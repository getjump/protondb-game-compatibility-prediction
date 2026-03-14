"""ProtonDB dump download, parsing, and database ingestion.

Two modes:
  - **check**: query GitHub API for latest release, compare with meta table.
  - **sync**: download (or load local) dump, parse JSON, UPSERT into DB.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from protondb_settings.db.connection import get_connection
from protondb_settings.db.migrations import ensure_schema

console = Console()

GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/bdefore/protondb-data/releases/latest"
)

BATCH_SIZE = 5_000

# ---------------------------------------------------------------------------
# Field mapping: JSON camelCase paths -> DB snake_case columns
# ---------------------------------------------------------------------------

# responses.customizationsUsed.* -> cust_* INTEGER (0/1)
_CUST_MAP: dict[str, str] = {
    "winetricks": "cust_winetricks",
    "protontricks": "cust_protontricks",
    "configChange": "cust_config_change",
    "customPrefix": "cust_custom_prefix",
    "customProton": "cust_custom_proton",
    "lutris": "cust_lutris",
    "mediaFoundation": "cust_media_foundation",
    "protonfixes": "cust_protonfixes",
    "native2Proton": "cust_native2proton",
    "notListed": "cust_not_listed",
}

# responses.launchFlagsUsed.* -> flag_* INTEGER (0/1)
_FLAG_MAP: dict[str, str] = {
    "useWineD3d11": "flag_use_wine_d3d11",
    "disableEsync": "flag_disable_esync",
    "enableNvapi": "flag_enable_nvapi",
    "disableFsync": "flag_disable_fsync",
    "useWineD9vk": "flag_use_wine_d9vk",
    "largeAddressAware": "flag_large_address_aware",
    "disableD3d11": "flag_disable_d3d11",
    "hideNvidia": "flag_hide_nvidia",
    "gameDrive": "flag_game_drive",
    "noWriteWatch": "flag_no_write_watch",
    "noXim": "flag_no_xim",
    "oldGlString": "flag_old_gl_string",
    "useSeccomp": "flag_use_seccomp",
    "fullscreenIntegerScaling": "flag_fullscreen_integer_scaling",
}

# responses.followUp.* -> followup_*_json TEXT (serialized JSON)
_FOLLOWUP_MAP: dict[str, str] = {
    "audioFaults": "followup_audio_faults_json",
    "graphicalFaults": "followup_graphical_faults_json",
    "inputFaults": "followup_input_faults_json",
    "performanceFaults": "followup_performance_faults_json",
    "stabilityFaults": "followup_stability_faults_json",
    "windowingFaults": "followup_windowing_faults_json",
    "saveGameFaults": "followup_save_game_faults_json",
    "isImpactedByAntiCheat": "followup_anticheat_json",
    "controlLayoutCustomization": "followup_control_cust_json",
}

# responses.notes.* -> notes_* TEXT
_NOTES_MAP: dict[str, str] = {
    "verdict": "notes_verdict",
    "audioFaults": "notes_audio_faults",
    "graphicalFaults": "notes_graphical_faults",
    "performanceFaults": "notes_performance_faults",
    "stabilityFaults": "notes_stability_faults",
    "windowingFaults": "notes_windowing_faults",
    "inputFaults": "notes_input_faults",
    "significantBugs": "notes_significant_bugs",
    "saveGameFaults": "notes_save_game_faults",
    "extra": "notes_extra",
    "launchFlagsUsed": "notes_launch_flags",
    "customizationsUsed": "notes_customizations",
    "variant": "notes_variant",
    "protonVersion": "notes_proton_version",
    "concludingNotes": "notes_concluding_notes",
    "controlLayout": "notes_control_layout",
    "controlLayoutCustomization": "notes_control_layout_customization",
    "batteryPerformance": "notes_battery_performance",
    "readability": "notes_readability",
    "launcher": "notes_launcher",
    "secondaryLauncher": "notes_secondary_launcher",
    "localMultiplayerAppraisal": "notes_local_mp",
    "onlineMultiplayerAppraisal": "notes_online_mp",
    "isImpactedByAntiCheat": "notes_anticheat",
    "tinkerOverride": "notes_tinker_override",
}


def _generate_report_id(record: dict) -> str:
    """Generate a deterministic report ID from record content.

    The dump does not contain explicit IDs, so we hash
    (app_id, timestamp, gpu, cpu, verdict) to produce a stable key.
    """
    app_id = record.get("app", {}).get("steam", {}).get("appId", "")
    ts = str(record.get("timestamp", ""))
    si = record.get("systemInfo", {})
    resp = record.get("responses", {})
    key = f"{app_id}|{ts}|{si.get('gpu', '')}|{si.get('cpu', '')}|{resp.get('verdict', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _bool_int(val: object) -> int | None:
    """Convert a boolean-ish value to 0/1 or None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val else 0
    return None


def _map_record(record: dict) -> dict:
    """Convert a single JSON record to a flat dict matching the reports schema."""
    app = record.get("app", {})
    steam = app.get("steam", {})
    si = record.get("systemInfo", {})
    resp = record.get("responses", {})

    app_id_str = steam.get("appId", resp.get("answerToWhatGame", ""))
    try:
        app_id = int(app_id_str)
    except (ValueError, TypeError):
        return {}  # skip records without a valid app_id

    row: dict = {
        "id": _generate_report_id(record),
        "app_id": app_id,
        "timestamp": str(record.get("timestamp", "")),
        # systemInfo
        "gpu": si.get("gpu"),
        "gpu_driver": si.get("gpuDriver"),
        "cpu": si.get("cpu"),
        "ram": si.get("ram"),
        "os": si.get("os"),
        "ram_mb": None,  # filled by preprocessing/cleaning
        "kernel": si.get("kernel"),
        "window_manager": si.get("xWindowManager"),
        # responses: main
        "type": resp.get("type"),
        "variant": resp.get("variant"),
        "proton_version": resp.get("protonVersion"),
        "custom_proton_version": resp.get("customProtonVersion"),
        "verdict": resp.get("verdict"),
        "tried_oob": resp.get("triedOob"),
        "verdict_oob": resp.get("verdictOob"),
        "tinker_override": resp.get("tinkerOverride"),
        # responses: funnel
        "installs": resp.get("installs"),
        "opens": resp.get("opens"),
        "starts_play": resp.get("startsPlay"),
        "duration": resp.get("duration"),
        "extra": resp.get("extra"),
        # faults
        "audio_faults": resp.get("audioFaults"),
        "graphical_faults": resp.get("graphicalFaults"),
        "input_faults": resp.get("inputFaults"),
        "performance_faults": resp.get("performanceFaults"),
        "stability_faults": resp.get("stabilityFaults"),
        "windowing_faults": resp.get("windowingFaults"),
        "save_game_faults": resp.get("saveGameFaults"),
        "significant_bugs": resp.get("significantBugs"),
        # text
        "launch_options": resp.get("launchOptions"),
        "concluding_notes": resp.get("concludingNotes"),
        # Steam Deck
        "battery_performance": resp.get("batteryPerformance"),
        "readability": resp.get("readability"),
        "control_layout": resp.get("controlLayout"),
        "did_change_control_layout": resp.get("didChangeControlLayout"),
        "control_layout_customization": resp.get("controlLayoutCustomization"),
        "frame_rate": resp.get("frameRate"),
        # Launcher
        "launcher": resp.get("launcher"),
        "secondary_launcher": resp.get("secondaryLauncher"),
        # Multiplayer
        "is_multiplayer_important": resp.get("isMultiplayerImportant"),
        "local_mp_attempted": resp.get("localMultiplayerAttempted"),
        "local_mp_played": resp.get("localMultiplayerPlayed"),
        "local_mp_appraisal": resp.get("localMultiplayerAppraisal"),
        "online_mp_attempted": resp.get("onlineMultiplayerAttempted"),
        "online_mp_played": resp.get("onlineMultiplayerPlayed"),
        "online_mp_appraisal": resp.get("onlineMultiplayerAppraisal"),
        # Anti-cheat
        "is_impacted_by_anticheat": resp.get("isImpactedByAntiCheat"),
    }

    # --- followUp.* -> JSON strings ---
    followup = resp.get("followUp")
    if isinstance(followup, dict):
        for src_key, db_col in _FOLLOWUP_MAP.items():
            val = followup.get(src_key)
            row[db_col] = json.dumps(val) if val is not None else None
    else:
        for db_col in _FOLLOWUP_MAP.values():
            row[db_col] = None

    # --- customizationsUsed.* -> 0/1 ---
    cust = resp.get("customizationsUsed")
    if isinstance(cust, dict):
        for src_key, db_col in _CUST_MAP.items():
            row[db_col] = _bool_int(cust.get(src_key))
    else:
        for db_col in _CUST_MAP.values():
            row[db_col] = None

    # --- launchFlagsUsed.* -> 0/1 ---
    flags = resp.get("launchFlagsUsed")
    if isinstance(flags, dict):
        for src_key, db_col in _FLAG_MAP.items():
            row[db_col] = _bool_int(flags.get(src_key))
    else:
        for db_col in _FLAG_MAP.values():
            row[db_col] = None

    # --- notes.* -> text ---
    notes = resp.get("notes")
    if isinstance(notes, dict):
        for src_key, db_col in _NOTES_MAP.items():
            row[db_col] = notes.get(src_key)
    else:
        for db_col in _NOTES_MAP.values():
            row[db_col] = None

    return row


# ---------------------------------------------------------------------------
# Column list for INSERT — must match the reports table exactly
# ---------------------------------------------------------------------------

_REPORT_COLUMNS = [
    "id", "app_id", "timestamp",
    # systemInfo
    "gpu", "gpu_driver", "cpu", "ram", "os", "ram_mb", "kernel", "window_manager",
    # responses: main
    "type", "variant", "proton_version", "custom_proton_version",
    "verdict", "tried_oob", "verdict_oob", "tinker_override",
    # responses: funnel
    "installs", "opens", "starts_play", "duration", "extra",
    # faults
    "audio_faults", "graphical_faults", "input_faults", "performance_faults",
    "stability_faults", "windowing_faults", "save_game_faults", "significant_bugs",
    # followUp JSON
    "followup_audio_faults_json", "followup_graphical_faults_json",
    "followup_input_faults_json", "followup_performance_faults_json",
    "followup_stability_faults_json", "followup_windowing_faults_json",
    "followup_save_game_faults_json", "followup_anticheat_json",
    "followup_control_cust_json",
    # customizationsUsed
    "cust_winetricks", "cust_protontricks", "cust_config_change",
    "cust_custom_prefix", "cust_custom_proton", "cust_lutris",
    "cust_media_foundation", "cust_protonfixes", "cust_native2proton",
    "cust_not_listed",
    # launchFlagsUsed
    "flag_use_wine_d3d11", "flag_disable_esync", "flag_enable_nvapi",
    "flag_disable_fsync", "flag_use_wine_d9vk", "flag_large_address_aware",
    "flag_disable_d3d11", "flag_hide_nvidia", "flag_game_drive",
    "flag_no_write_watch", "flag_no_xim", "flag_old_gl_string",
    "flag_use_seccomp", "flag_fullscreen_integer_scaling",
    # text
    "launch_options", "concluding_notes",
    # notes
    "notes_verdict", "notes_audio_faults", "notes_graphical_faults",
    "notes_performance_faults", "notes_stability_faults", "notes_windowing_faults",
    "notes_input_faults", "notes_significant_bugs", "notes_save_game_faults",
    "notes_extra", "notes_launch_flags", "notes_customizations",
    "notes_variant", "notes_proton_version", "notes_concluding_notes",
    # Steam Deck
    "battery_performance", "readability", "control_layout",
    "did_change_control_layout", "control_layout_customization", "frame_rate",
    "notes_control_layout", "notes_control_layout_customization",
    "notes_battery_performance", "notes_readability",
    # Launcher
    "launcher", "secondary_launcher", "notes_launcher", "notes_secondary_launcher",
    # Multiplayer
    "is_multiplayer_important",
    "local_mp_attempted", "local_mp_played", "local_mp_appraisal", "notes_local_mp",
    "online_mp_attempted", "online_mp_played", "online_mp_appraisal", "notes_online_mp",
    # Anti-cheat
    "is_impacted_by_anticheat", "notes_anticheat",
    # Tinker override
    "notes_tinker_override",
]


def _build_upsert_sql() -> str:
    """Build the INSERT OR REPLACE statement for reports."""
    cols = ", ".join(_REPORT_COLUMNS)
    placeholders = ", ".join(["?"] * len(_REPORT_COLUMNS))
    return f"INSERT OR REPLACE INTO reports ({cols}) VALUES ({placeholders})"


def _row_tuple(row: dict) -> tuple:
    """Convert a mapped row dict to a tuple matching _REPORT_COLUMNS order."""
    return tuple(row.get(c) for c in _REPORT_COLUMNS)


# ---------------------------------------------------------------------------
# Streaming JSON parser — avoids loading the full 464 MB file into memory
# ---------------------------------------------------------------------------


def _iter_json_array(path: Path):
    """Yield individual JSON objects from a top-level JSON array file.

    Uses a simple bracket-counting approach rather than loading the entire
    file.  This handles strings with escaped characters correctly.
    """
    with open(path, "r", encoding="utf-8") as f:
        # Skip whitespace and opening bracket
        ch = ""
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch == "[":
                break

        decoder = json.JSONDecoder()
        buf = ""
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            buf += chunk

            while True:
                # Skip whitespace and commas between objects
                idx = 0
                while idx < len(buf) and buf[idx] in " \t\n\r,":
                    idx += 1
                buf = buf[idx:]

                if not buf or buf[0] == "]":
                    buf = ""
                    break

                if buf[0] != "{":
                    # Should not happen in well-formed input; skip char
                    buf = buf[1:]
                    continue

                try:
                    obj, end = decoder.raw_decode(buf, 0)
                    yield obj
                    buf = buf[end:]
                except json.JSONDecodeError:
                    # Incomplete object — need more data
                    break


# ---------------------------------------------------------------------------
# check / sync
# ---------------------------------------------------------------------------


async def check_for_update(db_path: Path) -> bool:
    """Query GitHub for the latest release and compare with stored tag.

    Returns True if a new dump is available.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(GITHUB_RELEASES_URL)
        resp.raise_for_status()
        release = resp.json()

    latest_tag = release.get("tag_name", "")

    conn = get_connection(db_path)
    ensure_schema(conn)
    try:
        meta_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'dump_release_tag'"
        ).fetchone()
        current_tag = meta_row["value"] if meta_row else None
    finally:
        conn.close()

    if current_tag == latest_tag:
        console.print(f"[green]Up to date:[/green] {latest_tag}")
        return False

    console.print(
        f"[yellow]New dump available:[/yellow] {latest_tag} "
        f"(current: {current_tag or 'none'})"
    )
    return True


async def sync_dump(db_path: Path, *, local_file: Path | None = None) -> None:
    """Download the latest dump (or use a local file) and import into the DB."""
    conn = get_connection(db_path)
    ensure_schema(conn)

    if local_file is not None:
        console.print(f"Loading from local file: [bold]{local_file}[/bold]")
        _import_json_file(conn, local_file, release_tag="local", sha256_hex="local")
    else:
        await _download_and_import(conn)

    conn.close()
    console.print("[green]Sync complete.[/green]")


async def _download_and_import(conn: sqlite3.Connection) -> None:
    """Download .tar.gz from latest GitHub release and import."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(GITHUB_RELEASES_URL)
        resp.raise_for_status()
        release = resp.json()

    tag = release.get("tag_name", "unknown")
    assets = release.get("assets", [])

    # Find the .tar.gz asset
    tar_url = None
    for asset in assets:
        name = asset.get("name", "")
        if name.endswith(".tar.gz"):
            tar_url = asset["browser_download_url"]
            break

    if not tar_url:
        console.print("[red]No .tar.gz asset found in the latest release.[/red]")
        return

    console.print(f"Downloading [bold]{tag}[/bold] from {tar_url} ...")

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        resp = await client.get(tar_url)
        resp.raise_for_status()
        data = resp.content

    sha256_hex = hashlib.sha256(data).hexdigest()

    # Check if we already have this exact file
    meta_row = conn.execute(
        "SELECT value FROM meta WHERE key = 'dump_sha256'"
    ).fetchone()
    if meta_row and meta_row["value"] == sha256_hex:
        console.print("[green]Dump already imported (SHA-256 matches). Skipping.[/green]")
        return

    # Extract JSON from tar.gz
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tf:
            tf.extractall(tmpdir_path)

        # Find JSON file(s) in the extracted contents
        json_files = list(tmpdir_path.rglob("*.json"))
        if not json_files:
            console.print("[red]No JSON files found in the archive.[/red]")
            return

        for jf in json_files:
            console.print(f"  Importing {jf.name} ...")
            _import_json_file(conn, jf, release_tag=tag, sha256_hex=sha256_hex)


def _import_json_file(
    conn: sqlite3.Connection,
    path: Path,
    *,
    release_tag: str,
    sha256_hex: str,
) -> None:
    """Parse a JSON array file and UPSERT records into the database."""
    upsert_sql = _build_upsert_sql()
    game_upsert = (
        "INSERT OR REPLACE INTO games (app_id, name, updated_at) "
        "VALUES (?, ?, datetime('now'))"
    )

    batch_reports: list[tuple] = []
    batch_games: dict[int, str] = {}
    total = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} records"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Importing reports...", total=None)

        for record in _iter_json_array(path):
            row = _map_record(record)
            if not row:
                skipped += 1
                continue

            app_id = row["app_id"]
            title = record.get("app", {}).get("title", f"Unknown ({app_id})")
            batch_games[app_id] = title
            batch_reports.append(_row_tuple(row))
            total += 1

            if len(batch_reports) >= BATCH_SIZE:
                _flush_batch(conn, upsert_sql, game_upsert, batch_reports, batch_games)
                batch_reports.clear()
                batch_games.clear()
                progress.update(task, completed=total)

        # Final batch
        if batch_reports:
            _flush_batch(conn, upsert_sql, game_upsert, batch_reports, batch_games)
            progress.update(task, completed=total)

    console.print(f"  Imported {total} reports ({skipped} skipped)")

    # Write meta
    now = datetime.now(timezone.utc).isoformat()
    _set_meta(conn, "dump_release_tag", release_tag)
    _set_meta(conn, "dump_sha256", sha256_hex)
    _set_meta(conn, "dump_imported_at", now)


def _flush_batch(
    conn: sqlite3.Connection,
    report_sql: str,
    game_sql: str,
    reports: list[tuple],
    games: dict[int, str],
) -> None:
    """Write a batch of reports and games inside a single transaction."""
    with conn:
        for app_id, name in games.items():
            conn.execute(game_sql, (app_id, name))
        conn.executemany(report_sql, reports)


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update a key in the meta table."""
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
