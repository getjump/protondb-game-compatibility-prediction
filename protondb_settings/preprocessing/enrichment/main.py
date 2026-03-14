"""Enrichment orchestrator — fetches game metadata from external APIs.

Architecture:
  Phase 1 (parallel): threads per source fetch into enrichment_cache
    - Thread "steam_deck": Steam Store + Deck Verified (shared rate limit)
    - Thread "pcgamingwiki": PCGamingWiki Cargo API (batch)
    - Thread "protondb": ProtonDB Summary API (opt-in, --source protondb)
    - AWACY: pre-fetched in main thread (single file)
  Phase 2 (sequential): merge from cache → game_metadata

All errors are non-fatal. Ctrl+C finishes current items and exits.
API responses are cached in `enrichment_cache` — re-runs skip already-fetched.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from protondb_settings.config import (
    ENRICHMENT_BATCH_SIZE,
    PCGW_RATE_LIMIT,
    PROTONDB_RATE_LIMIT,
    STEAM_RATE_LIMIT,
)
from protondb_settings.db.connection import get_connection
from protondb_settings.preprocessing.enrichment.merger import merge_metadata
from protondb_settings.preprocessing.enrichment.models import (
    AWACYData,
    DeckData,
    PCGWData,
    ProtonDBData,
    ProtonGitHubData,
    SteamData,
)
from protondb_settings.preprocessing.enrichment.sources.anticheat import load_awacy
from protondb_settings.preprocessing.enrichment.sources.pcgamingwiki import (
    fetch_pcgw_batch,
)
from protondb_settings.preprocessing.enrichment.sources.protondb import (
    fetch_protondb_summary,
)
from protondb_settings.preprocessing.enrichment.sources.protondb_reports import (
    extract_contributor_data,
    fetch_reports as fetch_protondb_reports,
)
from protondb_settings.preprocessing.enrichment.sources.steam import (
    fetch_deck_status,
    fetch_steam,
)
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import upsert_rows

log = logging.getLogger(__name__)


# ── Interruption handling ─────────────────────────────────────────────
# Uses shared shutdown_requested event from interrupt module, which handles
# SIGINT, SIGTERM, and kitty keyboard protocol escape sequences (Cursor terminal).

from protondb_settings.preprocessing.interrupt import (
    install_handlers,
    restore_handlers,
    shutdown_requested,
)


# ── Rate limit helper ─────────────────────────────────────────────────


def _rate_sleep(rate: float) -> None:
    """Sleep for 1/rate seconds, guarding against zero/negative."""
    if rate > 0:
        time.sleep(1.0 / rate)


# ── Source cache (thread-safe via per-thread connections) ─────────────

_EMPTY: MappingProxyType[str, bool] = MappingProxyType({"_empty": True})


def _set_cached(conn: sqlite3.Connection, app_id: int, source: str, data: Any) -> None:
    data_json = json.dumps(dict(data) if isinstance(data, MappingProxyType) else data) if data is not None else None
    conn.execute(
        "INSERT INTO enrichment_cache (app_id, source, data_json) VALUES (?, ?, ?) "
        "ON CONFLICT(app_id, source) DO UPDATE SET data_json = excluded.data_json, "
        "fetched_at = datetime('now')",
        (app_id, source, data_json),
    )


def _get_cached_set(conn: sqlite3.Connection, source: str) -> set[int]:
    rows = conn.execute(
        "SELECT app_id FROM enrichment_cache WHERE source = ?", (source,)
    ).fetchall()
    return {row["app_id"] for row in rows}


def _invalidate_stale_cache(conn: sqlite3.Connection, days: int) -> int:
    """Delete cache entries older than N days so they get re-fetched."""
    cur = conn.execute(
        "DELETE FROM enrichment_cache WHERE fetched_at < datetime('now', ?)",
        (f"-{days} days",),
    )
    conn.commit()
    return cur.rowcount


def _from_cache(cached: dict | None, model_cls: type) -> Any:
    if cached is None or "_empty" in cached:
        return None
    try:
        return model_cls.model_validate(cached)
    except Exception:
        return None


# ── Per-source fetch workers ─────────────────────────────────────────


class _WorkerProgress:
    """Thread-safe progress wrapper backed by a shared rich Progress bar."""

    def __init__(self, progress: Progress, task_id: int, name: str) -> None:
        self._progress = progress
        self._task_id = task_id
        self._name = name
        self._lock = threading.Lock()
        self._errors = 0

    def advance(self, n: int = 1) -> None:
        self._progress.update(self._task_id, advance=n)

    def error(self) -> None:
        with self._lock:
            self._errors += 1
            errs = self._errors
        self._progress.update(
            self._task_id, description=f"{self._name} [red]({errs} err)[/]",
        )

    def print(self, msg: str) -> None:
        """Print a message without breaking the progress display."""
        self._progress.console.print(msg)

    @property
    def errors(self) -> int:
        with self._lock:
            return self._errors


def _worker_steam_deck(
    db_path: Path, app_ids: list[int], progress: _WorkerProgress,
) -> None:
    """Fetch Steam Store + Deck Verified for all app_ids."""
    conn = get_connection(db_path)
    try:
        cached_steam = _get_cached_set(conn, "steam")
        cached_deck = _get_cached_set(conn, "deck")

        for app_id in app_ids:
            if shutdown_requested.is_set():
                break
            try:
                if app_id not in cached_steam:
                    result = fetch_steam(app_id)
                    _set_cached(conn, app_id, "steam", result.model_dump() if result else _EMPTY)
                    conn.commit()  # release write lock before sleeping
                    _rate_sleep(STEAM_RATE_LIMIT)

                if app_id not in cached_deck:
                    result = fetch_deck_status(app_id)
                    _set_cached(conn, app_id, "deck", result.model_dump() if result else _EMPTY)
                    conn.commit()  # release write lock before sleeping
                    _rate_sleep(STEAM_RATE_LIMIT)
            except Exception as exc:
                progress.print(f"[yellow]Steam+Deck: failed for app {app_id}: {exc}[/]")
                progress.error()
                try:
                    conn.rollback()
                except Exception:
                    pass

            progress.advance()
    finally:
        conn.close()


def _worker_protondb(
    db_path: Path, app_ids: list[int], progress: _WorkerProgress,
) -> None:
    """Fetch ProtonDB summaries for all app_ids."""
    conn = get_connection(db_path)
    try:
        cached = _get_cached_set(conn, "protondb")

        for app_id in app_ids:
            if shutdown_requested.is_set():
                break
            if app_id in cached:
                progress.advance()
                continue
            try:
                result = fetch_protondb_summary(app_id)
                _set_cached(conn, app_id, "protondb", result.model_dump() if result else _EMPTY)
                conn.commit()
                _rate_sleep(PROTONDB_RATE_LIMIT)
            except Exception as exc:
                progress.print(f"[yellow]ProtonDB: failed for app {app_id}: {exc}[/]")
                progress.error()
                try:
                    conn.rollback()
                except Exception:
                    pass

            progress.advance()
    finally:
        conn.close()


def _worker_protondb_reports(
    db_path: Path, app_ids: list[int], progress: _WorkerProgress,
) -> None:
    """Fetch ProtonDB individual reports with contributor data.

    Stores contributor info in report_contributors table.
    Caches raw API response in enrichment_cache (source='protondb_reports').
    """
    conn = get_connection(db_path)
    try:
        cached = _get_cached_set(conn, "protondb_reports")

        for app_id in app_ids:
            if shutdown_requested.is_set():
                break
            if app_id in cached:
                progress.advance()
                continue
            try:
                reports = fetch_protondb_reports(app_id)
                if reports is None:
                    _set_cached(conn, app_id, "protondb_reports", _EMPTY)
                    conn.commit()
                    _rate_sleep(PROTONDB_RATE_LIMIT)
                    progress.advance()
                    continue

                # Build DB lookup for matching: (timestamp, gpu) → report_id
                db_rows = conn.execute(
                    "SELECT id, timestamp, gpu FROM reports WHERE app_id = ?",
                    (app_id,),
                ).fetchall()
                db_lookup: dict[tuple[int, str], str] = {}
                for r in db_rows:
                    ts = int(r["timestamp"]) if r["timestamp"] else 0
                    gpu = r["gpu"] or ""
                    db_lookup[(ts, gpu)] = r["id"]

                # Extract and store contributor data, matching by (timestamp, gpu)
                matched = 0
                for report in reports:
                    contrib = extract_contributor_data(report)
                    if not contrib:
                        continue
                    key = (contrib["timestamp"], contrib["gpu"])
                    db_id = db_lookup.get(key)
                    if db_id:
                        conn.execute(
                            "INSERT OR REPLACE INTO report_contributors "
                            "(report_id, contributor_id, report_tally, playtime, playtime_linux) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                db_id,
                                contrib["contributor_id"],
                                contrib["report_tally"],
                                contrib["playtime"],
                                contrib["playtime_linux"],
                            ),
                        )
                        matched += 1

                _set_cached(conn, app_id, "protondb_reports", {"total": len(reports), "matched": matched})
                conn.commit()
                _rate_sleep(PROTONDB_RATE_LIMIT)
            except Exception as exc:
                progress.print(f"[yellow]ProtonDB reports: failed for app {app_id}: {exc}[/]")
                progress.error()
                try:
                    conn.rollback()
                except Exception:
                    pass

            progress.advance()
    finally:
        conn.close()


def _worker_pcgamingwiki(
    db_path: Path, app_ids: list[int], progress: _WorkerProgress,
) -> None:
    """Fetch PCGamingWiki data in batches of 10."""
    conn = get_connection(db_path)
    try:
        cached = _get_cached_set(conn, "pcgamingwiki")
        to_fetch = [aid for aid in app_ids if aid not in cached]

        cached_count = len(app_ids) - len(to_fetch)
        progress.print(f"PCGamingWiki: {len(to_fetch)} to fetch ({cached_count} cached)")
        progress.advance(cached_count)

        for batch in chunked(to_fetch, 50):
            if shutdown_requested.is_set():
                break
            try:
                batch_result = fetch_pcgw_batch(batch)
                for aid in batch:
                    if aid in batch_result:
                        _set_cached(conn, aid, "pcgamingwiki", batch_result[aid].model_dump())
                    else:
                        _set_cached(conn, aid, "pcgamingwiki", _EMPTY)
                conn.commit()
                _rate_sleep(PCGW_RATE_LIMIT)
            except Exception as exc:
                progress.print(f"[yellow]PCGamingWiki: batch failed ({len(batch)} ids): {exc}[/]")
                progress.error()
                try:
                    conn.rollback()
                except Exception:
                    pass

            progress.advance(len(batch))
    finally:
        conn.close()


# ── Main orchestrator ─────────────────────────────────────────────────


def _get_pending_app_ids(
    conn: sqlite3.Connection,
    *,
    min_reports: int = 1,
    refresh_older_than_days: int | None = None,
    source: str | None = None,
) -> list[int]:
    # protondb_reports writes to report_contributors, not game_metadata —
    # pending list is based on enrichment_cache, not game_metadata.
    if source == "protondb_reports":
        rows = conn.execute(
            """
            SELECT r.app_id, COUNT(*) as cnt FROM reports r
            WHERE r.app_id NOT IN (
                SELECT app_id FROM enrichment_cache WHERE source = 'protondb_reports'
            )
            GROUP BY r.app_id
            HAVING cnt >= ?
            ORDER BY cnt DESC
            """,
            (min_reports,),
        ).fetchall()
        return [row["app_id"] for row in rows]

    if refresh_older_than_days is not None:
        rows = conn.execute(
            """
            SELECT r.app_id, COUNT(*) as cnt FROM reports r
            LEFT JOIN game_metadata gm ON r.app_id = gm.app_id
            WHERE gm.app_id IS NULL
               OR gm.enriched_at < datetime('now', ?)
            GROUP BY r.app_id
            HAVING cnt >= ?
            ORDER BY cnt DESC
            """,
            (f"-{refresh_older_than_days} days", min_reports),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT r.app_id, COUNT(*) as cnt FROM reports r
            WHERE r.app_id NOT IN (SELECT app_id FROM game_metadata)
            GROUP BY r.app_id
            HAVING cnt >= ?
            ORDER BY cnt DESC
            """,
            (min_reports,),
        ).fetchall()

    pending = [row["app_id"] for row in rows]

    # If a specific source is requested, also include app_ids that are in
    # game_metadata but missing from enrichment_cache for that source.
    # This handles the case where enrichment was interrupted mid-source
    # (e.g. pcgamingwiki finished but steam was stopped).
    if source and source not in ("anticheat",):
        cache_source = source
        if source == "steam":
            # steam worker also fetches deck — check steam cache
            cache_source = "steam"
        all_ids_rows = conn.execute(
            """
            SELECT app_id FROM game_metadata
            WHERE app_id NOT IN (SELECT app_id FROM enrichment_cache WHERE source = ?)
            """,
            (cache_source,),
        ).fetchall()
        missing = {row["app_id"] for row in all_ids_rows}
        if missing:
            pending_set = set(pending)
            pending.extend(aid for aid in missing if aid not in pending_set)
            log.info(
                "Added %d app_ids missing %s cache (already in game_metadata)",
                len(missing), cache_source,
            )

    return pending


def get_pending_count(
    conn: sqlite3.Connection, *, min_reports: int = 1
) -> int:
    return len(_get_pending_app_ids(conn, min_reports=min_reports))


def run_enrichment(
    conn: sqlite3.Connection,
    *,
    min_reports: int = 1,
    source: str | None = None,
    force: bool = False,
    refresh_older_than_days: int | None = None,
) -> int:
    """Run the enrichment pipeline.

    Phase 1: parallel fetch (threads per source) → enrichment_cache
    Phase 2: merge from cache → game_metadata
    """
    install_handlers()
    try:
        return _run_enrichment_inner(
            conn,
            min_reports=min_reports,
            source=source,
            force=force,
            refresh_older_than_days=refresh_older_than_days,
        )
    finally:
        restore_handlers()


def _run_enrichment_inner(
    conn: sqlite3.Connection,
    *,
    min_reports: int,
    source: str | None,
    force: bool,
    refresh_older_than_days: int | None,
) -> int:
    if force:
        log.info("Force enrichment: deleting game_metadata and cache")
        conn.execute("DELETE FROM game_metadata")
        conn.execute("DELETE FROM enrichment_cache")
        conn.commit()

    # Invalidate stale cache entries so they get re-fetched
    if refresh_older_than_days is not None:
        n = _invalidate_stale_cache(conn, refresh_older_than_days)
        if n:
            log.info("Invalidated %d stale cache entries (older than %d days)", n, refresh_older_than_days)

    app_ids = _get_pending_app_ids(
        conn,
        min_reports=min_reports,
        refresh_older_than_days=refresh_older_than_days,
        source=source,
    )

    if not app_ids:
        log.info("Enrichment: nothing to do")
        return 0

    log.info("Enrichment: %d app_ids to process (min_reports=%d)", len(app_ids), min_reports)

    db_path = Path(conn.execute("PRAGMA database_list").fetchone()["file"])

    # ── Phase 1: Parallel fetch into enrichment_cache ──────────────

    # protondb_reports is report-level (→ report_contributors), not game-level
    # (→ game_metadata). Skip AWACY/GitHub pre-fetch and Phase 2 merge.
    skip_merge = source == "protondb_reports"

    awacy_index: dict[int, AWACYData] = {}
    github_index: dict[int, ProtonGitHubData] = {}

    if not skip_merge:
        # Pre-fetch AWACY in main thread (single file, fast).
        # Always load AWACY regardless of --source filter — it's a single JSON fetch
        # and anticheat_status must be populated during every merge pass.
        log.info("Fetching AreWeAntiCheatYet data...")
        try:
            awacy_index = load_awacy(conn)
        except Exception:
            log.warning("AWACY: failed, continuing without anticheat data", exc_info=True)

        # Pre-fetch GitHub Proton issues (single bulk fetch via `gh`)
        if source is None or source == "github":
            log.info("Fetching GitHub Proton issues...")
            try:
                from protondb_settings.preprocessing.enrichment.sources.github_proton import (
                    build_github_index,
                    fetch_all_issues,
                )
                raw_issues = fetch_all_issues()
                github_index = build_github_index(raw_issues)
            except Exception:
                log.warning("GitHub: failed, continuing without issue data", exc_info=True)

    # Suppress httpx/httpcore INFO logs — they break rich progress display
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Launch parallel source workers with shared progress display
    workers: dict[str, tuple[Any, _WorkerProgress]] = {}
    pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="enrich")

    fetch_progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
    )

    # Add all tasks BEFORE starting the display to avoid partial renders
    task_ids: dict[str, int] = {}
    worker_configs: list[tuple[str, Any, list[int]]] = []

    if source is None or source == "steam":
        task_ids["steam+deck"] = fetch_progress.add_task("steam+deck", total=len(app_ids))
        worker_configs.append(("steam+deck", _worker_steam_deck, app_ids))

    if source == "protondb":
        task_ids["protondb"] = fetch_progress.add_task("protondb", total=len(app_ids))
        worker_configs.append(("protondb", _worker_protondb, app_ids))

    if source == "protondb_reports":
        task_ids["protondb_reports"] = fetch_progress.add_task("protondb_reports", total=len(app_ids))
        worker_configs.append(("protondb_reports", _worker_protondb_reports, app_ids))

    if source is None or source == "pcgamingwiki":
        task_ids["pcgamingwiki"] = fetch_progress.add_task("pcgamingwiki", total=len(app_ids))
        worker_configs.append(("pcgamingwiki", _worker_pcgamingwiki, app_ids))

    if worker_configs:
        log.info(
            "Fetching from %d sources in parallel: %s",
            len(worker_configs), ", ".join(task_ids.keys()),
        )

    fetch_progress.start()

    # Submit workers after display is started
    for name, worker_fn, ids in worker_configs:
        wp = _WorkerProgress(fetch_progress, task_ids[name], name)
        f = pool.submit(worker_fn, db_path, ids, wp)
        workers[name] = (f, wp)

    if workers:
        # Poll with timeout so SIGINT can be delivered between iterations
        # (future.result() without timeout blocks at C level, preventing signal delivery)
        failed: dict[str, Exception] = {}
        pending = dict(workers)
        try:
            while pending:
                newly_done = []
                for name, (future, wp) in pending.items():
                    try:
                        future.result(timeout=0.5)
                        newly_done.append(name)
                    except TimeoutError:
                        continue
                    except Exception as exc:
                        failed[name] = exc
                        newly_done.append(name)

                for name in newly_done:
                    pending.pop(name)

                if shutdown_requested.is_set():
                    break
        except KeyboardInterrupt:
            shutdown_requested.set()

    fetch_progress.stop()
    pool.shutdown(wait=True)

    # Log worker results AFTER progress display is stopped
    if workers:
        for name, (_, wp) in workers.items():
            if name in failed:
                log.warning("  %s: worker failed: %s", name, failed[name])
            elif wp.errors:
                log.warning("  %s: done with %d errors", name, wp.errors)

    if shutdown_requested.is_set():
        log.info("Interrupted during fetch phase — partial cache saved")
        return len(app_ids)

    if skip_merge:
        log.info("protondb_reports: %d app_ids processed (report_contributors updated)", len(app_ids))
        return len(app_ids)

    # ── Phase 2: Merge from cache → game_metadata ─────────────────

    log.info("Merging cached data into game_metadata...")

    # Batch-load entire cache into memory to avoid 4 queries per app_id
    cache_by_source: dict[str, dict[int, dict | None]] = {}
    for src in ("steam", "deck", "protondb", "pcgamingwiki"):
        rows = conn.execute(
            "SELECT app_id, data_json FROM enrichment_cache WHERE source = ?",
            (src,),
        ).fetchall()
        parsed: dict[int, dict | None] = {}
        for row in rows:
            try:
                parsed[row["app_id"]] = json.loads(row["data_json"]) if row["data_json"] else None
            except (json.JSONDecodeError, TypeError):
                parsed[row["app_id"]] = None
        cache_by_source[src] = parsed

    def _cached(app_id: int, source: str) -> dict | None:
        return cache_by_source.get(source, {}).get(app_id)

    processed = 0
    errors = 0

    with PipelineStep(conn, "enrichment", len(app_ids)) as step:
        for batch in chunked(app_ids, ENRICHMENT_BATCH_SIZE):
            if shutdown_requested.is_set():
                break

            rows_to_upsert = []
            actual_count = 0
            for app_id in batch:
                if shutdown_requested.is_set():
                    break
                actual_count += 1
                try:
                    steam_data = _from_cache(_cached(app_id, "steam"), SteamData)
                    deck_data = _from_cache(_cached(app_id, "deck"), DeckData)
                    protondb_data = _from_cache(_cached(app_id, "protondb"), ProtonDBData)
                    pcgw_data = _from_cache(_cached(app_id, "pcgamingwiki"), PCGWData)

                    merged = merge_metadata(
                        app_id,
                        steam=steam_data,
                        deck=deck_data,
                        pcgw=pcgw_data,
                        awacy=awacy_index.get(app_id),
                        protondb=protondb_data,
                        github=github_index.get(app_id),
                    )
                    rows_to_upsert.append(merged)
                except Exception:
                    errors += 1
                    log.debug("Merge: failed for app_id %d", app_id, exc_info=True)

            if rows_to_upsert:
                try:
                    upsert_rows(conn, "game_metadata", rows_to_upsert, "app_id")
                    conn.commit()
                except Exception:
                    log.warning("Merge: batch upsert failed", exc_info=True)
                    conn.rollback()

            step.advance(actual_count)
            step.sync_run()
            processed += actual_count

    if errors:
        log.warning("Enrichment completed with %d merge errors out of %d", errors, processed)
    if shutdown_requested.is_set():
        log.info("Enrichment interrupted after %d app_ids (will resume)", processed)

    return processed
