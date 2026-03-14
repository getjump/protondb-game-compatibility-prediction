"""AreWeAntiCheatYet data loader with ETag caching."""

from __future__ import annotations

import logging
import sqlite3

import httpx

from protondb_settings.preprocessing.enrichment.models import AWACYData

log = logging.getLogger(__name__)

_AWACY_URL = (
    "https://raw.githubusercontent.com/"
    "AreWeAntiCheatYet/AreWeAntiCheatYet/HEAD/games.json"
)


def check_awacy_stale(conn: sqlite3.Connection) -> bool:
    """Check if AreWeAntiCheatYet data is stale via HTTP HEAD + ETag.

    Returns True if data needs to be re-fetched.
    """
    row = conn.execute(
        "SELECT value FROM meta WHERE key = 'awacy_etag'"
    ).fetchone()
    stored_etag = row["value"] if row else None

    if stored_etag is None:
        return True

    try:
        resp = httpx.head(_AWACY_URL, timeout=10, headers={"If-None-Match": stored_etag})
        if resp.status_code == 304:
            return False
        return True
    except Exception:
        log.warning("AWACY: HEAD check failed", exc_info=True)
        return True


def load_awacy(conn: sqlite3.Connection | None = None) -> dict[int, AWACYData]:
    """Fetch games.json from AreWeAntiCheatYet and build an index by Steam app_id.

    If *conn* is provided, caches the ETag in the ``meta`` table.
    """
    try:
        resp = httpx.get(_AWACY_URL, timeout=60)
        resp.raise_for_status()
    except Exception:
        log.warning("AWACY: failed to fetch games.json", exc_info=True)
        return {}

    # Store ETag for future stale checks
    if conn is not None:
        etag = resp.headers.get("ETag")
        if etag:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('awacy_etag', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (etag,),
            )
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('awacy_fetched_at', datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value = datetime('now')",
            )
            conn.commit()

    index: dict[int, AWACYData] = {}
    games = resp.json()
    for game in games:
        # storeIds is a dict (not list): {"steam": "730"} or {"epic": {...}}
        store_ids = game.get("storeIds", {})
        steam_id = store_ids.get("steam")
        if steam_id is None:
            continue
        try:
            sid = int(steam_id)
        except (ValueError, TypeError):
            continue

        try:
            # anticheats can be list[str] or list[dict] depending on data version
            raw_anticheats = game.get("anticheats", [])
            anticheats: list[str] = []
            for ac in raw_anticheats:
                if isinstance(ac, str):
                    anticheats.append(ac)
                elif isinstance(ac, dict):
                    name = ac.get("name", "")
                    if name:
                        anticheats.append(name)

            index[sid] = AWACYData(
                anticheats=anticheats,
                status=game.get("status"),
            )
        except Exception:
            log.debug("AWACY: failed to parse game %s", game.get("name", "?"))
            continue

    log.info("AWACY: loaded %d games with Steam IDs", len(index))
    return index
