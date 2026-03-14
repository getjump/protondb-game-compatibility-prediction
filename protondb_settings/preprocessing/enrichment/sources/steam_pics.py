"""Steam PICS (Package Info Cache System) client via steam[client] library.

Fetches app metadata in bulk via the Steam CM protocol — much faster than
Store API (~60 apps/s in batches vs ~0.6 apps/s with rate limits).

Data includes: oslist, osarch, launch configs, depot info, Deck compatibility
(granular tests + recommended_runtime), review_score, developer/publisher.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


def _get_client(force_new: bool = False):
    """Get or create a Steam client with anonymous login."""
    global _client
    if force_new and _client is not None:
        try:
            _client.disconnect()
        except Exception:
            pass
        _client = None

    if _client is None:
        with _client_lock:
            if _client is None:
                from steam.client import SteamClient
                c = SteamClient()
                c.anonymous_login()
                _client = c
                log.info("Steam PICS: anonymous login OK")
    return _client


def disconnect():
    """Disconnect the Steam client."""
    global _client
    if _client is not None:
        _client.disconnect()
        _client = None


@dataclass
class SteamPICSData:
    """Extracted fields from PICS app_info."""

    app_id: int
    name: str | None = None
    app_type: str | None = None  # Game, DLC, Tool, etc.
    oslist: str | None = None  # "windows,linux,macos"
    osarch: str | None = None  # "64", "", etc.

    # Deck compatibility
    deck_category: int | None = None  # 0=unknown, 1=unsupported, 2=playable, 3=verified
    steamos_compatibility: int | None = None  # separate from deck_category
    recommended_runtime: str | None = None  # "proton-experimental", "native", etc.
    deck_test_results: list[dict] | None = None  # [{display, token}, ...]

    # Launch configs
    has_linux_launch: bool = False
    has_windows_launch: bool = False
    launch_count: int = 0

    # Depots
    linux_depot_count: int = 0
    windows_depot_count: int = 0
    total_depot_count: int = 0

    # Review
    review_score: int | None = None  # 1-9
    review_percentage: int | None = None  # 0-100

    # Extended
    developer: str | None = None
    publisher: str | None = None
    is_free: bool = False

    # Primary genre
    primary_genre: int | None = None


def extract_pics_data(app_id: int, raw: dict) -> SteamPICSData:
    """Extract useful fields from raw PICS app_info."""
    common = raw.get("common", {})
    config = raw.get("config", {})
    extended = raw.get("extended", {})
    depots = raw.get("depots", {})
    deck = common.get("steam_deck_compatibility", {})
    deck_config = deck.get("configuration", {})

    # Launch configs
    launch = config.get("launch", {})
    has_linux = any(
        l.get("config", {}).get("oslist", "") == "linux"
        for l in launch.values()
    ) if isinstance(launch, dict) else False
    has_windows = any(
        l.get("config", {}).get("oslist", "") in ("windows", "")
        for l in launch.values()
    ) if isinstance(launch, dict) else False

    # Depots
    numeric_depots = {k: v for k, v in depots.items() if isinstance(k, str) and k.isdigit()}
    linux_depots = sum(
        1 for d in numeric_depots.values()
        if isinstance(d, dict) and d.get("config", {}).get("oslist", "") == "linux"
    )
    windows_depots = sum(
        1 for d in numeric_depots.values()
        if isinstance(d, dict) and d.get("config", {}).get("oslist", "") == "windows"
    )

    # Deck tests
    deck_tests = None
    raw_tests = deck.get("tests", {})
    if raw_tests and isinstance(raw_tests, dict):
        deck_tests = [
            {"display": int(t.get("display", 0)), "token": t.get("token", "")}
            for t in raw_tests.values()
        ]

    # Parse integers safely
    def _int(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    return SteamPICSData(
        app_id=app_id,
        name=common.get("name"),
        app_type=common.get("type"),
        oslist=common.get("oslist"),
        osarch=common.get("osarch") or None,
        deck_category=_int(deck.get("category")),
        steamos_compatibility=_int(deck.get("steamos_compatibility")),
        recommended_runtime=deck_config.get("recommended_runtime"),
        deck_test_results=deck_tests,
        has_linux_launch=has_linux,
        has_windows_launch=has_windows,
        launch_count=len(launch) if isinstance(launch, dict) else 0,
        linux_depot_count=linux_depots,
        windows_depot_count=windows_depots,
        total_depot_count=len(numeric_depots),
        review_score=_int(common.get("review_score")),
        review_percentage=_int(common.get("review_percentage")),
        developer=extended.get("developer"),
        publisher=extended.get("publisher"),
        is_free=extended.get("isfreeapp") == "1",
        primary_genre=_int(common.get("primary_genre")),
    )


def fetch_pics_batch(app_ids: list[int], _retry: bool = True) -> dict[int, SteamPICSData]:
    """Fetch PICS data for a batch of app_ids.

    Returns dict mapping app_id → SteamPICSData (only for successful fetches).
    Reconnects on timeout and retries once.
    """
    client = _get_client()

    try:
        info = client.get_product_info(apps=app_ids, timeout=30)
    except Exception as e:
        if _retry:
            log.warning("Steam PICS batch failed (%s), reconnecting...", e)
            client = _get_client(force_new=True)
            return fetch_pics_batch(app_ids, _retry=False)
        log.warning("Steam PICS batch failed after retry: %s", e)
        return {}

    apps_data = info.get("apps", {})
    result = {}
    for app_id, raw in apps_data.items():
        try:
            result[app_id] = extract_pics_data(app_id, raw)
        except Exception as e:
            log.debug("Steam PICS: failed to extract app %d: %s", app_id, e)

    return result
