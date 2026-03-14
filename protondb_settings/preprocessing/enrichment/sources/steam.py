"""Steam Store API and Deck Verified API clients with retry."""

from __future__ import annotations

import logging
import threading
import time

import httpx

from protondb_settings.preprocessing.enrichment.models import DeckData, SteamData

log = logging.getLogger(__name__)

_STORE_URL = "https://store.steampowered.com/api/appdetails"
_DECK_URL = "https://store.steampowered.com/saleaction/ajaxgetdeckappcompatibilityreport"

_MAX_RETRIES = 3
_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(timeout=30, follow_redirects=True)
    return _client


def fetch_steam(app_id: int) -> SteamData | None:
    """Fetch game info from Steam Store API.

    Returns None if the game doesn't exist or the request fails.
    """
    client = _get_client()
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(_STORE_URL, params={"appids": str(app_id)})
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.info("Steam Store: rate limited for app %d, waiting %ds", app_id, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json().get(str(app_id), {})
            if not data.get("success"):
                log.debug("Steam Store: app %d not found or not available", app_id)
                return None
            d = data["data"]

            return SteamData(
                developer=(d.get("developers") or [None])[0],
                publisher=(d.get("publishers") or [None])[0],
                genres=[g["description"] for g in d.get("genres", [])],
                categories=[c["description"] for c in d.get("categories", [])],
                release_date=d.get("release_date", {}).get("date"),
                has_linux_native=d.get("platforms", {}).get("linux", False),
            )
        except httpx.TimeoutException:
            wait = 2 ** attempt
            log.warning("Steam Store: timeout for app %d (attempt %d/%d)", app_id, attempt + 1, _MAX_RETRIES)
            time.sleep(wait)
        except Exception:
            log.warning("Steam Store: failed to fetch app %d (attempt %d/%d)", app_id, attempt + 1, _MAX_RETRIES, exc_info=True)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


def fetch_deck_status(app_id: int) -> DeckData | None:
    """Fetch Steam Deck compatibility report for a game.

    Returns None if no data is available.
    """
    client = _get_client()
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(_DECK_URL, params={"nAppID": str(app_id)})
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.info("Deck Verified: rate limited for app %d, waiting %ds", app_id, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success") or not data.get("results"):
                return None
            results = data["results"]
            return DeckData(
                status=results.get("resolved_category", 0),
                tests=results.get("resolved_items", []),
            )
        except httpx.TimeoutException:
            log.warning("Deck Verified: timeout for app %d (attempt %d/%d)", app_id, attempt + 1, _MAX_RETRIES)
            time.sleep(2 ** attempt)
        except Exception:
            log.warning("Deck Verified: failed for app %d (attempt %d/%d)", app_id, attempt + 1, _MAX_RETRIES, exc_info=True)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


