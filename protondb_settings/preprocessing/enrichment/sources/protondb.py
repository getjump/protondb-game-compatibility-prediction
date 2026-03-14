"""ProtonDB Summary API client with retry."""

from __future__ import annotations

import logging
import threading
import time

import httpx

from protondb_settings.preprocessing.enrichment.models import ProtonDBData

log = logging.getLogger(__name__)

_SUMMARY_URL = "https://www.protondb.com/api/v1/reports/summaries/{app_id}.json"

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


def fetch_protondb_summary(app_id: int) -> ProtonDBData | None:
    """Fetch community summary from ProtonDB.

    Returns None on 404 (no data) or request failure.
    """
    client = _get_client()
    url = _SUMMARY_URL.format(app_id=app_id)
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(url)
            if resp.status_code == 404:
                log.debug("ProtonDB: no summary for app %d", app_id)
                return None
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.info("ProtonDB: rate limited for app %d, waiting %ds", app_id, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return ProtonDBData(
                tier=data.get("tier"),
                score=data.get("score"),
                confidence=data.get("confidence"),
                trending_tier=data.get("trendingTier"),
            )
        except httpx.TimeoutException:
            log.warning("ProtonDB: timeout for app %d (attempt %d/%d)", app_id, attempt + 1, _MAX_RETRIES)
            time.sleep(2 ** attempt)
        except Exception:
            log.warning("ProtonDB: failed for app %d (attempt %d/%d)", app_id, attempt + 1, _MAX_RETRIES, exc_info=True)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


