"""ProtonDB Reports API client — fetches individual reports with contributor data.

Uses the undocumented hash-protected endpoint:
  /data/reports/all-devices/app/{hashedId}.json

Returns up to 40 most recent reports per game with:
  - contributor.id — user identity (allows per-annotator grouping)
  - contributor.reportTally — total reports by user (experience proxy)
  - contributor.steam.playtime — total playtime
  - contributor.steam.playtimeLinux — Linux playtime

The hash algorithm requires fetching /data/counts.json first to get
(reports, timestamp) needed for ID calculation.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

_COUNTS_URL = "https://www.protondb.com/data/counts.json"
_REPORTS_URL = "https://www.protondb.com/data/reports/all-devices/app/{hashed_id}.json"
_DECK_REPORTS_URL = "https://www.protondb.com/data/reports/steam-deck/app/{hashed_id}.json"

_MAX_RETRIES = 3
_client: httpx.Client | None = None
_client_lock = threading.Lock()

# Cached counts (fetched once per session)
_counts: dict | None = None
_counts_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    timeout=30,
                    follow_redirects=True,
                    headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                        "Accept": "application/json",
                        "Referer": "https://www.protondb.com/",
                    },
                )
    return _client


# ── Hash algorithm (reverse-engineered from ProtonDB frontend) ────────


def _get_hash(n1: int, n2: int, timestamp: int) -> str:
    return f"{n2}p{n1 * (n2 % timestamp)}"


def _get_protondb_id(hash_str: str) -> int:
    val = 0
    for c in hash_str + "m":
        val = ((val << 5) - val + ord(c)) & 0xFFFFFFFF
        if val > 0x7FFFFFFF:
            val -= 0x100000000
    return abs(val)


def _calculate_hashed_id(steam_app_id: int, num_reports: int, counts_timestamp: int) -> int:
    """Calculate the hashed app ID needed for the reports endpoint."""
    hash1 = _get_hash(steam_app_id, num_reports, counts_timestamp)
    hash2 = _get_hash(1, steam_app_id, counts_timestamp)
    hash3 = f"p{hash1}*vRT{hash2}undefined"
    return _get_protondb_id(hash3)


# ── API fetchers ──────────────────────────────────────────────────────


def fetch_counts() -> dict:
    """Fetch /data/counts.json (cached per session).

    Returns dict with keys: reports, timestamp, uniqueGames, etc.
    """
    global _counts
    if _counts is not None:
        return _counts

    with _counts_lock:
        if _counts is not None:
            return _counts

        client = _get_client()
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.get(_COUNTS_URL)
                resp.raise_for_status()
                _counts = resp.json()
                log.info(
                    "ProtonDB counts: %d reports, %d games, ts=%d",
                    _counts.get("reports", 0),
                    _counts.get("uniqueGames", 0),
                    _counts.get("timestamp", 0),
                )
                return _counts
            except Exception:
                log.warning(
                    "ProtonDB counts: attempt %d/%d failed",
                    attempt + 1, _MAX_RETRIES, exc_info=True,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError("Failed to fetch ProtonDB counts.json")


def fetch_reports(app_id: int) -> list[dict] | None:
    """Fetch up to 40 recent reports for a game with contributor data.

    Returns list of report dicts, or None on failure/404.
    Each report contains contributor.id, contributor.reportTally,
    contributor.steam.playtime, etc.
    """
    counts = fetch_counts()
    num_reports = counts["reports"]
    counts_ts = counts["timestamp"]

    hashed_id = _calculate_hashed_id(app_id, num_reports, counts_ts)
    url = _REPORTS_URL.format(hashed_id=hashed_id)

    client = _get_client()
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(url)
            if resp.status_code == 404:
                log.debug("ProtonDB reports: no data for app %d", app_id)
                return None
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.info("ProtonDB reports: rate limited for app %d, waiting %ds", app_id, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            reports = data.get("reports", [])
            return reports
        except httpx.TimeoutException:
            log.warning(
                "ProtonDB reports: timeout for app %d (attempt %d/%d)",
                app_id, attempt + 1, _MAX_RETRIES,
            )
            time.sleep(2 ** attempt)
        except Exception:
            log.warning(
                "ProtonDB reports: failed for app %d (attempt %d/%d)",
                app_id, attempt + 1, _MAX_RETRIES, exc_info=True,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    return None


def extract_contributor_data(report: dict) -> dict | None:
    """Extract contributor and matching fields from a single API report.

    Returns dict with:
        timestamp: int — for matching to DB reports
        gpu: str — for matching to DB reports
        contributor_id: str
        report_tally: int
        playtime: int (minutes)
        playtime_linux: int (minutes)
    Or None if contributor data is missing.
    """
    contributor = report.get("contributor")
    if not contributor:
        return None

    timestamp = report.get("timestamp")
    if not timestamp:
        return None

    # GPU for matching (API uses device.inferred.steam.gpu)
    device = report.get("device", {}) or {}
    inferred = device.get("inferred", {}) or {}
    steam_hw = inferred.get("steam", {}) or {}
    gpu = steam_hw.get("gpu", "")

    steam = contributor.get("steam", {}) or {}

    return {
        "timestamp": timestamp,
        "gpu": gpu,
        "contributor_id": contributor.get("id"),
        "report_tally": contributor.get("reportTally", 0),
        "playtime": steam.get("playtime", 0),
        "playtime_linux": steam.get("playtimeLinux", 0),
    }
