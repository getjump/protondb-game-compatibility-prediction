"""PCGamingWiki Cargo API client with batch support and retry."""

from __future__ import annotations

import logging
import threading
import time

import httpx

from protondb_settings.preprocessing.enrichment.models import PCGWData

log = logging.getLogger(__name__)

_API_URL = "https://www.pcgamingwiki.com/w/api.php"

_MAX_RETRIES = 3
_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(timeout=60, follow_redirects=True)
    return _client


def _parse_graphics_apis(title: dict) -> list[str]:
    """Parse graphics API fields from PCGamingWiki cargo row."""
    apis = []
    for field, label in [
        ("Direct3D versions", "DirectX"),
        ("Direct3D_versions", "DirectX"),
        ("Vulkan versions", "Vulkan"),
        ("Vulkan_versions", "Vulkan"),
        ("OpenGL versions", "OpenGL"),
        ("OpenGL_versions", "OpenGL"),
    ]:
        val = title.get(field)
        if val and val.strip() and val.strip().lower() not in ("", "false"):
            # Vulkan_versions returns "true" (string) rather than a version number
            if val.strip().lower() == "true":
                apis.append(label)
            else:
                apis.append(f"{label} {val.strip()}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for api in apis:
        key = api.split()[0]  # Dedupe by label prefix
        if key not in seen:
            seen.add(key)
            deduped.append(api)
    return deduped


def fetch_pcgw_batch(app_ids: list[int]) -> dict[int, PCGWData]:
    """Fetch PCGamingWiki data for a batch of app_ids (max ~10 per request).

    Uses the Cargo API with OR in WHERE clause to batch multiple app_ids.
    Returns a dict mapping app_id -> PCGWData.
    """
    if not app_ids:
        return {}

    client = _get_client()
    where_clause = " OR ".join(
        f'Infobox_game.Steam_AppID HOLDS "{aid}"' for aid in app_ids
    )
    params = {
        "action": "cargoquery",
        "tables": "Infobox_game,API,Middleware,Availability",
        "join_on": (
            "Infobox_game._pageName=API._pageName,"
            "Infobox_game._pageName=Middleware._pageName,"
            "Infobox_game._pageName=Availability._pageName"
        ),
        "fields": (
            "Infobox_game.Steam_AppID,"
            "Infobox_game.Engines,"
            "API.Direct3D_versions,"
            "API.Vulkan_versions,"
            "API.OpenGL_versions,"
            "Middleware.Anticheat,"
            "Availability.Uses_DRM"
        ),
        "where": where_clause,
        "limit": "500",
        "format": "json",
    }

    data = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.get(_API_URL, params=params)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                log.info("PCGamingWiki: rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except httpx.TimeoutException:
            log.warning("PCGamingWiki: timeout (attempt %d/%d) for %s", attempt + 1, _MAX_RETRIES, app_ids)
            time.sleep(2 ** attempt)
        except Exception:
            log.warning("PCGamingWiki: request failed (attempt %d/%d) for %s", attempt + 1, _MAX_RETRIES, app_ids, exc_info=True)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    if data is None:
        return {}

    results: dict[int, PCGWData] = {}
    for row in data.get("cargoquery", []):
        try:
            title = row.get("title", {})
            raw_id = title.get("Steam AppID") or title.get("Steam_AppID") or ""
            # Steam_AppID can be comma-separated ("1091500,1495710")
            for sid in raw_id.split(","):
                sid = sid.strip()
                if not sid.isdigit():
                    continue
                steam_id = int(sid)

                # Engine comes with "Engine:" prefix
                engine_raw = (title.get("Engines") or "").removeprefix("Engine:").strip()
                engine = engine_raw or None

                # DRM: comma-separated, filter "DRM-free"
                drm_raw = title.get("Uses DRM") or title.get("Uses_DRM") or ""
                drm_list = [
                    d.strip()
                    for d in drm_raw.split(",")
                    if d.strip() and d.strip().lower() != "drm-free"
                ]

                anticheat = (title.get("Anticheat") or "").strip() or None

                # Use setdefault: first row from Cargo is the primary record
                # (JOIN can produce duplicates, e.g. CS2: Source 2 + Source)
                results.setdefault(
                    steam_id,
                    PCGWData(
                        engine=engine,
                        graphics_apis=_parse_graphics_apis(title),
                        anticheat=anticheat,
                        drm=drm_list or None,
                    ),
                )
        except Exception:
            log.debug("PCGamingWiki: failed to parse row %s", row, exc_info=True)
            continue

    return results


