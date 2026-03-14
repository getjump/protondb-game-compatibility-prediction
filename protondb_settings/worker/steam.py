"""Steam Store API client for game name lookups.

Currently a placeholder that returns the title from the dump data.
A full implementation would query the Steam Store API with rate limiting.
"""

from __future__ import annotations

import asyncio
import time

import httpx


_last_request_time: float = 0.0
_MIN_INTERVAL: float = 1.0  # 1 request per second


async def fetch_game_name(app_id: int) -> str | None:
    """Fetch the game name from the Steam Store API.

    Applies rate limiting of 1 request per second.  Returns *None* when
    the store returns no useful data (e.g. removed/hidden games).
    """
    global _last_request_time

    now = time.monotonic()
    wait = _MIN_INTERVAL - (now - _last_request_time)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_request_time = time.monotonic()

    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    entry = data.get(str(app_id))
    if not entry or not entry.get("success"):
        return None

    return entry.get("data", {}).get("name")
