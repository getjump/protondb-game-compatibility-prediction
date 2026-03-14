"""Merge enrichment data from all sources into a single game_metadata row."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from protondb_settings.preprocessing.enrichment.models import (
    AWACYData,
    DeckData,
    PCGWData,
    ProtonDBData,
    ProtonGitHubData,
    SteamData,
)


def merge_metadata(
    app_id: int,
    steam: SteamData | None = None,
    deck: DeckData | None = None,
    pcgw: PCGWData | None = None,
    awacy: AWACYData | None = None,
    protondb: ProtonDBData | None = None,
    github: ProtonGitHubData | None = None,
) -> dict[str, Any]:
    """Merge data from all sources into a dict suitable for UPSERT into game_metadata.

    Missing sources result in NULL columns.
    """
    row: dict[str, Any] = {
        "app_id": app_id,
        "enriched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Steam Store
    if steam:
        row["developer"] = steam.developer
        row["publisher"] = steam.publisher
        row["genres"] = json.dumps(steam.genres) if steam.genres else None
        row["categories"] = json.dumps(steam.categories) if steam.categories else None
        row["release_date"] = steam.release_date
        row["has_linux_native"] = int(steam.has_linux_native)
    else:
        row.update(
            developer=None, publisher=None, genres=None, categories=None,
            release_date=None, has_linux_native=None,
        )

    # Deck Verified
    if deck:
        row["deck_status"] = deck.status
        row["deck_tests_json"] = json.dumps(deck.tests) if deck.tests else None
    else:
        row["deck_status"] = None
        row["deck_tests_json"] = None

    # PCGamingWiki
    if pcgw:
        row["engine"] = pcgw.engine
        row["graphics_apis"] = json.dumps(pcgw.graphics_apis) if pcgw.graphics_apis else None
        row["drm"] = json.dumps(pcgw.drm) if pcgw.drm else None
        # PCGamingWiki anticheat (prefer this if AWACY not available)
        if pcgw.anticheat:
            row.setdefault("anticheat", pcgw.anticheat)
    else:
        row.setdefault("engine", None)
        row.setdefault("graphics_apis", None)
        row.setdefault("drm", None)

    # AreWeAntiCheatYet
    if awacy:
        anticheats_str = ", ".join(ac for ac in awacy.anticheats if ac)
        row["anticheat"] = anticheats_str or row.get("anticheat")
        row["anticheat_status"] = awacy.status
    else:
        row.setdefault("anticheat", None)
        row["anticheat_status"] = None

    # ProtonDB Summary
    if protondb:
        row["protondb_tier"] = protondb.tier
        row["protondb_score"] = protondb.score
        row["protondb_confidence"] = protondb.confidence
        row["protondb_trending"] = protondb.trending_tier
    else:
        row.update(
            protondb_tier=None, protondb_score=None,
            protondb_confidence=None, protondb_trending=None,
        )

    # GitHub Proton Issues
    if github:
        row["github_issue_count"] = github.issue_count
        row["github_open_count"] = github.open_count
        row["github_closed_completed"] = github.closed_completed
        row["github_closed_not_planned"] = github.closed_not_planned
        row["github_has_regression"] = int(github.has_regression)
        row["github_latest_issue_date"] = github.latest_issue_date
    else:
        row.update(
            github_issue_count=None, github_open_count=None,
            github_closed_completed=None, github_closed_not_planned=None,
            github_has_regression=None, github_latest_issue_date=None,
        )

    return row
