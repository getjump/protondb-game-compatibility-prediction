"""Fetch issues from ValveSoftware/Proton GitHub repo via `gh` CLI.

Extracts Steam app_id from issue titles and bodies, groups by app_id,
and returns aggregated issue data per game.

Requires `gh` CLI to be installed and authenticated.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections import defaultdict

from protondb_settings.preprocessing.enrichment.models import (
    ProtonGitHubData,
    ProtonGitHubIssue,
)

log = logging.getLogger(__name__)

_REPO = "ValveSoftware/Proton"

# Patterns to extract Steam app_id
_RE_TITLE_APPID = re.compile(r"\((\d{4,8})\)\s*$")
_RE_BODY_APPID = re.compile(
    r"(?:Steam\s+)?App\s*ID\s*(?:of\s+the\s+game)?[:\s]+(\d{4,8})", re.IGNORECASE,
)


def _extract_app_id(title: str, body: str | None) -> int | None:
    """Extract Steam app_id from issue title or body.

    Patterns:
    - Title: "Game Name (123456)"
    - Body: "Steam AppID of the game: 123456"
    """
    # Try title first — most reliable
    m = _RE_TITLE_APPID.search(title)
    if m:
        return int(m.group(1))

    # Try body
    if body:
        m = _RE_BODY_APPID.search(body[:500])  # only search first 500 chars
        if m:
            return int(m.group(1))

    return None


def fetch_all_issues(*, limit: int = 10000) -> list[dict]:
    """Fetch all issues from ValveSoftware/Proton using `gh` CLI.

    Returns raw list of issue dicts with fields:
    number, title, body, state, labels, createdAt, closedAt.
    """
    log.info("Fetching issues from %s (limit=%d)...", _REPO, limit)

    cmd = [
        "gh", "issue", "list",
        "-R", _REPO,
        "--state", "all",
        "--limit", str(limit),
        "--json", "number,title,body,state,stateReason,labels,createdAt,closedAt",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"gh issue list failed: {result.stderr.strip()}")

    issues = json.loads(result.stdout)
    log.info("Fetched %d issues from %s", len(issues), _REPO)
    return issues


def build_github_index(
    raw_issues: list[dict],
) -> dict[int, ProtonGitHubData]:
    """Parse raw issues and group by Steam app_id.

    Returns dict mapping app_id → ProtonGitHubData.
    Issues without a recognizable app_id are skipped.
    """
    by_app: dict[int, list[ProtonGitHubIssue]] = defaultdict(list)
    matched = 0
    skipped = 0

    for raw in raw_issues:
        title = raw.get("title", "")
        body = raw.get("body", "")
        app_id = _extract_app_id(title, body)

        if app_id is None:
            skipped += 1
            continue

        matched += 1
        labels = [lbl["name"] for lbl in raw.get("labels", []) if isinstance(lbl, dict)]

        by_app[app_id].append(ProtonGitHubIssue(
            number=raw["number"],
            title=title,
            state=raw.get("state", "").upper(),
            state_reason=raw.get("stateReason"),
            labels=labels,
            created_at=raw.get("createdAt"),
            closed_at=raw.get("closedAt"),
        ))

    log.info(
        "GitHub issues: %d matched to app_id, %d skipped (no app_id), %d unique games",
        matched, skipped, len(by_app),
    )

    result: dict[int, ProtonGitHubData] = {}
    for app_id, issues in by_app.items():
        open_count = sum(1 for i in issues if i.state == "OPEN")
        closed_completed = sum(
            1 for i in issues
            if i.state == "CLOSED" and (i.state_reason or "").upper() == "COMPLETED"
        )
        closed_not_planned = sum(
            1 for i in issues
            if i.state == "CLOSED" and (i.state_reason or "").upper() == "NOT_PLANNED"
        )
        closed_duplicate = sum(
            1 for i in issues
            if i.state == "CLOSED" and (i.state_reason or "").upper() == "DUPLICATE"
        )

        all_labels: list[str] = []
        for issue in issues:
            all_labels.extend(issue.labels)
        unique_labels = sorted(set(all_labels))

        has_regression = any("Regression" in lbl for lbl in unique_labels)

        # Latest issue date
        dates = [i.created_at for i in issues if i.created_at]
        latest = max(dates) if dates else None

        result[app_id] = ProtonGitHubData(
            issue_count=len(issues),
            open_count=open_count,
            closed_completed=closed_completed,
            closed_not_planned=closed_not_planned,
            closed_duplicate=closed_duplicate,
            has_regression=has_regression,
            labels=unique_labels,
            latest_issue_date=latest,
            issues=issues,
        )

    return result
