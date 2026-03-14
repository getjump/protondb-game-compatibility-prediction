"""Pydantic models for enrichment data from external APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SteamData(BaseModel):
    """Data from Steam Store API ``appdetails``."""

    developer: str | None = None
    publisher: str | None = None
    genres: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    release_date: str | None = None
    has_linux_native: bool = False


class DeckData(BaseModel):
    """Data from Steam Deck Verified API."""

    status: int = 0  # 0=unknown, 1=unsupported, 2=playable, 3=verified
    tests: list[dict] = Field(default_factory=list)


class PCGWData(BaseModel):
    """Data from PCGamingWiki Cargo API."""

    engine: str | None = None
    graphics_apis: list[str] = Field(default_factory=list)
    anticheat: str | None = None
    drm: list[str] | None = None


class AWACYData(BaseModel):
    """Data from AreWeAntiCheatYet."""

    anticheats: list[str] = Field(default_factory=list)
    status: str | None = None


class ProtonDBData(BaseModel):
    """Data from ProtonDB Summary API."""

    tier: str | None = None  # platinum/gold/silver/bronze/borked
    score: float | None = None
    confidence: str | None = None  # strong/good/weak
    trending_tier: str | None = None


class ProtonGitHubIssue(BaseModel):
    """Single issue from ValveSoftware/Proton GitHub repo."""

    number: int
    title: str = ""
    state: str = ""  # OPEN / CLOSED
    state_reason: str | None = None  # completed / not_planned / duplicate / reopened
    labels: list[str] = Field(default_factory=list)
    created_at: str | None = None
    closed_at: str | None = None


class ProtonGitHubData(BaseModel):
    """Aggregated GitHub issues data for a single app_id."""

    issue_count: int = 0
    open_count: int = 0
    closed_completed: int = 0  # fixed bugs
    closed_not_planned: int = 0  # won't fix
    closed_duplicate: int = 0
    has_regression: bool = False
    labels: list[str] = Field(default_factory=list)
    latest_issue_date: str | None = None
    issues: list[ProtonGitHubIssue] = Field(default_factory=list)
