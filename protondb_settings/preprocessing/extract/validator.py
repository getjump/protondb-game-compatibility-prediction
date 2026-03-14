"""Layer 3: Post-validation of LLM extraction results.

Applies:
1. Pydantic validation (JSON structure, enum values)
2. Risk override (sudo, /etc, rm -rf -> force risky)
3. Scope validation (file_patch paths)
4. Sanitization (env_var values, path traversal)
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from protondb_settings.config import RISKY_COMMANDS, RISKY_PATH_PREFIXES, RISKY_SCOPES
from protondb_settings.preprocessing.extract.models import (
    Action,
    ExtractionResult,
    Observation,
)

log = logging.getLogger(__name__)

# Valid env var value pattern
_ENV_VAR_VALUE_RE = re.compile(r"^[A-Za-z0-9_=,.\-/: ]+$")

# Path traversal detection
_PATH_TRAVERSAL_RE = re.compile(r"\.\.")

# Pre-compiled word-boundary patterns for risky commands
_RISKY_CMD_PATTERNS = [
    re.compile(rf"\b{re.escape(cmd)}\b", re.IGNORECASE)
    for cmd in RISKY_COMMANDS
]

_RISKY_SCOPE_PATTERNS = [
    re.compile(rf"\b{re.escape(scope)}\b", re.IGNORECASE)
    for scope in RISKY_SCOPES
]


def _check_risk_override(action: Action) -> Action:
    """Force risk=risky if action contains risky patterns."""
    combined = f"{action.value} {action.detail or ''}"

    # Check risky paths
    for prefix in RISKY_PATH_PREFIXES:
        if prefix in combined:
            action.risk = "risky"
            return action

    # Check risky commands (word boundaries to avoid false positives)
    for pattern in _RISKY_CMD_PATTERNS:
        if pattern.search(combined):
            action.risk = "risky"
            return action

    # Check risky scopes (word boundaries)
    for pattern in _RISKY_SCOPE_PATTERNS:
        if pattern.search(combined):
            action.risk = "risky"
            return action

    # system_tweak and dependency_install are always risky
    if action.type in ("system_tweak", "dependency_install"):
        action.risk = "risky"

    return action


def _validate_scope(action: Action) -> Action:
    """Validate file_patch paths — should be within game dir or ~/.steam/."""
    if action.type != "file_patch":
        return action

    path = action.value
    safe_prefixes = (
        "~/.steam/", "~/.local/share/Steam/",
        "~/.config/", "~/.",
        # Relative paths within game directory
        "./", "../",
    )

    is_safe = any(path.startswith(p) for p in safe_prefixes)
    if not is_safe and path.startswith("/"):
        action.risk = "risky"

    return action


def _sanitize_action(action: Action) -> Action | None:
    """Sanitize action values. Returns None if the action should be dropped."""
    # Check for path traversal in file paths
    if action.type == "file_patch" and _PATH_TRAVERSAL_RE.search(action.value):
        action.risk = "risky"

    # Sanitize env_var values
    if action.type == "env_var":
        # Value should match KEY=VALUE pattern
        if "=" in action.value:
            key, _, val = action.value.partition("=")
            if val and not _ENV_VAR_VALUE_RE.match(val):
                log.warning("Suspicious env_var value: %s", action.value)
                action.risk = "risky"

    return action


def validate_extraction(raw_result: dict | None) -> ExtractionResult:
    """Validate and sanitize a raw LLM extraction result.

    Returns a validated ExtractionResult, potentially with modified risk levels
    and dropped invalid actions.
    """
    if raw_result is None:
        return ExtractionResult()

    # Step 1: Pydantic validation
    try:
        result = ExtractionResult.model_validate(raw_result)
    except ValidationError as e:
        log.warning("Extraction validation failed: %s", e)
        # Try to salvage what we can
        actions = []
        for raw_action in raw_result.get("actions", []):
            try:
                actions.append(Action.model_validate(raw_action))
            except ValidationError:
                continue

        observations = []
        for raw_obs in raw_result.get("observations", []):
            try:
                observations.append(Observation.model_validate(raw_obs))
            except ValidationError:
                continue

        result = ExtractionResult(
            actions=actions,
            observations=observations,
            useful=bool(raw_result.get("useful", False)),
        )

    # Step 2: Risk override
    result.actions = [_check_risk_override(a) for a in result.actions]

    # Step 3: Scope validation
    result.actions = [_validate_scope(a) for a in result.actions]

    # Step 4: Sanitization
    sanitized_actions = []
    for action in result.actions:
        sanitized = _sanitize_action(action)
        if sanitized is not None:
            sanitized_actions.append(sanitized)
    result.actions = sanitized_actions

    return result
