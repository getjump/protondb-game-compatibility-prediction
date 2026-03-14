"""Layer 1: Deterministic regex spotting — pre-extracts entities from text before LLM.

Results are passed into the LLM prompt as hints to reduce hallucination and
improve accuracy.
"""

from __future__ import annotations

import re

# Pattern registry: name -> compiled regex
PATTERNS: dict[str, re.Pattern] = {
    "env_var": re.compile(r"\b([A-Z][A-Z0-9_]{2,})=([A-Za-z0-9_.\-/]+)"),
    "proton_version": re.compile(
        r"(?:GE-)?Proton[\s-]*[\d.]+(?:-GE-\d+)?|Proton\s+Experimental",
        re.IGNORECASE,
    ),
    "wine_version": re.compile(r"Wine[\s-]*\d+\.\d+", re.IGNORECASE),
    "wrapper_tool": re.compile(
        r"\b(gamescope|mangohud|gamemoderun|prime-run|protontricks|winetricks)\b",
        re.IGNORECASE,
    ),
    "game_arg": re.compile(
        r"(?<=\s)-(?:dx\d+|vulkan|windowed|fullscreen|skipintro|nointro"
        r"|nobattleye|force-[\w]+)\b",
        re.IGNORECASE,
    ),
    "file_path": re.compile(
        r"[~/][\w./\\-]+\.(?:ini|cfg|conf|json|xml|dll|exe|so|reg)"
    ),
    "package": re.compile(
        r"\b(?:vcrun\d+|dotnet\d+|d3dcompiler_\d+|dxvk|vkd3d|mf|faudio)\b",
        re.IGNORECASE,
    ),
    "dll_override": re.compile(r"\b\w+\.dll\b", re.IGNORECASE),
}


def spot_entities(text: str) -> dict[str, list[str]]:
    """Run all regex patterns against *text* and return matched entities.

    Returns a dict mapping pattern name -> list of unique matched strings.
    """
    if not text:
        return {}

    found: dict[str, list[str]] = {}
    for name, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # findall returns tuples for groups > 1 (e.g. env_var)
            if isinstance(matches[0], tuple):
                # For env_var: join KEY=VALUE
                unique = list(dict.fromkeys(f"{m[0]}={m[1]}" for m in matches))
            else:
                unique = list(dict.fromkeys(matches))
            if unique:
                found[name] = unique

    return found


def format_spotted(spotted: dict[str, list[str]]) -> str:
    """Format spotted entities into a human-readable string for the LLM prompt."""
    if not spotted:
        return "none"

    parts = []
    for category, items in spotted.items():
        label = category.replace("_", " ")
        joined = ", ".join(items[:10])  # Limit to 10 per category
        if len(items) > 10:
            joined += f" (+{len(items) - 10} more)"
        parts.append(f"{label}: {joined}")

    return "; ".join(parts)
