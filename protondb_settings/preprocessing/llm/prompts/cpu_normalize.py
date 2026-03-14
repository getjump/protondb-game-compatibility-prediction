"""CPU normalization prompt for LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a CPU hardware classifier for Linux gaming compatibility reports.
You normalize raw CPU strings into structured JSON.
Return valid JSON only, no explanations."""


def format_single_prompt(raw_cpu: str) -> str:
    """Format prompt for a single CPU string."""
    return f"""\
Normalize this CPU string from a Linux gaming report.
Return JSON only.

Input: "{raw_cpu}"

{{
  "vendor": "intel|amd|unknown",
  "family": "string",
  "model": "string",
  "normalized_name": "string",
  "generation": null|int,
  "is_apu": true/false
}}

Rules:
- "VirtualApple" -> vendor="unknown" (virtual/emulated).
- Random garbage ("0x0", "Spicy Silicon") -> all "unknown", generation=null, is_apu=false.
- Custom APUs (Steam Deck, ROG Ally) -> is_apu=true, appropriate family.
- family examples: "zen3", "zen4", "alder_lake", "raptor_lake", "custom_apu", "unknown"
- model examples: "ryzen7_5800x", "i7_12700k", "steam_deck_apu", "unknown"
- generation: Intel 12th gen -> 12, AMD Zen 3 -> 3, unknown -> null
"""


def format_batch_prompt(raw_cpus: list[str]) -> str:
    """Format prompt for a batch of CPU strings."""
    numbered = "\n".join(f'{i+1}. "{cpu}"' for i, cpu in enumerate(raw_cpus))
    return f"""\
Normalize these CPU strings from Linux gaming compatibility reports.
Return a JSON array with one result object per input, in the same order.

Inputs:
{numbered}

Each result must have:
{{
  "vendor": "intel|amd|unknown",
  "family": "string",
  "model": "string",
  "normalized_name": "string",
  "generation": null|int,
  "is_apu": true/false
}}

Return format: {{"results": [...]}}

Rules:
- "VirtualApple" -> vendor="unknown" (virtual/emulated).
- Random garbage ("0x0", "Spicy Silicon") -> all "unknown", generation=null, is_apu=false.
- Custom APUs (Steam Deck, ROG Ally) -> is_apu=true, appropriate family.
- family examples: "zen3", "zen4", "alder_lake", "raptor_lake", "custom_apu", "unknown"
- model examples: "ryzen7_5800x", "i7_12700k", "steam_deck_apu", "unknown"
- generation: Intel 12th gen -> 12, AMD Zen 3 -> 3, unknown -> null
"""
