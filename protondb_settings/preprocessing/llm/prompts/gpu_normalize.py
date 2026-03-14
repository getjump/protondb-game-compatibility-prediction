"""GPU normalization prompt for LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a GPU hardware classifier for Linux gaming compatibility reports.
You normalize raw GPU strings into structured JSON.
Return valid JSON only, no explanations."""


def format_single_prompt(raw_gpu: str) -> str:
    """Format prompt for a single GPU string."""
    return f"""\
Normalize this GPU string from a Linux gaming compatibility report.
Return JSON only.

Input: "{raw_gpu}"

{{
  "vendor": "nvidia|amd|intel|unknown",
  "family": "string",
  "model": "string",
  "normalized_name": "string",
  "is_apu": true/false,
  "is_igpu": true/false,
  "is_virtual": true/false
}}

Rules:
- "nouveau NVxx" = NVIDIA (open-source driver). Parse chip code to family.
- "llvmpipe", "virgl", "VMware" = virtual GPU, vendor from context or "unknown".
- If the string is clearly not a GPU (OS name, random text) -> all fields "unknown", \
is_apu/is_igpu/is_virtual = false.
- Preserve all info. Never discard unusual hardware (APUs, embedded, server GPUs).
- family examples: "rtx30", "rtx40", "gtx10", "gtx16", "rdna2", "rdna3", "gcn4", \
"hd600", "uhd700", "custom_apu", "unknown"
- model examples: "rtx3070", "rx6800xt", "steam_deck_apu", "i7_12700k", "unknown"
- normalized_name: human-readable like "NVIDIA RTX 3070", "AMD Custom APU (Steam Deck)"
"""


def format_batch_prompt(raw_gpus: list[str]) -> str:
    """Format prompt for a batch of GPU strings."""
    numbered = "\n".join(f'{i+1}. "{gpu}"' for i, gpu in enumerate(raw_gpus))
    return f"""\
Normalize these GPU strings from Linux gaming compatibility reports.
Return a JSON array with one result object per input, in the same order.

Inputs:
{numbered}

Each result must have:
{{
  "vendor": "nvidia|amd|intel|unknown",
  "family": "string",
  "model": "string",
  "normalized_name": "string",
  "is_apu": true/false,
  "is_igpu": true/false,
  "is_virtual": true/false
}}

Return format: {{"results": [...]}}

Rules:
- "nouveau NVxx" = NVIDIA (open-source driver). Parse chip code to family.
- "llvmpipe", "virgl", "VMware" = virtual GPU, vendor from context or "unknown".
- If the string is clearly not a GPU (OS name, random text) -> all fields "unknown", \
is_apu/is_igpu/is_virtual = false.
- Preserve all info. Never discard unusual hardware (APUs, embedded, server GPUs).
- family examples: "rtx30", "rtx40", "gtx10", "gtx16", "rdna2", "rdna3", "gcn4", \
"hd600", "uhd700", "custom_apu", "unknown"
- model examples: "rtx3070", "rx6800xt", "steam_deck_apu", "unknown"
- normalized_name: human-readable like "NVIDIA RTX 3070", "AMD Custom APU (Steam Deck)"
"""
