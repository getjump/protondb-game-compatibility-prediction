"""Text extraction prompt for LLM — extracts actions and observations from report text."""

from __future__ import annotations

from typing import Any

from protondb_settings.config import REPORT_TEXT_FIELDS

SYSTEM_PROMPT = """\
You are an expert extractor for Linux gaming compatibility reports from ProtonDB.
You extract actionable information and observations from free-text user reports.
Return valid JSON only, no explanations."""

_ACTION_TYPES = (
    "env_var|game_arg|wrapper_config|runner_selection|protontricks_verb|"
    "dll_override|prefix_action|file_patch|registry_patch|executable_override|"
    "dependency_install|session_requirement|system_tweak"
)

_SYMPTOM_TYPES = (
    "crash_on_launch|black_screen|stutter|no_audio|controller_issue|"
    "launcher_crash|anti_cheat_fail|other"
)


def format_prompt(
    *,
    title: str = "Unknown",
    engine: str | None = None,
    graphics_apis: str | None = None,
    anticheat: str | None = None,
    gpu: str | None = None,
    cpu: str | None = None,
    os_name: str | None = None,
    kernel: str | None = None,
    proton_version: str | None = None,
    variant: str | None = None,
    custom_proton_version: str | None = None,
    launcher: str | None = None,
    active_customizations: str = "none",
    active_launch_flags: str = "none",
    fault_summary: str = "none",
    pre_extracted_entities: str = "none",
    combined_text: str = "",
) -> str:
    """Format the text extraction prompt with full context."""
    return f"""\
Extract actionable Linux gaming compatibility information from this ProtonDB report.

Game: {title}
Engine: {engine or "unknown"} | Graphics API: {graphics_apis or "unknown"} | \
Anti-cheat: {anticheat or "none"}
Hardware: {gpu or "unknown"}, {cpu or "unknown"}, {os_name or "unknown"}, \
kernel {kernel or "unknown"}
Proton: {proton_version or "unknown"} ({variant or "unknown"}) | \
Custom: {custom_proton_version or "none"}
Launcher: {launcher or "unknown"}
Structured data already known:
  - Customizations: {active_customizations}
  - Launch flags: {active_launch_flags}
  - Faults: {fault_summary}
Detected entities (regex): {pre_extracted_entities}

User text:
---
{combined_text}
---

Return JSON only:
{{
  "actions": [
    {{
      "type": "{_ACTION_TYPES}",
      "value": "exact value from text",
      "detail": "additional context if needed or null",
      "reported_effect": "effective|ineffective|unclear",
      "conditions": [{{"kind": "gpu_vendor|symptom|display_server|distro|proton_version", \
"value": "..."}}],
      "risk": "safe|risky"
    }}
  ],
  "observations": [
    {{
      "symptom": "{_SYMPTOM_TYPES}",
      "description": "short text",
      "hardware_specific": true/false
    }}
  ],
  "useful": true/false
}}

Rules:
- Extract ONLY what is explicitly stated. Never infer actions not mentioned.
- Distinguish "X helped" (effective) from "X didn't help" (ineffective) -- CRITICAL.
- Don't duplicate what's already in "Structured data already known".
- env_var: exact KEY=VALUE. Common: PROTON_ENABLE_NVAPI, PROTON_USE_WINED3D, DXVK_ASYNC, \
VKD3D_CONFIG, WINE_FULLSCREEN_FSR, PROTON_NO_ESYNC/FSYNC.
- game_arg: flags for game exe (-dx11, -vulkan, -windowed, -skipintro), NOT env vars.
- wrapper_config: full gamescope/mangohud command with flags.
- runner_selection: exact version string (GE-Proton9-25, Proton Experimental).
- protontricks_verb: exact verb name (vcrun2019, dotnet48, d3dcompiler_47).
- file_patch: include file path and what to change.
- risk=risky: anything requiring sudo, system-wide changes, /etc, GRUB, kernel params, \
disabling security.
- "works fine" / "no issues" with no details -> useful=false, empty actions.
- Observations are gold -- capture symptoms even without fix.
- conditions: only add if the action is explicitly linked to a condition in the text.
"""


def build_context_from_report(report: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, str]:
    """Build context dict from a report row and optional game_metadata for prompt formatting."""
    # Active customizations
    cust_fields = [
        ("cust_winetricks", "winetricks"),
        ("cust_protontricks", "protontricks"),
        ("cust_config_change", "config change"),
        ("cust_custom_prefix", "custom prefix"),
        ("cust_custom_proton", "custom proton"),
        ("cust_lutris", "lutris"),
        ("cust_media_foundation", "media foundation"),
        ("cust_protonfixes", "protonfixes"),
        ("cust_native2proton", "native2proton"),
    ]
    active_custs = [label for key, label in cust_fields if report.get(key)]
    active_customizations = ", ".join(active_custs) if active_custs else "none"

    # Active launch flags
    flag_fields = [
        ("flag_use_wine_d3d11", "PROTON_USE_WINED3D=1"),
        ("flag_disable_esync", "PROTON_NO_ESYNC=1"),
        ("flag_enable_nvapi", "PROTON_ENABLE_NVAPI=1"),
        ("flag_disable_fsync", "PROTON_NO_FSYNC=1"),
        ("flag_use_wine_d9vk", "PROTON_USE_WINE_D9VK=1"),
        ("flag_large_address_aware", "PROTON_LARGE_ADDRESS_AWARE=1"),
        ("flag_disable_d3d11", "PROTON_NO_D3D11=1"),
        ("flag_hide_nvidia", "PROTON_HIDE_NVIDIA_GPU=1"),
    ]
    active_flags = [label for key, label in flag_fields if report.get(key)]
    active_launch_flags = ", ".join(active_flags) if active_flags else "none"

    # Fault summary
    fault_fields = [
        "audio_faults", "graphical_faults", "input_faults",
        "performance_faults", "stability_faults", "windowing_faults",
        "save_game_faults", "significant_bugs",
    ]
    active_faults = [f.replace("_", " ") for f in fault_fields if report.get(f) == "yes"]
    fault_summary = ", ".join(active_faults) if active_faults else "none"

    # Combined text from all notes fields
    text_fields = REPORT_TEXT_FIELDS
    texts = []
    for field in text_fields:
        val = report.get(field)
        if val and isinstance(val, str) and len(val.strip()) > 0:
            label = field.replace("notes_", "").replace("_", " ")
            texts.append(f"[{label}]: {val.strip()}")
    combined_text = "\n".join(texts)

    ctx: dict[str, str] = {
        "title": (metadata or {}).get("name") or "Unknown",
        "engine": (metadata or {}).get("engine"),
        "graphics_apis": (metadata or {}).get("graphics_apis"),
        "anticheat": (metadata or {}).get("anticheat"),
        "gpu": report.get("gpu"),
        "cpu": report.get("cpu"),
        "os_name": report.get("os"),
        "kernel": report.get("kernel"),
        "proton_version": report.get("proton_version"),
        "variant": report.get("variant"),
        "custom_proton_version": report.get("custom_proton_version"),
        "launcher": report.get("launcher"),
        "active_customizations": active_customizations,
        "active_launch_flags": active_launch_flags,
        "fault_summary": fault_summary,
        "combined_text": combined_text,
    }
    return ctx
