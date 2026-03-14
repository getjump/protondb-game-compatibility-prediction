"""Application configuration.

All tunables are configurable via environment variables (or .env file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


# ── Paths & server ───────────────────────────────────────────────────

DEFAULT_DB_PATH = Path(os.environ.get("PROTONDB_DB_PATH", "./data/protondb.db"))
DEFAULT_PORT = _int("PROTONDB_PORT", 8080)
DEFAULT_MODEL_PATH = Path(os.environ.get("PROTONDB_MODEL_PATH", "./data/model.pkl"))
DEFAULT_EMBEDDINGS_PATH = Path(os.environ.get("PROTONDB_EMBEDDINGS_PATH", "./data/embeddings.npz"))


@dataclass
class Config:
    """Central configuration object passed through the application."""

    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)
    port: int = DEFAULT_PORT
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    embeddings_path: Path = field(default_factory=lambda: DEFAULT_EMBEDDINGS_PATH)


# ── Normalized data source ────────────────────────────────────────────
# "heuristic" (default, fast regex) or "llm" (LLM-processed tables)
NORMALIZED_DATA_SOURCE = os.environ.get("NORMALIZED_DATA_SOURCE", "heuristic")

# ── Rate limits (requests per second) ────────────────────────────────

STEAM_RATE_LIMIT = _float("STEAM_RATE_LIMIT", 0.6)       # ~200 req/5min (Valve hard limit)
PROTONDB_RATE_LIMIT = _float("PROTONDB_RATE_LIMIT", 3.0)  # CDN-served static JSON
PCGW_RATE_LIMIT = _float("PCGW_RATE_LIMIT", 2.0)          # MediaWiki, no hard read limit

# ── Batch sizes ──────────────────────────────────────────────────────

CLEANING_BATCH_SIZE = _int("CLEANING_BATCH_SIZE", 500)
ENRICHMENT_BATCH_SIZE = _int("ENRICHMENT_BATCH_SIZE", 50)

# LLM batch sizes: _CLOUD for remote providers, _LOCAL for local llama.cpp/ollama
GPU_NORM_BATCH_CLOUD = _int("GPU_NORM_BATCH_CLOUD", 30)
GPU_NORM_BATCH_LOCAL = _int("GPU_NORM_BATCH_LOCAL", 5)
CPU_NORM_BATCH_CLOUD = _int("CPU_NORM_BATCH_CLOUD", 30)
CPU_NORM_BATCH_LOCAL = _int("CPU_NORM_BATCH_LOCAL", 5)
LAUNCH_OPT_BATCH_CLOUD = _int("LAUNCH_OPT_BATCH_CLOUD", 15)
LAUNCH_OPT_BATCH_LOCAL = _int("LAUNCH_OPT_BATCH_LOCAL", 3)
EXTRACT_BATCH_CLOUD = _int("EXTRACT_BATCH_CLOUD", 10)
EXTRACT_BATCH_LOCAL = _int("EXTRACT_BATCH_LOCAL", 3)
# ── LLM client defaults ─────────────────────────────────────────────

DEFAULT_LLM_CONCURRENCY = _int("LLM_CONCURRENCY", 10)
DEFAULT_LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 5)
DEFAULT_LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.0)
DEFAULT_LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 4096)

# ── Text fields in reports that contain user-written notes ───────────

REPORT_TEXT_FIELDS = [
    "concluding_notes", "notes_verdict", "notes_extra",
    "notes_customizations", "notes_launch_flags", "notes_variant",
    "notes_proton_version", "notes_concluding_notes",
    "notes_audio_faults", "notes_graphical_faults",
    "notes_performance_faults", "notes_stability_faults",
    "notes_windowing_faults", "notes_input_faults",
    "notes_significant_bugs", "notes_save_game_faults",
    "notes_anticheat", "notes_tinker_override",
    "notes_launcher", "notes_secondary_launcher",
]

# ── Risk override patterns for text extraction validation ────────────

RISKY_PATH_PREFIXES = ("/etc/", "/boot/", "~/.ssh/", "/usr/lib/", "/usr/bin/")
RISKY_COMMANDS = ("sudo", "rm -rf", "curl", "wget", "chmod -R 777", "chmod 777")
RISKY_SCOPES = ("sysctl", "grub", "modprobe", "udev")
