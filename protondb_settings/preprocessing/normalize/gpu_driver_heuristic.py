"""GPU driver normalization via regex heuristics.

Two main formats:
  - NVIDIA: "NVIDIA 535.183.01" → vendor=nvidia, version=535.183.01
  - Mesa: "4.6 Mesa 22.0.0-devel (git-...)" → vendor=mesa, version=22.0.0
  - AMDGPU-PRO: "AMDGPU-PRO (Ver. 19.30)" → vendor=amdgpu-pro, version=19.30
"""

from __future__ import annotations

import logging
import re
import sqlite3

from protondb_settings.preprocessing.interrupt import shutdown_requested
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import get_pending_strings, upsert_rows

log = logging.getLogger(__name__)

_BATCH_SIZE = 500

# ── Regex patterns ──────────────────────────────────────────────────────────

# NVIDIA 535.183.01 or NVIDIA 440.44
_RE_NVIDIA = re.compile(
    r"NVIDIA\s+(\d{3,4})\.(\d+)(?:\.(\d+))?",
)

# Mesa version: "Mesa 22.0.0", "Mesa 21.0.3-devel", "Mesa 24.1.5-arch1.1"
_RE_MESA = re.compile(
    r"Mesa\s+(\d+)\.(\d+)\.(\d+)",
)

# AMDGPU-PRO driver
_RE_AMDGPU_PRO = re.compile(
    r"AMDGPU-?PRO[^0-9]*(\d+\.\d+)",
    re.IGNORECASE,
)

# AMD Compatibility Profile Context (old AMDGPU-PRO / Windows GL string)
# e.g. "4.6.13572 Compatibility Profile Context 5.0.73.19.30"
_RE_AMD_COMPAT = re.compile(
    r"(\d+\.\d+\.\d+)\s+Compatibility Profile Context(?:\s+(\d+[\d.]+))?",
)

# Bare NVIDIA version without "NVIDIA" prefix: "440.44", "440.44-2"
_RE_BARE_NVIDIA_VER = re.compile(
    r"^(\d{3,4})\.(\d{2,3})(?:\.(\d+))?(?:-\d+)?$",
)


def parse_gpu_driver(raw: str) -> dict:
    """Parse a raw gpu_driver string into a normalized dict."""
    result = {
        "raw_string": raw,
        "driver_vendor": "unknown",
        "driver_version": "unknown",
        "driver_major": None,
        "driver_minor": None,
        "driver_patch": None,
    }

    s = raw.strip()
    if not s:
        return result

    # ── NVIDIA ──
    m = _RE_NVIDIA.search(s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3)) if m.group(3) else None
        ver = f"{major}.{minor}"
        if patch is not None:
            ver += f".{patch:02d}"
        result["driver_vendor"] = "nvidia"
        result["driver_version"] = ver
        result["driver_major"] = major
        result["driver_minor"] = minor
        result["driver_patch"] = patch
        return result

    # ── Mesa ──
    m = _RE_MESA.search(s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3))
        result["driver_vendor"] = "mesa"
        result["driver_version"] = f"{major}.{minor}.{patch}"
        result["driver_major"] = major
        result["driver_minor"] = minor
        result["driver_patch"] = patch
        return result

    # ── AMDGPU-PRO ──
    m = _RE_AMDGPU_PRO.search(s)
    if m:
        ver = m.group(1)
        parts = ver.split(".")
        result["driver_vendor"] = "amdgpu-pro"
        result["driver_version"] = ver
        result["driver_major"] = int(parts[0]) if len(parts) > 0 else None
        result["driver_minor"] = int(parts[1]) if len(parts) > 1 else None
        return result

    # ── AMD Compatibility Profile Context ──
    m = _RE_AMD_COMPAT.search(s)
    if m:
        gl_ver = m.group(1)  # "4.6.13572"
        driver_ver = m.group(2)  # "5.0.73" or None
        result["driver_vendor"] = "amdgpu-pro"
        result["driver_version"] = driver_ver or gl_ver
        return result

    # ── Bare NVIDIA version (no prefix) ──
    m = _RE_BARE_NVIDIA_VER.match(s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3)) if m.group(3) else None
        # NVIDIA driver versions have major >= 300
        if major >= 300:
            ver = f"{major}.{minor}"
            if patch is not None:
                ver += f".{patch:02d}"
            result["driver_vendor"] = "nvidia"
            result["driver_version"] = ver
            result["driver_major"] = major
            result["driver_minor"] = minor
            result["driver_patch"] = patch
            return result

    return result


def get_pending_driver_strings(conn: sqlite3.Connection) -> list[str]:
    """Get distinct gpu_driver strings not yet in gpu_driver_normalization."""
    return get_pending_strings(
        conn, source_table="reports", source_column="gpu_driver",
        norm_table="gpu_driver_normalization",
    )


def get_pending_count(conn: sqlite3.Connection) -> int:
    return len(get_pending_driver_strings(conn))


def normalize_gpu_drivers(
    conn: sqlite3.Connection,
    *,
    force: bool = False,
) -> int:
    """Run GPU driver normalization via regex heuristics."""
    if force:
        log.info("Force GPU driver normalization: clearing table")
        conn.execute("DELETE FROM gpu_driver_normalization")
        conn.commit()

    pending = get_pending_driver_strings(conn)
    if not pending:
        log.info("GPU driver normalization: nothing to do")
        return 0

    log.info("GPU driver normalization: %d strings to process", len(pending))

    processed = 0
    with PipelineStep(conn, "normalize_gpu_driver", len(pending)) as step:
        for batch in chunked(pending, _BATCH_SIZE):
            if shutdown_requested.is_set():
                log.info("GPU driver normalization: interrupted at %d", processed)
                break

            all_rows = [parse_gpu_driver(s) for s in batch]

            if all_rows:
                upsert_rows(conn, "gpu_driver_normalization", all_rows, "raw_string")
                conn.commit()

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
