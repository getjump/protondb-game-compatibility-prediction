"""Hardware features from reports + normalization tables."""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from .encoding import (
    extract_cpu_family,
    extract_cpu_generation,
    extract_cpu_vendor,
    extract_gpu_family,
    extract_gpu_vendor,
    gpu_tier_from_family,
    os_family_from_string,
)

log = logging.getLogger(__name__)

# Table names per normalized data source
_GPU_TABLES = {
    "heuristic": "gpu_normalization_heuristic",
    "llm": "gpu_normalization",
}
_CPU_TABLES = {
    "heuristic": "cpu_normalization_heuristic",
    "llm": "cpu_normalization",
}


def _parse_ram_gb(ram_str: str | None, ram_mb: int | None) -> float | None:
    """Parse RAM to GB from ram text or ram_mb column."""
    if ram_mb is not None:
        return ram_mb / 1024.0
    if not ram_str:
        return None
    m = re.search(r"(\d+)\s*GB", ram_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_kernel_major(kernel: str | None) -> float | None:
    """Extract major.minor kernel version."""
    if not kernel:
        return None
    m = re.search(r"(\d+\.\d+)", kernel)
    if m:
        return float(m.group(1))
    return None


_RE_GE_PROTON_NEW = re.compile(r"GE-Proton(\d+)", re.IGNORECASE)
_RE_GE_PROTON_OLD = re.compile(r"(?:Proton-)?(\d+)[\.\-].*GE", re.IGNORECASE)
_RE_PROTON_MAJOR = re.compile(r"(\d+)")


def _parse_proton_features(
    proton_version: str | None, custom_proton_version: str | None,
) -> dict[str, Any]:
    """Extract proton version features from report fields.

    Covers formats:
    - GE-Proton9-27 (new GE, custom_proton_version)
    - Proton-6.21-GE-2, 7.2-GE-2 (old GE, either field)
    - 6.3-8, 5.0-10 (official, proton_version)
    - Experimental / Proton Experimental (proton_version)
    """
    # 1. custom_proton_version — check first
    if custom_proton_version:
        cpv = custom_proton_version.strip().strip("'\"")
        # New format: GE-Proton9-27
        m = _RE_GE_PROTON_NEW.search(cpv)
        if m:
            return {"proton_major": int(m.group(1)), "is_ge_proton": 1, "has_proton_version": 1}
        # Old format: 7.2-GE-2, Proton-6.21-GE-2
        m = _RE_GE_PROTON_OLD.search(cpv)
        if m:
            return {"proton_major": int(m.group(1)), "is_ge_proton": 1, "has_proton_version": 1}
        # Bare major: extract first number as fallback
        m = _RE_PROTON_MAJOR.search(cpv)
        if m:
            return {"proton_major": int(m.group(1)), "is_ge_proton": 1, "has_proton_version": 1}

    # 2. proton_version
    if proton_version:
        pv = proton_version.strip()
        if pv.lower() in ("experimental", "proton experimental"):
            return {"proton_major": None, "is_ge_proton": 0, "has_proton_version": 1}
        # Old GE in proton_version field: Proton-6.21-GE-2
        if "GE" in pv.upper():
            m = _RE_GE_PROTON_OLD.search(pv)
            if m:
                return {"proton_major": int(m.group(1)), "is_ge_proton": 1, "has_proton_version": 1}
        # Official: 6.3-8, 5.0-10
        m = _RE_PROTON_MAJOR.match(pv)
        if m:
            return {"proton_major": int(m.group(1)), "is_ge_proton": 0, "has_proton_version": 1}

    return {"proton_major": None, "is_ge_proton": None, "has_proton_version": 0}


def _build_gpu_lookup(
    conn: sqlite3.Connection, *, source: str = "heuristic",
) -> dict[str, dict[str, Any]]:
    """Build a lookup dict from GPU normalized data table (raw_string → row).

    *source*: ``"heuristic"`` (default) or ``"llm"``.
    """
    table = _GPU_TABLES.get(source, _GPU_TABLES["heuristic"])
    # Check which form-factor columns exist (is_mobile may not exist in LLM table)
    col_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    available_cols = {row[1] for row in col_info}
    extra_cols = [c for c in ("is_apu", "is_igpu", "is_mobile") if c in available_cols]
    extra_sql = (", " + ", ".join(extra_cols)) if extra_cols else ""

    rows = conn.execute(
        f"SELECT raw_string, vendor, family, model, normalized_name{extra_sql} FROM {table}"
    ).fetchall()
    if not rows:
        return {}
    log.info("Loaded %d GPU normalized entries from %s", len(rows), table)
    return {
        row["raw_string"]: {
            "vendor": row["vendor"],
            "family": row["family"],
            "model": row["model"],
            **{c: row[c] for c in extra_cols},
        }
        for row in rows
    }


def _build_cpu_lookup(
    conn: sqlite3.Connection, *, source: str = "heuristic",
) -> dict[str, dict[str, Any]]:
    """Build a lookup dict from CPU normalized data table.

    *source*: ``"heuristic"`` (default) or ``"llm"``.
    """
    table = _CPU_TABLES.get(source, _CPU_TABLES["heuristic"])
    rows = conn.execute(
        f"SELECT raw_string, vendor, family, model, generation FROM {table}"
    ).fetchall()
    if not rows:
        return {}
    log.info("Loaded %d CPU normalized entries from %s", len(rows), table)
    return {
        row["raw_string"]: {
            "vendor": row["vendor"],
            "family": row["family"],
            "generation": row["generation"],
        }
        for row in rows
    }


def _build_driver_lookup(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Build a lookup dict from gpu_driver_normalization table."""
    rows = conn.execute(
        "SELECT raw_string, driver_vendor, driver_version, driver_major, driver_minor "
        "FROM gpu_driver_normalization"
    ).fetchall()
    if not rows:
        return {}
    log.info("Loaded %d GPU driver normalized entries", len(rows))
    return {
        row["raw_string"]: {
            "driver_vendor": row["driver_vendor"],
            "driver_version": row["driver_version"],
            "driver_major": row["driver_major"],
            "driver_minor": row["driver_minor"],
        }
        for row in rows
    }


def _parse_driver_major(gpu_driver: str | None) -> int | None:
    """Extract major driver version number (fallback when lookup misses)."""
    if not gpu_driver:
        return None
    m = re.search(r"(\d+)\.", gpu_driver)
    if m:
        return int(m.group(1))
    return None


def extract_hardware_features(
    report: dict[str, Any],
    gpu_lookup: dict[str, dict[str, Any]],
    cpu_lookup: dict[str, dict[str, Any]],
    driver_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract hardware features from a single report row.

    Uses normalized data tables if populated; falls back to heuristic extraction.
    """
    gpu_raw = report.get("gpu")
    cpu_raw = report.get("cpu")

    # GPU features
    gpu_data = gpu_lookup.get(gpu_raw) if gpu_raw else None
    if gpu_data:
        gpu_vendor = gpu_data["vendor"]
        gpu_family = gpu_data["family"]
        is_apu = gpu_data.get("is_apu", 0) or 0
        is_igpu = gpu_data.get("is_igpu", 0) or 0
        is_mobile = gpu_data.get("is_mobile", 0) or 0
    else:
        gpu_vendor = extract_gpu_vendor(gpu_raw)
        gpu_family = extract_gpu_family(gpu_raw)
        is_apu = 0
        is_igpu = 0
        is_mobile = 0

    # Steam Deck detection: vangogh chip or battery_performance field present
    gpu_lower = (gpu_raw or "").lower()
    is_steam_deck = 1 if ("vangogh" in gpu_lower or "van gogh" in gpu_lower) else 0
    if not is_steam_deck and report.get("battery_performance") is not None:
        is_steam_deck = 1

    gpu_tier = gpu_tier_from_family(gpu_family)

    # CPU features
    cpu_data = cpu_lookup.get(cpu_raw) if cpu_raw else None
    if cpu_data:
        cpu_vendor = cpu_data["vendor"]
        cpu_generation = cpu_data.get("generation")
    else:
        cpu_vendor = extract_cpu_vendor(cpu_raw)
        cpu_generation = extract_cpu_generation(cpu_raw)

    # Driver features — split by vendor to avoid mixing NVIDIA/Mesa scales
    driver_raw = report.get("gpu_driver")
    driver_data = driver_lookup.get(driver_raw) if driver_lookup and driver_raw else None

    nvidia_driver_version = None
    mesa_driver_version = None
    driver_major = None

    if driver_data:
        d_vendor = driver_data.get("driver_vendor")
        d_major = driver_data.get("driver_major")
        d_minor = driver_data.get("driver_minor")
        if d_vendor == "nvidia" and d_major is not None:
            nvidia_driver_version = d_major + (d_minor or 0) / 1000.0
        elif d_vendor == "mesa" and d_major is not None:
            mesa_driver_version = d_major + (d_minor or 0) / 10.0
    else:
        # Fallback for reports not in driver lookup
        driver_major = _parse_driver_major(driver_raw)

    # After Phase 7 ablation, removed features with ΔF1 < 0.002:
    # gpu_vendor, gpu_tier, cpu_vendor, cpu_generation, ram_gb,
    # kernel_major, os_family, proton_major, is_ge_proton, has_proton_version,
    # driver_major.  See PLAN_ML_7.md for details.
    return {
        "gpu_family": gpu_family,
        "nvidia_driver_version": nvidia_driver_version,
        "mesa_driver_version": mesa_driver_version,
        "is_apu": is_apu,
        "is_igpu": is_igpu,
        "is_mobile": is_mobile,
        "is_steam_deck": is_steam_deck,
    }


def build_hardware_lookups(
    conn: sqlite3.Connection,
    *,
    source: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build GPU, CPU, and driver normalized data lookups.

    *source*: ``"heuristic"`` or ``"llm"``.  If *None*, reads from
    ``NORMALIZED_DATA_SOURCE`` env var (default ``"heuristic"``).

    Returns ``(gpu_lookup, cpu_lookup, driver_lookup)``.
    """
    if source is None:
        from protondb_settings.config import NORMALIZED_DATA_SOURCE
        source = NORMALIZED_DATA_SOURCE

    log.info("Using normalized data source: %s", source)
    return (
        _build_gpu_lookup(conn, source=source),
        _build_cpu_lookup(conn, source=source),
        _build_driver_lookup(conn),
    )
