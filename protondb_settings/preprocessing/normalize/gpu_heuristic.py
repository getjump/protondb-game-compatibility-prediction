"""GPU normalization via regex heuristics — fast alternative to LLM.

Covers ~99.7% of reports by pattern-matching known GPU string formats.
Processes 35K unique strings in seconds instead of hours.
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

# ── AMD chip codename → (family, model_prefix, generation) ──────────────────

_AMD_CHIPS: dict[str, tuple[str, str, str | None]] = {
    # RDNA 4
    "gfx1201": ("rdna4", "rx9", None),
    # RDNA 3.5
    "gfx1150": ("rdna3.5", "radeon_8xx", None),
    # RDNA 3
    "navi31": ("rdna3", "rx7900", None),
    "navi32": ("rdna3", "rx7800", None),
    "navi33": ("rdna3", "rx7600", None),
    # RDNA 2
    "navi21": ("rdna2", "rx6800", None),
    "navi22": ("rdna2", "rx6700", None),
    "navi23": ("rdna2", "rx6600", None),
    "navi24": ("rdna2", "rx6400", None),
    "navi10": ("rdna1", "rx5700", None),
    "navi12": ("rdna1", "rx5600m", None),
    "navi14": ("rdna1", "rx5500", None),
    # GCN 5 (Vega)
    "vega20": ("vega", "vega_vii", None),
    "vega10": ("vega", "vega56", None),
    "vega12": ("vega", "vega_mobile", None),
    # APUs (Vega iGPU)
    "raven": ("vega", "vega_apu", None),
    "raven2": ("vega", "vega_apu", None),
    "picasso": ("vega", "vega_apu", None),
    "renoir": ("vega", "vega_apu", None),
    "cezanne": ("vega", "vega_apu", None),
    "barcelo": ("vega", "vega_apu", None),
    "lucienne": ("vega", "vega_apu", None),
    # RDNA 2 APUs
    "vangogh": ("rdna2", "steam_deck_apu", None),
    "rembrandt": ("rdna2", "rdna2_apu", None),
    "mendocino": ("rdna2", "rdna2_apu", None),
    "raphael_mendocino": ("rdna2", "rdna2_apu", None),
    "phoenix": ("rdna3", "rdna3_apu", None),
    "phoenix2": ("rdna3", "rdna3_apu", None),
    # RDNA 3.5 APUs
    "strix_point": ("rdna3.5", "rdna3.5_apu", None),
    # GCN 4
    "polaris10": ("gcn4", "rx480", None),
    "polaris11": ("gcn4", "rx460", None),
    "polaris12": ("gcn4", "rx550", None),
    "ellesmere": ("gcn4", "rx480", None),
    "baffin": ("gcn4", "rx460", None),
    "lexa": ("gcn4", "rx550", None),
    # GCN 3
    "fiji": ("gcn3", "fury", None),
    "tonga": ("gcn3", "r9_380", None),
    "topaz": ("gcn3", "r7_360", None),
    # GCN 2
    "bonaire": ("gcn2", "r7_260", None),
    "hawaii": ("gcn2", "r9_290", None),
    "pitcairn": ("gcn2", "r9_270", None),
    # GCN 1
    "tahiti": ("gcn1", "hd7900", None),
    "verde": ("gcn1", "hd7700", None),
    "oland": ("gcn1", "r7_240", None),
    "cape_verde": ("gcn1", "hd7700", None),
    "hainan": ("gcn1", "hd8400", None),
    # CDNA (server / BC-250)
    "gfx1013": ("cdna", "bc250", None),
    "gfx908": ("cdna", "mi100", None),
    "gfx90a": ("cdna2", "mi200", None),
}

_APU_CHIPS = {
    "raven", "raven2", "picasso", "renoir", "cezanne", "barcelo", "lucienne",
    "vangogh", "rembrandt", "mendocino", "raphael_mendocino",
    "phoenix", "phoenix2", "strix_point",
}

# ── NVIDIA model number → family ────────────────────────────────────────────

def _nvidia_family(model_num: int, prefix: str) -> str:
    """Determine NVIDIA family from model number and prefix (GTX/RTX/GT)."""
    if prefix.upper() == "RTX":
        if model_num >= 5000:
            return "rtx50"
        if model_num >= 4000:
            return "rtx40"
        if model_num >= 3000:
            return "rtx30"
        if model_num >= 2000:
            return "rtx20"
        return "rtx"
    if prefix.upper() in ("GTX", "GT"):
        if model_num >= 1600:
            return "gtx16"
        if model_num >= 1000:
            return "gtx10"
        if model_num >= 900:
            return "gtx9"
        if model_num >= 700:
            return "gtx7"
        if model_num >= 600:
            return "gtx6"
        if model_num >= 400:
            return "gtx4"
        return "gt"
    return "unknown"


# ── Regex patterns ──────────────────────────────────────────────────────────

# NVIDIA GeForce GTX/RTX 3070 Ti / SUPER
_RE_NVIDIA_GEFORCE = re.compile(
    r"(?:NVIDIA\s+)?GeForce\s+(GTX|RTX|GT)\s*(\d{3,4})\s*(Ti|SUPER|Ti\s+SUPER)?",
    re.IGNORECASE,
)

# NVIDIA GeForce without GTX/RTX prefix: "NVIDIA GeForce 210", "GeForce 920MX"
_RE_NVIDIA_GEFORCE_BARE = re.compile(
    r"(?:NVIDIA\s+)?GeForce\s+(\d{3,4})\s*(M|MX)?",
    re.IGNORECASE,
)

# Bare NVIDIA model: "3060Ti", "4080 Founders Edition", "1660ti mobile"
_RE_NVIDIA_BARE = re.compile(
    r"^(\d{4})\s*(Ti|SUPER)?\b",
    re.IGNORECASE,
)

# AMD Radeon RX 5700 XT or just "Radeon RX 580"
_RE_AMD_RADEON_RX = re.compile(
    r"(?:AMD\s+)?Radeon\s+RX\s+(\d{3,4})\s*(XT|XTX)?\b",
    re.IGNORECASE,
)

# AMD Radeon R9/R7/R5 series
_RE_AMD_RADEON_R = re.compile(
    r"(?:AMD\s+)?Radeon\s+(R\d)\s+(\d{3})\s*(X)?",
    re.IGNORECASE,
)

# AMD bare model: "AMD 6800 XT", "AMD 5700XT"
_RE_AMD_BARE_MODEL = re.compile(
    r"AMD\s+(\d{4})\s*(XT|XTX)?\b",
    re.IGNORECASE,
)

# AMD radeonsi driver string with chip codename
_RE_AMD_RADEONSI = re.compile(
    r"radeonsi,\s*(\w+)",
    re.IGNORECASE,
)

# AMD Custom GPU (Steam Deck and others)
_RE_AMD_CUSTOM = re.compile(
    r"AMD\s+Custom\s+GPU\s+\w+",
    re.IGNORECASE,
)

# Intel Mesa Intel HD/UHD/Iris (Plus/Pro/Xe)
_RE_INTEL = re.compile(
    r"(?:Mesa\s+)?Intel\s*(?:\(R\)\s*)?(?:HD|UHD|Iris)\s*(Plus|Pro|Xe)?\s*(?:Graphics\s*)?(\d{3,4})?",
    re.IGNORECASE,
)

# Nouveau open-source NVIDIA driver
_RE_NOUVEAU = re.compile(
    r"nouveau\s*(?:.*NV(\w+))?",
    re.IGNORECASE,
)

# Virtual GPUs
_RE_VIRTUAL = re.compile(
    r"(llvmpipe|virgl|vmware|swrast|softpipe|lavapipe|zink)",
    re.IGNORECASE,
)

# PCI device ID format: "GA107M [GeForce RTX 3050 Mobile]"
_RE_PCI_BRACKET = re.compile(
    r"\[([^\]]+)\]",
)

# NVIDIA bare with prefix: "NVIDIA 4080", "NVIDIA 3050 Laptop", "Nvidia RTX 3060"
_RE_NVIDIA_BARE_PREFIX = re.compile(
    r"NVIDIA\s+(?:GeForce\s+)?(GTX|RTX)?\s*(\d{3,4})\s*(Ti|SUPER)?",
    re.IGNORECASE,
)

# GeForce MX series: "GeForce MX110", "GeForce MX450"
_RE_NVIDIA_MX = re.compile(
    r"(?:NVIDIA\s+)?GeForce\s+MX\s*(\d{3})",
    re.IGNORECASE,
)

# NVIDIA driver version only: "4.6.0 NVIDIA 535.183.01" or "535.183.01"
_RE_NVIDIA_DRIVER_ONLY = re.compile(
    r"^(?:[\d.]+\s+)?NVIDIA\s+(\d{3}\.\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


def _amd_rx_family(model_num: int) -> str:
    """Determine AMD Radeon RX family from model number."""
    if model_num >= 9000:
        return "rdna4"
    if model_num >= 7000:
        return "rdna3"
    if model_num >= 6000:
        return "rdna2"
    if model_num >= 5000:
        return "rdna1"
    if model_num >= 400:
        return "gcn4"
    if model_num >= 200:
        return "gcn3"
    return "gcn"


def parse_gpu(raw: str) -> dict:
    """Parse a raw GPU string into a normalized dict.

    Returns dict with keys: vendor, family, model, normalized_name, is_apu, is_igpu, is_mobile, is_virtual.
    """
    result = {
        "raw_string": raw,
        "vendor": "unknown",
        "family": "unknown",
        "model": "unknown",
        "normalized_name": "unknown",
        "is_apu": 0,
        "is_igpu": 0,
        "is_mobile": 0,
        "is_virtual": 0,
    }

    s = raw.strip()
    if not s:
        return result

    # ── Mobile / Laptop detection (applies to all vendors) ──
    s_lower = s.lower()
    if any(kw in s_lower for kw in ("mobile", "laptop", "max-q", "max-p", " m ", "notebook")):
        result["is_mobile"] = 1
    elif re.search(r"\bGA\d+M\b|\bGN\d+M\b|\bGP\d+M\b|\bTU\d+M\b|\bAD\d+M\b", s):
        # NVIDIA mobile chip codes: GA107M, TU116M, etc.
        result["is_mobile"] = 1

    # ── Virtual GPU ──
    m = _RE_VIRTUAL.search(s)
    if m:
        virt = m.group(1).lower()
        result["vendor"] = "virtual"
        result["family"] = "virtual"
        result["model"] = virt
        result["normalized_name"] = virt
        result["is_virtual"] = 1
        return result

    # ── PCI bracket format: extract inner name and re-parse ──
    m = _RE_PCI_BRACKET.search(s)
    if m:
        inner = m.group(1)
        inner_result = parse_gpu(inner)
        if inner_result["vendor"] != "unknown":
            inner_result["raw_string"] = raw
            return inner_result

    # ── NVIDIA driver-only string (no GPU info) ──
    m = _RE_NVIDIA_DRIVER_ONLY.match(s)
    if m:
        result["vendor"] = "nvidia"
        result["family"] = "unknown"
        result["model"] = "driver_only"
        result["normalized_name"] = f"NVIDIA (driver {m.group(1)})"
        return result

    # ── NVIDIA GeForce GTX/RTX/GT ──
    m = _RE_NVIDIA_GEFORCE.search(s)
    if m:
        prefix = m.group(1).upper()  # GTX, RTX, GT
        num = int(m.group(2))
        suffix = (m.group(3) or "").strip()
        family = _nvidia_family(num, prefix)
        model_str = f"{prefix.lower()}{num}"
        if suffix:
            model_str += suffix.lower().replace(" ", "_")
        name = f"NVIDIA GeForce {prefix} {num}"
        if suffix:
            name += f" {suffix}"
        result["vendor"] = "nvidia"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = name
        return result

    # ── NVIDIA GeForce bare number: "GeForce 210", "GeForce 920MX" ──
    m = _RE_NVIDIA_GEFORCE_BARE.search(s)
    if m:
        num = int(m.group(1))
        suffix = (m.group(2) or "").strip()
        family = _nvidia_family(num, "GT")
        model_str = f"gt{num}"
        if suffix:
            model_str += suffix.lower()
        name = f"NVIDIA GeForce {num}"
        if suffix:
            name += suffix
        result["vendor"] = "nvidia"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = name
        return result

    # ── AMD Radeon RX ──
    m = _RE_AMD_RADEON_RX.search(s)
    if m:
        num = int(m.group(1))
        suffix = (m.group(2) or "").strip().upper()
        family = _amd_rx_family(num)
        model_str = f"rx{num}"
        if suffix:
            model_str += suffix.lower()
        name = f"AMD Radeon RX {num}"
        if suffix:
            name += f" {suffix}"
        result["vendor"] = "amd"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = name
        return result

    # ── AMD Radeon R9/R7/R5 ──
    m = _RE_AMD_RADEON_R.search(s)
    if m:
        tier = m.group(1).upper()
        num = int(m.group(2))
        suffix = (m.group(3) or "").strip().upper()
        family = "gcn3" if num >= 300 else "gcn2"
        model_str = f"{tier.lower()}_{num}"
        if suffix:
            model_str += suffix.lower()
        name = f"AMD Radeon {tier} {num}"
        if suffix:
            name += suffix
        result["vendor"] = "amd"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = name
        return result

    # ── AMD radeonsi driver string (with chip codename) ──
    m = _RE_AMD_RADEONSI.search(s)
    if m:
        chip = m.group(1).lower()
        chip_info = _AMD_CHIPS.get(chip)
        is_apu = 1 if chip in _APU_CHIPS else 0

        if chip_info:
            family, model_prefix, _ = chip_info
        else:
            family = "unknown"
            model_prefix = chip

        # Try to also extract RX model from the prefix
        rx_match = _RE_AMD_RADEON_RX.search(s)
        if rx_match:
            num = int(rx_match.group(1))
            suffix = (rx_match.group(2) or "").strip().upper()
            model_str = f"rx{num}"
            if suffix:
                model_str += suffix.lower()
            name = f"AMD Radeon RX {num}"
            if suffix:
                name += f" {suffix}"
        elif "Custom GPU" in s or chip == "vangogh":
            model_str = "steam_deck_apu" if chip == "vangogh" else model_prefix
            name = "AMD Custom APU (Steam Deck)" if chip == "vangogh" else f"AMD Custom GPU ({chip})"
        else:
            model_str = model_prefix
            name = f"AMD Radeon ({chip})"

        result["vendor"] = "amd"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = name
        result["is_apu"] = is_apu
        result["is_igpu"] = is_apu
        return result

    # ── AMD Custom GPU (without radeonsi — older format) ──
    m = _RE_AMD_CUSTOM.search(s)
    if m:
        is_vangogh = "vangogh" in s.lower()
        result["vendor"] = "amd"
        result["family"] = "rdna2" if is_vangogh else "custom"
        result["model"] = "steam_deck_apu" if is_vangogh else "custom_gpu"
        result["normalized_name"] = "AMD Custom APU (Steam Deck)" if is_vangogh else "AMD Custom GPU"
        result["is_apu"] = 1 if is_vangogh else 0
        result["is_igpu"] = 1 if is_vangogh else 0
        return result

    # ── AMD bare model: "AMD 6800 XT" ──
    m = _RE_AMD_BARE_MODEL.search(s)
    if m:
        num = int(m.group(1))
        suffix = (m.group(2) or "").strip().upper()
        family = _amd_rx_family(num)
        model_str = f"rx{num}"
        if suffix:
            model_str += suffix.lower()
        name = f"AMD Radeon RX {num}"
        if suffix:
            name += f" {suffix}"
        result["vendor"] = "amd"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = name
        return result

    # ── Intel iGPU ──
    m = _RE_INTEL.search(s)
    if m:
        variant = (m.group(1) or "").strip()
        num = m.group(2)

        if "Iris" in s:
            if "Xe" in s or variant.lower() == "xe":
                family = "xe"
            else:
                family = "iris"
        elif num:
            n = int(num)
            if n >= 700:
                family = "uhd700"
            elif n >= 600:
                family = "uhd600"
            else:
                family = "hd"
        else:
            family = "intel_igpu"

        model_parts = []
        if "Iris" in s:
            model_parts.append("iris")
        elif "UHD" in s.upper():
            model_parts.append("uhd")
        else:
            model_parts.append("hd")
        if variant:
            model_parts.append(variant.lower())
        if num:
            model_parts.append(num)
        model_str = "_".join(model_parts)

        name_parts = ["Intel"]
        if "Iris" in s:
            name_parts.append("Iris")
        elif "UHD" in s.upper():
            name_parts.append("UHD Graphics")
        else:
            name_parts.append("HD Graphics")
        if variant:
            name_parts.append(variant)
        if num:
            name_parts.append(num)

        result["vendor"] = "intel"
        result["family"] = family
        result["model"] = model_str
        result["normalized_name"] = " ".join(name_parts)
        result["is_igpu"] = 1
        return result

    # ── NVIDIA MX series ──
    m = _RE_NVIDIA_MX.search(s)
    if m:
        num = int(m.group(1))
        result["vendor"] = "nvidia"
        result["family"] = "mx"
        result["model"] = f"mx{num}"
        result["normalized_name"] = f"NVIDIA GeForce MX{num}"
        return result

    # ── NVIDIA bare with prefix: "NVIDIA 4080", "NVIDIA RTX 3060" ──
    m = _RE_NVIDIA_BARE_PREFIX.search(s)
    if m and "driver" not in s.lower() and "smi" not in s.lower():
        prefix = (m.group(1) or "").upper()
        num = int(m.group(2))
        suffix = (m.group(3) or "").strip()
        if num >= 1000:
            if not prefix:
                prefix = "RTX" if num >= 2000 else "GTX"
            family = _nvidia_family(num, prefix)
            model_str = f"{prefix.lower()}{num}"
            if suffix:
                model_str += suffix.lower()
            name = f"NVIDIA GeForce {prefix} {num}"
            if suffix:
                name += f" {suffix}"
            result["vendor"] = "nvidia"
            result["family"] = family
            result["model"] = model_str
            result["normalized_name"] = name
            return result

    # ── Nouveau ──
    m = _RE_NOUVEAU.search(s)
    if m:
        chip = m.group(1) or "unknown"
        result["vendor"] = "nvidia"
        result["family"] = "nouveau"
        result["model"] = f"nv{chip.lower()}"
        result["normalized_name"] = f"NVIDIA (nouveau NV{chip.upper()})"
        return result

    # ── Bare NVIDIA model number at start: "3060Ti", "4080 Founders Edition" ──
    m = _RE_NVIDIA_BARE.match(s)
    if m:
        num = int(m.group(1))
        suffix = (m.group(2) or "").strip()
        if num >= 1000:
            prefix = "RTX" if num >= 2000 else "GTX"
            family = _nvidia_family(num, prefix)
            model_str = f"{prefix.lower()}{num}"
            if suffix:
                model_str += suffix.lower()
            name = f"NVIDIA GeForce {prefix} {num}"
            if suffix:
                name += f" {suffix}"
            result["vendor"] = "nvidia"
            result["family"] = family
            result["model"] = model_str
            result["normalized_name"] = name
            return result

    # ── Fallback: try to detect vendor from keywords ──
    s_lower = s.lower()
    if "nvidia" in s_lower or "geforce" in s_lower:
        result["vendor"] = "nvidia"
        result["normalized_name"] = s
    elif "amd" in s_lower or "radeon" in s_lower or "ati" in s_lower:
        result["vendor"] = "amd"
        result["normalized_name"] = s
    elif "intel" in s_lower:
        result["vendor"] = "intel"
        result["normalized_name"] = s
        result["is_igpu"] = 1

    return result


def get_pending_gpu_strings(conn: sqlite3.Connection) -> list[str]:
    """Get distinct GPU strings not yet in gpu_normalization_heuristic."""
    return get_pending_strings(
        conn, source_table="reports", source_column="gpu",
        norm_table="gpu_normalization_heuristic",
    )


def get_pending_count(conn: sqlite3.Connection) -> int:
    return len(get_pending_gpu_strings(conn))


def normalize_gpus_heuristic(
    conn: sqlite3.Connection,
    *,
    force: bool = False,
) -> int:
    """Run GPU normalization via regex heuristics.

    Returns the number of strings processed.
    """
    if force:
        log.info("Force GPU heuristic normalization: clearing table")
        conn.execute("DELETE FROM gpu_normalization_heuristic")
        conn.commit()

    pending = get_pending_gpu_strings(conn)
    if not pending:
        log.info("GPU heuristic normalization: nothing to do")
        return 0

    log.info("GPU heuristic normalization: %d strings to process", len(pending))

    processed = 0
    with PipelineStep(conn, "normalize_gpu_heuristic", len(pending)) as step:
        for batch in chunked(pending, _BATCH_SIZE):
            if shutdown_requested.is_set():
                log.info("GPU heuristic normalization: interrupted at %d", processed)
                break

            rows = [parse_gpu(s) for s in batch]
            rows = [r for r in rows if r["vendor"] != "unknown" or r["normalized_name"] != "unknown"]

            # Always insert even unknown — so we don't retry them
            all_rows = []
            for s in batch:
                parsed = parse_gpu(s)
                all_rows.append(parsed)

            if all_rows:
                upsert_rows(conn, "gpu_normalization_heuristic", all_rows, "raw_string")
                conn.commit()

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
