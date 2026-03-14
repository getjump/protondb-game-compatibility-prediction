"""CPU normalization via regex heuristics — fast alternative to LLM.

CPU strings from ProtonDB are very structured (2.4K unique values).
Two main formats:
  - AMD: "AMD Ryzen 7 5800X 8-Core"
  - Intel: "Intel Core i7-6700K @ 4.00GHz"
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

# ── AMD Ryzen generation from model number ──────────────────────────────────

def _amd_generation(model_num: int) -> int:
    """Determine AMD Ryzen generation from the first digit of the model number."""
    first = model_num // 1000
    # Ryzen 9000 = gen 5 (Zen 5), 7000 = gen 4, etc. but numbering is direct
    return first


def _amd_family(series: str) -> str:
    """Normalize AMD series string to family."""
    s = series.strip().lower()
    if "ryzen" in s:
        return "ryzen"
    if "threadripper" in s:
        return "threadripper"
    if "epyc" in s:
        return "epyc"
    if "athlon" in s:
        return "athlon"
    if "fx" in s:
        return "fx"
    if "a10" in s or "a8" in s or "a6" in s or "a4" in s:
        return "apu"
    if "phenom" in s:
        return "phenom"
    return "amd_other"


# ── Intel generation from model number ──────────────────────────────────────

def _intel_generation(model_str: str) -> int:
    """Determine Intel Core generation from model number string.

    "6700K" → 6, "12700K" → 12, "1200" → 1
    """
    digits = re.match(r"(\d+)", model_str)
    if not digits:
        return 0
    num = int(digits.group(1))
    if num >= 10000:
        return num // 1000  # 12700 → 12, 14900 → 14
    return num // 100  # 6700 → 6 (wait, that's 67 — wrong)


# Actually, Intel naming:
# i7-6700K → gen 6 (first digit before last 3)
# i7-12700K → gen 12 (first 2 digits before last 3)
# i7-920 → gen 1 (Nehalem, 3 digits)
def _intel_gen_from_model(model_str: str) -> int:
    """Extract Intel Core generation from model number.

    Examples: 920 → 1, 6700K → 6, 12700K → 12, 1240P → 12, 14900K → 14
    """
    digits = re.match(r"(\d+)", model_str)
    if not digits:
        return 0
    num = int(digits.group(1))
    if num < 1000:
        return 1  # 3-digit models are gen 1 (920, 860, etc.)
    # 10th gen+ use 5-digit model numbers (10100, 12700, 14900)
    # but some like 1240P are also 4 digits with gen 12
    # Heuristic: if num >= 10000, gen = num // 1000
    # if 1000-9999: first digit is gen (2xxx=2nd, 6xxx=6th, etc.)
    # Exception: 10xx, 11xx, 12xx, 13xx, 14xx → could be gen 10-14
    # Intel skipped 5-digit for mobile: i5-1240P is gen 12 (1240 → 12)
    if num >= 10000:
        return num // 1000  # 12700 → 12, 14900 → 14
    # 4-digit: first digit is generation for gens 2-9
    # but gens 10-14 also use 4-digit mobile models (1035G1=10, 1165G7=11, 1240P=12, 1370P=13)
    first_digit = num // 1000
    if first_digit >= 2:
        return first_digit  # 6700 → 6, 9900 → 9
    # first_digit is 1: could be gen 1 (1xxx old) or gen 10-14 (mobile)
    # gen 10+ mobile: 10xx, 11xx, 12xx, 13xx, 14xx
    second_digit = (num // 100) % 10
    if second_digit >= 0:
        gen = num // 100  # 1240 → 12, 1035 → 10
        if gen >= 10:
            return gen
    return 1  # actual gen 1 (like 1xxx old Nehalem)


# ── Regex patterns ──────────────────────────────────────────────────────────

# AMD Ryzen 7 5800X3D 8-Core
_RE_AMD_RYZEN = re.compile(
    r"AMD\s+(Ryzen\s+\d)\s+(\d{4}\w*)",
    re.IGNORECASE,
)

# AMD Ryzen Threadripper 3990X 64-Core
_RE_AMD_THREADRIPPER = re.compile(
    r"AMD\s+Ryzen\s+(Threadripper)\s+(\d{4}\w*)",
    re.IGNORECASE,
)

# AMD EPYC 7742 64-Core
_RE_AMD_EPYC = re.compile(
    r"AMD\s+(EPYC)\s+(\d{4}\w*)",
    re.IGNORECASE,
)

# AMD FX-8350 / AMD FX(tm)-8350
_RE_AMD_FX = re.compile(
    r"AMD\s+FX(?:\(tm\))?[- ]+(\d{4}\w*)",
    re.IGNORECASE,
)

# AMD Athlon / A10 / A8 / Phenom
_RE_AMD_OTHER = re.compile(
    r"AMD\s+(Athlon|A\d+|Phenom)\s*(?:II\s*)?(?:X\d\s*)?(\w*\d+\w*)?",
    re.IGNORECASE,
)

# AMD Custom APU 0405 (Steam Deck)
_RE_AMD_CUSTOM_APU = re.compile(
    r"AMD\s+Custom\s+APU\s+(\w+)",
    re.IGNORECASE,
)

# Intel Core i7-6700K @ 4.00GHz
_RE_INTEL_CORE = re.compile(
    r"Intel\s+Core\s+(i\d+)\s*-?\s*(\d{3,5}\w*)",
    re.IGNORECASE,
)

# Intel Core Ultra 7 155H
_RE_INTEL_ULTRA = re.compile(
    r"Intel\s+Core\s+Ultra\s+(\d)\s+(\d{3}\w*)",
    re.IGNORECASE,
)

# Intel Xeon E5-2670 / W-2145
_RE_INTEL_XEON = re.compile(
    r"Intel\s+Xeon\s*(?:\(R\)\s*)?(?:CPU\s+)?(\w[\w-]*\d+\w*)",
    re.IGNORECASE,
)

# Intel Pentium / Celeron / N-series
_RE_INTEL_OTHER = re.compile(
    r"Intel\s+(Pentium|Celeron|Atom)\s*(?:\(R\)\s*)?(?:CPU\s+)?(?:Gold\s+|Silver\s+)?(\w*\d+\w*)?",
    re.IGNORECASE,
)

# Intel N100, N95 etc (no "Core" prefix)
_RE_INTEL_N = re.compile(
    r"Intel\s*(?:\(R\)\s*)?N(\d{2,3})",
    re.IGNORECASE,
)


def parse_cpu(raw: str) -> dict:
    """Parse a raw CPU string into a normalized dict."""
    result = {
        "raw_string": raw,
        "vendor": "unknown",
        "family": "unknown",
        "model": "unknown",
        "normalized_name": "unknown",
        "generation": None,
        "is_apu": 0,
    }

    s = raw.strip()
    if not s:
        return result

    # ── AMD Custom APU (Steam Deck) ──
    m = _RE_AMD_CUSTOM_APU.match(s)
    if m:
        result["vendor"] = "amd"
        result["family"] = "custom_apu"
        result["model"] = "steam_deck_apu"
        result["normalized_name"] = "AMD Custom APU (Steam Deck)"
        result["is_apu"] = 1
        return result

    # ── AMD Threadripper ──
    m = _RE_AMD_THREADRIPPER.search(s)
    if m:
        model = m.group(2)
        gen = _amd_generation(int(re.match(r"\d+", model).group()))
        result["vendor"] = "amd"
        result["family"] = "threadripper"
        result["model"] = f"threadripper_{model.lower()}"
        result["normalized_name"] = f"AMD Ryzen Threadripper {model}"
        result["generation"] = gen
        return result

    # ── AMD Ryzen ──
    m = _RE_AMD_RYZEN.search(s)
    if m:
        series = m.group(1)  # "Ryzen 7"
        model = m.group(2)   # "5800X3D"
        tier = series.split()[-1]  # "7"
        model_num_match = re.match(r"(\d+)", model)
        model_num = int(model_num_match.group()) if model_num_match else 0
        gen = _amd_generation(model_num)
        is_apu = 1 if model.upper().endswith("G") or model.upper().endswith("GE") or model.upper().endswith("U") or "APU" in s.upper() else 0

        result["vendor"] = "amd"
        result["family"] = "ryzen"
        result["model"] = f"ryzen_{tier}_{model.lower()}"
        result["normalized_name"] = f"AMD {series} {model}"
        result["generation"] = gen
        result["is_apu"] = is_apu
        return result

    # ── AMD EPYC ──
    m = _RE_AMD_EPYC.search(s)
    if m:
        model = m.group(2)
        result["vendor"] = "amd"
        result["family"] = "epyc"
        result["model"] = f"epyc_{model.lower()}"
        result["normalized_name"] = f"AMD EPYC {model}"
        return result

    # ── AMD FX ──
    m = _RE_AMD_FX.search(s)
    if m:
        model = m.group(1)
        result["vendor"] = "amd"
        result["family"] = "fx"
        result["model"] = f"fx_{model.lower()}"
        result["normalized_name"] = f"AMD FX-{model}"
        return result

    # ── AMD other (Athlon, A-series, Phenom) ──
    m = _RE_AMD_OTHER.search(s)
    if m:
        series = m.group(1)
        model = m.group(2) or ""
        family = _amd_family(series)
        result["vendor"] = "amd"
        result["family"] = family
        result["model"] = f"{series.lower()}_{model.lower()}".rstrip("_")
        result["normalized_name"] = f"AMD {series} {model}".strip()
        result["is_apu"] = 1 if family == "apu" else 0
        return result

    # ── Intel Core Ultra ──
    m = _RE_INTEL_ULTRA.search(s)
    if m:
        tier = m.group(1)  # "7"
        model = m.group(2)  # "155H"
        result["vendor"] = "intel"
        result["family"] = "core_ultra"
        result["model"] = f"ultra_{tier}_{model.lower()}"
        result["normalized_name"] = f"Intel Core Ultra {tier} {model}"
        result["generation"] = 1  # first gen Ultra
        return result

    # ── Intel Core ──
    m = _RE_INTEL_CORE.search(s)
    if m:
        tier = m.group(1).lower()  # "i7"
        model = m.group(2)          # "6700K"
        gen = _intel_gen_from_model(model)
        result["vendor"] = "intel"
        result["family"] = "core"
        result["model"] = f"{tier}_{model.lower()}"
        result["normalized_name"] = f"Intel Core {tier}-{model}"
        result["generation"] = gen
        return result

    # ── Intel Xeon ──
    m = _RE_INTEL_XEON.search(s)
    if m:
        model = m.group(1)
        result["vendor"] = "intel"
        result["family"] = "xeon"
        result["model"] = f"xeon_{model.lower()}"
        result["normalized_name"] = f"Intel Xeon {model}"
        return result

    # ── Intel N-series (N100 etc) ──
    m = _RE_INTEL_N.search(s)
    if m:
        num = m.group(1)
        result["vendor"] = "intel"
        result["family"] = "n_series"
        result["model"] = f"n{num}"
        result["normalized_name"] = f"Intel N{num}"
        return result

    # ── Intel other (Pentium, Celeron, Atom) ──
    m = _RE_INTEL_OTHER.search(s)
    if m:
        series = m.group(1)
        model = m.group(2) or ""
        result["vendor"] = "intel"
        result["family"] = series.lower()
        result["model"] = f"{series.lower()}_{model.lower()}".rstrip("_")
        result["normalized_name"] = f"Intel {series} {model}".strip()
        return result

    # ── Fallback: detect vendor from keywords ──
    s_lower = s.lower()
    if "amd" in s_lower:
        result["vendor"] = "amd"
        result["normalized_name"] = s
    elif "intel" in s_lower:
        result["vendor"] = "intel"
        result["normalized_name"] = s

    return result


def get_pending_cpu_strings(conn: sqlite3.Connection) -> list[str]:
    """Get distinct CPU strings not yet in cpu_normalization_heuristic."""
    return get_pending_strings(
        conn, source_table="reports", source_column="cpu",
        norm_table="cpu_normalization_heuristic",
    )


def get_pending_count(conn: sqlite3.Connection) -> int:
    return len(get_pending_cpu_strings(conn))


def normalize_cpus_heuristic(
    conn: sqlite3.Connection,
    *,
    force: bool = False,
) -> int:
    """Run CPU normalization via regex heuristics.

    Returns the number of strings processed.
    """
    if force:
        log.info("Force CPU heuristic normalization: clearing table")
        conn.execute("DELETE FROM cpu_normalization_heuristic")
        conn.commit()

    pending = get_pending_cpu_strings(conn)
    if not pending:
        log.info("CPU heuristic normalization: nothing to do")
        return 0

    log.info("CPU heuristic normalization: %d strings to process", len(pending))

    processed = 0
    with PipelineStep(conn, "normalize_cpu_heuristic", len(pending)) as step:
        for batch in chunked(pending, _BATCH_SIZE):
            if shutdown_requested.is_set():
                log.info("CPU heuristic normalization: interrupted at %d", processed)
                break

            all_rows = [parse_cpu(s) for s in batch]

            if all_rows:
                upsert_rows(conn, "cpu_normalization_heuristic", all_rows, "raw_string")
                conn.commit()

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
