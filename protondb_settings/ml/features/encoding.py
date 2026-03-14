"""Categorical encoding, label maps, tier/family mappings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# GPU tier mapping: family -> tier
# ---------------------------------------------------------------------------

_GPU_TIER_MAP: dict[str, str] = {
    # Low
    "gtx10": "low",
    "gtx16": "low",
    "rx500": "low",
    "hd600": "low",
    "vega": "low",
    "gt10": "low",
    "gt7": "low",
    "hd5": "low",
    "hd7": "low",
    "hd4": "low",
    "r7": "low",
    "r9": "low",
    "gma": "low",
    # Mid
    "rtx20": "mid",
    "rx5000": "mid",
    "rdna2": "mid",
    "xe": "mid",
    "rx6000": "mid",
    "gtx9": "mid",
    # High
    "rtx30": "high",
    "rdna3": "high",
    "arc": "high",
    "rx7000": "high",
    # Flagship
    "rtx40": "flagship",
    "rdna4": "flagship",
    "rtx50": "flagship",
    "rx9000": "flagship",
}


def gpu_tier_from_family(family: str | None) -> str | None:
    """Map a GPU family string to a tier (low/mid/high/flagship)."""
    if not family:
        return None
    family_lower = family.lower().strip()
    for prefix, tier in _GPU_TIER_MAP.items():
        if family_lower.startswith(prefix) or family_lower == prefix:
            return tier
    return None


# ---------------------------------------------------------------------------
# OS family mapping from raw os string
# ---------------------------------------------------------------------------

_OS_PATTERNS: list[tuple[str, str]] = [
    (r"arch", "arch"),
    (r"manjaro", "arch"),
    (r"endeavour", "arch"),
    (r"cachyos", "arch"),
    (r"garuda", "arch"),
    (r"arcolinux", "arch"),
    (r"ubuntu", "ubuntu"),
    (r"pop!?_?os|pop os", "ubuntu"),
    (r"linux mint", "ubuntu"),
    (r"zorin", "ubuntu"),
    (r"elementary", "ubuntu"),
    (r"fedora", "fedora"),
    (r"nobara", "fedora"),
    (r"bazzite", "fedora"),
    (r"opensuse|suse", "suse"),
    (r"debian", "debian"),
    (r"gentoo", "gentoo"),
    (r"nixos", "nixos"),
    (r"void", "void"),
    (r"solus", "solus"),
    (r"steamos", "steamos"),
]


def os_family_from_string(os_str: str | None) -> str | None:
    """Extract OS family from a raw os string."""
    if not os_str:
        return None
    os_lower = os_str.lower()
    for pattern, family in _OS_PATTERNS:
        if re.search(pattern, os_lower):
            return family
    return "other"


# ---------------------------------------------------------------------------
# GPU vendor/family extraction from raw GPU string (fallback when
# gpu_normalization is empty)
# ---------------------------------------------------------------------------

_GPU_VENDOR_PATTERNS = [
    (r"nvidia|geforce|gtx|rtx|quadro", "nvidia"),
    (r"amd|radeon|ati", "amd"),
    (r"intel|iris|uhd|hd graphics|xe", "intel"),
]

_GPU_FAMILY_PATTERNS = [
    # NVIDIA
    (r"rtx\s*50\d0", "rtx50"),
    (r"rtx\s*40\d0", "rtx40"),
    (r"rtx\s*30\d0", "rtx30"),
    (r"rtx\s*20\d0", "rtx20"),
    (r"gtx\s*16\d0", "gtx16"),
    (r"gtx\s*10\d0", "gtx10"),
    (r"gtx\s*9\d0", "gtx9"),
    (r"gtx\s*7\d0", "gt7"),
    (r"gt\s*10\d0", "gt10"),
    # AMD
    (r"rx\s*9\d{3}", "rx9000"),
    (r"rx\s*7\d{3}", "rx7000"),
    (r"rx\s*6\d{3}", "rx6000"),
    (r"rx\s*5\d{3}", "rx5000"),
    (r"rx\s*[45]\d{2}\b", "rx500"),
    (r"r9\s", "r9"),
    (r"r7\s", "r7"),
    (r"vega", "vega"),
    (r"rdna\s*4", "rdna4"),
    (r"rdna\s*3", "rdna3"),
    (r"rdna\s*2", "rdna2"),
    # Intel
    (r"arc\s*[ab]", "arc"),
    (r"iris\s*xe", "xe"),
    (r"uhd\s*7", "xe"),
    (r"uhd\s*6", "hd600"),
    (r"hd\s*6\d{2}", "hd600"),
    (r"hd\s*5\d{2}", "hd5"),
    (r"hd\s*4\d{2}", "hd4"),
]


def extract_gpu_vendor(gpu_str: str | None) -> str | None:
    """Extract GPU vendor from a raw string."""
    if not gpu_str:
        return None
    gpu_lower = gpu_str.lower()
    for pattern, vendor in _GPU_VENDOR_PATTERNS:
        if re.search(pattern, gpu_lower):
            return vendor
    return None


def extract_gpu_family(gpu_str: str | None) -> str | None:
    """Extract GPU family from a raw string (heuristic fallback)."""
    if not gpu_str:
        return None
    gpu_lower = gpu_str.lower()
    for pattern, family in _GPU_FAMILY_PATTERNS:
        if re.search(pattern, gpu_lower):
            return family
    return None


# ---------------------------------------------------------------------------
# CPU vendor/generation extraction from raw CPU string
# ---------------------------------------------------------------------------

def extract_cpu_vendor(cpu_str: str | None) -> str | None:
    """Extract CPU vendor from a raw string."""
    if not cpu_str:
        return None
    cpu_lower = cpu_str.lower()
    if "intel" in cpu_lower:
        return "intel"
    if "amd" in cpu_lower or "ryzen" in cpu_lower or "athlon" in cpu_lower:
        return "amd"
    return None


def extract_cpu_family(cpu_str: str | None) -> str | None:
    """Extract CPU family from a raw string (heuristic fallback)."""
    if not cpu_str:
        return None
    cpu_lower = cpu_str.lower()
    # AMD Ryzen
    m = re.search(r"ryzen\s*(\d)\s*(\d)", cpu_lower)
    if m:
        return f"ryzen_{m.group(1)}_{m.group(2)}xxx"
    m = re.search(r"ryzen\s*(\d)", cpu_lower)
    if m:
        return f"ryzen_{m.group(1)}"
    # Intel Core i-series
    m = re.search(r"i[3579]-(\d{1,2})\d{2,3}", cpu_lower)
    if m:
        gen = m.group(1)
        return f"intel_gen{gen}"
    # Intel 12th/13th/14th gen
    m = re.search(r"(\d{2})th gen", cpu_lower)
    if m:
        return f"intel_gen{m.group(1)}"
    return None


def extract_cpu_generation(cpu_str: str | None) -> int | None:
    """Extract CPU generation number from a raw string."""
    if not cpu_str:
        return None
    cpu_lower = cpu_str.lower()
    # AMD Ryzen: generation from model number (Ryzen 5 5600X -> gen 5)
    m = re.search(r"ryzen\s*\d\s*(\d)", cpu_lower)
    if m:
        return int(m.group(1))
    # Intel: i7-12700K -> gen 12
    m = re.search(r"i[3579]-(\d{1,2})\d{2,3}", cpu_lower)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Top-N categorical encoding
# ---------------------------------------------------------------------------


class LabelEncoder:
    """Simple label encoder with top-N support and 'other' bucket."""

    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.label_map: dict[str, int] = {}
        self.inverse_map: dict[int, str] = {}

    def fit(self, values: list[str | None]) -> "LabelEncoder":
        """Fit on a list of values, keeping top-N by frequency."""
        from collections import Counter

        counts = Counter(v for v in values if v is not None)
        top = [k for k, _ in counts.most_common(self.top_n)]
        self.label_map = {"__other__": 0}
        for i, label in enumerate(top, start=1):
            self.label_map[label] = i
        self.inverse_map = {v: k for k, v in self.label_map.items()}
        return self

    def transform(self, value: str | None) -> int:
        """Transform a single value to its encoded integer."""
        if value is None:
            return -1  # missing
        return self.label_map.get(value, 0)  # 0 = __other__

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        return {"top_n": self.top_n, "label_map": self.label_map}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelEncoder":
        """Deserialize from a dict."""
        enc = cls(top_n=d["top_n"])
        enc.label_map = d["label_map"]
        enc.inverse_map = {v: k for k, v in enc.label_map.items()}
        return enc


class LabelMaps:
    """Collection of label encoders for all categorical features."""

    def __init__(self) -> None:
        self.encoders: dict[str, LabelEncoder] = {}

    def fit_column(self, name: str, values: list[str | None], top_n: int = 50) -> None:
        """Fit an encoder for a named column."""
        enc = LabelEncoder(top_n=top_n)
        enc.fit(values)
        self.encoders[name] = enc

    def transform(self, name: str, value: str | None) -> int:
        """Transform a value using a named encoder."""
        if name not in self.encoders:
            return -1
        return self.encoders[name].transform(value)

    def save(self, path: Path) -> None:
        """Save all encoders to a JSON file."""
        data = {name: enc.to_dict() for name, enc in self.encoders.items()}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LabelMaps":
        """Load encoders from a JSON file."""
        maps = cls()
        data = json.loads(path.read_text())
        for name, d in data.items():
            maps.encoders[name] = LabelEncoder.from_dict(d)
        return maps
