"""Single-sample prediction: will a game work on the current PC?

Loads trained cascade model + embeddings, detects local GPU hardware,
builds a feature vector for a given app_id, and runs prediction.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local GPU detection
# ---------------------------------------------------------------------------

def detect_gpu() -> dict[str, str | None]:
    """Detect local GPU name and driver version.

    Returns dict with keys: gpu_raw, driver_version, vendor.
    """
    gpu_raw = None
    driver_version = None
    vendor = None

    # Try nvidia-smi first (NVIDIA)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = out.stdout.strip().split(",")
            gpu_raw = parts[0].strip()
            driver_version = parts[1].strip() if len(parts) > 1 else None
            vendor = "nvidia"
            return {"gpu_raw": gpu_raw, "driver_version": driver_version, "vendor": vendor}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try lspci (AMD/Intel/any)
    try:
        out = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                if "VGA" in line or "3D" in line:
                    # Extract GPU name after the bracket
                    m = re.search(r":\s+(.+)$", line)
                    if m:
                        gpu_raw = m.group(1).strip()
                        if "nvidia" in gpu_raw.lower():
                            vendor = "nvidia"
                        elif "amd" in gpu_raw.lower() or "radeon" in gpu_raw.lower():
                            vendor = "amd"
                        elif "intel" in gpu_raw.lower():
                            vendor = "intel"
                        break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try to get Mesa driver version
    if vendor != "nvidia":
        try:
            out = subprocess.run(
                ["glxinfo"], capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                m = re.search(r"OpenGL version string:.*?(\d+\.\d+)", out.stdout)
                if m:
                    driver_version = m.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return {"gpu_raw": gpu_raw, "driver_version": driver_version, "vendor": vendor}


def _parse_nvidia_driver(version_str: str | None) -> float | None:
    """Parse NVIDIA driver version string to float (e.g. '550.127.05' -> 550.127)."""
    if not version_str:
        return None
    parts = version_str.split(".")
    if len(parts) >= 2:
        try:
            return float(parts[0]) + float(parts[1]) / 1000.0
        except ValueError:
            return None
    return None


def _parse_mesa_driver(version_str: str | None) -> float | None:
    """Parse Mesa driver version to float (e.g. '24.1' -> 24.1)."""
    if not version_str:
        return None
    try:
        return float(version_str)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Feature building for a single prediction
# ---------------------------------------------------------------------------

def build_single_features(
    app_id: int,
    gpu_info: dict[str, str | None],
    emb_data: dict[str, Any],
    variant: str = "official",
) -> pd.DataFrame:
    """Build a feature DataFrame for a single app_id + current GPU.

    Text features are set to 0/NaN (not available at inference time).
    """
    from .features.encoding import extract_gpu_family

    gpu_raw = gpu_info.get("gpu_raw")
    gpu_family = extract_gpu_family(gpu_raw)
    vendor = gpu_info.get("vendor")

    nvidia_driver_version = None
    mesa_driver_version = None
    if vendor == "nvidia":
        nvidia_driver_version = _parse_nvidia_driver(gpu_info.get("driver_version"))
    else:
        mesa_driver_version = _parse_mesa_driver(gpu_info.get("driver_version"))

    # Check Steam Deck
    gpu_lower = (gpu_raw or "").lower()
    is_steam_deck = 1 if ("vangogh" in gpu_lower or "van gogh" in gpu_lower) else 0

    record: dict[str, Any] = {
        "gpu_family": gpu_family,
        "nvidia_driver_version": nvidia_driver_version,
        "mesa_driver_version": mesa_driver_version,
        "is_apu": 0,
        "is_igpu": 0,
        "is_mobile": 0,
        "is_steam_deck": is_steam_deck,
        "variant": variant,
        # Text features — not available at inference, set to zero
        "has_concluding_notes": 0,
        "concluding_notes_length": 0,
        "fault_notes_count": 0,
        "has_customization_notes": 0,
        "total_notes_length": 0,
        "mentions_crash": 0,
        "mentions_fix": 0,
        "mentions_perfect": 0,
        "mentions_proton_version": 0,
        "mentions_env_var": 0,
        "mentions_performance": 0,
        "sentiment_negative_words": 0,
        "sentiment_positive_words": 0,
    }

    # GPU embeddings
    gpu_families = emb_data.get("gpu_families", [])
    gpu_emb_matrix = emb_data.get("gpu_embeddings", np.array([]))
    gpu_family_to_idx = {f: i for i, f in enumerate(gpu_families)}

    n_gpu_emb = gpu_emb_matrix.shape[1] if gpu_emb_matrix.ndim == 2 and gpu_emb_matrix.size > 0 else 0

    if gpu_family and gpu_family in gpu_family_to_idx and gpu_emb_matrix.size > 0:
        idx = gpu_family_to_idx[gpu_family]
        for d in range(n_gpu_emb):
            record[f"gpu_emb_{d}"] = float(gpu_emb_matrix[idx, d])
    else:
        for d in range(n_gpu_emb):
            record[f"gpu_emb_{d}"] = np.nan

    # Game embeddings
    game_ids = emb_data.get("game_ids", [])
    game_emb_matrix = emb_data.get("game_embeddings", np.array([]))
    game_id_to_idx = {int(g): i for i, g in enumerate(game_ids)}

    if app_id in game_id_to_idx and game_emb_matrix.size > 0:
        idx = game_id_to_idx[app_id]
        for d in range(n_gpu_emb):
            record[f"game_emb_{d}"] = float(game_emb_matrix[idx, d])
    else:
        for d in range(n_gpu_emb):
            record[f"game_emb_{d}"] = np.nan

    # Per-game aggregate features (Phase 9.2: cust_* + flag_*)
    game_agg_cust = emb_data.get("game_agg_cust")
    game_agg_flag = emb_data.get("game_agg_flag")
    if game_agg_cust is not None and app_id in game_id_to_idx:
        idx = game_id_to_idx[app_id]
        for j, col in enumerate(emb_data["game_agg_columns_cust"]):
            record[col] = float(game_agg_cust[idx, j])
        for j, col in enumerate(emb_data["game_agg_columns_flag"]):
            record[col] = float(game_agg_flag[idx, j])
    elif game_agg_cust is not None:
        for col in emb_data["game_agg_columns_cust"]:
            record[col] = np.nan
        for col in emb_data["game_agg_columns_flag"]:
            record[col] = np.nan

    # Text embeddings — NaN at inference
    n_text_emb = emb_data.get("text_n_components", 0)
    for d in range(n_text_emb):
        record[f"text_emb_{d}"] = np.nan

    X = pd.DataFrame([record])

    # Coerce numeric columns that may be None (object dtype)
    for col in ("nvidia_driver_version", "mesa_driver_version"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Coerce any remaining object columns (except variant)
    for col in X.columns:
        if X[col].dtype == object and col != "variant":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Ensure variant is categorical
    if "variant" in X.columns:
        X["variant"] = X["variant"].astype("category")

    return X


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

def _get_game_metadata(app_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    """Load game metadata from DB for override signals."""
    if db_path is None:
        # Try default path relative to model
        db_path = Path("data/protondb.db")
    if not db_path.exists():
        return None
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT anticheat, anticheat_status, deck_status, protondb_tier "
            "FROM game_metadata WHERE app_id = ?",
            (app_id,),
        ).fetchone()
        conn.close()
        if row:
            return dict(row)
    except Exception:
        pass
    return None


# Anticheat systems known to block Linux without explicit support
_BLOCKING_ANTICHEATS = {"battleye", "easyanticheat", "easy anti-cheat", "vanguard", "ricochet"}

# Anticheats that are NOT blocking (client-side or Proton-friendly)
_SAFE_ANTICHEATS = {"valve anti-cheat", "vac"}


def _check_metadata_override(
    meta: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Check if game metadata indicates borked (blocking anticheat, etc.).

    Uses AWACY anticheat_status (Supported/Running/Broken/Denied) and
    deck_status as strong signals.

    Returns override dict with prediction/probabilities, or None if no override.
    """
    if meta is None:
        return None

    anticheat_raw = (meta.get("anticheat") or "").lower()
    anticheat_status = (meta.get("anticheat_status") or "").lower()
    deck_status = meta.get("deck_status")  # 1=unsupported, 2=playable, 3=verified

    # AWACY says Supported/Running → anticheat works on Linux, no override
    if anticheat_status in ("supported", "running"):
        return None

    # Safe anticheats (VAC etc.) — never override
    if anticheat_raw and all(
        any(safe in part for safe in _SAFE_ANTICHEATS)
        for part in anticheat_raw.split(",")
        if part.strip()
    ):
        return None

    # AWACY explicitly says Broken or Denied → borked,
    # BUT Deck verified/playable overrides (AWACY can be stale)
    if anticheat_status in ("broken", "denied"):
        if deck_status is not None and deck_status >= 2:
            # AWACY says broken but Valve says it works — trust Valve
            return None
        reason = f"AWACY status: {meta.get('anticheat_status')}, anticheat: {meta.get('anticheat')}"
        if deck_status == 1:
            reason += ", Deck unsupported"
        return {
            "override": True,
            "override_reason": reason,
            "prediction": "borked",
            "probabilities": {"borked": 0.95, "tinkering": 0.04, "works_oob": 0.01},
            "confidence": 0.95,
            "is_confident": True,
        }

    # No AWACY status — fallback to heuristic: blocking anticheat name + deck status
    has_blocker = any(ac in anticheat_raw for ac in _BLOCKING_ANTICHEATS)
    if not has_blocker:
        return None

    # Deck verified/playable = anticheat works on Linux despite blocker name
    if deck_status is not None and deck_status >= 2:
        return None

    reason = f"blocking anticheat: {meta.get('anticheat')}"
    if deck_status == 1:
        reason += ", Deck unsupported"

    return {
        "override": True,
        "override_reason": reason,
        "prediction": "borked",
        "probabilities": {"borked": 0.95, "tinkering": 0.04, "works_oob": 0.01},
        "confidence": 0.95,
        "is_confident": True,
    }


def predict_for_app(
    app_id: int,
    model_path: Path,
    embeddings_path: Path,
    variant: str = "official",
    gpu_override: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Predict compatibility for app_id on the current PC.

    Uses metadata overrides for obvious cases (blocking anticheat),
    falls back to cascade model prediction.

    Returns dict with prediction, probabilities, confidence, and details.
    """
    import joblib

    from .features.embeddings import load_embeddings
    from .features.encoding import extract_gpu_family

    # Detect GPU early (needed for all paths)
    if gpu_override:
        gpu_info = {"gpu_raw": gpu_override, "driver_version": None, "vendor": None}
        gl = gpu_override.lower()
        if "nvidia" in gl or "geforce" in gl or "rtx" in gl or "gtx" in gl:
            gpu_info["vendor"] = "nvidia"
        elif "amd" in gl or "radeon" in gl:
            gpu_info["vendor"] = "amd"
        elif "intel" in gl:
            gpu_info["vendor"] = "intel"
    else:
        gpu_info = detect_gpu()

    # Check metadata overrides (blocking anticheat, etc.)
    meta = _get_game_metadata(app_id, db_path)
    override = _check_metadata_override(meta)

    emb_data = load_embeddings(embeddings_path)
    has_game_emb = app_id in {int(g) for g in emb_data.get("game_ids", [])}

    if override:
        return {
            "app_id": app_id,
            **{k: override[k] for k in ("prediction", "probabilities", "confidence", "is_confident")},
            "override": True,
            "override_reason": override["override_reason"],
            "gpu": gpu_info.get("gpu_raw"),
            "gpu_family": extract_gpu_family(gpu_info.get("gpu_raw")),
            "driver_version": gpu_info.get("driver_version"),
            "variant": variant,
            "has_game_embedding": has_game_emb,
        }

    # Load model and predict
    cascade = joblib.load(model_path)
    X = build_single_features(app_id, gpu_info, emb_data, variant=variant)

    # Ensure feature order matches model expectations
    if hasattr(cascade, 'stage1') and hasattr(cascade.stage1, 'feature_name_'):
        expected_cols = cascade.stage1.feature_name_
        for col in expected_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[expected_cols]

    # Ensure categorical columns
    from .models.classifier import CATEGORICAL_FEATURES
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    result = cascade.predict_with_confidence(X)

    labels = {0: "borked", 1: "tinkering", 2: "works_oob"}
    pred_idx = int(result["prediction"][0])

    return {
        "app_id": app_id,
        "prediction": labels[pred_idx],
        "probabilities": {
            "borked": float(result["probabilities"][0, 0]),
            "tinkering": float(result["probabilities"][0, 1]),
            "works_oob": float(result["probabilities"][0, 2]),
        },
        "confidence": float(result["confidence"][0]),
        "is_confident": bool(result["is_confident"][0]),
        "override": False,
        "override_reason": None,
        "gpu": gpu_info.get("gpu_raw"),
        "gpu_family": extract_gpu_family(gpu_info.get("gpu_raw")),
        "driver_version": gpu_info.get("driver_version"),
        "variant": variant,
        "has_game_embedding": has_game_emb,
    }
