#!/usr/bin/env python3
"""Phase 6a: Zero-cost report features experiments.

Tests:
  A4: Per-report cust_* and flag_* fields
  A1: 8 fault booleans (audio_faults..significant_bugs)
  A3: duration + tried_oob (with leak check via tinker_override)
  A5: is_impacted_by_anticheat
  B1: Variant debiasing (drop variant from Stage 2)
  COMBINED: Best features together
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.cascade import (
    CascadeClassifier,
    STAGE2_DROP_FEATURES,
    train_stage1,
    train_stage2,
)
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.train import _build_feature_matrix, _compute_target

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]

# ──── Feature definitions ────

# A4: cust_* and flag_* (per-report customization/flag fields)
CUST_FIELDS = [
    "cust_protontricks", "cust_config_change", "cust_custom_prefix",
    "cust_custom_proton", "cust_lutris", "cust_media_foundation",
    "cust_protonfixes", "cust_native2proton", "cust_not_listed", "cust_winetricks",
]
FLAG_FIELDS = [
    "flag_disable_esync", "flag_disable_fsync", "flag_enable_nvapi",
    "flag_use_wine_d3d11", "flag_use_wine_d9vk", "flag_large_address_aware",
    "flag_disable_d3d11", "flag_hide_nvidia", "flag_game_drive",
]
# Aggregate features derived from cust/flag
CUST_FLAG_DERIVED = [
    "cust_any", "cust_count", "flag_any", "flag_count",
]

# A1: fault booleans
FAULT_FIELDS = [
    "audio_faults", "graphical_faults", "input_faults", "performance_faults",
    "stability_faults", "windowing_faults", "save_game_faults", "significant_bugs",
]
FAULT_DERIVED = ["fault_any", "fault_count"]

# A3: duration + tried_oob + tinker_override
DURATION_FIELDS = ["duration_ord", "tried_oob_bin", "tinker_override_bin"]

# A5: is_impacted_by_anticheat
ANTICHEAT_FIELDS = ["is_impacted_by_anticheat_bin"]

DURATION_MAP = {
    "lessThanFifteenMinutes": 0,
    "lessThanAnHour": 1,
    "aboutAnHour": 2,
    "severalHours": 3,
    "moreThanTenHours": 4,
}


def load_data():
    """Load baseline features + all new per-report fields."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = (
        emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    )
    emb_data["n_components_cpu"] = (
        emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0
    )

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Fetch new fields aligned with X
    # We need the same filtering as _build_feature_matrix (skip reports with target=None)
    all_fields = CUST_FIELDS + FLAG_FIELDS + FAULT_FIELDS + [
        "duration", "tried_oob", "tinker_override", "is_impacted_by_anticheat",
    ]
    cols_sql = ", ".join(all_fields)
    rows = conn.execute(
        f"SELECT verdict, verdict_oob, {cols_sql} FROM reports"
    ).fetchall()
    conn.close()

    records = []
    for row in rows:
        target = _compute_target(row["verdict"], row["verdict_oob"])
        if target is None:
            continue

        rec = {}

        # A4: cust/flag fields (INTEGER in DB, NULL → NaN, else 0/1)
        for f in CUST_FIELDS + FLAG_FIELDS:
            val = row[f]
            rec[f] = float(val) if val is not None else np.nan

        # A4 derived: any/count
        cust_vals = [row[f] for f in CUST_FIELDS]
        cust_known = [v for v in cust_vals if v is not None]
        rec["cust_any"] = float(any(v == 1 for v in cust_known)) if cust_known else np.nan
        rec["cust_count"] = float(sum(v == 1 for v in cust_known)) if cust_known else np.nan

        flag_vals = [row[f] for f in FLAG_FIELDS]
        flag_known = [v for v in flag_vals if v is not None]
        rec["flag_any"] = float(any(v == 1 for v in flag_known)) if flag_known else np.nan
        rec["flag_count"] = float(sum(v == 1 for v in flag_known)) if flag_known else np.nan

        # A1: fault booleans (TEXT yes/no → 1/0)
        fault_vals = []
        for f in FAULT_FIELDS:
            val = row[f]
            if val == "yes":
                rec[f] = 1.0
            elif val == "no":
                rec[f] = 0.0
            else:
                rec[f] = np.nan
            fault_vals.append(rec[f])

        known_faults = [v for v in fault_vals if not np.isnan(v)]
        rec["fault_any"] = float(any(v == 1 for v in known_faults)) if known_faults else np.nan
        rec["fault_count"] = float(sum(v for v in known_faults)) if known_faults else np.nan

        # A3: duration (ordinal), tried_oob, tinker_override
        dur = row["duration"]
        rec["duration_ord"] = float(DURATION_MAP[dur]) if dur in DURATION_MAP else np.nan

        rec["tried_oob_bin"] = (
            1.0 if row["tried_oob"] == "yes" else
            0.0 if row["tried_oob"] == "no" else np.nan
        )
        rec["tinker_override_bin"] = (
            1.0 if row["tinker_override"] == "yes" else
            0.0 if row["tinker_override"] == "no" else np.nan
        )

        # A5: anticheat
        rec["is_impacted_by_anticheat_bin"] = (
            1.0 if row["is_impacted_by_anticheat"] == "yes" else
            0.0 if row["is_impacted_by_anticheat"] == "no" else np.nan
        )

        records.append(rec)

    new_df = pd.DataFrame(records)
    assert len(new_df) == len(X), f"Mismatch: {len(new_df)} vs {len(X)}"

    return X, y, timestamps, new_df


def train_and_eval(X_train, y_train, X_test, y_test, label: str,
                   stage2_extra_drop=None):
    """Train cascade and evaluate. Returns (f1_macro, cascade)."""
    s1 = train_stage1(X_train, y_train, X_test, y_test)

    if stage2_extra_drop:
        drop_cols = list(set(STAGE2_DROP_FEATURES + stage2_extra_drop))
        s2, s2_dropped = train_stage2(X_train, y_train, X_test, y_test,
                                       drop_features=drop_cols)
    else:
        s2, s2_dropped = train_stage2(X_train, y_train, X_test, y_test)

    cascade = CascadeClassifier(s1, s2, s2_dropped)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES_3, digits=4))
    return f1, cascade


def add_features(X, new_df, columns):
    """Concatenate selected new features to X."""
    return pd.concat([X, new_df[columns].reset_index(drop=True)], axis=1)


def print_importance(cascade, feature_prefix_list, label):
    """Print importance for features matching prefixes."""
    print(f"\n{'#' * 60}")
    print(f"# {label} — feature importance")
    print(f"{'#' * 60}")

    for stage_name, model in [("Stage 1", cascade.stage1), ("Stage 2", cascade.stage2)]:
        features = model.feature_name_
        importances = model.feature_importances_
        matched = [f for f in features if any(f.startswith(p) or f == p for p in feature_prefix_list)]
        if matched:
            print(f"\n  {stage_name}:")
            for fname in sorted(matched, key=lambda f: -importances[features.index(f)]):
                idx = features.index(fname)
                print(f"    {fname:42s} {importances[idx]:10.1f}")


def main():
    print("Loading data...")
    X, y, timestamps, new_df = load_data()
    print(f"Loaded {len(X)} samples, {X.shape[1]} baseline features")
    print(f"New features available: {new_df.shape[1]} columns")

    # Coverage stats
    for group_name, cols in [
        ("A4 cust", CUST_FIELDS),
        ("A4 flag", FLAG_FIELDS),
        ("A1 faults", FAULT_FIELDS),
        ("A3 duration", ["duration_ord"]),
        ("A3 tried_oob", ["tried_oob_bin"]),
        ("A3 tinker_override", ["tinker_override_bin"]),
        ("A5 anticheat", ["is_impacted_by_anticheat_bin"]),
    ]:
        non_nan = new_df[cols].notna().any(axis=1).sum()
        print(f"  {group_name:25s}: {non_nan:6d}/{len(new_df)} ({non_nan/len(new_df):.1%})")

    # Time-based split
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    new_train = new_df.iloc[train_idx].reset_index(drop=True)
    new_test = new_df.iloc[test_idx].reset_index(drop=True)

    results = {}

    # ──── Baseline ────
    f1_base, _ = train_and_eval(X_train, y_train, X_test, y_test, "Baseline")
    results["baseline"] = f1_base

    # ──── A1: Fault booleans ────
    a1_cols = FAULT_FIELDS + FAULT_DERIVED
    Xtr_a1 = pd.concat([X_train, new_train[a1_cols]], axis=1)
    Xte_a1 = pd.concat([X_test, new_test[a1_cols]], axis=1)
    f1_a1, cascade_a1 = train_and_eval(Xtr_a1, y_train, Xte_a1, y_test, "A1: Fault booleans (10 features)")
    results["A1_faults"] = f1_a1
    print_importance(cascade_a1, FAULT_FIELDS + FAULT_DERIVED, "A1 faults")

    # ──── A4: cust/flag per-report fields ────
    a4_cols = CUST_FIELDS + FLAG_FIELDS + CUST_FLAG_DERIVED
    Xtr_a4 = pd.concat([X_train, new_train[a4_cols]], axis=1)
    Xte_a4 = pd.concat([X_test, new_test[a4_cols]], axis=1)
    f1_a4, cascade_a4 = train_and_eval(Xtr_a4, y_train, Xte_a4, y_test, "A4: Cust+Flag features (23 features)")
    results["A4_cust_flag"] = f1_a4
    print_importance(cascade_a4, CUST_FIELDS + FLAG_FIELDS + CUST_FLAG_DERIVED, "A4 cust+flag")

    # ──── A3: duration + tried_oob ────
    # First: tried_oob + duration (no tinker_override — potential leak check)
    a3_cols_safe = ["duration_ord", "tried_oob_bin"]
    Xtr_a3 = pd.concat([X_train, new_train[a3_cols_safe]], axis=1)
    Xte_a3 = pd.concat([X_test, new_test[a3_cols_safe]], axis=1)
    f1_a3, cascade_a3 = train_and_eval(Xtr_a3, y_train, Xte_a3, y_test, "A3: duration + tried_oob")
    results["A3_dur_triedoob"] = f1_a3

    # A3b: with tinker_override (leak check)
    a3_all = ["duration_ord", "tried_oob_bin", "tinker_override_bin"]
    Xtr_a3b = pd.concat([X_train, new_train[a3_all]], axis=1)
    Xte_a3b = pd.concat([X_test, new_test[a3_all]], axis=1)
    f1_a3b, cascade_a3b = train_and_eval(Xtr_a3b, y_train, Xte_a3b, y_test,
                                          "A3b: duration + tried_oob + tinker_override (LEAK CHECK)")
    results["A3b_with_tinker"] = f1_a3b
    print_importance(cascade_a3b, a3_all, "A3b duration+tried+tinker")

    # ──── A5: is_impacted_by_anticheat ────
    Xtr_a5 = pd.concat([X_train, new_train[ANTICHEAT_FIELDS]], axis=1)
    Xte_a5 = pd.concat([X_test, new_test[ANTICHEAT_FIELDS]], axis=1)
    f1_a5, _ = train_and_eval(Xtr_a5, y_train, Xte_a5, y_test, "A5: is_impacted_by_anticheat")
    results["A5_anticheat"] = f1_a5

    # ──── B1: Variant debiasing (drop variant from Stage 2) ────
    f1_b1, cascade_b1 = train_and_eval(X_train, y_train, X_test, y_test,
                                         "B1: Drop variant from Stage 2",
                                         stage2_extra_drop=["variant"])
    results["B1_no_variant_s2"] = f1_b1

    # ──── COMBINED: best features together ────
    # Combine A1 + A4 + A3(safe) + A5
    combined_cols = (
        FAULT_FIELDS + FAULT_DERIVED +
        CUST_FIELDS + FLAG_FIELDS + CUST_FLAG_DERIVED +
        a3_cols_safe + ANTICHEAT_FIELDS
    )
    Xtr_comb = pd.concat([X_train, new_train[combined_cols]], axis=1)
    Xte_comb = pd.concat([X_test, new_test[combined_cols]], axis=1)
    f1_comb, cascade_comb = train_and_eval(
        Xtr_comb, y_train, Xte_comb, y_test,
        "COMBINED: A1+A4+A3+A5 (all zero-cost features)"
    )
    results["COMBINED"] = f1_comb
    print_importance(cascade_comb,
                     FAULT_FIELDS + FAULT_DERIVED + CUST_FIELDS + FLAG_FIELDS +
                     CUST_FLAG_DERIVED + a3_cols_safe + ANTICHEAT_FIELDS,
                     "COMBINED all new features")

    # ──── COMBINED + B1: all features + variant debiasing ────
    f1_comb_b1, cascade_comb_b1 = train_and_eval(
        Xtr_comb, y_train, Xte_comb, y_test,
        "COMBINED + B1: all features + drop variant S2",
        stage2_extra_drop=["variant"]
    )
    results["COMBINED_B1"] = f1_comb_b1

    # ──── Summary ────
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, f1 in sorted(results.items(), key=lambda x: -x[1]):
        delta = f1 - f1_base
        marker = " <<<" if delta > 0.002 else ""
        print(f"  {name:25s}  F1={f1:.4f}  Δ={delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
