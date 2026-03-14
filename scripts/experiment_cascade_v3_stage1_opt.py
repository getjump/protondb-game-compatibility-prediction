#!/usr/bin/env python3
"""Experiment: Stage 1 optimization (borked vs works).

Tests from PLAN_ML_4.md step 4:
  4a. Remove report_age_days from Stage 1
  4b. Add pct_stability_faults × is_steam_deck interaction
  4c. Threshold tuning for P(borked)
  4d. Class weight grid search
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.models.cascade import train_stage2, CascadeClassifier

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")

CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]


def load_data():
    """Load feature matrix and split."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_test, y_train, y_test = _time_based_split(X, y, timestamps, 0.2)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_test.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    conn.close()
    return X_train, X_test, y_train, y_test, cat_cols


def train_stage1_variant(
    X_train, y_train, X_test, y_test, cat_cols,
    class_weight=None, drop_features=None, add_interactions=False,
):
    """Train a Stage 1 variant and return (model, X_train_used, X_test_used)."""
    if class_weight is None:
        class_weight = {0: 3.0, 1: 1.0}
    if drop_features is None:
        drop_features = []

    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Drop features
    for f in drop_features:
        if f in X_tr.columns:
            X_tr = X_tr.drop(columns=[f])
            X_te = X_te.drop(columns=[f])

    # Add interactions
    if add_interactions:
        if "pct_stability_faults" in X_tr.columns:
            is_deck = (X_tr.get("is_steam_deck", pd.Series(0, index=X_tr.index))).fillna(0)
            X_tr["stability_x_deck"] = X_tr["pct_stability_faults"] * is_deck
            is_deck_te = (X_te.get("is_steam_deck", pd.Series(0, index=X_te.index))).fillna(0)
            X_te["stability_x_deck"] = X_te["pct_stability_faults"] * is_deck_te

    y_tr_bin = (y_train > 0).astype(int)
    y_te_bin = (y_test > 0).astype(int)

    cat_cols_used = [c for c in cat_cols if c in X_tr.columns]
    for col in cat_cols_used:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")

    model = lgb.LGBMClassifier(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight=class_weight, n_jobs=-1, random_state=42,
        verbose=-1, importance_type="gain",
    )

    model.fit(
        X_tr, y_tr_bin,
        eval_set=[(X_te, y_te_bin)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=500),
        ],
        categorical_feature=cat_cols_used,
    )
    return model, X_tr, X_te


def cascade_evaluate(s1_model, X_test_s1, s2_model, s2_dropped, X_test_full, y_test, threshold=0.5):
    """Evaluate full cascade with a given Stage 1 model and threshold."""
    p_s1 = s1_model.predict_proba(X_test_s1)
    borked_mask = p_s1[:, 0] >= threshold

    result = np.full(len(X_test_full), 1, dtype=int)
    result[borked_mask] = 0

    works_mask = ~borked_mask
    if works_mask.any():
        X_s2 = X_test_full[works_mask].copy()
        cols_to_drop = [c for c in s2_dropped if c in X_s2.columns]
        if cols_to_drop:
            X_s2 = X_s2.drop(columns=cols_to_drop)
        p_s2 = s2_model.predict_proba(X_s2)
        result[works_mask] = np.where(p_s2[:, 1] >= 0.5, 2, 1)

    acc = accuracy_score(y_test, result)
    f1 = f1_score(y_test, result, average="macro", zero_division=0)

    borked_r = (result[y_test == 0] == 0).mean() if (y_test == 0).any() else 0
    borked_p = (y_test[result == 0] == 0).mean() if (result == 0).any() else 0
    oob_r = (result[y_test == 2] == 2).mean() if (y_test == 2).any() else 0
    oob_p = (y_test[result == 2] == 2).mean() if (result == 2).any() else 0

    return {
        "acc": acc, "f1": f1,
        "borked_r": borked_r, "borked_p": borked_p,
        "oob_r": oob_r, "oob_p": oob_p,
        "y_pred": result,
    }


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, cat_cols = load_data()
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}\n")

    # Train Stage 2 once (best config from v2: no report_age_days)
    print("=" * 60)
    print("Training Stage 2 (fixed: no report_age_days, weight={0:1, 1:2})")
    print("=" * 60)
    s2_model, s2_dropped = train_stage2(
        X_train, y_train, X_test, y_test,
        class_weight={0: 1.0, 1: 2.0},
    )
    print(f"  Stage 2 best iteration: {s2_model.best_iteration_}")
    print(f"  Dropped: {s2_dropped}\n")

    # ===== BASELINE: current Stage 1 =====
    print("=" * 60)
    print("BASELINE: Stage 1 with all features, weight={0:3, 1:1}")
    print("=" * 60)
    s1_base, X_tr_base, X_te_base = train_stage1_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 3.0, 1: 1.0},
    )
    r_base = cascade_evaluate(s1_base, X_te_base, s2_model, s2_dropped, X_test, y_test)
    print(f"  Cascade: acc={r_base['acc']:.4f}, F1={r_base['f1']:.4f}")
    print(f"  borked: R={r_base['borked_r']:.4f}, P={r_base['borked_p']:.4f}")
    print(f"  oob:    R={r_base['oob_r']:.4f}, P={r_base['oob_p']:.4f}")
    print()

    # ===== 4a: Remove report_age_days from Stage 1 =====
    print("=" * 60)
    print("4a: Stage 1 WITHOUT report_age_days")
    print("=" * 60)
    s1_4a, X_tr_4a, X_te_4a = train_stage1_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 3.0, 1: 1.0},
        drop_features=["report_age_days"],
    )
    r_4a = cascade_evaluate(s1_4a, X_te_4a, s2_model, s2_dropped, X_test, y_test)
    print(f"  Cascade: acc={r_4a['acc']:.4f}, F1={r_4a['f1']:.4f}")
    print(f"  borked: R={r_4a['borked_r']:.4f}, P={r_4a['borked_p']:.4f}")
    print(f"  oob:    R={r_4a['oob_r']:.4f}, P={r_4a['oob_p']:.4f}")
    delta = r_4a['f1'] - r_base['f1']
    print(f"  vs baseline: F1 {delta:+.4f}\n")

    # ===== 4b: Add interaction feature =====
    print("=" * 60)
    print("4b: Stage 1 + pct_stability_faults × is_steam_deck")
    print("=" * 60)
    s1_4b, X_tr_4b, X_te_4b = train_stage1_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 3.0, 1: 1.0},
        add_interactions=True,
    )
    r_4b = cascade_evaluate(s1_4b, X_te_4b, s2_model, s2_dropped, X_test, y_test)
    print(f"  Cascade: acc={r_4b['acc']:.4f}, F1={r_4b['f1']:.4f}")
    print(f"  borked: R={r_4b['borked_r']:.4f}, P={r_4b['borked_p']:.4f}")
    print(f"  oob:    R={r_4b['oob_r']:.4f}, P={r_4b['oob_p']:.4f}")
    delta = r_4b['f1'] - r_base['f1']
    print(f"  vs baseline: F1 {delta:+.4f}\n")

    # ===== 4b+4a combined =====
    print("=" * 60)
    print("4a+4b: No report_age_days + interaction")
    print("=" * 60)
    s1_4ab, X_tr_4ab, X_te_4ab = train_stage1_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 3.0, 1: 1.0},
        drop_features=["report_age_days"],
        add_interactions=True,
    )
    r_4ab = cascade_evaluate(s1_4ab, X_te_4ab, s2_model, s2_dropped, X_test, y_test)
    print(f"  Cascade: acc={r_4ab['acc']:.4f}, F1={r_4ab['f1']:.4f}")
    print(f"  borked: R={r_4ab['borked_r']:.4f}, P={r_4ab['borked_p']:.4f}")
    print(f"  oob:    R={r_4ab['oob_r']:.4f}, P={r_4ab['oob_p']:.4f}")
    delta = r_4ab['f1'] - r_base['f1']
    print(f"  vs baseline: F1 {delta:+.4f}\n")

    # ===== 4d: Class weight grid search =====
    print("=" * 60)
    print("4d: Class weight grid search (Stage 1)")
    print("=" * 60)

    best_config = None
    best_f1 = 0
    results_grid = []

    for w_borked in [2.0, 3.0, 4.0, 5.0, 6.0]:
        cw = {0: w_borked, 1: 1.0}
        s1_w, X_tr_w, X_te_w = train_stage1_variant(
            X_train, y_train, X_test, y_test, cat_cols,
            class_weight=cw,
        )
        r_w = cascade_evaluate(s1_w, X_te_w, s2_model, s2_dropped, X_test, y_test)
        results_grid.append((w_borked, r_w))
        print(f"  w={w_borked:.0f}: acc={r_w['acc']:.4f} F1={r_w['f1']:.4f} "
              f"borked R={r_w['borked_r']:.4f} P={r_w['borked_p']:.4f} "
              f"oob R={r_w['oob_r']:.4f} P={r_w['oob_p']:.4f}")

        if r_w['f1'] > best_f1:
            best_f1 = r_w['f1']
            best_config = (cw, s1_w, X_tr_w, X_te_w)

    print(f"\n  Best weight: {best_config[0]}, F1={best_f1:.4f}\n")

    # ===== 4c: Threshold tuning on best Stage 1 =====
    print("=" * 60)
    print("4c: Threshold tuning for P(borked)")
    print("=" * 60)

    # Use baseline Stage 1 for threshold sweep
    print(f"  {'Threshold':>10}  {'Acc':>7}  {'F1':>7}  {'borked R':>9}  {'borked P':>9}  {'oob R':>7}  {'oob P':>7}")
    best_thresh = 0.5
    best_thresh_f1 = 0
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        r_t = cascade_evaluate(s1_base, X_te_base, s2_model, s2_dropped, X_test, y_test, threshold=t)
        marker = ""
        if r_t['f1'] > best_thresh_f1:
            best_thresh_f1 = r_t['f1']
            best_thresh = t
            marker = " ←"
        print(f"  {t:>10.2f}  {r_t['acc']:>7.4f}  {r_t['f1']:>7.4f}  "
              f"{r_t['borked_r']:>9.4f}  {r_t['borked_p']:>9.4f}  "
              f"{r_t['oob_r']:>7.4f}  {r_t['oob_p']:>7.4f}{marker}")

    print(f"\n  Best threshold: {best_thresh}, F1={best_thresh_f1:.4f}\n")

    # ===== COMBINED BEST =====
    print("=" * 60)
    print("COMBINED: best weight + best threshold")
    print("=" * 60)

    # Try all combinations of best features
    variants = {
        "baseline": {"drop": [], "interactions": False},
        "no_age": {"drop": ["report_age_days"], "interactions": False},
        "interactions": {"drop": [], "interactions": True},
        "no_age+inter": {"drop": ["report_age_days"], "interactions": True},
    }

    for w_borked in [3.0, 4.0, 5.0]:
        for name, cfg in variants.items():
            s1_c, _, X_te_c = train_stage1_variant(
                X_train, y_train, X_test, y_test, cat_cols,
                class_weight={0: w_borked, 1: 1.0},
                drop_features=cfg["drop"],
                add_interactions=cfg["interactions"],
            )
            # Try thresholds around best
            for t in [0.4, 0.45, 0.5, 0.55]:
                r_c = cascade_evaluate(s1_c, X_te_c, s2_model, s2_dropped, X_test, y_test, threshold=t)
                print(f"  w={w_borked:.0f} {name:15s} t={t:.2f}: "
                      f"F1={r_c['f1']:.4f} borked R={r_c['borked_r']:.4f} P={r_c['borked_p']:.4f} "
                      f"oob R={r_c['oob_r']:.4f} P={r_c['oob_p']:.4f}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline:         F1={r_base['f1']:.4f}")
    print(f"  4a (no age):      F1={r_4a['f1']:.4f} ({r_4a['f1']-r_base['f1']:+.4f})")
    print(f"  4b (interaction): F1={r_4b['f1']:.4f} ({r_4b['f1']-r_base['f1']:+.4f})")
    print(f"  4a+4b:            F1={r_4ab['f1']:.4f} ({r_4ab['f1']-r_base['f1']:+.4f})")
    print(f"  4d (best weight): F1={best_f1:.4f} ({best_f1-r_base['f1']:+.4f})")
    print(f"  4c (best thresh): F1={best_thresh_f1:.4f} ({best_thresh_f1-r_base['f1']:+.4f})")


if __name__ == "__main__":
    main()
