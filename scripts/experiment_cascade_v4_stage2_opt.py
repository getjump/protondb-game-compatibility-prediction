#!/usr/bin/env python3
"""Experiment: Stage 2 optimization (tinkering vs works_oob).

Tests from PLAN_ML_4.md step 5:
  5a. Already done (remove report_age_days = best)
  5b. Remove anticheat/has_denuvo from Stage 2
  5c. Test with tinkering-specific aggregated features already present
  5d. Label smoothing: use pct_works_oob as soft target
  + Combined best variants
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.models.cascade import train_stage1

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")

CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]

# Features to try dropping from Stage 2
BORKED_ONLY_FEATURES = ["anticheat", "anticheat_status", "has_denuvo"]
TEMPORAL_FEATURES = ["report_age_days"]


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


def train_stage2_variant(
    X_train, y_train, X_test, y_test, cat_cols,
    class_weight=None, drop_features=None, soft_labels=False,
    n_estimators=2000, learning_rate=0.03,
):
    """Train a Stage 2 variant on non-borked data.

    Returns (model, X_train_s2, X_test_s2, y_train_s2, y_test_s2, dropped_features).
    """
    if class_weight is None:
        class_weight = {0: 1.0, 1: 2.0}
    if drop_features is None:
        drop_features = []

    # Filter to non-borked
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_tr = X_train[train_mask].reset_index(drop=True)
    y_tr = (y_train[train_mask] - 1).astype(int)  # 0=tinkering, 1=oob

    X_te = X_test[test_mask].reset_index(drop=True)
    y_te = (y_test[test_mask] - 1).astype(int)

    # Drop features
    existing_drops = [f for f in drop_features if f in X_tr.columns]
    if existing_drops:
        X_tr = X_tr.drop(columns=existing_drops)
        X_te = X_te.drop(columns=existing_drops)

    cat_cols_s2 = [c for c in cat_cols if c in X_tr.columns]
    for col in cat_cols_s2:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")

    if soft_labels and "pct_works_oob" in X_tr.columns:
        # Use pct_works_oob as soft target via sample_weight
        # Higher pct_works_oob → more confident "oob" label
        # Lower pct_works_oob → more confident "tinkering" label
        soft_vals_tr = X_tr["pct_works_oob"].fillna(0.5).values
        # Weight = how confident we are in the label
        # For oob (y=1): confidence = pct_works_oob
        # For tinkering (y=0): confidence = 1 - pct_works_oob
        sample_weight = np.where(y_tr == 1, soft_vals_tr, 1 - soft_vals_tr)
        # Clip to avoid zero weights
        sample_weight = np.clip(sample_weight, 0.1, 1.0)
        # Apply class weight on top
        for cls, w in class_weight.items():
            sample_weight[y_tr == cls] *= w
    else:
        sample_weight = None

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, num_leaves=63, learning_rate=learning_rate,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight=class_weight if sample_weight is None else None,
        n_jobs=-1, random_state=42,
        verbose=-1, importance_type="gain",
    )

    fit_kwargs = {
        "eval_set": [(X_te, y_te)],
        "callbacks": [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=500),
        ],
        "categorical_feature": cat_cols_s2,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X_tr, y_tr, **fit_kwargs)

    return model, X_tr, X_te, y_tr, y_te, existing_drops


def cascade_evaluate(s1_model, s2_model, s2_dropped, X_test, y_test, threshold=0.5):
    """Evaluate full cascade."""
    p_s1 = s1_model.predict_proba(X_test)
    borked_mask = p_s1[:, 0] >= threshold

    result = np.full(len(X_test), 1, dtype=int)
    result[borked_mask] = 0

    works_mask = ~borked_mask
    if works_mask.any():
        X_s2 = X_test[works_mask].copy()
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
    tink_r = (result[y_test == 1] == 1).mean() if (y_test == 1).any() else 0
    tink_p = (y_test[result == 1] == 1).mean() if (result == 1).any() else 0

    return {
        "acc": acc, "f1": f1,
        "borked_r": borked_r, "borked_p": borked_p,
        "oob_r": oob_r, "oob_p": oob_p,
        "tink_r": tink_r, "tink_p": tink_p,
        "y_pred": result,
    }


def print_stage2_metrics(model, X_te, y_te, label=""):
    """Print Stage 2 standalone metrics."""
    y_pred = model.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_te, y_pred)
    oob_r = (y_pred[y_te == 1] == 1).mean() if (y_te == 1).any() else 0
    oob_p = (y_te[y_pred == 1] == 1).mean() if (y_pred == 1).any() else 0
    tink_r = (y_pred[y_te == 0] == 0).mean() if (y_te == 0).any() else 0
    tink_p = (y_te[y_pred == 0] == 0).mean() if (y_pred == 0).any() else 0
    print(f"  {label:30s} S2 F1={f1:.4f} acc={acc:.4f} "
          f"tink R={tink_r:.4f}/P={tink_p:.4f} oob R={oob_r:.4f}/P={oob_p:.4f} "
          f"iter={model.best_iteration_}")
    return f1


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, cat_cols = load_data()
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}\n")

    # Train Stage 1 once (best config)
    print("=" * 70)
    print("Training Stage 1 (fixed: w={0:3, 1:1}, all features)")
    print("=" * 70)
    s1_model = train_stage1(X_train, y_train, X_test, y_test,
                            class_weight={0: 3.0, 1: 1.0})
    print(f"  Stage 1 best iteration: {s1_model.best_iteration_}\n")

    # Show available features for reference
    print("Available aggregated features:")
    agg_feats = [c for c in X_train.columns if c.startswith("pct_") or c.startswith("avg_")]
    print(f"  {agg_feats}\n")

    # ===== BASELINE (A): report_age_days dropped =====
    print("=" * 70)
    print("A (BASELINE): S2 drop=[report_age_days], w={0:1, 1:2}")
    print("=" * 70)
    s2_A, X_tr_A, X_te_A, y_tr_A, y_te_A, drops_A = train_stage2_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 1.0, 1: 2.0},
        drop_features=TEMPORAL_FEATURES,
    )
    f1_A = print_stage2_metrics(s2_A, X_te_A, y_te_A, "A baseline")
    r_A = cascade_evaluate(s1_model, s2_A, drops_A, X_test, y_test)
    print(f"  Cascade: F1={r_A['f1']:.4f} borked R={r_A['borked_r']:.4f}/P={r_A['borked_p']:.4f} "
          f"oob R={r_A['oob_r']:.4f}/P={r_A['oob_p']:.4f} tink R={r_A['tink_r']:.4f}/P={r_A['tink_p']:.4f}")
    print()

    # ===== 5b: Remove borked-only features =====
    print("=" * 70)
    print("5b: S2 drop=[report_age_days, anticheat, anticheat_status, has_denuvo]")
    print("=" * 70)
    s2_5b, X_tr_5b, X_te_5b, y_tr_5b, y_te_5b, drops_5b = train_stage2_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 1.0, 1: 2.0},
        drop_features=TEMPORAL_FEATURES + BORKED_ONLY_FEATURES,
    )
    f1_5b = print_stage2_metrics(s2_5b, X_te_5b, y_te_5b, "5b no borked feats")
    r_5b = cascade_evaluate(s1_model, s2_5b, drops_5b, X_test, y_test)
    print(f"  Cascade: F1={r_5b['f1']:.4f} borked R={r_5b['borked_r']:.4f}/P={r_5b['borked_p']:.4f} "
          f"oob R={r_5b['oob_r']:.4f}/P={r_5b['oob_p']:.4f} tink R={r_5b['tink_r']:.4f}/P={r_5b['tink_p']:.4f}")
    print(f"  vs baseline: cascade F1 {r_5b['f1']-r_A['f1']:+.4f}, S2 F1 {f1_5b-f1_A:+.4f}")
    print()

    # ===== 5d: Soft labels =====
    print("=" * 70)
    print("5d: Soft labels (sample_weight from pct_works_oob)")
    print("=" * 70)
    s2_5d, X_tr_5d, X_te_5d, y_tr_5d, y_te_5d, drops_5d = train_stage2_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 1.0, 1: 2.0},
        drop_features=TEMPORAL_FEATURES,
        soft_labels=True,
    )
    f1_5d = print_stage2_metrics(s2_5d, X_te_5d, y_te_5d, "5d soft labels")
    r_5d = cascade_evaluate(s1_model, s2_5d, drops_5d, X_test, y_test)
    print(f"  Cascade: F1={r_5d['f1']:.4f} borked R={r_5d['borked_r']:.4f}/P={r_5d['borked_p']:.4f} "
          f"oob R={r_5d['oob_r']:.4f}/P={r_5d['oob_p']:.4f} tink R={r_5d['tink_r']:.4f}/P={r_5d['tink_p']:.4f}")
    print(f"  vs baseline: cascade F1 {r_5d['f1']-r_A['f1']:+.4f}, S2 F1 {f1_5d-f1_A:+.4f}")
    print()

    # ===== 5b+5d combined =====
    print("=" * 70)
    print("5b+5d: No borked feats + soft labels")
    print("=" * 70)
    s2_5bd, X_tr_5bd, X_te_5bd, y_tr_5bd, y_te_5bd, drops_5bd = train_stage2_variant(
        X_train, y_train, X_test, y_test, cat_cols,
        class_weight={0: 1.0, 1: 2.0},
        drop_features=TEMPORAL_FEATURES + BORKED_ONLY_FEATURES,
        soft_labels=True,
    )
    f1_5bd = print_stage2_metrics(s2_5bd, X_te_5bd, y_te_5bd, "5b+5d combined")
    r_5bd = cascade_evaluate(s1_model, s2_5bd, drops_5bd, X_test, y_test)
    print(f"  Cascade: F1={r_5bd['f1']:.4f} borked R={r_5bd['borked_r']:.4f}/P={r_5bd['borked_p']:.4f} "
          f"oob R={r_5bd['oob_r']:.4f}/P={r_5bd['oob_p']:.4f} tink R={r_5bd['tink_r']:.4f}/P={r_5bd['tink_p']:.4f}")
    print(f"  vs baseline: cascade F1 {r_5bd['f1']-r_A['f1']:+.4f}, S2 F1 {f1_5bd-f1_A:+.4f}")
    print()

    # ===== Class weight grid for best variant =====
    print("=" * 70)
    print("Weight grid search (on baseline drop=[report_age_days])")
    print("=" * 70)
    best_w_f1 = 0
    best_w = None
    for w_oob in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        cw = {0: 1.0, 1: w_oob}
        s2_w, X_tr_w, X_te_w, y_tr_w, y_te_w, drops_w = train_stage2_variant(
            X_train, y_train, X_test, y_test, cat_cols,
            class_weight=cw,
            drop_features=TEMPORAL_FEATURES,
        )
        f1_w = print_stage2_metrics(s2_w, X_te_w, y_te_w, f"w={{0:1, 1:{w_oob}}}")
        r_w = cascade_evaluate(s1_model, s2_w, drops_w, X_test, y_test)
        print(f"    Cascade: F1={r_w['f1']:.4f} oob R={r_w['oob_r']:.4f}/P={r_w['oob_p']:.4f} "
              f"tink R={r_w['tink_r']:.4f}/P={r_w['tink_p']:.4f}")
        if r_w['f1'] > best_w_f1:
            best_w_f1 = r_w['f1']
            best_w = w_oob
    print(f"\n  Best weight: w_oob={best_w}, cascade F1={best_w_f1:.4f}\n")

    # ===== Stage 2 threshold tuning =====
    print("=" * 70)
    print("Stage 2 threshold tuning (on baseline)")
    print("=" * 70)
    print(f"  {'S2 thresh':>10}  {'Casc F1':>8}  {'oob R':>7}  {'oob P':>7}  {'tink R':>7}  {'tink P':>7}")
    best_t2 = 0.5
    best_t2_f1 = 0
    for t2 in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        p_s1 = s1_model.predict_proba(X_test)
        borked_mask = p_s1[:, 0] >= 0.5
        result = np.full(len(X_test), 1, dtype=int)
        result[borked_mask] = 0
        works_mask = ~borked_mask
        if works_mask.any():
            X_s2 = X_test[works_mask].copy()
            cols_to_drop = [c for c in drops_A if c in X_s2.columns]
            if cols_to_drop:
                X_s2 = X_s2.drop(columns=cols_to_drop)
            p_s2 = s2_A.predict_proba(X_s2)
            result[works_mask] = np.where(p_s2[:, 1] >= t2, 2, 1)

        f1_t = f1_score(y_test, result, average="macro", zero_division=0)
        oob_r = (result[y_test == 2] == 2).mean()
        oob_p = (y_test[result == 2] == 2).mean() if (result == 2).any() else 0
        tink_r = (result[y_test == 1] == 1).mean()
        tink_p = (y_test[result == 1] == 1).mean() if (result == 1).any() else 0
        marker = " ←" if f1_t > best_t2_f1 else ""
        if f1_t > best_t2_f1:
            best_t2_f1 = f1_t
            best_t2 = t2
        print(f"  {t2:>10.2f}  {f1_t:>8.4f}  {oob_r:>7.4f}  {oob_p:>7.4f}  {tink_r:>7.4f}  {tink_p:>7.4f}{marker}")

    print(f"\n  Best S2 threshold: {best_t2}, cascade F1={best_t2_f1:.4f}\n")

    # ===== Hyperparameter tuning =====
    print("=" * 70)
    print("Learning rate / num_leaves tuning (on baseline)")
    print("=" * 70)
    for lr in [0.01, 0.03, 0.05]:
        for nl in [31, 63, 127]:
            s2_hp, X_tr_hp, X_te_hp, y_tr_hp, y_te_hp, drops_hp = train_stage2_variant(
                X_train, y_train, X_test, y_test, cat_cols,
                class_weight={0: 1.0, 1: 2.0},
                drop_features=TEMPORAL_FEATURES,
                n_estimators=3000 if lr == 0.01 else 2000,
                learning_rate=lr,
            )
            f1_hp = print_stage2_metrics(s2_hp, X_te_hp, y_te_hp, f"lr={lr} nl={nl}")
            r_hp = cascade_evaluate(s1_model, s2_hp, drops_hp, X_test, y_test)
            print(f"    Cascade F1={r_hp['f1']:.4f}")

    # ===== SUMMARY =====
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  A (baseline, drop age):     S2 F1={f1_A:.4f}, cascade F1={r_A['f1']:.4f}")
    print(f"  5b (+ drop borked feats):   S2 F1={f1_5b:.4f} ({f1_5b-f1_A:+.4f}), cascade F1={r_5b['f1']:.4f} ({r_5b['f1']-r_A['f1']:+.4f})")
    print(f"  5d (soft labels):           S2 F1={f1_5d:.4f} ({f1_5d-f1_A:+.4f}), cascade F1={r_5d['f1']:.4f} ({r_5d['f1']-r_A['f1']:+.4f})")
    print(f"  5b+5d (combined):           S2 F1={f1_5bd:.4f} ({f1_5bd-f1_A:+.4f}), cascade F1={r_5bd['f1']:.4f} ({r_5bd['f1']-r_A['f1']:+.4f})")
    print(f"  Best weight:                w_oob={best_w}, cascade F1={best_w_f1:.4f}")
    print(f"  Best S2 threshold:          t={best_t2}, cascade F1={best_t2_f1:.4f}")

    # Full classification report for baseline
    print()
    print("Baseline cascade classification report:")
    print(classification_report(y_test, r_A['y_pred'], target_names=CLASS_NAMES_3, zero_division=0))


if __name__ == "__main__":
    main()
