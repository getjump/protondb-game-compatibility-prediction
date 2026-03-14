#!/usr/bin/env python3
"""Experiment v2: cascade with report_age_days alternatives in Stage 2."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
MODEL_PATH = Path("data/model.pkl")
EMB_PATH = Path("data/embeddings.npz")

CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]


def load_data():
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


def train_binary(X_train, y_train, X_test, y_test, cat_cols, class_weight=None):
    model = lgb.LGBMClassifier(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight=class_weight, n_jobs=-1, random_state=42,
        verbose=-1, importance_type="gain",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=500)],
        categorical_feature=cat_cols,
    )
    return model


def evaluate_3class(y_test, y_cascade, label=""):
    acc = accuracy_score(y_test, y_cascade)
    f1 = f1_score(y_test, y_cascade, average="macro", zero_division=0)
    print(f"  {label} Accuracy: {acc:.4f}, F1 macro: {f1:.4f}")
    print(classification_report(y_test, y_cascade, target_names=CLASS_NAMES_3, zero_division=0))

    cm = confusion_matrix(y_test, y_cascade, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("  Confusion matrix (normalized):")
    for i, name in enumerate(CLASS_NAMES_3):
        row = "  ".join(f"{cm_norm[i,j]:>7.1%}" for j in range(3))
        print(f"    {name:>15}  {row}")
    print()
    return acc, f1


def shap_top(model, X, top_n=10):
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.iloc[:500])
    feature_names = list(X.columns)
    if isinstance(sv, list):
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    elif sv.ndim == 3:
        mean_shap = np.mean(np.abs(sv), axis=(0, 2))
    else:
        mean_shap = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_shap)[::-1][:top_n]
    print(f"  Top {top_n} SHAP:")
    for i in top_idx:
        print(f"    {feature_names[i]:30s} {mean_shap[i]:.4f}")
    print()


def make_proton_era(X):
    """Create proton_era from proton_major: discretize into eras."""
    pm = X["proton_major"].copy() if "proton_major" in X.columns else pd.Series(np.nan, index=X.index)
    # Era: 0=unknown, 1=<=4, 2=5-6, 3=7, 4=8, 5=9+
    era = pd.Series(0, index=X.index, dtype="int8")
    era[pm <= 4] = 1
    era[(pm >= 5) & (pm <= 6)] = 2
    era[pm == 7] = 3
    era[pm == 8] = 4
    era[pm >= 9] = 5
    era[pm.isna()] = 0
    return era


def make_report_age_relative(X_full, y_full, timestamps_not_available=True):
    """report_age_days normalized per-game. Approximation: use report_age_days quantiles."""
    # We don't have per-game max/min easily, so use quantile binning instead
    rad = X_full["report_age_days"].copy()
    # Bin into deciles
    bins = pd.qcut(rad, q=10, labels=False, duplicates="drop")
    return bins


def cascade_evaluate(s1_model, s2_model, X_test_s1, X_test_s2, y_test, threshold=0.5):
    """Run cascade and evaluate. s1 and s2 may use different feature sets."""
    p_s1 = s1_model.predict_proba(X_test_s1)
    borked_mask = p_s1[:, 0] >= threshold

    y_cascade = np.full(len(X_test_s1), -1, dtype=int)
    y_cascade[borked_mask] = 0

    works_mask = ~borked_mask
    if works_mask.any():
        y_pred_s2 = s2_model.predict(X_test_s2[works_mask])
        y_cascade[works_mask] = np.where(y_pred_s2 == 1, 2, 1)

    return y_cascade


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, cat_cols = load_data()
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}\n")

    # ===== REFERENCE =====
    print("=" * 60)
    print("REFERENCE: Single 3-class model")
    print("=" * 60)
    model_3 = joblib.load(MODEL_PATH)
    y_pred_3 = model_3.predict(X_test)
    ref_acc, ref_f1 = evaluate_3class(y_test, y_pred_3, "Single")

    # ===== STAGE 1 (same for all variants) =====
    print("=" * 60)
    print("STAGE 1: works vs borked (class_weight={0:3, 1:1})")
    print("=" * 60)
    y_train_s1 = (y_train > 0).astype(int)
    y_test_s1 = (y_test > 0).astype(int)
    s1_model = train_binary(X_train, y_train_s1, X_test, y_test_s1, cat_cols, class_weight={0: 3.0, 1: 1.0})
    y_pred_s1 = s1_model.predict(X_test)
    br = (y_pred_s1[y_test_s1 == 0] == 0).mean()
    bp = (y_test_s1[y_pred_s1 == 0] == 0).mean()
    print(f"  borked recall={br:.4f}, precision={bp:.4f}, best_iter={s1_model.best_iteration_}\n")

    # ===== STAGE 2 variants =====
    train_works = y_train > 0
    test_works = y_test > 0

    X_train_s2 = X_train[train_works].reset_index(drop=True)
    y_train_s2 = (y_train[train_works] - 1).astype(int)
    X_test_s2 = X_test[test_works].reset_index(drop=True)
    y_test_s2 = (y_test[test_works] - 1).astype(int)

    for col in cat_cols:
        if col in X_train_s2.columns:
            X_train_s2[col] = X_train_s2[col].astype("category")
            X_test_s2[col] = X_test_s2[col].astype("category")

    variants = {}

    # --- Variant A: baseline (with report_age_days) ---
    print("=" * 60)
    print("STAGE 2A: baseline (with report_age_days)")
    print("=" * 60)
    s2a = train_binary(X_train_s2, y_train_s2, X_test_s2, y_test_s2, cat_cols, class_weight={0: 1.0, 1: 2.0})
    y_pred_s2a = s2a.predict(X_test_s2)
    f1_s2a = f1_score(y_test_s2, y_pred_s2a, average="macro")
    print(f"  Stage 2 F1 macro: {f1_s2a:.4f}, best_iter: {s2a.best_iteration_}")
    print(classification_report(y_test_s2, y_pred_s2a, target_names=["tinkering", "works_oob"], zero_division=0))
    shap_top(s2a, X_test_s2)
    y_cascade_a = cascade_evaluate(s1_model, s2a, X_test, X_test, y_test)
    acc_a, f1_a = evaluate_3class(y_test, y_cascade_a, "Cascade A (baseline)")
    variants["A: baseline"] = f1_a

    # --- Variant B: without report_age_days ---
    print("=" * 60)
    print("STAGE 2B: without report_age_days")
    print("=" * 60)
    drop_col = "report_age_days"
    X_train_s2b = X_train_s2.drop(columns=[drop_col])
    X_test_s2b = X_test_s2.drop(columns=[drop_col])
    cat_cols_b = [c for c in cat_cols if c != drop_col]
    s2b = train_binary(X_train_s2b, y_train_s2, X_test_s2b, y_test_s2, cat_cols_b, class_weight={0: 1.0, 1: 2.0})
    y_pred_s2b = s2b.predict(X_test_s2b)
    f1_s2b = f1_score(y_test_s2, y_pred_s2b, average="macro")
    print(f"  Stage 2 F1 macro: {f1_s2b:.4f}, best_iter: {s2b.best_iteration_}")
    print(classification_report(y_test_s2, y_pred_s2b, target_names=["tinkering", "works_oob"], zero_division=0))
    shap_top(s2b, X_test_s2b)
    # For cascade: s1 gets full X_test, s2 gets X_test without report_age_days
    X_test_no_rad = X_test.drop(columns=[drop_col])
    y_cascade_b = cascade_evaluate(s1_model, s2b, X_test, X_test_no_rad, y_test)
    acc_b, f1_b = evaluate_3class(y_test, y_cascade_b, "Cascade B (no report_age_days)")
    variants["B: no report_age_days"] = f1_b

    # --- Variant C: replace report_age_days with proton_era ---
    print("=" * 60)
    print("STAGE 2C: proton_era instead of report_age_days")
    print("=" * 60)
    X_train_s2c = X_train_s2.drop(columns=[drop_col]).copy()
    X_test_s2c = X_test_s2.drop(columns=[drop_col]).copy()
    X_train_s2c["proton_era"] = make_proton_era(X_train_s2).values
    X_test_s2c["proton_era"] = make_proton_era(X_test_s2).values
    s2c = train_binary(X_train_s2c, y_train_s2, X_test_s2c, y_test_s2, cat_cols_b, class_weight={0: 1.0, 1: 2.0})
    y_pred_s2c = s2c.predict(X_test_s2c)
    f1_s2c = f1_score(y_test_s2, y_pred_s2c, average="macro")
    print(f"  Stage 2 F1 macro: {f1_s2c:.4f}, best_iter: {s2c.best_iteration_}")
    print(classification_report(y_test_s2, y_pred_s2c, target_names=["tinkering", "works_oob"], zero_division=0))
    shap_top(s2c, X_test_s2c)
    X_test_c = X_test.drop(columns=[drop_col]).copy()
    X_test_c["proton_era"] = make_proton_era(X_test).values
    y_cascade_c = cascade_evaluate(s1_model, s2c, X_test, X_test_c, y_test)
    acc_c, f1_c = evaluate_3class(y_test, y_cascade_c, "Cascade C (proton_era)")
    variants["C: proton_era"] = f1_c

    # --- Variant D: report_age_days capped at 365 ---
    print("=" * 60)
    print("STAGE 2D: report_age_days capped at 365")
    print("=" * 60)
    X_train_s2d = X_train_s2.copy()
    X_test_s2d = X_test_s2.copy()
    X_train_s2d["report_age_days"] = X_train_s2d["report_age_days"].clip(upper=365)
    X_test_s2d["report_age_days"] = X_test_s2d["report_age_days"].clip(upper=365)
    s2d = train_binary(X_train_s2d, y_train_s2, X_test_s2d, y_test_s2, cat_cols, class_weight={0: 1.0, 1: 2.0})
    y_pred_s2d = s2d.predict(X_test_s2d)
    f1_s2d = f1_score(y_test_s2, y_pred_s2d, average="macro")
    print(f"  Stage 2 F1 macro: {f1_s2d:.4f}, best_iter: {s2d.best_iteration_}")
    print(classification_report(y_test_s2, y_pred_s2d, target_names=["tinkering", "works_oob"], zero_division=0))
    shap_top(s2d, X_test_s2d)
    X_test_d = X_test.copy()
    X_test_d["report_age_days"] = X_test_d["report_age_days"].clip(upper=365)
    y_cascade_d = cascade_evaluate(s1_model, s2d, X_test, X_test_d, y_test)
    acc_d, f1_d = evaluate_3class(y_test, y_cascade_d, "Cascade D (capped 365)")
    variants["D: capped 365"] = f1_d

    # --- Variant E: report_age_days capped + proton_era ---
    print("=" * 60)
    print("STAGE 2E: capped 365 + proton_era")
    print("=" * 60)
    X_train_s2e = X_train_s2.copy()
    X_test_s2e = X_test_s2.copy()
    X_train_s2e["report_age_days"] = X_train_s2e["report_age_days"].clip(upper=365)
    X_test_s2e["report_age_days"] = X_test_s2e["report_age_days"].clip(upper=365)
    X_train_s2e["proton_era"] = make_proton_era(X_train_s2).values
    X_test_s2e["proton_era"] = make_proton_era(X_test_s2).values
    s2e = train_binary(X_train_s2e, y_train_s2, X_test_s2e, y_test_s2, cat_cols, class_weight={0: 1.0, 1: 2.0})
    y_pred_s2e = s2e.predict(X_test_s2e)
    f1_s2e = f1_score(y_test_s2, y_pred_s2e, average="macro")
    print(f"  Stage 2 F1 macro: {f1_s2e:.4f}, best_iter: {s2e.best_iteration_}")
    print(classification_report(y_test_s2, y_pred_s2e, target_names=["tinkering", "works_oob"], zero_division=0))
    shap_top(s2e, X_test_s2e)
    X_test_e = X_test.copy()
    X_test_e["report_age_days"] = X_test_e["report_age_days"].clip(upper=365)
    X_test_e["proton_era"] = make_proton_era(X_test).values
    y_cascade_e = cascade_evaluate(s1_model, s2e, X_test, X_test_e, y_test)
    acc_e, f1_e = evaluate_3class(y_test, y_cascade_e, "Cascade E (capped+era)")
    variants["E: capped+era"] = f1_e

    # ===== SUMMARY =====
    print("=" * 60)
    print("SUMMARY: Cascade F1 macro by Stage 2 variant")
    print("=" * 60)
    print(f"  {'Single 3-class':>25s}  {ref_f1:.4f}")
    for name, f1 in sorted(variants.items(), key=lambda x: -x[1]):
        delta = f1 - ref_f1
        print(f"  {name:>25s}  {f1:.4f}  ({delta:+.4f})")


if __name__ == "__main__":
    main()
