#!/usr/bin/env python3
"""Experiment: two-stage cascade classifier (works/borked → tinkering/oob)."""

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
from protondb_settings.ml.models.classifier import TARGET_NAMES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
MODEL_PATH = Path("data/model.pkl")
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


def train_binary(X_train, y_train, X_test, y_test, cat_cols, class_weight=None,
                 n_estimators=2000, learning_rate=0.03):
    """Train a binary LightGBM classifier."""
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=63,
        learning_rate=learning_rate,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        importance_type="gain",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=500),
        ],
        categorical_feature=cat_cols,
    )
    return model


def evaluate_binary(model, X_test, y_test, class_names):
    """Evaluate binary model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"  Best iteration: {model.best_iteration_}")
    print()
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion matrix:")
    header = "          " + "  ".join(f"{n:>10}" for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i,j]:>10}" for j in range(len(class_names)))
        print(f"  {name:>8}  {row}")
    print()
    return y_pred, acc, f1


def shap_top(model, X_test, top_n=15):
    """Quick SHAP analysis."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test.iloc[:500])

    feature_names = list(X_test.columns)
    if isinstance(sv, list):
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    elif sv.ndim == 3:
        mean_shap = np.mean(np.abs(sv), axis=(0, 2))
    else:
        mean_shap = np.abs(sv).mean(axis=0)

    top_idx = np.argsort(mean_shap)[::-1][:top_n]
    print(f"  Top {top_n} features (SHAP):")
    for i in top_idx:
        print(f"    {feature_names[i]:30s} {mean_shap[i]:.4f}")
    print()


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, cat_cols = load_data()
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}\n")

    # ===== CURRENT SINGLE MODEL (for comparison) =====
    print("=" * 60)
    print("REFERENCE: Current 3-class model")
    print("=" * 60)
    model_3class = joblib.load(MODEL_PATH)
    y_pred_3 = model_3class.predict(X_test)
    acc_3 = accuracy_score(y_test, y_pred_3)
    f1_3 = f1_score(y_test, y_pred_3, average="macro", zero_division=0)
    print(f"  Accuracy: {acc_3:.4f}, F1 macro: {f1_3:.4f}")
    print(classification_report(y_test, y_pred_3, target_names=CLASS_NAMES_3, zero_division=0))

    # Collapsed: borked vs works
    y_test_binary = (y_test > 0).astype(int)  # 0=borked, 1=works
    y_pred_3_binary = (y_pred_3 > 0).astype(int)
    print("  3-class collapsed to binary (borked vs works):")
    print(f"    borked recall: {(y_pred_3_binary[y_test_binary == 0] == 0).mean():.4f}")
    print(f"    borked precision: {(y_test_binary[y_pred_3_binary == 0] == 0).mean():.4f}")
    print(f"    F1 borked: {f1_score(y_test_binary, y_pred_3_binary, pos_label=0):.4f}")
    print()

    # ===== STAGE 1: works vs borked =====
    print("=" * 60)
    print("STAGE 1: works vs borked")
    print("=" * 60)

    y_train_s1 = (y_train > 0).astype(int)  # 0=borked, 1=works
    y_test_s1 = (y_test > 0).astype(int)
    print(f"  Train: borked={((y_train_s1==0).sum())}, works={((y_train_s1==1).sum())}")
    print(f"  Test:  borked={((y_test_s1==0).sum())}, works={((y_test_s1==1).sum())}")
    print()

    # Try different class weights for Stage 1
    best_s1_model = None
    best_s1_f1 = 0
    best_s1_weight = None

    for weight_borked in [2.0, 3.0, 4.0, 5.0]:
        cw = {0: weight_borked, 1: 1.0}
        print(f"  --- class_weight = {cw} ---")
        model_s1 = train_binary(X_train, y_train_s1, X_test, y_test_s1, cat_cols, class_weight=cw)
        y_pred_s1, acc_s1, f1_s1 = evaluate_binary(model_s1, X_test, y_test_s1, ["borked", "works"])

        borked_recall = (y_pred_s1[y_test_s1 == 0] == 0).mean()
        borked_precision = (y_test_s1[y_pred_s1 == 0] == 0).mean() if (y_pred_s1 == 0).any() else 0
        print(f"  borked recall={borked_recall:.4f}, precision={borked_precision:.4f}")
        print()

        if f1_s1 > best_s1_f1:
            best_s1_f1 = f1_s1
            best_s1_model = model_s1
            best_s1_weight = cw

    print(f"  Best Stage 1: class_weight={best_s1_weight}, F1={best_s1_f1:.4f}")
    print()

    print("  SHAP (best Stage 1):")
    shap_top(best_s1_model, X_test)

    # ===== STAGE 2: tinkering vs works_oob =====
    print("=" * 60)
    print("STAGE 2: tinkering vs works_oob (non-borked only)")
    print("=" * 60)

    # Filter to non-borked
    train_works_mask = y_train > 0
    test_works_mask = y_test > 0

    X_train_s2 = X_train[train_works_mask].reset_index(drop=True)
    y_train_s2 = (y_train[train_works_mask] - 1).astype(int)  # 0=tinkering, 1=oob

    X_test_s2 = X_test[test_works_mask].reset_index(drop=True)
    y_test_s2 = (y_test[test_works_mask] - 1).astype(int)

    # Re-apply category dtype
    for col in cat_cols:
        if col in X_train_s2.columns:
            X_train_s2[col] = X_train_s2[col].astype("category")
            X_test_s2[col] = X_test_s2[col].astype("category")

    print(f"  Train: tinkering={((y_train_s2==0).sum())}, oob={((y_train_s2==1).sum())}")
    print(f"  Test:  tinkering={((y_test_s2==0).sum())}, oob={((y_test_s2==1).sum())}")
    print()

    best_s2_model = None
    best_s2_f1 = 0
    best_s2_weight = None

    for weight_oob in [1.5, 2.0, 2.5, 3.0]:
        cw = {0: 1.0, 1: weight_oob}
        print(f"  --- class_weight = {cw} ---")
        model_s2 = train_binary(X_train_s2, y_train_s2, X_test_s2, y_test_s2, cat_cols, class_weight=cw)
        y_pred_s2, acc_s2, f1_s2 = evaluate_binary(model_s2, X_test_s2, y_test_s2, ["tinkering", "works_oob"])

        if f1_s2 > best_s2_f1:
            best_s2_f1 = f1_s2
            best_s2_model = model_s2
            best_s2_weight = cw

    print(f"  Best Stage 2: class_weight={best_s2_weight}, F1={best_s2_f1:.4f}")
    print()

    print("  SHAP (best Stage 2):")
    shap_top(best_s2_model, X_test_s2)

    # ===== CASCADE: combine Stage 1 + Stage 2 =====
    print("=" * 60)
    print("CASCADE: Stage 1 + Stage 2 combined")
    print("=" * 60)

    # Stage 1 predictions on full test
    y_pred_s1_full = best_s1_model.predict(X_test)
    p_s1 = best_s1_model.predict_proba(X_test)

    # Stage 2 predictions on ALL test (even borked, for combining)
    # Need to re-apply category for full X_test
    p_s2 = best_s2_model.predict_proba(X_test)

    # Cascade: if stage1 says borked → borked, else stage2 decides
    y_cascade = np.full(len(X_test), -1, dtype=int)
    borked_mask = y_pred_s1_full == 0
    y_cascade[borked_mask] = 0  # borked

    works_mask = ~borked_mask
    y_pred_s2_on_works = best_s2_model.predict(X_test[works_mask])
    y_cascade[works_mask] = np.where(y_pred_s2_on_works == 1, 2, 1)  # 2=oob, 1=tinkering

    # Evaluate cascade as 3-class
    acc_cascade = accuracy_score(y_test, y_cascade)
    f1_cascade = f1_score(y_test, y_cascade, average="macro", zero_division=0)

    print(f"  Cascade Accuracy: {acc_cascade:.4f} (single: {acc_3:.4f})")
    print(f"  Cascade F1 macro: {f1_cascade:.4f} (single: {f1_3:.4f})")
    print()
    print(classification_report(y_test, y_cascade, target_names=CLASS_NAMES_3, zero_division=0))

    cm = confusion_matrix(y_test, y_cascade, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("  Confusion matrix (normalized):")
    header = "              " + "  ".join(f"{n:>12}" for n in CLASS_NAMES_3)
    print(header)
    for i, name in enumerate(CLASS_NAMES_3):
        row = "  ".join(f"{cm_norm[i,j]:>11.1%}" for j in range(3))
        print(f"  {name:>12}  {row}")
    print()

    # Combined probabilities
    proba_cascade = np.zeros((len(X_test), 3))
    proba_cascade[:, 0] = p_s1[:, 0]           # P(borked) from stage1 (class 0)
    proba_cascade[:, 1] = p_s1[:, 1] * p_s2[:, 0]  # P(works) × P(tinkering|works)
    proba_cascade[:, 2] = p_s1[:, 1] * p_s2[:, 1]  # P(works) × P(oob|works)

    # Threshold sweep for Stage 1
    print("  --- Stage 1 threshold sweep ---")
    print(f"  {'Threshold':>10}  {'Acc':>7}  {'F1 macro':>9}  {'borked R':>9}  {'borked P':>9}  {'oob R':>7}  {'oob P':>7}")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_t = np.full(len(X_test), -1, dtype=int)
        borked_t = p_s1[:, 0] >= threshold  # P(borked) >= threshold
        y_t[borked_t] = 0
        works_t = ~borked_t
        y_pred_s2_t = best_s2_model.predict(X_test[works_t])
        y_t[works_t] = np.where(y_pred_s2_t == 1, 2, 1)

        acc_t = accuracy_score(y_test, y_t)
        f1_t = f1_score(y_test, y_t, average="macro", zero_division=0)
        br = (y_t[y_test == 0] == 0).mean()
        bp = (y_test[y_t == 0] == 0).mean() if (y_t == 0).any() else 0
        oob_r = (y_t[y_test == 2] == 2).mean()
        oob_p = (y_test[y_t == 2] == 2).mean() if (y_t == 2).any() else 0
        print(f"  {threshold:>10.1f}  {acc_t:>7.4f}  {f1_t:>9.4f}  {br:>9.4f}  {bp:>9.4f}  {oob_r:>7.4f}  {oob_p:>7.4f}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Single 3-class: Accuracy={acc_3:.4f}, F1 macro={f1_3:.4f}")
    print(f"  Cascade:        Accuracy={acc_cascade:.4f}, F1 macro={f1_cascade:.4f}")
    delta_f1 = f1_cascade - f1_3
    print(f"  Delta F1: {delta_f1:+.4f}")


if __name__ == "__main__":
    main()
