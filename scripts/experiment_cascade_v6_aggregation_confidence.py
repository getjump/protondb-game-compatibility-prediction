#!/usr/bin/env python3
"""Experiment: Step 11 (aggregation by game+hardware) + Step 6 (confidence) + Step 7 (calibration).

Step 11: Aggregate predictions per (game, hardware_config)
Step 6: Confidence-aware output — measure uncertain fraction
Step 7: Post-hoc calibration (isotonic regression)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]
STAGE2_DROP = ["report_age_days"]


def load_data_with_groups():
    """Load data with group keys for aggregation."""
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

    # Get app_id for grouping (ordered by timestamp to match feature matrix)
    app_ids = pd.read_sql("SELECT app_id FROM reports ORDER BY timestamp ASC", conn)["app_id"].values
    # Use feature matrix columns for grouping
    group_cols = []
    for c in ["gpu_family", "cpu_vendor", "is_steam_deck"]:
        if c in X.columns:
            group_cols.append(c)
    group_df = X[group_cols].copy()
    group_df["app_id"] = app_ids[:len(X)]

    X_train, X_test, y_train, y_test = _time_based_split(X, y, timestamps, 0.2)
    split_idx = len(X_train)
    groups_test = group_df.iloc[split_idx:].reset_index(drop=True)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_test.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    conn.close()
    return X_train, X_test, y_train, y_test, cat_cols, groups_test


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, cat_cols, groups_test = load_data_with_groups()
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}\n")

    # Train cascade
    print("=" * 70)
    print("Training cascade (baseline)")
    print("=" * 70)
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_drops)

    y_pred = cascade.predict(X_test)
    y_proba = cascade.predict_proba(X_test)
    f1_base = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Cascade F1={f1_base:.4f}\n")

    # ===== STEP 11: Aggregation =====
    print("=" * 70)
    print("STEP 11: Aggregation by (game, gpu_family, cpu_vendor, is_deck)")
    print("=" * 70)

    # Create group key
    key_parts = [groups_test["app_id"].astype(str)]
    for c in ["gpu_family", "cpu_vendor", "is_steam_deck"]:
        if c in groups_test.columns:
            key_parts.append(groups_test[c].fillna("unk").astype(str))
    groups_test["group_key"] = key_parts[0]
    for kp in key_parts[1:]:
        groups_test["group_key"] = groups_test["group_key"] + "_" + kp

    n_groups = groups_test["group_key"].nunique()
    print(f"  Unique groups: {n_groups} (from {len(X_test)} reports)")

    # Per-group: majority vote and mean probability
    group_results = []
    for gk, idx in groups_test.groupby("group_key").groups.items():
        idx_arr = np.array(list(idx))
        y_true_g = y_test[idx_arr]
        y_pred_g = y_pred[idx_arr]
        y_proba_g = y_proba[idx_arr]

        # Majority true label
        true_majority = np.bincount(y_true_g, minlength=3).argmax()
        true_agreement = np.bincount(y_true_g, minlength=3).max() / len(y_true_g)

        # Majority predicted
        pred_majority = np.bincount(y_pred_g, minlength=3).argmax()

        # Mean probability → argmax
        mean_proba = y_proba_g.mean(axis=0)
        pred_mean_proba = mean_proba.argmax()

        group_results.append({
            "group_key": gk,
            "n_reports": len(idx_arr),
            "true_majority": true_majority,
            "true_agreement": true_agreement,
            "pred_majority_vote": pred_majority,
            "pred_mean_proba": pred_mean_proba,
            "mean_proba": mean_proba,
        })

    gr_df = pd.DataFrame(group_results)

    # Filter to groups with some consensus
    for min_agreement in [0.0, 0.5, 0.6, 0.7, 0.8]:
        mask = gr_df["true_agreement"] >= min_agreement
        sub = gr_df[mask]
        if len(sub) == 0:
            continue

        y_true_agg = sub["true_majority"].values
        y_pred_vote = sub["pred_majority_vote"].values
        y_pred_mprob = sub["pred_mean_proba"].values

        f1_vote = f1_score(y_true_agg, y_pred_vote, average="macro", zero_division=0)
        f1_mprob = f1_score(y_true_agg, y_pred_mprob, average="macro", zero_division=0)
        acc_vote = accuracy_score(y_true_agg, y_pred_vote)

        print(f"  agreement>={min_agreement:.1f}: {len(sub)} groups, "
              f"vote F1={f1_vote:.4f} acc={acc_vote:.4f}, "
              f"mean_proba F1={f1_mprob:.4f}")

    # Group size distribution
    sizes = gr_df["n_reports"].values
    print(f"\n  Group sizes: mean={sizes.mean():.1f}, median={np.median(sizes):.0f}, "
          f"max={sizes.max()}, 1-report={((sizes==1).sum())}/{len(sizes)}")

    # Multi-report groups only
    multi = gr_df[gr_df["n_reports"] > 1]
    if len(multi) > 0:
        y_true_m = multi["true_majority"].values
        y_pred_m = multi["pred_majority_vote"].values
        y_pred_mp = multi["pred_mean_proba"].values
        f1_m = f1_score(y_true_m, y_pred_m, average="macro", zero_division=0)
        f1_mp = f1_score(y_true_m, y_pred_mp, average="macro", zero_division=0)
        print(f"  Multi-report groups ({len(multi)}): vote F1={f1_m:.4f}, mean_proba F1={f1_mp:.4f}")

    print()

    # ===== STEP 6: Confidence-aware output =====
    print("=" * 70)
    print("STEP 6: Confidence-aware output")
    print("=" * 70)

    max_proba = y_proba.max(axis=1)
    entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
    max_entropy = np.log(3)  # uniform distribution

    print(f"  Max probability: mean={max_proba.mean():.3f}, median={np.median(max_proba):.3f}")
    print(f"  Entropy: mean={entropy.mean():.3f}, max possible={max_entropy:.3f}")

    # Confidence thresholds
    print(f"\n  {'Conf thresh':>12}  {'Confident%':>10}  {'F1 conf':>8}  {'F1 unconf':>9}  {'Acc conf':>8}")
    for conf_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        confident = max_proba >= conf_thresh
        n_conf = confident.sum()
        pct_conf = n_conf / len(y_test) * 100

        if n_conf > 0 and (~confident).sum() > 0:
            f1_conf = f1_score(y_test[confident], y_pred[confident], average="macro", zero_division=0)
            f1_unc = f1_score(y_test[~confident], y_pred[~confident], average="macro", zero_division=0)
            acc_conf = accuracy_score(y_test[confident], y_pred[confident])
        elif n_conf > 0:
            f1_conf = f1_score(y_test[confident], y_pred[confident], average="macro", zero_division=0)
            f1_unc = 0
            acc_conf = accuracy_score(y_test[confident], y_pred[confident])
        else:
            continue
        print(f"  {conf_thresh:>12.1f}  {pct_conf:>9.1f}%  {f1_conf:>8.4f}  {f1_unc:>9.4f}  {acc_conf:>8.4f}")

    # Stage-specific uncertainty
    p_s1 = s1.predict_proba(X_test)
    borked_prob = p_s1[:, 0]

    # Stage 1 uncertainty zones
    print(f"\n  Stage 1 P(borked) distribution:")
    for lo, hi, label in [(0.0, 0.3, "confident works"), (0.3, 0.7, "UNCERTAIN"),
                          (0.7, 1.0, "confident borked")]:
        mask = (borked_prob >= lo) & (borked_prob < hi)
        n = mask.sum()
        if n > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    P(borked) [{lo:.1f},{hi:.1f}): {n} ({n/len(y_test)*100:.1f}%) acc={acc:.4f}")

    # Stage 2 uncertainty for non-borked
    works_mask = borked_prob < 0.5
    if works_mask.any():
        X_s2 = X_test[works_mask].copy()
        for c in s2_drops:
            if c in X_s2.columns:
                X_s2 = X_s2.drop(columns=[c])
        p_s2 = s2.predict_proba(X_s2)
        p_oob = p_s2[:, 1]

        print(f"\n  Stage 2 P(oob) distribution (non-borked only, n={works_mask.sum()}):")
        for lo, hi, label in [(0.0, 0.35, "confident tinkering"), (0.35, 0.65, "UNCERTAIN"),
                              (0.65, 1.0, "confident oob")]:
            mask_s2 = (p_oob >= lo) & (p_oob < hi)
            n = mask_s2.sum()
            print(f"    P(oob) [{lo:.2f},{hi:.2f}): {n} ({n/works_mask.sum()*100:.1f}%)")

    print()

    # ===== STEP 7: Calibration =====
    print("=" * 70)
    print("STEP 7: Post-hoc calibration")
    print("=" * 70)

    # Split test into calibration and final eval
    n_cal = len(X_test) // 2
    y_proba_cal = y_proba[:n_cal]
    y_test_cal = y_test[:n_cal]
    y_proba_eval = y_proba[n_cal:]
    y_test_eval = y_test[n_cal:]

    # ECE before calibration
    def compute_ece(y_true, y_proba, n_bins=10):
        """Expected Calibration Error for multiclass."""
        y_pred = y_proba.argmax(axis=1)
        confidences = y_proba.max(axis=1)
        accuracies = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if mask.sum() > 0:
                avg_conf = confidences[mask].mean()
                avg_acc = accuracies[mask].mean()
                ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
        return ece

    ece_before = compute_ece(y_test_eval, y_proba_eval)
    print(f"  ECE before calibration: {ece_before:.4f}")

    # Per-class Brier scores
    for cls in range(3):
        y_bin = (y_test_eval == cls).astype(int)
        brier = brier_score_loss(y_bin, y_proba_eval[:, cls])
        print(f"  Brier score {CLASS_NAMES_3[cls]}: {brier:.4f}")

    # Calibrate each class with isotonic regression
    calibrators = {}
    for cls in range(3):
        y_bin_cal = (y_test_cal == cls).astype(int)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        iso.fit(y_proba_cal[:, cls], y_bin_cal)
        calibrators[cls] = iso

    # Apply calibration
    y_proba_calibrated = np.zeros_like(y_proba_eval)
    for cls in range(3):
        y_proba_calibrated[:, cls] = calibrators[cls].predict(y_proba_eval[:, cls])

    # Renormalize
    row_sums = y_proba_calibrated.sum(axis=1, keepdims=True)
    y_proba_calibrated = y_proba_calibrated / row_sums

    ece_after = compute_ece(y_test_eval, y_proba_calibrated)
    print(f"\n  ECE after calibration:  {ece_after:.4f} (delta: {ece_after-ece_before:+.4f})")

    # Per-class Brier after
    for cls in range(3):
        y_bin = (y_test_eval == cls).astype(int)
        brier_before = brier_score_loss(y_bin, y_proba_eval[:, cls])
        brier_after = brier_score_loss(y_bin, y_proba_calibrated[:, cls])
        print(f"  Brier {CLASS_NAMES_3[cls]}: {brier_before:.4f} → {brier_after:.4f} ({brier_after-brier_before:+.4f})")

    # F1 after calibration (should be similar — calibration doesn't change ranking much)
    y_pred_cal = y_proba_calibrated.argmax(axis=1)
    f1_cal = f1_score(y_test_eval, y_pred_cal, average="macro", zero_division=0)
    y_pred_uncal = y_proba_eval.argmax(axis=1)
    f1_uncal = f1_score(y_test_eval, y_pred_uncal, average="macro", zero_division=0)
    print(f"\n  F1 uncalibrated: {f1_uncal:.4f}")
    print(f"  F1 calibrated:   {f1_cal:.4f}")

    # Calibration curve per class
    print(f"\n  Calibration curves (10 bins):")
    for cls in range(3):
        y_bin = (y_test_eval == cls).astype(int)

        # Before
        prob_true_b, prob_pred_b = calibration_curve(y_bin, y_proba_eval[:, cls], n_bins=5, strategy="uniform")
        # After
        prob_true_a, prob_pred_a = calibration_curve(y_bin, y_proba_calibrated[:, cls], n_bins=5, strategy="uniform")

        print(f"\n  {CLASS_NAMES_3[cls]}:")
        print(f"    {'Bin':>5}  {'Pred(before)':>13}  {'True(before)':>13}  {'Pred(after)':>12}  {'True(after)':>12}")
        for i in range(len(prob_true_b)):
            pa = prob_pred_a[i] if i < len(prob_pred_a) else float('nan')
            ta = prob_true_a[i] if i < len(prob_true_a) else float('nan')
            print(f"    {i:>5}  {prob_pred_b[i]:>13.4f}  {prob_true_b[i]:>13.4f}  {pa:>12.4f}  {ta:>12.4f}")

    # ===== SUMMARY =====
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline cascade F1: {f1_base:.4f}")
    print(f"  Unique (game,hw) groups: {n_groups}")
    if len(multi) > 0:
        print(f"  Multi-report group F1 (vote): {f1_m:.4f}")
    print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}")
    print(f"  Stage 1 uncertain (P(borked) 0.3-0.7): "
          f"{((borked_prob >= 0.3) & (borked_prob < 0.7)).sum()/len(y_test)*100:.1f}%")
    if works_mask.any():
        print(f"  Stage 2 uncertain (P(oob) 0.35-0.65): "
              f"{((p_oob >= 0.35) & (p_oob < 0.65)).sum()/works_mask.sum()*100:.1f}%")


if __name__ == "__main__":
    main()
