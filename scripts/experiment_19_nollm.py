"""Phase 19.1-19.2: Data quality without LLM.

Experiments:
  19.1a — Stage 2 only on explicit verdict_oob labels
  19.1b — Explicit labels weight=1.0, inferred weight=0.3
  19.1c — Explicit + IRT-relabeled inferred (trusted subset)
  19.2a — Temporal filter: drop reports > 4 years
  19.2b — Temporal filter: drop reports > 3 years
  Combined — best filters together

Usage:
  python scripts/experiment_19_nollm.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features,
    )
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y_raw, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y_raw, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)
    theta, difficulty = fit_irt(conn)
    X_train = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_test = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    X_train = add_error_targeted_features(X_train, train_rids, conn)
    X_test = add_error_targeted_features(X_test, test_rids, conn)

    # Get verdict_oob info for each train report
    has_explicit_oob = set()
    report_ts = {}
    for r in conn.execute("SELECT id, verdict_oob, timestamp FROM reports").fetchall():
        if r["verdict_oob"] is not None:
            has_explicit_oob.add(r["id"])
        try:
            report_ts[r["id"]] = int(r["timestamp"]) if r["timestamp"] else 0
        except (ValueError, TypeError):
            report_ts[r["id"]] = 0

    conn2 = get_connection(args.db)
    y_baseline, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn2, theta)
    conn2.close()
    conn.close()

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Timestamps for filtering
    train_timestamps = np.array([report_ts.get(rid, 0) for rid in train_rids])
    max_ts = train_timestamps.max()
    train_has_oob = np.array([rid in has_explicit_oob for rid in train_rids])

    logger.info("Train: %d total, %d with explicit oob (%.1f%%)",
                len(train_rids), train_has_oob.sum(), train_has_oob.mean() * 100)

    results = []

    def train_eval(X_tr, y_tr, label, sample_weight=None):
        s1 = train_stage1(X_tr, y_tr, X_test, y_test)
        # Custom Stage 2 with optional weights
        from protondb_settings.ml.models.cascade import STAGE2_DROP_FEATURES
        mask_tr, mask_te = y_tr > 0, y_test > 0
        X2_tr = X_tr[mask_tr].reset_index(drop=True)
        y2_tr = (y_tr[mask_tr] - 1).astype(float)
        X2_te = X_test[mask_te].reset_index(drop=True)
        y2_te = (y_test[mask_te] - 1).astype(float)
        drops = [c for c in STAGE2_DROP_FEATURES if c in X2_tr.columns]
        if drops:
            X2_tr = X2_tr.drop(columns=drops)
            X2_te = X2_te.drop(columns=drops)
        cats = [c for c in CATEGORICAL_FEATURES if c in X2_tr.columns]
        for c in cats:
            X2_tr[c] = X2_tr[c].astype("category")
            X2_te[c] = X2_te[c].astype("category")
        y_smooth = y2_tr * 0.85 + (1 - y2_tr) * 0.15
        w = np.ones(len(y2_tr))
        w[y2_tr >= 0.5] = 1.8
        if sample_weight is not None:
            w = w * sample_weight[mask_tr]
        ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
        ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)
        s2 = lgb.train(
            {"objective": "cross_entropy", "metric": "binary_logloss",
             "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
             "reg_alpha": 1.0, "reg_lambda": 1.0, "min_split_gain": 0.05, "verbose": -1},
            ds_tr, num_boost_round=2000, valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
        )
        cascade = CascadeClassifier(s1, s2, drops)
        y_pred = cascade.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        per = f1_score(y_test, y_pred, average=None)
        oob_r = (y_pred[y_test == 2] == 2).mean()
        r = {"label": label, "f1_macro": f1, "borked_f1": per[0],
             "tinkering_f1": per[1], "works_oob_f1": per[2], "oob_recall": oob_r}
        results.append(r)
        print(f"  {label:40s} F1={f1:.4f} b={per[0]:.3f} t={per[1]:.3f} o={per[2]:.3f} oob_r={oob_r:.3f}")
        return f1

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)
    train_eval(X_train, y_baseline, "baseline")

    # ── 19.1a: Stage 2 only on explicit oob labels ───────────────────
    print("\n" + "=" * 70)
    print("19.1a: Explicit oob labels only (hard filter)")
    print("=" * 70)
    # Keep all data for Stage 1, but for Stage 2 only use explicit oob
    # Approach: set inferred tinkering (y=1, no oob) to borked-like (excluded from Stage 2)
    # Better: use sample weight = 0 for inferred in Stage 2
    w_explicit = np.ones(len(y_baseline))
    for i, rid in enumerate(train_rids):
        if y_baseline[i] in (1, 2) and rid not in has_explicit_oob:
            w_explicit[i] = 0.0  # exclude from Stage 2
    n_s2 = (w_explicit[y_baseline > 0] > 0).sum()
    logger.info("19.1a: %d Stage 2 samples (explicit only)", n_s2)
    train_eval(X_train, y_baseline, "19.1a_explicit_only", sample_weight=w_explicit)

    # ── 19.1b: Explicit weight=1.0, inferred weight=0.3 ─────────────
    print("\n" + "=" * 70)
    print("19.1b: Explicit=1.0, inferred=0.3")
    print("=" * 70)
    w_soft = np.ones(len(y_baseline))
    for i, rid in enumerate(train_rids):
        if y_baseline[i] in (1, 2) and rid not in has_explicit_oob:
            w_soft[i] = 0.3
    train_eval(X_train, y_baseline, "19.1b_inferred_weight_0.3", sample_weight=w_soft)

    # ── 19.1c: Explicit weight=1.0, inferred weight=0.5 ─────────────
    print("\n" + "=" * 70)
    print("19.1c: Explicit=1.0, inferred=0.5")
    print("=" * 70)
    w_soft2 = np.ones(len(y_baseline))
    for i, rid in enumerate(train_rids):
        if y_baseline[i] in (1, 2) and rid not in has_explicit_oob:
            w_soft2[i] = 0.5
    train_eval(X_train, y_baseline, "19.1c_inferred_weight_0.5", sample_weight=w_soft2)

    # ── 19.2a: Temporal filter > 4 years ─────────────────────────────
    print("\n" + "=" * 70)
    print("19.2a: Drop reports > 4 years old")
    print("=" * 70)
    cutoff_4y = max_ts - 4 * 365 * 86400
    mask_4y = train_timestamps >= cutoff_4y
    X_tr_4y = X_train[mask_4y].reset_index(drop=True)
    y_tr_4y = y_baseline[mask_4y]
    logger.info("19.2a: %d → %d (dropped %d, %.1f%%)",
                len(y_baseline), len(y_tr_4y), (~mask_4y).sum(), (~mask_4y).mean() * 100)
    train_eval(X_tr_4y, y_tr_4y, "19.2a_drop_4y")

    # ── 19.2b: Temporal filter > 3 years ─────────────────────────────
    print("\n" + "=" * 70)
    print("19.2b: Drop reports > 3 years old")
    print("=" * 70)
    cutoff_3y = max_ts - 3 * 365 * 86400
    mask_3y = train_timestamps >= cutoff_3y
    X_tr_3y = X_train[mask_3y].reset_index(drop=True)
    y_tr_3y = y_baseline[mask_3y]
    logger.info("19.2b: %d → %d (dropped %d, %.1f%%)",
                len(y_baseline), len(y_tr_3y), (~mask_3y).sum(), (~mask_3y).mean() * 100)
    train_eval(X_tr_3y, y_tr_3y, "19.2b_drop_3y")

    # ── Combined: temporal + explicit weighting ──────────────────────
    print("\n" + "=" * 70)
    print("COMBINED: Drop >4y + inferred weight 0.3")
    print("=" * 70)
    w_combined = w_soft[mask_4y]
    train_eval(X_tr_4y, y_tr_4y, "combined_4y+weight0.3", sample_weight=w_combined)

    print("\n" + "=" * 70)
    print("COMBINED: Drop >3y + inferred weight 0.5")
    print("=" * 70)
    w_combined2 = w_soft2[mask_3y]
    train_eval(X_tr_3y, y_tr_3y, "combined_3y+weight0.5", sample_weight=w_combined2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<40s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s} {'oob_r':>7s}")
    print("-" * 90)
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<40s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f} {r.get('oob_recall',0):>7.3f}")


if __name__ == "__main__":
    main()
