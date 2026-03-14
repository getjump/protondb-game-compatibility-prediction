"""Phase 20.2-20.3: Per-game evaluation + variant ablation.

Experiments:
  20.2  — Per-(game, gpu_family) aggregated evaluation
  20.3a — Stage 2 without variant
  20.3b — Stage 2 variant interaction features

Usage:
  python scripts/experiment_20_reformulation.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features,
    )
    from protondb_settings.ml.models.cascade import (
        train_stage1, train_stage2, CascadeClassifier, STAGE2_DROP_FEATURES,
    )
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
    y_train, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)

    # Build report metadata
    report_meta = {}
    for r in conn.execute("SELECT id, app_id, gpu FROM reports").fetchall():
        report_meta[r["id"]] = {"app_id": r["app_id"], "gpu_family": extract_gpu_family(r["gpu"]) if r["gpu"] else "unknown"}
    conn.close()

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # ── Train baseline cascade ───────────────────────────────────────
    print("=" * 70)
    print("Training baseline cascade...")
    print("=" * 70)
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_drops)

    y_pred = cascade.predict(X_test)
    y_proba = cascade.predict_proba(X_test)
    f1_baseline = f1_score(y_test, y_pred, average="macro")
    per_baseline = f1_score(y_test, y_pred, average=None)
    print(f"\nBaseline per-report: F1={f1_baseline:.4f} b={per_baseline[0]:.3f} "
          f"t={per_baseline[1]:.3f} o={per_baseline[2]:.3f}")

    # ── 20.2: Per-(game, gpu_family) aggregated evaluation ───────────
    print(f"\n{'='*70}")
    print("20.2: Per-(game, gpu_family) aggregated evaluation")
    print(f"{'='*70}")

    # Group test reports by (app_id, gpu_family)
    pair_preds = defaultdict(list)    # (app_id, gpu_fam) → [pred1, pred2, ...]
    pair_truths = defaultdict(list)
    pair_probas = defaultdict(list)

    for i, rid in enumerate(test_rids):
        meta = report_meta.get(rid, {})
        app_id = meta.get("app_id")
        gpu_fam = meta.get("gpu_family", "unknown")
        if app_id is None:
            continue
        pair = (app_id, gpu_fam)
        pair_preds[pair].append(y_pred[i])
        pair_truths[pair].append(y_test[i])
        pair_probas[pair].append(y_proba[i])

    # Aggregated predictions
    agg_true = []
    agg_pred_vote = []      # majority vote
    agg_pred_proba = []     # argmax of mean probabilities
    agg_pred_conf = []      # confidence-weighted
    pair_sizes = []

    for pair, preds in pair_preds.items():
        truths = pair_truths[pair]
        probas = np.array(pair_probas[pair])

        # Ground truth: majority vote of actual verdicts
        true_counts = Counter(truths)
        true_agg = true_counts.most_common(1)[0][0]
        agg_true.append(true_agg)

        # Prediction: majority vote
        pred_counts = Counter(preds)
        agg_pred_vote.append(pred_counts.most_common(1)[0][0])

        # Prediction: mean probability → argmax
        mean_proba = probas.mean(axis=0)
        agg_pred_proba.append(mean_proba.argmax())

        # Prediction: confidence-weighted (weight by max probability)
        weights = probas.max(axis=1)
        weighted_proba = (probas * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
        agg_pred_conf.append(weighted_proba.argmax())

        pair_sizes.append(len(preds))

    agg_true = np.array(agg_true)
    agg_pred_vote = np.array(agg_pred_vote)
    agg_pred_proba = np.array(agg_pred_proba)
    agg_pred_conf = np.array(agg_pred_conf)
    pair_sizes = np.array(pair_sizes)

    target_names = ["borked", "tinkering", "works_oob"]

    print(f"\nTotal pairs: {len(agg_true)}")
    print(f"Pair sizes: median={np.median(pair_sizes):.0f}, mean={np.mean(pair_sizes):.1f}, "
          f"max={pair_sizes.max()}")

    # Per-report baseline on test
    print(f"\n  Per-report:           F1={f1_baseline:.4f}")

    # Aggregated metrics
    for name, pred in [("vote", agg_pred_vote), ("mean_proba", agg_pred_proba),
                       ("conf_weighted", agg_pred_conf)]:
        f1 = f1_score(agg_true, pred, average="macro")
        acc = accuracy_score(agg_true, pred)
        per = f1_score(agg_true, pred, average=None)
        print(f"  Per-pair ({name:14s}): F1={f1:.4f} acc={acc:.4f} "
              f"b={per[0]:.3f} t={per[1]:.3f} o={per[2]:.3f}")

    # Aggregated on pairs with 3+ reports (more reliable)
    mask_3 = pair_sizes >= 3
    if mask_3.sum() > 50:
        f1_3 = f1_score(agg_true[mask_3], agg_pred_proba[mask_3], average="macro")
        acc_3 = accuracy_score(agg_true[mask_3], agg_pred_proba[mask_3])
        print(f"\n  Per-pair (3+ reports, n={mask_3.sum()}): F1={f1_3:.4f} acc={acc_3:.4f}")

    mask_5 = pair_sizes >= 5
    if mask_5.sum() > 50:
        f1_5 = f1_score(agg_true[mask_5], agg_pred_proba[mask_5], average="macro")
        acc_5 = accuracy_score(agg_true[mask_5], agg_pred_proba[mask_5])
        print(f"  Per-pair (5+ reports, n={mask_5.sum()}): F1={f1_5:.4f} acc={acc_5:.4f}")

    # Binary aggregated: borked vs works
    print(f"\n  Binary (borked vs works):")
    agg_true_bin = (agg_true > 0).astype(int)
    agg_pred_bin = (agg_pred_proba > 0).astype(int)
    f1_bin = f1_score(agg_true_bin, agg_pred_bin, average="macro")
    acc_bin = accuracy_score(agg_true_bin, agg_pred_bin)
    print(f"    Per-pair: F1={f1_bin:.4f} acc={acc_bin:.4f}")

    if mask_3.sum() > 50:
        f1_bin_3 = f1_score(agg_true_bin[mask_3], agg_pred_bin[mask_3], average="macro")
        print(f"    Per-pair (3+ reports): F1={f1_bin_3:.4f}")

    # ── 20.3a: Stage 2 without variant ───────────────────────────────
    print(f"\n{'='*70}")
    print("20.3a: Stage 2 WITHOUT variant")
    print(f"{'='*70}")

    s2_no_var, drops_no_var = train_stage2(
        X_train, y_train, X_test, y_test,
        drop_features=list(STAGE2_DROP_FEATURES) + ["variant"])
    cascade_no_var = CascadeClassifier(s1, s2_no_var, drops_no_var)
    y_pred_nv = cascade_no_var.predict(X_test)
    f1_nv = f1_score(y_test, y_pred_nv, average="macro")
    per_nv = f1_score(y_test, y_pred_nv, average=None)
    oob_r_nv = (y_pred_nv[y_test == 2] == 2).mean()

    print(f"  Without variant: F1={f1_nv:.4f} (Δ={f1_nv-f1_baseline:+.4f}) "
          f"b={per_nv[0]:.3f} t={per_nv[1]:.3f} o={per_nv[2]:.3f} oob_r={oob_r_nv:.3f}")

    # What took variant's place?
    s2_nv_names = s2_no_var.feature_name()
    s2_nv_imp = pd.Series(s2_no_var.feature_importance(importance_type="gain"),
                          index=s2_nv_names).sort_values(ascending=False)
    print(f"\n  Top-10 without variant:")
    for feat, imp in s2_nv_imp.head(10).items():
        print(f"    {feat:35s} {imp:10.1f}")

    # ── 20.3b: Stage 2 with variant interactions ─────────────────────
    print(f"\n{'='*70}")
    print("20.3b: Stage 2 with variant × IRT interactions")
    print(f"{'='*70}")

    X_train_vi = X_train.copy()
    X_test_vi = X_test.copy()

    # variant × irt_difficulty interaction
    if "variant" in X_train_vi.columns and "irt_game_difficulty" in X_train_vi.columns:
        # Encode variant as numeric for interaction
        var_map = {"official": 0, "ge": 1, "experimental": 2, "native": 3, "notListed": 4, "older": 5}
        for df in [X_train_vi, X_test_vi]:
            var_numeric = df["variant"].astype(str).map(var_map).fillna(-1).astype(float)
            df["variant_x_irt_diff"] = var_numeric * df["irt_game_difficulty"]
            df["variant_x_irt_strict"] = var_numeric * df["irt_contributor_strictness"]

    for col in CATEGORICAL_FEATURES:
        if col in X_train_vi.columns:
            X_train_vi[col] = X_train_vi[col].astype("category")
        if col in X_test_vi.columns:
            X_test_vi[col] = X_test_vi[col].astype("category")

    s2_vi, drops_vi = train_stage2(X_train_vi, y_train, X_test_vi, y_test)
    cascade_vi = CascadeClassifier(s1, s2_vi, drops_vi)
    y_pred_vi = cascade_vi.predict(X_test_vi)
    f1_vi = f1_score(y_test, y_pred_vi, average="macro")
    per_vi = f1_score(y_test, y_pred_vi, average=None)

    print(f"  With interactions: F1={f1_vi:.4f} (Δ={f1_vi-f1_baseline:+.4f}) "
          f"b={per_vi[0]:.3f} t={per_vi[1]:.3f} o={per_vi[2]:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Per-report baseline:          F1={f1_baseline:.4f}")
    print(f"  Per-pair aggregated (proba):  F1={f1_score(agg_true, agg_pred_proba, average='macro'):.4f}")
    if mask_3.sum() > 50:
        print(f"  Per-pair 3+ reports:          F1={f1_3:.4f}")
    if mask_5.sum() > 50:
        print(f"  Per-pair 5+ reports:          F1={f1_5:.4f}")
    print(f"  Per-pair binary (borked/works): F1={f1_bin:.4f}")
    print(f"  Without variant:              F1={f1_nv:.4f} (Δ={f1_nv-f1_baseline:+.4f})")
    print(f"  With variant interactions:    F1={f1_vi:.4f} (Δ={f1_vi-f1_baseline:+.4f})")


if __name__ == "__main__":
    main()
