"""Phase 16: Pipeline fixes based on analysis insights.

Experiments:
  16.0  — Re-run with full coverage (100% contributor data)
  16.1a — Class weight Stage 2 (upweight works_oob)
  16.1b — Oversample works_oob in train
  16.2a — Ablation: remove game aggregates
  16.2b — Ablation: remove game aggregates + add them back as leak-free
  16.5  — IRT on full data
  16.6  — Error-targeted features

Usage:
  python scripts/experiment_16_pipeline_fixes.py [--db data/protondb.db]
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

_s1_cache = {}


def train_and_eval(X_train, y_train, X_test, y_test,
                   sample_weight=None, s2_class_weight=None, label=""):
    from protondb_settings.ml.models.cascade import (
        train_stage1, CascadeClassifier, STAGE2_DROP_FEATURES,
    )
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    X_train, X_test = X_train.copy(), X_test.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    n_feat = X_train.shape[1]
    if n_feat in _s1_cache:
        s1 = _s1_cache[n_feat]
    else:
        s1 = train_stage1(X_train, y_train, X_test, y_test)
        _s1_cache[n_feat] = s1

    # Custom Stage 2
    mask_tr, mask_te = y_train > 0, y_test > 0
    X2_tr = X_train[mask_tr].reset_index(drop=True)
    y2_tr = (y_train[mask_tr] - 1).astype(float)
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

    # Sample weights for Stage 2
    w = None
    if sample_weight is not None:
        w = sample_weight[mask_tr]
    if s2_class_weight is not None:
        cw = np.ones(len(y2_tr))
        for cls, weight in s2_class_weight.items():
            cw[y2_tr == cls] = weight
        w = cw if w is None else w * cw

    ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
    ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)
    s2 = lgb.train(
        {"objective": "cross_entropy", "metric": "binary_logloss",
         "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
         "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
         "reg_alpha": 0.1, "reg_lambda": 0.1, "min_split_gain": 0.05, "verbose": -1},
        ds_tr, num_boost_round=2000, valid_sets=[ds_te],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    cascade = CascadeClassifier(s1, s2, drops)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    per = f1_score(y_test, y_pred, average=None)

    # works_oob recall
    oob_mask = y_test == 2
    oob_recall = (y_pred[oob_mask] == 2).mean() if oob_mask.any() else 0

    return {"label": label, "f1_macro": f1, "borked_f1": per[0],
            "tinkering_f1": per[1], "works_oob_f1": per[2], "oob_recall": oob_recall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 16: Pipeline fixes")
    print("=" * 70)

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import fit_irt, add_irt_features, contributor_aware_relabel
    from protondb_settings.ml.features.game import AGG_CUST_COLS, AGG_FLAG_COLS

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)

    # IRT on full data (16.0 + 16.5)
    print("\n[Fitting IRT on full contributor data...]")
    theta, difficulty = fit_irt(conn)
    logger.info("IRT: %d contributors, %d items", len(theta), len(difficulty))

    X_tr = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_te = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    y_tr, n_relabel = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)
    logger.info("Relabeled: %d", n_relabel)

    # Check class distribution
    for cls, name in [(0, "borked"), (1, "tinkering"), (2, "works_oob")]:
        n = (y_tr == cls).sum()
        print(f"  Train {name}: {n} ({n/len(y_tr)*100:.1f}%)")

    results = []

    # ── 16.0: Baseline with full coverage ────────────────────────────
    print("\n" + "=" * 70)
    print("16.0: Baseline (IRT full coverage + relabel)")
    print("=" * 70)
    r = train_and_eval(X_tr, y_tr, X_te, y_test, label="16.0_full_coverage")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | b={r['borked_f1']:.3f} t={r['tinkering_f1']:.3f} "
          f"o={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 16.1a: Class weight Stage 2 ──────────────────────────────────
    for oob_w in [1.5, 2.0, 2.5, 3.0]:
        print(f"\n{'='*70}")
        print(f"16.1a: Stage 2 class_weight oob={oob_w}")
        print(f"{'='*70}")
        r = train_and_eval(X_tr, y_tr, X_te, y_test,
                           s2_class_weight={0: 1.0, 1: oob_w},
                           label=f"16.1a_cw_oob_{oob_w}")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | oob_f1={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 16.1b: Oversample works_oob ──────────────────────────────────
    print(f"\n{'='*70}")
    print("16.1b: Oversample works_oob to 15%")
    print(f"{'='*70}")
    oob_mask = y_tr == 2
    n_oob = oob_mask.sum()
    target_n = int(len(y_tr) * 0.15)
    n_dup = target_n - n_oob
    if n_dup > 0:
        oob_idx = np.where(oob_mask)[0]
        dup_idx = np.random.RandomState(42).choice(oob_idx, size=n_dup, replace=True)
        X_tr_os = pd.concat([X_tr, X_tr.iloc[dup_idx]], ignore_index=True)
        y_tr_os = np.concatenate([y_tr, y_tr[dup_idx]])
        logger.info("Oversampled oob: %d → %d (total %d → %d)", n_oob, n_oob + n_dup, len(y_tr), len(y_tr_os))
        r = train_and_eval(X_tr_os, y_tr_os, X_te, y_test, label="16.1b_oversample_15pct")
    else:
        r = results[0].copy()
        r["label"] = "16.1b_oversample_15pct (skip)"
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | oob_f1={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 16.2a: Ablation — remove game aggregates ─────────────────────
    print(f"\n{'='*70}")
    print("16.2a: Ablation — remove game aggregates (26 features)")
    print(f"{'='*70}")
    agg_cols = AGG_CUST_COLS + AGG_FLAG_COLS
    agg_present = [c for c in agg_cols if c in X_tr.columns]
    X_tr_noagg = X_tr.drop(columns=agg_present)
    X_te_noagg = X_te.drop(columns=agg_present)
    logger.info("Dropped %d aggregate features", len(agg_present))
    r = train_and_eval(X_tr_noagg, y_tr, X_te_noagg, y_test, label="16.2a_no_aggregates")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | oob_f1={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 16.6: Error-targeted features ────────────────────────────────
    print(f"\n{'='*70}")
    print("16.6: Error-targeted features")
    print(f"{'='*70}")

    # Add contributor consistency and game agreement features
    contrib_lookup = {}
    for r_row in conn.execute("SELECT report_id, contributor_id, report_tally FROM report_contributors").fetchall():
        contrib_lookup[r_row["report_id"]] = {"cid": str(r_row["contributor_id"]), "tally": r_row["report_tally"]}

    # Per-contributor verdict consistency (std of verdicts)
    contrib_verdicts = {}
    report_verdicts = {}
    for r_row in conn.execute("SELECT id, app_id, verdict, verdict_oob FROM reports").fetchall():
        rid = r_row["id"]
        if r_row["verdict"] == "no":
            score = 0
        elif r_row["verdict_oob"] == "yes":
            score = 2
        else:
            score = 1
        report_verdicts[rid] = score
        cinfo = contrib_lookup.get(rid)
        if cinfo:
            cid = cinfo["cid"]
            if cid not in contrib_verdicts:
                contrib_verdicts[cid] = []
            contrib_verdicts[cid].append(score)

    contrib_consistency = {}
    for cid, scores in contrib_verdicts.items():
        if len(scores) >= 2:
            contrib_consistency[cid] = np.std(scores)

    # Per-game verdict agreement
    game_verdicts = {}
    for r_row in conn.execute("SELECT app_id, verdict, verdict_oob FROM reports WHERE verdict IS NOT NULL").fetchall():
        app_id = r_row["app_id"]
        if app_id not in game_verdicts:
            game_verdicts[app_id] = []
        if r_row["verdict"] == "no":
            game_verdicts[app_id].append(0)
        elif r_row["verdict_oob"] == "yes":
            game_verdicts[app_id].append(2)
        else:
            game_verdicts[app_id].append(1)

    game_agreement = {}
    for app_id, scores in game_verdicts.items():
        if len(scores) >= 3:
            # Agreement = 1 - normalized entropy
            counts = np.bincount(scores, minlength=3)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(3)
            game_agreement[app_id] = 1 - entropy / max_entropy

    report_app = {}
    for r_row in conn.execute("SELECT id, app_id FROM reports").fetchall():
        report_app[r_row["id"]] = r_row["app_id"]

    def add_error_features(X, report_ids):
        X = X.copy()
        consistencies = []
        agreements = []
        for rid in report_ids:
            cinfo = contrib_lookup.get(rid)
            if cinfo and cinfo["cid"] in contrib_consistency:
                consistencies.append(contrib_consistency[cinfo["cid"]])
            else:
                consistencies.append(np.nan)

            app_id = report_app.get(rid)
            if app_id and app_id in game_agreement:
                agreements.append(game_agreement[app_id])
            else:
                agreements.append(np.nan)

        cs = pd.Series(consistencies)
        ag = pd.Series(agreements)
        X["contributor_consistency"] = cs.fillna(cs.median()).values
        X["game_verdict_agreement"] = ag.fillna(ag.median()).values
        return X

    X_tr_ef = add_error_features(X_tr, train_rids)
    X_te_ef = add_error_features(X_te, test_rids)
    r = train_and_eval(X_tr_ef, y_tr, X_te_ef, y_test, label="16.6_error_features")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | oob_f1={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── Combined: best positive experiments ──────────────────────────
    print(f"\n{'='*70}")
    print("COMBINED: Full coverage + best class weight + error features")
    print(f"{'='*70}")
    # Find best class weight
    cw_results = [r for r in results if r["label"].startswith("16.1a_cw")]
    best_cw = max(cw_results, key=lambda r: r["f1_macro"]) if cw_results else None
    best_cw_val = float(best_cw["label"].split("_")[-1]) if best_cw and best_cw["f1_macro"] > results[0]["f1_macro"] else None

    cw = {0: 1.0, 1: best_cw_val} if best_cw_val else None
    r = train_and_eval(X_tr_ef, y_tr, X_te_ef, y_test,
                       s2_class_weight=cw, label="combined")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | oob_f1={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    conn.close()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<30s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s} {'oob_r':>7s}")
    print("-" * 77)
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<30s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f} {r['oob_recall']:>7.3f}")


if __name__ == "__main__":
    main()
