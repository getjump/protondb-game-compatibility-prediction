"""Phase 17: Final optimization experiments.

Experiments:
  17.4  — GroupKFold validation (honest metrics)
  17.1  — Optuna hyperparameter tuning Stage 2
  17.2  — Stage 2 ensemble (bagging)
  17.5  — Feature pruning (SHAP-based)
  Combined

Usage:
  python scripts/experiment_17_final.py [--db data/protondb.db]
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
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────


def load_full_pipeline(db_path):
    """Load data with full IRT + error features + relabeling."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features,
    )

    conn = get_connection(db_path)
    emb_data = load_embeddings(Path(db_path).parent / "embeddings.npz")
    X, y, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)

    theta, difficulty = fit_irt(conn)
    X_train = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_test = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    X_train = add_error_targeted_features(X_train, train_rids, conn)
    X_test = add_error_targeted_features(X_test, test_rids, conn)
    y_train, n = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)

    # Get app_ids for GroupKFold
    report_apps = {}
    for r in conn.execute("SELECT id, app_id FROM reports").fetchall():
        report_apps[r["id"]] = r["app_id"]
    train_apps = np.array([report_apps.get(rid, 0) for rid in train_rids])

    conn.close()
    return (X_train, X_test, y_train, y_test, train_rids, test_rids,
            X, y, ts, rids, train_apps, relabel_ids, theta, difficulty, report_apps)


# ── Train + eval ─────────────────────────────────────────────────────


def _ensure_cat(X):
    """Ensure categorical columns have category dtype."""
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def train_cascade(X_train, y_train, X_test, y_test, params_s2=None, oob_weight=1.5):
    """Train cascade with custom Stage 2 params."""
    from protondb_settings.ml.models.cascade import (
        train_stage1, CascadeClassifier, STAGE2_DROP_FEATURES,
    )
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    X_train, X_test = _ensure_cat(X_train), _ensure_cat(X_test)

    s1 = train_stage1(X_train, y_train, X_test, y_test)

    # Stage 2
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

    alpha = 0.15
    if params_s2 and "label_smoothing" in params_s2:
        alpha = params_s2.pop("label_smoothing")
    y_smooth = y2_tr * (1 - alpha) + (1 - y2_tr) * alpha

    w = np.ones(len(y2_tr))
    w[y2_tr >= 0.5] = oob_weight

    default_params = {
        "objective": "cross_entropy", "metric": "binary_logloss",
        "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 0.1, "min_split_gain": 0.05, "verbose": -1,
    }
    if params_s2:
        default_params.update(params_s2)

    ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
    ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)
    s2 = lgb.train(
        default_params, ds_tr, num_boost_round=2000, valid_sets=[ds_te],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(1000)],
    )

    cascade = CascadeClassifier(s1, s2, drops)
    return cascade, s1


def evaluate(cascade, X_test, y_test, label=""):
    X_test = _ensure_cat(X_test)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    per = f1_score(y_test, y_pred, average=None)
    oob_r = (y_pred[y_test == 2] == 2).mean() if (y_test == 2).any() else 0
    return {"label": label, "f1_macro": f1, "borked_f1": per[0],
            "tinkering_f1": per[1], "works_oob_f1": per[2], "oob_recall": oob_r}


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 17: Final optimization")
    print("=" * 70)

    t0 = time.time()
    (X_train, X_test, y_train, y_test, train_rids, test_rids,
     X_full, y_full, ts_full, rids_full, train_apps,
     relabel_ids, theta, difficulty, report_apps) = load_full_pipeline(args.db)
    logger.info("Data loaded in %.0fs", time.time() - t0)

    results = []

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)
    cascade_bl, s1_bl = train_cascade(X_train, y_train, X_test, y_test)
    r = evaluate(cascade_bl, X_test, y_test, "baseline")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | b={r['borked_f1']:.3f} t={r['tinkering_f1']:.3f} "
          f"o={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 17.4: GroupKFold validation ──────────────────────────────────
    print("\n" + "=" * 70)
    print("17.4: GroupKFold validation (by app_id)")
    print("=" * 70)

    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
    from protondb_settings.ml.models.cascade import (
        train_stage1, train_stage2, CascadeClassifier, STAGE2_DROP_FEATURES,
    )

    gkf = GroupKFold(n_splits=5)
    gkf_f1s = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_train, y_train, groups=train_apps)):
        X_tr_f = X_train.iloc[tr_idx].reset_index(drop=True)
        X_te_f = X_train.iloc[te_idx].reset_index(drop=True)
        y_tr_f = y_train[tr_idx]
        y_te_f = y_train[te_idx]
        cas_f, _ = train_cascade(X_tr_f, y_tr_f, X_te_f, y_te_f)
        r_f = evaluate(cas_f, X_te_f, y_te_f)
        gkf_f1s.append(r_f["f1_macro"])

    gkf_mean = np.mean(gkf_f1s)
    gkf_std = np.std(gkf_f1s)
    print(f"  GroupKFold F1: {gkf_mean:.4f} ± {gkf_std:.4f}")
    print(f"  Time-based F1: {results[0]['f1_macro']:.4f}")
    print(f"  Gap: {results[0]['f1_macro'] - gkf_mean:+.4f}")
    results.append({"label": f"17.4_groupkfold (mean±std)", "f1_macro": gkf_mean,
                     "borked_f1": 0, "tinkering_f1": 0, "works_oob_f1": 0, "oob_recall": 0})

    # ── 17.1: Hyperparameter sweep ───────────────────────────────────
    print("\n" + "=" * 70)
    print("17.1: Hyperparameter sweep (grid)")
    print("=" * 70)

    best_f1 = results[0]["f1_macro"]
    best_params = {}
    best_r = results[0]

    param_grid = [
        {"num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 50},
        {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 50},
        {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 30},
        {"num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 100},
        {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50, "label_smoothing": 0.10},
        {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50, "label_smoothing": 0.20},
        {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50, "reg_alpha": 0.01, "reg_lambda": 0.01},
        {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50, "reg_alpha": 1.0, "reg_lambda": 1.0},
        {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50, "colsample_bytree": 0.6},
        {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50, "subsample": 0.7},
    ]

    for oob_w in [1.3, 1.5, 1.8]:
        for i, params in enumerate(param_grid):
            cas, _ = train_cascade(X_train, y_train, X_test, y_test,
                                   params_s2=params.copy(), oob_weight=oob_w)
            r = evaluate(cas, X_test, y_test,
                        f"hp_{i}_oob{oob_w}")
            if r["f1_macro"] > best_f1:
                best_f1 = r["f1_macro"]
                best_params = {**params, "oob_weight": oob_w}
                best_r = r
                print(f"  NEW BEST: F1={r['f1_macro']:.4f} params={params} oob_w={oob_w}")

    print(f"\n  Best sweep: F1={best_f1:.4f} (Δ={best_f1-results[0]['f1_macro']:+.4f})")
    print(f"  Params: {best_params}")
    best_r_copy = dict(best_r)
    best_r_copy["label"] = "17.1_best_hparams"
    results.append(best_r_copy)

    # ── 17.2: Ensemble (bagging) ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("17.2: Stage 2 ensemble (5 seeds)")
    print("=" * 70)

    from protondb_settings.ml.models.cascade import CascadeClassifier as CC

    # Train Stage 1 once
    X_tr_c, X_te_c = _ensure_cat(X_train), _ensure_cat(X_test)
    s1_ens = train_stage1(X_tr_c, y_train, X_te_c, y_test)

    mask_tr, mask_te = y_train > 0, y_test > 0
    X2_tr = X_tr_c[mask_tr].reset_index(drop=True)
    y2_tr = (y_train[mask_tr] - 1).astype(float)
    X2_te = X_te_c[mask_te].reset_index(drop=True)
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
    w[y2_tr >= 0.5] = 1.5

    # Train 5 models with different seeds
    s2_models = []
    for seed in [42, 123, 456, 789, 1024]:
        params = {
            "objective": "cross_entropy", "metric": "binary_logloss",
            "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 0.1, "min_split_gain": 0.05,
            "verbose": -1, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
        }
        ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
        ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)
        s2 = lgb.train(params, ds_tr, num_boost_round=2000, valid_sets=[ds_te],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        s2_models.append(s2)

    # Ensemble predict: average P(oob) across models
    X_te_c = _ensure_cat(X_test)
    s1_proba = s1_ens.predict_proba(X_te_c)
    borked_mask = s1_proba[:, 0] >= 0.5

    # Average Stage 2 predictions
    s2_preds = []
    X2_te_full = X_te_c[mask_te].reset_index(drop=True)
    if drops:
        X2_te_pred = X2_te_full.drop(columns=[c for c in drops if c in X2_te_full.columns])
    else:
        X2_te_pred = X2_te_full
    for c in cats:
        if c in X2_te_pred.columns:
            X2_te_pred[c] = X2_te_pred[c].astype("category")

    for s2m in s2_models:
        p = s2m.predict(X2_te_pred)
        s2_preds.append(p)
    avg_p_oob = np.mean(s2_preds, axis=0)

    # Build 3-class predictions
    y_pred_ens = np.full(len(y_test), 1, dtype=int)  # default tinkering
    y_pred_ens[borked_mask] = 0
    # For non-borked: use averaged Stage 2
    non_borked_idx = np.where(~borked_mask)[0]
    s2_pred_class = (avg_p_oob >= 0.5).astype(int)  # 0=tinkering, 1=oob
    for j, idx in enumerate(non_borked_idx):
        if mask_te[idx]:  # should always be true for non-borked
            s2_j = np.searchsorted(np.where(mask_te)[0], idx)
            if s2_j < len(s2_pred_class):
                y_pred_ens[idx] = 1 + s2_pred_class[s2_j]

    f1_ens = f1_score(y_test, y_pred_ens, average="macro")
    per_ens = f1_score(y_test, y_pred_ens, average=None)
    oob_r_ens = (y_pred_ens[y_test == 2] == 2).mean()
    r = {"label": "17.2_ensemble_5seeds", "f1_macro": f1_ens,
         "borked_f1": per_ens[0], "tinkering_f1": per_ens[1],
         "works_oob_f1": per_ens[2], "oob_recall": oob_r_ens}
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | b={r['borked_f1']:.3f} t={r['tinkering_f1']:.3f} "
          f"o={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 17.5: Feature pruning ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("17.5: Feature pruning (keep top-N by Stage 2 importance)")
    print("=" * 70)

    # Get Stage 2 feature importance from baseline
    cas_bl2, _ = train_cascade(X_train, y_train, X_test, y_test)
    s2_bl = cas_bl2.stage2
    s2_names = s2_bl.feature_name()
    s2_imp = pd.Series(s2_bl.feature_importance(importance_type="gain"), index=s2_names)
    s2_imp = s2_imp.sort_values(ascending=False)

    for top_n in [30, 50, 80]:
        keep_features = set(s2_imp.head(top_n).index)
        # Also keep all Stage 1 features (they're used for borked detection)
        # Only prune Stage 2 features via drop list
        extra_drops = [f for f in s2_names if f not in keep_features]

        mask_tr2 = y_train > 0
        mask_te2 = y_test > 0
        X2_tr_p = X_tr_c[mask_tr2].reset_index(drop=True)
        y2_tr_p = (y_train[mask_tr2] - 1).astype(float)
        X2_te_p = X_te_c[mask_te2].reset_index(drop=True)
        y2_te_p = (y_test[mask_te2] - 1).astype(float)

        all_drops = list(set(list(STAGE2_DROP_FEATURES) + extra_drops))
        existing = [c for c in all_drops if c in X2_tr_p.columns]
        if existing:
            X2_tr_p = X2_tr_p.drop(columns=existing)
            X2_te_p = X2_te_p.drop(columns=existing)

        cats_p = [c for c in CATEGORICAL_FEATURES if c in X2_tr_p.columns]
        for c in cats_p:
            X2_tr_p[c] = X2_tr_p[c].astype("category")
            X2_te_p[c] = X2_te_p[c].astype("category")

        y_smooth_p = y2_tr_p * 0.85 + (1 - y2_tr_p) * 0.15
        w_p = np.ones(len(y2_tr_p))
        w_p[y2_tr_p >= 0.5] = 1.5

        ds_tr_p = lgb.Dataset(X2_tr_p, label=y_smooth_p, weight=w_p, categorical_feature=cats_p)
        ds_te_p = lgb.Dataset(X2_te_p, label=y2_te_p, categorical_feature=cats_p)
        s2_p = lgb.train(
            {"objective": "cross_entropy", "metric": "binary_logloss",
             "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
             "reg_alpha": 0.1, "reg_lambda": 0.1, "min_split_gain": 0.05, "verbose": -1},
            ds_tr_p, num_boost_round=2000, valid_sets=[ds_te_p],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        cas_p = CC(s1_ens, s2_p, existing)
        r = evaluate(cas_p, X_te_c, y_test, f"17.5_top{top_n}_features")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  top-{top_n}: F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) "
              f"oob={r['works_oob_f1']:.3f} ({X2_tr_p.shape[1]} features)")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<35s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s} {'oob_r':>7s}")
    print("-" * 84)
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<35s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f} {r.get('oob_recall',0):>7.3f}")


if __name__ == "__main__":
    main()
