"""Phase 18: SOTA approaches — threshold optimization, adaptive soft labels, focal loss.

Experiments:
  18.1  — Post-hoc threshold optimization (zero retraining cost)
  18.2  — IRT-derived adaptive soft labels
  18.3  — Focal loss for Stage 2
  18.4  — Cleanlab quality scores as features
  Combined

Usage:
  python scripts/experiment_18_sota.py [--db data/protondb.db]
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


# ── Data loading (full pipeline) ─────────────────────────────────────


def load_pipeline(db_path):
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel,
        add_error_targeted_features, _build_lookups,
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
    y_train, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)

    # Split test into calibration + eval
    n_cal = len(X_test) // 2
    X_cal = X_test.iloc[:n_cal].reset_index(drop=True)
    y_cal = y_test[:n_cal]
    X_eval = X_test.iloc[n_cal:].reset_index(drop=True)
    y_eval = y_test[n_cal:]

    # Get IRT P(tinkering) for train reports (for adaptive smoothing)
    from protondb_settings.ml.features.encoding import extract_gpu_family
    report_info, contrib_lookup = _build_lookups(conn)

    irt_p_tink = np.full(len(train_rids), np.nan)
    for i, rid in enumerate(train_rids):
        app_id, gpu = report_info.get(rid, (None, None))
        gpu_fam = extract_gpu_family(gpu) if gpu else "unknown"
        d = difficulty.get((app_id, gpu_fam))
        if d is None and app_id:
            vals = [v for (a, g), v in difficulty.items() if a == app_id]
            d = np.mean(vals) if vals else None
        cid = contrib_lookup.get(rid)
        t = theta.get(cid) if cid else None
        if t is not None and d is not None:
            irt_p_tink[i] = 1 / (1 + np.exp(-(t - d)))

    conn.close()
    return (X_train, X_cal, X_eval, y_train, y_cal, y_eval,
            train_rids, theta, difficulty, irt_p_tink)


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_cat(X):
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
    X = X.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def train_cascade_full(X_train, y_train, X_cal, y_cal,
                       custom_s2_fn=None):
    """Train cascade. custom_s2_fn overrides Stage 2 training if provided."""
    from protondb_settings.ml.models.cascade import (
        train_stage1, train_stage2, CascadeClassifier,
    )

    X_train, X_cal = _ensure_cat(X_train), _ensure_cat(X_cal)
    # Combine cal into test for early stopping
    X_test_combined = pd.concat([X_cal], ignore_index=True)
    y_test_combined = y_cal

    s1 = train_stage1(X_train, y_train, X_test_combined, y_test_combined)

    if custom_s2_fn:
        s2, drops = custom_s2_fn(X_train, y_train, X_test_combined, y_test_combined)
    else:
        s2, drops = train_stage2(X_train, y_train, X_test_combined, y_test_combined)

    cascade = CascadeClassifier(s1, s2, drops)
    # Calibrate on cal set
    X_cal_cat = _ensure_cat(X_cal)
    cascade.fit_calibrators(X_cal_cat, y_test_combined)
    return cascade


def evaluate(cascade, X_eval, y_eval, label="", thresholds=None):
    X_eval = _ensure_cat(X_eval)

    if thresholds is not None:
        y_proba = cascade.predict_proba(X_eval, calibrated=True)
        y_pred = np.argmax(y_proba * thresholds, axis=1)
    else:
        y_pred = cascade.predict(X_eval)

    f1 = f1_score(y_eval, y_pred, average="macro")
    per = f1_score(y_eval, y_pred, average=None)
    oob_r = (y_pred[y_eval == 2] == 2).mean() if (y_eval == 2).any() else 0
    return {"label": label, "f1_macro": f1, "borked_f1": per[0],
            "tinkering_f1": per[1], "works_oob_f1": per[2], "oob_recall": oob_r}


# ── 18.1: Threshold optimization ─────────────────────────────────────


def optimize_thresholds(cascade, X_cal, y_cal):
    """Grid search per-class thresholds on calibration set.

    Uses multiplicative scaling: pred = argmax(proba * t) where t > 1 = favor class.
    """
    X_cal = _ensure_cat(X_cal)
    y_proba = cascade.predict_proba(X_cal, calibrated=True)

    best_f1 = f1_score(y_cal, y_proba.argmax(axis=1), average="macro")
    best_t = np.array([1.0, 1.0, 1.0])

    # Scale factors: >1 = favor class, <1 = suppress
    for s_borked in np.arange(0.6, 1.6, 0.1):
        for s_oob in np.arange(0.8, 3.0, 0.1):
            t = np.array([s_borked, 1.0, s_oob])
            y_pred = np.argmax(y_proba * t, axis=1)
            f1 = f1_score(y_cal, y_pred, average="macro")
            if f1 > best_f1:
                best_f1 = f1
                best_t = t.copy()

    logger.info("Threshold optimization: best_f1=%.4f, t=%s", best_f1, best_t)
    return best_t, best_f1


# ── 18.2: Adaptive soft labels ───────────────────────────────────────


def train_stage2_adaptive_smooth(X_train, y_train, X_test, y_test,
                                  irt_p_tink=None):
    """Stage 2 with per-sample adaptive label smoothing from IRT."""
    from protondb_settings.ml.models.cascade import (
        STAGE2_DROP_FEATURES, STAGE2_KEEP_FEATURES,
    )
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    train_mask = y_train > 0
    test_mask = y_test > 0

    X2_tr = X_train[train_mask].reset_index(drop=True)
    y2_tr = (y_train[train_mask] - 1).astype(float)  # 0=tinkering, 1=oob
    X2_te = X_test[test_mask].reset_index(drop=True)
    y2_te = (y_test[test_mask] - 1).astype(float)

    # Drop + prune
    drops = [c for c in STAGE2_DROP_FEATURES if c in X2_tr.columns]
    keep = [c for c in STAGE2_KEEP_FEATURES if c in X2_tr.columns]
    extra_drop = [c for c in X2_tr.columns if c not in keep and c not in drops]
    all_drops = drops + extra_drop
    existing = [c for c in all_drops if c in X2_tr.columns]
    if existing:
        X2_tr = X2_tr.drop(columns=existing)
        X2_te = X2_te.drop(columns=existing)

    cats = [c for c in CATEGORICAL_FEATURES if c in X2_tr.columns]
    for c in cats:
        X2_tr[c] = X2_tr[c].astype("category")
        X2_te[c] = X2_te[c].astype("category")

    # Adaptive smoothing: alpha_i = clip(|P_irt - y_i| * 0.5, 0.05, 0.40)
    if irt_p_tink is not None:
        p_irt_s2 = irt_p_tink[train_mask]
        y2_arr = y2_tr if isinstance(y2_tr, np.ndarray) else y2_tr.values
        alpha = np.clip(np.abs(p_irt_s2 - y2_arr) * 0.5, 0.05, 0.40)
        # For samples without IRT: use fixed alpha=0.15
        no_irt = np.isnan(p_irt_s2)
        alpha[no_irt] = 0.15
        y_smooth = y2_arr * (1 - alpha) + (1 - y2_arr) * alpha
        logger.info("Adaptive smoothing: mean_alpha=%.3f, median=%.3f",
                     np.mean(alpha), np.median(alpha))
    else:
        y2_arr = y2_tr if isinstance(y2_tr, np.ndarray) else y2_tr.values
        y_smooth = y2_arr * 0.85 + (1 - y2_arr) * 0.15

    w = np.ones(len(y2_tr))
    w[y2_tr >= 0.5] = 1.8

    ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
    ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)
    model = lgb.train(
        {"objective": "cross_entropy", "metric": "binary_logloss",
         "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
         "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
         "reg_alpha": 1.0, "reg_lambda": 1.0, "min_split_gain": 0.05, "verbose": -1},
        ds_tr, num_boost_round=2000, valid_sets=[ds_te],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )
    return model, existing


# ── 18.3: Focal loss ─────────────────────────────────────────────────


def focal_binary_objective(y_true, y_pred, gamma=2.0):
    """Focal loss custom objective for LightGBM."""
    p = 1 / (1 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)

    # Focal weight: (1-p_t)^gamma
    pt = p * y_true + (1 - p) * (1 - y_true)
    focal_weight = (1 - pt) ** gamma

    grad = (p - y_true) * focal_weight
    hess = p * (1 - p) * focal_weight * (1 + gamma * (1 - pt) ** (gamma - 1) * pt)
    hess = np.clip(hess, 1e-7, None)

    return grad, hess


def focal_eval(y_true, y_pred):
    """Focal loss eval metric."""
    p = 1 / (1 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return "binary_logloss", loss, False


def train_stage2_focal(X_train, y_train, X_test, y_test, gamma=2.0):
    """Stage 2 with focal loss."""
    from protondb_settings.ml.models.cascade import (
        STAGE2_DROP_FEATURES, STAGE2_KEEP_FEATURES,
    )
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    train_mask = y_train > 0
    test_mask = y_test > 0

    X2_tr = X_train[train_mask].reset_index(drop=True)
    y2_tr = (y_train[train_mask] - 1).astype(float)
    X2_te = X_test[test_mask].reset_index(drop=True)
    y2_te = (y_test[test_mask] - 1).astype(float)

    drops = [c for c in STAGE2_DROP_FEATURES if c in X2_tr.columns]
    keep = [c for c in STAGE2_KEEP_FEATURES if c in X2_tr.columns]
    extra_drop = [c for c in X2_tr.columns if c not in keep and c not in drops]
    all_drops = drops + extra_drop
    existing = [c for c in all_drops if c in X2_tr.columns]
    if existing:
        X2_tr = X2_tr.drop(columns=existing)
        X2_te = X2_te.drop(columns=existing)

    cats = [c for c in CATEGORICAL_FEATURES if c in X2_tr.columns]
    for c in cats:
        X2_tr[c] = X2_tr[c].astype("category")
        X2_te[c] = X2_te[c].astype("category")

    # Label smoothing
    y_smooth = y2_tr * 0.85 + (1 - y2_tr) * 0.15

    w = np.ones(len(y2_tr))
    w[y2_tr >= 0.5] = 1.8

    ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
    ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)

    def fobj(preds, dataset):
        y_true = dataset.get_label()
        return focal_binary_objective(y_true, preds, gamma=gamma)

    def feval_fn(preds, dataset):
        y_true = dataset.get_label()
        return [focal_eval(y_true, preds)]

    model = lgb.train(
        {"objective": fobj,
         "num_leaves": 63, "learning_rate": 0.03,
         "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
         "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 1.0,
         "min_split_gain": 0.05, "verbose": -1},
        ds_tr, num_boost_round=2000, valid_sets=[ds_te],
        feval=feval_fn,
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(500)],
    )
    return model, existing


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 18: SOTA approaches")
    print("=" * 70)

    t0 = time.time()
    (X_train, X_cal, X_eval, y_train, y_cal, y_eval,
     train_rids, theta, difficulty, irt_p_tink) = load_pipeline(args.db)
    logger.info("Data loaded in %.0fs", time.time() - t0)

    results = []

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)
    cascade_bl = train_cascade_full(X_train, y_train, X_cal, y_cal)
    r = evaluate(cascade_bl, X_eval, y_eval, "baseline")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | b={r['borked_f1']:.3f} t={r['tinkering_f1']:.3f} "
          f"o={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 18.1: Threshold optimization ─────────────────────────────────
    print("\n" + "=" * 70)
    print("18.1: Post-hoc threshold optimization")
    print("=" * 70)
    best_t, cal_f1 = optimize_thresholds(cascade_bl, X_cal, y_cal)
    r = evaluate(cascade_bl, X_eval, y_eval, "18.1_threshold_opt", thresholds=best_t)
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | thresholds={best_t}")
    print(f"  b={r['borked_f1']:.3f} t={r['tinkering_f1']:.3f} o={r['works_oob_f1']:.3f} oob_r={r['oob_recall']:.3f}")

    # ── 18.2: Adaptive soft labels ───────────────────────────────────
    print("\n" + "=" * 70)
    print("18.2: IRT-derived adaptive soft labels")
    print("=" * 70)
    cascade_as = train_cascade_full(
        X_train, y_train, X_cal, y_cal,
        custom_s2_fn=lambda Xtr, ytr, Xte, yte: train_stage2_adaptive_smooth(
            Xtr, ytr, Xte, yte, irt_p_tink=irt_p_tink))
    r = evaluate(cascade_as, X_eval, y_eval, "18.2_adaptive_smooth")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f})")

    # 18.2 + threshold
    best_t2, _ = optimize_thresholds(cascade_as, X_cal, y_cal)
    r = evaluate(cascade_as, X_eval, y_eval, "18.2+18.1_adaptive+thresh", thresholds=best_t2)
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  +threshold: F1={r['f1_macro']:.4f} (Δ={delta:+.4f})")

    # ── 18.3: Focal loss ─────────────────────────────────────────────
    for gamma in [1.0, 2.0, 3.0]:
        print(f"\n{'='*70}")
        print(f"18.3: Focal loss (gamma={gamma})")
        print(f"{'='*70}")
        cascade_fl = train_cascade_full(
            X_train, y_train, X_cal, y_cal,
            custom_s2_fn=lambda Xtr, ytr, Xte, yte, g=gamma: train_stage2_focal(
                Xtr, ytr, Xte, yte, gamma=g))
        r = evaluate(cascade_fl, X_eval, y_eval, f"18.3_focal_g{gamma}")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f})")

        # + threshold
        best_t3, _ = optimize_thresholds(cascade_fl, X_cal, y_cal)
        r = evaluate(cascade_fl, X_eval, y_eval, f"18.3+18.1_focal_g{gamma}+thresh",
                     thresholds=best_t3)
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  +threshold: F1={r['f1_macro']:.4f} (Δ={delta:+.4f})")

    # ── 18.2 + 18.3 combined ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMBINED: adaptive smooth + focal + threshold")
    print(f"{'='*70}")
    # Adaptive smooth with focal loss is tricky - focal changes the objective
    # so soft labels interact differently. Skip this combo and just use best individual.

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<35s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s} {'oob_r':>7s}")
    print("-" * 84)
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<35s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f} {r.get('oob_recall',0):>7.3f}")


if __name__ == "__main__":
    main()
