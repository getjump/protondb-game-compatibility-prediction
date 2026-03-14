#!/usr/bin/env python3
"""Phase 11.2 experiment: Alternative models for Stage 2.

Tests CatBoost, XGBoost, and HistGradientBoosting as drop-in replacements
for LightGBM in Stage 2 (tinkering vs works_oob).

Stage 1 (borked vs works) stays LightGBM — shared across all experiments.
Only Stage 2 is swapped.

Hypothesis:
  - CatBoost: ordered boosting = implicit regularization for noisy labels,
    native categoricals without encoding loss
  - XGBoost: different split algorithm, may find different patterns
  - HistGBM: sklearn native, supports monotonic constraints
"""

import gc
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protondb_settings.ml.models.cascade import (
    STAGE2_DROP_FEATURES,
    train_stage1,
)
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

console = Console()

# LightGBM Stage 2 params (Phase 9.1 baseline)
S2_PARAMS_LGB = {
    "objective": "cross_entropy",
    "metric": "binary_logloss",
    "num_leaves": 63,
    "learning_rate": 0.02,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_split_gain": 0.05,
    "verbose": -1,
}


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _prepare_stage2(X_train, y_train, X_test, y_test, drop_features=None):
    """Filter to non-borked and prepare Stage 2 data."""
    if drop_features is None:
        drop_features = list(STAGE2_DROP_FEATURES)

    train_mask = y_train > 0
    test_mask = y_test > 0

    X_tr = X_train[train_mask].reset_index(drop=True)
    y_tr = (y_train[train_mask] - 1).astype(float)  # 0=tinkering, 1=oob

    X_te = X_test[test_mask].reset_index(drop=True)
    y_te = (y_test[test_mask] - 1).astype(float)

    drops = [c for c in drop_features if c in X_tr.columns]
    if drops:
        X_tr = X_tr.drop(columns=drops)
        X_te = X_te.drop(columns=drops)

    return X_tr, y_tr, X_te, y_te, drops


def _eval_cascade(s1_model, s2_predict_fn, s2_drops, X_eval, y_eval, label=""):
    """Evaluate cascade: Stage 1 (LightGBM) + Stage 2 (any model).

    s2_predict_fn: callable(X) -> P(works_oob) array
    """
    # Ensure categorical dtypes for Stage 1
    X_ev = X_eval.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_ev.columns:
            X_ev[col] = X_ev[col].astype("category")

    # Stage 1: borked vs works
    p_s1 = s1_model.predict_proba(X_ev)
    borked_mask = p_s1[:, 0] >= 0.5

    result = np.full(len(X_ev), 1, dtype=int)  # default tinkering
    result[borked_mask] = 0

    # Stage 2: tinkering vs works_oob (non-borked only)
    works_mask = ~borked_mask
    if works_mask.any():
        X_s2 = X_ev[works_mask].copy()
        drops = [c for c in s2_drops if c in X_s2.columns]
        if drops:
            X_s2 = X_s2.drop(columns=drops)
        p_oob = s2_predict_fn(X_s2)
        result[works_mask] = np.where(p_oob >= 0.5, 2, 1)

    f1 = f1_score(y_eval, result, average="macro")
    per_class = f1_score(y_eval, result, average=None, labels=[0, 1, 2])
    acc = accuracy_score(y_eval, result)

    return {
        "label": label,
        "f1_eval": round(f1, 4),
        "f1_borked": round(per_class[0], 3),
        "f1_tinkering": round(per_class[1], 3),
        "f1_works_oob": round(per_class[2], 3),
        "accuracy": round(acc, 4),
    }


# ═══════════════════════════════════════════════════════════════════
# Model trainers
# ═══════════════════════════════════════════════════════════════════

def train_s2_lightgbm(X_tr, y_tr, X_te, y_te, cat_cols, alpha=0.15):
    """Baseline LightGBM Stage 2 with label smoothing."""
    y_smooth = y_tr.copy()
    if alpha > 0:
        y_smooth = y_smooth * (1 - alpha) + (1 - y_smooth) * alpha

    for col in cat_cols:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype("category")
            X_te[col] = X_te[col].astype("category")

    ds_train = lgb.Dataset(X_tr, label=y_smooth, categorical_feature=cat_cols)
    ds_test = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_cols)

    model = lgb.train(
        S2_PARAMS_LGB, ds_train,
        num_boost_round=3000,
        valid_sets=[ds_test],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    console.print(f"    LightGBM: {model.best_iteration} iters")

    def predict_fn(X):
        return model.predict(X)
    return predict_fn


def train_s2_catboost(X_tr, y_tr, X_te, y_te, cat_cols, alpha=0.15,
                      variant="default"):
    """CatBoost Stage 2.

    Variants:
      - default: ordered boosting, native categoricals
      - noise_robust: higher l2_leaf_reg, deeper min_data_in_leaf
      - label_smooth: use label_smoothing param (CatBoost native)
    """
    from catboost import CatBoostClassifier, Pool

    # CatBoost handles categoricals natively — convert to string
    X_tr_cb = X_tr.copy()
    X_te_cb = X_te.copy()
    cb_cat_indices = []
    for col in cat_cols:
        if col in X_tr_cb.columns:
            X_tr_cb[col] = X_tr_cb[col].astype(str).fillna("__nan__")
            X_te_cb[col] = X_te_cb[col].astype(str).fillna("__nan__")
            cb_cat_indices.append(X_tr_cb.columns.get_loc(col))

    # Apply label smoothing to targets (same approach as LightGBM baseline)
    y_tr_int = y_tr.astype(int)
    y_te_int = y_te.astype(int)

    params = {
        "iterations": 3000,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 0,
        "eval_metric": "Logloss",
        "early_stopping_rounds": 100,
        "task_type": "CPU",
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.0,
    }

    if variant == "noise_robust":
        params.update({
            "l2_leaf_reg": 10.0,
            "min_data_in_leaf": 50,
            "random_strength": 2.0,
            "model_size_reg": 0.5,
        })
    elif variant == "deeper":
        params.update({
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "min_data_in_leaf": 30,
        })
    elif variant == "langevin":
        params.update({
            "langevin": True,
            "diffusion_temperature": 10000,
            "l2_leaf_reg": 5.0,
        })

    train_pool = Pool(X_tr_cb, label=y_tr_int, cat_features=cb_cat_indices)
    eval_pool = Pool(X_te_cb, label=y_te_int, cat_features=cb_cat_indices)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=eval_pool)
    best_iter = model.get_best_iteration()
    console.print(f"    CatBoost ({variant}): {best_iter} iters")

    def predict_fn(X):
        X_pred = X.copy()
        for col in cat_cols:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype(str).fillna("__nan__")
        return model.predict_proba(X_pred)[:, 1]
    return predict_fn


def train_s2_xgboost(X_tr, y_tr, X_te, y_te, cat_cols, alpha=0.15,
                     variant="default"):
    """XGBoost Stage 2."""
    import xgboost as xgb

    # XGBoost: encode categoricals as int (enable_categorical in newer versions)
    X_tr_xg = X_tr.copy()
    X_te_xg = X_te.copy()
    for col in cat_cols:
        if col in X_tr_xg.columns:
            X_tr_xg[col] = X_tr_xg[col].astype("category").cat.codes
            X_te_xg[col] = X_te_xg[col].astype("category").cat.codes

    # Apply label smoothing
    y_smooth = y_tr.copy()
    if alpha > 0:
        y_smooth = y_smooth * (1 - alpha) + (1 - y_smooth) * alpha

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.02,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "seed": 42,
        "verbosity": 0,
        "tree_method": "hist",
    }

    num_rounds = 3000
    if variant == "dart":
        params["booster"] = "dart"
        params["rate_drop"] = 0.1
        params["skip_drop"] = 0.5
        # DART is extremely slow — limit iterations
        num_rounds = 500
    elif variant == "deep":
        params["max_depth"] = 8
        params["min_child_weight"] = 30
        params["max_leaves"] = 63

    dtrain = xgb.DMatrix(X_tr_xg, label=y_smooth)
    dtest = xgb.DMatrix(X_te_xg, label=y_te)

    model = xgb.train(
        params, dtrain,
        num_boost_round=num_rounds,
        evals=[(dtest, "eval")],
        early_stopping_rounds=100,
        verbose_eval=500,
    )
    console.print(f"    XGBoost ({variant}): {model.best_iteration} iters")

    def predict_fn(X):
        X_pred = X.copy()
        for col in cat_cols:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category").cat.codes
        return model.predict(xgb.DMatrix(X_pred))
    return predict_fn


def train_s2_histgbm(X_tr, y_tr, X_te, y_te, cat_cols, alpha=0.15,
                     variant="default"):
    """sklearn HistGradientBoostingClassifier Stage 2."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    # HistGBM: mark categoricals via categorical_features parameter
    X_tr_h = X_tr.copy()
    X_te_h = X_te.copy()
    cat_mask = np.zeros(X_tr_h.shape[1], dtype=bool)
    for col in cat_cols:
        if col in X_tr_h.columns:
            idx = X_tr_h.columns.get_loc(col)
            cat_mask[idx] = True
            X_tr_h[col] = X_tr_h[col].astype("category").cat.codes.astype(float)
            X_te_h[col] = X_te_h[col].astype("category").cat.codes.astype(float)

    # Fill NaN for HistGBM (it doesn't handle NaN in categoricals well)
    X_tr_h = X_tr_h.fillna(-1)
    X_te_h = X_te_h.fillna(-1)

    # Convert soft labels to hard for HistGBM (doesn't support float targets for classification)
    y_tr_int = y_tr.round().astype(int)
    y_te_int = y_te.round().astype(int)

    params = {
        "max_iter": 3000,
        "learning_rate": 0.02,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 50,
        "max_depth": None,
        "l2_regularization": 0.1,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": None,
        "n_iter_no_change": 100,
        "verbose": 0,
        "categorical_features": cat_mask if cat_mask.any() else None,
    }

    if variant == "monotonic":
        # Domain knowledge: more reports = more reliable signal
        # (only set constraints for features we're confident about)
        params["monotonic_cst"] = None  # would need per-feature array

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_tr_h, y_tr_int, sample_weight=None)
    n_iters = model.n_iter_
    console.print(f"    HistGBM ({variant}): {n_iters} iters")

    def predict_fn(X):
        X_pred = X.copy()
        for col in cat_cols:
            if col in X_pred.columns:
                X_pred[col] = X_pred[col].astype("category").cat.codes.astype(float)
        X_pred = X_pred.fillna(-1)
        return model.predict_proba(X_pred)[:, 1]
    return predict_fn


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.noise import find_noisy_samples
    from protondb_settings.ml.relabeling import apply_relabeling, get_relabel_ids
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split

    db_path = Path("data/protondb.db")
    emb_path = Path("data/embeddings.npz")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── Load data ──
    console.print("[bold]Loading cached embeddings...[/bold]")
    emb_data = load_embeddings(emb_path)
    console.print(f"  GPU: {emb_data.get('n_components_gpu', '?')} dims, "
                  f"Text: {emb_data.get('text_n_components', 0)} dims")

    from protondb_settings.config import NORMALIZED_DATA_SOURCE

    console.print("[bold]Building feature matrix...[/bold]")
    X, y, timestamps, report_ids, label_maps = _build_feature_matrix(
        conn, emb_data, normalized_data_source=NORMALIZED_DATA_SOURCE,
    )
    console.print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    del emb_data
    gc.collect()

    # ── Relabeling ──
    relabel_ids = get_relabel_ids(conn)

    # ── Split ──
    console.print("[bold]Time-based split...[/bold]")
    X_full, X_test_full, y_train_full, y_test_full, train_rids, _ = _time_based_split(
        X, y, timestamps, test_fraction=0.2, report_ids=report_ids,
    )

    del X, y, timestamps, report_ids
    gc.collect()

    y_train_full, n_relabeled = apply_relabeling(y_train_full, train_rids, relabel_ids)
    console.print(f"  Relabeled {n_relabeled}")

    # ── Cleanlab noise removal ──
    console.print("[bold]Cleanlab noise removal (3%)...[/bold]")
    keep_mask = find_noisy_samples(X_full, y_train_full, frac_remove=0.03,
                                   cache_dir="data/")
    n_removed = (~keep_mask).sum()
    X_train = X_full[keep_mask].reset_index(drop=True)
    y_train = y_train_full[keep_mask]
    console.print(f"  Removed {n_removed} noisy samples")

    del X_full, y_train_full, keep_mask
    gc.collect()

    # Split test into ES + eval
    n_cal = len(X_test_full) // 2
    X_test_es = X_test_full.iloc[:n_cal].copy().reset_index(drop=True)
    y_test_es = y_test_full[:n_cal]
    X_eval = X_test_full.iloc[n_cal:].copy().reset_index(drop=True)
    y_eval = y_test_full[n_cal:]

    del X_test_full, y_test_full
    gc.collect()

    console.print(f"  Train: {len(X_train)}, ES: {len(X_test_es)}, Eval: {len(X_eval)}")

    # Ensure categorical dtypes
    for col in CATEGORICAL_FEATURES:
        for df in [X_train, X_test_es, X_eval]:
            if col in df.columns:
                df[col] = df[col].astype("category")

    # ── Train shared Stage 1 (LightGBM, same for all experiments) ──
    console.print("\n[bold cyan]Training shared Stage 1 (LightGBM)...[/bold cyan]")
    t0 = time.time()
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es)
    console.print(f"  Stage 1: {s1.best_iteration_} iters ({time.time()-t0:.0f}s)")

    # ── Prepare Stage 2 data ──
    X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, s2_drops = _prepare_stage2(
        X_train, y_train, X_test_es, y_test_es,
    )
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr_s2.columns]
    console.print(f"  Stage 2 data: {len(X_tr_s2)} train, {len(X_te_s2)} ES, "
                  f"{X_tr_s2.shape[1]} features, {len(cat_cols)} categorical")

    all_results = []

    # ═══════════════════════════════════════════════════════════════
    # Baseline: LightGBM Stage 2
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]Baseline: LightGBM Stage 2[/bold cyan]")
    t0 = time.time()
    lgb_fn = train_s2_lightgbm(X_tr_s2.copy(), y_tr_s2.copy(),
                                X_te_s2.copy(), y_te_s2.copy(), cat_cols)
    r = _eval_cascade(s1, lgb_fn, s2_drops, X_eval, y_eval, "LightGBM (baseline)")
    r["time_s"] = round(time.time() - t0, 1)
    all_results.append(r)
    console.print(f"  F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f} ({r['time_s']}s)")

    # ═══════════════════════════════════════════════════════════════
    # R4: CatBoost variants
    # ═══════════════════════════════════════════════════════════════
    for cb_variant in ["default", "noise_robust", "deeper", "langevin"]:
        console.print(f"\n[bold cyan]R4: CatBoost ({cb_variant})[/bold cyan]")
        t0 = time.time()
        try:
            cb_fn = train_s2_catboost(X_tr_s2.copy(), y_tr_s2.copy(),
                                       X_te_s2.copy(), y_te_s2.copy(), cat_cols,
                                       variant=cb_variant)
            r = _eval_cascade(s1, cb_fn, s2_drops, X_eval, y_eval,
                              f"CatBoost ({cb_variant})")
            r["time_s"] = round(time.time() - t0, 1)
            all_results.append(r)
            console.print(f"  F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f} ({r['time_s']}s)")
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # R5a: XGBoost variants
    # ═══════════════════════════════════════════════════════════════
    for xgb_variant in ["default", "deep"]:
        console.print(f"\n[bold cyan]R5a: XGBoost ({xgb_variant})[/bold cyan]")
        t0 = time.time()
        try:
            xgb_fn = train_s2_xgboost(X_tr_s2.copy(), y_tr_s2.copy(),
                                       X_te_s2.copy(), y_te_s2.copy(), cat_cols,
                                       variant=xgb_variant)
            r = _eval_cascade(s1, xgb_fn, s2_drops, X_eval, y_eval,
                              f"XGBoost ({xgb_variant})")
            r["time_s"] = round(time.time() - t0, 1)
            all_results.append(r)
            console.print(f"  F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f} ({r['time_s']}s)")
        except Exception as e:
            console.print(f"  [red]FAILED: {e}[/red]")
        gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # R5b: HistGradientBoosting
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]R5b: HistGradientBoosting[/bold cyan]")
    t0 = time.time()
    try:
        hgb_fn = train_s2_histgbm(X_tr_s2.copy(), y_tr_s2.copy(),
                                   X_te_s2.copy(), y_te_s2.copy(), cat_cols)
        r = _eval_cascade(s1, hgb_fn, s2_drops, X_eval, y_eval,
                          "HistGBM (default)")
        r["time_s"] = round(time.time() - t0, 1)
        all_results.append(r)
        console.print(f"  F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f} ({r['time_s']}s)")
    except Exception as e:
        console.print(f"  [red]FAILED: {e}[/red]")

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    console.print("\n")
    table = Table(title="Phase 11.2: Alternative Models for Stage 2")
    table.add_column("Experiment", style="cyan")
    table.add_column("F1 eval", justify="right")
    table.add_column("ΔF1", justify="right")
    table.add_column("borked", justify="right")
    table.add_column("tinkering", justify="right")
    table.add_column("works_oob", justify="right")
    table.add_column("accuracy", justify="right")
    table.add_column("time", justify="right")

    baseline_f1 = all_results[0]["f1_eval"]
    for r in all_results:
        delta = r["f1_eval"] - baseline_f1
        delta_str = f"{delta:+.4f}" if r["label"] != "LightGBM (baseline)" else "—"
        style = "green" if delta > 0.002 else ("red" if delta < -0.002 else "")
        table.add_row(
            r["label"],
            f"{r['f1_eval']:.4f}",
            delta_str,
            f"{r['f1_borked']:.3f}",
            f"{r['f1_tinkering']:.3f}",
            f"{r['f1_works_oob']:.3f}",
            f"{r['accuracy']:.4f}",
            f"{r.get('time_s', '?')}s",
            style=style,
        )

    console.print(table)
    conn.close()


if __name__ == "__main__":
    main()
