#!/usr/bin/env python3
"""Phase 9.5 experiment: Advanced features.

Tests new features added ON TOP of existing 119-feature baseline.
Each experiment adds features to the existing feature matrix and
evaluates the cascade pipeline.

Experiments:
  P18 — ProtonDB tier/score as features
  P11 — Hierarchical target encoding for app_id (app→engine→global)
  P20 — Temporal decay tinkering rate per (app_id, variant)
  P19 — Cross-entity conditional statistics (gpu×engine, variant×engine)
  ALL — Combined best features
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
    CascadeClassifier,
    STAGE2_DROP_FEATURES,
    train_stage1,
)
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

console = Console()

# Stage 2 params (Phase 9.1)
S2_PARAMS = {
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
    y_tr = (y_train[train_mask] - 1).astype(float)

    X_te = X_test[test_mask].reset_index(drop=True)
    y_te = (y_test[test_mask] - 1).astype(float)

    drops = [c for c in drop_features if c in X_tr.columns]
    if drops:
        X_tr = X_tr.drop(columns=drops)
        X_te = X_te.drop(columns=drops)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")

    return X_tr, y_tr, X_te, y_te, cat_cols, drops


def _train_s2_booster(X_tr, y_tr_labels, X_te, y_te, cat_cols, alpha=0.15):
    """Train Stage 2 Booster with label smoothing."""
    params = dict(S2_PARAMS)
    y_smooth = y_tr_labels.copy()
    if alpha > 0:
        y_smooth = y_smooth * (1 - alpha) + (1 - y_smooth) * alpha

    ds_train = lgb.Dataset(X_tr, label=y_smooth, categorical_feature=cat_cols)
    ds_test = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_cols)

    model = lgb.train(
        params, ds_train,
        num_boost_round=3000,
        valid_sets=[ds_test],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    return model


def run_cascade_eval(s1, s2, s2_dropped, X_eval, y_eval, label=""):
    """Evaluate cascade and return metrics dict."""
    cascade = CascadeClassifier(s1, s2, s2_dropped)

    for col in CATEGORICAL_FEATURES:
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype("category")

    preds = cascade.predict(X_eval)
    f1 = f1_score(y_eval, preds, average="macro")
    per_class = f1_score(y_eval, preds, average=None, labels=[0, 1, 2])
    acc = accuracy_score(y_eval, preds)

    return {
        "label": label,
        "f1_eval": round(f1, 4),
        "f1_borked": round(per_class[0], 3),
        "f1_tinkering": round(per_class[1], 3),
        "f1_works_oob": round(per_class[2], 3),
        "accuracy": round(acc, 4),
    }


def run_full_cascade(s1_shared, X_train, y_train, X_test_es, y_test_es,
                     X_eval, y_eval, label=""):
    """Train Stage 2 on given features and evaluate full cascade."""
    X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols, drops = _prepare_stage2(
        X_train, y_train, X_test_es, y_test_es
    )
    s2 = _train_s2_booster(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols)
    r = run_cascade_eval(s1_shared, s2, drops, X_eval, y_eval, label)
    del s2
    gc.collect()
    return r


# ═══════════════════════════════════════════════════════════════════
# Feature builders
# ═══════════════════════════════════════════════════════════════════

def build_p18_features(conn, app_ids: pd.Series) -> pd.DataFrame:
    """P18: ProtonDB tier + score as features.

    protondb_tier → ordinal (borked=0, bronze=1, silver=2, gold=3, platinum=4)
    protondb_score → continuous [0..1]
    """
    tier_map = {"borked": 0, "bronze": 1, "silver": 2, "gold": 3, "platinum": 4}

    rows = conn.execute(
        "SELECT app_id, protondb_tier, protondb_score FROM game_metadata "
        "WHERE protondb_tier IS NOT NULL OR protondb_score IS NOT NULL"
    ).fetchall()

    lookup = {}
    for r in rows:
        tier_val = tier_map.get(r["protondb_tier"])
        score = r["protondb_score"]
        lookup[r["app_id"]] = {"protondb_tier_ord": tier_val, "protondb_score": score}

    df = pd.DataFrame.from_dict(lookup, orient="index")
    result = df.reindex(app_ids.values)
    result.index = app_ids.index

    n_mapped = result["protondb_tier_ord"].notna().sum()
    console.print(f"    P18: mapped {n_mapped}/{len(app_ids)} ({n_mapped/len(app_ids)*100:.1f}%)")
    return result


def _compute_te_rates(app_ids_arr, y_arr, engine_map, smoothing_m=20.0):
    """Compute hierarchical target encoding rates for a set of samples.

    Returns: (app_oob_dict, app_tink_dict, engine_stats, global_oob, global_tink)
    """
    global_oob = (y_arr == 2).mean()
    global_tink = (y_arr == 1).mean()

    engine_labels = np.array([engine_map.get(a, "__none__") for a in app_ids_arr])
    engine_stats = {}
    for eng in np.unique(engine_labels):
        mask = engine_labels == eng
        n = mask.sum()
        if n >= 5:
            engine_stats[eng] = {
                "oob_rate": (y_arr[mask] == 2).mean(),
                "tinkering_rate": (y_arr[mask] == 1).mean(),
                "n": n,
            }

    app_oob = {}
    app_tink = {}
    for app_id in set(app_ids_arr):
        mask = app_ids_arr == app_id
        n = mask.sum()
        oob_count = (y_arr[mask] == 2).sum()
        tink_count = (y_arr[mask] == 1).sum()

        eng = engine_map.get(app_id, "__none__")
        eng_stat = engine_stats.get(eng)
        if eng_stat and eng_stat["n"] >= 10:
            prior_oob = eng_stat["oob_rate"]
            prior_tink = eng_stat["tinkering_rate"]
        else:
            prior_oob = global_oob
            prior_tink = global_tink

        app_oob[app_id] = (oob_count + smoothing_m * prior_oob) / (n + smoothing_m)
        app_tink[app_id] = (tink_count + smoothing_m * prior_tink) / (n + smoothing_m)

    return app_oob, app_tink, engine_stats, global_oob, global_tink


def _map_te_rates(app_ids, app_oob, app_tink, engine_map, engine_stats, global_oob, global_tink):
    """Map precomputed rates to a Series of app_ids, with fallbacks."""
    te_oob = app_ids.map(app_oob).astype(float)
    te_tink = app_ids.map(app_tink).astype(float)

    unmapped = te_oob.isna()
    if unmapped.any():
        for idx in app_ids[unmapped].index:
            app_id = app_ids[idx]
            eng = engine_map.get(app_id, "__none__")
            eng_stat = engine_stats.get(eng)
            if eng_stat:
                te_oob.at[idx] = eng_stat["oob_rate"]
                te_tink.at[idx] = eng_stat["tinkering_rate"]
            else:
                te_oob.at[idx] = global_oob
                te_tink.at[idx] = global_tink

    return te_oob.values, te_tink.values


def build_p11_features(
    conn, app_ids: pd.Series, y: np.ndarray,
    train_mask: np.ndarray, smoothing_m: float = 20.0,
    n_folds: int = 5,
) -> pd.DataFrame:
    """P11: Hierarchical target encoding for app_id.

    Hierarchy: app_id → engine → global mean.
    Uses 5-fold OOF encoding on training set to prevent leakage.
    Test/eval rows use full training set statistics.

    Returns: DataFrame with 'te_app_oob_rate', 'te_app_tinkering_rate'
    """
    from sklearn.model_selection import KFold

    engine_rows = conn.execute(
        "SELECT app_id, engine FROM game_metadata WHERE engine IS NOT NULL"
    ).fetchall()
    engine_map = {r["app_id"]: r["engine"].strip().split(",")[0].strip().lower()
                  for r in engine_rows if r["engine"]}

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    train_app_ids = app_ids.iloc[train_idx].values
    train_y = y[train_idx]

    # OOF encoding for training data
    te_oob_arr = np.full(len(app_ids), np.nan)
    te_tink_arr = np.full(len(app_ids), np.nan)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_train, fold_val in kf.split(train_idx):
        # Compute rates from fold_train, apply to fold_val
        fold_apps = train_app_ids[fold_train]
        fold_y = train_y[fold_train]
        app_oob, app_tink, eng_stats, g_oob, g_tink = _compute_te_rates(
            fold_apps, fold_y, engine_map, smoothing_m
        )
        val_global_idx = train_idx[fold_val]
        val_app_ids = app_ids.iloc[val_global_idx]
        oob_vals, tink_vals = _map_te_rates(
            val_app_ids, app_oob, app_tink, engine_map, eng_stats, g_oob, g_tink
        )
        te_oob_arr[val_global_idx] = oob_vals
        te_tink_arr[val_global_idx] = tink_vals

    # Full training rates for test/eval
    app_oob_full, app_tink_full, eng_stats_full, g_oob_full, g_tink_full = _compute_te_rates(
        train_app_ids, train_y, engine_map, smoothing_m
    )
    if len(test_idx) > 0:
        test_app_ids = app_ids.iloc[test_idx]
        oob_vals, tink_vals = _map_te_rates(
            test_app_ids, app_oob_full, app_tink_full, engine_map,
            eng_stats_full, g_oob_full, g_tink_full
        )
        te_oob_arr[test_idx] = oob_vals
        te_tink_arr[test_idx] = tink_vals

    result = pd.DataFrame({
        "te_app_oob_rate": te_oob_arr,
        "te_app_tinkering_rate": te_tink_arr,
    }, index=app_ids.index)

    console.print(f"    P11: global oob={g_oob_full:.3f}, tink={g_tink_full:.3f}, "
                  f"engines={len(eng_stats_full)}, apps={len(app_oob_full)}, "
                  f"OOF {n_folds}-fold")
    return result


def _compute_td_rates(apps, vars_, ts, y_arr, ref_time, lam, smoothing_w=5.0):
    """Compute temporal-decay rates for (app_id, variant) pairs.

    smoothing_w: pseudo-weight added as prior (towards global rate).
    """
    days_ago = (ref_time - ts) / 86400.0
    weights = np.exp(-lam * days_ago)

    global_w_oob = (weights * (y_arr == 2)).sum()
    global_w_tink = (weights * (y_arr == 1)).sum()
    global_w_total = weights.sum()
    global_oob = global_w_oob / global_w_total if global_w_total > 0 else 0.0
    global_tink = global_w_tink / global_w_total if global_w_total > 0 else 0.0

    pair_stats = {}
    for i in range(len(apps)):
        key = (apps[i], vars_[i] if pd.notna(vars_[i]) else "__none__")
        if key not in pair_stats:
            pair_stats[key] = {"w_oob": 0.0, "w_tink": 0.0, "w_total": 0.0}
        w = weights[i]
        pair_stats[key]["w_total"] += w
        if y_arr[i] == 2:
            pair_stats[key]["w_oob"] += w
        elif y_arr[i] == 1:
            pair_stats[key]["w_tink"] += w

    # Bayesian smoothed rates
    pair_oob = {}
    pair_tink = {}
    for key, s in pair_stats.items():
        wt = s["w_total"]
        pair_oob[key] = (s["w_oob"] + smoothing_w * global_oob) / (wt + smoothing_w)
        pair_tink[key] = (s["w_tink"] + smoothing_w * global_tink) / (wt + smoothing_w)

    return pair_oob, pair_tink, global_oob, global_tink


def build_p20_features(
    conn, app_ids: pd.Series, variants: pd.Series,
    timestamps: np.ndarray, train_mask: np.ndarray,
    y: np.ndarray, half_life_days: float = 180.0, n_folds: int = 5,
) -> pd.DataFrame:
    """P20: Temporal decay tinkering rate per (app_id, variant).

    Uses OOF encoding on training data to prevent leakage.
    Bayesian smoothed with pseudo-weight towards global rate.
    """
    from sklearn.model_selection import KFold

    lam = np.log(2) / half_life_days
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]

    train_apps = app_ids.iloc[train_idx].values
    train_vars = variants.iloc[train_idx].values
    train_ts = timestamps[train_idx]
    train_y = y[train_idx]
    ref_time = train_ts.max()

    oob_arr = np.full(len(app_ids), np.nan)
    tink_arr = np.full(len(app_ids), np.nan)

    # OOF for training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_train, fold_val in kf.split(train_idx):
        ft_apps = train_apps[fold_train]
        ft_vars = train_vars[fold_train]
        ft_ts = train_ts[fold_train]
        ft_y = train_y[fold_train]

        pair_oob, pair_tink, g_oob, g_tink = _compute_td_rates(
            ft_apps, ft_vars, ft_ts, ft_y, ref_time, lam
        )

        for vi in fold_val:
            gi = train_idx[vi]
            key = (train_apps[vi], train_vars[vi] if pd.notna(train_vars[vi]) else "__none__")
            oob_arr[gi] = pair_oob.get(key, g_oob)
            tink_arr[gi] = pair_tink.get(key, g_tink)

    # Full training rates for test/eval
    pair_oob_full, pair_tink_full, g_oob_full, g_tink_full = _compute_td_rates(
        train_apps, train_vars, train_ts, train_y, ref_time, lam
    )
    for ti in test_idx:
        app = app_ids.iloc[ti]
        var = variants.iloc[ti]
        key = (app, var if pd.notna(var) else "__none__")
        oob_arr[ti] = pair_oob_full.get(key, g_oob_full)
        tink_arr[ti] = pair_tink_full.get(key, g_tink_full)

    result = pd.DataFrame({
        "td_oob_rate": oob_arr,
        "td_tink_rate": tink_arr,
    }, index=app_ids.index)

    n_mapped = np.isfinite(oob_arr).sum()
    console.print(f"    P20: {len(pair_oob_full)} pairs, OOF {n_folds}-fold, "
                  f"mapped {n_mapped}/{len(app_ids)}")
    return result


def _compute_cross_rates(keys1, keys2, y_arr, smoothing_m=10.0):
    """Compute cross-entity rates for (key1, key2) → oob/tink rates."""
    global_oob = (y_arr == 2).mean()
    global_tink = (y_arr == 1).mean()

    stats = {}
    for i in range(len(keys1)):
        k1, k2 = keys1[i], keys2[i]
        if pd.isna(k1) or pd.isna(k2) or k1 == "__none__" or k2 == "__none__":
            continue
        key = (k1, k2)
        if key not in stats:
            stats[key] = {"n": 0, "oob": 0, "tink": 0}
        stats[key]["n"] += 1
        if y_arr[i] == 2:
            stats[key]["oob"] += 1
        elif y_arr[i] == 1:
            stats[key]["tink"] += 1

    rates = {}
    for key, s in stats.items():
        n = s["n"]
        rates[key] = {
            "oob": (s["oob"] + smoothing_m * global_oob) / (n + smoothing_m),
            "tink": (s["tink"] + smoothing_m * global_tink) / (n + smoothing_m),
        }
    return rates


def build_p19_features(
    conn, app_ids: pd.Series, gpu_families: pd.Series,
    variants: pd.Series, train_mask: np.ndarray, y: np.ndarray,
    smoothing_m: float = 10.0, n_folds: int = 5,
) -> pd.DataFrame:
    """P19: Cross-entity conditional statistics with OOF encoding.

    - gpu_family × engine → oob_rate, tinkering_rate
    - variant × engine → oob_rate, tinkering_rate
    Bayesian smoothing with m=10, 5-fold OOF for training data.
    """
    from sklearn.model_selection import KFold

    engine_rows = conn.execute(
        "SELECT app_id, engine FROM game_metadata WHERE engine IS NOT NULL"
    ).fetchall()
    engine_map = {r["app_id"]: r["engine"].strip().split(",")[0].strip().lower()
                  for r in engine_rows if r["engine"]}

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]

    all_engines = np.array([engine_map.get(a, "__none__") for a in app_ids.values])

    n = len(app_ids)
    ge_oob = np.full(n, np.nan)
    ge_tink = np.full(n, np.nan)
    ve_oob = np.full(n, np.nan)
    ve_tink = np.full(n, np.nan)

    # OOF for training data
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_train, fold_val in kf.split(train_idx):
        ft_idx = train_idx[fold_train]
        fv_idx = train_idx[fold_val]

        ft_gpu = gpu_families.iloc[ft_idx].values
        ft_eng = all_engines[ft_idx]
        ft_var = variants.iloc[ft_idx].values
        ft_y = y[ft_idx]

        ge_rates = _compute_cross_rates(ft_gpu, ft_eng, ft_y, smoothing_m)
        ve_rates = _compute_cross_rates(ft_var, ft_eng, ft_y, smoothing_m)

        for gi in fv_idx:
            gpu, eng, var = gpu_families.iloc[gi], all_engines[gi], variants.iloc[gi]
            r = ge_rates.get((gpu, eng))
            if r:
                ge_oob[gi] = r["oob"]
                ge_tink[gi] = r["tink"]
            r = ve_rates.get((var, eng))
            if r:
                ve_oob[gi] = r["oob"]
                ve_tink[gi] = r["tink"]

    # Full training rates for test/eval
    train_gpu = gpu_families.iloc[train_idx].values
    train_eng = all_engines[train_idx]
    train_var = variants.iloc[train_idx].values
    train_y = y[train_idx]

    ge_rates_full = _compute_cross_rates(train_gpu, train_eng, train_y, smoothing_m)
    ve_rates_full = _compute_cross_rates(train_var, train_eng, train_y, smoothing_m)

    console.print(f"    P19 gpu×eng: {len(ge_rates_full)} pairs, var×eng: {len(ve_rates_full)} pairs")

    for ti in test_idx:
        gpu, eng, var = gpu_families.iloc[ti], all_engines[ti], variants.iloc[ti]
        r = ge_rates_full.get((gpu, eng))
        if r:
            ge_oob[ti] = r["oob"]
            ge_tink[ti] = r["tink"]
        r = ve_rates_full.get((var, eng))
        if r:
            ve_oob[ti] = r["oob"]
            ve_tink[ti] = r["tink"]

    result = pd.DataFrame({
        "x_gpu_eng_oob": ge_oob,
        "x_gpu_eng_tink": ge_tink,
        "x_var_eng_oob": ve_oob,
        "x_var_eng_tink": ve_tink,
    }, index=app_ids.index)

    n_ge = np.isfinite(ge_oob).sum()
    n_ve = np.isfinite(ve_oob).sum()
    console.print(f"    P19: gpu×eng mapped={n_ge}, var×eng mapped={n_ve}, OOF {n_folds}-fold")
    return result


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

    # We need app_id, variant, gpu_family for new features — save before dropping
    # Re-fetch from DB since _build_feature_matrix drops them
    console.print("[bold]Loading auxiliary columns for new features...[/bold]")
    aux_df = pd.read_sql_query(
        "SELECT id, app_id, variant, gpu FROM reports", conn,
    )
    # Align with valid samples (same filtering as _build_feature_matrix)
    aux_df = aux_df.set_index("id").reindex(report_ids).reset_index()
    app_ids = aux_df["app_id"]
    variants = aux_df["variant"]
    # gpu_family is already in X
    gpu_families = X["gpu_family"] if "gpu_family" in X.columns else pd.Series(np.nan, index=X.index)

    del emb_data
    gc.collect()

    # ── Relabeling ──
    relabel_ids = get_relabel_ids(conn)

    # ── Split ──
    console.print("[bold]Time-based split...[/bold]")
    X_full, X_test_full, y_train_full, y_test_full, train_rids, _ = _time_based_split(
        X, y, timestamps, test_fraction=0.2, report_ids=report_ids,
    )

    # Save split indices for auxiliary data alignment
    sorted_indices = np.argsort(timestamps)
    split_point = int(len(sorted_indices) * 0.8)
    train_idx = sorted_indices[:split_point]
    test_idx = sorted_indices[split_point:]

    # Align auxiliary data
    app_ids_train = app_ids.iloc[train_idx].reset_index(drop=True)
    app_ids_test = app_ids.iloc[test_idx].reset_index(drop=True)
    variants_train = variants.iloc[train_idx].reset_index(drop=True)
    variants_test = variants.iloc[test_idx].reset_index(drop=True)
    gpu_families_train = gpu_families.iloc[train_idx].reset_index(drop=True)
    gpu_families_test = gpu_families.iloc[test_idx].reset_index(drop=True)
    ts_train = timestamps[train_idx]

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
    app_ids_train = app_ids_train[keep_mask].reset_index(drop=True)
    variants_train = variants_train[keep_mask].reset_index(drop=True)
    gpu_families_train = gpu_families_train[keep_mask].reset_index(drop=True)
    ts_train = ts_train[keep_mask]
    console.print(f"  Removed {n_removed} noisy samples")

    del X_full, y_train_full, keep_mask
    gc.collect()

    # Split test into ES + eval
    n_cal = len(X_test_full) // 2
    X_test_es = X_test_full.iloc[:n_cal].copy().reset_index(drop=True)
    y_test_es = y_test_full[:n_cal]
    X_eval = X_test_full.iloc[n_cal:].copy().reset_index(drop=True)
    y_eval = y_test_full[n_cal:]

    app_ids_es = app_ids_test.iloc[:n_cal].reset_index(drop=True)
    app_ids_eval = app_ids_test.iloc[n_cal:].reset_index(drop=True)
    variants_es = variants_test.iloc[:n_cal].reset_index(drop=True)
    variants_eval = variants_test.iloc[n_cal:].reset_index(drop=True)
    gpu_families_es = gpu_families_test.iloc[:n_cal].reset_index(drop=True)
    gpu_families_eval = gpu_families_test.iloc[n_cal:].reset_index(drop=True)

    del X_test_full, y_test_full
    gc.collect()

    console.print(f"  Train: {len(X_train)}, ES: {len(X_test_es)}, Eval: {len(X_eval)}")

    # Ensure categorical dtypes
    for col in CATEGORICAL_FEATURES:
        for df in [X_train, X_test_es, X_eval]:
            if col in df.columns:
                df[col] = df[col].astype("category")

    # ── Train shared Stage 1 ──
    console.print("\n[bold cyan]Training shared Stage 1...[/bold cyan]")
    t0 = time.time()
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es)
    console.print(f"  Stage 1: {s1.best_iteration_} iters ({time.time()-t0:.0f}s)")

    # ── Baseline ──
    console.print("\n[bold cyan]Baseline[/bold cyan]")
    baseline_r = run_full_cascade(s1, X_train, y_train, X_test_es, y_test_es,
                                   X_eval, y_eval, "baseline")
    console.print(f"  baseline: F1={baseline_r['f1_eval']:.4f} oob={baseline_r['f1_works_oob']:.3f}")

    all_results = [baseline_r]

    # ═══════════════════════════════════════════════════════════════
    # P18: ProtonDB tier + score
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]P18: ProtonDB tier/score[/bold cyan]")
    p18_train = build_p18_features(conn, app_ids_train)
    p18_es = build_p18_features(conn, app_ids_es)
    p18_eval = build_p18_features(conn, app_ids_eval)

    X_tr_p18 = pd.concat([X_train, p18_train.set_index(X_train.index)], axis=1)
    X_es_p18 = pd.concat([X_test_es, p18_es.set_index(X_test_es.index)], axis=1)
    X_ev_p18 = pd.concat([X_eval, p18_eval.set_index(X_eval.index)], axis=1)

    # Need to retrain Stage 1 with new features
    console.print("  Training Stage 1 with P18...")
    s1_p18 = train_stage1(X_tr_p18.copy(), y_train, X_es_p18.copy(), y_test_es)
    r = run_full_cascade(s1_p18, X_tr_p18, y_train, X_es_p18, y_test_es,
                          X_ev_p18, y_eval, "P18 tier+score")
    all_results.append(r)
    console.print(f"  P18: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
    del s1_p18
    gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # P11: Hierarchical target encoding
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]P11: Hierarchical target encoding[/bold cyan]")
    # Need combined app_ids for building the encoding
    n_train = len(X_train)
    n_es = len(X_test_es)
    n_eval = len(X_eval)
    all_app_ids = pd.concat([app_ids_train, app_ids_es, app_ids_eval], ignore_index=True)
    all_y = np.concatenate([y_train, y_test_es, y_eval])
    train_mask_all = np.zeros(len(all_app_ids), dtype=bool)
    train_mask_all[:n_train] = True

    p11_all = build_p11_features(conn, all_app_ids, all_y, train_mask_all)
    p11_train = p11_all.iloc[:n_train].reset_index(drop=True)
    p11_es = p11_all.iloc[n_train:n_train+n_es].reset_index(drop=True)
    p11_eval = p11_all.iloc[n_train+n_es:].reset_index(drop=True)

    X_tr_p11 = pd.concat([X_train, p11_train], axis=1)
    X_es_p11 = pd.concat([X_test_es, p11_es], axis=1)
    X_ev_p11 = pd.concat([X_eval, p11_eval], axis=1)

    s1_p11 = train_stage1(X_tr_p11.copy(), y_train, X_es_p11.copy(), y_test_es)
    r = run_full_cascade(s1_p11, X_tr_p11, y_train, X_es_p11, y_test_es,
                          X_ev_p11, y_eval, "P11 target enc")
    all_results.append(r)
    console.print(f"  P11: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
    del s1_p11
    gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # P20: Temporal decay rates
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]P20: Temporal decay tinkering rate[/bold cyan]")
    all_variants = pd.concat([variants_train, variants_es, variants_eval], ignore_index=True)
    all_ts = np.concatenate([ts_train, np.zeros(n_es + n_eval)])  # test timestamps not needed

    p20_all = build_p20_features(conn, all_app_ids, all_variants, all_ts,
                                  train_mask_all, all_y)
    p20_train = p20_all.iloc[:n_train].reset_index(drop=True)
    p20_es = p20_all.iloc[n_train:n_train+n_es].reset_index(drop=True)
    p20_eval = p20_all.iloc[n_train+n_es:].reset_index(drop=True)

    X_tr_p20 = pd.concat([X_train, p20_train], axis=1)
    X_es_p20 = pd.concat([X_test_es, p20_es], axis=1)
    X_ev_p20 = pd.concat([X_eval, p20_eval], axis=1)

    s1_p20 = train_stage1(X_tr_p20.copy(), y_train, X_es_p20.copy(), y_test_es)
    r = run_full_cascade(s1_p20, X_tr_p20, y_train, X_es_p20, y_test_es,
                          X_ev_p20, y_eval, "P20 temporal")
    all_results.append(r)
    console.print(f"  P20: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
    del s1_p20
    gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # P19: Cross-entity conditional stats
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]P19: Cross-entity stats[/bold cyan]")
    all_gpu = pd.concat([gpu_families_train, gpu_families_es, gpu_families_eval], ignore_index=True)

    p19_all = build_p19_features(conn, all_app_ids, all_gpu, all_variants,
                                  train_mask_all, all_y)
    p19_train = p19_all.iloc[:n_train].reset_index(drop=True)
    p19_es = p19_all.iloc[n_train:n_train+n_es].reset_index(drop=True)
    p19_eval = p19_all.iloc[n_train+n_es:].reset_index(drop=True)

    X_tr_p19 = pd.concat([X_train, p19_train], axis=1)
    X_es_p19 = pd.concat([X_test_es, p19_es], axis=1)
    X_ev_p19 = pd.concat([X_eval, p19_eval], axis=1)

    s1_p19 = train_stage1(X_tr_p19.copy(), y_train, X_es_p19.copy(), y_test_es)
    r = run_full_cascade(s1_p19, X_tr_p19, y_train, X_es_p19, y_test_es,
                          X_ev_p19, y_eval, "P19 cross-entity")
    all_results.append(r)
    console.print(f"  P19: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
    del s1_p19
    gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # ALL: Combined best features
    # ═══════════════════════════════════════════════════════════════
    console.print("\n[bold cyan]ALL: Combined features[/bold cyan]")
    X_tr_all = pd.concat([X_train, p18_train.set_index(X_train.index),
                           p11_train, p20_train, p19_train], axis=1)
    X_es_all = pd.concat([X_test_es, p18_es.set_index(X_test_es.index),
                           p11_es, p20_es, p19_es], axis=1)
    X_ev_all = pd.concat([X_eval, p18_eval.set_index(X_eval.index),
                           p11_eval, p20_eval, p19_eval], axis=1)

    s1_all = train_stage1(X_tr_all.copy(), y_train, X_es_all.copy(), y_test_es)
    r = run_full_cascade(s1_all, X_tr_all, y_train, X_es_all, y_test_es,
                          X_ev_all, y_eval, "ALL combined")
    all_results.append(r)
    console.print(f"  ALL: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")

    # ── Feature importance for combined model ──
    console.print("\n[bold]Stage 1 top features (ALL):[/bold]")
    importances = s1_all.feature_importances_
    feat_names = s1_all.feature_name_
    top_idx = np.argsort(importances)[-15:][::-1]
    for i in top_idx:
        console.print(f"  {feat_names[i]:30s} {importances[i]:>10.0f}")

    del s1_all
    gc.collect()

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    console.print("\n")
    table = Table(title="Phase 9.5 Results")
    table.add_column("Experiment", style="cyan")
    table.add_column("F1 eval", justify="right")
    table.add_column("ΔF1", justify="right")
    table.add_column("borked", justify="right")
    table.add_column("tinkering", justify="right")
    table.add_column("works_oob", justify="right")
    table.add_column("accuracy", justify="right")

    baseline_f1 = all_results[0]["f1_eval"]
    for r in all_results:
        delta = r["f1_eval"] - baseline_f1
        delta_str = f"{delta:+.4f}" if r["label"] != "baseline" else "—"
        style = "green" if delta > 0.002 else ("red" if delta < -0.002 else "")
        table.add_row(
            r["label"],
            f"{r['f1_eval']:.4f}",
            delta_str,
            f"{r['f1_borked']:.3f}",
            f"{r['f1_tinkering']:.3f}",
            f"{r['f1_works_oob']:.3f}",
            f"{r['accuracy']:.4f}",
            style=style,
        )

    console.print(table)
    conn.close()


if __name__ == "__main__":
    main()
