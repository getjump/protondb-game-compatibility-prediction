#!/usr/bin/env python3
"""Phase 9.4 experiment: Distillation and model changes.

Experiments:
  P8  — Teacher-student distillation (text features → soft targets for student)
  P15 — Focal loss for hard boundary samples
  P16 — Variant-specific sub-models (GE vs non-GE)
  P13 — Ordinal regression via cumulative logit approach

All experiments evaluated on held-out eval set (not used for ES).
"""

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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protondb_settings.ml.models.cascade import (
    CascadeClassifier,
    STAGE2_DROP_FEATURES,
    train_stage1,
)
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

console = Console()

# ── Stage 2 base params (Phase 9.1) ────────────────────────────────

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


def _train_s2_booster(X_tr, y_tr_labels, X_te, y_te, cat_cols, params=None, alpha=0.15):
    """Train Stage 2 lgb.Booster with label smoothing."""
    if params is None:
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

    # Ensure categorical dtypes for prediction
    X_ev = X_eval.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_ev.columns:
            X_ev[col] = X_ev[col].astype("category")

    y_pred = cascade.predict(X_ev)
    f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_eval, y_pred)
    f1_per = f1_score(y_eval, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "label": label,
        "f1_eval": f1,
        "accuracy": acc,
        "f1_borked": f1_per[0],
        "f1_tinkering": f1_per[1],
        "f1_works_oob": f1_per[2],
    }


# ── P8: Teacher-student distillation ────────────────────────────────

def experiment_p8_distillation(X_train, y_train, X_test_es, y_test_es, X_eval, y_eval):
    """Teacher-student distillation: teacher uses text_emb, student doesn't.

    1. Train teacher Stage 2 with all features (including text_emb)
    2. Get 3-fold OOF soft predictions from teacher
    3. Train student Stage 2 without text_emb features, using:
       target = α × teacher_prob + (1-α) × hard_label
    """
    import gc

    console.print("\n[bold cyan]P8: Teacher-student distillation[/bold cyan]")
    results = []

    text_cols = [c for c in X_train.columns if c.startswith("text_emb_")]
    console.print(f"  Text embedding features: {len(text_cols)}")
    if not text_cols:
        console.print("  [red]No text_emb features found, skipping P8[/red]")
        return results

    # Reuse shared Stage 1 — train_stage1 already called in main
    # We just need the s1 model for eval. Re-train here to keep P8 self-contained.
    t0 = time.time()
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es)
    console.print(f"  Stage 1: {s1.best_iteration_} iters ({time.time()-t0:.0f}s)")

    # ── No-text baseline: drop text_emb ──
    X_eval_notext = X_eval.drop(columns=text_cols)

    # Stage 2 without text features — but Stage 1 still uses all features
    drops_nt = list(STAGE2_DROP_FEATURES) + text_cols
    X_tr_nt, y_tr_nt, X_te_nt, y_te_nt, cat_nt, _ = _prepare_stage2(
        X_train.drop(columns=text_cols), y_train,
        X_test_es.drop(columns=text_cols), y_test_es,
    )
    s2_notext = _train_s2_booster(X_tr_nt, y_tr_nt, X_te_nt, y_te_nt, cat_nt)
    # Use full X_eval for Stage 1 (needs all features), drops_nt tells cascade to strip text for Stage 2
    r = run_cascade_eval(s1, s2_notext, drops_nt, X_eval, y_eval, "no text_emb")
    results.append(r)
    console.print(f"  no text: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
    del s2_notext
    gc.collect()

    # ── Teacher OOF predictions (3-fold to save memory) ──
    console.print("  Building teacher OOF predictions (3-fold)...")
    train_mask = y_train > 0
    X_s2_all = X_train[train_mask].reset_index(drop=True)
    y_s2_all = (y_train[train_mask] - 1).astype(float)

    s2_drops = [c for c in STAGE2_DROP_FEATURES if c in X_s2_all.columns]
    if s2_drops:
        X_s2_all = X_s2_all.drop(columns=s2_drops)

    cat_all = [c for c in CATEGORICAL_FEATURES if c in X_s2_all.columns]

    teacher_oof = np.zeros(len(X_s2_all))
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    y_int = y_s2_all.astype(int)

    y_s2_arr = np.asarray(y_s2_all, dtype=np.float64)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_s2_all, y_int)):
        X_fold_tr = X_s2_all.iloc[tr_idx]
        X_fold_val = X_s2_all.iloc[val_idx]
        y_fold_tr = y_s2_arr[tr_idx]

        # Label smoothing
        y_smooth = y_fold_tr * 0.85 + (1 - y_fold_tr) * 0.15

        ds_tr = lgb.Dataset(X_fold_tr, label=y_smooth, categorical_feature=cat_all)
        ds_val = lgb.Dataset(X_fold_val, label=y_s2_arr[val_idx],
                             categorical_feature=cat_all)

        teacher = lgb.train(
            S2_PARAMS, ds_tr, num_boost_round=2000, valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        teacher_oof[val_idx] = teacher.predict(X_fold_val)
        del teacher, ds_tr, ds_val
        gc.collect()
        console.print(f"    Fold {fold}: done")

    console.print(f"  Teacher OOF: mean={teacher_oof.mean():.3f}, std={teacher_oof.std():.3f}")

    # ── Student models with distillation ──
    y_s2_hard = np.asarray(y_s2_all, dtype=np.float64).copy()
    X_s2_notext = X_s2_all.drop(columns=text_cols, errors="ignore")
    del X_s2_all, y_s2_all
    gc.collect()

    cat_nt_s2 = [c for c in cat_all if c in X_s2_notext.columns]

    # Test set without text for ES
    test_mask = y_test_es > 0
    X_te_nt2 = X_test_es[test_mask].drop(
        columns=[c for c in STAGE2_DROP_FEATURES + text_cols if c in X_test_es.columns],
        errors="ignore",
    ).reset_index(drop=True)
    y_te_nt2 = (y_test_es[test_mask] - 1).astype(float)
    for col in cat_nt_s2:
        if col in X_te_nt2.columns:
            X_te_nt2[col] = X_te_nt2[col].astype("category")

    for alpha in [0.3, 0.5, 0.7]:
        y_distilled = alpha * teacher_oof + (1 - alpha) * y_s2_hard
        y_distilled = y_distilled * 0.85 + (1 - y_distilled) * 0.15

        ds_tr = lgb.Dataset(X_s2_notext, label=y_distilled, categorical_feature=cat_nt_s2)
        ds_te = lgb.Dataset(X_te_nt2, label=y_te_nt2, categorical_feature=cat_nt_s2)

        student = lgb.train(
            S2_PARAMS, ds_tr, num_boost_round=2000, valid_sets=[ds_te],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)],
        )

        r = run_cascade_eval(s1, student, drops_nt, X_eval, y_eval,
                             f"P8 distill α={alpha}")
        results.append(r)
        console.print(f"  α={alpha}: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f} "
                       f"(iters={student.best_iteration})")
        del student, ds_tr, ds_te
        gc.collect()

    return results


# ── P15: Focal loss ─────────────────────────────────────────────────

def _focal_loss(gamma=2.0):
    """Create focal loss objective and eval metric for LightGBM."""
    def focal_obj(preds, dtrain):
        labels = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))  # sigmoid
        grad = p - labels  # base gradient
        # Focal weight: (1-p_t)^gamma where p_t = p if y=1, (1-p) if y=0
        p_t = labels * p + (1 - labels) * (1 - p)
        w = (1 - p_t) ** gamma
        # Also add gamma * p_t * log(p_t) term for correct focal gradient
        grad = w * grad
        hess = w * p * (1 - p)
        hess = np.maximum(hess, 1e-7)
        return grad, hess

    def focal_eval(preds, dtrain):
        labels = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        p_t = labels * p + (1 - labels) * (1 - p)
        loss = -((1 - p_t) ** gamma) * np.log(p_t)
        return "focal_loss", loss.mean(), False

    return focal_obj, focal_eval


def experiment_p15_focal(X_train, y_train, X_test_es, y_test_es, X_eval, y_eval, s1):
    """Focal loss for Stage 2."""
    console.print("\n[bold cyan]P15: Focal loss[/bold cyan]")
    results = []

    X_tr, y_tr, X_te, y_te, cat_cols, drops = _prepare_stage2(
        X_train, y_train, X_test_es, y_test_es
    )

    y_tr_arr = np.asarray(y_tr, dtype=np.float64)
    y_te_arr = np.asarray(y_te, dtype=np.float64)

    for gamma in [1.0, 2.0, 3.0]:
        focal_obj, focal_eval = _focal_loss(gamma)

        ds_train = lgb.Dataset(X_tr, label=y_tr_arr, categorical_feature=cat_cols)
        ds_test = lgb.Dataset(X_te, label=y_te_arr, categorical_feature=cat_cols)

        params = {k: v for k, v in S2_PARAMS.items()
                  if k not in ("objective", "metric")}
        params["objective"] = focal_obj
        params["metric"] = "None"

        model = lgb.train(
            params, ds_train,
            num_boost_round=3000,
            valid_sets=[ds_test],
            feval=focal_eval,
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        r = run_cascade_eval(s1, model, drops, X_eval, y_eval, f"P15 focal γ={gamma}")
        results.append(r)
        console.print(f"  γ={gamma}: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f} "
                       f"(iters={model.best_iteration})")

    # Focal + label smoothing combo
    for gamma in [2.0]:
        focal_obj, focal_eval = _focal_loss(gamma)
        y_smooth = y_tr_arr * 0.85 + (1 - y_tr_arr) * 0.15
        ds_train = lgb.Dataset(X_tr, label=y_smooth, categorical_feature=cat_cols)
        ds_test = lgb.Dataset(X_te, label=y_te_arr, categorical_feature=cat_cols)

        params_s = dict(params)
        params_s["objective"] = focal_obj

        model = lgb.train(
            params_s, ds_train,
            num_boost_round=3000,
            valid_sets=[ds_test],
            feval=focal_eval,
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        r = run_cascade_eval(s1, model, drops, X_eval, y_eval,
                             f"P15 focal γ={gamma}+smooth")
        results.append(r)
        console.print(f"  γ={gamma}+smooth: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")

    return results


# ── P16: Variant-specific sub-models ───────────────────────────────

def experiment_p16_variant_split(X_train, y_train, X_test_es, y_test_es, X_eval, y_eval, s1):
    """Train separate Stage 2 models for GE vs non-GE variants."""
    console.print("\n[bold cyan]P16: Variant-specific sub-models[/bold cyan]")
    results = []

    # Identify variant groups
    # variant is encoded — we need to detect GE
    # In the feature matrix, "variant" column exists
    X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols, drops = _prepare_stage2(
        X_train, y_train, X_test_es, y_test_es
    )

    if "variant" not in X_tr_s2.columns:
        console.print("  [red]No variant column, skipping P16[/red]")
        return results

    # Get variant values (they're category-encoded)
    variant_train = X_tr_s2["variant"].astype(str)
    variant_test = X_te_s2["variant"].astype(str)

    # GE detection: look for 'ge' pattern
    ge_mask_train = variant_train.str.contains("ge", case=False, na=False)
    ge_mask_test = variant_test.str.contains("ge", case=False, na=False)

    console.print(f"  Train: {ge_mask_train.sum()} GE, {(~ge_mask_train).sum()} non-GE")
    console.print(f"  Test:  {ge_mask_test.sum()} GE, {(~ge_mask_test).sum()} non-GE")

    # If variant values are numeric (label-encoded), need different approach
    if ge_mask_train.sum() == 0:
        console.print("  Variant seems label-encoded, using value-based split...")
        # Try splitting by unique variant values
        unique_variants = variant_train.unique()
        console.print(f"  Unique variants: {unique_variants[:10]}")

        # Fall back to dropping variant and training without it
        X_tr_novar = X_tr_s2.drop(columns=["variant"])
        X_te_novar = X_te_s2.drop(columns=["variant"])
        cat_novar = [c for c in cat_cols if c != "variant"]

        for col in cat_novar:
            X_tr_novar[col] = X_tr_novar[col].astype("category")
            X_te_novar[col] = X_te_novar[col].astype("category")

        s2 = _train_s2_booster(X_tr_novar, y_tr_s2, X_te_novar, y_te_s2, cat_novar)
        X_eval_novar = X_eval.copy()
        if "variant" in X_eval_novar.columns:
            X_eval_novar = X_eval_novar.drop(columns=["variant"])
        drops_novar = drops + ["variant"] if "variant" not in drops else drops
        r = run_cascade_eval(s1, s2, drops_novar, X_eval_novar, y_eval, "P16 no-variant")
        results.append(r)
        console.print(f"  no-variant: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
        return results

    # Train separate models for GE and non-GE
    # Drop variant from features (it's constant within each sub-model)
    feats_novars = [c for c in X_tr_s2.columns if c != "variant"]
    cat_novar = [c for c in cat_cols if c != "variant"]

    # GE sub-model
    X_ge_tr = X_tr_s2.loc[ge_mask_train.values, feats_novars].reset_index(drop=True)
    y_ge_tr = np.asarray(y_tr_s2)[ge_mask_train.values]
    X_ge_te = X_te_s2.loc[ge_mask_test.values, feats_novars].reset_index(drop=True)
    y_ge_te = np.asarray(y_te_s2)[ge_mask_test.values]

    for col in cat_novar:
        if col in X_ge_tr.columns:
            X_ge_tr[col] = X_ge_tr[col].astype("category")
            X_ge_te[col] = X_ge_te[col].astype("category")

    console.print(f"  Training GE sub-model ({len(X_ge_tr)} samples)...")
    s2_ge = _train_s2_booster(X_ge_tr, y_ge_tr, X_ge_te, y_ge_te, cat_novar)

    # Non-GE sub-model
    X_nge_tr = X_tr_s2.loc[~ge_mask_train.values, feats_novars].reset_index(drop=True)
    y_nge_tr = np.asarray(y_tr_s2)[~ge_mask_train.values]
    X_nge_te = X_te_s2.loc[~ge_mask_test.values, feats_novars].reset_index(drop=True)
    y_nge_te = np.asarray(y_te_s2)[~ge_mask_test.values]

    for col in cat_novar:
        if col in X_nge_tr.columns:
            X_nge_tr[col] = X_nge_tr[col].astype("category")
            X_nge_te[col] = X_nge_te[col].astype("category")

    console.print(f"  Training non-GE sub-model ({len(X_nge_tr)} samples)...")
    s2_nge = _train_s2_booster(X_nge_tr, y_nge_tr, X_nge_te, y_nge_te, cat_novar)

    # Evaluate: route by variant
    # Need to evaluate on eval set — detect GE in eval
    X_eval_s2, y_eval_s2 = X_eval.copy(), y_eval.copy()
    # Re-run stage1 to get works_mask
    y_s1_pred = s1.predict(X_eval_s2)
    works_mask = y_s1_pred == 1

    y_pred = np.full(len(X_eval), 1, dtype=int)  # default tinkering
    y_pred[~works_mask] = 0  # borked

    if works_mask.any():
        X_works = X_eval_s2[works_mask].reset_index(drop=True)
        variant_eval = X_works["variant"].astype(str) if "variant" in X_works.columns else pd.Series([""] * len(X_works))
        ge_eval = variant_eval.str.contains("ge", case=False, na=False)

        X_works_novars = X_works.drop(columns=["variant"] + [c for c in STAGE2_DROP_FEATURES if c in X_works.columns],
                                       errors="ignore")

        preds_s2 = np.zeros(len(X_works))
        if ge_eval.any():
            preds_s2[ge_eval.values] = s2_ge.predict(X_works_novars[ge_eval])
        if (~ge_eval).any():
            preds_s2[~ge_eval.values] = s2_nge.predict(X_works_novars[~ge_eval])

        y_pred[works_mask] = np.where(preds_s2 >= 0.5, 2, 1)

    f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_eval, y_pred)
    f1_per = f1_score(y_eval, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    r = {
        "label": "P16 GE/non-GE split",
        "f1_eval": f1, "accuracy": acc,
        "f1_borked": f1_per[0], "f1_tinkering": f1_per[1], "f1_works_oob": f1_per[2],
    }
    results.append(r)
    console.print(f"  split: F1={f1:.4f} oob={f1_per[2]:.3f}")

    return results


# ── P13: Ordinal approach (cumulative binary) ──────────────────────

def experiment_p13_ordinal(X_train, y_train, X_test_es, y_test_es, X_eval, y_eval):
    """Ordinal regression: two cumulative binary models.

    Model A: P(y > 0) = P(not borked)
    Model B: P(y > 1) = P(works_oob)
    P(borked) = 1 - P(y>0)
    P(tinkering) = P(y>0) - P(y>1)
    P(works_oob) = P(y>1)
    """
    console.print("\n[bold cyan]P13: Ordinal (cumulative binary)[/bold cyan]")
    results = []

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    X_tr = X_train.copy()
    X_te = X_test_es.copy()
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype("category")
        X_te[col] = X_te[col].astype("category")

    # Model A: borked vs rest (same as Stage 1)
    y_a_train = (y_train > 0).astype(float)
    y_a_test = (y_test_es > 0).astype(float)

    ds_a_tr = lgb.Dataset(X_tr, label=y_a_train, categorical_feature=cat_cols)
    ds_a_te = lgb.Dataset(X_te, label=y_a_test, categorical_feature=cat_cols)

    params_a = dict(S2_PARAMS)
    params_a["min_child_samples"] = 20  # like stage1

    console.print("  Training Model A (P(y>0))...")
    model_a = lgb.train(
        params_a, ds_a_tr, num_boost_round=2000, valid_sets=[ds_a_te],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)],
    )

    # Model B: borked+tinkering vs works_oob
    y_b_train = (y_train > 1).astype(float)
    y_b_test = (y_test_es > 1).astype(float)

    # Apply label smoothing to B
    y_b_smooth = y_b_train * 0.85 + (1 - y_b_train) * 0.15

    ds_b_tr = lgb.Dataset(X_tr, label=y_b_smooth, categorical_feature=cat_cols)
    ds_b_te = lgb.Dataset(X_te, label=y_b_test, categorical_feature=cat_cols)

    console.print("  Training Model B (P(y>1))...")
    model_b = lgb.train(
        S2_PARAMS, ds_b_tr, num_boost_round=3000, valid_sets=[ds_b_te],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    # Predict
    X_ev = X_eval.copy()
    for col in cat_cols:
        if col in X_ev.columns:
            X_ev[col] = X_ev[col].astype("category")

    p_gt0 = model_a.predict(X_ev)  # P(not borked)
    p_gt1 = model_b.predict(X_ev)  # P(works_oob)

    # Ensure ordering: P(y>1) <= P(y>0)
    p_gt1 = np.minimum(p_gt1, p_gt0)

    p_borked = 1 - p_gt0
    p_tinkering = p_gt0 - p_gt1
    p_oob = p_gt1

    proba = np.column_stack([p_borked, p_tinkering, p_oob])
    y_pred = proba.argmax(axis=1)

    f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_eval, y_pred)
    f1_per = f1_score(y_eval, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    r = {
        "label": "P13 ordinal",
        "f1_eval": f1, "accuracy": acc,
        "f1_borked": f1_per[0], "f1_tinkering": f1_per[1], "f1_works_oob": f1_per[2],
    }
    results.append(r)
    console.print(f"  ordinal: F1={f1:.4f} oob={f1_per[2]:.3f} "
                   f"(A:{model_a.best_iteration} B:{model_b.best_iteration})")

    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    import gc

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

    # Free emb_data — no longer needed
    del emb_data
    gc.collect()

    # ── Relabeling ──
    relabel_ids = get_relabel_ids(conn)

    # ── Split ──
    console.print("[bold]Time-based split...[/bold]")
    X_train, X_test, y_train, y_test, train_rids, _ = _time_based_split(
        X, y, timestamps, test_fraction=0.2, report_ids=report_ids,
    )
    del X, y, timestamps, report_ids
    gc.collect()

    y_train, n_relabeled = apply_relabeling(y_train, train_rids, relabel_ids)
    console.print(f"  Relabeled {n_relabeled}")

    # ── Cleanlab noise removal (match production pipeline) ──
    console.print("[bold]Cleanlab noise removal (3%)...[/bold]")
    keep_mask = find_noisy_samples(X_train, y_train, frac_remove=0.03,
                                   cache_dir="data/")
    n_removed = (~keep_mask).sum()
    X_train = X_train[keep_mask].reset_index(drop=True)
    y_train = y_train[keep_mask]
    train_rids = [rid for rid, keep in zip(train_rids, keep_mask) if keep]
    console.print(f"  Removed {n_removed} noisy samples")

    # Split test into ES + eval
    n_cal = len(X_test) // 2
    X_test_es = X_test.iloc[:n_cal].copy().reset_index(drop=True)
    y_test_es = y_test[:n_cal]
    X_eval = X_test.iloc[n_cal:].copy().reset_index(drop=True)
    y_eval = y_test[n_cal:]
    del X_test, y_test
    gc.collect()

    console.print(f"  Train: {len(X_train)}, ES: {len(X_test_es)}, Eval: {len(X_eval)}")

    # ── Ensure categorical dtypes on all sets ──
    for col in CATEGORICAL_FEATURES:
        for df in [X_train, X_test_es, X_eval]:
            if col in df.columns:
                df[col] = df[col].astype("category")

    # ── Train shared Stage 1 ──
    console.print("\n[bold cyan]Training shared Stage 1...[/bold cyan]")
    t0 = time.time()
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es)
    console.print(f"  Stage 1: {s1.best_iteration_} iters ({time.time()-t0:.0f}s)")

    # ── Run experiments ──
    all_results = []

    # Baseline Stage 2
    console.print("\n[bold cyan]Baseline[/bold cyan]")
    X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols, drops = _prepare_stage2(
        X_train, y_train, X_test_es, y_test_es
    )
    s2 = _train_s2_booster(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols)
    r = run_cascade_eval(s1, s2, drops, X_eval, y_eval, "baseline")
    all_results.append(r)
    console.print(f"  baseline: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")
    del s2
    gc.collect()

    # P13: Ordinal
    p13_results = experiment_p13_ordinal(X_train, y_train, X_test_es, y_test_es, X_eval, y_eval)
    all_results.extend(p13_results)
    gc.collect()

    # P8: Distillation (heaviest — run last)
    p8_results = experiment_p8_distillation(X_train, y_train, X_test_es, y_test_es, X_eval, y_eval)
    all_results.extend(p8_results)

    # P15 (focal) and P16 (variant split) already tested — both harmful:
    # P15 focal γ=1: F1=0.643, γ=2: 0.619, γ=3: 0.592 (all worse than baseline 0.725)
    # P16 GE/non-GE split: F1=0.684 (worse, variant signal lost when splitting)

    # ── Summary table ──
    console.print("\n")
    table = Table(title="Phase 9.4 Results")
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
