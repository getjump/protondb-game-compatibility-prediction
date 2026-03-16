"""Optuna-based hyperparameter optimization for cascade classifier."""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from .irt import (
    add_error_targeted_features,
    add_irt_features,
    contributor_aware_relabel,
    fit_irt,
)
from .models.cascade import STAGE2_DROP_FEATURES, CascadeClassifier
from .models.classifier import CATEGORICAL_FEATURES
from .relabeling import get_relabel_ids

logger = logging.getLogger(__name__)


def _suggest_stage1_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": 3000,
        "num_leaves": trial.suggest_int("s1_num_leaves", 31, 127),
        "learning_rate": trial.suggest_float("s1_lr", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("s1_min_child_samples", 10, 100),
        "subsample": trial.suggest_float("s1_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("s1_colsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("s1_reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("s1_reg_lambda", 1e-3, 10.0, log=True),
        "max_depth": -1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
        "importance_type": "gain",
    }


def _suggest_stage1_class_weight(trial: optuna.Trial) -> dict[int, float]:
    borked_w = trial.suggest_float("s1_borked_weight", 1.0, 6.0)
    return {0: borked_w, 1: 1.0}


def _suggest_stage2_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "cross_entropy",
        "metric": "binary_logloss",
        "num_leaves": trial.suggest_int("s2_num_leaves", 31, 127),
        "learning_rate": trial.suggest_float("s2_lr", 0.005, 0.1, log=True),
        "min_child_samples": trial.suggest_int("s2_min_child_samples", 20, 150),
        "subsample": trial.suggest_float("s2_subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("s2_colsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("s2_reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("s2_reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("s2_min_split_gain", 0.0, 0.2),
        "verbose": -1,
    }


def _train_and_evaluate(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    train_rids: list[str],
    val_rids: list[str],
    conn: sqlite3.Connection,
    irt_theta: dict[str, float],
    irt_difficulty: dict[tuple[int, str], float],
    relabel_ids: set[str],
) -> float:
    """Single trial: train cascade with suggested HPs, return F1 macro."""
    # --- Suggest hyperparameters ---
    s1_params = _suggest_stage1_params(trial)
    s1_class_weight = _suggest_stage1_class_weight(trial)
    s2_params = _suggest_stage2_params(trial)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.3)
    oob_weight = trial.suggest_float("oob_weight", 1.0, 3.0)
    theta_threshold = trial.suggest_float("theta_threshold", 0.0, 2.0)

    # --- Relabeling ---
    y_tr = y_train.copy()
    if irt_theta:
        y_tr, _ = contributor_aware_relabel(
            y_tr, train_rids, relabel_ids, conn, irt_theta,
            theta_threshold=theta_threshold,
        )

    # --- Stage 1: borked vs works ---
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    X_tr_s1 = X_train.copy()
    X_val_s1 = X_val.copy()
    for col in cat_cols:
        X_tr_s1[col] = X_tr_s1[col].astype("category")
        X_val_s1[col] = X_val_s1[col].astype("category")

    y_tr_bin = (y_tr > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)

    s1_model = lgb.LGBMClassifier(**s1_params, class_weight=s1_class_weight)
    s1_model.fit(
        X_tr_s1, y_tr_bin,
        eval_set=[(X_val_s1, y_val_bin)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),  # silent
        ],
        categorical_feature=cat_cols,
    )

    # --- Stage 2: tinkering vs works_oob ---
    train_works = y_tr > 0
    val_works = y_val > 0

    X_tr_s2 = X_train[train_works].reset_index(drop=True)
    y_tr_s2 = (y_tr[train_works] - 1).astype(float)
    X_val_s2 = X_val[val_works].reset_index(drop=True)
    y_val_s2 = (y_val[val_works] - 1).astype(float)

    # Drop temporal features
    for df in (X_tr_s2, X_val_s2):
        for col in STAGE2_DROP_FEATURES:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    cat_cols_s2 = [c for c in cat_cols if c in X_tr_s2.columns]
    for col in cat_cols_s2:
        X_tr_s2[col] = X_tr_s2[col].astype("category")
        X_val_s2[col] = X_val_s2[col].astype("category")

    # Label smoothing
    y_smooth = y_tr_s2.copy()
    if label_smoothing > 0:
        y_smooth = y_smooth * (1 - label_smoothing) + (1 - y_smooth) * label_smoothing

    sample_weight = np.ones(len(y_tr_s2))
    sample_weight[y_tr_s2 >= 0.5] = oob_weight

    ds_train = lgb.Dataset(X_tr_s2, label=y_smooth, weight=sample_weight,
                           categorical_feature=cat_cols_s2)
    ds_val = lgb.Dataset(X_val_s2, label=y_val_s2, categorical_feature=cat_cols_s2)

    s2_model = lgb.train(
        s2_params, ds_train,
        num_boost_round=2000,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # --- Evaluate cascade ---
    dropped = [c for c in STAGE2_DROP_FEATURES if c in X_train.columns]
    cascade = CascadeClassifier(s1_model, s2_model, dropped)

    y_pred = cascade.predict(X_val_s1)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

    return f1


def run_optimization(
    conn: sqlite3.Connection,
    n_trials: int = 50,
    output_dir: str | Path = "data/",
    test_fraction: float = 0.2,
    normalized_data_source: str | None = None,
) -> dict[str, Any]:
    """Run Optuna optimization for cascade hyperparameters.

    Returns dict with best_params, best_f1, study.
    """
    from rich.console import Console

    from .features.embeddings import load_embeddings
    from .train import _build_feature_matrix, _time_based_split

    console = Console()
    output_dir = Path(output_dir)

    if normalized_data_source is None:
        from protondb_settings.config import NORMALIZED_DATA_SOURCE
        normalized_data_source = NORMALIZED_DATA_SOURCE

    # Load cached embeddings
    emb_path = output_dir / "embeddings.npz"
    if not emb_path.exists():
        console.print("[red]No cached embeddings. Run train-cascade first.[/red]")
        raise FileNotFoundError(emb_path)

    emb_data = load_embeddings(emb_path)

    # Build features
    console.print("[bold]Building features...[/bold]")
    X, y, timestamps, report_ids, label_maps = _build_feature_matrix(
        conn, emb_data, normalized_data_source=normalized_data_source,
    )
    console.print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

    # IRT
    console.print("[bold]Fitting IRT...[/bold]")
    irt_theta, irt_difficulty = fit_irt(conn)
    relabel_ids = get_relabel_ids(conn)

    # Split
    X_train, X_test, y_train, y_test, train_rids, test_rids = _time_based_split(
        X, y, timestamps, test_fraction, report_ids=report_ids,
    )

    # Add IRT + error features
    if irt_theta:
        X_train = add_irt_features(X_train, train_rids, conn, irt_theta, irt_difficulty)
        X_test = add_irt_features(X_test, test_rids, conn, irt_theta, irt_difficulty)
    X_train = add_error_targeted_features(X_train, train_rids, conn)
    X_test = add_error_targeted_features(X_test, test_rids, conn)

    # Use first half of test for optimization, second for final eval
    n_opt = len(X_test) // 2
    X_opt = X_test.iloc[:n_opt].copy().reset_index(drop=True)
    y_opt = y_test[:n_opt]
    opt_rids = test_rids[:n_opt]
    X_eval = X_test.iloc[n_opt:].copy().reset_index(drop=True)
    y_eval = y_test[n_opt:]
    eval_rids = test_rids[n_opt:]

    console.print(f"  Train: {len(X_train)}, Optimize: {len(X_opt)}, Eval: {len(X_eval)}")

    # Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="cascade_hpo")

    t0 = time.time()
    best_so_far = 0.0

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_so_far
        f1 = _train_and_evaluate(
            trial, X_train, y_train, X_opt, y_opt,
            train_rids, opt_rids, conn,
            irt_theta, irt_difficulty, relabel_ids,
        )
        if f1 > best_so_far:
            best_so_far = f1
            console.print(
                f"  [green]Trial {trial.number}: F1={f1:.4f} (new best)[/green]"
            )
        else:
            console.print(f"  Trial {trial.number}: F1={f1:.4f}")
        return f1

    console.print(f"\n[bold]Running {n_trials} Optuna trials...[/bold]")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    elapsed = time.time() - t0
    best = study.best_trial
    console.print(f"\n[bold]Optimization complete in {elapsed:.0f}s[/bold]")
    console.print(f"  Best F1: [green]{best.value:.4f}[/green] (trial {best.number})")

    # Print best params
    console.print("\n[bold]Best hyperparameters:[/bold]")
    for k, v in sorted(best.params.items()):
        console.print(f"  {k}: {v}")

    # Retrain with best params on full train, evaluate on held-out eval
    console.print("\n[bold]Retraining with best params on eval split...[/bold]")
    best_trial_for_eval = optuna.trial.FixedTrial(best.params)
    f1_eval = _train_and_evaluate(
        best_trial_for_eval, X_train, y_train, X_eval, y_eval,
        train_rids, eval_rids, conn,
        irt_theta, irt_difficulty, relabel_ids,
    )
    console.print(f"  Eval F1 (held-out): [green]{f1_eval:.4f}[/green]")

    # Save best params
    import json
    params_path = output_dir / "best_hparams.json"
    with open(params_path, "w") as f:
        json.dump({"best_f1_opt": best.value, "best_f1_eval": f1_eval,
                    "params": best.params, "n_trials": n_trials}, f, indent=2)
    console.print(f"  Saved to {params_path}")

    return {
        "best_f1_opt": best.value,
        "best_f1_eval": f1_eval,
        "best_params": best.params,
        "study": study,
    }
