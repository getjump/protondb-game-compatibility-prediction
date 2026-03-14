#!/usr/bin/env python3
"""Phase 9.3 experiment: Label noise modeling.

P5 — Dawid-Skene soft labels: treat reports as annotations from crowd,
     use EM to estimate consensus probabilities, train with cross_entropy.
P7 — Cleanlab: find high-confidence mislabels via out-of-fold CV, remove/downweight.

Each is tested independently and combined, vs baseline.
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_dawid_skene_soft_labels(
    conn: sqlite3.Connection,
    report_ids: list[str],
    y: np.ndarray,
) -> np.ndarray:
    """Compute Dawid-Skene soft labels for Stage 2 (tinkering vs works_oob).

    Groups reports by app_id+variant as "tasks". Each report is an "annotator"
    (ProtonDB users are anonymous, so we use report_id as worker_id).
    DS estimates consensus P(works_oob) for each task. Reports in the same
    task group get the same soft label.

    For single-report tasks, falls back to label smoothing (alpha=0.15).

    Returns: float array same shape as y, with soft labels [0, 1].
    """
    from crowdkit.aggregation import DawidSkene

    # Get app_id + variant for each report (batched to avoid SQL variable limit)
    report_meta = {}
    batch_size = 500
    rid_list = list(report_ids)
    for start in range(0, len(rid_list), batch_size):
        batch = rid_list[start:start + batch_size]
        placeholders = ",".join(["?"] * len(batch))
        rows = conn.execute(
            f"SELECT id, app_id, variant FROM reports WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        for r in rows:
            report_meta[r["id"]] = (r["app_id"], r["variant"] or "unknown")

    # Build annotation DataFrame: worker=report_id, task=app_id:variant, label=binary
    # Only non-borked (Stage 2 relevant)
    records = []
    for i, rid in enumerate(report_ids):
        if y[i] == 0:  # borked — skip for Stage 2 DS
            continue
        meta = report_meta.get(rid)
        if meta is None:
            continue
        app_id, variant = meta
        task_key = f"{app_id}:{variant}"
        binary_label = 1 if y[i] == 2 else 0  # 0=tinkering, 1=oob
        records.append({"worker": rid, "task": task_key, "label": binary_label})

    ann_df = pd.DataFrame(records)

    # Count reports per task
    task_counts = ann_df.groupby("task").size()
    multi_tasks = set(task_counts[task_counts >= 2].index)

    console = Console()
    console.print(f"  DS: {len(records)} annotations, {len(task_counts)} tasks, "
                  f"{len(multi_tasks)} multi-annotator tasks")

    # Run Dawid-Skene only on multi-annotator tasks
    ann_multi = ann_df[ann_df["task"].isin(multi_tasks)].copy()

    if len(ann_multi) > 0:
        ds = DawidSkene(n_iter=30)
        ds_proba = ds.fit_predict_proba(ann_multi)
        # ds_proba: DataFrame indexed by task, columns [0, 1] = P(tinkering), P(oob)
        task_p_oob = ds_proba[1].to_dict() if 1 in ds_proba.columns else {}
        console.print(f"  DS converged: {len(task_p_oob)} tasks with soft labels")
    else:
        task_p_oob = {}

    # Build soft label array
    alpha = 0.15  # fallback label smoothing for single-report tasks
    y_soft = np.full(len(y), np.nan)  # NaN for borked (not used by Stage 2)

    for i, rid in enumerate(report_ids):
        if y[i] == 0:
            y_soft[i] = np.nan  # borked — not in Stage 2
            continue
        meta = report_meta.get(rid)
        if meta is None:
            # Fallback
            y_soft[i] = float(y[i] == 2) * (1 - alpha) + float(y[i] != 2) * alpha
            continue
        task_key = f"{meta[0]}:{meta[1]}"
        if task_key in task_p_oob:
            y_soft[i] = task_p_oob[task_key]
        else:
            # Single-report task → label smoothing
            hard = float(y[i] == 2)
            y_soft[i] = hard * (1 - alpha) + (1 - hard) * alpha

    return y_soft


def build_cleanlab_mask(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_folds: int = 5,
    frac_remove: float = 0.05,
) -> np.ndarray:
    """Find likely mislabeled samples using Cleanlab confident learning.

    Returns boolean mask: True = keep, False = remove (suspected mislabel).
    """
    from cleanlab.filter import find_label_issues
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb

    console = Console()

    # Build out-of-fold predicted probabilities
    pred_proba = np.zeros((len(X_train), 3))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]

        from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_tr.columns]
        for col in cat_cols:
            X_tr[col] = X_tr[col].astype("category")
            X_val[col] = X_val[col].astype("category")

        model = lgb.LGBMClassifier(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_train[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
            categorical_feature=cat_cols,
        )
        pred_proba[val_idx] = model.predict_proba(X_val)

    # Find label issues
    issues = find_label_issues(
        labels=y_train,
        pred_probs=pred_proba,
        return_indices_ranked_by="self_confidence",
    )

    n_remove = int(len(y_train) * frac_remove)
    remove_indices = set(issues[:n_remove])

    keep_mask = np.ones(len(y_train), dtype=bool)
    keep_mask[list(remove_indices)] = False

    # Stats
    removed_labels = y_train[~keep_mask]
    console.print(f"  Cleanlab: {len(issues)} issues found, removing top {n_remove} "
                  f"({frac_remove*100:.0f}%)")
    for cls in [0, 1, 2]:
        n_cls = (removed_labels == cls).sum()
        console.print(f"    class {cls}: {n_cls} removed")

    return keep_mask


def run_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test_es: pd.DataFrame,
    y_test_es: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    label: str,
    console: Console,
    y_soft_train: np.ndarray | None = None,
) -> dict:
    """Train cascade and return metrics.

    If y_soft_train is provided, use it as Stage 2 soft labels instead of
    the default label smoothing.
    """
    from protondb_settings.ml.models.cascade import CascadeClassifier, train_stage1, train_stage2
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    t0 = time.time()

    # Stage 1 (unchanged)
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es, categorical_cols=cat_cols)

    # Stage 2 — optionally with DS soft labels
    if y_soft_train is not None:
        s2, s2_dropped = _train_stage2_soft(
            X_train.copy(), y_train, y_soft_train,
            X_test_es.copy(), y_test_es,
            categorical_cols=cat_cols,
        )
    else:
        s2, s2_dropped = train_stage2(
            X_train.copy(), y_train, X_test_es.copy(), y_test_es,
            categorical_cols=cat_cols,
        )

    cascade = CascadeClassifier(s1, s2, s2_dropped)

    # Categorical dtypes for prediction
    for col in cat_cols:
        if col in X_test_es.columns:
            X_test_es[col] = X_test_es[col].astype("category")
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype("category")

    # Metrics
    y_pred_es = cascade.predict(X_test_es)
    f1_es = f1_score(y_test_es, y_pred_es, average="macro", zero_division=0)

    y_pred_eval = cascade.predict(X_eval)
    acc_eval = accuracy_score(y_eval, y_pred_eval)
    f1_eval = f1_score(y_eval, y_pred_eval, average="macro", zero_division=0)
    f1_per = f1_score(y_eval, y_pred_eval, average=None, labels=[0, 1, 2], zero_division=0)

    elapsed = time.time() - t0

    console.print(f"  [{label}] ES F1={f1_es:.4f}  eval F1={f1_eval:.4f}  acc={acc_eval:.4f}  "
                  f"borked={f1_per[0]:.3f} tink={f1_per[1]:.3f} oob={f1_per[2]:.3f}  "
                  f"({elapsed:.0f}s)")

    return {
        "label": label,
        "f1_macro": f1_es,
        "f1_eval": f1_eval,
        "accuracy": acc_eval,
        "f1_borked": f1_per[0],
        "f1_tinkering": f1_per[1],
        "f1_works_oob": f1_per[2],
        "elapsed": elapsed,
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
    }


def _train_stage2_soft(
    X_train, y_train, y_soft_train,
    X_test, y_test,
    categorical_cols=None,
):
    """Train Stage 2 with Dawid-Skene soft labels (replacing default label smoothing)."""
    import lightgbm as lgb
    from protondb_settings.ml.models.cascade import STAGE2_DROP_FEATURES
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    drop_features = list(STAGE2_DROP_FEATURES)

    # Filter to non-borked
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_train_s2 = X_train[train_mask].reset_index(drop=True)
    y_soft_s2 = y_soft_train[train_mask]  # DS soft labels: P(works_oob)

    X_test_s2 = X_test[test_mask].reset_index(drop=True)
    y_test_s2 = (y_test[test_mask] - 1).astype(float)

    # Drop temporal bias features
    existing_drops = [c for c in drop_features if c in X_train_s2.columns]
    if existing_drops:
        X_train_s2 = X_train_s2.drop(columns=existing_drops)
        X_test_s2 = X_test_s2.drop(columns=existing_drops)

    cat_cols_s2 = [c for c in categorical_cols if c in X_train_s2.columns]
    for col in cat_cols_s2:
        X_train_s2[col] = X_train_s2[col].astype("category")
        X_test_s2[col] = X_test_s2[col].astype("category")

    # Handle NaN in soft labels (shouldn't happen, but safety)
    nan_mask = np.isnan(y_soft_s2)
    if nan_mask.any():
        # Replace NaN with hard labels + smoothing
        hard = (y_train[train_mask] - 1).astype(float)
        y_soft_s2 = np.where(nan_mask, hard * 0.85 + (1 - hard) * 0.15, y_soft_s2)

    ds_train = lgb.Dataset(X_train_s2, label=y_soft_s2, categorical_feature=cat_cols_s2)
    ds_test = lgb.Dataset(X_test_s2, label=y_test_s2, categorical_feature=cat_cols_s2)

    params = {
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

    model = lgb.train(
        params, ds_train, num_boost_round=3000,
        valid_sets=[ds_test],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )

    return model, existing_drops


def main():
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.relabeling import apply_relabeling, get_relabel_ids
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split

    console = Console()
    db_path = Path("data/protondb.db")
    emb_path = Path("data/embeddings.npz")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── Step 1: Build feature matrix ──
    console.print("[bold]Loading embeddings + building features...[/bold]")
    emb_data = load_embeddings(emb_path)

    from protondb_settings.config import NORMALIZED_DATA_SOURCE
    X, y, timestamps, report_ids, label_maps = _build_feature_matrix(
        conn, emb_data, normalized_data_source=NORMALIZED_DATA_SOURCE,
    )
    console.print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    # ── Step 2: Split ──
    console.print("[bold]Time-based split...[/bold]")
    relabel_ids = get_relabel_ids(conn)

    X_train, X_test, y_train, y_test, train_rids, test_rids = _time_based_split(
        X, y, timestamps, test_fraction=0.2, report_ids=report_ids,
    )
    y_train, n_relabeled = apply_relabeling(y_train, train_rids, relabel_ids)
    console.print(f"  Relabeled {n_relabeled} training samples")

    n_cal = len(X_test) // 2
    X_eval = X_test.iloc[n_cal:].copy().reset_index(drop=True)
    y_eval = y_test[n_cal:]
    X_test_es = X_test.iloc[:n_cal].copy().reset_index(drop=True)
    y_test_es = y_test[:n_cal]

    console.print(f"  Train: {len(X_train)}, ES: {len(X_test_es)}, Eval: {len(X_eval)}")

    results = []

    # ── Baseline ──
    console.print("\n[cyan]Baseline (label smoothing α=0.15):[/cyan]")
    r = run_experiment(X_train.copy(), y_train, X_test_es.copy(), y_test_es,
                       X_eval.copy(), y_eval, "baseline", console)
    results.append(r)

    # ── P5: Dawid-Skene soft labels ──
    console.print("\n[bold]Computing Dawid-Skene soft labels...[/bold]")
    y_soft = build_dawid_skene_soft_labels(conn, train_rids, y_train)

    # Stats on soft labels
    non_borked = y_train > 0
    soft_non_borked = y_soft[non_borked]
    console.print(f"  Soft label stats (non-borked): mean={np.nanmean(soft_non_borked):.3f}, "
                  f"std={np.nanstd(soft_non_borked):.3f}, "
                  f"min={np.nanmin(soft_non_borked):.3f}, max={np.nanmax(soft_non_borked):.3f}")
    # How many differ from hard smoothed labels?
    hard_smooth = np.where(y_train[non_borked] == 2, 0.85, 0.15)
    diff = np.abs(soft_non_borked - hard_smooth)
    console.print(f"  Samples where DS differs from hard-smoothed by >0.1: "
                  f"{(diff > 0.1).sum()} ({(diff > 0.1).mean()*100:.1f}%)")

    console.print("\n[cyan]P5: Dawid-Skene soft labels:[/cyan]")
    r = run_experiment(X_train.copy(), y_train, X_test_es.copy(), y_test_es,
                       X_eval.copy(), y_eval, "P5_dawid_skene", console,
                       y_soft_train=y_soft)
    results.append(r)

    # ── P7: Cleanlab ──
    console.print("\n[bold]Running Cleanlab (5-fold CV)...[/bold]")

    for frac in [0.03, 0.05, 0.10]:
        keep_mask = build_cleanlab_mask(X_train.copy(), y_train, frac_remove=frac)
        X_clean = X_train[keep_mask].reset_index(drop=True)
        y_clean = y_train[keep_mask]

        label = f"P7_cleanlab_{int(frac*100)}pct"
        console.print(f"\n[cyan]{label} (remove {frac*100:.0f}%, keep {keep_mask.sum()}):[/cyan]")
        r = run_experiment(X_clean.copy(), y_clean, X_test_es.copy(), y_test_es,
                           X_eval.copy(), y_eval, label, console)
        results.append(r)

    # ── P5+P7 combined: DS soft labels + Cleanlab removal ──
    console.print("\n[bold]P5+P7 combined: DS soft labels + Cleanlab 5% removal...[/bold]")
    keep_mask_5 = build_cleanlab_mask(X_train.copy(), y_train, frac_remove=0.05)
    X_combined = X_train[keep_mask_5].reset_index(drop=True)
    y_combined = y_train[keep_mask_5]
    y_soft_combined = y_soft[keep_mask_5]

    console.print(f"\n[cyan]P5+P7 (DS + Cleanlab 5%):[/cyan]")
    r = run_experiment(X_combined.copy(), y_combined, X_test_es.copy(), y_test_es,
                       X_eval.copy(), y_eval, "P5+P7_combined", console,
                       y_soft_train=y_soft_combined)
    results.append(r)

    # ── Summary ──
    console.print("\n")
    table = Table(title="Phase 9.3 — Label Noise Modeling")
    table.add_column("Experiment", style="cyan")
    table.add_column("Train", justify="right")
    table.add_column("F1 (ES)", justify="right")
    table.add_column("F1 (eval)", justify="right", style="bold")
    table.add_column("ΔF1", justify="right")
    table.add_column("borked", justify="right")
    table.add_column("tinkering", justify="right")
    table.add_column("works_oob", justify="right")

    baseline_f1 = results[0]["f1_eval"]
    for r in results:
        delta = r["f1_eval"] - baseline_f1
        delta_str = f"{delta:+.4f}"
        if delta > 0.002:
            delta_str = f"[green]{delta_str}[/green]"
        elif delta < -0.002:
            delta_str = f"[red]{delta_str}[/red]"

        table.add_row(
            r["label"],
            str(r["n_train"]),
            f"{r['f1_macro']:.4f}",
            f"{r['f1_eval']:.4f}",
            delta_str,
            f"{r['f1_borked']:.3f}",
            f"{r['f1_tinkering']:.3f}",
            f"{r['f1_works_oob']:.3f}",
        )

    console.print(table)
    conn.close()


if __name__ == "__main__":
    main()
