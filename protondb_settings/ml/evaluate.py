"""Evaluation: accuracy, F1, confusion matrix, SHAP, external validation."""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .models.classifier import TARGET_NAMES

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a trained model on test data.

    Returns a dict with accuracy, f1_macro, confusion_matrix, classification_report,
    and optionally SHAP top features.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=[TARGET_NAMES[i] for i in range(3)],
        zero_division=0,
    )

    results = {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    # SHAP analysis
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        # Use a sample for SHAP if the test set is large
        n_sample = min(1000, len(X_test))
        X_sample = X_test.iloc[:n_sample]
        shap_values = explainer.shap_values(X_sample)

        if feature_names is None:
            feature_names = list(X_test.columns)

        # Get mean absolute SHAP values across all classes
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
            if shap_arr.shape[0] == 3 and shap_arr.shape[0] != shap_arr.shape[2]:
                # (n_classes, n_samples, n_features)
                mean_shap = np.mean(np.abs(shap_arr), axis=(0, 1))
            else:
                # (n_samples, n_features, n_classes)
                mean_shap = np.mean(np.abs(shap_arr), axis=(0, 2))
        elif isinstance(shap_values, list):
            mean_shap = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_shap = np.abs(shap_arr).mean(axis=0)

        top_indices = np.argsort(mean_shap)[::-1][:20]
        top_features = [
            (feature_names[i], float(mean_shap[i])) for i in top_indices
        ]
        results["shap_top_features"] = top_features
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)
        results["shap_top_features"] = []

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print evaluation results using Rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold]Model Evaluation Results[/bold]")
    console.print(f"  Accuracy: [green]{results['accuracy']:.4f}[/green]")
    console.print(f"  F1 (macro): [green]{results['f1_macro']:.4f}[/green]")

    # Confusion matrix
    console.print("\n[bold]Confusion Matrix[/bold]")
    cm = results["confusion_matrix"]
    cm_table = Table(title="Predicted vs Actual")
    cm_table.add_column("", style="bold")
    for name in [TARGET_NAMES[i] for i in range(3)]:
        cm_table.add_column(name)
    for i in range(3):
        row_vals = [str(cm[i][j]) for j in range(3)]
        cm_table.add_row(TARGET_NAMES[i], *row_vals)
    console.print(cm_table)

    # Classification report
    console.print("\n[bold]Classification Report[/bold]")
    console.print(results["classification_report"])

    # SHAP top features
    if results.get("shap_top_features"):
        console.print("\n[bold]Top Features (SHAP)[/bold]")
        feat_table = Table()
        feat_table.add_column("Feature")
        feat_table.add_column("Mean |SHAP|", justify="right")
        for name, importance in results["shap_top_features"][:15]:
            feat_table.add_row(name, f"{importance:.4f}")
        console.print(feat_table)


def evaluate_per_game(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    test_report_ids: list[str],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Aggregate per-report predictions to per-game via majority vote and evaluate.

    Returns dict with per-game F1 (3-class, binary), classification report, game counts.
    """
    # Map report_id → app_id
    app_map: dict[str, int] = {}
    for row in conn.execute("SELECT id, app_id FROM reports").fetchall():
        app_map[row["id"] if isinstance(row, sqlite3.Row) else row[0]] = (
            row["app_id"] if isinstance(row, sqlite3.Row) else row[1]
        )

    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    # Group by game
    game_preds: dict[int, list[int]] = defaultdict(list)
    game_truths: dict[int, list[int]] = defaultdict(list)
    game_probas: dict[int, list[np.ndarray]] = defaultdict(list)

    for i, rid in enumerate(test_report_ids):
        aid = app_map.get(rid)
        if aid is None:
            continue
        game_preds[aid].append(int(y_pred[i]))
        game_truths[aid].append(int(y_test[i]))
        if y_proba is not None:
            game_probas[aid].append(y_proba[i])

    # Aggregate
    agg_true, agg_vote, agg_proba = [], [], []
    sizes = []
    for aid in game_preds:
        preds = game_preds[aid]
        truths = game_truths[aid]
        agg_true.append(Counter(truths).most_common(1)[0][0])
        agg_vote.append(Counter(preds).most_common(1)[0][0])
        if game_probas[aid]:
            probas = np.array(game_probas[aid])
            agg_proba.append(int(probas.mean(axis=0).argmax()))
        else:
            agg_proba.append(Counter(preds).most_common(1)[0][0])
        sizes.append(len(preds))

    agg_true = np.array(agg_true)
    agg_vote = np.array(agg_vote)
    agg_proba = np.array(agg_proba)
    sizes = np.array(sizes)

    f1_vote = float(f1_score(agg_true, agg_vote, average="macro", zero_division=0))
    f1_proba = float(f1_score(agg_true, agg_proba, average="macro", zero_division=0))

    bin_true = (agg_true > 0).astype(int)
    bin_vote = (agg_vote > 0).astype(int)
    f1_binary = float(f1_score(bin_true, bin_vote, average="macro", zero_division=0))

    report_vote = classification_report(
        agg_true, agg_vote,
        labels=[0, 1, 2],
        target_names=[TARGET_NAMES[i] for i in range(3)],
        zero_division=0,
    )

    result: dict[str, Any] = {
        "n_games": len(agg_true),
        "f1_vote": f1_vote,
        "f1_proba": f1_proba,
        "f1_binary": f1_binary,
        "classification_report": report_vote,
        "sizes_median": float(np.median(sizes)),
        "sizes_mean": float(np.mean(sizes)),
    }

    # Games with 3+ reports
    mask3 = sizes >= 3
    if mask3.sum() > 50:
        result["n_games_3plus"] = int(mask3.sum())
        result["f1_vote_3plus"] = float(
            f1_score(agg_true[mask3], agg_vote[mask3], average="macro", zero_division=0)
        )
        result["f1_binary_3plus"] = float(
            f1_score(
                (agg_true[mask3] > 0).astype(int),
                (agg_vote[mask3] > 0).astype(int),
                average="macro", zero_division=0,
            )
        )

    return result


def print_per_game_results(results: dict[str, Any]) -> None:
    """Print per-game evaluation results using Rich."""
    from rich.console import Console

    console = Console()

    console.print("\n[bold]Per-Game Aggregated Evaluation[/bold]")
    console.print(f"  Games: {results['n_games']} "
                  f"(median {results['sizes_median']:.0f} reports, "
                  f"mean {results['sizes_mean']:.1f})")
    console.print(f"  Majority vote F1:  [green]{results['f1_vote']:.4f}[/green]")
    console.print(f"  Mean proba F1:     [green]{results['f1_proba']:.4f}[/green]")
    console.print(f"  Binary F1:         [green]{results['f1_binary']:.4f}[/green]")

    if "n_games_3plus" in results:
        console.print(f"\n  Games with 3+ reports ({results['n_games_3plus']}):")
        console.print(f"    3-class F1: [green]{results['f1_vote_3plus']:.4f}[/green]")
        console.print(f"    Binary F1:  [green]{results['f1_binary_3plus']:.4f}[/green]")

    console.print(f"\n{results['classification_report']}")


def validate_against_protondb(
    model: Any,
    X: pd.DataFrame,
    app_ids: np.ndarray,
    game_metadata_lookup: dict[int, dict],
    min_reports: int = 50,
    report_counts: dict[int, int] | None = None,
) -> float | None:
    """Validate predictions against ProtonDB community tiers.

    Returns agreement rate or None if no game_metadata available.
    """
    if not game_metadata_lookup:
        logger.info("No game_metadata available for ProtonDB validation")
        return None

    tier_mapping = {
        "platinum": 2,
        "gold": 2,
        "silver": 1,
        "bronze": 0,
        "borked": 0,
    }

    valid_mask = []
    protondb_targets = []

    for i, app_id in enumerate(app_ids):
        meta = game_metadata_lookup.get(int(app_id))
        if meta is None:
            valid_mask.append(False)
            protondb_targets.append(-1)
            continue
        tier = meta.get("protondb_tier", "").lower() if meta.get("protondb_tier") else ""
        mapped = tier_mapping.get(tier)
        if mapped is None:
            valid_mask.append(False)
            protondb_targets.append(-1)
            continue
        # Check report count threshold
        if report_counts and report_counts.get(int(app_id), 0) < min_reports:
            valid_mask.append(False)
            protondb_targets.append(-1)
            continue
        valid_mask.append(True)
        protondb_targets.append(mapped)

    valid_mask = np.array(valid_mask)
    if not valid_mask.any():
        logger.info("No games with ProtonDB tier and >= %d reports", min_reports)
        return None

    X_valid = X[valid_mask]
    y_protondb = np.array(protondb_targets)[valid_mask]
    y_pred = model.predict(X_valid)
    agreement = (y_pred == y_protondb).mean()
    logger.info("ProtonDB agreement: %.1f%% (%d games)", agreement * 100, valid_mask.sum())
    return float(agreement)
