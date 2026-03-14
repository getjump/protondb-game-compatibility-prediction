#!/usr/bin/env python3
"""Deep analysis & visualization of the cascade model.

Produces:
  data/plots/cascade_*.png — all visualizations
  stdout — textual insights and recommendations
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.cascade import CascadeClassifier, train_stage1, train_stage2
from protondb_settings.ml.train import _build_feature_matrix, _compute_target

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(exist_ok=True)
CLASS_NAMES = ["borked", "tinkering", "works_oob"]


def load_and_train():
    """Load data, train cascade, return everything needed for analysis."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Get extra fields for analysis
    rows = conn.execute(
        "SELECT app_id, verdict, verdict_oob, variant, proton_version FROM reports"
    ).fetchall()
    conn.close()

    app_ids, variants = [], []
    for row in rows:
        t = _compute_target(row["verdict"], row["verdict_oob"])
        if t is not None:
            app_ids.append(row["app_id"])
            variants.append(row["variant"])

    # Time-based split
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    ts_test = timestamps[test_idx]
    app_ids_test = [app_ids[i] for i in test_idx]
    variants_test = [variants[i] for i in test_idx]

    # Train
    print("Training cascade...")
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_dropped = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_dropped)

    # Calibrate — ensure categorical dtypes match training
    cal_idx = sorted_idx[split_point - split_point // 5 : split_point]
    X_cal = X.iloc[cal_idx].reset_index(drop=True)
    y_cal = y[cal_idx]
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
    for col in CATEGORICAL_FEATURES:
        if col in X_cal.columns:
            X_cal[col] = X_cal[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")
    cascade.fit_calibrators(X_cal, y_cal)

    return cascade, X_train, y_train, X_test, y_test, ts_test, app_ids_test, variants_test


def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=12)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(CLASS_NAMES)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_title(f"{title} — Counts")
    fig.colorbar(im1, ax=ax1)

    # Normalized
    im2 = ax2.imshow(cm_pct, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{cm_pct[i, j]:.1%}", ha="center", va="center",
                     color="white" if cm_pct[i, j] > 0.5 else "black", fontsize=12)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(CLASS_NAMES)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(CLASS_NAMES)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_title(f"{title} — Row-normalized")
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, title, path, top_n=30):
    features = model.feature_name_
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    colors = []
    for i in idx:
        name = features[i]
        if "emb_" in name:
            colors.append("#4CAF50")  # green — embeddings
        elif name.startswith(("audio_faults", "graphical_faults", "input_faults",
                              "performance_faults", "stability_faults", "windowing_faults",
                              "save_game_faults", "significant_bugs", "fault_any", "fault_count")):
            colors.append("#F44336")  # red — A1 fault booleans
        elif name.startswith(("cust_", "flag_")):
            colors.append("#E91E63")  # pink — A4 cust/flag
        elif name.startswith(("mentions_", "has_", "total_notes", "concluding", "sentiment_")):
            colors.append("#FF9800")  # orange — text features
        elif name.startswith(("game_", "genre", "engine", "graphics", "drm",
                              "anticheat", "linux", "multiplayer", "release",
                              "is_impacted_by_anticheat")):
            colors.append("#2196F3")  # blue — game metadata
        else:
            colors.append("#9C27B0")  # purple — other (hw, temporal, variant)

    ax.barh(range(len(idx)), importances[idx], color=colors)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([features[i] for i in idx], fontsize=8)
    ax.set_xlabel("Gain")
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Embeddings"),
        Patch(facecolor="#F44336", label="Fault booleans (A1)"),
        Patch(facecolor="#E91E63", label="Cust/Flag (A4)"),
        Patch(facecolor="#FF9800", label="Text features"),
        Patch(facecolor="#2196F3", label="Game metadata"),
        Patch(facecolor="#9C27B0", label="Report / HW"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, proba, path):
    """Reliability diagram per class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for cls, (ax, name) in enumerate(zip(axes, CLASS_NAMES)):
        y_bin = (y_true == cls).astype(int)
        p = proba[:, cls]

        # Bin
        n_bins = 15
        bins = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true = []
        bin_counts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() > 0:
                bin_means.append(p[mask].mean())
                bin_true.append(y_bin[mask].mean())
                bin_counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.bar(bin_means, bin_true, width=0.05, alpha=0.6, color="steelblue", label="Observed")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{name}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        # ECE
        total = sum(bin_counts)
        if total > 0 and bin_means:
            ece = sum(abs(m - t) * c for m, t, c in zip(bin_means, bin_true, bin_counts)) / total
            ax.text(0.05, 0.9, f"ECE={ece:.4f}", transform=ax.transAxes, fontsize=10)

        ax.legend(fontsize=8)

    plt.suptitle("Calibration (Reliability Diagrams)", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confidence_accuracy(y_true, proba, path):
    """Confidence vs accuracy + coverage trade-off."""
    y_pred = proba.argmax(axis=1)
    confidence = proba.max(axis=1)

    thresholds = np.arange(0.3, 1.0, 0.02)
    accuracies = []
    coverages = []
    f1s = []

    for t in thresholds:
        mask = confidence >= t
        if mask.sum() < 10:
            accuracies.append(np.nan)
            coverages.append(0)
            f1s.append(np.nan)
            continue
        accuracies.append((y_true[mask] == y_pred[mask]).mean())
        coverages.append(mask.mean())
        f1s.append(f1_score(y_true[mask], y_pred[mask], average="macro"))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(thresholds, accuracies, "b-o", markersize=3, label="Accuracy")
    ax1.plot(thresholds, f1s, "r-s", markersize=3, label="F1 macro")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy / F1")
    ax1.set_ylim(0.5, 1.0)

    ax2 = ax1.twinx()
    ax2.fill_between(thresholds, coverages, alpha=0.15, color="green")
    ax2.plot(thresholds, coverages, "g--", alpha=0.7, label="Coverage")
    ax2.set_ylabel("Coverage (fraction of data)")
    ax2.set_ylim(0, 1.0)

    # Mark key thresholds
    for t_mark in [0.5, 0.7, 0.8, 0.9]:
        mask = confidence >= t_mark
        if mask.sum() > 10:
            acc = (y_true[mask] == y_pred[mask]).mean()
            cov = mask.mean()
            ax1.axvline(t_mark, color="gray", alpha=0.3, linestyle=":")
            ax1.annotate(f"t={t_mark}\nacc={acc:.1%}\ncov={cov:.1%}",
                         xy=(t_mark, acc), fontsize=7, ha="center",
                         xytext=(t_mark + 0.03, acc - 0.05))

    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")
    ax1.set_title("Confidence → Accuracy / Coverage Trade-off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stage_probability_distributions(cascade, X_test, y_test, path):
    """Distribution of stage probabilities, colored by true class."""
    p_s1 = cascade.stage1.predict_proba(X_test)
    X_s2 = cascade._prepare_stage2_input(X_test)
    p_s2 = cascade.stage2.predict_proba(X_s2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage 1: P(borked)
    ax = axes[0]
    for cls, name, color in [(0, "borked", "red"), (1, "tinkering", "orange"), (2, "works_oob", "green")]:
        mask = y_test == cls
        ax.hist(p_s1[mask, 0], bins=50, alpha=0.5, color=color, label=name, density=True)
    ax.axvline(0.5, color="black", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("P(borked) — Stage 1")
    ax.set_ylabel("Density")
    ax.set_title("Stage 1: P(borked) by true class")
    ax.legend()

    # Stage 2: P(works_oob | not borked)
    ax = axes[1]
    for cls, name, color in [(1, "tinkering", "orange"), (2, "works_oob", "green")]:
        mask = y_test == cls
        ax.hist(p_s2[mask, 1], bins=50, alpha=0.5, color=color, label=name, density=True)
    borked_mask = y_test == 0
    ax.hist(p_s2[borked_mask, 1], bins=50, alpha=0.3, color="red", label="borked (leaked)", density=True)
    ax.axvline(0.5, color="black", linestyle="--", label="threshold=0.5")
    ax.set_xlabel("P(works_oob) — Stage 2")
    ax.set_ylabel("Density")
    ax.set_title("Stage 2: P(works_oob) by true class")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(y_true, proba, path):
    """ROC and PR curves per class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax = axes[0]
    for cls, name, color in [(0, "borked", "red"), (1, "tinkering", "orange"), (2, "works_oob", "green")]:
        y_bin = (y_true == cls).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba[:, cls])
        auc = roc_auc_score(y_bin, proba[:, cls])
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend()

    # PR
    ax = axes[1]
    for cls, name, color in [(0, "borked", "red"), (1, "tinkering", "orange"), (2, "works_oob", "green")]:
        y_bin = (y_true == cls).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, proba[:, cls])
        ap = average_precision_score(y_bin, proba[:, cls])
        ax.plot(rec, prec, color=color, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_analysis_by_variant(y_true, y_pred, variants, path):
    """Per-variant accuracy and error patterns."""
    df = pd.DataFrame({"true": y_true, "pred": y_pred, "variant": variants})

    variant_order = ["official", "ge", "experimental", "native", "notListed", "older"]
    variants_present = [v for v in variant_order if v in df["variant"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Per-variant accuracy and F1
    ax = axes[0]
    accs = []
    f1s = []
    counts = []
    for v in variants_present:
        mask = df["variant"] == v
        sub = df[mask]
        accs.append((sub["true"] == sub["pred"]).mean())
        f1s.append(f1_score(sub["true"], sub["pred"], average="macro", zero_division=0))
        counts.append(len(sub))

    x = range(len(variants_present))
    width = 0.35
    ax.bar([i - width / 2 for i in x], accs, width, label="Accuracy", color="steelblue")
    ax.bar([i + width / 2 for i in x], f1s, width, label="F1 macro", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(n={c})" for v, c in zip(variants_present, counts)], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Performance by Proton variant")
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    # Per-variant confusion: misclass distribution
    ax = axes[1]
    error_types = {v: {"borked→tinker": 0, "borked→oob": 0,
                        "tinker→borked": 0, "tinker→oob": 0,
                        "oob→borked": 0, "oob→tinker": 0}
                   for v in variants_present}
    label_map = {0: "borked", 1: "tinker", 2: "oob"}
    for _, row in df.iterrows():
        if row["true"] != row["pred"] and row["variant"] in error_types:
            key = f"{label_map[row['true']]}→{label_map[row['pred']]}"
            error_types[row["variant"]][key] += 1

    error_df = pd.DataFrame(error_types).T
    error_df_pct = error_df.div(error_df.sum(axis=1), axis=0).fillna(0)
    error_df_pct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Error distribution by variant")
    ax.set_ylabel("Fraction of errors")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_drift(y_true, y_pred, proba, timestamps, path):
    """Performance over time (test set)."""
    df = pd.DataFrame({
        "ts": timestamps,
        "true": y_true,
        "pred": y_pred,
        "confidence": proba.max(axis=1),
        "correct": (y_true == y_pred).astype(int),
    })
    df["date"] = pd.to_datetime(df["ts"], unit="s")
    df["month"] = df["date"].dt.to_period("M")

    monthly = df.groupby("month").agg(
        accuracy=("correct", "mean"),
        confidence=("confidence", "mean"),
        count=("correct", "count"),
        borked_rate=("true", lambda x: (x == 0).mean()),
    ).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)

    # Also compute monthly F1
    monthly_f1 = []
    for _, row in monthly.iterrows():
        mask = df["month"] == row["month"]
        sub = df[mask]
        if len(sub) > 10:
            monthly_f1.append(f1_score(sub["true"], sub["pred"], average="macro", zero_division=0))
        else:
            monthly_f1.append(np.nan)
    monthly["f1"] = monthly_f1

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Accuracy + F1
    ax = axes[0]
    ax.plot(monthly["month_str"], monthly["accuracy"], "b-o", markersize=3, label="Accuracy")
    ax.plot(monthly["month_str"], monthly["f1"], "r-s", markersize=3, label="F1 macro")
    ax.set_ylabel("Score")
    ax.set_title("Temporal drift: model performance over time (test set)")
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(axis="x", rotation=45)

    # Confidence
    ax = axes[1]
    ax.plot(monthly["month_str"], monthly["confidence"], "g-^", markersize=3)
    ax.set_ylabel("Mean confidence")
    ax.set_ylim(0.4, 1.0)
    ax.tick_params(axis="x", rotation=45)

    # Class distribution + volume
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(monthly["month_str"], monthly["borked_rate"], "r-", alpha=0.7, label="Borked rate")
    ax2.bar(monthly["month_str"], monthly["count"], alpha=0.2, color="gray", label="# reports")
    ax.set_ylabel("Borked rate")
    ax2.set_ylabel("Report count")
    ax.set_xlabel("Month")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=45)

    # Only show every Nth label
    n_months = len(monthly)
    step = max(1, n_months // 20)
    for ax_i in axes:
        labels = ax_i.get_xticklabels()
        for i, label in enumerate(labels):
            if i % step != 0:
                label.set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_stage_interaction(cascade, X_test, y_test, path):
    """2D scatter: Stage 1 P(borked) vs Stage 2 P(oob), colored by true class."""
    p_s1 = cascade.stage1.predict_proba(X_test)
    X_s2 = cascade._prepare_stage2_input(X_test)
    p_s2 = cascade.stage2.predict_proba(X_s2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample for readability
    n = min(10000, len(y_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_test), n, replace=False)

    colors = {0: "red", 1: "orange", 2: "green"}
    for cls, name in [(0, "borked"), (1, "tinkering"), (2, "works_oob")]:
        mask = y_test[idx] == cls
        ax.scatter(p_s1[idx[mask], 0], p_s2[idx[mask], 1],
                   c=colors[cls], alpha=0.15, s=5, label=name)

    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="S1 threshold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="S2 threshold")

    # Decision regions
    ax.text(0.75, 0.75, "BORKED\n(S1 decides)", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.1))
    ax.text(0.25, 0.75, "works_oob\n(S2 decides)", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="green", alpha=0.1))
    ax.text(0.25, 0.25, "tinkering\n(S2 decides)", fontsize=12, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="orange", alpha=0.1))

    ax.set_xlabel("P(borked) — Stage 1")
    ax.set_ylabel("P(works_oob) — Stage 2")
    ax.set_title("Stage interaction: decision landscape")
    ax.legend(markerscale=5)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_error_features(cascade, X_test, y_test, path):
    """For each misclassification type, show which features differ most from correct."""
    y_pred = cascade.predict(X_test)

    # Focus on the two most impactful errors
    pairs = [
        (1, 2, "tinkering→works_oob"),
        (2, 1, "works_oob→tinkering"),
        (0, 1, "borked→tinkering"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax, (true_cls, pred_cls, title) in zip(axes, pairs):
        correct_mask = (y_test == true_cls) & (y_pred == true_cls)
        error_mask = (y_test == true_cls) & (y_pred == pred_cls)

        if correct_mask.sum() < 10 or error_mask.sum() < 10:
            ax.set_title(f"{title} (insufficient data)")
            continue

        # Compare feature means
        numeric_cols = X_test.select_dtypes(include=[np.number]).columns
        correct_means = X_test.loc[correct_mask, numeric_cols].mean()
        error_means = X_test.loc[error_mask, numeric_cols].mean()
        correct_stds = X_test.loc[correct_mask, numeric_cols].std()

        # Compute effect size (Cohen's d approximation)
        effect = (error_means - correct_means) / (correct_stds + 1e-8)
        effect = effect.replace([np.inf, -np.inf], np.nan).dropna()

        # Top features by absolute effect
        top = effect.abs().nlargest(15)
        top_vals = effect[top.index]

        colors = ["#d32f2f" if v > 0 else "#1976d2" for v in top_vals]
        ax.barh(range(len(top_vals)), top_vals.values, color=colors)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(top_vals.index, fontsize=7)
        ax.set_xlabel("Effect size (Cohen's d)")
        ax.set_title(f"{title}\n(n_correct={correct_mask.sum()}, n_error={error_mask.sum()})")
        ax.axvline(0, color="gray", linewidth=0.5)

    plt.suptitle("Feature differences: correct vs misclassified", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def print_insights(cascade, X_test, y_test, proba, ts_test, variants_test):
    """Print deep textual analysis."""
    y_pred = cascade.predict(X_test)
    confidence = proba.max(axis=1)

    print("\n" + "=" * 70)
    print("  DEEP ANALYSIS: CASCADE MODEL")
    print("=" * 70)

    # 1. Overall metrics
    print("\n─── 1. OVERALL METRICS ───")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"  F1 macro:  {f1:.4f}")
    print(f"  Accuracy:  {(y_test == y_pred).mean():.4f}")

    # Per-class AUC
    for cls, name in enumerate(CLASS_NAMES):
        y_bin = (y_test == cls).astype(int)
        auc = roc_auc_score(y_bin, proba[:, cls])
        brier = brier_score_loss(y_bin, proba[:, cls])
        print(f"  {name:15s}  AUC={auc:.4f}  Brier={brier:.4f}")

    # 2. Stage-level analysis
    print("\n─── 2. STAGE-LEVEL ANALYSIS ───")
    p_s1 = cascade.stage1.predict_proba(X_test)
    y_s1 = (y_test > 0).astype(int)
    s1_pred = (p_s1[:, 1] >= 0.5).astype(int)
    s1_acc = (y_s1 == s1_pred).mean()
    s1_auc = roc_auc_score(y_s1, p_s1[:, 1])
    s1_logloss = log_loss(y_s1, p_s1)
    print(f"  Stage 1 (borked vs works):")
    print(f"    Accuracy: {s1_acc:.4f}, AUC: {s1_auc:.4f}, LogLoss: {s1_logloss:.4f}")
    print(f"    Borked recall:    {(s1_pred[y_test == 0] == 0).mean():.4f}")
    print(f"    Borked precision: {(y_test[s1_pred == 0] == 0).mean():.4f}" if (s1_pred == 0).sum() > 0 else "    (no borked predictions)")

    X_s2 = cascade._prepare_stage2_input(X_test)
    p_s2 = cascade.stage2.predict_proba(X_s2)
    works_mask = y_test > 0
    y_s2 = (y_test[works_mask] - 1).astype(int)
    s2_pred = (p_s2[works_mask, 1] >= 0.5).astype(int)
    s2_acc = (y_s2 == s2_pred).mean()
    s2_auc = roc_auc_score(y_s2, p_s2[works_mask, 1])
    s2_logloss = log_loss(y_s2, p_s2[works_mask])
    print(f"\n  Stage 2 (tinkering vs works_oob, on works only):")
    print(f"    Accuracy: {s2_acc:.4f}, AUC: {s2_auc:.4f}, LogLoss: {s2_logloss:.4f}")
    print(f"    works_oob recall:    {(s2_pred[y_s2 == 1] == 1).mean():.4f}")
    print(f"    works_oob precision: {(y_s2[s2_pred == 1] == 1).mean():.4f}" if (s2_pred == 1).sum() > 0 else "")

    # 3. Error cascade analysis
    print("\n─── 3. ERROR CASCADE ───")
    cm = confusion_matrix(y_test, y_pred)
    total_errors = (y_test != y_pred).sum()
    print(f"  Total errors: {total_errors} ({total_errors / len(y_test):.1%})")

    error_pairs = [
        ((1, 2), "tinkering→works_oob"),
        ((2, 1), "works_oob→tinkering"),
        ((0, 1), "borked→tinkering"),
        ((1, 0), "tinkering→borked"),
        ((0, 2), "borked→works_oob"),
        ((2, 0), "works_oob→borked"),
    ]
    for (t, p), label in error_pairs:
        n = cm[t, p]
        pct = n / total_errors * 100
        print(f"    {label:25s}  {n:5d} ({pct:5.1f}% of errors)")

    # Stage attribution
    s1_errors = (y_s1 != s1_pred).sum()
    s1_borked_miss = ((y_test == 0) & (s1_pred == 1)).sum()
    s1_false_borked = ((y_test > 0) & (s1_pred == 0)).sum()
    print(f"\n  Stage 1 errors: {s1_errors} (miss borked: {s1_borked_miss}, false borked: {s1_false_borked})")
    print(f"  Stage 2 errors (on works): {(y_s2 != s2_pred).sum()}")

    # 4. Confidence analysis
    print("\n─── 4. CONFIDENCE ANALYSIS ───")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = confidence >= t
        if mask.sum() > 0:
            acc = (y_test[mask] == y_pred[mask]).mean()
            f1_t = f1_score(y_test[mask], y_pred[mask], average="macro", zero_division=0)
            print(f"    conf ≥ {t:.1f}:  coverage={mask.mean():.1%}, "
                  f"accuracy={acc:.1%}, F1={f1_t:.4f}")

    # 5. Uncertainty zone
    print("\n─── 5. UNCERTAINTY ZONES ───")
    s1_uncertain = (p_s1[:, 0] >= 0.3) & (p_s1[:, 0] < 0.7)
    print(f"  Stage 1 uncertain (P(borked) in [0.3, 0.7]): {s1_uncertain.mean():.1%}")
    if s1_uncertain.sum() > 0:
        print(f"    Accuracy in zone: {(y_test[s1_uncertain] == y_pred[s1_uncertain]).mean():.1%}")
        print(f"    True class distribution: {dict(zip(*np.unique(y_test[s1_uncertain], return_counts=True)))}")

    s2_uncertain = (p_s2[:, 1] >= 0.35) & (p_s2[:, 1] < 0.65)
    s2_unc_on_works = s2_uncertain[works_mask]
    print(f"\n  Stage 2 uncertain (P(oob) in [0.35, 0.65]): {s2_unc_on_works.mean():.1%} of works")
    if s2_unc_on_works.sum() > 0:
        y_unc = y_test[works_mask][s2_unc_on_works]
        p_unc = (p_s2[works_mask][s2_unc_on_works, 1] >= 0.5).astype(int)
        y_unc_bin = (y_unc - 1).astype(int)
        print(f"    Accuracy in zone: {(y_unc_bin == p_unc).mean():.1%}")

    # 6. Per-variant analysis
    print("\n─── 6. PER-VARIANT PERFORMANCE ───")
    df_var = pd.DataFrame({"true": y_test, "pred": y_pred, "variant": variants_test})
    for v in ["official", "ge", "experimental", "native", "notListed", "older"]:
        mask = df_var["variant"] == v
        if mask.sum() < 50:
            continue
        sub = df_var[mask]
        acc = (sub["true"] == sub["pred"]).mean()
        f1_v = f1_score(sub["true"], sub["pred"], average="macro", zero_division=0)
        dist = sub["true"].value_counts(normalize=True).sort_index()
        print(f"  {v:15s}  n={len(sub):5d}  acc={acc:.3f}  F1={f1_v:.3f}  "
              f"borked={dist.get(0, 0):.1%} tink={dist.get(1, 0):.1%} oob={dist.get(2, 0):.1%}")

    # 7. Bottleneck identification
    print("\n─── 7. BOTTLENECK IDENTIFICATION ───")
    print(f"  Stage 1 LogLoss: {s1_logloss:.4f}  (lower = better)")
    print(f"  Stage 2 LogLoss: {s2_logloss:.4f}  (lower = better)")
    bottleneck = "Stage 2" if s2_logloss > s1_logloss else "Stage 1"
    print(f"  → Bottleneck: **{bottleneck}** (higher logloss)")

    if s2_logloss > s1_logloss:
        print(f"\n  Stage 2 is the weak link — tinkering vs works_oob is inherently ambiguous.")
        print(f"  Stage 2 AUC={s2_auc:.4f} vs Stage 1 AUC={s1_auc:.4f}")
        print(f"  The {s2_logloss / s1_logloss:.1f}x higher logloss confirms this.")

    # Feature importance comparison
    print("\n─── 8. TOP FEATURES BY STAGE ───")
    for stage_name, model in [("Stage 1", cascade.stage1), ("Stage 2", cascade.stage2)]:
        features = model.feature_name_
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        print(f"\n  {stage_name} top-10:")
        for i in top_idx:
            print(f"    {features[i]:35s}  gain={importances[i]:.0f}")

    return f1, s1_logloss, s2_logloss, s1_auc, s2_auc


def main():
    cascade, X_train, y_train, X_test, y_test, ts_test, app_ids_test, variants_test = load_and_train()

    proba = cascade.predict_proba(X_test)
    y_pred = cascade.predict(X_test)

    # Generate all plots
    print("\nGenerating visualizations...")

    plot_confusion_matrix(y_test, y_pred, "Cascade Classifier", PLOT_DIR / "cascade_confusion.png")
    print("  ✓ cascade_confusion.png")

    plot_feature_importance(cascade.stage1, "Stage 1: Feature Importance (borked vs works)",
                           PLOT_DIR / "cascade_importance_s1.png")
    plot_feature_importance(cascade.stage2, "Stage 2: Feature Importance (tinkering vs oob)",
                           PLOT_DIR / "cascade_importance_s2.png")
    print("  ✓ cascade_importance_s1.png, cascade_importance_s2.png")

    plot_calibration(y_test, proba, PLOT_DIR / "cascade_calibration.png")
    print("  ✓ cascade_calibration.png")

    plot_confidence_accuracy(y_test, proba, PLOT_DIR / "cascade_confidence.png")
    print("  ✓ cascade_confidence.png")

    plot_stage_probability_distributions(cascade, X_test, y_test, PLOT_DIR / "cascade_stage_distributions.png")
    print("  ✓ cascade_stage_distributions.png")

    plot_roc_pr_curves(y_test, proba, PLOT_DIR / "cascade_roc_pr.png")
    print("  ✓ cascade_roc_pr.png")

    plot_error_analysis_by_variant(y_test, y_pred, variants_test, PLOT_DIR / "cascade_variant_errors.png")
    print("  ✓ cascade_variant_errors.png")

    plot_temporal_drift(y_test, y_pred, proba, ts_test, PLOT_DIR / "cascade_temporal.png")
    print("  ✓ cascade_temporal.png")

    plot_stage_interaction(cascade, X_test, y_test, PLOT_DIR / "cascade_stage_interaction.png")
    print("  ✓ cascade_stage_interaction.png")

    plot_per_class_error_features(cascade, X_test, y_test, PLOT_DIR / "cascade_error_features.png")
    print("  ✓ cascade_error_features.png")

    # Textual analysis
    f1, s1_ll, s2_ll, s1_auc, s2_auc = print_insights(
        cascade, X_test, y_test, proba, ts_test, variants_test
    )

    # Recommendations
    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)

    print("""
  1. STAGE 2 IS THE BOTTLENECK
     LogLoss Stage2/Stage1 ratio shows Stage 2 is the harder problem.
     tinkering vs works_oob is semantically ambiguous — users who say
     "works fine" may have configured something without realizing it.

  2. LABEL NOISE IS THE CEILING
     Many "tinkering" reports describe setups that "just worked" after
     choosing the right Proton version (is that tinkering or OOB?).
     Consider: relabel reports where the only "tinkering" was version selection.

  3. CONFIDENCE-BASED DEPLOYMENT
     Use confidence thresholds for production:
     - conf ≥ 0.8: show as "confident prediction"
     - conf 0.6-0.8: show as "likely" with caveat
     - conf < 0.6: show "uncertain, check reports"
     This maximizes user trust.

  4. VARIANT-SPECIFIC INSIGHTS
     "native" and "notListed" variants likely have different distributions.
     Consider: variant-specific threshold tuning or separate models.

  5. TEMPORAL DRIFT
     If performance degrades over time, consider periodic retraining
     or adding temporal features like "months since game release".

  6. NEXT IMPROVEMENT VECTORS (by expected ROI)
     a) LLM-based text features (PLAN_ML_5_LLM.md) — extract structured
        sentiment, effort level, playability from free text
     b) Increase GitHub issues coverage via broader matching
     c) Rethink label definition: 3-class might not be the right granularity
""")


if __name__ == "__main__":
    main()
