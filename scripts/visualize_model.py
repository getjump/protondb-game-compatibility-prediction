#!/usr/bin/env python3
"""Comprehensive model diagnostics: confusion matrix, calibration, SHAP, error analysis."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import build_embeddings, load_embeddings
from protondb_settings.ml.features.hardware import build_hardware_lookups
from protondb_settings.ml.models.classifier import TARGET_NAMES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/protondb.db")
MODEL_PATH = Path("data/model.pkl")
EMB_PATH = Path("data/embeddings.npz")
OUT_DIR = Path("data/plots")

CLASS_NAMES = ["borked", "needs_tinkering", "works_oob"]
CLASS_COLORS = ["#e74c3c", "#f39c12", "#2ecc71"]


def load_test_data():
    """Rebuild feature matrix and return test split + model."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    # Load embeddings
    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    # Build features
    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Coerce any remaining object columns to numeric
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_test, y_train, y_test = _time_based_split(X, y, timestamps, 0.2)

    # Convert categoricals
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_test.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Also get app_ids for error analysis
    sorted_indices = np.argsort(timestamps)
    split_point = int(len(sorted_indices) * 0.8)
    test_idx = sorted_indices[split_point:]

    # Fetch app_ids from reports
    rows = conn.execute("SELECT app_id FROM reports").fetchall()
    all_app_ids = np.array([r["app_id"] for r in rows])

    # Game names
    name_map = {}
    for row in conn.execute("SELECT app_id, name FROM games"):
        name_map[row["app_id"]] = row["name"]

    conn.close()

    return model, X_train, X_test, y_train, y_test, test_idx, all_app_ids, name_map


def plot_confusion_matrix(model, X_test, y_test):
    """1. Confusion matrix — normalized and absolute."""
    from sklearn.metrics import confusion_matrix

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Absolute
    im1 = ax1.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax1.set_yticks(range(3)); ax1.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix (absolute)")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Normalized
    im2 = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax2.text(j, i, f"{cm_norm[i, j]:.2%}", ha="center", va="center", color=color, fontsize=13)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(CLASS_NAMES, fontsize=10)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (row-normalized)")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.tight_layout()
    path = OUT_DIR / "1_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_probability_distribution(model, X_test, y_test):
    """2. Predicted probability distributions per true class."""
    y_proba = model.predict_proba(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for cls_idx in range(3):
        ax = axes[cls_idx]
        for true_cls in range(3):
            mask = y_test == true_cls
            probs = y_proba[mask, cls_idx]
            ax.hist(probs, bins=50, alpha=0.6, label=f"true={CLASS_NAMES[true_cls]}",
                    color=CLASS_COLORS[true_cls], density=True)
        ax.set_xlabel(f"P({CLASS_NAMES[cls_idx]})")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of P({CLASS_NAMES[cls_idx]})")
        ax.legend(fontsize=8)

    fig.suptitle("Predicted Probability Distributions by True Class", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "2_probability_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_calibration_curves(model, X_test, y_test):
    """3. Calibration curves (reliability diagrams) per class."""
    from sklearn.calibration import calibration_curve

    y_proba = model.predict_proba(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for cls_idx in range(3):
        ax = axes[cls_idx]
        y_binary = (y_test == cls_idx).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, y_proba[:, cls_idx], n_bins=15)

        ax.plot(prob_pred, prob_true, "o-", color=CLASS_COLORS[cls_idx], label=CLASS_NAMES[cls_idx])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration: {CLASS_NAMES[cls_idx]}")
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    fig.suptitle("Calibration Curves (Reliability Diagrams)", fontsize=13)
    fig.tight_layout()
    path = OUT_DIR / "3_calibration_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_shap_dependency(model, X_test, y_test):
    """4. SHAP dependency plots for top features."""
    import shap

    explainer = shap.TreeExplainer(model)
    n_sample = min(2000, len(X_test))
    X_sample = X_test.iloc[:n_sample]
    shap_values = explainer.shap_values(X_sample)

    # Normalize shap_values to (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # list of (n_samples, n_features) per class -> stack
        shap_all = np.stack(shap_values, axis=-1)  # (n_samples, n_features, n_classes)
    else:
        shap_all = shap_values
        # Detect shape: could be (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if shap_all.ndim == 3 and shap_all.shape[0] == 3 and shap_all.shape[2] != 3:
            shap_all = np.transpose(shap_all, (1, 2, 0))  # -> (n_samples, n_features, n_classes)

    # Mean |SHAP| across classes and samples: (n_features,)
    mean_shap = np.mean(np.abs(shap_all), axis=(0, 2))
    feature_names = list(X_test.columns)

    # Only pick numeric features for dependency plots (skip categoricals)
    cat_set = set(CATEGORICAL_FEATURES)
    numeric_top_idx = [i for i in np.argsort(mean_shap)[::-1] if feature_names[i] not in cat_set]
    top_idx = numeric_top_idx[:9]

    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    for plot_i, feat_i in enumerate(top_idx):
        ax = axes[plot_i // 3][plot_i % 3]
        feat_name = feature_names[feat_i]
        feat_values = X_sample.iloc[:, feat_i].values.astype(float)

        # Filter out NaN for scatter
        valid = ~np.isnan(feat_values)

        # Plot SHAP for each class
        for cls_idx in range(3):
            sv = shap_all[:, feat_i, cls_idx]
            ax.scatter(feat_values[valid], sv[valid], alpha=0.12, s=4, c=CLASS_COLORS[cls_idx],
                       label=CLASS_NAMES[cls_idx], rasterized=True)

        ax.set_xlabel(feat_name, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(f"{feat_name} (|SHAP|={mean_shap[feat_i]:.3f})", fontsize=10)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        if plot_i == 0:
            ax.legend(fontsize=7, markerscale=3)

    fig.suptitle("SHAP Dependency Plots — Top 9 Features", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "4_shap_dependency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    return shap_values, X_sample, feature_names


def plot_shap_summary(shap_values, X_sample, feature_names):
    """5. SHAP summary bar plot (feature importance per class)."""
    # Normalize to (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_all = np.stack(shap_values, axis=-1)
    else:
        shap_all = shap_values
        if shap_all.ndim == 3 and shap_all.shape[0] == 3 and shap_all.shape[2] != 3:
            shap_all = np.transpose(shap_all, (1, 2, 0))

    # Per-class mean |SHAP|
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for cls_idx in range(3):
        ax = axes[cls_idx]
        sv = np.abs(shap_all[:, :, cls_idx]).mean(axis=0)  # (n_features,)

        top_n = min(20, len(sv))
        top_idx = np.argsort(sv)[::-1][:top_n]
        top_names = [feature_names[i] for i in top_idx]
        top_vals = sv[top_idx]

        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_vals[::-1], color=CLASS_COLORS[cls_idx], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=7)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"{CLASS_NAMES[cls_idx]}", fontsize=12)

    fig.suptitle("Feature Importance by Class (SHAP)", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "5_shap_per_class.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_error_analysis(model, X_test, y_test, test_idx, all_app_ids, name_map):
    """6. Error analysis — which games does the model struggle with."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Get app_ids for test set
    # test_idx indexes into the full filtered dataset (after _compute_target filtering)
    # But all_app_ids comes from raw reports — we need to match them
    # Simpler: analyze by error type and confidence

    errors = y_pred != y_test
    correct = ~errors

    # Confidence of wrong predictions
    wrong_confidence = np.max(y_proba[errors], axis=1) if errors.any() else np.array([])
    right_confidence = np.max(y_proba[correct], axis=1) if correct.any() else np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Confidence distribution: correct vs wrong
    ax = axes[0, 0]
    if len(right_confidence) > 0:
        ax.hist(right_confidence, bins=50, alpha=0.6, color="#2ecc71", label=f"Correct ({len(right_confidence)})", density=True)
    if len(wrong_confidence) > 0:
        ax.hist(wrong_confidence, bins=50, alpha=0.6, color="#e74c3c", label=f"Wrong ({len(wrong_confidence)})", density=True)
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Model Confidence: Correct vs Wrong Predictions")
    ax.legend()

    # 2. Error rate by predicted class
    ax = axes[0, 1]
    for cls in range(3):
        pred_mask = y_pred == cls
        if pred_mask.sum() > 0:
            err_rate = (y_test[pred_mask] != cls).mean()
            total = pred_mask.sum()
            ax.bar(cls, err_rate, color=CLASS_COLORS[cls], alpha=0.8)
            ax.text(cls, err_rate + 0.01, f"{err_rate:.1%}\n(n={total})", ha="center", fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Error rate")
    ax.set_title("Error Rate by Predicted Class")
    ax.set_ylim(0, 1)

    # 3. Misclassification flow (what gets confused with what)
    ax = axes[1, 0]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    # Show off-diagonal as bars
    pairs = []
    values = []
    colors = []
    for i in range(3):
        for j in range(3):
            if i != j:
                pairs.append(f"{CLASS_NAMES[i][:4]}→{CLASS_NAMES[j][:4]}")
                values.append(cm[i, j])
                colors.append(CLASS_COLORS[i])
    ax.barh(range(len(pairs)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("Misclassification Flows")
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, str(v), va="center", fontsize=9)

    # 4. Accuracy by confidence bin
    ax = axes[1, 1]
    max_probs = np.max(y_proba, axis=1)
    bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_counts = []
    bin_centers = []
    for b_lo, b_hi in zip(bins[:-1], bins[1:]):
        mask = (max_probs >= b_lo) & (max_probs < b_hi)
        if mask.sum() > 0:
            bin_accs.append((y_pred[mask] == y_test[mask]).mean())
            bin_counts.append(mask.sum())
            bin_centers.append((b_lo + b_hi) / 2)

    ax2 = ax.twinx()
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="#3498db", label="Accuracy")
    ax2.plot(bin_centers, bin_counts, "ro-", markersize=5, label="Count")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Accuracy", color="#3498db")
    ax2.set_ylabel("Sample count", color="red")
    ax.set_title("Accuracy by Confidence Bin")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    fig.suptitle("Error Analysis", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "6_error_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and model...")
    model, X_train, X_test, y_train, y_test, test_idx, all_app_ids, name_map = load_test_data()

    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, f1_score
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Test samples: {len(y_test)}")
    print()

    print("=== 1/6: Confusion Matrix ===")
    plot_confusion_matrix(model, X_test, y_test)

    print("=== 2/6: Probability Distributions ===")
    plot_probability_distribution(model, X_test, y_test)

    print("=== 3/6: Calibration Curves ===")
    plot_calibration_curves(model, X_test, y_test)

    print("=== 4/6: SHAP Dependency Plots ===")
    shap_values, X_sample, feature_names = plot_shap_dependency(model, X_test, y_test)

    print("=== 5/6: SHAP Per-Class Importance ===")
    plot_shap_summary(shap_values, X_sample, feature_names)

    print("=== 6/6: Error Analysis ===")
    plot_error_analysis(model, X_test, y_test, test_idx, all_app_ids, name_map)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
