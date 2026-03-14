#!/usr/bin/env python3
"""Deep analysis: t-SNE of features, correlations, interactions, error clusters."""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import umap

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import TARGET_NAMES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.WARNING)
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

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_test, y_train, y_test = _time_based_split(X, y, timestamps, 0.2)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_test.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = joblib.load(MODEL_PATH)
    conn.close()

    return model, X_train, X_test, y_train, y_test


def plot_feature_space_tsne_umap(model, X_test, y_test):
    """1. t-SNE and UMAP of full feature space — true labels vs predicted."""
    print("  Preparing numeric features...")
    # Drop categorical, fill NaN
    cat_set = set(CATEGORICAL_FEATURES)
    numeric_cols = [c for c in X_test.columns if c not in cat_set]
    X_num = X_test[numeric_cols].copy()
    X_num = X_num.fillna(0).astype(float)

    # Sample for speed
    n_sample = min(8000, len(X_num))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_num), n_sample, replace=False)
    X_sub = X_num.iloc[idx].values
    y_sub = y_test[idx]
    y_pred_sub = model.predict(X_test)[idx]

    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X_sub)

    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    print("  Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, n_jobs=-1)
    X_umap = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    for col_idx, (X_2d, method) in enumerate([(X_tsne, "t-SNE"), (X_umap, "UMAP")]):
        # True labels
        ax = axes[0, col_idx]
        for cls in range(3):
            mask = y_sub == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=CLASS_COLORS[cls],
                       s=4, alpha=0.4, label=f"{CLASS_NAMES[cls]} ({mask.sum()})", rasterized=True)
        ax.set_title(f"{method} — True Labels", fontsize=12)
        ax.legend(fontsize=8, markerscale=3)

        # Predicted labels
        ax = axes[1, col_idx]
        for cls in range(3):
            mask = y_pred_sub == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=CLASS_COLORS[cls],
                       s=4, alpha=0.4, label=f"pred {CLASS_NAMES[cls]} ({mask.sum()})", rasterized=True)
        # Highlight errors
        errors = y_pred_sub != y_sub
        ax.scatter(X_2d[errors, 0], X_2d[errors, 1], c="none", edgecolors="black",
                   s=10, alpha=0.3, linewidths=0.5, label=f"errors ({errors.sum()})")
        ax.set_title(f"{method} — Predicted (errors circled)", fontsize=12)
        ax.legend(fontsize=8, markerscale=3)

    fig.suptitle("Feature Space Projections (all numeric features)", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "7_feature_space_projections.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return X_scaled, X_tsne, X_umap, y_sub, y_pred_sub, idx


def plot_feature_correlations(X_test):
    """2. Correlation heatmap of top features."""
    cat_set = set(CATEGORICAL_FEATURES)
    numeric_cols = [c for c in X_test.columns if c not in cat_set]
    X_num = X_test[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Select top features by variance (skip near-zero)
    variances = X_num.var()
    top_cols = variances.nlargest(40).index.tolist()
    corr = X_num[top_cols].corr()

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix (top 40 by variance)", fontsize=13)

    fig.tight_layout()
    path = OUT_DIR / "8_feature_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_interactions(model, X_test):
    """3. SHAP interaction values — top feature pairs."""
    import shap

    print("  Computing SHAP interaction values (200 samples, tree_limit=200)...")
    explainer = shap.TreeExplainer(model)
    n_sample = 200
    X_sample = X_test.iloc[:n_sample].copy()
    # Encode categoricals as codes for SHAP interaction (requires numeric)
    cat_set = set(CATEGORICAL_FEATURES)
    for col in X_sample.columns:
        if col in cat_set and hasattr(X_sample[col], "cat"):
            X_sample[col] = X_sample[col].cat.codes.replace(-1, np.nan).astype(float)
        elif X_sample[col].dtype.name == "category":
            X_sample[col] = X_sample[col].cat.codes.replace(-1, np.nan).astype(float)
    # tree_limit reduces from 2000 to 200 trees — 10x speedup
    interaction_values = explainer.shap_interaction_values(X_sample, tree_limit=200)

    # interaction_values: (n_samples, n_features, n_features, n_classes) or list
    if isinstance(interaction_values, list):
        # list of (n_samples, n_features, n_features) per class
        iv_mean = np.mean([np.abs(iv).mean(axis=0) for iv in interaction_values], axis=0)
    elif interaction_values.ndim == 4:
        # (n_samples, n_features, n_features, n_classes)
        iv_mean = np.mean(np.abs(interaction_values), axis=(0, 3))
    else:
        iv_mean = np.mean(np.abs(interaction_values), axis=0)

    # Zero out diagonal (self-interaction = main effect)
    np.fill_diagonal(iv_mean, 0)

    feature_names = list(X_test.columns)

    # Top interactions
    n_feat = iv_mean.shape[0]
    pairs = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            pairs.append((iv_mean[i, j], i, j))
    pairs.sort(reverse=True)
    top_pairs = pairs[:20]

    fig, ax = plt.subplots(figsize=(14, 8))
    labels = [f"{feature_names[p[1]][:20]} × {feature_names[p[2]][:20]}" for p in top_pairs]
    values = [p[0] for p in top_pairs]
    y_pos = np.arange(len(top_pairs))
    ax.barh(y_pos, values[::-1], color="#3498db", alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP interaction|")
    ax.set_title("Top 20 Feature Interactions (SHAP)", fontsize=13)

    fig.tight_layout()
    path = OUT_DIR / "9_shap_interactions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Also plot top-3 interaction scatter plots
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    for plot_i in range(min(3, len(top_pairs))):
        ax = axes2[plot_i]
        _, fi, fj = top_pairs[plot_i]
        name_i = feature_names[fi]
        name_j = feature_names[fj]

        X_sample_np = X_sample.iloc[:, [fi, fj]].values.astype(float)
        valid = ~np.isnan(X_sample_np).any(axis=1)

        if isinstance(interaction_values, list):
            # Average across classes
            iv_ij = np.mean([iv[:, fi, fj] for iv in interaction_values], axis=0)
        elif interaction_values.ndim == 4:
            iv_ij = np.mean(interaction_values[:, fi, fj, :], axis=1)
        else:
            iv_ij = interaction_values[:, fi, fj]

        sc = ax.scatter(X_sample_np[valid, 0], X_sample_np[valid, 1],
                        c=iv_ij[valid], cmap="RdBu_r", s=8, alpha=0.7, rasterized=True)
        ax.set_xlabel(name_i[:25], fontsize=9)
        ax.set_ylabel(name_j[:25], fontsize=9)
        ax.set_title(f"Interaction: {name_i[:15]} × {name_j[:15]}", fontsize=10)
        fig2.colorbar(sc, ax=ax, label="SHAP interaction")

    fig2.suptitle("Top 3 Feature Interaction Patterns", fontsize=13)
    fig2.tight_layout()
    path2 = OUT_DIR / "10_shap_interaction_scatter.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")


def plot_error_clusters(X_tsne, X_umap, y_sub, y_pred_sub, X_test, idx):
    """4. Error analysis in 2D — what kind of errors cluster together."""
    errors = y_pred_sub != y_sub

    # Classify error types
    error_types = np.full(len(y_sub), "", dtype=object)
    error_types[~errors] = "correct"
    for true_cls in range(3):
        for pred_cls in range(3):
            if true_cls == pred_cls:
                continue
            mask = (y_sub == true_cls) & (y_pred_sub == pred_cls)
            label = f"{CLASS_NAMES[true_cls][:4]}→{CLASS_NAMES[pred_cls][:4]}"
            error_types[mask] = label

    unique_errors = sorted(set(error_types) - {"correct"})
    error_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_errors)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    for X_2d, ax, method in [(X_tsne, ax1, "t-SNE"), (X_umap, ax2, "UMAP")]:
        # Background: correct
        correct_mask = error_types == "correct"
        ax.scatter(X_2d[correct_mask, 0], X_2d[correct_mask, 1],
                   c="#dddddd", s=3, alpha=0.3, label=f"correct ({correct_mask.sum()})", rasterized=True)

        # Error types
        for i, etype in enumerate(unique_errors):
            mask = error_types == etype
            if mask.sum() == 0:
                continue
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[error_colors[i]],
                       s=8, alpha=0.6, label=f"{etype} ({mask.sum()})", rasterized=True)

        ax.set_title(f"{method} — Error Types", fontsize=12)
        ax.legend(fontsize=7, markerscale=3, loc="upper right")

    fig.suptitle("Misclassification Patterns in Feature Space", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "11_error_clusters.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_by_class(X_test, y_test, model):
    """5. Feature value distributions per class — top 12 numeric features."""
    import shap

    cat_set = set(CATEGORICAL_FEATURES)

    # Quick SHAP for feature ranking
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test.iloc[:1000])
    if isinstance(sv, list):
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        mean_shap = np.mean(np.abs(sv), axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)

    feature_names = list(X_test.columns)
    numeric_feats = [(i, feature_names[i]) for i in range(len(feature_names)) if feature_names[i] not in cat_set]
    numeric_feats.sort(key=lambda x: mean_shap[x[0]], reverse=True)
    top_12 = numeric_feats[:12]

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))

    for plot_i, (feat_i, feat_name) in enumerate(top_12):
        ax = axes[plot_i // 4][plot_i % 4]
        for cls in range(3):
            mask = y_test == cls
            vals = X_test.iloc[mask, feat_i].dropna().values.astype(float)
            if len(vals) > 0:
                # Clip outliers for better visualization
                p1, p99 = np.percentile(vals, [1, 99])
                vals_clip = vals[(vals >= p1) & (vals <= p99)]
                if len(vals_clip) > 0:
                    ax.hist(vals_clip, bins=50, alpha=0.5, density=True,
                            color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])

        ax.set_title(f"{feat_name} (SHAP={mean_shap[feat_i]:.3f})", fontsize=9)
        ax.set_xlabel(feat_name, fontsize=7)
        if plot_i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Feature Value Distributions by True Class (top 12 by SHAP)", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "12_feature_distributions_by_class.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_prediction_confidence_map(X_umap, y_sub, model, X_test, idx):
    """6. UMAP colored by prediction confidence (max probability)."""
    y_proba = model.predict_proba(X_test)
    proba_sub = y_proba[idx]
    confidence = np.max(proba_sub, axis=1)
    entropy = -np.sum(proba_sub * np.log(proba_sub + 1e-10), axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Confidence
    sc1 = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=confidence, cmap="RdYlGn",
                      s=5, alpha=0.5, vmin=0.3, vmax=1.0, rasterized=True)
    fig.colorbar(sc1, ax=ax1, label="Max P(class)")
    ax1.set_title("UMAP — Prediction Confidence", fontsize=12)

    # Entropy (uncertainty)
    sc2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=entropy, cmap="hot_r",
                      s=5, alpha=0.5, rasterized=True)
    fig.colorbar(sc2, ax=ax2, label="Prediction Entropy")
    ax2.set_title("UMAP — Prediction Uncertainty (Entropy)", fontsize=12)

    fig.suptitle("Model Confidence in Feature Space", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "13_confidence_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_temporal_analysis(X_test, y_test, model):
    """7. How model performance changes over time (test set is already sorted by time)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Split test set into 10 temporal buckets
    n_buckets = 10
    bucket_size = len(y_test) // n_buckets

    accs = []
    f1s = []
    cls_recalls = {0: [], 1: [], 2: []}
    confidences = []

    from sklearn.metrics import accuracy_score, f1_score, recall_score

    for i in range(n_buckets):
        start = i * bucket_size
        end = start + bucket_size if i < n_buckets - 1 else len(y_test)
        yp = y_pred[start:end]
        yt = y_test[start:end]
        prob = y_proba[start:end]

        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        confidences.append(np.max(prob, axis=1).mean())
        for cls in range(3):
            cls_mask = yt == cls
            if cls_mask.sum() > 0:
                cls_recalls[cls].append((yp[cls_mask] == cls).mean())
            else:
                cls_recalls[cls].append(np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    x = np.arange(n_buckets)
    labels = [f"Bucket {i+1}\n(oldest)" if i == 0 else (f"Bucket {i+1}\n(newest)" if i == n_buckets-1 else f"B{i+1}") for i in range(n_buckets)]

    # Accuracy + F1
    ax = axes[0, 0]
    ax.plot(x, accs, "bo-", label="Accuracy", markersize=6)
    ax.plot(x, f1s, "rs-", label="F1 macro", markersize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Score"); ax.set_title("Accuracy & F1 by Time Bucket")
    ax.legend(); ax.set_ylim(0, 1)

    # Per-class recall
    ax = axes[0, 1]
    for cls in range(3):
        ax.plot(x, cls_recalls[cls], "o-", color=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls], markersize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Recall"); ax.set_title("Per-class Recall by Time Bucket")
    ax.legend(); ax.set_ylim(0, 1)

    # Confidence
    ax = axes[1, 0]
    ax.plot(x, confidences, "go-", markersize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean max P"); ax.set_title("Mean Prediction Confidence by Time Bucket")
    ax.set_ylim(0.4, 1)

    # Class distribution
    ax = axes[1, 1]
    for cls in range(3):
        fracs = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < n_buckets - 1 else len(y_test)
            fracs.append((y_test[start:end] == cls).mean())
        ax.plot(x, fracs, "o-", color=CLASS_COLORS[cls], label=CLASS_NAMES[cls], markersize=6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Fraction"); ax.set_title("Class Distribution by Time Bucket")
    ax.legend(); ax.set_ylim(0, 1)

    fig.suptitle("Temporal Analysis of Model Performance (test set, oldest→newest)", fontsize=14)
    fig.tight_layout()
    path = OUT_DIR / "14_temporal_analysis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and model...")
    model, X_train, X_test, y_train, y_test = load_test_data()
    print(f"  Test: {len(y_test)} samples, {X_test.shape[1]} features\n")

    print("=== 1/7: Feature Space Projections (t-SNE + UMAP) ===")
    X_scaled, X_tsne, X_umap, y_sub, y_pred_sub, idx = plot_feature_space_tsne_umap(model, X_test, y_test)

    print("=== 2/7: Feature Correlations ===")
    plot_feature_correlations(X_test)

    print("=== 3/7: SHAP Interactions ===")
    plot_shap_interactions(model, X_test)

    print("=== 4/7: Error Clusters ===")
    plot_error_clusters(X_tsne, X_umap, y_sub, y_pred_sub, X_test, idx)

    print("=== 5/7: Feature Distributions by Class ===")
    plot_feature_by_class(X_test, y_test, model)

    print("=== 6/7: Confidence Map ===")
    plot_prediction_confidence_map(X_umap, y_sub, model, X_test, idx)

    print("=== 7/7: Temporal Analysis ===")
    plot_temporal_analysis(X_test, y_test, model)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
