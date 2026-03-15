"""Generate visualization plots for the final pipeline.

Saves plots to data/plots/ directory.

Usage:
  python scripts/visualize_plots.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COLORS = {"borked": "#e74c3c", "tinkering": "#f39c12", "works_oob": "#2ecc71"}
PLOT_DIR = Path("data/plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features,
    )
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y_raw, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y_raw, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)
    theta, difficulty = fit_irt(conn)
    X_train = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_test = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    X_train = add_error_targeted_features(X_train, train_rids, conn)
    X_test = add_error_targeted_features(X_test, test_rids, conn)
    y_train, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)

    report_meta = {}
    for r in conn.execute("""
        SELECT id, app_id, gpu, variant, timestamp,
               CASE WHEN gpu LIKE '%anGogh%' OR gpu LIKE '%an Gogh%'
                    OR battery_performance IS NOT NULL THEN 1 ELSE 0 END as is_deck
        FROM reports
    """).fetchall():
        gpu = r["gpu"] or ""
        vendor = "nvidia" if any(k in gpu.lower() for k in ("nvidia","geforce","gtx","rtx")) \
            else "amd" if any(k in gpu.lower() for k in ("amd","radeon","rx ")) \
            else "intel" if "intel" in gpu.lower() else "other"
        report_meta[r["id"]] = {
            "app_id": r["app_id"], "vendor": vendor,
            "is_deck": bool(r["is_deck"]), "variant": r["variant"],
        }
    conn.close()

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    logger.info("Training model...")
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_drops)
    y_pred = cascade.predict(X_test)
    y_proba = cascade.predict_proba(X_test)
    confidence = y_proba.max(axis=1)
    labels = ["borked", "tinkering", "works_oob"]
    errors = y_pred != y_test

    # Per-game aggregation
    game_groups = defaultdict(lambda: {"preds": [], "truths": []})
    for i, rid in enumerate(test_rids):
        app_id = report_meta.get(rid, {}).get("app_id")
        if app_id:
            game_groups[app_id]["preds"].append(y_pred[i])
            game_groups[app_id]["truths"].append(y_test[i])

    agg_true, agg_pred, agg_sizes = [], [], []
    for data in game_groups.values():
        agg_true.append(Counter(data["truths"]).most_common(1)[0][0])
        agg_pred.append(Counter(data["preds"]).most_common(1)[0][0])
        agg_sizes.append(len(data["preds"]))
    agg_true, agg_pred, agg_sizes = np.array(agg_true), np.array(agg_pred), np.array(agg_sizes)

    logger.info("Generating plots...")

    # ── 1. Cumulative progress ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    phases = ["Phase 11\nBaseline", "+IRT\n(Phase 12)", "+Relabel\n(Phase 13)",
              "+ClassWeight\n(Phase 16)", "+HP Tuning\n(Phase 17)", "Per-game\nvote (21)"]
    f1s = [0.7245, 0.7545, 0.7711, 0.7776, 0.7801, 0.8712]
    colors_bar = ["#95a5a6"] * 5 + ["#2ecc71"]
    bars = ax.bar(phases, f1s, color=colors_bar, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("F1 Macro", fontsize=12)
    ax.set_title("Cumulative ML Pipeline Progress", fontsize=14, fontweight="bold")
    ax.set_ylim(0.7, 0.9)
    ax.axhline(y=0.7245, color="#e74c3c", linestyle="--", alpha=0.5, label="Original baseline")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "01_cumulative_progress.png", dpi=150)
    plt.close()

    # ── 2. Confusion matrices (per-report + per-game) ────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, y_t, y_p, title in [
        (axes[0], y_test, y_pred, "Per-Report (F1=0.780)"),
        (axes[1], agg_true, agg_pred, f"Per-Game Vote (F1={f1_score(agg_true, agg_pred, average='macro'):.3f})"),
    ]:
        cm = confusion_matrix(y_t, y_p)
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_pct, cmap="YlOrRd", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                color = "white" if cm_pct[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.0%})", ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "02_confusion_matrices.png", dpi=150)
    plt.close()

    # ── 3. Confidence distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidence[~errors], bins=50, alpha=0.7, color="#2ecc71", label=f"Correct ({(~errors).sum()})", density=True)
    ax.hist(confidence[errors], bins=50, alpha=0.7, color="#e74c3c", label=f"Errors ({errors.sum()})", density=True)
    ax.axvline(x=0.7, color="black", linestyle="--", alpha=0.5, label="Threshold 0.7")
    ax.set_xlabel("Model Confidence", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Errors", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "03_confidence_distribution.png", dpi=150)
    plt.close()

    # ── 4. Feature importance (Stage 2) ──────────────────────────────
    s2_imp = pd.Series(s2.feature_importance(importance_type="gain"),
                       index=s2.feature_name()).sort_values(ascending=True)
    top20 = s2_imp.tail(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_fi = []
    for f in top20.index:
        if f.startswith("irt_") or f == "contributor_consistency":
            colors_fi.append("#3498db")
        elif f.startswith("game_emb"):
            colors_fi.append("#9b59b6")
        elif f == "variant":
            colors_fi.append("#e74c3c")
        elif f == "game_verdict_agreement":
            colors_fi.append("#1abc9c")
        else:
            colors_fi.append("#95a5a6")
    ax.barh(range(len(top20)), top20.values, color=colors_fi)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_xlabel("Gain", fontsize=12)
    ax.set_title("Stage 2 Feature Importance (Top 20)", fontsize=14, fontweight="bold")
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="Proton variant"),
        mpatches.Patch(color="#3498db", label="IRT / contributor"),
        mpatches.Patch(color="#9b59b6", label="Game embeddings"),
        mpatches.Patch(color="#1abc9c", label="Agreement"),
        mpatches.Patch(color="#95a5a6", label="Other"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "04_feature_importance_stage2.png", dpi=150)
    plt.close()

    # ── 5. IRT parameter distributions ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    theta_vals = np.array(list(theta.values()))
    axes[0].hist(theta_vals, bins=60, color="#3498db", alpha=0.8, edgecolor="white")
    axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[0].axvline(x=theta_vals.mean(), color="#e74c3c", linestyle="-", alpha=0.7,
                    label=f"mean={theta_vals.mean():.2f}")
    axes[0].set_xlabel("θ (strictness)", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Contributor Strictness (θ)", fontsize=13, fontweight="bold")
    axes[0].legend()

    diff_vals = np.array(list(difficulty.values()))
    axes[1].hist(diff_vals, bins=60, color="#e67e22", alpha=0.8, edgecolor="white")
    axes[1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[1].axvline(x=diff_vals.mean(), color="#e74c3c", linestyle="-", alpha=0.7,
                    label=f"mean={diff_vals.mean():.2f}")
    axes[1].set_xlabel("d (difficulty)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Game×GPU Difficulty (d)", fontsize=13, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "05_irt_distributions.png", dpi=150)
    plt.close()

    # ── 6. Per-game F1 by report count ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    buckets = [(1,2,"1-2"), (3,5,"3-5"), (6,10,"6-10"), (11,20,"11-20"), (21,50,"21-50"), (51,9999,"51+")]
    bucket_f1s, bucket_accs, bucket_labels, bucket_ns = [], [], [], []
    for mn, mx, label in buckets:
        mask = (agg_sizes >= mn) & (agg_sizes <= mx)
        if mask.sum() > 10:
            bucket_f1s.append(f1_score(agg_true[mask], agg_pred[mask], average="macro"))
            bucket_accs.append(accuracy_score(agg_true[mask], agg_pred[mask]))
            bucket_labels.append(label)
            bucket_ns.append(mask.sum())

    x = np.arange(len(bucket_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, bucket_f1s, w, label="F1 Macro", color="#3498db")
    bars2 = ax.bar(x + w/2, bucket_accs, w, label="Accuracy", color="#2ecc71")
    for bar, n in zip(bars1, bucket_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Reports per Game", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Game Metrics by Report Count", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "06_per_game_by_reports.png", dpi=150)
    plt.close()

    # ── 7. Per-vendor + deck comparison ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    segments = []
    for vendor in ["nvidia", "amd", "intel"]:
        mask = np.array([report_meta.get(rid, {}).get("vendor") == vendor for rid in test_rids])
        if mask.sum() > 100:
            f = f1_score(y_test[mask], y_pred[mask], average="macro")
            segments.append((vendor.upper(), f, mask.sum()))
    # Deck
    mask_deck = np.array([report_meta.get(rid, {}).get("is_deck", False) for rid in test_rids])
    mask_desktop = ~mask_deck
    segments.append(("Deck", f1_score(y_test[mask_deck], y_pred[mask_deck], average="macro"), mask_deck.sum()))
    segments.append(("Desktop", f1_score(y_test[mask_desktop], y_pred[mask_desktop], average="macro"), mask_desktop.sum()))

    seg_names = [s[0] for s in segments]
    seg_f1s = [s[1] for s in segments]
    seg_ns = [s[2] for s in segments]
    seg_colors = ["#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"]

    bars = ax.bar(seg_names, seg_f1s, color=seg_colors[:len(segments)], edgecolor="white", linewidth=1.5)
    for bar, f, n in zip(bars, seg_f1s, seg_ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{f:.3f}\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("F1 Macro", fontsize=12)
    ax.set_title("Per-Report F1 by GPU Vendor & Device", fontsize=14, fontweight="bold")
    ax.set_ylim(0.7, 0.82)
    ax.axhline(y=f1_score(y_test, y_pred, average="macro"), color="gray", linestyle="--", alpha=0.5, label="Overall")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "07_vendor_deck_comparison.png", dpi=150)
    plt.close()

    # ── 8. Error type breakdown ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    error_types = Counter()
    for i in range(len(y_test)):
        if errors[i]:
            t = labels[y_test[i]]
            p = labels[y_pred[i]]
            error_types[f"{t} → {p}"] += 1

    sorted_errors = error_types.most_common()
    err_labels = [e[0] for e in sorted_errors]
    err_counts = [e[1] for e in sorted_errors]
    err_colors = []
    for e in err_labels:
        if "oob" in e and "tinkering" in e:
            err_colors.append("#f39c12")
        elif "borked" in e:
            err_colors.append("#e74c3c")
        else:
            err_colors.append("#95a5a6")

    ax.barh(err_labels[::-1], err_counts[::-1], color=err_colors[::-1])
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("Error Type Breakdown", fontsize=14, fontweight="bold")
    for i, (label, count) in enumerate(zip(err_labels[::-1], err_counts[::-1])):
        ax.text(count + 50, i, f"{count/sum(err_counts)*100:.0f}%", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "08_error_breakdown.png", dpi=150)
    plt.close()

    # ── 9. Per-report vs Per-game comparison ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = {
        "Per-report\n3-class": f1_score(y_test, y_pred, average="macro"),
        "Per-game\n3-class": f1_score(agg_true, agg_pred, average="macro"),
        "Per-game\nbinary": f1_score((agg_true > 0).astype(int), (agg_pred > 0).astype(int), average="macro"),
    }
    bar_colors = ["#3498db", "#2ecc71", "#27ae60"]
    bars = ax.bar(metrics.keys(), metrics.values(), color=bar_colors, edgecolor="white", linewidth=2)
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylabel("F1 Macro", fontsize=12)
    ax.set_title("Per-Report vs Per-Game Prediction Quality", fontsize=14, fontweight="bold")
    ax.set_ylim(0.6, 1.0)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "09_report_vs_game.png", dpi=150)
    plt.close()

    # ── 10. Class distribution train vs test ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    w = 0.35
    train_pcts = [(y_train == c).mean() for c in range(3)]
    test_pcts = [(y_test == c).mean() for c in range(3)]
    ax.bar(x - w/2, train_pcts, w, label="Train", color="#3498db", alpha=0.8)
    ax.bar(x + w/2, test_pcts, w, label="Test", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("Class Distribution: Train vs Test (temporal shift)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    for i in range(3):
        ax.text(i - w/2, train_pcts[i] + 0.01, f"{train_pcts[i]:.1%}", ha="center", fontsize=9)
        ax.text(i + w/2, test_pcts[i] + 0.01, f"{test_pcts[i]:.1%}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "10_class_distribution.png", dpi=150)
    plt.close()

    print(f"\nSaved 10 plots to {PLOT_DIR}/")
    for p in sorted(PLOT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
