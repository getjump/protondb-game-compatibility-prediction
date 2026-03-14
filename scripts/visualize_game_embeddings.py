#!/usr/bin/env python3
"""Visualize game embeddings via t-SNE and UMAP with verdict/engine coloring."""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
OUT_DIR = Path("data")


def load_data():
    """Load embeddings and metadata from DB."""
    emb = np.load(EMB_PATH, allow_pickle=True)
    game_embeddings = emb["game_embeddings"]  # (n_games, 16)
    game_ids = emb["game_ids"].astype(int)
    print(f"Loaded {len(game_ids)} game embeddings, dim={game_embeddings.shape[1]}")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Game names
    name_map = {}
    for row in conn.execute("SELECT app_id, name FROM games"):
        name_map[row["app_id"]] = row["name"]

    # Engine info
    engine_map = {}
    for row in conn.execute("SELECT app_id, engine FROM game_metadata WHERE engine IS NOT NULL"):
        engine_map[row["app_id"]] = row["engine"]

    # Per-game verdict stats
    verdict_rows = conn.execute("""
        SELECT app_id,
               SUM(CASE WHEN verdict_oob = 'yes' THEN 1 ELSE 0 END) as works,
               SUM(CASE WHEN verdict = 'no' THEN 1 ELSE 0 END) as borked,
               COUNT(*) as total
        FROM reports
        GROUP BY app_id
        HAVING total >= 3
    """).fetchall()

    verdict_map = {}
    for row in verdict_rows:
        total = row["total"]
        works_pct = row["works"] / total
        borked_pct = row["borked"] / total
        tinkering_pct = 1 - works_pct - borked_pct
        if borked_pct >= works_pct and borked_pct >= tinkering_pct:
            verdict_map[row["app_id"]] = "borked"
        elif works_pct >= tinkering_pct:
            verdict_map[row["app_id"]] = "works_oob"
        else:
            verdict_map[row["app_id"]] = "tinkering"

    # Report counts
    report_counts = {}
    for row in conn.execute("SELECT app_id, count(*) as cnt FROM reports GROUP BY app_id"):
        report_counts[row["app_id"]] = row["cnt"]

    conn.close()

    # Filter to games with verdicts
    mask = np.array([gid in verdict_map for gid in game_ids])
    X = game_embeddings[mask]
    ids = game_ids[mask]
    print(f"Games with verdicts: {len(ids)}")

    verdicts = [verdict_map[gid] for gid in ids]
    names = [name_map.get(gid, str(gid)) for gid in ids]
    engines = [engine_map.get(gid) for gid in ids]
    counts = np.array([report_counts.get(gid, 1) for gid in ids])

    return X, ids, verdicts, names, engines, counts


def plot_verdict(X_2d, verdicts, names, counts, method_name, ax):
    """Plot with verdict coloring."""
    color_map = {
        "works_oob": "#2ecc71",
        "tinkering": "#f39c12",
        "borked": "#e74c3c",
    }
    sizes = np.clip(np.log1p(counts) * 3, 3, 40)

    for label, color in color_map.items():
        idx = [i for i, v in enumerate(verdicts) if v == label]
        ax.scatter(
            X_2d[idx, 0], X_2d[idx, 1],
            c=color, s=sizes[idx], alpha=0.5,
            label=f"{label} ({len(idx)})",
            edgecolors="none",
        )

    # Annotate top games
    top_n = 40
    top_indices = np.argsort(-counts)[:top_n]
    for rank, i in enumerate(top_indices):
        angle = rank * 2.399
        r = 8 + (rank % 3) * 4  # vary radius
        dx = np.cos(angle) * r
        dy = np.sin(angle) * r
        ax.annotate(
            names[i][:22],
            (X_2d[i, 0], X_2d[i, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.5, alpha=0.85,
            arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.4),
        )

    ax.set_title(f"Game Embeddings ({method_name}) — verdict", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)


def plot_engine(X_2d, engines, names, counts, method_name, ax):
    """Plot with engine coloring (top engines only)."""
    from collections import Counter

    # Top engines
    engine_counts = Counter(e for e in engines if e)
    top_engines = [e for e, _ in engine_counts.most_common(10)]
    engine_color_map = {}
    cmap = plt.cm.tab10
    for i, eng in enumerate(top_engines):
        engine_color_map[eng] = cmap(i)

    sizes = np.clip(np.log1p(counts) * 3, 3, 40)

    # "Other" engines first (background)
    other_idx = [i for i, e in enumerate(engines) if e not in engine_color_map]
    ax.scatter(
        X_2d[other_idx, 0], X_2d[other_idx, 1],
        c="#cccccc", s=sizes[other_idx], alpha=0.3,
        label=f"other ({len(other_idx)})",
        edgecolors="none",
    )

    for eng, color in engine_color_map.items():
        idx = [i for i, e in enumerate(engines) if e == eng]
        ax.scatter(
            X_2d[idx, 0], X_2d[idx, 1],
            c=[color], s=sizes[idx], alpha=0.6,
            label=f"{eng} ({len(idx)})",
            edgecolors="none",
        )

    ax.set_title(f"Game Embeddings ({method_name}) — engine", fontsize=12)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def main():
    X, ids, verdicts, names, engines, counts = load_data()

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    print(f"t-SNE done. KL divergence: {tsne.kl_divergence_:.4f}")

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)
    X_umap = reducer.fit_transform(X)
    print("UMAP done.")

    # === Figure 1: 2x2 grid (t-SNE verdict, UMAP verdict, t-SNE engine, UMAP engine) ===
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    plot_verdict(X_tsne, verdicts, names, counts, "t-SNE", axes[0, 0])
    plot_verdict(X_umap, verdicts, names, counts, "UMAP", axes[0, 1])
    plot_engine(X_tsne, engines, names, counts, "t-SNE", axes[1, 0])
    plot_engine(X_umap, engines, names, counts, "UMAP", axes[1, 1])

    fig.suptitle("Game Embeddings Visualization (SVD → 2D projection)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out1 = OUT_DIR / "game_embeddings_overview.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved overview: {out1}")
    plt.close()

    # === Figure 2: UMAP verdict — zoomed on dense region ===
    fig2, ax2 = plt.subplots(figsize=(18, 14))
    plot_verdict(X_umap, verdicts, names, counts, "UMAP", ax2)

    # Zoom to central 90% of points
    p5, p95 = np.percentile(X_umap, [5, 95], axis=0)
    margin = (p95 - p5) * 0.1
    ax2.set_xlim(p5[0] - margin[0], p95[0] + margin[0])
    ax2.set_ylim(p5[1] - margin[1], p95[1] + margin[1])
    ax2.set_title("Game Embeddings (UMAP) — zoomed, verdict coloring", fontsize=14)

    out2 = OUT_DIR / "game_embeddings_umap_zoom.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved zoomed UMAP: {out2}")
    plt.close()


if __name__ == "__main__":
    main()
