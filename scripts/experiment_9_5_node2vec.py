#!/usr/bin/env python3
"""Phase 9.5 P10: Node2Vec embeddings on game-GPU-variant graph.

Builds a tripartite graph:
  - Nodes: games (app_id), GPU families, variants
  - Edges: each report creates edges game↔gpu_family and game↔variant
  - Edge weights: verdict score (borked=0, tinkering=0.5, oob=1.0)

Node2Vec random walks capture higher-order neighborhood structure
that SVD (linear) might miss.

Experiment:
  1. Build graph from training data only
  2. Run Node2Vec → embeddings for games, GPUs, variants
  3. Add as features alongside existing SVD embeddings
  4. Evaluate cascade pipeline
"""

import gc
import sqlite3
import sys
import time
from pathlib import Path

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from pecanpy import pecanpy
from gensim.models import Word2Vec
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
    params = dict(S2_PARAMS)
    y_smooth = y_tr_labels.copy()
    if alpha > 0:
        y_smooth = y_smooth * (1 - alpha) + (1 - y_smooth) * alpha
    ds_train = lgb.Dataset(X_tr, label=y_smooth, categorical_feature=cat_cols)
    ds_test = lgb.Dataset(X_te, label=y_te, categorical_feature=cat_cols)
    model = lgb.train(
        params, ds_train, num_boost_round=3000, valid_sets=[ds_test],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=500)],
    )
    return model


def run_cascade_eval(s1, s2, s2_dropped, X_eval, y_eval, label=""):
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


def run_full_cascade(s1, X_train, y_train, X_test_es, y_test_es, X_eval, y_eval, label=""):
    X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols, drops = _prepare_stage2(
        X_train, y_train, X_test_es, y_test_es
    )
    s2 = _train_s2_booster(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols)
    r = run_cascade_eval(s1, s2, drops, X_eval, y_eval, label)
    del s2
    gc.collect()
    return r


# ═══════════════════════════════════════════════════════════════════
# Node2Vec graph building
# ═══════════════════════════════════════════════════════════════════

def build_node2vec_embeddings(
    conn,
    app_ids_train: np.ndarray,
    gpu_families_train: np.ndarray,
    variants_train: np.ndarray,
    y_train: np.ndarray,
    n_dims: int = 32,
    walk_length: int = 40,
    num_walks: int = 20,
    p: float = 1.0,
    q: float = 0.5,
    window: int = 5,
    workers: int = 8,
) -> dict:
    """Build Node2Vec embeddings from tripartite graph.

    Graph structure:
      - Game nodes: "g_{app_id}"
      - GPU nodes: "gpu_{family}"
      - Variant nodes: "var_{variant}"
      - Edges: game↔gpu and game↔variant, weighted by verdict score

    Returns dict with game_emb, gpu_emb, var_emb lookups.
    """
    import tempfile

    console.print(f"  Building tripartite graph...")
    t0 = time.time()

    # Score mapping: borked=0.1 (low but nonzero), tinkering=0.5, oob=1.0
    score_map = {0: 0.1, 1: 0.5, 2: 1.0}

    # Aggregate edges: (node1, node2) → list of scores
    edge_scores = {}
    n_edges_raw = 0
    for i in range(len(app_ids_train)):
        app_id = app_ids_train[i]
        gpu = gpu_families_train[i]
        var = variants_train[i]
        score = score_map[y_train[i]]

        game_node = f"g_{app_id}"

        if pd.notna(gpu) and gpu:
            gpu_node = f"gpu_{gpu}"
            key = (game_node, gpu_node)
            edge_scores.setdefault(key, []).append(score)
            n_edges_raw += 1

        if pd.notna(var) and var:
            var_node = f"var_{var}"
            key = (game_node, var_node)
            edge_scores.setdefault(key, []).append(score)
            n_edges_raw += 1

    # Build weighted edge list (mean score as weight)
    # Write to temp file for pecanpy (tab-delimited edgelist)
    tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.edg', delete=False)
    n_nodes = set()
    for (n1, n2), scores in edge_scores.items():
        weight = np.mean(scores)
        tmpfile.write(f"{n1}\t{n2}\t{weight:.4f}\n")
        n_nodes.add(n1)
        n_nodes.add(n2)
    tmpfile.close()

    n_games = sum(1 for n in n_nodes if n.startswith("g_"))
    n_gpus = sum(1 for n in n_nodes if n.startswith("gpu_"))
    n_vars = sum(1 for n in n_nodes if n.startswith("var_"))

    console.print(f"  Graph: {len(n_nodes)} nodes ({n_games} games, {n_gpus} GPUs, {n_vars} variants), "
                  f"{len(edge_scores)} edges ({n_edges_raw} raw)")
    console.print(f"  Graph built in {time.time()-t0:.1f}s")

    # Run Node2Vec via pecanpy
    console.print(f"  Running Node2Vec (dims={n_dims}, walks={num_walks}×{walk_length}, p={p}, q={q})...")
    t1 = time.time()

    g = pecanpy.SparseOTF(p=p, q=q, workers=workers)
    g.read_edg(tmpfile.name, weighted=True, directed=False)
    walks = g.simulate_walks(num_walks=num_walks, walk_length=walk_length)

    console.print(f"  Walks generated in {time.time()-t1:.1f}s ({len(walks)} walks)")

    # Train Word2Vec on walks
    t2 = time.time()
    model = Word2Vec(
        walks, vector_size=n_dims, window=window,
        min_count=1, sg=1, workers=workers, epochs=5,
    )
    console.print(f"  Word2Vec trained in {time.time()-t2:.1f}s")

    # Extract embeddings by node type
    game_emb = {}
    gpu_emb = {}
    var_emb = {}

    for node in model.wv.index_to_key:
        vec = model.wv[node]
        if node.startswith("g_"):
            game_emb[int(node[2:])] = vec
        elif node.startswith("gpu_"):
            gpu_emb[node[4:]] = vec
        elif node.startswith("var_"):
            var_emb[node[4:]] = vec

    console.print(f"  Embeddings: {len(game_emb)} games, {len(gpu_emb)} GPUs, {len(var_emb)} variants")

    # Cleanup
    import os
    os.unlink(tmpfile.name)

    return {
        "game_emb": game_emb,
        "gpu_emb": gpu_emb,
        "var_emb": var_emb,
        "n_dims": n_dims,
    }


def add_n2v_features(
    X: pd.DataFrame,
    app_ids: pd.Series,
    gpu_families: pd.Series,
    variants: pd.Series,
    n2v_data: dict,
    prefix: str = "n2v",
) -> pd.DataFrame:
    """Add Node2Vec embedding features to DataFrame."""
    n_dims = n2v_data["n_dims"]
    game_emb = n2v_data["game_emb"]
    gpu_emb = n2v_data["gpu_emb"]
    var_emb = n2v_data["var_emb"]

    emb_frames = []

    # Game embeddings
    game_arr = np.full((len(X), n_dims), np.nan)
    for i in range(len(X)):
        app = app_ids.iloc[i]
        if app in game_emb:
            game_arr[i] = game_emb[app]
    emb_frames.append(pd.DataFrame(
        game_arr, columns=[f"{prefix}_game_{d}" for d in range(n_dims)], index=X.index,
    ))

    # GPU embeddings
    gpu_arr = np.full((len(X), n_dims), np.nan)
    for i in range(len(X)):
        gpu = gpu_families.iloc[i]
        if pd.notna(gpu) and gpu in gpu_emb:
            gpu_arr[i] = gpu_emb[gpu]
    emb_frames.append(pd.DataFrame(
        gpu_arr, columns=[f"{prefix}_gpu_{d}" for d in range(n_dims)], index=X.index,
    ))

    # Variant embeddings
    var_arr = np.full((len(X), n_dims), np.nan)
    for i in range(len(X)):
        var = variants.iloc[i]
        if pd.notna(var) and var in var_emb:
            var_arr[i] = var_emb[var]
    emb_frames.append(pd.DataFrame(
        var_arr, columns=[f"{prefix}_var_{d}" for d in range(n_dims)], index=X.index,
    ))

    result = pd.concat([X] + emb_frames, axis=1)

    n_game = np.isfinite(game_arr[:, 0]).sum()
    n_gpu = np.isfinite(gpu_arr[:, 0]).sum()
    n_var = np.isfinite(var_arr[:, 0]).sum()
    console.print(f"    N2V features: {3*n_dims} dims. "
                  f"Coverage: game={n_game}/{len(X)}, gpu={n_gpu}/{len(X)}, var={n_var}/{len(X)}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.noise import find_noisy_samples
    from protondb_settings.ml.relabeling import apply_relabeling, get_relabel_ids
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split

    db_path = Path("data/protondb.db")
    emb_path = Path("data/embeddings.npz")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

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

    # Auxiliary columns
    console.print("[bold]Loading auxiliary columns...[/bold]")
    aux_df = pd.read_sql_query("SELECT id, app_id, variant, gpu FROM reports", conn)
    aux_df = aux_df.set_index("id").reindex(report_ids).reset_index()
    app_ids = aux_df["app_id"]
    variants = aux_df["variant"]
    gpu_families = X["gpu_family"] if "gpu_family" in X.columns else pd.Series(np.nan, index=X.index)

    del emb_data
    gc.collect()

    # Relabeling
    relabel_ids = get_relabel_ids(conn)

    # Split
    console.print("[bold]Time-based split...[/bold]")
    X_train, X_test, y_train, y_test, train_rids, _ = _time_based_split(
        X, y, timestamps, test_fraction=0.2, report_ids=report_ids,
    )

    sorted_indices = np.argsort(timestamps)
    split_point = int(len(sorted_indices) * 0.8)
    train_idx = sorted_indices[:split_point]
    test_idx = sorted_indices[split_point:]

    app_ids_train = app_ids.iloc[train_idx].reset_index(drop=True)
    app_ids_test = app_ids.iloc[test_idx].reset_index(drop=True)
    variants_train = variants.iloc[train_idx].reset_index(drop=True)
    variants_test = variants.iloc[test_idx].reset_index(drop=True)
    gpu_families_train = gpu_families.iloc[train_idx].reset_index(drop=True)
    gpu_families_test = gpu_families.iloc[test_idx].reset_index(drop=True)

    del X, y, timestamps, report_ids
    gc.collect()

    y_train, n_relabeled = apply_relabeling(y_train, train_rids, relabel_ids)
    console.print(f"  Relabeled {n_relabeled}")

    # Cleanlab
    console.print("[bold]Cleanlab noise removal (3%)...[/bold]")
    keep_mask = find_noisy_samples(X_train, y_train, frac_remove=0.03, cache_dir="data/")
    n_removed = (~keep_mask).sum()
    X_train = X_train[keep_mask].reset_index(drop=True)
    y_train = y_train[keep_mask]
    app_ids_train = app_ids_train[keep_mask].reset_index(drop=True)
    variants_train = variants_train[keep_mask].reset_index(drop=True)
    gpu_families_train = gpu_families_train[keep_mask].reset_index(drop=True)
    console.print(f"  Removed {n_removed} noisy samples")

    # ES/eval split
    n_cal = len(X_test) // 2
    X_test_es = X_test.iloc[:n_cal].copy().reset_index(drop=True)
    y_test_es = y_test[:n_cal]
    X_eval = X_test.iloc[n_cal:].copy().reset_index(drop=True)
    y_eval = y_test[n_cal:]

    app_ids_es = app_ids_test.iloc[:n_cal].reset_index(drop=True)
    app_ids_eval = app_ids_test.iloc[n_cal:].reset_index(drop=True)
    variants_es = variants_test.iloc[:n_cal].reset_index(drop=True)
    variants_eval = variants_test.iloc[n_cal:].reset_index(drop=True)
    gpu_families_es = gpu_families_test.iloc[:n_cal].reset_index(drop=True)
    gpu_families_eval = gpu_families_test.iloc[n_cal:].reset_index(drop=True)

    del X_test, y_test
    gc.collect()

    console.print(f"  Train: {len(X_train)}, ES: {len(X_test_es)}, Eval: {len(X_eval)}")

    for col in CATEGORICAL_FEATURES:
        for df in [X_train, X_test_es, X_eval]:
            if col in df.columns:
                df[col] = df[col].astype("category")

    # ── Baseline ──
    console.print("\n[bold cyan]Training baseline Stage 1...[/bold cyan]")
    t0 = time.time()
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es)
    console.print(f"  Stage 1: {s1.best_iteration_} iters ({time.time()-t0:.0f}s)")

    baseline_r = run_full_cascade(s1, X_train, y_train, X_test_es, y_test_es,
                                   X_eval, y_eval, "baseline")
    console.print(f"  baseline: F1={baseline_r['f1_eval']:.4f} oob={baseline_r['f1_works_oob']:.3f}")
    all_results = [baseline_r]

    # ── Build Node2Vec embeddings ──
    console.print("\n[bold cyan]P10: Node2Vec[/bold cyan]")

    for n_dims in [16, 32]:
        console.print(f"\n  [bold]Node2Vec dims={n_dims}[/bold]")

        n2v_data = build_node2vec_embeddings(
            conn,
            app_ids_train.values,
            gpu_families_train.values,
            variants_train.values,
            y_train,
            n_dims=n_dims,
            walk_length=40,
            num_walks=20,
            p=1.0,
            q=0.5,
        )

        # Add N2V features to all sets
        X_tr_n2v = add_n2v_features(X_train.copy(), app_ids_train, gpu_families_train,
                                     variants_train, n2v_data)
        X_es_n2v = add_n2v_features(X_test_es.copy(), app_ids_es, gpu_families_es,
                                     variants_es, n2v_data)
        X_ev_n2v = add_n2v_features(X_eval.copy(), app_ids_eval, gpu_families_eval,
                                     variants_eval, n2v_data)

        console.print(f"  Training Stage 1 with N2V dims={n_dims}...")
        s1_n2v = train_stage1(X_tr_n2v.copy(), y_train, X_es_n2v.copy(), y_test_es)
        r = run_full_cascade(s1_n2v, X_tr_n2v, y_train, X_es_n2v, y_test_es,
                              X_ev_n2v, y_eval, f"N2V d={n_dims}")
        all_results.append(r)
        console.print(f"  N2V d={n_dims}: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")

        # Feature importance
        console.print(f"  Stage 1 top-10 features:")
        importances = s1_n2v.feature_importances_
        feat_names = s1_n2v.feature_name_
        top_idx = np.argsort(importances)[-10:][::-1]
        for i in top_idx:
            console.print(f"    {feat_names[i]:30s} {importances[i]:>10.0f}")

        # N2V-only features gain
        n2v_cols = [c for c in feat_names if c.startswith("n2v_")]
        n2v_gain = sum(importances[feat_names.index(c)] for c in n2v_cols if c in feat_names)
        total_gain = sum(importances)
        console.print(f"  N2V features: {len(n2v_cols)} cols, {n2v_gain/total_gain*100:.1f}% of total gain")

        del s1_n2v, X_tr_n2v, X_es_n2v, X_ev_n2v, n2v_data
        gc.collect()

    # ── N2V replacing SVD (not additive) ──
    console.print("\n  [bold]Node2Vec d=32 REPLACING SVD embeddings[/bold]")
    n2v_data = build_node2vec_embeddings(
        conn, app_ids_train.values, gpu_families_train.values,
        variants_train.values, y_train, n_dims=32,
    )

    # Drop existing SVD embeddings, add N2V
    svd_cols = [c for c in X_train.columns if c.startswith(("gpu_emb_", "game_emb_"))]
    console.print(f"  Dropping {len(svd_cols)} SVD columns, adding N2V...")

    X_tr_rep = X_train.drop(columns=svd_cols).copy()
    X_es_rep = X_test_es.drop(columns=svd_cols).copy()
    X_ev_rep = X_eval.drop(columns=svd_cols).copy()

    X_tr_rep = add_n2v_features(X_tr_rep, app_ids_train, gpu_families_train,
                                 variants_train, n2v_data)
    X_es_rep = add_n2v_features(X_es_rep, app_ids_es, gpu_families_es,
                                 variants_es, n2v_data)
    X_ev_rep = add_n2v_features(X_ev_rep, app_ids_eval, gpu_families_eval,
                                 variants_eval, n2v_data)

    s1_rep = train_stage1(X_tr_rep.copy(), y_train, X_es_rep.copy(), y_test_es)
    r = run_full_cascade(s1_rep, X_tr_rep, y_train, X_es_rep, y_test_es,
                          X_ev_rep, y_eval, "N2V replace SVD")
    all_results.append(r)
    console.print(f"  N2V replace: F1={r['f1_eval']:.4f} oob={r['f1_works_oob']:.3f}")

    del s1_rep, X_tr_rep, X_es_rep, X_ev_rep, n2v_data
    gc.collect()

    # ── Summary ──
    console.print("\n")
    table = Table(title="P10 Node2Vec Results")
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
            r["label"], f"{r['f1_eval']:.4f}", delta_str,
            f"{r['f1_borked']:.3f}", f"{r['f1_tinkering']:.3f}",
            f"{r['f1_works_oob']:.3f}", f"{r['accuracy']:.4f}",
            style=style,
        )
    console.print(table)
    conn.close()


if __name__ == "__main__":
    main()
