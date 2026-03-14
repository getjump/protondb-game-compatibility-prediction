#!/usr/bin/env python3
"""Experiment: Text embeddings (PLAN_ML_5_EMBEDDINGS experiments E1-E5).

E1: concluding_notes embeddings (34% coverage) + SVD 16
E2: all_text embeddings (91% coverage) + SVD 16
E3: notes_verdict embeddings (74% coverage) + SVD 16
E4: Dimension sweep (8, 16, 32, 64) on best source
E5: Best embeddings + D+E features combined
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split, _compute_target
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]


def load_data():
    """Load feature matrix + text fields."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    print("Loading text fields...")
    rows = conn.execute("""
        SELECT concluding_notes, notes_verdict, notes_extra, notes_customizations,
            notes_audio_faults, notes_graphical_faults, notes_performance_faults,
            notes_stability_faults, notes_windowing_faults, notes_input_faults,
            notes_significant_bugs, notes_save_game_faults, notes_concluding_notes,
            verdict, verdict_oob
        FROM reports
    """).fetchall()
    conn.close()

    concluding = []
    verdicts = []
    all_texts = []

    for row in rows:
        target = _compute_target(row["verdict"], row["verdict_oob"])
        if target is None:
            continue

        concluding.append(row["concluding_notes"] or "")
        verdicts.append(row["notes_verdict"] or "")

        parts = [
            row["concluding_notes"], row["notes_verdict"], row["notes_extra"],
            row["notes_customizations"], row["notes_concluding_notes"],
            row["notes_audio_faults"], row["notes_graphical_faults"],
            row["notes_performance_faults"], row["notes_stability_faults"],
            row["notes_windowing_faults"], row["notes_input_faults"],
            row["notes_significant_bugs"], row["notes_save_game_faults"],
        ]
        all_texts.append(" ".join(t for t in parts if t) or "")

    assert len(concluding) == len(X), f"Mismatch: {len(concluding)} vs {len(X)}"

    return X, y, timestamps, concluding, verdicts, all_texts


def encode_texts(model: SentenceTransformer, texts: list[str], label: str) -> np.ndarray:
    """Encode texts, returning embeddings. Empty strings get zero vectors."""
    print(f"  Encoding {label}...")
    non_empty_mask = np.array([bool(t.strip()) for t in texts])
    non_empty_texts = [t for t in texts if t.strip()]
    coverage = non_empty_mask.mean()
    print(f"  Coverage: {coverage:.1%} ({non_empty_mask.sum()}/{len(texts)})")

    if not non_empty_texts:
        return np.zeros((len(texts), model.get_sentence_embedding_dimension()))

    t0 = time.time()
    non_empty_embs = model.encode(non_empty_texts, batch_size=256, show_progress_bar=True)
    elapsed = time.time() - t0
    print(f"  Encoded {len(non_empty_texts)} texts in {elapsed:.1f}s")

    dim = non_empty_embs.shape[1]
    all_embs = np.zeros((len(texts), dim), dtype=np.float32)
    all_embs[non_empty_mask] = non_empty_embs

    return all_embs, non_empty_mask


def svd_reduce(embeddings: np.ndarray, mask: np.ndarray, n_components: int,
               train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SVD reduce embeddings, fit on train only. NaN for empty texts."""
    # Fit SVD on non-empty train embeddings
    train_mask = mask[train_idx]
    train_embs = embeddings[train_idx][train_mask]

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(train_embs)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD {n_components} dims, explained variance: {explained:.3f}")

    # Transform all
    all_reduced = svd.transform(embeddings)

    # Build feature arrays with NaN for empty
    def _build(idx):
        reduced = all_reduced[idx]
        m = mask[idx]
        result = np.full_like(reduced, np.nan)
        result[m] = reduced[m]
        return result

    return _build(train_idx), _build(test_idx)


def make_emb_df(reduced: np.ndarray, prefix: str) -> pd.DataFrame:
    """Convert reduced embeddings to DataFrame."""
    cols = {f"{prefix}_{i}": reduced[:, i] for i in range(reduced.shape[1])}
    return pd.DataFrame(cols)


def train_cascade_and_evaluate(X_train, y_train, X_test, y_test, label: str):
    """Train cascade, return F1 macro."""
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_dropped = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_dropped)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES_3, digits=4))
    return f1, cascade


def main():
    print("Loading data...")
    X, y, timestamps, concluding, verdicts, all_texts = load_data()
    print(f"Loaded {len(X)} samples, {X.shape[1]} features")

    # Split indices
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Baseline (already has D+E features)
    print("\n" + "=" * 60)
    print("BASELINE (with D+E text features)")
    print("=" * 60)
    f1_baseline, _ = train_cascade_and_evaluate(X_train, y_train, X_test, y_test, "Baseline (D+E)")

    results = {"baseline_de": f1_baseline}

    # Load sentence transformer
    print("\nLoading sentence-transformers model...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Model dim: {st_model.get_sentence_embedding_dimension()}")

    # ── Encode all text sources ──
    print("\n" + "#" * 60)
    print("# Encoding text sources")
    print("#" * 60)

    concluding_embs, concluding_mask = encode_texts(st_model, concluding, "concluding_notes")
    verdict_embs, verdict_mask = encode_texts(st_model, verdicts, "notes_verdict")
    alltext_embs, alltext_mask = encode_texts(st_model, all_texts, "all_text")

    n_dims = 16  # default

    # ── E1: concluding_notes embeddings ──
    print("\n" + "#" * 60)
    print("# E1: concluding_notes embeddings (SVD 16)")
    print("#" * 60)

    cn_train, cn_test = svd_reduce(concluding_embs, concluding_mask, n_dims, train_idx, test_idx)
    X_train_e1 = pd.concat([X_train, make_emb_df(cn_train, "cn_emb")], axis=1)
    X_test_e1 = pd.concat([X_test, make_emb_df(cn_test, "cn_emb")], axis=1)

    f1_e1, _ = train_cascade_and_evaluate(X_train_e1, y_train, X_test_e1, y_test, "E1: concluding_notes SVD16")
    results["E1_concluding_16"] = f1_e1

    # ── E2: all_text embeddings ──
    print("\n" + "#" * 60)
    print("# E2: all_text embeddings (SVD 16)")
    print("#" * 60)

    at_train, at_test = svd_reduce(alltext_embs, alltext_mask, n_dims, train_idx, test_idx)
    X_train_e2 = pd.concat([X_train, make_emb_df(at_train, "at_emb")], axis=1)
    X_test_e2 = pd.concat([X_test, make_emb_df(at_test, "at_emb")], axis=1)

    f1_e2, _ = train_cascade_and_evaluate(X_train_e2, y_train, X_test_e2, y_test, "E2: all_text SVD16")
    results["E2_alltext_16"] = f1_e2

    # ── E3: notes_verdict embeddings ──
    print("\n" + "#" * 60)
    print("# E3: notes_verdict embeddings (SVD 16)")
    print("#" * 60)

    nv_train, nv_test = svd_reduce(verdict_embs, verdict_mask, n_dims, train_idx, test_idx)
    X_train_e3 = pd.concat([X_train, make_emb_df(nv_train, "nv_emb")], axis=1)
    X_test_e3 = pd.concat([X_test, make_emb_df(nv_test, "nv_emb")], axis=1)

    f1_e3, _ = train_cascade_and_evaluate(X_train_e3, y_train, X_test_e3, y_test, "E3: notes_verdict SVD16")
    results["E3_verdict_16"] = f1_e3

    # ── E4: Dimension sweep on best source ──
    print("\n" + "#" * 60)
    print("# E4: Dimension sweep")
    print("#" * 60)

    # Determine best source
    source_scores = {
        "concluding": (f1_e1, concluding_embs, concluding_mask, "cn_emb"),
        "alltext": (f1_e2, alltext_embs, alltext_mask, "at_emb"),
        "verdict": (f1_e3, verdict_embs, verdict_mask, "nv_emb"),
    }
    best_src = max(source_scores, key=lambda k: source_scores[k][0])
    _, best_embs, best_mask, best_prefix = source_scores[best_src]
    print(f"  Best source: {best_src} (F1={source_scores[best_src][0]:.4f})")

    for dims in [8, 32, 64]:
        print(f"\n  --- SVD {dims} dims ---")
        b_train, b_test = svd_reduce(best_embs, best_mask, dims, train_idx, test_idx)
        X_tr = pd.concat([X_train, make_emb_df(b_train, best_prefix)], axis=1)
        X_te = pd.concat([X_test, make_emb_df(b_test, best_prefix)], axis=1)
        f1_d, _ = train_cascade_and_evaluate(X_tr, y_train, X_te, y_test, f"E4: {best_src} SVD{dims}")
        results[f"E4_{best_src}_{dims}"] = f1_d

    # ── E5: Best embeddings combo ──
    print("\n" + "#" * 60)
    print("# E5: Multiple embedding sources combined")
    print("#" * 60)

    # Combine concluding + verdict (different sources)
    cn16_tr, cn16_te = svd_reduce(concluding_embs, concluding_mask, 16, train_idx, test_idx)
    nv16_tr, nv16_te = svd_reduce(verdict_embs, verdict_mask, 16, train_idx, test_idx)

    X_train_e5 = pd.concat([
        X_train,
        make_emb_df(cn16_tr, "cn_emb"),
        make_emb_df(nv16_tr, "nv_emb"),
    ], axis=1)
    X_test_e5 = pd.concat([
        X_test,
        make_emb_df(cn16_te, "cn_emb"),
        make_emb_df(nv16_te, "nv_emb"),
    ], axis=1)

    f1_e5, cascade_e5 = train_cascade_and_evaluate(
        X_train_e5, y_train, X_test_e5, y_test, "E5: concluding+verdict SVD16+16"
    )
    results["E5_cn_nv_16_16"] = f1_e5

    # ── Feature importance for embedding features ──
    print("\n" + "#" * 60)
    print("# Embedding feature importance (Stage 2)")
    print("#" * 60)

    s2_features = cascade_e5.stage2.feature_name_
    s2_importances = cascade_e5.stage2.feature_importances_

    emb_feats = [f for f in s2_features if "_emb_" in f]
    if emb_feats:
        print("\nStage 2 — embedding features by importance:")
        for fname in sorted(emb_feats, key=lambda f: -s2_importances[s2_features.index(f)]):
            idx = s2_features.index(fname)
            print(f"  {fname:25s} {s2_importances[idx]:10.1f}")

    # ── Summary ──
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        delta = f1 - f1_baseline
        marker = " <<<" if delta > 0.002 else ""
        print(f"  {name:25s}  F1={f1:.4f}  Δ={delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
