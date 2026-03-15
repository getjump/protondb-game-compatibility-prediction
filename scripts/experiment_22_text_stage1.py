"""Phase 22: Text embeddings upgrade + Stage 1 optimization.

Experiments:
  22.1a — All text fields (not just notes_verdict), SVD 32
  22.1b — notes_verdict only, no SVD (384 dims)
  22.1c — All text fields, no SVD (384 dims)
  22.1d — All text fields, SVD 64
  22.2a — Stage 1 borked class_weight sweep
  22.2b — Stage 1 threshold sweep
  22.3  — Contradictory report features

Usage:
  python scripts/experiment_22_text_stage1.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_text_embeddings_custom(conn, report_ids, model_name="all-MiniLM-L6-v2",
                                  n_components=None, use_all_fields=False):
    """Build text embeddings with custom settings."""
    from protondb_settings.config import REPORT_TEXT_FIELDS

    logger.info("Building text embeddings (model=%s, svd=%s, all_fields=%s)...",
                model_name, n_components, use_all_fields)

    if use_all_fields:
        text_fields = [
            "concluding_notes", "notes_verdict", "notes_extra", "notes_customizations",
            "notes_audio_faults", "notes_graphical_faults", "notes_performance_faults",
            "notes_stability_faults", "notes_windowing_faults", "notes_input_faults",
            "notes_significant_bugs", "notes_save_game_faults", "notes_concluding_notes",
        ]
        select = ", ".join(f"r.{f}" for f in text_fields)
    else:
        text_fields = ["notes_verdict"]
        select = "r.notes_verdict"

    # Fetch all reports (faster than IN clause with 278K ids)
    rows = conn.execute(f"SELECT r.id, {select} FROM reports r").fetchall()
    report_id_set = set(report_ids)
    rows = [r for r in rows if r["id"] in report_id_set]

    texts = {}
    for r in rows:
        parts = []
        for f in text_fields:
            val = r[f]
            if val and isinstance(val, str) and len(val.strip()) > 3:
                parts.append(val.strip())
        if parts:
            texts[r["id"]] = " ".join(parts)

    logger.info("Reports with text: %d/%d (%.1f%%)", len(texts), len(report_ids),
                len(texts) / len(report_ids) * 100)

    if not texts:
        dim = n_components or 384
        return np.full((len(report_ids), dim), np.nan)

    # Encode
    model = SentenceTransformer(model_name)
    text_list = [texts.get(rid, "") for rid in report_ids]
    has_text = [bool(texts.get(rid)) for rid in report_ids]

    t0 = time.time()
    embeddings = model.encode(text_list, show_progress_bar=True, batch_size=256)
    logger.info("Encoded %d texts in %.1fs", len(text_list), time.time() - t0)

    # Set empty to NaN
    for i, ht in enumerate(has_text):
        if not ht:
            embeddings[i] = np.nan

    # SVD if requested
    if n_components is not None and n_components < embeddings.shape[1]:
        valid_mask = np.array(has_text)
        valid_emb = embeddings[valid_mask]
        mean = np.nanmean(valid_emb, axis=0)
        centered = valid_emb - mean

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_valid = svd.fit_transform(centered)
        logger.info("SVD %d → %d (explained variance: %.1f%%)",
                    embeddings.shape[1], n_components, svd.explained_variance_ratio_.sum() * 100)

        reduced = np.full((len(report_ids), n_components), np.nan)
        reduced[valid_mask] = reduced_valid
        return reduced

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features,
    )
    from protondb_settings.ml.models.cascade import (
        train_stage1, train_stage2, CascadeClassifier, STAGE2_DROP_FEATURES,
    )
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

    # Current text embedding columns
    text_cols = [c for c in X_train.columns if c.startswith("text_emb_")]
    n_current_text = len(text_cols)
    logger.info("Current text embedding dims: %d", n_current_text)

    results = []

    def ensure_cat(X):
        X = X.copy()
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype("category")
        return X

    def train_eval(X_tr, X_te, y_tr, label, s1_class_weight=None, borked_threshold=0.5):
        X_tr, X_te = ensure_cat(X_tr), ensure_cat(X_te)
        cw = s1_class_weight or {0: 3.0, 1: 1.0}
        s1 = lgb.LGBMClassifier(
            n_estimators=3000, num_leaves=63, learning_rate=0.05,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, class_weight=cw,
            n_jobs=-1, random_state=42, verbose=-1)
        s1.fit(X_tr, (y_tr > 0).astype(int), eval_set=[(X_te, (y_test > 0).astype(int))],
               callbacks=[lgb.early_stopping(50, verbose=False)],
               categorical_feature=[c for c in CATEGORICAL_FEATURES if c in X_tr.columns])

        s2, drops = train_stage2(X_tr, y_tr, X_te, y_test)
        cascade = CascadeClassifier(s1, s2, drops, borked_threshold=borked_threshold)
        y_pred = cascade.predict(X_te)
        f1 = f1_score(y_test, y_pred, average="macro")
        per = f1_score(y_test, y_pred, average=None)

        # Borked recall/precision
        y_bin_test = (y_test > 0).astype(int)
        y_bin_pred = s1.predict(X_te)
        borked_r = (y_bin_pred[y_bin_test == 0] == 0).mean()
        borked_p = (y_bin_test[y_bin_pred == 0] == 0).mean() if (y_bin_pred == 0).any() else 0

        r = {"label": label, "f1_macro": f1, "borked_f1": per[0],
             "tinkering_f1": per[1], "works_oob_f1": per[2],
             "borked_recall": borked_r, "borked_precision": borked_p}
        results.append(r)
        print(f"  {label:45s} F1={f1:.4f} b={per[0]:.3f} t={per[1]:.3f} o={per[2]:.3f} "
              f"b_r={borked_r:.3f} b_p={borked_p:.3f}")
        return f1

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE (current: MiniLM, notes_verdict only, SVD 32)")
    print("=" * 70)
    train_eval(X_train, X_test, y_train, "baseline")

    # ── 22.1a: All text fields, SVD 32 ───────────────────────────────
    print("\n" + "=" * 70)
    print("22.1a: All text fields, SVD 32")
    print("=" * 70)
    emb_all_32_tr = build_text_embeddings_custom(conn, train_rids, n_components=32, use_all_fields=True)
    emb_all_32_te = build_text_embeddings_custom(conn, test_rids, n_components=32, use_all_fields=True)
    X_tr_a = X_train.copy()
    X_te_a = X_test.copy()
    for d in range(32):
        col = f"text_emb_{d}"
        X_tr_a[col] = emb_all_32_tr[:, d]
        X_te_a[col] = emb_all_32_te[:, d]
    train_eval(X_tr_a, X_te_a, y_train, "22.1a_all_fields_svd32")

    # ── 22.1b: notes_verdict only, no SVD (384 dims) ────────────────
    print("\n" + "=" * 70)
    print("22.1b: notes_verdict only, no SVD (384 dims)")
    print("=" * 70)
    emb_nv_full_tr = build_text_embeddings_custom(conn, train_rids, n_components=None, use_all_fields=False)
    emb_nv_full_te = build_text_embeddings_custom(conn, test_rids, n_components=None, use_all_fields=False)
    X_tr_b = X_train.drop(columns=text_cols).copy()
    X_te_b = X_test.drop(columns=text_cols).copy()
    for d in range(emb_nv_full_tr.shape[1]):
        X_tr_b[f"text_emb_{d}"] = emb_nv_full_tr[:, d]
        X_te_b[f"text_emb_{d}"] = emb_nv_full_te[:, d]
    train_eval(X_tr_b, X_te_b, y_train, "22.1b_verdict_only_384d")

    # ── 22.1c: All text fields, no SVD (384 dims) ───────────────────
    print("\n" + "=" * 70)
    print("22.1c: All text fields, no SVD (384 dims)")
    print("=" * 70)
    emb_all_full_tr = build_text_embeddings_custom(conn, train_rids, n_components=None, use_all_fields=True)
    emb_all_full_te = build_text_embeddings_custom(conn, test_rids, n_components=None, use_all_fields=True)
    X_tr_c = X_train.drop(columns=text_cols).copy()
    X_te_c = X_test.drop(columns=text_cols).copy()
    for d in range(emb_all_full_tr.shape[1]):
        X_tr_c[f"text_emb_{d}"] = emb_all_full_tr[:, d]
        X_te_c[f"text_emb_{d}"] = emb_all_full_te[:, d]
    train_eval(X_tr_c, X_te_c, y_train, "22.1c_all_fields_384d")

    # ── 22.1d: All text fields, SVD 64 ──────────────────────────────
    print("\n" + "=" * 70)
    print("22.1d: All text fields, SVD 64")
    print("=" * 70)
    emb_all_64_tr = build_text_embeddings_custom(conn, train_rids, n_components=64, use_all_fields=True)
    emb_all_64_te = build_text_embeddings_custom(conn, test_rids, n_components=64, use_all_fields=True)
    X_tr_d = X_train.drop(columns=text_cols).copy()
    X_te_d = X_test.drop(columns=text_cols).copy()
    for d in range(64):
        X_tr_d[f"text_emb_{d}"] = emb_all_64_tr[:, d]
        X_te_d[f"text_emb_{d}"] = emb_all_64_te[:, d]
    train_eval(X_tr_d, X_te_d, y_train, "22.1d_all_fields_svd64")

    # ── 22.2a: Stage 1 class_weight sweep ────────────────────────────
    print("\n" + "=" * 70)
    print("22.2: Stage 1 class_weight sweep")
    print("=" * 70)
    for w in [4.0, 5.0, 6.0]:
        train_eval(X_train, X_test, y_train, f"22.2a_cw_{w}", s1_class_weight={0: w, 1: 1.0})

    # ── 22.2b: Stage 1 threshold sweep ───────────────────────────────
    print("\n" + "=" * 70)
    print("22.2b: Stage 1 borked threshold sweep")
    print("=" * 70)
    for t in [0.45, 0.40, 0.35]:
        train_eval(X_train, X_test, y_train, f"22.2b_thresh_{t}", borked_threshold=t)

    # ── 22.3: Contradictory features ─────────────────────────────────
    print("\n" + "=" * 70)
    print("22.3: Contradictory report features")
    print("=" * 70)
    X_tr_contra = X_train.copy()
    X_te_contra = X_test.copy()
    for X_df in [X_tr_contra, X_te_contra]:
        X_df["contradictory"] = ((X_df["mentions_perfect"] > 0) &
                                  (X_df["fault_notes_count"] > 0)).astype(int)
        X_df["fix_in_borked"] = ((X_df["mentions_fix"] > 0) &
                                  (X_df["mentions_crash"] > 0)).astype(int)
    train_eval(X_tr_contra, X_te_contra, y_train, "22.3_contradictory_features")

    conn.close()

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<45s} {'F1':>7s} {'ΔF1':>7s} {'b':>6s} {'t':>6s} {'o':>6s} {'b_r':>6s}")
    print("-" * 84)
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<45s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>6.3f} {r['tinkering_f1']:>6.3f} "
              f"{r['works_oob_f1']:>6.3f} {r['borked_recall']:>6.3f}")


if __name__ == "__main__":
    main()
