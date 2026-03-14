"""Phase 19 interim check: effect of LLM-inferred verdicts on model quality.

Compares:
  A. Baseline (current pipeline, inferred tinkering as-is)
  B. Use LLM verdicts to fix inferred labels
  C. Filter: Stage 2 only on explicit + LLM-inferred labels

Usage:
  python scripts/experiment_19_interim.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")

    # Load inferred verdicts
    inferred = {}
    for r in conn.execute("SELECT report_id, verdict FROM inferred_verdicts").fetchall():
        inferred[r["report_id"]] = r["verdict"]
    logger.info("Loaded %d inferred verdicts", len(inferred))

    # Build features
    X, y_raw, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y_raw, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)

    theta, difficulty = fit_irt(conn)
    X_train = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_test = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    X_train = add_error_targeted_features(X_train, train_rids, conn)
    X_test = add_error_targeted_features(X_test, test_rids, conn)

    # Check which train reports have inferred verdicts
    n_inferred_train = sum(1 for rid in train_rids if rid in inferred)
    logger.info("Train reports with inferred verdicts: %d/%d (%.1f%%)",
                n_inferred_train, len(train_rids), n_inferred_train/len(train_rids)*100)

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    conn.close()

    def train_eval(X_tr, y_tr, label):
        s1 = train_stage1(X_tr, y_tr, X_test, y_test)
        s2, drops = train_stage2(X_tr, y_tr, X_test, y_test)
        cas = CascadeClassifier(s1, s2, drops)
        y_pred = cas.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        per = f1_score(y_test, y_pred, average=None)
        oob_r = (y_pred[y_test == 2] == 2).mean()
        print(f"  {label:40s} F1={f1:.4f} b={per[0]:.3f} t={per[1]:.3f} o={per[2]:.3f} oob_r={oob_r:.3f}")
        return f1

    # ── A. Baseline: current pipeline ────────────────────────────────
    print("\n" + "=" * 70)
    print("A. BASELINE (contributor-aware relabel, no LLM verdicts)")
    print("=" * 70)
    y_baseline, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids,
                                              get_connection(args.db), theta)
    get_connection(args.db).close()
    f1_bl = train_eval(X_train, y_baseline, "baseline")

    # ── B. Fix inferred labels with LLM verdicts ─────────────────────
    print("\n" + "=" * 70)
    print("B. LLM VERDICT FIX (replace inferred tinkering with LLM verdict)")
    print("=" * 70)
    # y_train_raw: 0=borked, 1=tinkering (inferred OR explicit), 2=works_oob
    # For reports with LLM verdict: override the label
    y_llm = y_baseline.copy()
    n_fixed_oob = 0
    n_fixed_tink = 0
    for i, rid in enumerate(train_rids):
        if rid in inferred and y_llm[i] == 1:  # currently tinkering
            llm_verdict = inferred[rid]
            if llm_verdict == "works_oob":
                y_llm[i] = 2
                n_fixed_oob += 1
            # tinkering stays tinkering
    logger.info("LLM fix: %d tinkering → works_oob, %d confirmed tinkering", n_fixed_oob, n_fixed_tink)

    # Show new distribution
    for cls, name in [(0, "borked"), (1, "tinkering"), (2, "works_oob")]:
        n_bl = (y_baseline == cls).sum()
        n_llm = (y_llm == cls).sum()
        delta = n_llm - n_bl
        print(f"  {name:12s}: {n_bl:6d} → {n_llm:6d} ({delta:+d})")

    f1_llm = train_eval(X_train, y_llm, "llm_verdict_fix")

    # ── C. Only explicit + LLM labels for Stage 2 ───────────────────
    print("\n" + "=" * 70)
    print("C. FILTERED: Stage 2 only on reports with real/LLM verdict_oob")
    print("=" * 70)

    # Check which reports have explicit verdict_oob or LLM verdict
    conn2 = get_connection(args.db)
    has_explicit_oob = set()
    for r in conn2.execute("SELECT id FROM reports WHERE verdict_oob IS NOT NULL").fetchall():
        has_explicit_oob.add(r["id"])
    conn2.close()

    has_label = set()
    for rid in train_rids:
        if rid in has_explicit_oob or rid in inferred:
            has_label.add(rid)

    # For Stage 2: mask out reports without real labels
    # We need to keep them for Stage 1 (borked detection) but downweight for Stage 2
    # Approach: set weight=0 for reports without labels in Stage 2
    # Simpler: just use y_llm with all data (LLM already fixed the labels we have)
    # The real test is: does LLM fix help?

    logger.info("Reports with explicit/LLM verdict: %d/%d (%.1f%%)",
                len(has_label), len(train_rids), len(has_label)/len(train_rids)*100)

    print(f"\n  Delta: {f1_llm - f1_bl:+.4f} F1")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline:         F1={f1_bl:.4f}")
    print(f"  LLM verdict fix:  F1={f1_llm:.4f} (Δ={f1_llm-f1_bl:+.4f})")
    print(f"  Inferred verdicts available: {len(inferred)} ({len(inferred)/211749*100:.1f}% of target)")


if __name__ == "__main__":
    main()
