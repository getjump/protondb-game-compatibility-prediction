"""Deep analysis of current best pipeline (IRT + contributor-aware relabel).

Outputs:
  - Feature importance (SHAP top-30)
  - Per-class error analysis
  - Confusion matrix
  - Confidence distribution
  - IRT parameter analysis
  - Coverage impact analysis
  - Leakage detection
  - Recommendations

Usage:
  python scripts/analyze_current_pipeline.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, log_loss,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Pipeline Analysis: IRT + Contributor-Aware Relabel")
    print("=" * 70)

    # ── Load data and train ──────────────────────────────────────────
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import fit_irt, add_irt_features, contributor_aware_relabel
    from protondb_settings.ml.models.cascade import (
        train_stage1, train_stage2, CascadeClassifier, STAGE2_DROP_FEATURES,
    )
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES, TARGET_NAMES

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y, ts, 0.2, report_ids=rids)

    relabel_ids = get_relabel_ids(conn)
    theta, difficulty = fit_irt(conn)
    X_train = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_test = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    y_train, n_relabel = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)

    # Build report metadata for analysis
    report_meta = {}
    for r in conn.execute("SELECT id, app_id, gpu, timestamp, proton_version FROM reports").fetchall():
        report_meta[r["id"]] = dict(r)
    contrib_lookup = {}
    for r in conn.execute("SELECT report_id, contributor_id, report_tally FROM report_contributors").fetchall():
        contrib_lookup[r["report_id"]] = {"contributor_id": str(r["contributor_id"]), "tally": r["report_tally"]}

    conn.close()

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Relabeled: {n_relabel}")
    print(f"IRT: {len(theta)} contributors, {len(difficulty)} items")

    # Train
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_drops)

    y_pred = cascade.predict(X_test)
    y_proba = cascade.predict_proba(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    per_class = f1_score(y_test, y_pred, average=None)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*70}")
    print(f"METRICS: F1={f1:.4f}, Acc={acc:.4f}")
    print(f"  borked:    F1={per_class[0]:.4f}")
    print(f"  tinkering: F1={per_class[1]:.4f}")
    print(f"  works_oob: F1={per_class[2]:.4f}")

    # ── 1. Confusion Matrix ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("1. CONFUSION MATRIX")
    print(f"{'='*70}")
    cm = confusion_matrix(y_test, y_pred)
    labels = ["borked", "tinkering", "works_oob"]
    print(f"{'':>12s} {'pred_bork':>10s} {'pred_tink':>10s} {'pred_oob':>10s} {'total':>8s} {'recall':>8s}")
    for i, label in enumerate(labels):
        row = cm[i]
        recall = row[i] / row.sum() if row.sum() > 0 else 0
        print(f"{label:>12s} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d} {row.sum():>8d} {recall:>8.3f}")
    print(f"{'precision':>12s}", end="")
    for j in range(3):
        prec = cm[j, j] / cm[:, j].sum() if cm[:, j].sum() > 0 else 0
        print(f" {prec:>10.3f}", end="")
    print()

    # ── 2. Error Analysis ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("2. ERROR ANALYSIS")
    print(f"{'='*70}")

    errors = y_pred != y_test
    n_errors = errors.sum()
    print(f"Total errors: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

    # Error types
    error_types = Counter()
    for i in range(len(y_test)):
        if errors[i]:
            true_name = TARGET_NAMES[y_test[i]]
            pred_name = TARGET_NAMES[y_pred[i]]
            error_types[f"{true_name} → {pred_name}"] += 1

    print("\nError breakdown:")
    for err_type, count in error_types.most_common():
        pct = count / n_errors * 100
        print(f"  {err_type:30s} {count:5d} ({pct:.1f}%)")

    # Stage 2 boundary errors
    s2_errors = sum(1 for i in range(len(y_test))
                    if errors[i] and y_test[i] in (1, 2) and y_pred[i] in (1, 2))
    print(f"\nStage 2 boundary errors (tinkering↔oob): {s2_errors}/{n_errors} ({s2_errors/n_errors*100:.1f}%)")

    # ── 3. Confidence Analysis ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("3. CONFIDENCE ANALYSIS")
    print(f"{'='*70}")

    confidence = y_proba.max(axis=1)
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = confidence >= thresh
        if mask.sum() > 0:
            acc_conf = accuracy_score(y_test[mask], y_pred[mask])
            f1_conf = f1_score(y_test[mask], y_pred[mask], average="macro")
            print(f"  conf >= {thresh:.1f}: {mask.sum():5d} ({mask.sum()/len(y_test)*100:.0f}%) "
                  f"acc={acc_conf:.4f} f1={f1_conf:.4f}")

    # Confidence by correctness
    correct_conf = confidence[~errors].mean()
    error_conf = confidence[errors].mean()
    print(f"\n  Mean confidence — correct: {correct_conf:.3f}, errors: {error_conf:.3f}")

    # ── 4. Feature Importance (Stage 1 + Stage 2) ────────────────────
    print(f"\n{'='*70}")
    print("4. FEATURE IMPORTANCE")
    print(f"{'='*70}")

    # Stage 1 importance
    s1_imp = pd.Series(s1.feature_importances_, index=X_train.columns)
    s1_imp = s1_imp.sort_values(ascending=False)
    print("\nStage 1 (borked vs works) — top 15:")
    for feat, imp in s1_imp.head(15).items():
        print(f"  {feat:35s} {imp:8.0f}")

    # Stage 2 importance
    s2_features = [c for c in X_train.columns if c not in s2_drops]
    s2_imp_raw = s2.feature_importance(importance_type="gain")
    s2_names = s2.feature_name()
    s2_imp = pd.Series(s2_imp_raw, index=s2_names).sort_values(ascending=False)
    print("\nStage 2 (tinkering vs oob) — top 15:")
    for feat, imp in s2_imp.head(15).items():
        print(f"  {feat:35s} {imp:8.1f}")

    # ── 5. IRT Analysis ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("5. IRT PARAMETER ANALYSIS")
    print(f"{'='*70}")

    theta_vals = list(theta.values())
    diff_vals = list(difficulty.values())
    print(f"\nTheta (contributor strictness):")
    print(f"  count={len(theta_vals)}, mean={np.mean(theta_vals):.2f}, std={np.std(theta_vals):.2f}")
    print(f"  range=[{np.min(theta_vals):.2f}, {np.max(theta_vals):.2f}]")
    for pct in [10, 25, 50, 75, 90]:
        print(f"  p{pct}={np.percentile(theta_vals, pct):.2f}", end="")
    print()

    print(f"\nDifficulty (game×gpu):")
    print(f"  count={len(diff_vals)}, mean={np.mean(diff_vals):.2f}, std={np.std(diff_vals):.2f}")
    print(f"  range=[{np.min(diff_vals):.2f}, {np.max(diff_vals):.2f}]")

    # IRT feature importance in Stage 2
    irt_features = [f for f in s2_names if f.startswith("irt_")]
    if irt_features:
        print(f"\nIRT features in Stage 2:")
        for f in irt_features:
            print(f"  {f:35s} gain={s2_imp.get(f, 0):.1f} (rank {list(s2_imp.index).index(f)+1}/{len(s2_imp)})")

    # ── 6. Coverage Impact ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("6. COVERAGE IMPACT (contributor data vs none)")
    print(f"{'='*70}")

    has_contrib = np.array([rid in contrib_lookup for rid in test_rids])
    if has_contrib.any() and (~has_contrib).any():
        f1_with = f1_score(y_test[has_contrib], y_pred[has_contrib], average="macro")
        f1_without = f1_score(y_test[~has_contrib], y_pred[~has_contrib], average="macro")
        acc_with = accuracy_score(y_test[has_contrib], y_pred[has_contrib])
        acc_without = accuracy_score(y_test[~has_contrib], y_pred[~has_contrib])
        print(f"  With contributor data:    n={has_contrib.sum():5d} ({has_contrib.mean()*100:.1f}%) F1={f1_with:.4f} acc={acc_with:.4f}")
        print(f"  Without contributor data: n={(~has_contrib).sum():5d} ({(~has_contrib).mean()*100:.1f}%) F1={f1_without:.4f} acc={acc_without:.4f}")
        print(f"  Gap: {f1_with - f1_without:+.4f} F1")

    # ── 7. Per-class deep dive ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("7. PER-CLASS DEEP DIVE")
    print(f"{'='*70}")

    for cls in range(3):
        cls_name = TARGET_NAMES[cls]
        mask_true = y_test == cls
        mask_pred = y_pred == cls

        # False negatives: true=cls but predicted as other
        fn_mask = mask_true & ~mask_pred
        # False positives: predicted=cls but true is other
        fp_mask = mask_pred & ~mask_true

        print(f"\n  {cls_name.upper()}:")
        print(f"    True: {mask_true.sum()}, Predicted: {mask_pred.sum()}")
        print(f"    FN: {fn_mask.sum()} (missed), FP: {fp_mask.sum()} (false alarm)")

        # Confidence on FN (how confident model was in wrong prediction)
        if fn_mask.any():
            fn_conf = confidence[fn_mask].mean()
            fn_pred_dist = Counter(y_pred[fn_mask])
            print(f"    FN mean confidence: {fn_conf:.3f}")
            print(f"    FN predicted as: {dict(fn_pred_dist)}")

        # IRT difficulty distribution for errors
        if "irt_game_difficulty" in X_test.columns:
            irt_d = X_test["irt_game_difficulty"].values
            if fn_mask.any():
                print(f"    FN irt_difficulty: mean={irt_d[fn_mask].mean():.2f} vs all={irt_d[mask_true].mean():.2f}")

    # ── 8. Train vs Test Distribution ────────────────────────────────
    print(f"\n{'='*70}")
    print("8. TRAIN vs TEST DISTRIBUTION")
    print(f"{'='*70}")

    for cls in range(3):
        n_train = (y_train == cls).sum()
        n_test = (y_test == cls).sum()
        print(f"  {TARGET_NAMES[cls]:12s}: train={n_train:6d} ({n_train/len(y_train)*100:.1f}%) "
              f"test={n_test:6d} ({n_test/len(y_test)*100:.1f}%)")

    # ── 9. Leakage Check ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("9. LEAKAGE CHECK")
    print(f"{'='*70}")

    # Check if game aggregates leak (same game in train and test)
    train_apps = set()
    test_apps = set()
    for rid in train_rids:
        if rid in report_meta:
            train_apps.add(report_meta[rid]["app_id"])
    for rid in test_rids:
        if rid in report_meta:
            test_apps.add(report_meta[rid]["app_id"])

    overlap = train_apps & test_apps
    test_only = test_apps - train_apps
    print(f"  Train games: {len(train_apps)}, Test games: {len(test_apps)}")
    print(f"  Overlap: {len(overlap)} ({len(overlap)/len(test_apps)*100:.1f}% of test)")
    print(f"  Test-only (cold start): {len(test_only)} ({len(test_only)/len(test_apps)*100:.1f}%)")

    # F1 on overlapping vs test-only games
    overlap_mask = np.array([report_meta.get(rid, {}).get("app_id") in overlap for rid in test_rids])
    cold_mask = ~overlap_mask
    if overlap_mask.any() and cold_mask.any():
        f1_overlap = f1_score(y_test[overlap_mask], y_pred[overlap_mask], average="macro")
        f1_cold = f1_score(y_test[cold_mask], y_pred[cold_mask], average="macro")
        print(f"  F1 overlap games:   {f1_overlap:.4f} (n={overlap_mask.sum()})")
        print(f"  F1 cold-start games: {f1_cold:.4f} (n={cold_mask.sum()})")
        print(f"  Gap: {f1_overlap - f1_cold:+.4f}")

    # ── 10. Hardest Games ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("10. HARDEST GAMES (most errors)")
    print(f"{'='*70}")

    game_errors = Counter()
    game_totals = Counter()
    for i, rid in enumerate(test_rids):
        app_id = report_meta.get(rid, {}).get("app_id")
        if app_id:
            game_totals[app_id] += 1
            if errors[i]:
                game_errors[app_id] += 1

    # Games with most errors (min 5 reports)
    error_rates = {}
    for app_id, n_err in game_errors.items():
        n_total = game_totals[app_id]
        if n_total >= 5:
            error_rates[app_id] = (n_err, n_total, n_err / n_total)

    print(f"\n  Games with highest error rate (min 5 test reports):")
    sorted_games = sorted(error_rates.items(), key=lambda x: -x[1][2])[:15]

    # Load game names from PICS cache
    game_names = {}
    try:
        from protondb_settings.db.connection import get_connection as gc
        conn2 = gc(args.db)
        for r in conn2.execute(
            "SELECT app_id, data_json FROM enrichment_cache WHERE source='steam_pics'"
        ).fetchall():
            try:
                d = json.loads(r["data_json"])
                if d.get("name"):
                    game_names[r["app_id"]] = d["name"]
            except Exception:
                pass
        conn2.close()
    except Exception:
        pass

    for app_id, (n_err, n_total, rate) in sorted_games:
        name = game_names.get(app_id, f"app_{app_id}")[:30]
        print(f"  {name:30s} {n_err:3d}/{n_total:3d} ({rate*100:.0f}%) errors")

    # ── 11. Recommendations ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("11. RECOMMENDATIONS")
    print(f"{'='*70}")

    s2_boundary_pct = s2_errors / n_errors * 100 if n_errors > 0 else 0
    cold_gap = (f1_overlap - f1_cold) if overlap_mask.any() and cold_mask.any() else 0
    contrib_gap = (f1_with - f1_without) if has_contrib.any() and (~has_contrib).any() else 0

    print(f"""
  Stage 2 boundary errors: {s2_boundary_pct:.0f}% of all errors
  Cold-start gap: {cold_gap:+.4f} F1
  Contributor coverage gap: {contrib_gap:+.4f} F1

  Based on analysis:""")

    if s2_boundary_pct > 60:
        print("  → Stage 2 boundary still dominates errors. Focus on tinkering↔oob denoising.")
    if cold_gap > 0.05:
        print(f"  → Cold-start gap is significant ({cold_gap:+.4f}). Game metadata features help seen games but not new ones.")
        print("    Consider: game-type features (engine, genre, DX) instead of per-game aggregates for cold start.")
    if contrib_gap > 0.02:
        print(f"  → Contributor coverage matters ({contrib_gap:+.4f}). More contributor data = better predictions.")
    if error_conf > 0.55:
        print(f"  → Errors have high confidence ({error_conf:.3f}). Model is overconfident on mistakes.")
        print("    Consider: better calibration or conformal prediction.")

    print(f"\n  Current best: F1={f1:.4f} (borked={per_class[0]:.3f}, tink={per_class[1]:.3f}, oob={per_class[2]:.3f})")


if __name__ == "__main__":
    main()
