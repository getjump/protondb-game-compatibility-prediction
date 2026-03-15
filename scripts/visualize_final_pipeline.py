"""Final pipeline visualization and analysis.

Generates comprehensive analysis of the production pipeline:
  - Per-report vs per-game metrics comparison
  - Confusion matrices (per-report and per-game)
  - Confidence distribution and calibration
  - Per-class deep dive
  - Feature importance (Stage 1 + Stage 2)
  - IRT parameter distributions
  - Error analysis by game popularity
  - Cold-start analysis
  - Per-vendor / per-deck breakdown
  - Proton version impact
  - Summary statistics for README/paper

Usage:
  python scripts/visualize_final_pipeline.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import (
        fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features,
    )
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES, TARGET_NAMES
    from protondb_settings.ml.aggregate import aggregate_predictions

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

    # Report metadata
    report_meta = {}
    for r in conn.execute("""
        SELECT id, app_id, gpu, variant, timestamp, proton_version,
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
            "timestamp": int(r["timestamp"]) if r["timestamp"] else 0,
            "proton_version": r["proton_version"],
        }
    conn.close()

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Train
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_drops)

    y_pred = cascade.predict(X_test)
    y_proba = cascade.predict_proba(X_test)
    confidence = y_proba.max(axis=1)
    labels = ["borked", "tinkering", "works_oob"]

    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  FINAL PIPELINE ANALYSIS")
    print("  Per-report model + per-game aggregation")
    print("=" * 80)

    # ── 1. Per-report metrics ────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("1. PER-REPORT METRICS")
    print(f"{'─'*80}")
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    per = f1_score(y_test, y_pred, average=None)
    print(f"  F1 macro: {f1:.4f}    Accuracy: {acc:.4f}")
    print(f"  borked:    F1={per[0]:.4f}")
    print(f"  tinkering: F1={per[1]:.4f}")
    print(f"  works_oob: F1={per[2]:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix:")
    print(f"  {'':>12s} {'pred_b':>8s} {'pred_t':>8s} {'pred_o':>8s}  {'recall':>7s}")
    for i, label in enumerate(labels):
        row = cm[i]
        recall = row[i] / row.sum()
        print(f"  {label:>12s} {row[0]:>8d} {row[1]:>8d} {row[2]:>8d}  {recall:>7.3f}")

    # ── 2. Per-game aggregated metrics ───────────────────────────────
    print(f"\n{'─'*80}")
    print("2. PER-GAME AGGREGATED METRICS (majority vote)")
    print(f"{'─'*80}")

    game_groups = defaultdict(lambda: {"preds": [], "truths": [], "probas": []})
    for i, rid in enumerate(test_rids):
        app_id = report_meta.get(rid, {}).get("app_id")
        if app_id:
            game_groups[app_id]["preds"].append(y_pred[i])
            game_groups[app_id]["truths"].append(y_test[i])
            game_groups[app_id]["probas"].append(y_proba[i])

    agg_true, agg_pred, agg_sizes = [], [], []
    for app_id, data in game_groups.items():
        agg_true.append(Counter(data["truths"]).most_common(1)[0][0])
        agg_pred.append(Counter(data["preds"]).most_common(1)[0][0])
        agg_sizes.append(len(data["preds"]))

    agg_true = np.array(agg_true)
    agg_pred = np.array(agg_pred)
    agg_sizes = np.array(agg_sizes)

    f1_game = f1_score(agg_true, agg_pred, average="macro")
    acc_game = accuracy_score(agg_true, agg_pred)
    per_game = f1_score(agg_true, agg_pred, average=None)

    print(f"  Games: {len(agg_true)}")
    print(f"  F1 macro: {f1_game:.4f}    Accuracy: {acc_game:.4f}")
    print(f"  borked:    F1={per_game[0]:.4f}")
    print(f"  tinkering: F1={per_game[1]:.4f}")
    print(f"  works_oob: F1={per_game[2]:.4f}")

    # Binary
    agg_true_bin = (agg_true > 0).astype(int)
    agg_pred_bin = (agg_pred > 0).astype(int)
    f1_bin = f1_score(agg_true_bin, agg_pred_bin, average="macro")
    print(f"\n  Binary (borked vs works): F1={f1_bin:.4f}")

    # By report count
    print(f"\n  By reports per game:")
    for min_n, max_n, label in [(1,2,"1-2"), (3,5,"3-5"), (6,10,"6-10"),
                                 (11,20,"11-20"), (21,50,"21-50"), (51,9999,"51+")]:
        mask = (agg_sizes >= min_n) & (agg_sizes <= max_n)
        if mask.sum() > 10:
            f = f1_score(agg_true[mask], agg_pred[mask], average="macro")
            a = accuracy_score(agg_true[mask], agg_pred[mask])
            print(f"    {label:>6s} reports: {mask.sum():5d} games, F1={f:.4f}, acc={a:.4f}")

    # ── 3. Confidence analysis ───────────────────────────────────────
    print(f"\n{'─'*80}")
    print("3. CONFIDENCE DISTRIBUTION")
    print(f"{'─'*80}")

    errors = y_pred != y_test
    print(f"  Mean confidence — correct: {confidence[~errors].mean():.3f}, errors: {confidence[errors].mean():.3f}")
    print(f"\n  Per-report:")
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = confidence >= thresh
        if mask.sum() > 0:
            f = f1_score(y_test[mask], y_pred[mask], average="macro")
            a = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    conf >= {thresh}: {mask.sum():5d} ({mask.mean()*100:.0f}%) F1={f:.4f} acc={a:.4f}")

    # Per-game confidence
    print(f"\n  Per-game:")
    for min_n in [1, 3, 5, 10]:
        mask = agg_sizes >= min_n
        if mask.sum() > 10:
            f = f1_score(agg_true[mask], agg_pred[mask], average="macro")
            agreement = np.array([Counter(game_groups[app]["preds"]).most_common(1)[0][1] / len(game_groups[app]["preds"])
                                  for app, s in zip(game_groups.keys(), agg_sizes) if s >= min_n])
            print(f"    {min_n}+ reports: {mask.sum():5d} games, F1={f:.4f}, mean agreement={agreement.mean():.3f}")

    # ── 4. Feature importance ────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("4. FEATURE IMPORTANCE")
    print(f"{'─'*80}")

    s1_imp = pd.Series(s1.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(f"\n  Stage 1 (borked vs works) — top 10:")
    for feat, imp in s1_imp.head(10).items():
        print(f"    {feat:35s} {imp:8.0f}")

    s2_imp = pd.Series(s2.feature_importance(importance_type="gain"),
                       index=s2.feature_name()).sort_values(ascending=False)
    print(f"\n  Stage 2 (tinkering vs oob) — top 10:")
    for feat, imp in s2_imp.head(10).items():
        print(f"    {feat:35s} {imp:10.1f}")

    # ── 5. IRT analysis ──────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("5. IRT PARAMETERS")
    print(f"{'─'*80}")

    theta_vals = np.array(list(theta.values()))
    diff_vals = np.array(list(difficulty.values()))
    print(f"  Contributors: {len(theta)}")
    print(f"    θ (strictness): mean={theta_vals.mean():.2f}, std={theta_vals.std():.2f}, "
          f"range=[{theta_vals.min():.2f}, {theta_vals.max():.2f}]")
    print(f"  Items (game×gpu): {len(difficulty)}")
    print(f"    d (difficulty): mean={diff_vals.mean():.2f}, std={diff_vals.std():.2f}, "
          f"range=[{diff_vals.min():.2f}, {diff_vals.max():.2f}]")

    # ── 6. Per-vendor breakdown ──────────────────────────────────────
    print(f"\n{'─'*80}")
    print("6. PER-VENDOR BREAKDOWN")
    print(f"{'─'*80}")

    for vendor in ["nvidia", "amd", "intel"]:
        mask = np.array([report_meta.get(rid, {}).get("vendor") == vendor for rid in test_rids])
        if mask.sum() > 100:
            f = f1_score(y_test[mask], y_pred[mask], average="macro")
            per_v = f1_score(y_test[mask], y_pred[mask], average=None)
            print(f"  {vendor:>8s}: n={mask.sum():5d} F1={f:.4f} b={per_v[0]:.3f} t={per_v[1]:.3f} o={per_v[2]:.3f}")

    # ── 7. Deck vs Desktop ───────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("7. STEAM DECK vs DESKTOP")
    print(f"{'─'*80}")

    for device, is_deck_val in [("Deck", True), ("Desktop", False)]:
        mask = np.array([report_meta.get(rid, {}).get("is_deck") == is_deck_val for rid in test_rids])
        if mask.sum() > 100:
            f = f1_score(y_test[mask], y_pred[mask], average="macro")
            per_d = f1_score(y_test[mask], y_pred[mask], average=None)
            print(f"  {device:>8s}: n={mask.sum():5d} F1={f:.4f} b={per_d[0]:.3f} t={per_d[1]:.3f} o={per_d[2]:.3f}")

    # ── 8. Per-variant breakdown ─────────────────────────────────────
    print(f"\n{'─'*80}")
    print("8. PER-VARIANT (Proton type)")
    print(f"{'─'*80}")

    for variant in ["official", "ge", "experimental", "native", "notListed", "older"]:
        mask = np.array([report_meta.get(rid, {}).get("variant") == variant for rid in test_rids])
        if mask.sum() > 100:
            f = f1_score(y_test[mask], y_pred[mask], average="macro")
            print(f"  {variant:>14s}: n={mask.sum():5d} F1={f:.4f}")

    # ── 9. Error analysis ────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("9. ERROR ANALYSIS")
    print(f"{'─'*80}")

    n_errors = errors.sum()
    print(f"  Total errors: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

    error_types = Counter()
    for i in range(len(y_test)):
        if errors[i]:
            error_types[f"{TARGET_NAMES[y_test[i]]} → {TARGET_NAMES[y_pred[i]]}"] += 1

    print(f"\n  Error breakdown:")
    for err, cnt in error_types.most_common():
        print(f"    {err:30s} {cnt:5d} ({cnt/n_errors*100:.1f}%)")

    s2_errors = sum(1 for i in range(len(y_test))
                    if errors[i] and y_test[i] in (1,2) and y_pred[i] in (1,2))
    print(f"\n  Stage 2 boundary (tinkering↔oob): {s2_errors}/{n_errors} ({s2_errors/n_errors*100:.1f}%)")

    # ── 10. Data summary ─────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("10. DATA SUMMARY")
    print(f"{'─'*80}")

    print(f"  Total reports: {len(X_train) + len(X_test)}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Games in test: {len(game_groups)}")

    for cls in range(3):
        n_tr = (y_train == cls).sum()
        n_te = (y_test == cls).sum()
        print(f"  {TARGET_NAMES[cls]:12s}: train {n_tr:6d} ({n_tr/len(y_train)*100:.1f}%), "
              f"test {n_te:6d} ({n_te/len(y_test)*100:.1f}%)")

    # ── 11. Cumulative progress ──────────────────────────────────────
    print(f"\n{'─'*80}")
    print("11. CUMULATIVE PROGRESS (Phases 11-21)")
    print(f"{'─'*80}")

    milestones = [
        ("Phase 11 baseline", 0.7245, 0.503),
        ("+ IRT features (Phase 12)", 0.7545, 0.569),
        ("+ Contributor relabel (Phase 13)", 0.7711, 0.591),
        ("+ Class weight + error features (Phase 16)", 0.7776, 0.603),
        ("+ HP tuning (Phase 17)", 0.7801, 0.614),
        ("Per-game vote (Phase 21)", f1_game, per_game[2]),
    ]

    print(f"\n  {'Stage':<45s} {'F1 macro':>9s} {'oob F1':>8s} {'ΔF1':>7s}")
    print(f"  {'─'*72}")
    prev = 0
    for name, f1_m, oob in milestones:
        delta = f1_m - prev if prev > 0 else 0
        marker = " ←" if name.startswith("Per-game") else ""
        print(f"  {name:<45s} {f1_m:>9.4f} {oob:>8.4f} {delta:>+7.4f}{marker}")
        prev = f1_m

    # ── 12. Production summary ───────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  PRODUCTION PIPELINE SUMMARY")
    print(f"{'═'*80}")
    print(f"""
  Model: Two-stage cascade LightGBM + IRT denoising
  Training data: {len(X_train)} reports, {X_train.shape[1]} features
  IRT: {len(theta)} contributors, {len(difficulty)} items

  Per-report:  F1={f1:.4f}  (borked={per[0]:.3f}, tinkering={per[1]:.3f}, oob={per[2]:.3f})
  Per-game:    F1={f1_game:.4f}  (borked={per_game[0]:.3f}, tinkering={per_game[1]:.3f}, oob={per_game[2]:.3f})
  Binary:      F1={f1_bin:.4f}  (borked vs works, per-game)

  Key innovations:
    1. IRT decomposition of annotator strictness + game difficulty (+0.030 F1)
    2. Contributor-aware relabeling replacing Cleanlab + Phase 8 (+0.017 F1)
    3. Per-game majority vote aggregation at inference (+0.09 F1)

  Total improvement: {f1_game - 0.7245:+.4f} F1 (0.7245 → {f1_game:.4f})
""")


if __name__ == "__main__":
    main()
