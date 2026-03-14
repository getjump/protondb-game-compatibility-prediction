"""Phase 21.1-21.3: Multi-level aggregation evaluation.

Experiments:
  21.1  — Compare aggregation groupings (game, game+vendor, game+deck, etc.)
  21.2  — Hierarchical fallback evaluation
  21.3  — Temporal-aware aggregation (recency weighting)

Usage:
  python scripts/experiment_21_aggregation.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

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

    # Build report metadata
    report_meta = {}
    for r in conn.execute("""
        SELECT id, app_id, gpu, variant, timestamp,
               CASE WHEN gpu LIKE '%anGogh%' OR gpu LIKE '%an Gogh%'
                    OR battery_performance IS NOT NULL THEN 1 ELSE 0 END as is_deck
        FROM reports
    """).fetchall():
        gpu = r["gpu"] or ""
        vendor = "nvidia" if any(k in gpu.lower() for k in ("nvidia", "geforce", "gtx", "rtx")) \
            else "amd" if any(k in gpu.lower() for k in ("amd", "radeon", "rx ")) \
            else "intel" if "intel" in gpu.lower() else "other"
        report_meta[r["id"]] = {
            "app_id": r["app_id"],
            "gpu_family": extract_gpu_family(gpu) if gpu else "unknown",
            "vendor": vendor,
            "variant": r["variant"] or "unknown",
            "is_deck": "deck" if r["is_deck"] else "desktop",
            "timestamp": int(r["timestamp"]) if r["timestamp"] else 0,
        }
    conn.close()

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Train cascade
    print("Training cascade...")
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_drops)

    y_pred = cascade.predict(X_test)
    y_proba = cascade.predict_proba(X_test)
    confidence = y_proba.max(axis=1)

    print(f"\nPer-report baseline: F1={f1_score(y_test, y_pred, average='macro'):.4f}")

    # ── 21.1: Multi-level aggregation ────────────────────────────────
    print(f"\n{'='*70}")
    print("21.1: Multi-level aggregation comparison")
    print(f"{'='*70}")

    def get_group_key(rid, dims):
        meta = report_meta.get(rid, {})
        parts = []
        for d in dims:
            if d == "game":
                parts.append(str(meta.get("app_id", "?")))
            elif d == "vendor":
                parts.append(meta.get("vendor", "?"))
            elif d == "gpu_family":
                parts.append(meta.get("gpu_family", "?"))
            elif d == "variant":
                parts.append(meta.get("variant", "?"))
            elif d == "deck":
                parts.append(meta.get("is_deck", "?"))
        return tuple(parts)

    def eval_aggregation(dims, name, temporal_weight=False):
        groups = defaultdict(lambda: {"preds": [], "truths": [], "probas": [], "timestamps": []})
        for i, rid in enumerate(test_rids):
            key = get_group_key(rid, dims)
            groups[key]["preds"].append(y_pred[i])
            groups[key]["truths"].append(y_test[i])
            groups[key]["probas"].append(y_proba[i])
            groups[key]["timestamps"].append(report_meta.get(rid, {}).get("timestamp", 0))

        agg_true_3, agg_pred_3 = [], []  # 3-class
        agg_true_2, agg_pred_2 = [], []  # binary
        sizes = []
        reliable_true_3, reliable_pred_3 = [], []  # 3+ reports
        reliable_true_2, reliable_pred_2 = [], []

        for key, data in groups.items():
            n = len(data["preds"])
            sizes.append(n)

            if temporal_weight and n > 1:
                # Recency-weighted vote
                ts_arr = np.array(data["timestamps"], dtype=float)
                max_ts = ts_arr.max()
                weights = np.exp(-(max_ts - ts_arr) / (365 * 86400) * np.log(2))
                # Weighted vote per class
                votes = np.zeros(3)
                for p, w in zip(data["preds"], weights):
                    votes[p] += w
                pred_agg = votes.argmax()
            else:
                pred_agg = Counter(data["preds"]).most_common(1)[0][0]

            true_agg = Counter(data["truths"]).most_common(1)[0][0]

            agg_true_3.append(true_agg)
            agg_pred_3.append(pred_agg)
            agg_true_2.append(int(true_agg > 0))
            agg_pred_2.append(int(pred_agg > 0))

            if n >= 3:
                reliable_true_3.append(true_agg)
                reliable_pred_3.append(pred_agg)
                reliable_true_2.append(int(true_agg > 0))
                reliable_pred_2.append(int(pred_agg > 0))

        agg_true_3 = np.array(agg_true_3)
        agg_pred_3 = np.array(agg_pred_3)

        f1_3 = f1_score(agg_true_3, agg_pred_3, average="macro")
        acc_3 = accuracy_score(agg_true_3, agg_pred_3)
        per_3 = f1_score(agg_true_3, agg_pred_3, average=None)
        f1_bin = f1_score(agg_true_2, agg_pred_2, average="macro")
        acc_bin = accuracy_score(agg_true_2, agg_pred_2)

        n_reliable = len(reliable_true_3)
        f1_rel = f1_score(reliable_true_3, reliable_pred_3, average="macro") if n_reliable > 50 else 0
        f1_rel_bin = f1_score(reliable_true_2, reliable_pred_2, average="macro") if n_reliable > 50 else 0

        print(f"\n  {name}")
        print(f"    Pairs: {len(agg_true_3):6d} (median {np.median(sizes):.0f} reports)")
        print(f"    3-class: F1={f1_3:.4f} acc={acc_3:.4f} b={per_3[0]:.3f} t={per_3[1]:.3f} o={per_3[2]:.3f}")
        print(f"    Binary:  F1={f1_bin:.4f} acc={acc_bin:.4f}")
        if n_reliable > 50:
            print(f"    3+ reports ({n_reliable:d} pairs): 3-class F1={f1_rel:.4f}, binary F1={f1_rel_bin:.4f}")

        return {"name": name, "pairs": len(agg_true_3), "f1_3": f1_3, "f1_bin": f1_bin,
                "f1_rel": f1_rel, "reliable": n_reliable}

    results = []

    # Different groupings
    groupings = [
        (["game"], "(game)"),
        (["game", "vendor"], "(game, vendor)"),
        (["game", "deck"], "(game, deck/desktop)"),
        (["game", "variant"], "(game, variant)"),
        (["game", "vendor", "deck"], "(game, vendor, deck)"),
        (["game", "vendor", "variant"], "(game, vendor, variant)"),
        (["game", "gpu_family"], "(game, gpu_family)"),
    ]

    for dims, name in groupings:
        r = eval_aggregation(dims, name)
        results.append(r)

    # ── 21.2: Hierarchical fallback ──────────────────────────────────
    print(f"\n{'='*70}")
    print("21.2: Hierarchical fallback")
    print(f"{'='*70}")

    # For each test report, find the most specific group with 3+ reports
    levels = [
        (["game", "vendor", "variant", "deck"], "L1: game+vendor+variant+deck"),
        (["game", "vendor", "deck"], "L2: game+vendor+deck"),
        (["game", "vendor"], "L3: game+vendor"),
        (["game", "deck"], "L4: game+deck"),
        (["game"], "L5: game"),
    ]

    # Pre-compute group data for all levels
    level_groups = {}
    for dims, _ in levels:
        groups = defaultdict(lambda: {"preds": [], "truths": []})
        for i, rid in enumerate(test_rids):
            key = get_group_key(rid, dims)
            groups[key]["preds"].append(y_pred[i])
            groups[key]["truths"].append(y_test[i])
        level_groups[tuple(dims)] = groups

    # Per-report: use most specific level with 3+ reports
    hier_preds = []
    hier_levels_used = Counter()

    for i, rid in enumerate(test_rids):
        predicted = False
        for dims, level_name in levels:
            key = get_group_key(rid, dims)
            group = level_groups[tuple(dims)].get(key)
            if group and len(group["preds"]) >= 3:
                pred = Counter(group["preds"]).most_common(1)[0][0]
                hier_preds.append(pred)
                hier_levels_used[level_name] += 1
                predicted = True
                break
        if not predicted:
            # Fallback to individual prediction
            hier_preds.append(y_pred[i])
            hier_levels_used["L6: individual"] += 1

    hier_preds = np.array(hier_preds)
    f1_hier = f1_score(y_test, hier_preds, average="macro")
    per_hier = f1_score(y_test, hier_preds, average=None)
    acc_hier = accuracy_score(y_test, hier_preds)

    print(f"\n  Hierarchical: F1={f1_hier:.4f} acc={acc_hier:.4f} "
          f"b={per_hier[0]:.3f} t={per_hier[1]:.3f} o={per_hier[2]:.3f}")
    print(f"  Level usage:")
    for level_name, count in sorted(hier_levels_used.items()):
        print(f"    {level_name:40s} {count:6d} ({count/len(y_test)*100:.1f}%)")

    # Binary hierarchical
    hier_bin = (hier_preds > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)
    f1_hier_bin = f1_score(y_test_bin, hier_bin, average="macro")
    print(f"\n  Hierarchical binary: F1={f1_hier_bin:.4f}")

    # ── 21.3: Temporal-aware aggregation ─────────────────────────────
    print(f"\n{'='*70}")
    print("21.3: Temporal-aware aggregation (recency weighting)")
    print(f"{'='*70}")

    for dims, name in [
        (["game"], "(game) temporal"),
        (["game", "vendor"], "(game, vendor) temporal"),
        (["game", "deck"], "(game, deck) temporal"),
    ]:
        eval_aggregation(dims, name, temporal_weight=True)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Per-report baseline:    F1={f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  Hierarchical fallback:  F1={f1_hier:.4f}")
    print(f"  Hierarchical binary:    F1={f1_hier_bin:.4f}")
    print(f"\n  {'Grouping':<35s} {'Pairs':>7s} {'3-class':>8s} {'Binary':>8s} {'Reliable':>8s}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['name']:<35s} {r['pairs']:>7d} {r['f1_3']:>8.4f} {r['f1_bin']:>8.4f} "
              f"{r['f1_rel']:>8.4f} ({r['reliable']})")


if __name__ == "__main__":
    main()
