"""Phase 21.5 leakage check: aggregated model without verdict-derived features.

Tests:
  A. Full features (including borked_pct, oob_pct etc.) — original
  B. Without verdict-derived features (honest evaluation)
  C. Leave-out: test pair aggregates computed from TRAIN reports only

Usage:
  python scripts/experiment_21c_leakage_check.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_pairs(df, group_cols, difficulty, game_meta, pics_data, split_ts):
    pairs = []
    for key, group in df.groupby(group_cols):
        app_id = key[0] if isinstance(key, tuple) else key
        vendor = key[1] if isinstance(key, tuple) and len(key) > 1 else "all"
        n = len(group)
        scores = group["vscore"].values
        majority = Counter(scores).most_common(1)[0][0]

        feat = {
            "app_id": app_id, "vendor": vendor,
            "n_reports": n,
            # Verdict-derived (potentially leaky)
            "borked_pct": (scores == 0).mean(),
            "tinkering_pct": (scores == 1).mean(),
            "oob_pct": (scores == 2).mean(),
            "avg_vscore": scores.mean(),
            "std_vscore": scores.std() if n >= 2 else 0,
            "verdict_entropy": 0,
            "agreement": 0,
            # Non-leaky report aggregates
            "has_deck_reports": int(group["is_deck"].any()),
            "deck_pct": group["is_deck"].mean(),
            "n_variants": group["variant"].nunique(),
            "has_ge": int((group["variant"] == "ge").any()),
            "has_native": int((group["variant"] == "native").any()),
            "has_experimental": int((group["variant"] == "experimental").any()),
            "recency_days": (split_ts - group["timestamp"].max()) / 86400,
            "report_span_days": (group["timestamp"].max() - group["timestamp"].min()) / 86400,
            "target_3class": majority,
            "target_binary": int(majority > 0),
        }

        counts = np.bincount(scores, minlength=3)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        feat["verdict_entropy"] = float(-np.sum(probs * np.log2(probs)))
        feat["agreement"] = float(counts.max() / counts.sum())

        game_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
        feat["irt_difficulty_mean"] = np.mean(game_diffs) if game_diffs else 0
        feat["irt_difficulty_std"] = np.std(game_diffs) if len(game_diffs) >= 2 else 0

        gm = game_meta.get(app_id, {})
        feat["engine"] = gm.get("engine")
        feat["has_linux_native"] = gm.get("has_linux_native", 0) or 0
        feat["deck_status"] = gm.get("deck_status", 0) or 0
        feat["anticheat"] = gm.get("anticheat")
        feat["anticheat_status"] = gm.get("anticheat_status")
        feat["protondb_tier"] = gm.get("protondb_tier")
        feat["protondb_score"] = gm.get("protondb_score")
        feat["github_issues"] = gm.get("github_issue_count", 0) or 0
        feat["github_regression"] = gm.get("github_has_regression", 0) or 0

        pics = pics_data.get(app_id, {})
        feat["review_score"] = pics.get("review_score", 0) or 0
        feat["review_pct"] = pics.get("review_percentage", 0) or 0
        feat["recommended_runtime"] = pics.get("recommended_runtime")
        feat["is_runtime_native"] = int(pics.get("recommended_runtime") == "native")
        feat["is_runtime_proton"] = int("proton" in str(pics.get("recommended_runtime", "")))
        feat["steamos_compat"] = pics.get("steamos_compatibility", 0) or 0

        pairs.append(feat)
    return pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.irt import fit_irt

    conn = get_connection(args.db)
    theta, difficulty = fit_irt(conn)

    reports = pd.read_sql_query("""
        SELECT r.id, r.app_id, r.gpu, r.variant, r.timestamp,
               r.verdict, r.verdict_oob,
               CASE WHEN r.gpu LIKE '%anGogh%' OR r.gpu LIKE '%an Gogh%'
                    OR r.battery_performance IS NOT NULL THEN 1 ELSE 0 END as is_deck
        FROM reports r
    """, conn)
    reports["timestamp"] = pd.to_numeric(reports["timestamp"], errors="coerce").fillna(0)

    def vscore(row):
        if row["verdict"] == "no": return 0
        if row["verdict_oob"] == "yes": return 2
        return 1
    reports["vscore"] = reports.apply(vscore, axis=1)

    def get_vendor(gpu):
        if not gpu: return "other"
        g = gpu.lower()
        if any(k in g for k in ("nvidia","geforce","gtx","rtx")): return "nvidia"
        if any(k in g for k in ("amd","radeon","rx ")): return "amd"
        if "intel" in g: return "intel"
        return "other"
    reports["vendor"] = reports["gpu"].apply(get_vendor)

    game_meta = {}
    for r in conn.execute("SELECT * FROM game_metadata").fetchall():
        game_meta[r["app_id"]] = dict(r)
    pics_data = {}
    for r in conn.execute("SELECT app_id, data_json FROM enrichment_cache WHERE source='steam_pics'").fetchall():
        try:
            d = json.loads(r["data_json"])
            if "_empty" not in d: pics_data[r["app_id"]] = d
        except: pass
    conn.close()

    # Time split
    reports = reports.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(reports) * 0.8)
    split_ts = reports.iloc[split_idx]["timestamp"]
    train_reports = reports[reports["timestamp"] <= split_ts]
    test_reports = reports[reports["timestamp"] > split_ts]

    group_cols = ["app_id", "vendor"]

    cat_cols = ["vendor", "engine", "anticheat", "anticheat_status",
                "protondb_tier", "recommended_runtime"]

    # All features
    all_feats = ["n_reports", "borked_pct", "tinkering_pct", "oob_pct",
                 "verdict_entropy", "agreement", "avg_vscore", "std_vscore",
                 "has_deck_reports", "deck_pct", "n_variants",
                 "has_ge", "has_native", "has_experimental",
                 "recency_days", "report_span_days",
                 "irt_difficulty_mean", "irt_difficulty_std",
                 "has_linux_native", "deck_status",
                 "github_issues", "github_regression",
                 "review_score", "review_pct",
                 "is_runtime_native", "is_runtime_proton", "steamos_compat"] + cat_cols

    # Leak-free: remove verdict-derived
    leaky = {"borked_pct", "tinkering_pct", "oob_pct", "avg_vscore",
             "std_vscore", "verdict_entropy", "agreement"}
    clean_feats = [f for f in all_feats if f not in leaky]

    def train_eval(X_tr, X_te, y_tr, y_te, feat_cols, label, binary=False):
        cats = [c for c in cat_cols if c in feat_cols]
        Xtr, Xte = X_tr[feat_cols].copy(), X_te[feat_cols].copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype("category")
            Xte[c] = Xte[c].astype("category")

        kw = {"class_weight": {0: 3.0, 1: 1.0}} if binary else {}
        model = lgb.LGBMClassifier(
            n_estimators=1000, num_leaves=31, learning_rate=0.05,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbose=-1, **kw)
        model.fit(Xtr, y_tr, eval_set=[(Xte, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False)],
                  categorical_feature=cats)
        yp = model.predict(Xte)
        f1 = f1_score(y_te, yp, average="macro")
        acc = accuracy_score(y_te, yp)
        if binary:
            per = f1_score(y_te, yp, average=None)
            print(f"  {label:45s} F1={f1:.4f} acc={acc:.4f} b_f1={per[0]:.3f} w_f1={per[1]:.3f}")
        else:
            per = f1_score(y_te, yp, average=None)
            print(f"  {label:45s} F1={f1:.4f} acc={acc:.4f} b={per[0]:.3f} t={per[1]:.3f} o={per[2]:.3f}")
        return f1

    # ── A. Standard: train pairs from train, test pairs from test ────
    print(f"\n{'='*70}")
    print("A. STANDARD (potential leakage: test aggregates from test reports)")
    print(f"{'='*70}")

    train_pairs = build_pairs(train_reports, group_cols, difficulty, game_meta, pics_data, split_ts)
    test_pairs = build_pairs(test_reports, group_cols, difficulty, game_meta, pics_data, split_ts)
    logger.info("Standard: train=%d, test=%d", len(train_pairs), len(test_pairs))

    print("\n  3-class:")
    train_eval(train_pairs, test_pairs, train_pairs["target_3class"].values,
               test_pairs["target_3class"].values, all_feats, "all features", binary=False)
    train_eval(train_pairs, test_pairs, train_pairs["target_3class"].values,
               test_pairs["target_3class"].values, clean_feats, "without verdict-derived", binary=False)

    print("\n  Binary:")
    train_eval(train_pairs, test_pairs, train_pairs["target_binary"].values,
               test_pairs["target_binary"].values, all_feats, "all features", binary=True)
    train_eval(train_pairs, test_pairs, train_pairs["target_binary"].values,
               test_pairs["target_binary"].values, clean_feats, "without verdict-derived", binary=True)

    # ── B. Leave-out: test pair aggregates from TRAIN reports ────────
    print(f"\n{'='*70}")
    print("B. LEAVE-OUT (test aggregates from train reports only — honest)")
    print(f"{'='*70}")

    # For test pairs: aggregate only train reports for same (game, vendor)
    train_by_pair = defaultdict(list)
    for _, row in train_reports.iterrows():
        key = (row["app_id"], get_vendor(row["gpu"]))
        train_by_pair[key].append(row["vscore"])

    # Rebuild test pairs using train-only aggregates
    test_pairs_lo = test_pairs.copy()
    for i, row in test_pairs_lo.iterrows():
        key = (row["app_id"], row["vendor"])
        train_scores = train_by_pair.get(key, [])
        if train_scores:
            scores = np.array(train_scores)
            test_pairs_lo.at[i, "borked_pct"] = (scores == 0).mean()
            test_pairs_lo.at[i, "tinkering_pct"] = (scores == 1).mean()
            test_pairs_lo.at[i, "oob_pct"] = (scores == 2).mean()
            test_pairs_lo.at[i, "avg_vscore"] = scores.mean()
            test_pairs_lo.at[i, "std_vscore"] = scores.std() if len(scores) >= 2 else 0
            counts = np.bincount(scores, minlength=3)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            test_pairs_lo.at[i, "verdict_entropy"] = float(-np.sum(probs * np.log2(probs)))
            test_pairs_lo.at[i, "agreement"] = float(counts.max() / counts.sum())
            test_pairs_lo.at[i, "n_reports"] = len(train_scores)
        else:
            # No train reports for this pair — cold start
            for col in ["borked_pct", "tinkering_pct", "oob_pct", "avg_vscore",
                         "std_vscore", "verdict_entropy", "agreement"]:
                test_pairs_lo.at[i, col] = 0
            test_pairs_lo.at[i, "n_reports"] = 0

    n_cold = (test_pairs_lo["n_reports"] == 0).sum()
    logger.info("Leave-out: %d test pairs, %d cold-start (%.1f%%)",
                len(test_pairs_lo), n_cold, n_cold / len(test_pairs_lo) * 100)

    print("\n  3-class:")
    train_eval(train_pairs, test_pairs_lo, train_pairs["target_3class"].values,
               test_pairs_lo["target_3class"].values, all_feats, "leave-out all features", binary=False)
    train_eval(train_pairs, test_pairs_lo, train_pairs["target_3class"].values,
               test_pairs_lo["target_3class"].values, clean_feats, "leave-out without verdict-derived", binary=False)

    print("\n  Binary:")
    f1_lo_all = train_eval(train_pairs, test_pairs_lo, train_pairs["target_binary"].values,
               test_pairs_lo["target_binary"].values, all_feats, "leave-out all features", binary=True)
    f1_lo_clean = train_eval(train_pairs, test_pairs_lo, train_pairs["target_binary"].values,
               test_pairs_lo["target_binary"].values, clean_feats, "leave-out without verdict-derived", binary=True)

    # ── C. Leave-out, exclude cold-start pairs ───────────────────────
    print(f"\n{'='*70}")
    print("C. LEAVE-OUT, only pairs with train reports (no cold-start)")
    print(f"{'='*70}")

    has_train = test_pairs_lo["n_reports"] > 0
    if has_train.sum() > 100:
        print(f"\n  Pairs with train reports: {has_train.sum()}/{len(test_pairs_lo)}")
        print("\n  3-class:")
        train_eval(train_pairs, test_pairs_lo[has_train].reset_index(drop=True),
                   train_pairs["target_3class"].values,
                   test_pairs_lo[has_train]["target_3class"].values,
                   all_feats, "known pairs, all features", binary=False)
        print("\n  Binary:")
        train_eval(train_pairs, test_pairs_lo[has_train].reset_index(drop=True),
                   train_pairs["target_binary"].values,
                   test_pairs_lo[has_train]["target_binary"].values,
                   all_feats, "known pairs, all features", binary=True)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Standard (leaky):           3-class, binary — see above")
    print(f"  Leave-out (honest):         binary F1={f1_lo_all:.4f} (all), {f1_lo_clean:.4f} (clean)")
    print(f"  Cold-start pairs: {n_cold}/{len(test_pairs_lo)} ({n_cold/len(test_pairs_lo)*100:.0f}%)")


if __name__ == "__main__":
    main()
