"""Phase 21.5: Aggregated model + confidence scoring + cold-start.

Experiments:
  21.5  — Model trained on per-(game, vendor) pairs instead of individual reports
  21.6  — Confidence scoring (n_reports, agreement, model confidence)
  21.7  — Cold-start model (game metadata only, no reports)

Usage:
  python scripts/experiment_21b_aggregated_model.py [--db data/protondb.db]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.irt import fit_irt

    conn = get_connection(args.db)

    # ── Build per-pair dataset ───────────────────────────────────────
    print("Building per-pair dataset...")

    theta, difficulty = fit_irt(conn)

    # Get all reports with metadata
    reports = pd.read_sql_query("""
        SELECT r.id, r.app_id, r.gpu, r.variant, r.timestamp,
               r.verdict, r.verdict_oob,
               r.battery_performance,
               CASE WHEN r.gpu LIKE '%anGogh%' OR r.gpu LIKE '%an Gogh%'
                    OR r.battery_performance IS NOT NULL THEN 1 ELSE 0 END as is_deck
        FROM reports r
    """, conn)
    reports["timestamp"] = pd.to_numeric(reports["timestamp"], errors="coerce").fillna(0)

    # Compute verdict score
    def verdict_score(row):
        if row["verdict"] == "no":
            return 0
        if row["verdict_oob"] == "yes":
            return 2
        return 1

    reports["vscore"] = reports.apply(verdict_score, axis=1)
    reports["gpu_family"] = reports["gpu"].apply(lambda g: extract_gpu_family(g) if g else "unknown")

    # GPU vendor
    def get_vendor(gpu):
        if not gpu:
            return "other"
        gpu_l = gpu.lower()
        if any(k in gpu_l for k in ("nvidia", "geforce", "gtx", "rtx")):
            return "nvidia"
        if any(k in gpu_l for k in ("amd", "radeon", "rx ")):
            return "amd"
        if "intel" in gpu_l:
            return "intel"
        return "other"

    reports["vendor"] = reports["gpu"].apply(get_vendor)

    # Game metadata
    game_meta = {}
    for r in conn.execute("""
        SELECT app_id, engine, has_linux_native, deck_status, anticheat,
               anticheat_status, protondb_tier, protondb_score,
               github_issue_count, github_has_regression
        FROM game_metadata
    """).fetchall():
        game_meta[r["app_id"]] = dict(r)

    # PICS data
    pics_data = {}
    for r in conn.execute("SELECT app_id, data_json FROM enrichment_cache WHERE source='steam_pics'").fetchall():
        try:
            d = json.loads(r["data_json"])
            if "_empty" not in d:
                pics_data[r["app_id"]] = d
        except Exception:
            pass

    conn.close()

    # ── Build per-(game, vendor) pairs ───────────────────────────────
    print("Aggregating pairs...")

    # Time-based split on reports first
    reports = reports.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(reports) * 0.8)
    split_ts = reports.iloc[split_idx]["timestamp"]

    train_reports = reports[reports["timestamp"] <= split_ts]
    test_reports = reports[reports["timestamp"] > split_ts]

    def build_pair_features(df, group_cols=["app_id", "vendor"]):
        pairs = []
        for key, group in df.groupby(group_cols):
            app_id = key[0] if isinstance(key, tuple) else key
            vendor = key[1] if isinstance(key, tuple) and len(key) > 1 else "all"

            n = len(group)
            scores = group["vscore"].values
            majority = Counter(scores).most_common(1)[0][0]

            # Aggregate features
            feat = {
                "app_id": app_id,
                "vendor": vendor,
                "n_reports": n,
                "borked_pct": (scores == 0).mean(),
                "tinkering_pct": (scores == 1).mean(),
                "oob_pct": (scores == 2).mean(),
                "verdict_entropy": 0,
                "agreement": 0,
                "avg_vscore": scores.mean(),
                "std_vscore": scores.std() if n >= 2 else 0,
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

            # Entropy
            counts = np.bincount(scores, minlength=3)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            feat["verdict_entropy"] = float(-np.sum(probs * np.log2(probs)))
            feat["agreement"] = float(counts.max() / counts.sum())

            # IRT difficulty for this game
            game_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
            feat["irt_difficulty_mean"] = np.mean(game_diffs) if game_diffs else 0
            feat["irt_difficulty_std"] = np.std(game_diffs) if len(game_diffs) >= 2 else 0

            # Game metadata
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

            # PICS
            pics = pics_data.get(app_id, {})
            feat["review_score"] = pics.get("review_score", 0) or 0
            feat["review_pct"] = pics.get("review_percentage", 0) or 0
            feat["recommended_runtime"] = pics.get("recommended_runtime")
            feat["is_runtime_native"] = int(pics.get("recommended_runtime") == "native")
            feat["is_runtime_proton"] = int("proton" in str(pics.get("recommended_runtime", "")))
            feat["steamos_compat"] = pics.get("steamos_compatibility", 0) or 0

            pairs.append(feat)

        return pd.DataFrame(pairs)

    train_pairs = build_pair_features(train_reports)
    test_pairs = build_pair_features(test_reports)

    logger.info("Train pairs: %d, Test pairs: %d", len(train_pairs), len(test_pairs))
    logger.info("Train class dist: %s", dict(Counter(train_pairs["target_3class"])))

    # ── Feature columns ──────────────────────────────────────────────
    cat_cols = ["vendor", "engine", "anticheat", "anticheat_status",
                "protondb_tier", "recommended_runtime"]
    num_cols = ["n_reports", "borked_pct", "tinkering_pct", "oob_pct",
                "verdict_entropy", "agreement", "avg_vscore", "std_vscore",
                "has_deck_reports", "deck_pct", "n_variants",
                "has_ge", "has_native", "has_experimental",
                "recency_days", "report_span_days",
                "irt_difficulty_mean", "irt_difficulty_std",
                "has_linux_native", "deck_status",
                "github_issues", "github_regression",
                "review_score", "review_pct",
                "is_runtime_native", "is_runtime_proton", "steamos_compat"]
    feature_cols = num_cols + cat_cols

    X_train_p = train_pairs[feature_cols].copy()
    X_test_p = test_pairs[feature_cols].copy()
    y_train_3 = train_pairs["target_3class"].values
    y_test_3 = test_pairs["target_3class"].values
    y_train_bin = train_pairs["target_binary"].values
    y_test_bin = test_pairs["target_binary"].values

    for c in cat_cols:
        X_train_p[c] = X_train_p[c].astype("category")
        X_test_p[c] = X_test_p[c].astype("category")

    # ── 21.5a: 3-class aggregated model ──────────────────────────────
    print(f"\n{'='*70}")
    print("21.5a: 3-class aggregated model (per-game-vendor pairs)")
    print(f"{'='*70}")

    model_3 = lgb.LGBMClassifier(
        n_estimators=1000, num_leaves=31, learning_rate=0.05,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=42, verbose=-1,
    )
    model_3.fit(X_train_p, y_train_3,
                eval_set=[(X_test_p, y_test_3)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
                categorical_feature=cat_cols)

    y_pred_3 = model_3.predict(X_test_p)
    f1_3 = f1_score(y_test_3, y_pred_3, average="macro")
    per_3 = f1_score(y_test_3, y_pred_3, average=None)
    acc_3 = accuracy_score(y_test_3, y_pred_3)
    print(f"  3-class: F1={f1_3:.4f} acc={acc_3:.4f} b={per_3[0]:.3f} t={per_3[1]:.3f} o={per_3[2]:.3f}")

    # Feature importance
    imp = pd.Series(model_3.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top-10 features:")
    for feat, v in imp.head(10).items():
        print(f"    {feat:30s} {v:6.0f}")

    # ── 21.5b: Binary aggregated model ───────────────────────────────
    print(f"\n{'='*70}")
    print("21.5b: Binary aggregated model (borked vs works)")
    print(f"{'='*70}")

    model_bin = lgb.LGBMClassifier(
        n_estimators=1000, num_leaves=31, learning_rate=0.05,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, class_weight={0: 3.0, 1: 1.0},
        n_jobs=-1, random_state=42, verbose=-1,
    )
    model_bin.fit(X_train_p, y_train_bin,
                  eval_set=[(X_test_p, y_test_bin)],
                  callbacks=[lgb.early_stopping(50, verbose=False)],
                  categorical_feature=cat_cols)

    y_pred_bin = model_bin.predict(X_test_p)
    y_proba_bin = model_bin.predict_proba(X_test_p)
    f1_bin = f1_score(y_test_bin, y_pred_bin, average="macro")
    acc_bin = accuracy_score(y_test_bin, y_pred_bin)
    borked_r = (y_pred_bin[y_test_bin == 0] == 0).mean() if (y_test_bin == 0).any() else 0
    borked_p = (y_test_bin[y_pred_bin == 0] == 0).mean() if (y_pred_bin == 0).any() else 0
    print(f"  Binary: F1={f1_bin:.4f} acc={acc_bin:.4f} borked_r={borked_r:.3f} borked_p={borked_p:.3f}")

    # ── 21.6: Confidence scoring ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("21.6: Confidence scoring")
    print(f"{'='*70}")

    # Confidence = f(n_reports, agreement, model_confidence)
    model_conf = y_proba_bin.max(axis=1)
    n_reports = test_pairs["n_reports"].values
    agreement = test_pairs["agreement"].values

    # Combined confidence
    conf_combined = (
        0.3 * np.clip(np.log1p(n_reports) / np.log1p(50), 0, 1) +  # report count
        0.3 * agreement +                                             # verdict agreement
        0.4 * model_conf                                              # model confidence
    )

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = conf_combined >= thresh
        if mask.sum() > 10:
            f1_c = f1_score(y_test_bin[mask], y_pred_bin[mask], average="macro")
            acc_c = accuracy_score(y_test_bin[mask], y_pred_bin[mask])
            print(f"  conf >= {thresh:.1f}: {mask.sum():5d} ({mask.sum()/len(y_test_bin)*100:.0f}%) "
                  f"F1={f1_c:.4f} acc={acc_c:.4f}")

    # ── 21.7: Cold-start model ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("21.7: Cold-start model (game metadata only, no report aggregates)")
    print(f"{'='*70}")

    cold_cols = ["vendor", "engine", "anticheat", "anticheat_status",
                 "has_linux_native", "deck_status", "protondb_tier",
                 "github_issues", "github_regression",
                 "review_score", "review_pct",
                 "is_runtime_native", "is_runtime_proton", "steamos_compat",
                 "irt_difficulty_mean"]

    X_train_cold = train_pairs[cold_cols].copy()
    X_test_cold = test_pairs[cold_cols].copy()
    cold_cats = [c for c in cat_cols if c in cold_cols]
    for c in cold_cats:
        X_train_cold[c] = X_train_cold[c].astype("category")
        X_test_cold[c] = X_test_cold[c].astype("category")

    model_cold = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=15, learning_rate=0.05,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight={0: 3.0, 1: 1.0},
        n_jobs=-1, random_state=42, verbose=-1,
    )
    model_cold.fit(X_train_cold, y_train_bin,
                   eval_set=[(X_test_cold, y_test_bin)],
                   callbacks=[lgb.early_stopping(50, verbose=False)],
                   categorical_feature=cold_cats)

    y_pred_cold = model_cold.predict(X_test_cold)
    f1_cold = f1_score(y_test_bin, y_pred_cold, average="macro")
    acc_cold = accuracy_score(y_test_bin, y_pred_cold)
    print(f"  Cold-start binary: F1={f1_cold:.4f} acc={acc_cold:.4f}")

    # Cold-start 3-class
    model_cold_3 = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=15, learning_rate=0.05,
        min_child_samples=20, n_jobs=-1, random_state=42, verbose=-1,
    )
    model_cold_3.fit(X_train_cold, y_train_3,
                     eval_set=[(X_test_cold, y_test_3)],
                     callbacks=[lgb.early_stopping(50, verbose=False)],
                     categorical_feature=cold_cats)

    y_pred_cold_3 = model_cold_3.predict(X_test_cold)
    f1_cold_3 = f1_score(y_test_3, y_pred_cold_3, average="macro")
    print(f"  Cold-start 3-class: F1={f1_cold_3:.4f}")

    imp_cold = pd.Series(model_cold.feature_importances_, index=cold_cols).sort_values(ascending=False)
    print(f"\n  Cold-start top features:")
    for feat, v in imp_cold.head(10).items():
        print(f"    {feat:30s} {v:6.0f}")

    # ── Compare: how much do reports help? ───────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON: report aggregates vs cold-start")
    print(f"{'='*70}")

    # Split test pairs by n_reports
    for min_n in [1, 3, 5, 10, 20]:
        mask = n_reports >= min_n
        if mask.sum() > 50:
            f1_full = f1_score(y_test_bin[mask], y_pred_bin[mask], average="macro")
            f1_cs = f1_score(y_test_bin[mask], y_pred_cold[mask], average="macro")
            print(f"  n>={min_n:2d}: full F1={f1_full:.4f}, cold F1={f1_cs:.4f}, "
                  f"gap={f1_full-f1_cs:+.4f} (n={mask.sum()})")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Per-report baseline (Phase 17):  F1=0.7801 (3-class)")
    print(f"  Per-pair vote (Phase 21.1):       F1=0.8712 (3-class), 0.9430 (binary)")
    print(f"  Aggregated model 3-class:        F1={f1_3:.4f}")
    print(f"  Aggregated model binary:         F1={f1_bin:.4f}")
    print(f"  Cold-start model binary:         F1={f1_cold:.4f}")
    print(f"  Cold-start model 3-class:        F1={f1_cold_3:.4f}")


if __name__ == "__main__":
    main()
