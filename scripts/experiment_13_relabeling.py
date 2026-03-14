"""Phase 13 experiments: contributor-aware relabeling.

Experiments:
  13.1  — IRT-informed relabeling (replace Cleanlab deletion with IRT correction)
  13.2  — Contributor-aware rule-based relabeling (graduated Phase 8)
  13.3  — Hybrid noise pipeline (IRT + actions + Cleanlab fallback)

Usage:
  python scripts/experiment_13_relabeling.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Data loading (shared with experiment_12) ─────────────────────────


def load_data_raw(db_path: str):
    """Load feature matrix WITHOUT relabeling or Cleanlab — we'll apply our own."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split

    conn = get_connection(db_path)

    emb_path = Path(db_path).parent / "embeddings.npz"
    emb_data = load_embeddings(emb_path)

    X, y, timestamps, report_ids, label_maps = _build_feature_matrix(conn, emb_data)

    X_train, X_test, y_train, y_test, train_rids, test_rids = _time_based_split(
        X, y, timestamps, 0.2, report_ids=report_ids
    )

    # Load contributor data
    contrib_df = pd.read_sql_query(
        "SELECT report_id, contributor_id, report_tally, playtime, playtime_linux "
        "FROM report_contributors", conn,
    )
    contrib_map = contrib_df.set_index("report_id")

    # Load relabel IDs (Phase 8 rule-based)
    from protondb_settings.ml.relabeling import get_relabel_ids
    relabel_ids = get_relabel_ids(conn)

    conn.close()
    return X_train, X_test, y_train, y_test, train_rids, test_rids, contrib_map, relabel_ids


# ── IRT fitting (reused from experiment_12) ──────────────────────────


def fit_irt(db_path: str) -> tuple[dict, dict]:
    """Fit 1PL IRT. Returns (theta, difficulty)."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    df = pd.read_sql_query("""
        SELECT r.id, r.app_id, r.gpu, r.verdict, r.verdict_oob, rc.contributor_id
        FROM reports r
        JOIN report_contributors rc ON r.id = rc.report_id
        WHERE r.verdict = 'yes'
    """, conn)
    conn.close()

    df["response"] = (df["verdict_oob"] != "yes").astype(int)
    df["gpu_family"] = df["gpu"].apply(lambda g: extract_gpu_family(g) if g else "unknown")
    df["item"] = df["app_id"].astype(str) + "_" + df["gpu_family"]

    # Iterative filtering
    for _ in range(3):
        item_counts = df["item"].value_counts()
        df = df[df["item"].isin(item_counts[item_counts >= 2].index)]
        contrib_counts = df["contributor_id"].value_counts()
        df = df[df["contributor_id"].isin(contrib_counts[contrib_counts >= 2].index)]

    if len(df) < 100:
        return {}, {}

    contributors = df["contributor_id"].unique()
    items = df["item"].unique()
    c_to_idx = {c: i for i, c in enumerate(contributors)}
    i_to_idx = {it: i for i, it in enumerate(items)}

    c_idx = df["contributor_id"].map(c_to_idx).values
    i_idx = df["item"].map(i_to_idx).values
    responses = df["response"].values.astype(float)

    n_c, n_i = len(contributors), len(items)

    def neg_log_likelihood(params):
        theta, d = params[:n_c], params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        p = np.clip(1 / (1 + np.exp(-logit)), 1e-7, 1 - 1e-7)
        ll = responses * np.log(p) + (1 - responses) * np.log(1 - p)
        return -ll.sum() + 0.01 * (np.sum(theta ** 2) + np.sum(d ** 2))

    def grad_nll(params):
        theta, d = params[:n_c], params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        residual = 1 / (1 + np.exp(-logit)) - responses
        g_theta, g_d = np.zeros(n_c), np.zeros(n_i)
        np.add.at(g_theta, c_idx, residual)
        np.add.at(g_d, i_idx, -residual)
        return np.concatenate([g_theta + 0.02 * theta, g_d + 0.02 * d])

    x0 = np.zeros(n_c + n_i)
    for c, idx in c_to_idx.items():
        ratio = np.clip(df.loc[df["contributor_id"] == c, "response"].mean(), 0.05, 0.95)
        x0[idx] = np.log(ratio / (1 - ratio))
    for item, idx in i_to_idx.items():
        ratio = np.clip(df.loc[df["item"] == item, "response"].mean(), 0.05, 0.95)
        x0[n_c + idx] = -np.log(ratio / (1 - ratio))

    result = minimize(neg_log_likelihood, x0, jac=grad_nll, method="L-BFGS-B",
                     options={"maxiter": 1000, "ftol": 1e-8})
    logger.info("IRT converged: %s, %.1fs, %d iters, nll=%.1f",
                result.success, 0, result.nit, result.fun)

    theta = {c: result.x[idx] for c, idx in c_to_idx.items()}
    difficulty = {}
    for item_key, idx in i_to_idx.items():
        parts = item_key.rsplit("_", 1)
        difficulty[(int(parts[0]), parts[1])] = result.x[n_c + idx]

    return theta, difficulty


# ── IRT feature addition (for cascade eval) ──────────────────────────


def add_irt_features(X, report_ids, contrib_map, theta, difficulty, db_path):
    """Add irt_game_difficulty and irt_contributor_strictness."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    report_info = {}
    for r in conn.execute("SELECT id, app_id, gpu FROM reports").fetchall():
        report_info[r["id"]] = (r["app_id"], r["gpu"])
    conn.close()

    X = X.copy()
    irt_d, irt_t = [], []
    for rid in report_ids:
        app_id, gpu = report_info.get(rid, (None, None))
        gpu_fam = extract_gpu_family(gpu) if gpu else "unknown"
        d = difficulty.get((app_id, gpu_fam))
        if d is None and app_id:
            app_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
            d = np.mean(app_diffs) if app_diffs else None
        irt_d.append(d if d is not None else np.nan)

        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            irt_t.append(theta.get(str(row["contributor_id"]), np.nan))
        else:
            irt_t.append(np.nan)

    X["irt_game_difficulty"] = pd.Series(irt_d).fillna(pd.Series(irt_d).median()).values
    X["irt_contributor_strictness"] = pd.Series(irt_t).fillna(0).values
    return X


# ── Cascade training and eval ────────────────────────────────────────


def train_and_eval(X_train, y_train, X_test, y_test, sample_weight=None, label=""):
    """Train cascade and return results dict."""
    from protondb_settings.ml.models.cascade import train_stage1, CascadeClassifier
    from protondb_settings.ml.models.cascade import STAGE2_DROP_FEATURES
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    s1 = train_stage1(X_train, y_train, X_test, y_test)

    # Stage 2 with optional weights
    train_mask = y_train > 0
    test_mask = y_test > 0
    X_tr2 = X_train[train_mask].reset_index(drop=True)
    y_tr2 = (y_train[train_mask] - 1).astype(float)
    X_te2 = X_test[test_mask].reset_index(drop=True)
    y_te2 = (y_test[test_mask] - 1).astype(float)

    drops = [c for c in STAGE2_DROP_FEATURES if c in X_tr2.columns]
    if drops:
        X_tr2 = X_tr2.drop(columns=drops)
        X_te2 = X_te2.drop(columns=drops)

    cats = [c for c in CATEGORICAL_FEATURES if c in X_tr2.columns]
    for c in cats:
        X_tr2[c] = X_tr2[c].astype("category")
        X_te2[c] = X_te2[c].astype("category")

    y_smooth = y_tr2 * 0.85 + (1 - y_tr2) * 0.15
    w = sample_weight[train_mask] if sample_weight is not None else None

    ds_tr = lgb.Dataset(X_tr2, label=y_smooth, weight=w, categorical_feature=cats)
    ds_te = lgb.Dataset(X_te2, label=y_te2, categorical_feature=cats)

    s2 = lgb.train(
        {"objective": "cross_entropy", "metric": "binary_logloss",
         "num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 50,
         "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
         "reg_alpha": 0.1, "reg_lambda": 0.1, "min_split_gain": 0.05, "verbose": -1},
        ds_tr, num_boost_round=3000, valid_sets=[ds_te],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    cascade = CascadeClassifier(s1, s2, drops)
    y_pred = cascade.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    per = f1_score(y_test, y_pred, average=None)
    return {"label": label, "f1_macro": f1, "borked_f1": per[0],
            "tinkering_f1": per[1], "works_oob_f1": per[2]}


# ── Relabeling strategies ────────────────────────────────────────────


def apply_phase8_relabel(y, report_ids, relabel_ids):
    """Original Phase 8: tinkering → oob if no effort markers."""
    y_new = y.copy()
    n = 0
    for i, rid in enumerate(report_ids):
        if rid in relabel_ids and y_new[i] == 1:
            y_new[i] = 2
            n += 1
    return y_new, n


def irt_relabel(y, report_ids, contrib_map, theta, difficulty, db_path,
                threshold_low=0.3, threshold_high=0.7):
    """Phase 13.1: IRT-informed relabeling for Stage 2 boundary."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    report_info = {}
    for r in conn.execute("SELECT id, app_id, gpu FROM reports").fetchall():
        report_info[r["id"]] = (r["app_id"], r["gpu"])
    conn.close()

    y_new = y.copy()
    n_tink_to_oob = 0
    n_oob_to_tink = 0

    for i, rid in enumerate(report_ids):
        if y_new[i] not in (1, 2):  # only fix Stage 2 boundary
            continue

        # Need contributor data
        if rid not in contrib_map.index:
            continue
        row = contrib_map.loc[rid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        cid = str(row["contributor_id"])
        t = theta.get(cid)
        if t is None:
            continue

        # Need item difficulty
        app_id, gpu = report_info.get(rid, (None, None))
        if app_id is None:
            continue
        gpu_fam = extract_gpu_family(gpu) if gpu else "unknown"
        d = difficulty.get((app_id, gpu_fam))
        if d is None:
            app_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
            d = np.mean(app_diffs) if app_diffs else None
        if d is None:
            continue

        # IRT prediction: P(tinkering) = sigmoid(theta - d)
        p_tink = 1 / (1 + np.exp(-(t - d)))

        # Relabel if IRT strongly disagrees with raw label
        if y_new[i] == 1 and p_tink < threshold_low:
            # Raw = tinkering, IRT says very likely oob
            y_new[i] = 2
            n_tink_to_oob += 1
        elif y_new[i] == 2 and p_tink > threshold_high:
            # Raw = oob, IRT says very likely tinkering
            y_new[i] = 1
            n_oob_to_tink += 1

    logger.info("IRT relabel: %d tinkering→oob, %d oob→tinkering (thresholds %.2f/%.2f)",
                n_tink_to_oob, n_oob_to_tink, threshold_low, threshold_high)
    return y_new, n_tink_to_oob + n_oob_to_tink


def contributor_aware_relabel(y, report_ids, relabel_ids, contrib_map, theta):
    """Phase 13.2: Graduated relabeling based on annotator strictness.

    Level 1 (hard): no actions + theta > 1.5 → tinkering → oob
    Level 2 (soft label): no actions + theta 0.5..1.5 → y = 1.7 (soft target)
    Level 3 (keep): has actions, or theta < 0.5, or no contributor data
    """
    y_new = y.copy().astype(float)
    n_hard = 0
    n_soft = 0

    for i, rid in enumerate(report_ids):
        if y_new[i] != 1:  # only tinkering
            continue

        is_phase8_relabel = rid in relabel_ids  # no effort markers

        if not is_phase8_relabel:
            continue  # has effort markers → keep as tinkering

        # Check contributor strictness
        t = None
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cid = str(row["contributor_id"])
            t = theta.get(cid)

        if t is not None and t > 1.5:
            # Level 1: very strict annotator, no actions → hard relabel
            y_new[i] = 2
            n_hard += 1
        elif t is not None and t > 0.5:
            # Level 2: moderately strict, no actions → soft label
            # For cross_entropy: 1.7 maps to P(oob)=0.7 after Stage 2 encoding
            y_new[i] = 2  # relabel but mark for soft weighting
            n_soft += 1
        # else: Level 3 — keep as tinkering (no data or lenient annotator)

    logger.info("Contributor-aware relabel: %d hard, %d soft (of %d tinkering)",
                n_hard, n_soft, (y == 1).sum())
    return y_new, n_hard + n_soft


def hybrid_relabel(y, report_ids, relabel_ids, contrib_map, theta, difficulty, db_path):
    """Phase 13.3: Hybrid pipeline combining IRT + actions + Cleanlab fallback.

    Priority:
    1. Has IRT + is Phase 8 candidate → IRT-informed relabeling
    2. Has IRT only → conservative IRT relabeling
    3. Phase 8 candidate, no IRT → original Phase 8 relabeling
    4. Neither → keep original
    """
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    report_info = {}
    for r in conn.execute("SELECT id, app_id, gpu FROM reports").fetchall():
        report_info[r["id"]] = (r["app_id"], r["gpu"])
    conn.close()

    y_new = y.copy()
    confidence = np.ones(len(y), dtype=float) * 0.5  # default confidence
    stats = {"irt_relabel": 0, "irt_confirm": 0, "phase8_relabel": 0, "keep": 0}

    for i, rid in enumerate(report_ids):
        if y_new[i] not in (1, 2):
            confidence[i] = 0.9  # borked labels are reliable
            continue

        is_phase8 = rid in relabel_ids and y_new[i] == 1

        # Try IRT
        has_irt = False
        p_tink = None
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cid = str(row["contributor_id"])
            t = theta.get(cid)
            if t is not None:
                app_id, gpu = report_info.get(rid, (None, None))
                if app_id:
                    gpu_fam = extract_gpu_family(gpu) if gpu else "unknown"
                    d = difficulty.get((app_id, gpu_fam))
                    if d is None:
                        app_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
                        d = np.mean(app_diffs) if app_diffs else None
                    if d is not None:
                        has_irt = True
                        p_tink = 1 / (1 + np.exp(-(t - d)))

        if has_irt and is_phase8:
            # Priority 1: IRT + actions agree on direction
            if p_tink < 0.35:
                y_new[i] = 2  # both say oob
                confidence[i] = 0.9
                stats["irt_relabel"] += 1
            else:
                # IRT says maybe tinkering, but no actions → softer confidence
                y_new[i] = 2  # still relabel (Phase 8)
                confidence[i] = 0.6
                stats["phase8_relabel"] += 1
        elif has_irt:
            # Priority 2: IRT only, conservative thresholds
            if y_new[i] == 1 and p_tink < 0.25:
                y_new[i] = 2
                confidence[i] = 0.7
                stats["irt_relabel"] += 1
            elif y_new[i] == 2 and p_tink > 0.75:
                y_new[i] = 1
                confidence[i] = 0.7
                stats["irt_relabel"] += 1
            else:
                confidence[i] = 0.5 + 0.3 * min(1, abs(p_tink - 0.5) * 4)
                stats["irt_confirm"] += 1
        elif is_phase8:
            # Priority 3: Phase 8 only (no IRT)
            y_new[i] = 2
            confidence[i] = 0.6
            stats["phase8_relabel"] += 1
        else:
            # Priority 4: keep original
            confidence[i] = 0.5
            stats["keep"] += 1

    logger.info("Hybrid relabel: irt=%d, phase8=%d, confirm=%d, keep=%d",
                stats["irt_relabel"], stats["phase8_relabel"],
                stats["irt_confirm"], stats["keep"])

    # Convert confidence to sample weights: w = 0.3 + 0.7 * confidence
    weights = 0.3 + 0.7 * confidence
    return y_new, weights


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 13: Contributor-aware relabeling experiments")
    print("=" * 70)

    # Load raw data (no relabeling, no Cleanlab)
    t0 = time.time()
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids, contrib_map, relabel_ids = \
        load_data_raw(args.db)
    logger.info("Data loaded in %.1fs: train=%d, test=%d", time.time() - t0, len(X_train), len(X_test))

    # Fit IRT
    print("\n[IRT fitting]")
    theta, difficulty = fit_irt(args.db)
    logger.info("IRT: %d contributors, %d items", len(theta), len(difficulty))

    # Add IRT features (used in all experiments since 12.8 proved it works)
    X_train_irt = add_irt_features(X_train, train_rids, contrib_map, theta, difficulty, args.db)
    X_test_irt = add_irt_features(X_test, test_rids, contrib_map, theta, difficulty, args.db)

    results = []

    # ── Baseline: Phase 8 relabel + Cleanlab (current pipeline) ──────
    print("\n" + "=" * 70)
    print("BASELINE: Phase 8 relabel + Cleanlab 3% + IRT features")
    print("=" * 70)
    y_baseline, n_r = apply_phase8_relabel(y_train_raw, train_rids, relabel_ids)
    logger.info("Phase 8: relabeled %d", n_r)

    from protondb_settings.ml.noise import find_noisy_samples
    keep_mask = find_noisy_samples(X_train_irt, y_baseline, frac_remove=0.03,
                                   cache_dir=Path(args.db).parent, force=False)
    X_bl = X_train_irt[keep_mask].reset_index(drop=True)
    y_bl = y_baseline[keep_mask]
    rids_bl = [r for r, k in zip(train_rids, keep_mask) if k]
    logger.info("Cleanlab: removed %d", (~keep_mask).sum())

    r = train_and_eval(X_bl, y_bl, X_test_irt, y_test, label="baseline_phase8+cleanlab+irt")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.1a: IRT relabel only (no Phase 8, no Cleanlab) ────────────
    print("\n" + "=" * 70)
    print("13.1a: IRT relabel only (no Phase 8, no Cleanlab)")
    print("=" * 70)
    y_irt, n = irt_relabel(y_train_raw, train_rids, contrib_map, theta, difficulty, args.db)
    r = train_and_eval(X_train_irt, y_irt, X_test_irt, y_test, label="13.1a_irt_only")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.1b: IRT relabel + Phase 8 (no Cleanlab) ──────────────────
    print("\n" + "=" * 70)
    print("13.1b: Phase 8 + IRT relabel (no Cleanlab)")
    print("=" * 70)
    y_p8, _ = apply_phase8_relabel(y_train_raw, train_rids, relabel_ids)
    y_p8_irt, n = irt_relabel(y_p8, train_rids, contrib_map, theta, difficulty, args.db)
    r = train_and_eval(X_train_irt, y_p8_irt, X_test_irt, y_test, label="13.1b_phase8+irt")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.1c: IRT relabel (aggressive thresholds) ───────────────────
    print("\n" + "=" * 70)
    print("13.1c: IRT relabel aggressive (thresholds 0.4/0.6)")
    print("=" * 70)
    y_irt_agg, n = irt_relabel(y_train_raw, train_rids, contrib_map, theta, difficulty, args.db,
                               threshold_low=0.4, threshold_high=0.6)
    r = train_and_eval(X_train_irt, y_irt_agg, X_test_irt, y_test, label="13.1c_irt_aggressive")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.2: Contributor-aware rule relabel ──────────────────────────
    print("\n" + "=" * 70)
    print("13.2: Contributor-aware rule-based relabel")
    print("=" * 70)
    y_ca, n = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, contrib_map, theta)
    y_ca = y_ca.astype(int)
    r = train_and_eval(X_train_irt, y_ca, X_test_irt, y_test, label="13.2_contributor_aware")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.3a: Hybrid pipeline (no Cleanlab) ─────────────────────────
    print("\n" + "=" * 70)
    print("13.3a: Hybrid pipeline (IRT + actions, confidence weights)")
    print("=" * 70)
    y_hyb, w_hyb = hybrid_relabel(y_train_raw, train_rids, relabel_ids,
                                   contrib_map, theta, difficulty, args.db)
    r = train_and_eval(X_train_irt, y_hyb, X_test_irt, y_test,
                       sample_weight=w_hyb, label="13.3a_hybrid")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.3b: Hybrid + Cleanlab fallback ────────────────────────────
    print("\n" + "=" * 70)
    print("13.3b: Hybrid + Cleanlab fallback (3%)")
    print("=" * 70)
    keep_mask_hyb = find_noisy_samples(X_train_irt, y_hyb, frac_remove=0.03,
                                       cache_dir=None, force=True)
    X_hyb_cl = X_train_irt[keep_mask_hyb].reset_index(drop=True)
    y_hyb_cl = y_hyb[keep_mask_hyb]
    w_hyb_cl = w_hyb[keep_mask_hyb]
    r = train_and_eval(X_hyb_cl, y_hyb_cl, X_test_irt, y_test,
                       sample_weight=w_hyb_cl, label="13.3b_hybrid+cleanlab")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<35s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s}")
    print("-" * 70)
    baseline_f1 = results[0]["f1_macro"]
    for r in results:
        delta = r["f1_macro"] - baseline_f1
        print(f"{r['label']:<35s} {r['f1_macro']:>7.4f} {delta:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f}")


if __name__ == "__main__":
    main()
