"""Phase 12 experiments: contributor-aware model improvements.

Experiments:
  12.1  — Contributor features (report_tally, playtime, linux_ratio)
  12.3  — Sample weighting by contributor reliability
  12.7  — Binary task (borked vs works) + rule-based tinkering/oob split
  12.8  — IRT (Item Response Theory) for label denoising

Usage:
  python scripts/experiment_12_contributor.py [--db data/protondb.db]
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
from sklearn.metrics import f1_score, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def load_data(db_path: str) -> tuple:
    """Load feature matrix + contributor data using existing pipeline."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import apply_relabeling, get_relabel_ids
    from protondb_settings.ml.noise import find_noisy_samples

    conn = get_connection(db_path)

    # Load cached embeddings
    emb_path = Path(db_path).parent / "embeddings.npz"
    emb_data = load_embeddings(emb_path)

    # Build features
    X, y, timestamps, report_ids, label_maps = _build_feature_matrix(conn, emb_data)

    # Time split
    X_train, X_test, y_train, y_test, train_rids, test_rids = _time_based_split(
        X, y, timestamps, 0.2, report_ids=report_ids
    )

    # Relabeling (Phase 8)
    relabel_ids = get_relabel_ids(conn)
    y_train, n_relabeled = apply_relabeling(y_train, train_rids, relabel_ids)
    logger.info("Relabeled %d training samples", n_relabeled)

    # Cleanlab noise removal
    keep_mask = find_noisy_samples(X_train, y_train, frac_remove=0.03,
                                   cache_dir=Path(db_path).parent, force=False)
    n_removed = (~keep_mask).sum()
    X_train = X_train[keep_mask].reset_index(drop=True)
    y_train = y_train[keep_mask]
    train_rids = [rid for rid, keep in zip(train_rids, keep_mask) if keep]
    logger.info("Cleanlab: removed %d noisy samples", n_removed)

    # Load contributor data
    contrib_df = pd.read_sql_query(
        "SELECT report_id, contributor_id, report_tally, playtime, playtime_linux "
        "FROM report_contributors",
        conn,
    )
    contrib_map = contrib_df.set_index("report_id")

    conn.close()
    return X_train, X_test, y_train, y_test, train_rids, test_rids, contrib_map, conn, db_path


def train_cascade_and_eval(
    X_train, y_train, X_test, y_test,
    sample_weight_train=None,
    label: str = "baseline",
):
    """Train cascade (Stage 1 + Stage 2) and evaluate. Returns results dict."""
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    # Stage 1: borked vs works
    s1_model = train_stage1(X_train, y_train, X_test, y_test)

    # Stage 2: tinkering vs works_oob (with sample weights if provided)
    s2_model, s2_dropped = train_stage2_weighted(
        X_train, y_train, X_test, y_test,
        sample_weight=sample_weight_train,
    )

    # Cascade predict
    cascade = CascadeClassifier(s1_model, s2_model, s2_dropped)
    y_pred = cascade.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    per_class = f1_score(y_test, y_pred, average=None)

    result = {
        "label": label,
        "f1_macro": f1,
        "accuracy": acc,
        "borked_f1": per_class[0],
        "tinkering_f1": per_class[1],
        "works_oob_f1": per_class[2],
    }
    return result


def train_stage2_weighted(
    X_train, y_train, X_test, y_test,
    sample_weight=None,
    label_smoothing=0.15,
):
    """Train Stage 2 with optional sample weights."""
    from protondb_settings.ml.models.cascade import STAGE2_DROP_FEATURES
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    drop_features = list(STAGE2_DROP_FEATURES)

    # Filter to non-borked
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_train_s2 = X_train[train_mask].reset_index(drop=True)
    y_train_s2 = (y_train[train_mask] - 1).astype(float)

    X_test_s2 = X_test[test_mask].reset_index(drop=True)
    y_test_s2 = (y_test[test_mask] - 1).astype(float)

    # Drop temporal bias features
    existing_drops = [c for c in drop_features if c in X_train_s2.columns]
    if existing_drops:
        X_train_s2 = X_train_s2.drop(columns=existing_drops)
        X_test_s2 = X_test_s2.drop(columns=existing_drops)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train_s2.columns]
    for col in cat_cols:
        X_train_s2[col] = X_train_s2[col].astype("category")
        X_test_s2[col] = X_test_s2[col].astype("category")

    # Label smoothing
    y_smooth = y_train_s2 * (1 - label_smoothing) + (1 - y_train_s2) * label_smoothing

    # Sample weights for non-borked subset
    w = None
    if sample_weight is not None:
        w = sample_weight[train_mask]

    ds_train = lgb.Dataset(X_train_s2, label=y_smooth, weight=w, categorical_feature=cat_cols)
    ds_test = lgb.Dataset(X_test_s2, label=y_test_s2, categorical_feature=cat_cols)

    params = {
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

    model = lgb.train(
        params, ds_train, num_boost_round=3000,
        valid_sets=[ds_test],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    return model, existing_drops


# ── Phase 12.1: Contributor features ─────────────────────────────────


def add_contributor_features(X: pd.DataFrame, report_ids: list[str], contrib_map: pd.DataFrame) -> pd.DataFrame:
    """Add contributor features to feature matrix."""
    X = X.copy()

    tally = []
    playtime = []
    linux_ratio = []

    for rid in report_ids:
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            # Handle duplicate report_ids (take first)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            tally.append(row["report_tally"] or 0)
            pt = row["playtime"] or 0
            ptl = row["playtime_linux"] or 0
            playtime.append(pt)
            linux_ratio.append(ptl / pt if pt > 0 else 0)
        else:
            tally.append(np.nan)
            playtime.append(np.nan)
            linux_ratio.append(np.nan)

    X["contributor_tally"] = tally
    X["contributor_playtime"] = playtime
    X["contributor_linux_ratio"] = linux_ratio
    # Log-transform tally and playtime for better distribution
    X["contributor_log_tally"] = np.log1p(X["contributor_tally"])
    X["contributor_log_playtime"] = np.log1p(X["contributor_playtime"])

    # Fill NaN with median (reports without contributor data)
    for col in ["contributor_tally", "contributor_playtime", "contributor_linux_ratio",
                "contributor_log_tally", "contributor_log_playtime"]:
        X[col] = X[col].fillna(X[col].median())

    coverage = sum(1 for rid in report_ids if rid in contrib_map.index) / len(report_ids) * 100
    logger.info("Contributor coverage: %.1f%% (%d/%d)", coverage,
                sum(1 for rid in report_ids if rid in contrib_map.index), len(report_ids))

    return X


# ── Phase 12.3: Sample weighting ─────────────────────────────────────


def compute_sample_weights(report_ids: list[str], contrib_map: pd.DataFrame, method: str = "log") -> np.ndarray:
    """Compute sample weights based on contributor reliability."""
    weights = np.ones(len(report_ids), dtype=np.float64)

    for i, rid in enumerate(report_ids):
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            tally = row["report_tally"] or 1

            if method == "log":
                # log-based: smooth, bounded [0.3, 1.0]
                weights[i] = min(1.0, max(0.3, np.log1p(tally) / np.log1p(100)))
            elif method == "bucket":
                # bucket-based
                if tally >= 20:
                    weights[i] = 1.0
                elif tally >= 5:
                    weights[i] = 0.8
                else:
                    weights[i] = 0.5
        else:
            weights[i] = 0.7  # unknown contributor

    return weights


# ── Phase 12.8: IRT ──────────────────────────────────────────────────


def fit_irt(db_path: str, contrib_map: pd.DataFrame) -> tuple[dict, dict]:
    """Fit 1PL IRT model on tinkering/oob boundary.

    Returns:
        theta: {contributor_id: strictness}
        difficulty: {(app_id, gpu_family): difficulty}
    """
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)

    # Get reports with contributor data AND tinkering/oob verdict
    df = pd.read_sql_query("""
        SELECT r.id, r.app_id, r.gpu, r.verdict, r.verdict_oob,
               rc.contributor_id
        FROM reports r
        JOIN report_contributors rc ON r.id = rc.report_id
        WHERE r.verdict = 'yes'
    """, conn)
    conn.close()

    # Compute label: 1 = tinkering (strict), 0 = works_oob (lenient)
    # verdict_oob = "yes" → works_oob → 0
    # verdict_oob = "no" or NULL → tinkering → 1
    df["response"] = (df["verdict_oob"] != "yes").astype(int)

    # Extract GPU family for item grouping
    df["gpu_family"] = df["gpu"].apply(lambda g: extract_gpu_family(g) if g else "unknown")

    # Item = (app_id, gpu_family)
    df["item"] = df["app_id"].astype(str) + "_" + df["gpu_family"]

    # Filter: need items with 2+ annotators AND annotators with 2+ items
    for _ in range(3):  # iterative filtering
        item_counts = df["item"].value_counts()
        df = df[df["item"].isin(item_counts[item_counts >= 2].index)]
        contrib_counts = df["contributor_id"].value_counts()
        df = df[df["contributor_id"].isin(contrib_counts[contrib_counts >= 2].index)]

    logger.info("IRT data: %d responses, %d items, %d contributors",
                len(df), df["item"].nunique(), df["contributor_id"].nunique())

    if len(df) < 100:
        logger.warning("Not enough data for IRT (need 100+, got %d)", len(df))
        return {}, {}

    # Encode to indices
    contributors = df["contributor_id"].unique()
    items = df["item"].unique()
    c_to_idx = {c: i for i, c in enumerate(contributors)}
    i_to_idx = {it: i for i, it in enumerate(items)}

    c_idx = df["contributor_id"].map(c_to_idx).values
    i_idx = df["item"].map(i_to_idx).values
    responses = df["response"].values.astype(float)

    n_c = len(contributors)
    n_i = len(items)

    # 1PL IRT: P(response=1) = sigmoid(theta[c] - d[i])
    # Optimize via L-BFGS-B
    def neg_log_likelihood(params):
        theta = params[:n_c]
        d = params[n_c:]

        logit = theta[c_idx] - d[i_idx]
        # Clip for numerical stability
        logit = np.clip(logit, -20, 20)
        p = 1 / (1 + np.exp(-logit))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        ll = responses * np.log(p) + (1 - responses) * np.log(1 - p)
        # L2 regularization (prior N(0,1))
        reg = 0.01 * (np.sum(theta ** 2) + np.sum(d ** 2))
        return -ll.sum() + reg

    # Initialize theta from personal tinkering ratio (clipped logit)
    x0 = np.zeros(n_c + n_i)
    for c, idx in c_to_idx.items():
        mask = df["contributor_id"] == c
        ratio = df.loc[mask, "response"].mean()
        ratio = np.clip(ratio, 0.05, 0.95)  # avoid log(0) and log(inf)
        x0[idx] = np.log(ratio / (1 - ratio))
    # Initialize difficulty from per-item tinkering ratio
    for item, idx in i_to_idx.items():
        mask = df["item"] == item
        ratio = df.loc[mask, "response"].mean()
        ratio = np.clip(ratio, 0.05, 0.95)
        # d = -logit(ratio) because P(1) = σ(θ - d), higher ratio → lower d
        x0[n_c + idx] = -np.log(ratio / (1 - ratio))

    # Analytical gradient for faster convergence
    def grad_nll(params):
        theta = params[:n_c]
        d = params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        p = 1 / (1 + np.exp(-logit))
        residual = p - responses  # dL/dlogit

        g_theta = np.zeros(n_c)
        g_d = np.zeros(n_i)
        np.add.at(g_theta, c_idx, residual)
        np.add.at(g_d, i_idx, -residual)
        # Regularization gradient
        g_theta += 0.02 * theta
        g_d += 0.02 * d
        return np.concatenate([g_theta, g_d])

    logger.info("Fitting IRT (1PL, %d params)...", n_c + n_i)
    t0 = time.time()
    result = minimize(neg_log_likelihood, x0, jac=grad_nll, method="L-BFGS-B",
                     options={"maxiter": 1000, "ftol": 1e-8})
    elapsed = time.time() - t0
    logger.info("IRT converged: %s, %.1fs, %d iterations, nll=%.1f",
                result.success, elapsed, result.nit, result.fun)

    theta_vals = result.x[:n_c]
    d_vals = result.x[n_c:]

    theta = {c: theta_vals[idx] for c, idx in c_to_idx.items()}
    difficulty = {}
    for item_key, idx in i_to_idx.items():
        parts = item_key.rsplit("_", 1)
        app_id = int(parts[0])
        gpu_fam = parts[1]
        difficulty[(app_id, gpu_fam)] = d_vals[idx]

    # Stats
    logger.info("IRT theta: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
                np.mean(theta_vals), np.std(theta_vals), np.min(theta_vals), np.max(theta_vals))
    logger.info("IRT difficulty: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
                np.mean(d_vals), np.std(d_vals), np.min(d_vals), np.max(d_vals))

    return theta, difficulty


def add_irt_features(X: pd.DataFrame, report_ids: list[str],
                     contrib_map: pd.DataFrame, theta: dict, difficulty: dict,
                     db_path: str) -> pd.DataFrame:
    """Add IRT-derived features to feature matrix."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    # Build report_id → (app_id, gpu) lookup
    report_info = {}
    rows = conn.execute("SELECT id, app_id, gpu FROM reports").fetchall()
    for r in rows:
        report_info[r["id"]] = (r["app_id"], r["gpu"])
    conn.close()

    X = X.copy()
    irt_difficulty = []
    irt_theta = []

    for rid in report_ids:
        app_id, gpu = report_info.get(rid, (None, None))
        gpu_fam = extract_gpu_family(gpu) if gpu else "unknown"

        # Game difficulty
        d = difficulty.get((app_id, gpu_fam))
        if d is None and app_id:
            # Fallback: average difficulty across all gpu families for this app
            app_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
            d = np.mean(app_diffs) if app_diffs else None
        irt_difficulty.append(d if d is not None else np.nan)

        # Contributor strictness
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cid = str(row["contributor_id"])
            irt_theta.append(theta.get(cid, np.nan))
        else:
            irt_theta.append(np.nan)

    X["irt_game_difficulty"] = irt_difficulty
    X["irt_contributor_strictness"] = irt_theta

    # Fill NaN
    X["irt_game_difficulty"] = X["irt_game_difficulty"].fillna(X["irt_game_difficulty"].median())
    X["irt_contributor_strictness"] = X["irt_contributor_strictness"].fillna(0)  # neutral

    coverage_d = sum(1 for d in irt_difficulty if not np.isnan(d)) / len(irt_difficulty) * 100
    coverage_t = sum(1 for t in irt_theta if not np.isnan(t)) / len(irt_theta) * 100
    logger.info("IRT coverage: difficulty=%.1f%%, theta=%.1f%%", coverage_d, coverage_t)

    return X


def irt_sample_weights(report_ids: list[str], contrib_map: pd.DataFrame, theta: dict) -> np.ndarray:
    """Sample weights based on IRT contributor strictness: downweight extreme annotators."""
    weights = np.ones(len(report_ids), dtype=np.float64)
    for i, rid in enumerate(report_ids):
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cid = str(row["contributor_id"])
            t = theta.get(cid)
            if t is not None:
                # Downweight extreme annotators: w = 1 / (1 + |theta|)
                weights[i] = 1.0 / (1.0 + abs(t) * 0.3)
        else:
            weights[i] = 0.7
    return weights


# ── Phase 12.7: Binary task ──────────────────────────────────────────


def eval_binary(X_train, y_train, X_test, y_test, sample_weight=None, label="binary"):
    """Train binary borked(0) vs works(1) and evaluate."""
    from protondb_settings.ml.models.cascade import train_stage1
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    y_train_bin = (y_train > 0).astype(int)
    y_test_bin = (y_test > 0).astype(int)

    s1_model = train_stage1(X_train, y_train, X_test, y_test)
    y_pred_bin = s1_model.predict(X_test)

    f1 = f1_score(y_test_bin, y_pred_bin, average="macro")
    acc = accuracy_score(y_test_bin, y_pred_bin)
    per_class = f1_score(y_test_bin, y_pred_bin, average=None)

    borked_r = (y_pred_bin[y_test_bin == 0] == 0).mean()
    borked_p = (y_test_bin[y_pred_bin == 0] == 0).mean() if (y_pred_bin == 0).any() else 0

    logger.info("[%s] Binary F1=%.4f, acc=%.4f, borked_r=%.3f, borked_p=%.3f",
                label, f1, acc, borked_r, borked_p)

    return {"label": label, "binary_f1": f1, "accuracy": acc,
            "borked_f1": per_class[0], "works_f1": per_class[1]}


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 12: Contributor-aware experiments")
    print("=" * 70)

    # Load data
    t0 = time.time()
    X_train, X_test, y_train, y_test, train_rids, test_rids, contrib_map, conn, db_path = load_data(args.db)
    logger.info("Data loaded in %.1fs: train=%d, test=%d", time.time() - t0, len(X_train), len(X_test))

    results = []

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE (current cascade)")
    print("=" * 70)
    r = train_cascade_and_eval(X_train, y_train, X_test, y_test, label="baseline")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | borked={r['borked_f1']:.3f} | "
          f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 12.1: Contributor features ───────────────────────────────────
    print("\n" + "=" * 70)
    print("12.1: Contributor features (tally, playtime, linux_ratio)")
    print("=" * 70)
    X_train_cf = add_contributor_features(X_train, train_rids, contrib_map)
    X_test_cf = add_contributor_features(X_test, test_rids, contrib_map)
    r = train_cascade_and_eval(X_train_cf, y_train, X_test_cf, y_test, label="12.1_contrib_features")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 12.3: Sample weighting (log) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("12.3a: Sample weighting (log-based)")
    print("=" * 70)
    w_log = compute_sample_weights(train_rids, contrib_map, method="log")
    r = train_cascade_and_eval(X_train, y_train, X_test, y_test,
                               sample_weight_train=w_log, label="12.3a_weight_log")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 12.3: Sample weighting (bucket) ──────────────────────────────
    print("\n" + "=" * 70)
    print("12.3b: Sample weighting (bucket-based)")
    print("=" * 70)
    w_bucket = compute_sample_weights(train_rids, contrib_map, method="bucket")
    r = train_cascade_and_eval(X_train, y_train, X_test, y_test,
                               sample_weight_train=w_bucket, label="12.3b_weight_bucket")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 12.1 + 12.3 combined ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("12.1 + 12.3: Features + weighting combined")
    print("=" * 70)
    r = train_cascade_and_eval(X_train_cf, y_train, X_test_cf, y_test,
                               sample_weight_train=w_log, label="12.1+12.3_combined")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 12.8: IRT ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("12.8: IRT (Item Response Theory)")
    print("=" * 70)
    theta, difficulty = fit_irt(args.db, contrib_map)

    if theta and difficulty:
        # 12.8a: IRT features only
        X_train_irt = add_irt_features(X_train, train_rids, contrib_map, theta, difficulty, args.db)
        X_test_irt = add_irt_features(X_test, test_rids, contrib_map, theta, difficulty, args.db)
        r = train_cascade_and_eval(X_train_irt, y_train, X_test_irt, y_test,
                                   label="12.8a_irt_features")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
              f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

        # 12.8b: IRT weights
        w_irt = irt_sample_weights(train_rids, contrib_map, theta)
        r = train_cascade_and_eval(X_train, y_train, X_test, y_test,
                                   sample_weight_train=w_irt, label="12.8b_irt_weights")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
              f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

        # 12.8c: IRT features + IRT weights
        r = train_cascade_and_eval(X_train_irt, y_train, X_test_irt, y_test,
                                   sample_weight_train=w_irt, label="12.8c_irt_full")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
              f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

        # 12.1 + 12.3 + 12.8 all combined
        X_train_all = add_contributor_features(X_train_irt, train_rids, contrib_map)
        X_test_all = add_contributor_features(X_test_irt, test_rids, contrib_map)
        r = train_cascade_and_eval(X_train_all, y_train, X_test_all, y_test,
                                   sample_weight_train=w_irt, label="12.1+12.3+12.8_all")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
              f"tinkering={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")
    else:
        print("  SKIPPED: Not enough IRT data")

    # ── 12.7: Binary task ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("12.7: Binary task (borked vs works)")
    print("=" * 70)
    eval_binary(X_train, y_train, X_test, y_test, label="12.7_binary_baseline")

    if theta:
        eval_binary(X_train_irt, y_train, X_test_irt, y_test, label="12.7_binary+irt")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY (3-class cascade)")
    print("=" * 70)
    print(f"{'Experiment':<30s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s}")
    print("-" * 70)
    baseline_f1 = results[0]["f1_macro"]
    for r in results:
        delta = r["f1_macro"] - baseline_f1
        print(f"{r['label']:<30s} {r['f1_macro']:>7.4f} {delta:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f}")


if __name__ == "__main__":
    main()
