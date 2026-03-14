"""Phase 13b: 2PL IRT experiment.

Compares 1PL vs 2PL IRT, tests discrimination-aware relabeling.

Usage:
  python scripts/experiment_13b_2pl_irt.py [--db data/protondb.db]
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
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── IRT models ───────────────────────────────────────────────────────


def fit_irt_1pl(df, c_to_idx, i_to_idx, c_idx, i_idx, responses):
    """Fit 1PL IRT. Returns (theta, d)."""
    n_c, n_i = len(c_to_idx), len(i_to_idx)

    def nll(params):
        theta, d = params[:n_c], params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        p = np.clip(1 / (1 + np.exp(-logit)), 1e-7, 1 - 1e-7)
        ll = responses * np.log(p) + (1 - responses) * np.log(1 - p)
        return -ll.sum() + 0.01 * (np.sum(theta**2) + np.sum(d**2))

    def grad(params):
        theta, d = params[:n_c], params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        r = 1 / (1 + np.exp(-logit)) - responses
        g_t, g_d = np.zeros(n_c), np.zeros(n_i)
        np.add.at(g_t, c_idx, r)
        np.add.at(g_d, i_idx, -r)
        return np.concatenate([g_t + 0.02 * theta, g_d + 0.02 * d])

    x0 = np.zeros(n_c + n_i)
    for c, idx in c_to_idx.items():
        ratio = np.clip(df.loc[df["contributor_id"] == c, "response"].mean(), 0.05, 0.95)
        x0[idx] = np.log(ratio / (1 - ratio))
    for item, idx in i_to_idx.items():
        ratio = np.clip(df.loc[df["item"] == item, "response"].mean(), 0.05, 0.95)
        x0[n_c + idx] = -np.log(ratio / (1 - ratio))

    t0 = time.time()
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-8})
    logger.info("1PL: converged=%s, %.1fs, %d iters, nll=%.1f", res.success, time.time()-t0, res.nit, res.fun)
    return res.x[:n_c], res.x[n_c:]


def fit_irt_2pl(df, c_to_idx, i_to_idx, c_idx, i_idx, responses):
    """Fit 2PL IRT: P(1) = σ(a_i * (θ_j - d_i)). Returns (theta, d, a)."""
    n_c, n_i = len(c_to_idx), len(i_to_idx)

    def nll(params):
        theta = params[:n_c]
        d = params[n_c:n_c+n_i]
        log_a = params[n_c+n_i:]  # log(a) to keep a > 0
        a = np.exp(np.clip(log_a, -3, 3))

        logit = np.clip(a[i_idx] * (theta[c_idx] - d[i_idx]), -20, 20)
        p = np.clip(1 / (1 + np.exp(-logit)), 1e-7, 1 - 1e-7)
        ll = responses * np.log(p) + (1 - responses) * np.log(1 - p)
        # Regularization: theta~N(0,1), d~N(0,1), log_a~N(0, 0.5)
        reg = 0.01 * (np.sum(theta**2) + np.sum(d**2)) + 0.02 * np.sum(log_a**2)
        return -ll.sum() + reg

    def grad(params):
        theta = params[:n_c]
        d = params[n_c:n_c+n_i]
        log_a = params[n_c+n_i:]
        a = np.exp(np.clip(log_a, -3, 3))

        logit = np.clip(a[i_idx] * (theta[c_idx] - d[i_idx]), -20, 20)
        p = 1 / (1 + np.exp(-logit))
        r = p - responses  # residual

        g_theta = np.zeros(n_c)
        g_d = np.zeros(n_i)
        g_loga = np.zeros(n_i)

        # dL/dtheta_j = sum_i a_i * residual
        np.add.at(g_theta, c_idx, a[i_idx] * r)
        # dL/dd_i = sum_j -a_i * residual
        np.add.at(g_d, i_idx, -a[i_idx] * r)
        # dL/dlog_a_i = sum_j (theta_j - d_i) * a_i * residual  (chain rule: d/dlog_a = a * d/da)
        np.add.at(g_loga, i_idx, (theta[c_idx] - d[i_idx]) * a[i_idx] * r)

        g_theta += 0.02 * theta
        g_d += 0.02 * d
        g_loga += 0.04 * log_a
        return np.concatenate([g_theta, g_d, g_loga])

    # Initialize from 1PL solution
    theta_1pl, d_1pl = fit_irt_1pl(df, c_to_idx, i_to_idx, c_idx, i_idx, responses)
    x0 = np.concatenate([theta_1pl, d_1pl, np.zeros(n_i)])  # log(a) = 0 → a = 1

    t0 = time.time()
    res = minimize(nll, x0, jac=grad, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-8})
    elapsed = time.time() - t0
    logger.info("2PL: converged=%s, %.1fs, %d iters, nll=%.1f", res.success, elapsed, res.nit, res.fun)

    theta = res.x[:n_c]
    d = res.x[n_c:n_c+n_i]
    a = np.exp(np.clip(res.x[n_c+n_i:], -3, 3))

    logger.info("  theta: mean=%.2f, std=%.2f", np.mean(theta), np.std(theta))
    logger.info("  d: mean=%.2f, std=%.2f", np.mean(d), np.std(d))
    logger.info("  a: mean=%.2f, std=%.2f, range=[%.2f, %.2f]", np.mean(a), np.std(a), np.min(a), np.max(a))

    return theta, d, a


# ── Data preparation ─────────────────────────────────────────────────


def prepare_irt_data(db_path):
    """Load and prepare data for IRT fitting."""
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

    for _ in range(3):
        item_counts = df["item"].value_counts()
        df = df[df["item"].isin(item_counts[item_counts >= 2].index)]
        contrib_counts = df["contributor_id"].value_counts()
        df = df[df["contributor_id"].isin(contrib_counts[contrib_counts >= 2].index)]

    contributors = df["contributor_id"].unique()
    items = df["item"].unique()
    c_to_idx = {c: i for i, c in enumerate(contributors)}
    i_to_idx = {it: i for i, it in enumerate(items)}
    c_idx = df["contributor_id"].map(c_to_idx).values
    i_idx = df["item"].map(i_to_idx).values
    responses = df["response"].values.astype(float)

    logger.info("IRT data: %d responses, %d items, %d contributors",
                len(df), len(items), len(contributors))

    return df, c_to_idx, i_to_idx, c_idx, i_idx, responses, contributors, items


def build_param_dicts(c_to_idx, i_to_idx, theta, d, a=None):
    """Convert arrays to dicts."""
    theta_dict = {c: theta[idx] for c, idx in c_to_idx.items()}
    diff_dict = {}
    disc_dict = {}
    for item_key, idx in i_to_idx.items():
        parts = item_key.rsplit("_", 1)
        key = (int(parts[0]), parts[1])
        diff_dict[key] = d[idx]
        if a is not None:
            disc_dict[key] = a[idx]
    return theta_dict, diff_dict, disc_dict


# ── Feature addition and relabeling ──────────────────────────────────


def add_irt_features(X, report_ids, contrib_map, theta_dict, diff_dict, disc_dict, db_path):
    """Add IRT features (difficulty + optional discrimination)."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    ri = {r["id"]: (r["app_id"], r["gpu"]) for r in conn.execute("SELECT id, app_id, gpu FROM reports")}
    conn.close()

    X = X.copy()
    irt_d, irt_t, irt_a = [], [], []

    for rid in report_ids:
        app_id, gpu = ri.get(rid, (None, None))
        gf = extract_gpu_family(gpu) if gpu else "unknown"

        d = diff_dict.get((app_id, gf))
        if d is None and app_id:
            vals = [v for (a, g), v in diff_dict.items() if a == app_id]
            d = np.mean(vals) if vals else None
        irt_d.append(d if d is not None else np.nan)

        a = disc_dict.get((app_id, gf)) if disc_dict else None
        if a is None and app_id and disc_dict:
            vals = [v for (ai, g), v in disc_dict.items() if ai == app_id]
            a = np.mean(vals) if vals else None
        irt_a.append(a if a is not None else np.nan)

        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            irt_t.append(theta_dict.get(str(row["contributor_id"]), np.nan))
        else:
            irt_t.append(np.nan)

    X["irt_game_difficulty"] = pd.Series(irt_d).fillna(pd.Series(irt_d).median()).values
    X["irt_contributor_strictness"] = pd.Series(irt_t).fillna(0).values
    if disc_dict:
        X["irt_game_discrimination"] = pd.Series(irt_a).fillna(pd.Series(irt_a).median()).values
    return X


def contributor_aware_relabel(y, report_ids, relabel_ids, contrib_map, theta_dict):
    """Phase 13.2 relabeling (winner from previous experiment)."""
    y_new = y.copy()
    n = 0
    for i, rid in enumerate(report_ids):
        if y_new[i] != 1:
            continue
        if rid not in relabel_ids:
            continue
        t = None
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            t = theta_dict.get(str(row["contributor_id"]))
        if t is not None and t > 0.5:
            y_new[i] = 2
            n += 1
    return y_new, n


def discrimination_aware_relabel(y, report_ids, relabel_ids, contrib_map,
                                  theta_dict, diff_dict, disc_dict, db_path,
                                  min_disc=0.5):
    """Phase 13.2 + discrimination: only relabel high-discrimination items."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.encoding import extract_gpu_family

    conn = get_connection(db_path)
    ri = {r["id"]: (r["app_id"], r["gpu"]) for r in conn.execute("SELECT id, app_id, gpu FROM reports")}
    conn.close()

    y_new = y.copy()
    n_relabel = 0
    n_skipped_low_disc = 0

    for i, rid in enumerate(report_ids):
        if y_new[i] != 1:
            continue
        if rid not in relabel_ids:
            continue

        t = None
        if rid in contrib_map.index:
            row = contrib_map.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            t = theta_dict.get(str(row["contributor_id"]))

        if t is None or t <= 0.5:
            continue

        # Check discrimination for this item
        app_id, gpu = ri.get(rid, (None, None))
        gf = extract_gpu_family(gpu) if gpu else "unknown"
        a = disc_dict.get((app_id, gf))
        if a is None and app_id:
            vals = [v for (ai, g), v in disc_dict.items() if ai == app_id]
            a = np.mean(vals) if vals else None

        if a is not None and a < min_disc:
            n_skipped_low_disc += 1
            continue  # low discrimination — genuine ambiguity, don't relabel

        y_new[i] = 2
        n_relabel += 1

    logger.info("Disc-aware relabel: %d relabeled, %d skipped (disc < %.2f)",
                n_relabel, n_skipped_low_disc, min_disc)
    return y_new, n_relabel


# ── Train + eval ─────────────────────────────────────────────────────


def train_and_eval(X_train, y_train, X_test, y_test, label=""):
    from protondb_settings.ml.models.cascade import train_stage1, CascadeClassifier, STAGE2_DROP_FEATURES
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    s1 = train_stage1(X_train, y_train, X_test, y_test)

    mask_tr, mask_te = y_train > 0, y_test > 0
    X2_tr = X_train[mask_tr].reset_index(drop=True)
    y2_tr = (y_train[mask_tr] - 1).astype(float)
    X2_te = X_test[mask_te].reset_index(drop=True)
    y2_te = (y_test[mask_te] - 1).astype(float)

    drops = [c for c in STAGE2_DROP_FEATURES if c in X2_tr.columns]
    if drops:
        X2_tr = X2_tr.drop(columns=drops)
        X2_te = X2_te.drop(columns=drops)

    cats = [c for c in CATEGORICAL_FEATURES if c in X2_tr.columns]
    for c in cats:
        X2_tr[c] = X2_tr[c].astype("category")
        X2_te[c] = X2_te[c].astype("category")

    y_smooth = y2_tr * 0.85 + (1 - y2_tr) * 0.15
    ds_tr = lgb.Dataset(X2_tr, label=y_smooth, categorical_feature=cats)
    ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)

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
    return {"label": label, "f1_macro": f1, "borked_f1": per[0], "tinkering_f1": per[1], "works_oob_f1": per[2]}


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 13b: 2PL IRT experiment")
    print("=" * 70)

    # Load data
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(X, y, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)

    contrib_df = pd.read_sql_query(
        "SELECT report_id, contributor_id, report_tally, playtime, playtime_linux FROM report_contributors", conn)
    contrib_map = contrib_df.set_index("report_id")
    conn.close()

    # Fit IRT models
    df_irt, c_to_idx, i_to_idx, c_idx, i_idx, responses, contributors, items = prepare_irt_data(args.db)

    print("\n[1PL IRT]")
    theta_1pl, d_1pl = fit_irt_1pl(df_irt, c_to_idx, i_to_idx, c_idx, i_idx, responses)
    theta_dict_1pl, diff_dict_1pl, _ = build_param_dicts(c_to_idx, i_to_idx, theta_1pl, d_1pl)

    print("\n[2PL IRT]")
    theta_2pl, d_2pl, a_2pl = fit_irt_2pl(df_irt, c_to_idx, i_to_idx, c_idx, i_idx, responses)
    theta_dict_2pl, diff_dict_2pl, disc_dict_2pl = build_param_dicts(c_to_idx, i_to_idx, theta_2pl, d_2pl, a_2pl)

    results = []

    # ── 1PL baseline (Phase 13.2 winner) ─────────────────────────────
    print("\n" + "=" * 70)
    print("1PL: features + contributor-aware relabel (Phase 13.2 winner)")
    print("=" * 70)
    X_tr_1pl = add_irt_features(X_train, train_rids, contrib_map, theta_dict_1pl, diff_dict_1pl, {}, args.db)
    X_te_1pl = add_irt_features(X_test, test_rids, contrib_map, theta_dict_1pl, diff_dict_1pl, {}, args.db)
    y_1pl, n = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, contrib_map, theta_dict_1pl)
    logger.info("1PL relabel: %d", n)
    r = train_and_eval(X_tr_1pl, y_1pl, X_te_1pl, y_test, label="1PL_baseline")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | borked={r['borked_f1']:.3f} | tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 2PL: features only (d + a) ──────────────────────────────────
    print("\n" + "=" * 70)
    print("2PL: features (difficulty + discrimination) + 1PL relabel")
    print("=" * 70)
    X_tr_2pl = add_irt_features(X_train, train_rids, contrib_map, theta_dict_2pl, diff_dict_2pl, disc_dict_2pl, args.db)
    X_te_2pl = add_irt_features(X_test, test_rids, contrib_map, theta_dict_2pl, diff_dict_2pl, disc_dict_2pl, args.db)
    # Use 2PL theta for relabeling
    y_2pl, n = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, contrib_map, theta_dict_2pl)
    logger.info("2PL relabel: %d", n)
    r = train_and_eval(X_tr_2pl, y_2pl, X_te_2pl, y_test, label="2PL_features+relabel")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 2PL: discrimination-aware relabeling ─────────────────────────
    for min_disc in [0.3, 0.5, 0.8, 1.0]:
        print("\n" + "=" * 70)
        print(f"2PL: discrimination-aware relabel (min_disc={min_disc})")
        print("=" * 70)
        y_disc, n = discrimination_aware_relabel(
            y_train_raw, train_rids, relabel_ids, contrib_map,
            theta_dict_2pl, diff_dict_2pl, disc_dict_2pl, args.db, min_disc=min_disc)
        r = train_and_eval(X_tr_2pl, y_disc, X_te_2pl, y_test,
                           label=f"2PL_disc_aware_{min_disc}")
        results.append(r)
        delta = r["f1_macro"] - results[0]["f1_macro"]
        print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30s} {'F1':>7s} {'ΔF1':>7s} {'borked':>7s} {'tink':>7s} {'oob':>7s}")
    print("-" * 70)
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<30s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f}")


if __name__ == "__main__":
    main()
