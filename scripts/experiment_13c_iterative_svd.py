"""Phase 13.4-13.5: Iterative IRT refinement + Annotator SVD embeddings.

Experiments:
  13.4  — Iterative: IRT → relabel → retrain → re-fit IRT (2-3 rounds)
  13.5  — Annotator SVD embeddings (contributor×game matrix)
  Combined — All together

Usage:
  python scripts/experiment_13c_iterative_svd.py [--db data/protondb.db]
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── IRT fitting ──────────────────────────────────────────────────────


def fit_irt_from_df(df):
    """Fit 1PL IRT from prepared DataFrame with response/item/contributor_id columns."""
    for _ in range(3):
        ic = df["item"].value_counts()
        df = df[df["item"].isin(ic[ic >= 2].index)]
        cc = df["contributor_id"].value_counts()
        df = df[df["contributor_id"].isin(cc[cc >= 2].index)]

    if len(df) < 100:
        return {}, {}, df

    contributors = df["contributor_id"].unique()
    items = df["item"].unique()
    c_to_idx = {c: i for i, c in enumerate(contributors)}
    i_to_idx = {it: i for i, it in enumerate(items)}
    c_idx = df["contributor_id"].map(c_to_idx).values
    i_idx = df["item"].map(i_to_idx).values
    responses = df["response"].values.astype(float)
    n_c, n_i = len(contributors), len(items)

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

    result = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                     options={"maxiter": 1000, "ftol": 1e-8})

    theta = {str(c): float(result.x[idx]) for c, idx in c_to_idx.items()}
    difficulty = {}
    for item_key, idx in i_to_idx.items():
        parts = item_key.rsplit("_", 1)
        difficulty[(int(parts[0]), parts[1])] = float(result.x[n_c + idx])

    return theta, difficulty, df


def prepare_irt_df(db_path):
    """Prepare IRT DataFrame from DB."""
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
    return df


# ── Cached lookups ───────────────────────────────────────────────────


_report_info_cache = None
_contrib_lookup_cache = None


def get_report_info(db_path):
    global _report_info_cache
    if _report_info_cache is None:
        from protondb_settings.db.connection import get_connection
        conn = get_connection(db_path)
        _report_info_cache = {}
        for r in conn.execute("SELECT id, app_id, gpu FROM reports").fetchall():
            _report_info_cache[r["id"]] = (r["app_id"], r["gpu"])
        conn.close()
    return _report_info_cache


def get_contrib_lookup(db_path):
    global _contrib_lookup_cache
    if _contrib_lookup_cache is None:
        from protondb_settings.db.connection import get_connection
        conn = get_connection(db_path)
        _contrib_lookup_cache = {}
        for r in conn.execute("SELECT report_id, contributor_id FROM report_contributors").fetchall():
            _contrib_lookup_cache[r["report_id"]] = str(r["contributor_id"])
        conn.close()
    return _contrib_lookup_cache


# ── Feature helpers ──────────────────────────────────────────────────


def add_irt_features(X, report_ids, theta, difficulty, db_path):
    from protondb_settings.ml.features.encoding import extract_gpu_family
    ri = get_report_info(db_path)
    cl = get_contrib_lookup(db_path)

    X = X.copy()
    irt_d, irt_t = [], []
    for rid in report_ids:
        app_id, gpu = ri.get(rid, (None, None))
        gf = extract_gpu_family(gpu) if gpu else "unknown"
        d = difficulty.get((app_id, gf))
        if d is None and app_id:
            vals = [v for (a, g), v in difficulty.items() if a == app_id]
            d = np.mean(vals) if vals else None
        irt_d.append(d if d is not None else np.nan)
        cid = cl.get(rid)
        irt_t.append(theta.get(cid, np.nan) if cid else np.nan)

    ds = pd.Series(irt_d)
    X["irt_game_difficulty"] = ds.fillna(ds.median()).values
    X["irt_contributor_strictness"] = pd.Series(irt_t).fillna(0).values
    return X


def contributor_aware_relabel(y, report_ids, relabel_ids, theta, db_path):
    cl = get_contrib_lookup(db_path)
    y_new = y.copy()
    n = 0
    for i, rid in enumerate(report_ids):
        if y_new[i] != 1 or rid not in relabel_ids:
            continue
        cid = cl.get(rid)
        t = theta.get(cid) if cid else None
        if t is not None and t > 0.5:
            y_new[i] = 2
            n += 1
    return y_new, n


# ── Phase 13.5: Annotator SVD embeddings ─────────────────────────────


def build_annotator_embeddings(db_path, n_components=8, min_reports_per_contributor=3):
    """Build SVD embeddings from contributor×game verdict matrix.

    Returns:
        contributor_emb: {contributor_id: np.array(n_components)}
        game_emb: {app_id: np.array(n_components)}
    """
    from protondb_settings.db.connection import get_connection

    conn = get_connection(db_path)
    df = pd.read_sql_query("""
        SELECT r.app_id, rc.contributor_id,
               CASE WHEN r.verdict_oob = 'yes' THEN 0.0
                    WHEN r.verdict = 'no' THEN -1.0
                    ELSE 1.0 END as score
        FROM reports r
        JOIN report_contributors rc ON r.id = rc.report_id
        WHERE r.verdict IS NOT NULL
    """, conn)
    conn.close()

    # Filter contributors with enough reports
    cc = df["contributor_id"].value_counts()
    df = df[df["contributor_id"].isin(cc[cc >= min_reports_per_contributor].index)]

    contributors = df["contributor_id"].unique()
    games = df["app_id"].unique()
    c_to_idx = {c: i for i, c in enumerate(contributors)}
    g_to_idx = {g: i for i, g in enumerate(games)}

    logger.info("Annotator SVD: %d contributors × %d games, %d entries",
                len(contributors), len(games), len(df))

    # Build sparse matrix
    rows = df["contributor_id"].map(c_to_idx).values
    cols = df["app_id"].map(g_to_idx).values
    vals = df["score"].values

    mat = csr_matrix((vals, (rows, cols)), shape=(len(contributors), len(games)))

    # SVD
    k = min(n_components, min(mat.shape) - 1)
    U, S, Vt = svds(mat.astype(float), k=k)

    # Sort by singular value (descending)
    order = np.argsort(-S)
    U = U[:, order]
    S = S[order]
    Vt = Vt[order, :]

    logger.info("Annotator SVD: %d components, singular values: %s",
                k, ", ".join(f"{s:.2f}" for s in S))

    # Contributor embeddings: U * S
    contrib_emb_matrix = U * S[np.newaxis, :]
    contributor_emb = {str(c): contrib_emb_matrix[idx] for c, idx in c_to_idx.items()}

    # Game embeddings: Vt.T * S (right singular vectors scaled)
    game_emb_matrix = Vt.T * S[np.newaxis, :]
    game_emb = {int(g): game_emb_matrix[idx] for g, idx in g_to_idx.items()}

    return contributor_emb, game_emb, k


def add_annotator_emb_features(X, report_ids, contributor_emb, game_emb, n_components, db_path):
    """Add annotator SVD embedding features."""
    ri = get_report_info(db_path)
    cl = get_contrib_lookup(db_path)

    X = X.copy()
    c_emb_arr = np.full((len(report_ids), n_components), np.nan)
    g_emb_arr = np.full((len(report_ids), n_components), np.nan)

    for i, rid in enumerate(report_ids):
        # Contributor embedding (train-time only, but we add it as feature)
        cid = cl.get(rid)
        if cid and cid in contributor_emb:
            c_emb_arr[i] = contributor_emb[cid]

        # Game embedding (inference-time)
        app_id, _ = ri.get(rid, (None, None))
        if app_id and app_id in game_emb:
            g_emb_arr[i] = game_emb[app_id]

    # Fill NaN with 0 (neutral)
    c_emb_arr = np.nan_to_num(c_emb_arr, 0)
    g_emb_arr = np.nan_to_num(g_emb_arr, 0)

    for d in range(n_components):
        X[f"ann_contrib_emb_{d}"] = c_emb_arr[:, d]
        X[f"ann_game_emb_{d}"] = g_emb_arr[:, d]

    c_coverage = sum(1 for rid in report_ids if cl.get(rid) in contributor_emb) / len(report_ids)
    g_coverage = sum(1 for rid in report_ids
                     if ri.get(rid, (None,))[0] in game_emb) / len(report_ids)
    logger.info("Annotator embeddings: contributor_coverage=%.1f%%, game_coverage=%.1f%%",
                c_coverage * 100, g_coverage * 100)

    return X


# ── Train + eval ─────────────────────────────────────────────────────


def train_and_eval(X_train, y_train, X_test, y_test, s1_model=None, label=""):
    """Train cascade. Reuse Stage 1 if feature count matches."""
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    # Ensure categorical dtypes match for predict
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Reuse Stage 1 only if feature count matches
    if s1_model is not None and s1_model.n_features_ != X_train.shape[1]:
        logger.info("Stage 1 feature mismatch (%d vs %d), retraining",
                     s1_model.n_features_, X_train.shape[1])
        s1_model = None

    if s1_model is None:
        s1_model = train_stage1(X_train, y_train, X_test, y_test)

    s2, drops = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1_model, s2, drops)
    y_pred = cascade.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    per = f1_score(y_test, y_pred, average=None)
    return {"label": label, "f1_macro": f1, "borked_f1": per[0],
            "tinkering_f1": per[1], "works_oob_f1": per[2]}, s1_model


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 13.4-13.5: Iterative IRT + Annotator SVD")
    print("=" * 70)

    # Load data (no relabeling, no Cleanlab)
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)
    conn.close()

    # Pre-cache lookups
    get_report_info(args.db)
    get_contrib_lookup(args.db)

    results = []

    # ── Baseline: 1-round IRT + 13.2 relabel (current best) ─────────
    print("\n" + "=" * 70)
    print("BASELINE: IRT features + contributor-aware relabel (Phase 13.2)")
    print("=" * 70)
    irt_df = prepare_irt_df(args.db)
    theta, difficulty, _ = fit_irt_from_df(irt_df.copy())
    logger.info("IRT round 0: %d contributors, %d items", len(theta), len(difficulty))

    X_tr = add_irt_features(X_train, train_rids, theta, difficulty, args.db)
    X_te = add_irt_features(X_test, test_rids, theta, difficulty, args.db)
    y_tr, n = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, theta, args.db)
    logger.info("Relabeled: %d", n)

    r, s1_model = train_and_eval(X_tr, y_tr, X_te, y_test, label="baseline_1round")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.4: Iterative IRT refinement (2 rounds) ────────────────────
    print("\n" + "=" * 70)
    print("13.4a: Iterative IRT (2 rounds)")
    print("=" * 70)

    # Round 1: use model to find high-confidence disagreements, update IRT labels
    from protondb_settings.ml.models.cascade import CascadeClassifier

    # Get Stage 2 predictions on training set for self-training
    y_tr_pred_proba = s1_model.predict_proba(X_tr)
    # For non-borked: use cascade to get 3-class predictions
    from protondb_settings.ml.models.cascade import train_stage2
    s2_tmp, drops_tmp = train_stage2(X_tr, y_tr, X_te, y_test)
    cascade_tmp = CascadeClassifier(s1_model, s2_tmp, drops_tmp)
    y_tr_pred = cascade_tmp.predict(X_tr)
    y_tr_proba = cascade_tmp.predict_proba(X_tr)

    # Find high-confidence disagreements (model vs label)
    confidence = y_tr_proba.max(axis=1)
    disagree = y_tr_pred != y_tr
    high_conf_disagree = disagree & (confidence > 0.85)
    n_disagree = high_conf_disagree.sum()
    logger.info("Round 1: %d high-confidence disagreements (%.1f%%)",
                n_disagree, n_disagree / len(y_tr) * 100)

    # Cap at 2% of data
    max_relabel = int(len(y_tr) * 0.02)
    if n_disagree > max_relabel:
        # Take top by confidence
        disagree_idx = np.where(high_conf_disagree)[0]
        top_idx = disagree_idx[np.argsort(-confidence[disagree_idx])[:max_relabel]]
        high_conf_disagree = np.zeros(len(y_tr), dtype=bool)
        high_conf_disagree[top_idx] = True
        n_disagree = max_relabel

    # Apply model-guided relabeling
    y_tr_r2 = y_tr.copy()
    y_tr_r2[high_conf_disagree] = y_tr_pred[high_conf_disagree]
    logger.info("Round 1: relabeled %d samples", n_disagree)

    # Re-fit IRT with updated labels
    irt_df_r2 = irt_df.copy()
    # Update IRT responses based on relabeled data
    cl = get_contrib_lookup(args.db)
    relabeled_rids = {train_rids[i] for i in np.where(high_conf_disagree)[0]}
    for i, rid in enumerate(train_rids):
        if rid in relabeled_rids and rid in irt_df_r2["id"].values:
            new_label = y_tr_r2[i]
            if new_label == 1:  # tinkering
                irt_df_r2.loc[irt_df_r2["id"] == rid, "response"] = 1
            elif new_label == 2:  # works_oob
                irt_df_r2.loc[irt_df_r2["id"] == rid, "response"] = 0

    theta_r2, difficulty_r2, _ = fit_irt_from_df(irt_df_r2)
    logger.info("IRT round 1: %d contributors, %d items", len(theta_r2), len(difficulty_r2))

    # Rebuild features with round-2 IRT
    X_tr_r2 = add_irt_features(X_train, train_rids, theta_r2, difficulty_r2, args.db)
    X_te_r2 = add_irt_features(X_test, test_rids, theta_r2, difficulty_r2, args.db)

    r, _ = train_and_eval(X_tr_r2, y_tr_r2, X_te_r2, y_test, s1_model=s1_model,
                          label="13.4a_iterative_2rounds")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.4b: 3 rounds ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("13.4b: Iterative IRT (3 rounds, threshold 0.80)")
    print("=" * 70)

    # Round 2: lower threshold
    s2_tmp2, drops_tmp2 = train_stage2(X_tr_r2, y_tr_r2, X_te_r2, y_test)
    cascade_tmp2 = CascadeClassifier(s1_model, s2_tmp2, drops_tmp2)
    y_tr_pred2 = cascade_tmp2.predict(X_tr_r2)
    y_tr_proba2 = cascade_tmp2.predict_proba(X_tr_r2)

    confidence2 = y_tr_proba2.max(axis=1)
    disagree2 = y_tr_pred2 != y_tr_r2
    high_conf2 = disagree2 & (confidence2 > 0.80)
    n_d2 = min(high_conf2.sum(), int(len(y_tr) * 0.02))

    if n_d2 > 0:
        d_idx = np.where(high_conf2)[0]
        top_idx = d_idx[np.argsort(-confidence2[d_idx])[:n_d2]]
        y_tr_r3 = y_tr_r2.copy()
        y_tr_r3[top_idx] = y_tr_pred2[top_idx]
        logger.info("Round 2: relabeled %d samples", n_d2)

        # Re-fit IRT
        irt_df_r3 = irt_df_r2.copy()
        for idx in top_idx:
            rid = train_rids[idx]
            if rid in irt_df_r3["id"].values:
                irt_df_r3.loc[irt_df_r3["id"] == rid, "response"] = int(y_tr_r3[idx] == 1)
        theta_r3, difficulty_r3, _ = fit_irt_from_df(irt_df_r3)

        X_tr_r3 = add_irt_features(X_train, train_rids, theta_r3, difficulty_r3, args.db)
        X_te_r3 = add_irt_features(X_test, test_rids, theta_r3, difficulty_r3, args.db)

        r, _ = train_and_eval(X_tr_r3, y_tr_r3, X_te_r3, y_test, s1_model=s1_model,
                              label="13.4b_iterative_3rounds")
    else:
        r = results[-1].copy()
        r["label"] = "13.4b_iterative_3rounds (no change)"
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.5a: Annotator SVD embeddings (game-side only) ─────────────
    print("\n" + "=" * 70)
    print("13.5a: Annotator SVD game embeddings (8 dims)")
    print("=" * 70)
    contributor_emb, game_emb, n_comp = build_annotator_embeddings(args.db, n_components=8)

    # Add only game-side embeddings (inference-time available)
    X_tr_ge = X_tr.copy()
    X_te_ge = X_te.copy()
    ri = get_report_info(args.db)
    for split_X, split_rids in [(X_tr_ge, train_rids), (X_te_ge, test_rids)]:
        g_arr = np.zeros((len(split_rids), n_comp))
        for i, rid in enumerate(split_rids):
            app_id = ri.get(rid, (None,))[0]
            if app_id and app_id in game_emb:
                g_arr[i] = game_emb[app_id]
        for d in range(n_comp):
            split_X[f"ann_game_emb_{d}"] = g_arr[:, d]

    r, _ = train_and_eval(X_tr_ge, y_tr, X_te_ge, y_test, s1_model=s1_model,
                          label="13.5a_game_emb_only")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 13.5b: Full annotator embeddings (contributor + game) ────────
    print("\n" + "=" * 70)
    print("13.5b: Full annotator embeddings (contributor + game, 8+8 dims)")
    print("=" * 70)
    X_tr_full = add_annotator_emb_features(X_tr, train_rids, contributor_emb, game_emb, n_comp, args.db)
    X_te_full = add_annotator_emb_features(X_te, test_rids, contributor_emb, game_emb, n_comp, args.db)

    r, _ = train_and_eval(X_tr_full, y_tr, X_te_full, y_test, s1_model=s1_model,
                          label="13.5b_full_emb")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── Combined: iterative + SVD ────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMBINED: Best iterative + SVD game embeddings")
    print("=" * 70)
    # Use round-2 IRT + game SVD embeddings
    X_tr_comb = add_irt_features(X_train, train_rids, theta_r2, difficulty_r2, args.db)
    X_te_comb = add_irt_features(X_test, test_rids, theta_r2, difficulty_r2, args.db)
    for split_X, split_rids in [(X_tr_comb, train_rids), (X_te_comb, test_rids)]:
        g_arr = np.zeros((len(split_rids), n_comp))
        for i, rid in enumerate(split_rids):
            app_id = ri.get(rid, (None,))[0]
            if app_id and app_id in game_emb:
                g_arr[i] = game_emb[app_id]
        for d in range(n_comp):
            split_X[f"ann_game_emb_{d}"] = g_arr[:, d]

    r, _ = train_and_eval(X_tr_comb, y_tr_r2, X_te_comb, y_test, s1_model=s1_model,
                          label="combined_iter+svd")
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
    bl = results[0]["f1_macro"]
    for r in results:
        d = r["f1_macro"] - bl
        print(f"{r['label']:<35s} {r['f1_macro']:>7.4f} {d:>+7.4f} "
              f"{r['borked_f1']:>7.3f} {r['tinkering_f1']:>7.3f} {r['works_oob_f1']:>7.3f}")


if __name__ == "__main__":
    main()
