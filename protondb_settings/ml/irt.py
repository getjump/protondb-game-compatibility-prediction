"""Item Response Theory (IRT) for annotator-game difficulty decomposition.

Phase 12.8: 1PL IRT model decomposes the tinkering/works_oob boundary into:
  - theta_j: per-contributor "strictness" (high = tends to say tinkering)
  - d_i: per-(game, gpu_family) "difficulty" (high = objectively harder)

P(tinkering | contributor_j, item_i) = sigmoid(theta_j - d_i)

Used for:
  1. Feature: irt_game_difficulty (inference-time, per game×gpu)
  2. Relabeling: contributor-aware label correction (train-time)
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def fit_irt(
    conn: sqlite3.Connection,
    min_annotators_per_item: int = 2,
    min_items_per_annotator: int = 2,
) -> tuple[dict[str, float], dict[tuple[int, str], float]]:
    """Fit 1PL IRT on tinkering/oob boundary.

    Returns:
        theta: {contributor_id: strictness_score}
        difficulty: {(app_id, gpu_family): difficulty_score}
    """
    from .features.encoding import extract_gpu_family

    df = pd.read_sql_query("""
        SELECT r.id, r.app_id, r.gpu, r.verdict_oob, rc.contributor_id
        FROM reports r
        JOIN report_contributors rc ON r.id = rc.report_id
        WHERE r.verdict = 'yes'
    """, conn)

    if df.empty:
        logger.warning("IRT: no reports with contributor data")
        return {}, {}

    # Response: 1 = tinkering (strict), 0 = works_oob (lenient)
    df["response"] = (df["verdict_oob"] != "yes").astype(int)
    df["gpu_family"] = df["gpu"].apply(lambda g: extract_gpu_family(g) if g else "unknown")
    df["item"] = df["app_id"].astype(str) + "_" + df["gpu_family"]

    # Iterative filtering: items with 2+ annotators, annotators with 2+ items
    for _ in range(3):
        ic = df["item"].value_counts()
        df = df[df["item"].isin(ic[ic >= min_annotators_per_item].index)]
        cc = df["contributor_id"].value_counts()
        df = df[df["contributor_id"].isin(cc[cc >= min_items_per_annotator].index)]

    if len(df) < 100:
        logger.warning("IRT: not enough data after filtering (%d)", len(df))
        return {}, {}

    contributors = df["contributor_id"].unique()
    items = df["item"].unique()
    c_to_idx = {c: i for i, c in enumerate(contributors)}
    i_to_idx = {it: i for i, it in enumerate(items)}
    c_idx = df["contributor_id"].map(c_to_idx).values
    i_idx = df["item"].map(i_to_idx).values
    responses = df["response"].values.astype(float)
    n_c, n_i = len(contributors), len(items)

    logger.info("IRT: %d responses, %d items, %d contributors", len(df), n_i, n_c)

    # NLL + L2 regularization
    def nll(params):
        theta, d = params[:n_c], params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        p = np.clip(1 / (1 + np.exp(-logit)), 1e-7, 1 - 1e-7)
        ll = responses * np.log(p) + (1 - responses) * np.log(1 - p)
        return -ll.sum() + 0.01 * (np.sum(theta**2) + np.sum(d**2))

    # Analytical gradient
    def grad(params):
        theta, d = params[:n_c], params[n_c:]
        logit = np.clip(theta[c_idx] - d[i_idx], -20, 20)
        r = 1 / (1 + np.exp(-logit)) - responses
        g_t, g_d = np.zeros(n_c), np.zeros(n_i)
        np.add.at(g_t, c_idx, r)
        np.add.at(g_d, i_idx, -r)
        return np.concatenate([g_t + 0.02 * theta, g_d + 0.02 * d])

    # Initialize from empirical log-odds (clipped)
    x0 = np.zeros(n_c + n_i)
    for c, idx in c_to_idx.items():
        ratio = np.clip(df.loc[df["contributor_id"] == c, "response"].mean(), 0.05, 0.95)
        x0[idx] = np.log(ratio / (1 - ratio))
    for item, idx in i_to_idx.items():
        ratio = np.clip(df.loc[df["item"] == item, "response"].mean(), 0.05, 0.95)
        x0[n_c + idx] = -np.log(ratio / (1 - ratio))

    result = minimize(nll, x0, jac=grad, method="L-BFGS-B",
                     options={"maxiter": 1000, "ftol": 1e-8})
    logger.info("IRT converged: %s, %d iterations", result.success, result.nit)

    theta_vals = result.x[:n_c]
    d_vals = result.x[n_c:]

    theta = {str(c): float(theta_vals[idx]) for c, idx in c_to_idx.items()}
    difficulty = {}
    for item_key, idx in i_to_idx.items():
        parts = item_key.rsplit("_", 1)
        difficulty[(int(parts[0]), parts[1])] = float(d_vals[idx])

    logger.info("IRT theta: mean=%.2f, std=%.2f, range=[%.2f, %.2f]",
                np.mean(theta_vals), np.std(theta_vals), np.min(theta_vals), np.max(theta_vals))
    logger.info("IRT difficulty: mean=%.2f, std=%.2f", np.mean(d_vals), np.std(d_vals))

    return theta, difficulty


def _build_lookups(conn: sqlite3.Connection) -> tuple[dict, dict]:
    """Build and cache report_info and contrib_lookup from DB.

    Returns:
        report_info: {report_id: (app_id, gpu)}
        contrib_lookup: {report_id: contributor_id}
    """
    if not hasattr(_build_lookups, "_cache"):
        _build_lookups._cache = {}

    db_path = conn.execute("PRAGMA database_list").fetchone()["file"]
    if db_path in _build_lookups._cache:
        return _build_lookups._cache[db_path]

    report_info: dict[str, tuple[int, str]] = {}
    for r in conn.execute("SELECT id, app_id, gpu FROM reports").fetchall():
        report_info[r["id"]] = (r["app_id"], r["gpu"])

    contrib_lookup: dict[str, str] = {}
    for r in conn.execute("SELECT report_id, contributor_id FROM report_contributors").fetchall():
        contrib_lookup[r["report_id"]] = str(r["contributor_id"])

    _build_lookups._cache[db_path] = (report_info, contrib_lookup)
    logger.info("IRT lookups cached: %d reports, %d contributors", len(report_info), len(contrib_lookup))
    return report_info, contrib_lookup


def add_irt_features(
    X: pd.DataFrame,
    report_ids: list[str],
    conn: sqlite3.Connection,
    theta: dict[str, float],
    difficulty: dict[tuple[int, str], float],
) -> pd.DataFrame:
    """Add irt_game_difficulty and irt_contributor_strictness to feature matrix."""
    from .features.encoding import extract_gpu_family

    report_info, contrib_lookup = _build_lookups(conn)

    irt_d = []
    irt_t = []

    for rid in report_ids:
        app_id, gpu = report_info.get(rid, (None, None))
        gpu_fam = extract_gpu_family(gpu) if gpu else "unknown"

        # Game difficulty
        d = difficulty.get((app_id, gpu_fam))
        if d is None and app_id is not None:
            app_diffs = [v for (a, g), v in difficulty.items() if a == app_id]
            d = np.mean(app_diffs) if app_diffs else None
        irt_d.append(d if d is not None else np.nan)

        # Contributor strictness
        cid = contrib_lookup.get(rid)
        irt_t.append(theta.get(cid, np.nan) if cid else np.nan)

    X = X.copy()
    irt_d_series = pd.Series(irt_d)
    X["irt_game_difficulty"] = irt_d_series.fillna(irt_d_series.median()).values
    X["irt_contributor_strictness"] = pd.Series(irt_t).fillna(0).values

    coverage_d = irt_d_series.notna().mean() * 100
    coverage_t = pd.Series(irt_t).notna().mean() * 100
    logger.info("IRT features: difficulty coverage=%.1f%%, theta coverage=%.1f%%", coverage_d, coverage_t)

    return X


def contributor_aware_relabel(
    y: np.ndarray,
    report_ids: list[str],
    relabel_ids: set[str],
    conn: sqlite3.Connection,
    theta: dict[str, float],
    theta_threshold: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Contributor-aware relabeling: tinkering → works_oob for strict annotators.

    Only relabels reports that:
    1. Are Phase 8 candidates (no effort markers) — from relabel_ids
    2. Have contributor with theta > threshold (strict annotator)

    Reports without contributor data or with lenient contributors keep original label.
    """
    _, contrib_lookup = _build_lookups(conn)

    y_new = y.copy()
    n_relabeled = 0

    for i, rid in enumerate(report_ids):
        if y_new[i] != 1:  # only tinkering
            continue
        if rid not in relabel_ids:  # has effort markers → keep
            continue

        cid = contrib_lookup.get(rid)
        t = theta.get(cid) if cid else None

        if t is not None and t > theta_threshold:
            y_new[i] = 2
            n_relabeled += 1

    logger.info("Contributor-aware relabel: %d tinkering → works_oob (theta > %.1f)",
                n_relabeled, theta_threshold)
    return y_new, n_relabeled


def add_error_targeted_features(
    X: pd.DataFrame,
    report_ids: list[str],
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Add contributor_consistency and game_verdict_agreement features.

    Phase 16.6: +0.006 F1 in combination with class weighting.
    """
    _, contrib_lookup = _build_lookups(conn)

    # Per-contributor verdict consistency (std of verdict scores)
    contrib_verdicts: dict[str, list[int]] = {}
    report_app: dict[str, int] = {}
    for r in conn.execute("SELECT id, app_id, verdict, verdict_oob FROM reports").fetchall():
        report_app[r["id"]] = r["app_id"]
        cid = contrib_lookup.get(r["id"])
        if not cid:
            continue
        score = 0 if r["verdict"] == "no" else (2 if r["verdict_oob"] == "yes" else 1)
        if cid not in contrib_verdicts:
            contrib_verdicts[cid] = []
        contrib_verdicts[cid].append(score)

    contrib_consistency = {
        cid: float(np.std(scores))
        for cid, scores in contrib_verdicts.items()
        if len(scores) >= 2
    }

    # Per-game verdict agreement (1 - normalized entropy)
    game_verdicts: dict[int, list[int]] = {}
    for r in conn.execute("SELECT app_id, verdict, verdict_oob FROM reports WHERE verdict IS NOT NULL").fetchall():
        app_id = r["app_id"]
        score = 0 if r["verdict"] == "no" else (2 if r["verdict_oob"] == "yes" else 1)
        if app_id not in game_verdicts:
            game_verdicts[app_id] = []
        game_verdicts[app_id].append(score)

    game_agreement = {}
    for app_id, scores in game_verdicts.items():
        if len(scores) >= 3:
            counts = np.bincount(scores, minlength=3)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            game_agreement[app_id] = 1 - entropy / np.log2(3)

    X = X.copy()
    consistencies = []
    agreements = []

    for rid in report_ids:
        cid = contrib_lookup.get(rid)
        consistencies.append(contrib_consistency.get(cid, np.nan) if cid else np.nan)
        app_id = report_app.get(rid)
        agreements.append(game_agreement.get(app_id, np.nan) if app_id else np.nan)

    cs = pd.Series(consistencies)
    ag = pd.Series(agreements)
    X["contributor_consistency"] = cs.fillna(cs.median()).values
    X["game_verdict_agreement"] = ag.fillna(ag.median()).values

    logger.info("Error features: consistency coverage=%.1f%%, agreement coverage=%.1f%%",
                cs.notna().mean() * 100, ag.notna().mean() * 100)
    return X
