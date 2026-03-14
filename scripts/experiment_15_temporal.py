"""Phase 15: Temporal validity experiments.

Experiments:
  15.1   — Proton version numeric
  15.2   — Time-decay sample weighting
  15.3   — Game-temporal features (verdict trend, latest score)
  15.3b  — Per-game optimal Proton
  15.5   — Temporal label correction
  15.6   — Proton × Game SVD embeddings
  15.7   — Factorization Machines (multi-way interactions)
  Combined

Usage:
  python scripts/experiment_15_temporal.py [--db data/protondb.db]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Cached lookups ───────────────────────────────────────────────────

_report_data = None  # DataFrame with id, app_id, timestamp, gpu, proton_version, verdict, verdict_oob


def get_report_data(db_path):
    global _report_data
    if _report_data is None:
        from protondb_settings.db.connection import get_connection
        conn = get_connection(db_path)
        _report_data = pd.read_sql_query(
            "SELECT id, app_id, timestamp, gpu, proton_version, custom_proton_version, "
            "verdict, verdict_oob FROM reports", conn)
        _report_data["timestamp"] = pd.to_numeric(_report_data["timestamp"], errors="coerce").fillna(0).astype(np.int64)
        conn.close()
        logger.info("Report data cached: %d rows", len(_report_data))
    return _report_data


# ── 15.1: Proton version parsing ─────────────────────────────────────

_PROTON_VER_RE = re.compile(r"(\d+)\.(\d+)-?(\d+)?")
_GE_RE = re.compile(r"GE-Proton(\d+)-(\d+)", re.IGNORECASE)


def parse_proton_version(pv: str | None) -> tuple[float, int, bool, bool]:
    """Parse proton_version → (numeric, major, is_ge, is_experimental).

    Returns (nan, 0, False, False) for unparseable.
    """
    if not pv or not isinstance(pv, str):
        return np.nan, 0, False, False

    pv = pv.strip()

    if "experimental" in pv.lower():
        return 99.0, 99, False, True

    m = _GE_RE.search(pv)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        return major + minor / 100, major, True, False

    m = _PROTON_VER_RE.search(pv)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3)) if m.group(3) else 0
        return major + minor / 10 + patch / 1000, major, False, False

    return np.nan, 0, False, False


def add_proton_features(X, report_ids, db_path):
    """Phase 15.1: Add proton version features."""
    rd = get_report_data(db_path).set_index("id")

    versions = []
    majors = []
    is_ge = []
    is_exp = []
    generations = []

    for rid in report_ids:
        if rid in rd.index:
            row = rd.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            pv = row.get("custom_proton_version") or row.get("proton_version")
            v, maj, ge, exp = parse_proton_version(pv)
        else:
            v, maj, ge, exp = np.nan, 0, False, False

        versions.append(v)
        majors.append(maj)
        is_ge.append(int(ge))
        is_exp.append(int(exp))
        # Generation buckets
        if exp:
            generations.append(4)
        elif maj >= 8:
            generations.append(3)  # modern
        elif maj >= 6:
            generations.append(2)  # stable
        elif maj >= 1:
            generations.append(1)  # legacy
        else:
            generations.append(0)  # unknown

    X = X.copy()
    vs = pd.Series(versions)
    X["proton_version_numeric"] = vs.fillna(vs.median()).values
    X["proton_major"] = majors
    X["proton_is_ge"] = is_ge
    X["proton_is_experimental"] = is_exp
    X["proton_generation"] = generations

    coverage = vs.notna().mean() * 100
    logger.info("Proton features: coverage=%.1f%%", coverage)
    return X


# ── 15.2: Time-decay weighting ───────────────────────────────────────


def compute_time_decay_weights(report_ids, db_path, half_life_days=730):
    """Phase 15.2: Sample weights with exponential time decay."""
    rd = get_report_data(db_path).set_index("id")
    now_ts = rd["timestamp"].max()

    weights = np.ones(len(report_ids))
    for i, rid in enumerate(report_ids):
        if rid in rd.index:
            row = rd.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            ts = row["timestamp"]
            age_days = (now_ts - ts) / 86400
            weights[i] = 0.3 + 0.7 * np.exp(-age_days * np.log(2) / half_life_days)

    return weights


# ── 15.3: Game-temporal features ─────────────────────────────────────


def _verdict_score(verdict, verdict_oob):
    if verdict == "no":
        return 0
    if verdict_oob == "yes":
        return 2
    return 1


def add_game_temporal_features(X, report_ids, db_path, train_rids_set=None):
    """Phase 15.3: Game-temporal features (trend, latest, etc.)."""
    rd = get_report_data(db_path)

    # Compute per-report verdict score
    rd = rd.copy()
    rd["vscore"] = rd.apply(lambda r: _verdict_score(r["verdict"], r["verdict_oob"]), axis=1)

    # Parse proton version for all reports
    rd["pv_numeric"] = rd["proton_version"].apply(lambda pv: parse_proton_version(pv)[0])

    # Only use train data for computing features (leakage protection)
    if train_rids_set:
        rd_train = rd[rd["id"].isin(train_rids_set)]
    else:
        rd_train = rd

    # Per-game temporal stats (from train data)
    game_stats = {}
    for app_id, group in rd_train.groupby("app_id"):
        group = group.sort_values("timestamp")
        scores = group["vscore"].values
        timestamps = group["timestamp"].values

        # Verdict trend (linear regression slope)
        trend = 0.0
        if len(scores) >= 3:
            t_norm = (timestamps - timestamps.min()) / max(1, timestamps.max() - timestamps.min())
            if np.std(t_norm) > 0:
                trend = np.corrcoef(t_norm, scores)[0, 1]
                if np.isnan(trend):
                    trend = 0.0

        # Latest verdict score (last 5)
        latest_score = scores[-5:].mean() if len(scores) > 0 else 1.0

        # Proton version range
        pvs = group["pv_numeric"].dropna()
        pv_range = (pvs.max() - pvs.min()) if len(pvs) >= 2 else 0.0

        # Has recent reports (last 6 months = 180 days)
        max_ts = rd["timestamp"].max()
        has_recent = int((max_ts - timestamps.max()) / 86400 < 180) if len(timestamps) > 0 else 0

        game_stats[app_id] = {
            "trend": trend,
            "latest_score": latest_score,
            "pv_range": pv_range if not np.isnan(pv_range) else 0.0,
            "has_recent": has_recent,
        }

    rd_idx = rd.set_index("id")
    X = X.copy()
    trends = []
    latest_scores = []
    pv_ranges = []
    has_recents = []
    proton_vs_median = []

    for rid in report_ids:
        if rid in rd_idx.index:
            row = rd_idx.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            app_id = row["app_id"]
            gs = game_stats.get(app_id, {})
            trends.append(gs.get("trend", 0))
            latest_scores.append(gs.get("latest_score", 1.0))
            pv_ranges.append(gs.get("pv_range", 0))
            has_recents.append(gs.get("has_recent", 0))

            # report proton vs game median proton
            pv = parse_proton_version(row.get("proton_version"))[0]
            game_reports = rd_train[rd_train["app_id"] == app_id]["pv_numeric"].dropna()
            median_pv = game_reports.median() if len(game_reports) > 0 else np.nan
            proton_vs_median.append(pv - median_pv if not np.isnan(pv) and not np.isnan(median_pv) else 0)
        else:
            trends.append(0)
            latest_scores.append(1.0)
            pv_ranges.append(0)
            has_recents.append(0)
            proton_vs_median.append(0)

    X["game_verdict_trend"] = trends
    X["game_latest_verdict_score"] = latest_scores
    X["game_proton_version_range"] = pv_ranges
    X["game_has_recent_reports"] = has_recents
    X["report_proton_vs_game_median"] = proton_vs_median

    return X


# ── 15.3b: Per-game optimal Proton ──────────────────────────────────


def add_optimal_proton_features(X, report_ids, db_path, pics_cache=None, train_rids_set=None):
    """Phase 15.3b: Per-game optimal proton version features."""
    rd = get_report_data(db_path)
    rd = rd.copy()
    rd["vscore"] = rd.apply(lambda r: _verdict_score(r["verdict"], r["verdict_oob"]), axis=1)
    rd["pv_numeric"] = rd["proton_version"].apply(lambda pv: parse_proton_version(pv)[0])

    if train_rids_set:
        rd_train = rd[rd["id"].isin(train_rids_set)]
    else:
        rd_train = rd

    # Per-game: best proton version and regression detection
    game_proton = {}
    for app_id, group in rd_train.groupby("app_id"):
        pv_scores = group.groupby("pv_numeric")["vscore"].mean()
        pv_scores = pv_scores[pv_scores.index.notna()]
        if len(pv_scores) == 0:
            continue
        best_pv = pv_scores.idxmax()
        best_score = pv_scores.max()

        # Regression detection: is there a version after best_pv with lower score?
        later = pv_scores[pv_scores.index > best_pv]
        has_regression = int(any(later < best_score - 0.3)) if len(later) > 0 else 0

        # Stability: variance of scores across proton versions
        stability = pv_scores.std() if len(pv_scores) >= 2 else 0.0

        game_proton[app_id] = {
            "best_pv": best_pv,
            "has_regression": has_regression,
            "stability": stability if not np.isnan(stability) else 0.0,
        }

    # PICS recommended_runtime matching
    rd_idx = rd.set_index("id")
    X = X.copy()
    vs_best = []
    regressions = []
    stabilities = []
    matches_recommended = []

    for rid in report_ids:
        if rid in rd_idx.index:
            row = rd_idx.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            app_id = row["app_id"]
            gp = game_proton.get(app_id, {})
            pv = parse_proton_version(row.get("proton_version"))[0]
            best = gp.get("best_pv", np.nan)
            vs_best.append(pv - best if not np.isnan(pv) and not np.isnan(best) else 0)
            regressions.append(gp.get("has_regression", 0))
            stabilities.append(gp.get("stability", 0))

            # Check PICS recommended runtime
            match = 0
            if pics_cache and app_id in pics_cache:
                rec = pics_cache[app_id].get("recommended_runtime", "")
                if rec:
                    rec_pv = parse_proton_version(rec)[0]
                    if not np.isnan(pv) and not np.isnan(rec_pv):
                        match = 1 if abs(pv - rec_pv) < 0.5 else 0
            matches_recommended.append(match)
        else:
            vs_best.append(0)
            regressions.append(0)
            stabilities.append(0)
            matches_recommended.append(0)

    X["report_proton_vs_best"] = vs_best
    X["game_has_proton_regression"] = regressions
    X["game_proton_stability"] = stabilities
    X["report_proton_matches_recommended"] = matches_recommended

    return X


# ── 15.5: Temporal label correction ──────────────────────────────────


def temporal_label_correction(y, report_ids, db_path, train_rids_set=None):
    """Phase 15.5: Relabel old borked → works if recent reports say works."""
    rd = get_report_data(db_path)
    rd = rd.copy()
    rd["vscore"] = rd.apply(lambda r: _verdict_score(r["verdict"], r["verdict_oob"]), axis=1)
    rd["pv_numeric"] = rd["proton_version"].apply(lambda pv: parse_proton_version(pv)[0])

    if train_rids_set:
        rd_use = rd[rd["id"].isin(train_rids_set)]
    else:
        rd_use = rd

    from protondb_settings.ml.features.encoding import extract_gpu_family

    # Per (app_id, gpu_family): check if recent reports are unanimously works
    rd_use = rd_use.copy()
    rd_use["gpu_family"] = rd_use["gpu"].apply(lambda g: extract_gpu_family(g) if g else "unknown")

    game_gpu_recent = {}
    for (app_id, gf), group in rd_use.groupby(["app_id", "gpu_family"]):
        group = group.sort_values("timestamp")
        if len(group) < 4:
            continue
        recent = group.tail(3)
        old = group.head(len(group) - 3)

        # Recent all works (score > 0) AND old has borked (score == 0)
        if recent["vscore"].min() > 0 and (old["vscore"] == 0).any():
            # Check proton version increased
            old_pv = old["pv_numeric"].dropna()
            recent_pv = recent["pv_numeric"].dropna()
            if len(old_pv) > 0 and len(recent_pv) > 0 and recent_pv.min() > old_pv.max():
                game_gpu_recent[(app_id, gf)] = set(old[old["vscore"] == 0]["id"].values)

    # Apply correction
    rd_idx = rd.set_index("id")
    y_new = y.copy()
    n_corrected = 0
    relabel_rids = set()
    for rids_set in game_gpu_recent.values():
        relabel_rids |= rids_set

    for i, rid in enumerate(report_ids):
        if rid in relabel_rids and y_new[i] == 0:  # borked → tinkering
            y_new[i] = 1
            n_corrected += 1

    logger.info("Temporal correction: %d borked → tinkering (%d game×gpu pairs)",
                n_corrected, len(game_gpu_recent))
    return y_new, n_corrected


# ── 15.6: Proton × Game SVD ──────────────────────────────────────────


def build_proton_game_svd(db_path, n_components=8, train_rids_set=None):
    """Phase 15.6: SVD on Proton × Game compatibility matrix."""
    rd = get_report_data(db_path)
    rd = rd.copy()
    rd["vscore"] = rd.apply(lambda r: _verdict_score(r["verdict"], r["verdict_oob"]), axis=1)

    # Bucket proton versions
    rd["pv_bucket"] = rd["proton_version"].apply(lambda pv: _proton_bucket(pv))

    if train_rids_set:
        rd = rd[rd["id"].isin(train_rids_set)]

    # Filter to known buckets
    rd = rd[rd["pv_bucket"] != "unknown"]

    buckets = sorted(rd["pv_bucket"].unique())
    games = rd["app_id"].unique()
    b_to_idx = {b: i for i, b in enumerate(buckets)}
    g_to_idx = {g: i for i, g in enumerate(games)}

    logger.info("Proton×Game SVD: %d buckets × %d games", len(buckets), len(games))

    # Build mean score matrix
    agg = rd.groupby(["pv_bucket", "app_id"])["vscore"].mean()

    rows, cols, vals = [], [], []
    for (bucket, app_id), score in agg.items():
        rows.append(b_to_idx[bucket])
        cols.append(g_to_idx[app_id])
        vals.append(score)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(buckets), len(games)))

    k = min(n_components, min(mat.shape) - 1)
    U, S, Vt = svds(mat.astype(float), k=k)
    order = np.argsort(-S)
    U, S, Vt = U[:, order], S[order], Vt[order, :]

    logger.info("Proton×Game SVD: %d components, σ: %s", k, ", ".join(f"{s:.2f}" for s in S))

    # Proton embeddings: U·S
    proton_emb = {b: (U[idx] * S) for b, idx in b_to_idx.items()}
    # Game embeddings: Vt.T·S
    game_emb_matrix = Vt.T * S[np.newaxis, :]
    game_emb = {g: game_emb_matrix[idx] for g, idx in g_to_idx.items()}

    return proton_emb, game_emb, k, buckets


def _proton_bucket(pv):
    if not pv or not isinstance(pv, str):
        return "unknown"
    pv = pv.strip().lower()
    if "experimental" in pv:
        return "experimental"
    m = _GE_RE.search(pv)
    if m:
        return f"ge_{m.group(1)}"
    m = _PROTON_VER_RE.search(pv)
    if m:
        major = int(m.group(1))
        if major <= 4:
            return "v3-4"
        return f"v{major}"
    return "unknown"


def add_proton_game_svd_features(X, report_ids, proton_emb, game_emb, n_comp, db_path):
    """Add Proton×Game SVD features."""
    rd = get_report_data(db_path).set_index("id")

    X = X.copy()
    p_arr = np.zeros((len(report_ids), n_comp))
    g_arr = np.zeros((len(report_ids), n_comp))
    dots = np.zeros(len(report_ids))

    for i, rid in enumerate(report_ids):
        if rid in rd.index:
            row = rd.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            bucket = _proton_bucket(row.get("proton_version"))
            app_id = row["app_id"]

            if bucket in proton_emb:
                p_arr[i] = proton_emb[bucket]
            if app_id in game_emb:
                g_arr[i] = game_emb[app_id]
            dots[i] = np.dot(p_arr[i], g_arr[i])

    for d in range(n_comp):
        X[f"proton_svd_{d}"] = p_arr[:, d]
        X[f"game_proton_svd_{d}"] = g_arr[:, d]
    X["proton_game_dot"] = dots

    return X


# ── 15.7: Factorization Machines ─────────────────────────────────────


def train_fm_and_extract(X, report_ids, y, db_path, k=8, train_rids_set=None):
    """Phase 15.7: Train FM on categorical interactions, extract score + embeddings."""
    rd = get_report_data(db_path)
    rd = rd.copy()
    rd["vscore"] = rd.apply(lambda r: _verdict_score(r["verdict"], r["verdict_oob"]), axis=1)
    rd["pv_bucket"] = rd["proton_version"].apply(lambda pv: _proton_bucket(pv))

    from protondb_settings.ml.features.encoding import extract_gpu_family
    rd["gpu_family"] = rd["gpu"].apply(lambda g: extract_gpu_family(g) if g else "unknown")

    if train_rids_set:
        rd_train = rd[rd["id"].isin(train_rids_set)]
    else:
        rd_train = rd

    # Build categorical encodings
    app_ids = rd_train["app_id"].unique()
    pv_buckets = [b for b in rd_train["pv_bucket"].unique() if b != "unknown"]
    gpu_families = rd_train["gpu_family"].unique()

    app_to_idx = {a: i for i, a in enumerate(app_ids)}
    pv_to_idx = {p: i for i, p in enumerate(pv_buckets)}
    gpu_to_idx = {g: i for i, g in enumerate(gpu_families)}

    n_app, n_pv, n_gpu = len(app_ids), len(pv_buckets), len(gpu_families)
    n_total = n_app + n_pv + n_gpu

    logger.info("FM: %d apps × %d proton × %d gpus = %d entities, k=%d",
                n_app, n_pv, n_gpu, n_total, k)

    # Build training data for FM
    fm_rows = []
    fm_y = []
    for _, row in rd_train.iterrows():
        a_idx = app_to_idx.get(row["app_id"])
        p_idx = pv_to_idx.get(row["pv_bucket"])
        g_idx = gpu_to_idx.get(row["gpu_family"])
        if a_idx is None or p_idx is None or g_idx is None:
            continue
        fm_rows.append((a_idx, n_app + p_idx, n_app + n_pv + g_idx))
        fm_y.append(row["vscore"] / 2.0)  # normalize to [0, 1]

    fm_y = np.array(fm_y)

    # Simple FM via SGD (2nd order interactions only)
    np.random.seed(42)
    V = np.random.randn(n_total, k) * 0.01
    w = np.zeros(n_total)
    w0 = fm_y.mean()
    lr = 0.01
    reg = 0.001

    logger.info("FM training: %d samples, %d iterations...", len(fm_rows), 20)
    for epoch in range(20):
        perm = np.random.permutation(len(fm_rows))
        total_loss = 0
        for idx in perm:
            indices = fm_rows[idx]
            yi = fm_y[idx]

            # Predict
            pred = w0 + sum(w[j] for j in indices)
            # 2nd order: sum of pairwise dot products
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    pred += np.dot(V[indices[a]], V[indices[b]])
            pred = np.clip(pred, 0, 1)

            err = pred - yi
            total_loss += err ** 2

            # Update
            for j in indices:
                w[j] -= lr * (err + reg * w[j])
                sum_vx = sum(V[jj] for jj in indices) - V[j]
                V[j] -= lr * (err * sum_vx + reg * V[j])
            w0 -= lr * err

        if (epoch + 1) % 5 == 0:
            logger.info("  FM epoch %d: MSE=%.4f", epoch + 1, total_loss / len(fm_rows))

    # Extract features for all reports
    rd_idx = rd.set_index("id")
    X = X.copy()
    fm_scores = []
    fm_game_embs = np.zeros((len(report_ids), k))
    fm_proton_embs = np.zeros((len(report_ids), k))

    for i, rid in enumerate(report_ids):
        if rid in rd_idx.index:
            row = rd_idx.loc[rid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            a_idx = app_to_idx.get(row["app_id"])
            p_idx = pv_to_idx.get(_proton_bucket(row.get("proton_version")))
            g_idx = gpu_to_idx.get(extract_gpu_family(row["gpu"]) if row["gpu"] else "unknown")

            score = w0
            indices = []
            if a_idx is not None:
                indices.append(a_idx)
                fm_game_embs[i] = V[a_idx]
            if p_idx is not None:
                indices.append(n_app + p_idx)
                fm_proton_embs[i] = V[n_app + p_idx]
            if g_idx is not None:
                indices.append(n_app + n_pv + g_idx)

            score += sum(w[j] for j in indices)
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    score += np.dot(V[indices[a]], V[indices[b]])
            fm_scores.append(np.clip(score, 0, 1))
        else:
            fm_scores.append(0.5)

    X["fm_score"] = fm_scores
    for d in range(k):
        X[f"fm_game_emb_{d}"] = fm_game_embs[:, d]
        X[f"fm_proton_emb_{d}"] = fm_proton_embs[:, d]

    logger.info("FM features added: score coverage=%.1f%%",
                sum(1 for s in fm_scores if s != 0.5) / len(fm_scores) * 100)
    return X


# ── Train + eval (Stage 1 cached by feature count) ──────────────────

_s1_cache = {}


def train_and_eval(X_train, y_train, X_test, y_test, sample_weight=None, label=""):
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    n_feat = X_train.shape[1]
    if n_feat in _s1_cache:
        s1 = _s1_cache[n_feat]
    else:
        s1 = train_stage1(X_train, y_train, X_test, y_test)
        _s1_cache[n_feat] = s1

    # Stage 2 with optional weights
    from protondb_settings.ml.models.cascade import STAGE2_DROP_FEATURES
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
    w = sample_weight[mask_tr] if sample_weight is not None else None

    ds_tr = lgb.Dataset(X2_tr, label=y_smooth, weight=w, categorical_feature=cats)
    ds_te = lgb.Dataset(X2_te, label=y2_te, categorical_feature=cats)
    s2 = lgb.train(
        {"objective": "cross_entropy", "metric": "binary_logloss",
         "num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 50,
         "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
         "reg_alpha": 0.1, "reg_lambda": 0.1, "min_split_gain": 0.05, "verbose": -1},
        ds_tr, num_boost_round=2000, valid_sets=[ds_te],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    cascade = CascadeClassifier(s1, s2, drops)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    per = f1_score(y_test, y_pred, average=None)
    return {"label": label, "f1_macro": f1, "borked_f1": per[0],
            "tinkering_f1": per[1], "works_oob_f1": per[2]}


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 15: Temporal validity experiments")
    print("=" * 70)

    # Load base data
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import fit_irt, add_irt_features, contributor_aware_relabel

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")
    X, y, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)

    # IRT baseline (Phase 12-13)
    theta, difficulty = fit_irt(conn)
    X_tr = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_te = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    y_tr, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)
    conn.close()

    train_rids_set = set(train_rids)

    # Load PICS cache
    pics_cache = {}
    try:
        pics_cache = load_pics_cache(args.db)
    except Exception:
        pass

    results = []

    # ── Baseline ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE: IRT + contributor-aware relabel")
    print("=" * 70)
    r = train_and_eval(X_tr.copy(), y_tr, X_te.copy(), y_test, label="baseline")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | b={r['borked_f1']:.3f} t={r['tinkering_f1']:.3f} o={r['works_oob_f1']:.3f}")

    # ── 15.1: Proton version numeric ─────────────────────────────────
    print("\n" + "=" * 70)
    print("15.1: Proton version numeric")
    print("=" * 70)
    X_tr_pv = add_proton_features(X_tr, train_rids, args.db)
    X_te_pv = add_proton_features(X_te, test_rids, args.db)
    r = train_and_eval(X_tr_pv.copy(), y_tr, X_te_pv.copy(), y_test, label="15.1_proton_numeric")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── 15.2: Time-decay weighting ───────────────────────────────────
    print("\n" + "=" * 70)
    print("15.2: Time-decay weighting")
    print("=" * 70)
    w_td = compute_time_decay_weights(train_rids, args.db)
    r = train_and_eval(X_tr.copy(), y_tr, X_te.copy(), y_test, sample_weight=w_td, label="15.2_time_decay")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── 15.3: Game-temporal features ─────────────────────────────────
    print("\n" + "=" * 70)
    print("15.3: Game-temporal features")
    print("=" * 70)
    X_tr_gt = add_game_temporal_features(X_tr, train_rids, args.db, train_rids_set)
    X_te_gt = add_game_temporal_features(X_te, test_rids, args.db, train_rids_set)
    r = train_and_eval(X_tr_gt.copy(), y_tr, X_te_gt.copy(), y_test, label="15.3_game_temporal")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── 15.3b: Per-game optimal Proton ───────────────────────────────
    print("\n" + "=" * 70)
    print("15.3b: Per-game optimal Proton")
    print("=" * 70)
    X_tr_op = add_optimal_proton_features(X_tr_pv, train_rids, args.db, pics_cache, train_rids_set)
    X_te_op = add_optimal_proton_features(X_te_pv, test_rids, args.db, pics_cache, train_rids_set)
    r = train_and_eval(X_tr_op.copy(), y_tr, X_te_op.copy(), y_test, label="15.3b_optimal_proton")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── 15.5: Temporal label correction ──────────────────────────────
    print("\n" + "=" * 70)
    print("15.5: Temporal label correction")
    print("=" * 70)
    y_tc, n_corr = temporal_label_correction(y_tr, train_rids, args.db, train_rids_set)
    r = train_and_eval(X_tr.copy(), y_tc, X_te.copy(), y_test, label="15.5_temporal_correction")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── 15.6: Proton × Game SVD ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("15.6: Proton × Game SVD embeddings")
    print("=" * 70)
    p_emb, g_emb, n_comp, _ = build_proton_game_svd(args.db, n_components=8, train_rids_set=train_rids_set)
    X_tr_svd = add_proton_game_svd_features(X_tr_pv, train_rids, p_emb, g_emb, n_comp, args.db)
    X_te_svd = add_proton_game_svd_features(X_te_pv, test_rids, p_emb, g_emb, n_comp, args.db)
    r = train_and_eval(X_tr_svd.copy(), y_tr, X_te_svd.copy(), y_test, label="15.6_proton_game_svd")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── 15.7: Factorization Machines ─────────────────────────────────
    print("\n" + "=" * 70)
    print("15.7: Factorization Machines")
    print("=" * 70)
    X_tr_fm = train_fm_and_extract(X_tr_pv, train_rids, y_tr, args.db, k=8, train_rids_set=train_rids_set)
    X_te_fm = train_fm_and_extract(X_te_pv, test_rids, y_test, args.db, k=8, train_rids_set=train_rids_set)
    r = train_and_eval(X_tr_fm.copy(), y_tr, X_te_fm.copy(), y_test, label="15.7_fm")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

    # ── Combined: best of each ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMBINED: 15.1 + 15.3 + 15.3b + 15.5 + 15.6")
    print("=" * 70)
    X_tr_c = add_proton_features(X_tr, train_rids, args.db)
    X_te_c = add_proton_features(X_te, test_rids, args.db)
    X_tr_c = add_game_temporal_features(X_tr_c, train_rids, args.db, train_rids_set)
    X_te_c = add_game_temporal_features(X_te_c, test_rids, args.db, train_rids_set)
    X_tr_c = add_optimal_proton_features(X_tr_c, train_rids, args.db, pics_cache, train_rids_set)
    X_te_c = add_optimal_proton_features(X_te_c, test_rids, args.db, pics_cache, train_rids_set)
    X_tr_c = add_proton_game_svd_features(X_tr_c, train_rids, p_emb, g_emb, n_comp, args.db)
    X_te_c = add_proton_game_svd_features(X_te_c, test_rids, p_emb, g_emb, n_comp, args.db)
    y_tc2, _ = temporal_label_correction(y_tr, train_rids, args.db, train_rids_set)
    r = train_and_eval(X_tr_c.copy(), y_tc2, X_te_c.copy(), y_test, label="combined_all")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} (Δ={r['f1_macro']-results[0]['f1_macro']:+.4f})")

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
