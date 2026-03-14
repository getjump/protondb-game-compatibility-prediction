"""Phase 14: Steam PICS features + DX version parsing.

Experiments:
  14.1a — recommended_runtime only
  14.1b — All PICS features (runtime + deck tests + review + osarch)
  14.3  — DX version from pc_requirements
  Combined — All together

Usage:
  python scripts/experiment_14_pics_features.py [--db data/protondb.db]
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
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── PICS feature extraction ─────────────────────────────────────────


def load_pics_cache(db_path: str) -> dict[int, dict]:
    """Load all PICS data from enrichment_cache into memory."""
    from protondb_settings.db.connection import get_connection
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT app_id, data_json FROM enrichment_cache WHERE source = 'steam_pics'"
    ).fetchall()
    conn.close()

    result = {}
    for r in rows:
        if r["data_json"]:
            try:
                d = json.loads(r["data_json"])
                if "_empty" not in d:
                    result[r["app_id"]] = d
            except (json.JSONDecodeError, TypeError):
                pass
    logger.info("PICS cache: %d apps loaded", len(result))
    return result


def _parse_runtime_version(runtime: str | None) -> float:
    """Parse proton version from recommended_runtime string to ordinal."""
    if not runtime:
        return np.nan
    if runtime == "native":
        return 0.0
    # "proton-experimental" → 99, "proton-10.0-beta" → 10.0, "proton-9.0-3RC" → 9.0
    m = re.search(r"proton-(\d+)\.(\d+)", runtime)
    if m:
        return float(m.group(1)) + float(m.group(2)) / 10
    if "experimental" in runtime:
        return 99.0
    return np.nan


def _parse_deck_tests(tests: list[dict] | None) -> dict:
    """Parse granular deck test results into binary features."""
    result = {
        "deck_test_controller_ok": 0,
        "deck_test_glyphs_match": 0,
        "deck_test_text_legible": 0,
        "deck_test_performant": 0,
        "deck_test_anticheat_fail": 0,
        "deck_test_pass_count": 0,
        "deck_test_warn_count": 0,
        "deck_test_fail_count": 0,
    }
    if not tests:
        return result

    for t in tests:
        display = t.get("display", 0)
        token = t.get("token", "")

        if display == 4:  # pass
            result["deck_test_pass_count"] += 1
        elif display == 3:  # warning
            result["deck_test_warn_count"] += 1
        elif display == 2:  # fail
            result["deck_test_fail_count"] += 1

        if "ControllerConfigFullyFunctional" in token and display == 4:
            result["deck_test_controller_ok"] = 1
        if "GlyphsMatchDeckDevice" in token and display == 4:
            result["deck_test_glyphs_match"] = 1
        if "InterfaceTextIsLegible" in token and display == 4:
            result["deck_test_text_legible"] = 1
        if "DefaultConfigurationIsPerformant" in token and display == 4:
            result["deck_test_performant"] = 1
        if "UnsupportedAntiCheat" in token:
            result["deck_test_anticheat_fail"] = 1

    return result


def add_pics_features(X: pd.DataFrame, report_ids: list[str],
                      pics_cache: dict, report_info: dict,
                      mode: str = "all") -> pd.DataFrame:
    """Add PICS features to feature matrix.

    mode: "runtime" — only recommended_runtime
          "all" — all PICS features
    """
    X = X.copy()

    runtime_native = []
    runtime_proton = []
    runtime_version = []
    steamos_compat = []
    osarch_64 = []
    has_linux_launch = []
    has_linux_depot = []
    review_score = []
    review_pct = []
    is_free = []
    # Deck test features
    deck_tests_data = []

    for rid in report_ids:
        app_id = report_info.get(rid, (None,))[0]
        pics = pics_cache.get(app_id, {}) if app_id else {}

        rt = pics.get("recommended_runtime")
        runtime_native.append(1 if rt == "native" else 0)
        runtime_proton.append(1 if rt and "proton" in str(rt) else 0)
        runtime_version.append(_parse_runtime_version(rt))

        steamos_compat.append(pics.get("steamos_compatibility"))
        osarch_64.append(1 if pics.get("osarch") == "64" else 0)
        has_linux_launch.append(1 if pics.get("has_linux_launch") else 0)

        ldc = pics.get("linux_depot_count", 0) or 0
        has_linux_depot.append(1 if ldc > 0 else 0)

        review_score.append(pics.get("review_score"))
        review_pct.append(pics.get("review_percentage"))
        is_free.append(1 if pics.get("is_free") else 0)

        deck_tests_data.append(_parse_deck_tests(pics.get("deck_test_results")))

    if mode in ("runtime", "all"):
        X["pics_runtime_native"] = runtime_native
        X["pics_runtime_proton"] = runtime_proton
        X["pics_runtime_version"] = pd.Series(runtime_version).fillna(-1).values

    if mode == "all":
        X["pics_steamos_compat"] = pd.Series(steamos_compat).fillna(-1).astype(int).values
        X["pics_osarch_64"] = osarch_64
        X["pics_has_linux_launch"] = has_linux_launch
        X["pics_has_linux_depot"] = has_linux_depot
        X["pics_review_score"] = pd.Series(review_score).fillna(0).astype(int).values
        X["pics_review_pct"] = pd.Series(review_pct).fillna(0).astype(int).values
        X["pics_is_free"] = is_free

        # Deck test details
        dt_df = pd.DataFrame(deck_tests_data)
        for col in dt_df.columns:
            X[col] = dt_df[col].values

    coverage = sum(1 for rid in report_ids
                   if report_info.get(rid, (None,))[0] in pics_cache) / len(report_ids) * 100
    logger.info("PICS features (mode=%s): coverage=%.1f%%", mode, coverage)

    return X


# ── DX version parsing (Phase 14.3) ─────────────────────────────────


_DX_RE = re.compile(r"DirectX.*?(?:Version\s*)?(\d+)", re.IGNORECASE)
_DX_SHORT_RE = re.compile(r"\bDX\s*(\d+)\b", re.IGNORECASE)
_VULKAN_RE = re.compile(r"\bVulkan\b", re.IGNORECASE)
_OPENGL_RE = re.compile(r"\bOpenGL\b", re.IGNORECASE)


def load_steam_cache(db_path: str) -> dict[int, dict]:
    """Load Steam Store appdetails cache."""
    from protondb_settings.db.connection import get_connection
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT app_id, data_json FROM enrichment_cache WHERE source = 'steam'"
    ).fetchall()
    conn.close()

    result = {}
    for r in rows:
        if r["data_json"]:
            try:
                d = json.loads(r["data_json"])
                if "_empty" not in d:
                    result[r["app_id"]] = d
            except (json.JSONDecodeError, TypeError):
                pass
    return result


def _extract_dx_version(steam_data: dict) -> tuple[int | None, bool, bool]:
    """Extract DX version, vulkan flag, opengl flag from Steam Store data.

    Steam Store cache stores the parsed SteamData model, not raw appdetails.
    pc_requirements HTML is NOT in our cache — we only have genres, categories, etc.
    But categories contain useful info like "VR Support", "Controller Support".
    """
    # Our Steam cache is SteamData model (developer, publisher, genres, categories, etc.)
    # It does NOT contain pc_requirements HTML.
    # We can only infer from categories/genres.
    return None, False, False


def add_dx_features_from_pics(X: pd.DataFrame, report_ids: list[str],
                               pics_cache: dict, report_info: dict) -> pd.DataFrame:
    """Since we don't have pc_requirements in cache, use PICS oslist as proxy."""
    # pc_requirements not available in our current Steam cache.
    # Would need to re-fetch with appdetails including requirements.
    # For now, skip — PICS oslist/osarch covers the most important signals.
    return X


# ── IRT + relabeling (reuse from experiment_13) ─────────────────────


def fit_irt_and_relabel(db_path, X_train, X_test, y_train_raw, train_rids, test_rids, relabel_ids):
    """Fit IRT, add features, apply contributor-aware relabeling."""
    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.irt import fit_irt, add_irt_features, contributor_aware_relabel

    conn = get_connection(db_path)
    theta, difficulty = fit_irt(conn)

    X_tr = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_te = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    y_tr, n = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn, theta)
    logger.info("IRT relabel: %d", n)

    conn.close()
    return X_tr, X_te, y_tr, theta


# ── Train + eval ─────────────────────────────────────────────────────


_s1_cache = {}


def train_and_eval(X_train, y_train, X_test, y_test, label=""):
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Reuse Stage 1 for same feature count
    n_feat = X_train.shape[1]
    if n_feat in _s1_cache:
        s1 = _s1_cache[n_feat]
    else:
        s1 = train_stage1(X_train, y_train, X_test, y_test)
        _s1_cache[n_feat] = s1

    s2, drops = train_stage2(X_train, y_train, X_test, y_test)
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
    print("Phase 14: Steam PICS features")
    print("=" * 70)

    # Load data
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

    # Load caches
    pics_cache = load_pics_cache(args.db)

    # Build report_info lookup (cached)
    from protondb_settings.db.connection import get_connection as gc
    conn2 = gc(args.db)
    report_info = {}
    for r in conn2.execute("SELECT id, app_id, gpu FROM reports").fetchall():
        report_info[r["id"]] = (r["app_id"], r["gpu"])
    conn2.close()

    # Apply IRT + relabeling (Phase 12-13 baseline)
    X_tr_irt, X_te_irt, y_tr, theta = fit_irt_and_relabel(
        args.db, X_train, X_test, y_train_raw, train_rids, test_rids, relabel_ids)

    results = []

    # ── Baseline: IRT only (no PICS) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("BASELINE: IRT features + contributor-aware relabel")
    print("=" * 70)
    r = train_and_eval(X_tr_irt.copy(), y_tr, X_te_irt.copy(), y_test, label="baseline_irt")
    results.append(r)
    print(f"  F1={r['f1_macro']:.4f} | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 14.1a: recommended_runtime only ──────────────────────────────
    print("\n" + "=" * 70)
    print("14.1a: recommended_runtime only (3 features)")
    print("=" * 70)
    X_tr_rt = add_pics_features(X_tr_irt, train_rids, pics_cache, report_info, mode="runtime")
    X_te_rt = add_pics_features(X_te_irt, test_rids, pics_cache, report_info, mode="runtime")
    r = train_and_eval(X_tr_rt, y_tr, X_te_rt, y_test, label="14.1a_runtime_only")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

    # ── 14.1b: All PICS features ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("14.1b: All PICS features (runtime + deck tests + review + osarch)")
    print("=" * 70)
    X_tr_all = add_pics_features(X_tr_irt, train_rids, pics_cache, report_info, mode="all")
    X_te_all = add_pics_features(X_te_irt, test_rids, pics_cache, report_info, mode="all")
    r = train_and_eval(X_tr_all, y_tr, X_te_all, y_test, label="14.1b_all_pics")
    results.append(r)
    delta = r["f1_macro"] - results[0]["f1_macro"]
    print(f"  F1={r['f1_macro']:.4f} (Δ={delta:+.4f}) | borked={r['borked_f1']:.3f} | "
          f"tink={r['tinkering_f1']:.3f} | oob={r['works_oob_f1']:.3f}")

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
