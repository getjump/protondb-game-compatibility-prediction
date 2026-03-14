#!/usr/bin/env python3
"""Experiment: GitHub Proton issues text embeddings + severity features.

A) Keyword severity features (per-game aggregated)
B) Text embeddings from issue titles (per-game, averaged → SVD)
C) Text embeddings from issue title+body (per-game, averaged → SVD)
D) Best embeddings + severity combined
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.cascade import CascadeClassifier, train_stage1, train_stage2
from protondb_settings.ml.train import _build_feature_matrix, _compute_target

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
ISSUES_PATH = Path("data/github_proton_issues.json")
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]

# Severity classification keywords
_CRITICAL_RE = re.compile(
    r"crash|won'?t launch|doesn'?t start|black screen|segfault|can'?t start|"
    r"fails to launch|won'?t start|cannot start|unable to launch",
    re.IGNORECASE,
)
_MAJOR_RE = re.compile(
    r"broken|freeze|hang|unplayable|regression|not working|can'?t play|"
    r"doesn'?t work|fails to run|stopped working",
    re.IGNORECASE,
)
_MINOR_RE = re.compile(
    r"glitch|audio|stutter|flickering|font|artifact|visual|minor|"
    r"graphical|performance|rendering",
    re.IGNORECASE,
)
_NOT_BUG_RE = re.compile(
    r"feature request|question|wiki|suggestion|enhancement",
    re.IGNORECASE,
)

# Extract app_id from issue
_RE_TITLE_APPID = re.compile(r"\((\d{4,8})\)\s*$")
_RE_BODY_APPID = re.compile(
    r"(?:Steam\s+)?App\s*ID\s*(?:of\s+the\s+game)?[:\s]+(\d{4,8})", re.IGNORECASE,
)


def _extract_app_id(title: str, body: str | None) -> int | None:
    m = _RE_TITLE_APPID.search(title)
    if m:
        return int(m.group(1))
    if body:
        m = _RE_BODY_APPID.search(body[:500])
        if m:
            return int(m.group(1))
    return None


def classify_severity(title: str, body: str, labels: list[str]) -> int:
    """0=not_bug, 1=minor, 2=major, 3=critical."""
    label_names = [l if isinstance(l, str) else l.get("name", "") for l in labels]

    if any("Feature Request" in l for l in label_names):
        return 0
    if _NOT_BUG_RE.search(title):
        return 0

    text = f"{title} {body[:500]}"
    has_regression = any("Regression" in l for l in label_names)

    if _CRITICAL_RE.search(text):
        return 3
    if _MAJOR_RE.search(text) or has_regression:
        return 2
    if _MINOR_RE.search(text):
        return 1
    return 1  # default minor


def load_issues() -> list[dict]:
    with open(ISSUES_PATH) as f:
        return json.load(f)


def build_per_game_data(issues: list[dict]) -> dict[int, dict]:
    """Build per-game aggregated data from raw issues."""
    by_app: dict[int, list[dict]] = defaultdict(list)

    for raw in issues:
        title = raw.get("title", "")
        body = raw.get("body", "") or ""
        app_id = _extract_app_id(title, body)
        if app_id is None:
            continue

        labels = raw.get("labels", [])
        severity = classify_severity(title, body, labels)

        by_app[app_id].append({
            "title": title,
            "body": body[:500],
            "severity": severity,
            "state": raw.get("state", ""),
            "labels": labels,
        })

    result = {}
    for app_id, app_issues in by_app.items():
        severities = [i["severity"] for i in app_issues]
        open_issues = [i for i in app_issues if i["state"] == "OPEN"]

        result[app_id] = {
            "issues": app_issues,
            "issue_count": len(app_issues),
            "max_severity": max(severities),
            "critical_count": sum(1 for s in severities if s == 3),
            "major_count": sum(1 for s in severities if s == 2),
            "has_open_critical": any(i["severity"] == 3 for i in open_issues),
            "has_regression": any(
                any("Regression" in (l if isinstance(l, str) else l.get("name", ""))
                    for l in i["labels"])
                for i in app_issues
            ),
            "open_count": len(open_issues),
            "fixed_ratio": (
                sum(1 for i in app_issues if i["state"] == "CLOSED") / len(app_issues)
                if app_issues else 0
            ),
            # Concatenated text for embeddings
            "all_titles": " ".join(i["title"] for i in app_issues),
            "all_text": " ".join(f"{i['title']}. {i['body']}" for i in app_issues),
        }

    return result


def make_severity_features(app_ids_series: pd.Series, game_data: dict) -> pd.DataFrame:
    """Build severity features aligned with report app_ids."""
    records = []
    for app_id in app_ids_series:
        gd = game_data.get(app_id)
        if gd:
            records.append({
                "gh_has_issue": 1,
                "gh_max_severity": gd["max_severity"],
                "gh_critical_count": gd["critical_count"],
                "gh_major_count": gd["major_count"],
                "gh_has_open_critical": int(gd["has_open_critical"]),
                "gh_has_regression": int(gd["has_regression"]),
                "gh_issue_count": gd["issue_count"],
                "gh_open_count": gd["open_count"],
                "gh_fixed_ratio": gd["fixed_ratio"],
            })
        else:
            records.append({
                "gh_has_issue": 0,
                "gh_max_severity": np.nan,
                "gh_critical_count": np.nan,
                "gh_major_count": np.nan,
                "gh_has_open_critical": np.nan,
                "gh_has_regression": np.nan,
                "gh_issue_count": np.nan,
                "gh_open_count": np.nan,
                "gh_fixed_ratio": np.nan,
            })
    return pd.DataFrame(records)


def build_game_text_embeddings(
    game_data: dict,
    st_model: SentenceTransformer,
    text_field: str,
    n_components: int = 16,
) -> tuple[dict[int, np.ndarray], TruncatedSVD]:
    """Build per-game text embeddings via sentence-transformers + SVD."""
    app_ids = sorted(game_data.keys())
    texts = [game_data[aid][text_field] for aid in app_ids]

    print(f"  Encoding {len(texts)} game texts ({text_field})...")
    raw_embs = st_model.encode(texts, batch_size=256, show_progress_bar=True)

    # SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(raw_embs)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD {n_components} dims, explained variance: {explained:.3f}")

    emb_lookup = {aid: reduced[i] for i, aid in enumerate(app_ids)}
    return emb_lookup, svd


def make_emb_features(
    app_ids_series: pd.Series,
    emb_lookup: dict[int, np.ndarray],
    n_components: int,
    prefix: str,
) -> pd.DataFrame:
    """Build embedding features aligned with report app_ids."""
    records = []
    for app_id in app_ids_series:
        emb = emb_lookup.get(app_id)
        if emb is not None:
            records.append({f"{prefix}_{i}": emb[i] for i in range(n_components)})
        else:
            records.append({f"{prefix}_{i}": np.nan for i in range(n_components)})
    return pd.DataFrame(records)


def train_and_eval(X_train, y_train, X_test, y_test, label: str):
    s1 = train_stage1(X_train, y_train, X_test, y_test)
    s2, s2_dropped = train_stage2(X_train, y_train, X_test, y_test)
    cascade = CascadeClassifier(s1, s2, s2_dropped)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  F1 macro: {f1:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES_3, digits=4))
    return f1, cascade


def main():
    # Load ML data
    print("Loading ML data...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Get app_ids aligned with X
    rows = conn.execute("SELECT app_id, verdict, verdict_oob FROM reports").fetchall()
    app_ids_list = []
    for row in rows:
        target = _compute_target(row["verdict"], row["verdict_oob"])
        if target is not None:
            app_ids_list.append(row["app_id"])
    conn.close()

    assert len(app_ids_list) == len(X)
    app_ids_series = pd.Series(app_ids_list)

    print(f"Loaded {len(X)} samples, {X.shape[1]} features")

    # Load GitHub issues
    print("Loading GitHub issues...")
    issues = load_issues()
    game_data = build_per_game_data(issues)
    print(f"  {len(issues)} issues → {len(game_data)} games")

    # Coverage
    covered = sum(1 for aid in app_ids_list if aid in game_data)
    print(f"  Report coverage: {covered}/{len(app_ids_list)} ({covered/len(app_ids_list):.1%})")

    # Split
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]
    app_train = app_ids_series.iloc[train_idx].reset_index(drop=True)
    app_test = app_ids_series.iloc[test_idx].reset_index(drop=True)

    # Baseline
    f1_base, _ = train_and_eval(X_train, y_train, X_test, y_test, "Baseline")
    results = {"baseline": f1_base}

    # A) Severity features
    print("\n" + "#" * 60)
    print("# A: Severity features")
    print("#" * 60)

    sev_train = make_severity_features(app_train, game_data)
    sev_test = make_severity_features(app_test, game_data)

    X_train_a = pd.concat([X_train, sev_train], axis=1)
    X_test_a = pd.concat([X_test, sev_test], axis=1)
    f1_a, _ = train_and_eval(X_train_a, y_train, X_test_a, y_test, "A: severity features")
    results["A_severity"] = f1_a

    # Load sentence transformer
    print("\nLoading sentence-transformers model...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # B) Title embeddings
    print("\n" + "#" * 60)
    print("# B: Title embeddings (SVD 16)")
    print("#" * 60)

    title_lookup, _ = build_game_text_embeddings(game_data, st_model, "all_titles", 16)
    title_train = make_emb_features(app_train, title_lookup, 16, "gh_title_emb")
    title_test = make_emb_features(app_test, title_lookup, 16, "gh_title_emb")

    X_train_b = pd.concat([X_train, title_train], axis=1)
    X_test_b = pd.concat([X_test, title_test], axis=1)
    f1_b, _ = train_and_eval(X_train_b, y_train, X_test_b, y_test, "B: title embeddings SVD16")
    results["B_title_emb16"] = f1_b

    # C) Title+body embeddings
    print("\n" + "#" * 60)
    print("# C: Title+body embeddings (SVD 16)")
    print("#" * 60)

    text_lookup, _ = build_game_text_embeddings(game_data, st_model, "all_text", 16)
    text_train = make_emb_features(app_train, text_lookup, 16, "gh_text_emb")
    text_test = make_emb_features(app_test, text_lookup, 16, "gh_text_emb")

    X_train_c = pd.concat([X_train, text_train], axis=1)
    X_test_c = pd.concat([X_test, text_test], axis=1)
    f1_c, _ = train_and_eval(X_train_c, y_train, X_test_c, y_test, "C: title+body embeddings SVD16")
    results["C_text_emb16"] = f1_c

    # D) Best embeddings + severity
    print("\n" + "#" * 60)
    print("# D: Best embeddings + severity combined")
    print("#" * 60)

    # Pick best embedding
    best_emb_name = "B" if f1_b >= f1_c else "C"
    best_emb_train = title_train if f1_b >= f1_c else text_train
    best_emb_test = title_test if f1_b >= f1_c else text_test

    X_train_d = pd.concat([X_train, sev_train, best_emb_train], axis=1)
    X_test_d = pd.concat([X_test, sev_test, best_emb_test], axis=1)
    f1_d, cascade_d = train_and_eval(X_train_d, y_train, X_test_d, y_test,
                                      f"D: severity + {best_emb_name} embeddings")
    results["D_sev_emb"] = f1_d

    # Feature importance
    print("\n" + "#" * 60)
    print("# GitHub feature importance (best model)")
    print("#" * 60)

    for stage_name, model in [("Stage 1", cascade_d.stage1), ("Stage 2", cascade_d.stage2)]:
        features = model.feature_name_
        importances = model.feature_importances_
        gh_feats = [f for f in features if f.startswith("gh_")]
        if gh_feats:
            print(f"\n{stage_name} — github features by importance:")
            for fname in sorted(gh_feats, key=lambda f: -importances[features.index(f)])[:15]:
                idx = features.index(fname)
                print(f"  {fname:35s} {importances[idx]:10.1f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        delta = f1 - f1_base
        marker = " <<<" if delta > 0.002 else ""
        print(f"  {name:25s}  F1={f1:.4f}  Δ={delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
