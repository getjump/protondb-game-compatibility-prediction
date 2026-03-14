#!/usr/bin/env python3
"""Experiment: GitHub Proton issues — version-specific interaction features.

Version-specific (game × proton):
  github_issue_this_version   — есть issue на эту (game, proton_major, family)
  github_severity_this_version — max severity на эту версию
  github_open_this_version    — есть открытый issue на эту версию

Version-independent (per-game):
  github_has_any_issue        — есть хотя бы один issue
  github_max_severity         — max severity по всем issues
  github_fixed_ratio          — completed / total

Tests:
  A) Version-independent only
  B) Version-specific only
  C) Both combined
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

# ── Severity classification ──

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


def classify_severity(title: str, body: str, labels: list[str]) -> int:
    label_names = [l if isinstance(l, str) else l.get("name", "") for l in labels]
    if any("Feature Request" in l for l in label_names):
        return 0
    if _NOT_BUG_RE.search(title):
        return 0
    text = f"{title} {body[:500]}"
    if _CRITICAL_RE.search(text):
        return 3
    if _MAJOR_RE.search(text) or any("Regression" in l for l in label_names):
        return 2
    if _MINOR_RE.search(text):
        return 1
    return 1


# ── Proton version extraction from issues ──

_RE_TITLE_APPID = re.compile(r"\((\d{4,8})\)\s*$")
_RE_BODY_APPID = re.compile(
    r"(?:Steam\s+)?App\s*ID\s*(?:of\s+the\s+game)?[:\s]+(\d{4,8})", re.IGNORECASE,
)
_RE_PROTON_VER = re.compile(
    r"Proton\s+version[:\s]+(.+?)(?:\n|$)", re.IGNORECASE
)
_RE_MAJOR_VER = re.compile(r"(\d+)\.\d+")


def _extract_app_id(title: str, body: str | None) -> int | None:
    m = _RE_TITLE_APPID.search(title)
    if m:
        return int(m.group(1))
    if body:
        m = _RE_BODY_APPID.search(body[:500])
        if m:
            return int(m.group(1))
    return None


def _extract_proton_info(body: str) -> list[tuple[int | None, str]]:
    """Extract (proton_major, family) from issue body.

    Returns list of (major_version, family) tuples.
    family: 'official', 'ge', 'experimental'
    """
    m = _RE_PROTON_VER.search(body[:1000])
    if not m:
        return []

    ver_text = m.group(1).strip().lower()
    results = []

    if "experimental" in ver_text:
        results.append((None, "experimental"))

    if "ge-proton" in ver_text or "ge proton" in ver_text:
        # Extract major from GE-Proton9-27 → 9
        ge_m = re.search(r"ge[- ]?proton(\d+)", ver_text, re.IGNORECASE)
        if ge_m:
            results.append((int(ge_m.group(1)), "ge"))
        else:
            results.append((None, "ge"))

    if "hotfix" in ver_text:
        results.append((None, "official"))

    # Official version: 9.0-4, 10.0, etc.
    official_m = re.findall(r"(\d+)\.\d+(?:-\d+)?", ver_text)
    for v in official_m:
        major = int(v)
        if major <= 15:  # sanity check
            results.append((major, "official"))

    if not results:
        # Try to get any version number
        any_m = _RE_MAJOR_VER.search(ver_text)
        if any_m and int(any_m.group(1)) <= 15:
            results.append((int(any_m.group(1)), "official"))

    return results


def _extract_report_proton_major(proton_version: str | None) -> int | None:
    """Extract major version from report's proton_version field (e.g. '9.0-4' → 9)."""
    if not proton_version:
        return None
    m = _RE_MAJOR_VER.match(proton_version)
    if m:
        return int(m.group(1))
    return None


def _extract_report_ge_major(custom_proton_version: str | None) -> int | None:
    """Extract major from GE version (e.g. 'GE-Proton9-27' → 9)."""
    if not custom_proton_version:
        return None
    m = re.search(r"GE[- ]?Proton(\d+)", custom_proton_version, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


# ── Build issue index ──

def build_issue_index(issues: list[dict]) -> tuple[
    dict[int, list[dict]],  # per-game issues
    dict[tuple[int, int | None, str], list[dict]],  # (app_id, major, family) → issues
]:
    """Build two indexes: per-game and per-(game, version)."""
    per_game: dict[int, list[dict]] = defaultdict(list)
    per_version: dict[tuple[int, int | None, str], list[dict]] = defaultdict(list)

    parsed_count = 0
    version_matched = 0

    for raw in issues:
        title = raw.get("title", "")
        body = raw.get("body", "") or ""
        app_id = _extract_app_id(title, body)
        if app_id is None:
            continue

        labels = raw.get("labels", [])
        severity = classify_severity(title, body, labels)
        state = raw.get("state", "")
        state_reason = (raw.get("stateReason") or "").upper()

        entry = {
            "severity": severity,
            "state": state,
            "is_open": state == "OPEN",
            "is_completed": state == "CLOSED" and state_reason == "COMPLETED",
        }

        per_game[app_id].append(entry)
        parsed_count += 1

        # Version-specific index
        proton_infos = _extract_proton_info(body)
        if proton_infos:
            version_matched += 1
            for major, family in proton_infos:
                per_version[(app_id, major, family)].append(entry)

    print(f"  Parsed {parsed_count} issues, {len(per_game)} games")
    print(f"  Version extracted: {version_matched}/{parsed_count} ({version_matched/parsed_count:.1%})")
    print(f"  Unique (game, major, family) keys: {len(per_version)}")

    return dict(per_game), dict(per_version)


# ── Feature builders ──

def make_version_independent_features(app_ids: list[int], per_game: dict) -> pd.DataFrame:
    records = []
    for app_id in app_ids:
        issues = per_game.get(app_id)
        if issues:
            severities = [i["severity"] for i in issues]
            completed = sum(1 for i in issues if i["is_completed"])
            records.append({
                "gh_has_any_issue": 1,
                "gh_max_severity": max(severities),
                "gh_fixed_ratio": completed / len(issues),
            })
        else:
            records.append({
                "gh_has_any_issue": 0,
                "gh_max_severity": np.nan,
                "gh_fixed_ratio": np.nan,
            })
    return pd.DataFrame(records)


def make_version_specific_features(
    app_ids: list[int],
    variants: list[str | None],
    proton_versions: list[str | None],
    custom_proton_versions: list[str | None],
    per_version: dict,
) -> pd.DataFrame:
    records = []
    for app_id, variant, pv, cpv in zip(app_ids, variants, proton_versions, custom_proton_versions):
        # Determine (major, family) for this report
        family = variant if variant in ("official", "ge", "experimental") else None
        if family == "official" or family is None:
            major = _extract_report_proton_major(pv)
            if family is None and major is not None:
                family = "official"
        elif family == "ge":
            major = _extract_report_ge_major(cpv)
        elif family == "experimental":
            major = None
        else:
            major = None

        # Lookup version-specific issues
        issues = None
        if family:
            issues = per_version.get((app_id, major, family))
            # Fallback: try without major version
            if not issues and major is not None:
                issues = per_version.get((app_id, None, family))

        if issues:
            severities = [i["severity"] for i in issues]
            records.append({
                "gh_issue_this_version": 1,
                "gh_severity_this_version": max(severities),
                "gh_open_this_version": int(any(i["is_open"] for i in issues)),
            })
        else:
            records.append({
                "gh_issue_this_version": 0,
                "gh_severity_this_version": np.nan,
                "gh_open_this_version": np.nan,
            })
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
    print("Loading ML data...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Get report-level proton info aligned with X
    rows = conn.execute(
        "SELECT app_id, verdict, verdict_oob, variant, proton_version, custom_proton_version "
        "FROM reports"
    ).fetchall()
    conn.close()

    app_ids_list = []
    variants_list = []
    pv_list = []
    cpv_list = []
    for row in rows:
        target = _compute_target(row["verdict"], row["verdict_oob"])
        if target is not None:
            app_ids_list.append(row["app_id"])
            variants_list.append(row["variant"])
            pv_list.append(row["proton_version"])
            cpv_list.append(row["custom_proton_version"])

    assert len(app_ids_list) == len(X)
    print(f"Loaded {len(X)} samples, {X.shape[1]} features")

    # Load GitHub issues
    print("\nLoading GitHub issues...")
    issues = json.load(open(ISSUES_PATH))
    per_game, per_version = build_issue_index(issues)

    # Coverage stats
    covered_game = sum(1 for aid in app_ids_list if aid in per_game)
    print(f"  Per-game coverage: {covered_game}/{len(app_ids_list)} ({covered_game/len(app_ids_list):.1%})")

    # Split
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Aligned arrays for train/test
    def _subset(lst, idx):
        return [lst[i] for i in idx]

    app_train = _subset(app_ids_list, train_idx)
    app_test = _subset(app_ids_list, test_idx)
    var_train = _subset(variants_list, train_idx)
    var_test = _subset(variants_list, test_idx)
    pv_train = _subset(pv_list, train_idx)
    pv_test = _subset(pv_list, test_idx)
    cpv_train = _subset(cpv_list, train_idx)
    cpv_test = _subset(cpv_list, test_idx)

    # Baseline
    f1_base, _ = train_and_eval(X_train, y_train, X_test, y_test, "Baseline")
    results = {"baseline": f1_base}

    # A) Version-independent features
    print("\n" + "#" * 60)
    print("# A: Version-independent features")
    print("#" * 60)

    vi_train = make_version_independent_features(app_train, per_game)
    vi_test = make_version_independent_features(app_test, per_game)

    X_train_a = pd.concat([X_train, vi_train], axis=1)
    X_test_a = pd.concat([X_test, vi_test], axis=1)
    f1_a, _ = train_and_eval(X_train_a, y_train, X_test_a, y_test, "A: version-independent")
    results["A_ver_indep"] = f1_a

    # B) Version-specific features
    print("\n" + "#" * 60)
    print("# B: Version-specific features")
    print("#" * 60)

    vs_train = make_version_specific_features(app_train, var_train, pv_train, cpv_train, per_version)
    vs_test = make_version_specific_features(app_test, var_test, pv_test, cpv_test, per_version)

    # Coverage of version-specific
    matched_train = (vs_train["gh_issue_this_version"] == 1).sum()
    matched_test = (vs_test["gh_issue_this_version"] == 1).sum()
    print(f"  Version-matched: train {matched_train}/{len(vs_train)} ({matched_train/len(vs_train):.1%}), "
          f"test {matched_test}/{len(vs_test)} ({matched_test/len(vs_test):.1%})")

    X_train_b = pd.concat([X_train, vs_train], axis=1)
    X_test_b = pd.concat([X_test, vs_test], axis=1)
    f1_b, _ = train_and_eval(X_train_b, y_train, X_test_b, y_test, "B: version-specific")
    results["B_ver_spec"] = f1_b

    # C) Both combined
    print("\n" + "#" * 60)
    print("# C: Both combined")
    print("#" * 60)

    X_train_c = pd.concat([X_train, vi_train, vs_train], axis=1)
    X_test_c = pd.concat([X_test, vi_test, vs_test], axis=1)
    f1_c, cascade_c = train_and_eval(X_train_c, y_train, X_test_c, y_test, "C: both combined")
    results["C_combined"] = f1_c

    # Feature importance
    print("\n" + "#" * 60)
    print("# GitHub feature importance (combined model)")
    print("#" * 60)

    for stage_name, model in [("Stage 1", cascade_c.stage1), ("Stage 2", cascade_c.stage2)]:
        features = model.feature_name_
        importances = model.feature_importances_
        gh_feats = [f for f in features if f.startswith("gh_")]
        if gh_feats:
            print(f"\n{stage_name}:")
            for fname in sorted(gh_feats, key=lambda f: -importances[features.index(f)]):
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
