#!/usr/bin/env python3
"""Experiment: Deck Verified features (deck_status + deck_tests_json).

Tests:
A) deck_status only (categorical 0-3)
B) deck_tests parsed (bool flags per test type + counts)
C) deck_status + deck_tests combined
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
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
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]

# Key test tokens to extract as features
DECK_TEST_FEATURES = {
    "deck_test_controller_full": "DefaultControllerConfigFullyFunctional",
    "deck_test_controller_bad": "DefaultControllerConfigNotFullyFunctional",
    "deck_test_performant": "DefaultConfigurationIsPerformant",
    "deck_test_not_performant": "DefaultConfigurationIsNotPerformant",
    "deck_test_text_legible": "InterfaceTextIsLegible",
    "deck_test_text_not_legible": "InterfaceTextIsNotLegible",
    "deck_test_resolution_not_supported": "NativeResolutionNotSupported",
    "deck_test_resolution_not_default": "NativeResolutionNotDefault",
    "deck_test_glyphs_match": "ControllerGlyphsMatchDeckDevice",
    "deck_test_glyphs_no_match": "ControllerGlyphsDoNotMatchDeckDevice",
    "deck_test_anticheat_unsupported": "UnsupportedAntiCheat",
    "deck_test_steamos_unsupported": "SteamOSDoesNotSupport",
    "deck_test_launcher_issues": "LauncherInteractionIssues",
    "deck_test_no_keyboard": "NotFullyFunctionalWithoutExternalKeyboard",
    "deck_test_gamepad_not_default": "GamepadNotEnabledByDefault",
    "deck_test_no_exit_clean": "GameOrLauncherDoesntExitCleanly",
    "deck_test_sleep_broken": "ResumeFromSleepNotFunctional",
    "deck_test_display_issues": "DisplayOutputHasNonblockingIssues",
    "deck_test_graphics_unsupported": "UnsupportedGraphicsPerformance",
}


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Load deck data aligned with reports
    rows = conn.execute("""
        SELECT r.app_id, r.verdict, r.verdict_oob,
               gm.deck_status, gm.deck_tests_json
        FROM reports r
        LEFT JOIN game_metadata gm ON r.app_id = gm.app_id
    """).fetchall()
    conn.close()

    deck_statuses = []
    deck_tests_list = []
    for row in rows:
        target = _compute_target(row["verdict"], row["verdict_oob"])
        if target is None:
            continue
        deck_statuses.append(row["deck_status"])
        deck_tests_list.append(row["deck_tests_json"])

    assert len(deck_statuses) == len(X), f"Mismatch: {len(deck_statuses)} vs {len(X)}"
    return X, y, timestamps, deck_statuses, deck_tests_list


def make_deck_status_features(statuses: list) -> pd.DataFrame:
    """deck_status as numeric feature."""
    return pd.DataFrame({"deck_status": [s if s is not None else np.nan for s in statuses]})


def make_deck_tests_features(tests_list: list) -> pd.DataFrame:
    """Parse deck_tests_json into boolean flags + aggregate counts."""
    records = []
    for tests_json in tests_list:
        record = {}
        if tests_json:
            try:
                tests = json.loads(tests_json)
            except (json.JSONDecodeError, TypeError):
                tests = []

            tokens = [t.get("loc_token", "") for t in tests]
            display_types = [t.get("display_type", 0) for t in tests]

            # Bool flags for specific tests
            for feat_name, token_part in DECK_TEST_FEATURES.items():
                record[feat_name] = int(any(token_part in tok for tok in tokens))

            # Aggregate counts
            record["deck_test_count"] = len(tests)
            record["deck_test_pass_count"] = sum(1 for d in display_types if d == 4)
            record["deck_test_warn_count"] = sum(1 for d in display_types if d == 3)
            record["deck_test_fail_count"] = sum(1 for d in display_types if d == 2)
        else:
            for feat_name in DECK_TEST_FEATURES:
                record[feat_name] = np.nan
            record["deck_test_count"] = np.nan
            record["deck_test_pass_count"] = np.nan
            record["deck_test_warn_count"] = np.nan
            record["deck_test_fail_count"] = np.nan

        records.append(record)

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
    print("Loading data...")
    X, y, timestamps, deck_statuses, deck_tests_list = load_data()
    print(f"Loaded {len(X)} samples, {X.shape[1]} features")

    # Coverage
    has_status = sum(1 for s in deck_statuses if s is not None)
    has_tests = sum(1 for t in deck_tests_list if t)
    print(f"deck_status coverage: {has_status}/{len(deck_statuses)} ({has_status/len(deck_statuses):.1%})")
    print(f"deck_tests coverage: {has_tests}/{len(deck_tests_list)} ({has_tests/len(deck_tests_list):.1%})")

    # Split
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Baseline
    f1_base, _ = train_and_eval(X_train, y_train, X_test, y_test, "Baseline (current features)")
    results = {"baseline": f1_base}

    # Prepare deck features
    status_df = make_deck_status_features(deck_statuses)
    tests_df = make_deck_tests_features(deck_tests_list)

    # A) deck_status only
    status_train = status_df.iloc[train_idx].reset_index(drop=True)
    status_test = status_df.iloc[test_idx].reset_index(drop=True)

    X_train_a = pd.concat([X_train, status_train], axis=1)
    X_test_a = pd.concat([X_test, status_test], axis=1)
    f1_a, _ = train_and_eval(X_train_a, y_train, X_test_a, y_test, "A: deck_status only")
    results["A_deck_status"] = f1_a

    # B) deck_tests only
    tests_train = tests_df.iloc[train_idx].reset_index(drop=True)
    tests_test = tests_df.iloc[test_idx].reset_index(drop=True)

    X_train_b = pd.concat([X_train, tests_train], axis=1)
    X_test_b = pd.concat([X_test, tests_test], axis=1)
    f1_b, _ = train_and_eval(X_train_b, y_train, X_test_b, y_test, "B: deck_tests only")
    results["B_deck_tests"] = f1_b

    # C) deck_status + deck_tests
    X_train_c = pd.concat([X_train, status_train, tests_train], axis=1)
    X_test_c = pd.concat([X_test, status_test, tests_test], axis=1)
    f1_c, cascade_c = train_and_eval(X_train_c, y_train, X_test_c, y_test, "C: deck_status + deck_tests")
    results["C_status_tests"] = f1_c

    # Feature importance for deck features
    print("\n" + "#" * 60)
    print("# Deck feature importance (best model)")
    print("#" * 60)

    best_cascade = cascade_c
    for stage_name, model in [("Stage 1", best_cascade.stage1), ("Stage 2", best_cascade.stage2)]:
        features = model.feature_name_
        importances = model.feature_importances_
        deck_feats = [f for f in features if f.startswith("deck_")]
        if deck_feats:
            print(f"\n{stage_name} — deck features by importance:")
            for fname in sorted(deck_feats, key=lambda f: -importances[features.index(f)]):
                idx = features.index(fname)
                print(f"  {fname:40s} {importances[idx]:10.1f}")

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
