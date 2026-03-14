#!/usr/bin/env python3
"""Experiment: Text-based features (PLAN_ML_5 experiments 4, 5, 6).

Exp 4: Text meta-features (Group D) — lengths, counts of filled fields
Exp 5: Keyword features (Group E) — regex crash/fix/perfect/env_var mentions
Exp 6: Aggregated text features (Group F) — per-game keyword/meta aggregation

Baseline: cascade with current features (F1 ≈ 0.593)
"""

from __future__ import annotations

import logging
import re
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]


# ── Keyword patterns ──────────────────────────────────────────────

CRASH_PATTERN = re.compile(
    r"\b(crash|crashes|crashing|segfault|sigsegv|sigabrt|freeze|freezes|freezing"
    r"|hang|hangs|hanging|won.?t\s+start|won.?t\s+launch|doesn.?t\s+start"
    r"|doesn.?t\s+launch|fail\s+to\s+start|fail\s+to\s+launch|broken|unplayable)\b",
    re.IGNORECASE,
)

FIX_PATTERN = re.compile(
    r"\b(fix|fixed|fixes|workaround|tweak|tweaked|solved|solution"
    r"|resolved|launch\s+option|protontricks|winetricks"
    r"|you\s+need\s+to|have\s+to|must\s+set|try\s+setting)\b",
    re.IGNORECASE,
)

PERFECT_PATTERN = re.compile(
    r"\b(perfect|flawless|no\s+issues|works?\s+great|works?\s+perfectly"
    r"|works?\s+fine|out\s+of\s+the\s+box|smooth|excellent|without\s+any\s+issue"
    r"|no\s+problems|runs?\s+great|runs?\s+perfectly|just\s+works)\b",
    re.IGNORECASE,
)

PROTON_VERSION_PATTERN = re.compile(
    r"\b(proton\s*\d|ge[-\s]?proton|proton[-\s]?ge|proton\s+experimental"
    r"|proton\s+hotfix|proton\s+[\d.]+)\b",
    re.IGNORECASE,
)

ENV_VAR_PATTERN = re.compile(
    r"\b[A-Z_]{3,}=[^\s]+",
)

PERFORMANCE_PATTERN = re.compile(
    r"\b(lag|laggy|lagging|stutter|stutters|stuttering|fps|frame.?rate"
    r"|slow|sluggish|performance|choppy)\b",
    re.IGNORECASE,
)

NEGATIVE_WORDS = re.compile(
    r"\b(broken|unplayable|garbage|terrible|horrible|awful|worst|useless"
    r"|waste|disappointed|frustrating|unbearable|atrocious|dreadful)\b",
    re.IGNORECASE,
)

POSITIVE_WORDS = re.compile(
    r"\b(great|excellent|smooth|perfect|fantastic|amazing|wonderful"
    r"|awesome|brilliant|superb|stellar|flawless|solid|stable)\b",
    re.IGNORECASE,
)


def count_matches(pattern: re.Pattern, text) -> int:
    if not isinstance(text, str):
        return 0
    return len(pattern.findall(text))


def has_match(pattern: re.Pattern, text) -> int:
    if not isinstance(text, str):
        return 0
    return 1 if pattern.search(text) else 0


# ── Data loading ──────────────────────────────────────────────────

def load_data():
    """Load base feature matrix + text fields from DB."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    # Load text fields and report IDs for joining
    print("Loading text fields from reports...")
    rows = conn.execute("""
        SELECT id, app_id,
            concluding_notes,
            notes_verdict,
            notes_extra,
            notes_customizations,
            notes_audio_faults,
            notes_graphical_faults,
            notes_performance_faults,
            notes_stability_faults,
            notes_windowing_faults,
            notes_input_faults,
            notes_significant_bugs,
            notes_save_game_faults,
            notes_concluding_notes,
            verdict, verdict_oob
        FROM reports
    """).fetchall()
    conn.close()

    # Build text data aligned with feature matrix
    # We need to match: only reports with valid targets in same order as _build_feature_matrix
    from protondb_settings.ml.train import _compute_target

    text_records = []
    app_ids = []
    for row in rows:
        target = _compute_target(row["verdict"], row["verdict_oob"])
        if target is None:
            continue

        fault_fields = [
            row["notes_audio_faults"],
            row["notes_graphical_faults"],
            row["notes_performance_faults"],
            row["notes_stability_faults"],
            row["notes_windowing_faults"],
            row["notes_input_faults"],
            row["notes_significant_bugs"],
            row["notes_save_game_faults"],
        ]

        all_text = " ".join(
            t for t in [
                row["concluding_notes"],
                row["notes_verdict"],
                row["notes_extra"],
                row["notes_customizations"],
                row["notes_concluding_notes"],
            ] + fault_fields
            if t
        )

        text_records.append({
            "concluding_notes": row["concluding_notes"],
            "notes_verdict": row["notes_verdict"],
            "notes_extra": row["notes_extra"],
            "notes_customizations": row["notes_customizations"],
            "all_text": all_text if all_text.strip() else None,
            "fault_fields_filled": sum(1 for f in fault_fields if f and f.strip()),
        })
        app_ids.append(row["app_id"])

    assert len(text_records) == len(X), f"Mismatch: {len(text_records)} text vs {len(X)} features"

    text_df = pd.DataFrame(text_records)
    app_id_series = pd.Series(app_ids, name="app_id")

    return X, y, timestamps, text_df, app_id_series


# ── Feature builders ──────────────────────────────────────────────

def build_group_d_features(text_df: pd.DataFrame) -> pd.DataFrame:
    """Group D: text meta-features (per-report)."""
    features = pd.DataFrame(index=text_df.index)

    features["has_concluding_notes"] = text_df["concluding_notes"].notna().astype(int)
    features["concluding_notes_length"] = text_df["concluding_notes"].fillna("").str.len()
    features["fault_notes_count"] = text_df["fault_fields_filled"]
    features["has_customization_notes"] = text_df["notes_customizations"].notna().astype(int)
    features["total_notes_length"] = text_df["all_text"].fillna("").str.len()

    return features


def build_group_e_features(text_df: pd.DataFrame) -> pd.DataFrame:
    """Group E: keyword features (per-report)."""
    features = pd.DataFrame(index=text_df.index)

    all_text = text_df["all_text"]

    features["mentions_crash"] = all_text.apply(lambda t: has_match(CRASH_PATTERN, t))
    features["mentions_fix"] = all_text.apply(lambda t: has_match(FIX_PATTERN, t))
    features["mentions_perfect"] = all_text.apply(lambda t: has_match(PERFECT_PATTERN, t))
    features["mentions_proton_version"] = all_text.apply(lambda t: has_match(PROTON_VERSION_PATTERN, t))
    features["mentions_env_var"] = all_text.apply(lambda t: has_match(ENV_VAR_PATTERN, t))
    features["mentions_performance"] = all_text.apply(lambda t: has_match(PERFORMANCE_PATTERN, t))
    features["sentiment_negative_words"] = all_text.apply(lambda t: count_matches(NEGATIVE_WORDS, t))
    features["sentiment_positive_words"] = all_text.apply(lambda t: count_matches(POSITIVE_WORDS, t))

    return features


def build_group_f_features(
    text_df: pd.DataFrame,
    app_ids: pd.Series,
    group_e: pd.DataFrame,
    group_d: pd.DataFrame,
) -> pd.DataFrame:
    """Group F: aggregated text features (per-game)."""
    # Build per-game aggregation dataframe
    agg_df = pd.DataFrame({
        "app_id": app_ids,
        "has_notes": group_d["has_concluding_notes"],
        "notes_length": group_d["concluding_notes_length"],
        "mentions_crash": group_e["mentions_crash"],
        "mentions_fix": group_e["mentions_fix"],
        "has_faults": (group_d["fault_notes_count"] > 0).astype(int),
    })

    game_agg = agg_df.groupby("app_id").agg(
        pct_reports_with_notes=("has_notes", "mean"),
        avg_notes_length=("notes_length", "mean"),
        pct_reports_mention_crash=("mentions_crash", "mean"),
        pct_reports_mention_fix=("mentions_fix", "mean"),
        pct_reports_with_faults=("has_faults", "mean"),
        game_report_count=("has_notes", "count"),
    ).reset_index()

    # Map back to per-report
    features = app_ids.to_frame("app_id").merge(game_agg, on="app_id", how="left")
    features = features.drop(columns=["app_id"]).reset_index(drop=True)

    return features


# ── Training helper ───────────────────────────────────────────────

def train_cascade_and_evaluate(X_train, y_train, X_test, y_test, label: str):
    """Train cascade and return F1 macro."""
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


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    X, y, timestamps, text_df, app_ids = load_data()
    print(f"Loaded {len(X)} samples, {X.shape[1]} features")
    print(f"Text coverage: concluding_notes={text_df['concluding_notes'].notna().mean():.1%}, "
          f"all_text={text_df['all_text'].notna().mean():.1%}")

    # Split
    X_train, X_test, y_train, y_test = _time_based_split(X, y, timestamps)
    n_train = len(X_train)
    n_test = len(X_test)
    text_train = text_df.iloc[:n_train + n_test]  # need full for agg
    app_ids_full = app_ids.iloc[:n_train + n_test]

    # The split indices
    sorted_idx = np.argsort(timestamps)
    split_point = int(len(sorted_idx) * 0.8)
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    text_df_sorted_train = text_df.iloc[train_idx].reset_index(drop=True)
    text_df_sorted_test = text_df.iloc[test_idx].reset_index(drop=True)
    app_ids_sorted_train = app_ids.iloc[train_idx].reset_index(drop=True)
    app_ids_sorted_test = app_ids.iloc[test_idx].reset_index(drop=True)

    # ── Baseline ──
    print("\n" + "=" * 60)
    print("BASELINE (current features)")
    print("=" * 60)
    f1_baseline, _ = train_cascade_and_evaluate(X_train, y_train, X_test, y_test, "Baseline")

    results = {"baseline": f1_baseline}

    # ── Experiment 4: Group D (text meta) ──
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 4: Group D — Text meta-features")
    print("#" * 60)

    d_train = build_group_d_features(text_df_sorted_train)
    d_test = build_group_d_features(text_df_sorted_test)

    X_train_d = pd.concat([X_train.reset_index(drop=True), d_train], axis=1)
    X_test_d = pd.concat([X_test.reset_index(drop=True), d_test], axis=1)

    print(f"Added {d_train.shape[1]} features: {list(d_train.columns)}")
    f1_d, _ = train_cascade_and_evaluate(X_train_d, y_train, X_test_d, y_test, "Exp 4: Group D")
    results["exp4_group_d"] = f1_d

    # ── Experiment 5: Group E (keywords) ──
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 5: Group E — Keyword features")
    print("#" * 60)

    e_train = build_group_e_features(text_df_sorted_train)
    e_test = build_group_e_features(text_df_sorted_test)

    X_train_e = pd.concat([X_train.reset_index(drop=True), e_train], axis=1)
    X_test_e = pd.concat([X_test.reset_index(drop=True), e_test], axis=1)

    print(f"Added {e_train.shape[1]} features: {list(e_train.columns)}")
    f1_e, _ = train_cascade_and_evaluate(X_train_e, y_train, X_test_e, y_test, "Exp 5: Group E")
    results["exp5_group_e"] = f1_e

    # ── Experiment 5b: Group D + E combined ──
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 5b: Group D + E combined")
    print("#" * 60)

    X_train_de = pd.concat([X_train.reset_index(drop=True), d_train, e_train], axis=1)
    X_test_de = pd.concat([X_test.reset_index(drop=True), d_test, e_test], axis=1)

    f1_de, _ = train_cascade_and_evaluate(X_train_de, y_train, X_test_de, y_test, "Exp 5b: Group D+E")
    results["exp5b_group_de"] = f1_de

    # ── Experiment 6: Group F (aggregated text, per-game) ──
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 6: Group F — Aggregated text features")
    print("#" * 60)

    # Build group E and D on full dataset first, then split
    d_full = build_group_d_features(text_df)
    e_full = build_group_e_features(text_df)
    f_full = build_group_f_features(text_df, app_ids, e_full, d_full)

    f_train = f_full.iloc[train_idx].reset_index(drop=True)
    f_test = f_full.iloc[test_idx].reset_index(drop=True)

    X_train_f = pd.concat([X_train.reset_index(drop=True), f_train], axis=1)
    X_test_f = pd.concat([X_test.reset_index(drop=True), f_test], axis=1)

    print(f"Added {f_train.shape[1]} features: {list(f_train.columns)}")
    f1_f, _ = train_cascade_and_evaluate(X_train_f, y_train, X_test_f, y_test, "Exp 6: Group F")
    results["exp6_group_f"] = f1_f

    # ── Experiment 6b: All groups D + E + F ──
    print("\n\n" + "#" * 60)
    print("# EXPERIMENT 6b: All groups D + E + F combined")
    print("#" * 60)

    X_train_all = pd.concat([X_train.reset_index(drop=True), d_train, e_train, f_train], axis=1)
    X_test_all = pd.concat([X_test.reset_index(drop=True), d_test, e_test, f_test], axis=1)

    f1_all, cascade_all = train_cascade_and_evaluate(
        X_train_all, y_train, X_test_all, y_test, "Exp 6b: All groups D+E+F"
    )
    results["exp6b_all_def"] = f1_all

    # ── Feature importance for new features ──
    print("\n\n" + "#" * 60)
    print("# Feature importance (new features in Stage 2)")
    print("#" * 60)

    new_feature_names = list(d_train.columns) + list(e_train.columns) + list(f_train.columns)
    s2_features = cascade_all.stage2.feature_name_
    s2_importances = cascade_all.stage2.feature_importances_

    print("\nStage 2 — new features by importance (gain):")
    for fname in new_feature_names:
        if fname in s2_features:
            idx = s2_features.index(fname)
            print(f"  {fname:40s} {s2_importances[idx]:10.1f}")

    s1_features = cascade_all.stage1.feature_name_
    s1_importances = cascade_all.stage1.feature_importances_

    print("\nStage 1 — new features by importance (gain):")
    for fname in new_feature_names:
        if fname in s1_features:
            idx = s1_features.index(fname)
            print(f"  {fname:40s} {s1_importances[idx]:10.1f}")

    # ── Summary ──
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
        delta = f1 - f1_baseline
        marker = " <<<" if delta > 0.002 else ""
        print(f"  {name:25s}  F1={f1:.4f}  Δ={delta:+.4f}{marker}")


if __name__ == "__main__":
    main()
