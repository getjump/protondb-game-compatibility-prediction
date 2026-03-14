#!/usr/bin/env python3
"""Experiment: Label noise analysis and mitigation (Steps 8-10 from PLAN_ML_4.md).

Step 8: Cleanlab — detect and remove noisy labels (class-balanced)
Step 9: Confident Learning — weight samples by game consensus
Step 10: Regression — continuous score instead of classification
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).parent.parent))

from protondb_settings.db.migrations import ensure_schema
from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
from protondb_settings.ml.features.embeddings import load_embeddings
from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES
from protondb_settings.ml.models.cascade import train_stage1, CascadeClassifier

logging.basicConfig(level=logging.WARNING)

DB_PATH = Path("data/protondb.db")
EMB_PATH = Path("data/embeddings.npz")
CLASS_NAMES_3 = ["borked", "needs_tinkering", "works_oob"]
STAGE2_DROP = ["report_age_days"]


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    emb_data = load_embeddings(EMB_PATH)
    emb_data["n_components_gpu"] = emb_data["gpu_embeddings"].shape[1] if emb_data["gpu_embeddings"].size else 0
    emb_data["n_components_cpu"] = emb_data["cpu_embeddings"].shape[1] if emb_data["cpu_embeddings"].size else 0

    X, y, timestamps, label_maps = _build_feature_matrix(conn, emb_data)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Load per-game pct_works_oob and avg_verdict_score for Steps 9/10
    game_stats = pd.read_sql("""
        SELECT app_id,
            AVG(CASE WHEN verdict_oob = 'yes' THEN 1.0 ELSE 0.0 END) as pct_works_oob,
            AVG(CASE
                WHEN verdict_oob = 'yes' THEN 1.0
                WHEN verdict_oob = 'no' AND verdict = 'yes' THEN 0.5
                WHEN verdict = 'no' THEN 0.0
                WHEN verdict = 'yes' THEN 0.5
                ELSE NULL
            END) as avg_verdict_score
        FROM reports GROUP BY app_id
    """, conn)
    game_stats_map = game_stats.set_index("app_id")

    # Get app_ids for each row
    app_ids = pd.read_sql("""
        SELECT app_id FROM reports ORDER BY timestamp ASC
    """, conn)["app_id"].values

    X_train, X_test, y_train, y_test = _time_based_split(X, y, timestamps, 0.2)
    split_idx = len(X_train)
    app_ids_train = app_ids[:split_idx]
    app_ids_test = app_ids[split_idx:]

    # Map pct_works_oob/avg_verdict_score to each sample
    pct_oob_train = np.array([game_stats_map.loc[aid, "pct_works_oob"]
                              if aid in game_stats_map.index else 0.5
                              for aid in app_ids_train])
    pct_oob_test = np.array([game_stats_map.loc[aid, "pct_works_oob"]
                             if aid in game_stats_map.index else 0.5
                             for aid in app_ids_test])
    avg_score_train = np.array([game_stats_map.loc[aid, "avg_verdict_score"]
                                if aid in game_stats_map.index else 0.5
                                for aid in app_ids_train])

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_test.columns]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    conn.close()
    return (X_train, X_test, y_train, y_test, cat_cols,
            pct_oob_train, pct_oob_test, avg_score_train)


def cascade_evaluate(s1_model, s2_model, s2_dropped, X_test, y_test):
    cascade = CascadeClassifier(s1_model, s2_model, s2_dropped)
    y_pred = cascade.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    borked_r = (y_pred[y_test == 0] == 0).mean() if (y_test == 0).any() else 0
    borked_p = (y_test[y_pred == 0] == 0).mean() if (y_pred == 0).any() else 0
    oob_r = (y_pred[y_test == 2] == 2).mean() if (y_test == 2).any() else 0
    oob_p = (y_test[y_pred == 2] == 2).mean() if (y_pred == 2).any() else 0
    return {"f1": f1, "borked_r": borked_r, "borked_p": borked_p,
            "oob_r": oob_r, "oob_p": oob_p, "y_pred": y_pred}


def train_s2(X_tr, y_tr, X_te, y_te, cat_cols, class_weight=None, sample_weight=None):
    if class_weight is None:
        class_weight = {0: 1.0, 1: 2.0}
    cat_cols_s2 = [c for c in cat_cols if c in X_tr.columns]
    model = lgb.LGBMClassifier(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        class_weight=class_weight if sample_weight is None else None,
        n_jobs=-1, random_state=42, verbose=-1, importance_type="gain",
    )
    fit_kw = {
        "eval_set": [(X_te, y_te)],
        "callbacks": [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=500)],
        "categorical_feature": cat_cols_s2,
    }
    if sample_weight is not None:
        fit_kw["sample_weight"] = sample_weight
    model.fit(X_tr, y_tr, **fit_kw)
    return model


def main():
    print("Loading data...")
    (X_train, X_test, y_train, y_test, cat_cols,
     pct_oob_train, pct_oob_test, avg_score_train) = load_data()
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}\n")

    # Train Stage 1
    print("=" * 70)
    print("Training Stage 1 (fixed)")
    print("=" * 70)
    s1_model = train_stage1(X_train, y_train, X_test, y_test)
    print(f"  Best iteration: {s1_model.best_iteration_}\n")

    # Prepare Stage 2 data
    train_mask = y_train > 0
    test_mask = y_test > 0

    X_tr_s2 = X_train[train_mask].reset_index(drop=True)
    y_tr_s2 = (y_train[train_mask] - 1).astype(int)
    pct_oob_tr_s2 = pct_oob_train[train_mask]

    X_te_s2 = X_test[test_mask].reset_index(drop=True)
    y_te_s2 = (y_test[test_mask] - 1).astype(int)

    # Drop report_age_days
    for df in [X_tr_s2, X_te_s2]:
        for c in STAGE2_DROP:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    cat_cols_s2 = [c for c in cat_cols if c in X_tr_s2.columns]
    for col in cat_cols_s2:
        X_tr_s2[col] = X_tr_s2[col].astype("category")
        X_te_s2[col] = X_te_s2[col].astype("category")

    print(f"  S2 train: {len(X_tr_s2)} (tink={((y_tr_s2==0).sum())}, oob={((y_tr_s2==1).sum())})")
    print(f"  S2 test:  {len(X_te_s2)} (tink={((y_te_s2==0).sum())}, oob={((y_te_s2==1).sum())})\n")

    # ===== BASELINE =====
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)
    s2_base = train_s2(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols_s2)
    y_pred_base = s2_base.predict(X_te_s2)
    f1_base = f1_score(y_te_s2, y_pred_base, average="macro", zero_division=0)
    r_base = cascade_evaluate(s1_model, s2_base, STAGE2_DROP, X_test, y_test)
    print(f"  S2 F1={f1_base:.4f}, Cascade F1={r_base['f1']:.4f} "
          f"borked R={r_base['borked_r']:.4f}/P={r_base['borked_p']:.4f} "
          f"oob R={r_base['oob_r']:.4f}/P={r_base['oob_p']:.4f}\n")

    # ===== STEP 8: CLEANLAB =====
    print("=" * 70)
    print("STEP 8: Cleanlab")
    print("=" * 70)

    from cleanlab.filter import find_label_issues

    # Cross-validated probabilities — encode categoricals for sklearn
    X_tr_numeric = X_tr_s2.copy()
    for col in cat_cols_s2:
        if col in X_tr_numeric.columns:
            X_tr_numeric[col] = X_tr_numeric[col].cat.codes

    cv_model = lgb.LGBMClassifier(
        n_estimators=500, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, class_weight={0: 1.0, 1: 2.0},
        n_jobs=-1, random_state=42, verbose=-1,
    )
    print("  Cross-validating (5-fold)...")
    pred_probs = cross_val_predict(cv_model, X_tr_numeric, y_tr_s2, cv=5, method="predict_proba")

    issue_indices = find_label_issues(
        labels=np.asarray(y_tr_s2),
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    n_issues = len(issue_indices)
    pct_issues = n_issues / len(y_tr_s2) * 100

    issues_in_tink = (np.asarray(y_tr_s2)[issue_indices] == 0).sum()
    issues_in_oob = (np.asarray(y_tr_s2)[issue_indices] == 1).sum()
    total_tink = (y_tr_s2 == 0).sum()
    total_oob = (y_tr_s2 == 1).sum()
    print(f"  Issues: {n_issues} ({pct_issues:.1f}%)")
    print(f"    tinkering: {issues_in_tink}/{total_tink} ({issues_in_tink/total_tink*100:.1f}%)")
    print(f"    works_oob: {issues_in_oob}/{total_oob} ({issues_in_oob/total_oob*100:.1f}%)")

    # 8a: Remove ONLY from majority class (tinkering) to avoid destroying minority
    print("\n  8a: Remove noisy tinkering only (preserve oob):")
    tink_issues = issue_indices[np.asarray(y_tr_s2)[issue_indices] == 0]
    for remove_pct in [25, 50, 75, 100]:
        n_remove = int(len(tink_issues) * remove_pct / 100)
        remove_set = set(tink_issues[:n_remove])
        keep = np.array([i not in remove_set for i in range(len(y_tr_s2))])
        X_c = X_tr_s2[keep].reset_index(drop=True)
        y_c = y_tr_s2[keep]
        for col in cat_cols_s2:
            if col in X_c.columns:
                X_c[col] = X_c[col].astype("category")
        s2_c = train_s2(X_c, y_c, X_te_s2, y_te_s2, cat_cols_s2)
        f1_c = f1_score(y_te_s2, s2_c.predict(X_te_s2), average="macro", zero_division=0)
        r_c = cascade_evaluate(s1_model, s2_c, STAGE2_DROP, X_test, y_test)
        print(f"    rm {remove_pct}% tink ({n_remove}): S2 F1={f1_c:.4f} ({f1_c-f1_base:+.4f}) "
              f"casc F1={r_c['f1']:.4f} ({r_c['f1']-r_base['f1']:+.4f}) "
              f"oob R={r_c['oob_r']:.4f}/P={r_c['oob_p']:.4f}")

    # 8b: Balanced removal — same % from each class
    print("\n  8b: Balanced removal (same % from each class):")
    oob_issues = issue_indices[np.asarray(y_tr_s2)[issue_indices] == 1]
    for remove_pct in [10, 20, 30]:
        n_rm_tink = int(len(tink_issues) * remove_pct / 100)
        n_rm_oob = int(len(oob_issues) * remove_pct / 100)
        remove_set = set(tink_issues[:n_rm_tink]) | set(oob_issues[:n_rm_oob])
        keep = np.array([i not in remove_set for i in range(len(y_tr_s2))])
        X_c = X_tr_s2[keep].reset_index(drop=True)
        y_c = y_tr_s2[keep]
        for col in cat_cols_s2:
            if col in X_c.columns:
                X_c[col] = X_c[col].astype("category")
        s2_c = train_s2(X_c, y_c, X_te_s2, y_te_s2, cat_cols_s2)
        f1_c = f1_score(y_te_s2, s2_c.predict(X_te_s2), average="macro", zero_division=0)
        r_c = cascade_evaluate(s1_model, s2_c, STAGE2_DROP, X_test, y_test)
        n_total_rm = len(remove_set)
        print(f"    rm {remove_pct}% balanced ({n_total_rm}): S2 F1={f1_c:.4f} ({f1_c-f1_base:+.4f}) "
              f"casc F1={r_c['f1']:.4f} ({r_c['f1']-r_base['f1']:+.4f}) "
              f"oob R={r_c['oob_r']:.4f}/P={r_c['oob_p']:.4f}")

    # 8c: Downweight noisy (per-class balanced)
    print("\n  8c: Downweight noisy labels:")
    issue_set = set(issue_indices)
    for noise_w in [0.1, 0.3, 0.5, 0.7]:
        sw = np.ones(len(y_tr_s2))
        for idx in issue_set:
            sw[idx] = noise_w
        # Apply class weight
        sw[y_tr_s2 == 1] *= 2.0
        s2_w = train_s2(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols_s2, sample_weight=sw)
        f1_w = f1_score(y_te_s2, s2_w.predict(X_te_s2), average="macro", zero_division=0)
        r_w = cascade_evaluate(s1_model, s2_w, STAGE2_DROP, X_test, y_test)
        print(f"    weight={noise_w}: S2 F1={f1_w:.4f} ({f1_w-f1_base:+.4f}) "
              f"casc F1={r_w['f1']:.4f} ({r_w['f1']-r_base['f1']:+.4f}) "
              f"oob R={r_w['oob_r']:.4f}/P={r_w['oob_p']:.4f}")

    print()

    # ===== STEP 9: Consensus weighting =====
    print("=" * 70)
    print("STEP 9: Consensus weighting (pct_works_oob)")
    print("=" * 70)

    # Label-aligned confidence
    label_aligned = np.where(
        y_tr_s2 == 1,
        pct_oob_tr_s2,        # oob: confidence = pct_oob
        1 - pct_oob_tr_s2,    # tinkering: confidence = 1 - pct_oob
    )
    print(f"  label_aligned: mean={label_aligned.mean():.3f}, median={np.median(label_aligned):.3f}")
    print(f"  Low confidence (<0.3): {(label_aligned < 0.3).sum()} ({(label_aligned < 0.3).mean()*100:.1f}%)")
    print(f"  Low confidence (<0.2): {(label_aligned < 0.2).sum()} ({(label_aligned < 0.2).mean()*100:.1f}%)")

    # 9a: Weight by consensus
    print("\n  9a: Weight by consensus confidence:")
    for floor in [0.05, 0.1, 0.2, 0.3]:
        sw = np.clip(label_aligned, floor, 1.0)
        sw[y_tr_s2 == 1] *= 2.0
        s2_conf = train_s2(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2, cat_cols_s2, sample_weight=sw)
        f1_conf = f1_score(y_te_s2, s2_conf.predict(X_te_s2), average="macro", zero_division=0)
        r_conf = cascade_evaluate(s1_model, s2_conf, STAGE2_DROP, X_test, y_test)
        print(f"    floor={floor}: S2 F1={f1_conf:.4f} ({f1_conf-f1_base:+.4f}) "
              f"casc F1={r_conf['f1']:.4f} ({r_conf['f1']-r_base['f1']:+.4f}) "
              f"oob R={r_conf['oob_r']:.4f}/P={r_conf['oob_p']:.4f}")

    # 9b: Filter low-confidence samples
    print("\n  9b: Filter by consensus (remove low-confidence):")
    for min_conf in [0.15, 0.2, 0.25, 0.3, 0.4]:
        keep = label_aligned >= min_conf
        X_f = X_tr_s2[keep].reset_index(drop=True)
        y_f = y_tr_s2[keep]
        for col in cat_cols_s2:
            if col in X_f.columns:
                X_f[col] = X_f[col].astype("category")
        s2_f = train_s2(X_f, y_f, X_te_s2, y_te_s2, cat_cols_s2)
        f1_f = f1_score(y_te_s2, s2_f.predict(X_te_s2), average="macro", zero_division=0)
        r_f = cascade_evaluate(s1_model, s2_f, STAGE2_DROP, X_test, y_test)
        n_kept = keep.sum()
        print(f"    min_conf={min_conf}: kept {n_kept} ({n_kept/len(y_tr_s2)*100:.0f}%) "
              f"S2 F1={f1_f:.4f} ({f1_f-f1_base:+.4f}) "
              f"casc F1={r_f['f1']:.4f} ({r_f['f1']-r_base['f1']:+.4f}) "
              f"oob R={r_f['oob_r']:.4f}/P={r_f['oob_p']:.4f}")

    print()

    # ===== STEP 10: Regression =====
    print("=" * 70)
    print("STEP 10: Regression")
    print("=" * 70)

    X_tr_reg = X_train.copy()
    X_te_reg = X_test.copy()
    cat_cols_reg = [c for c in cat_cols if c in X_tr_reg.columns]
    for col in cat_cols_reg:
        X_tr_reg[col] = X_tr_reg[col].astype("category")
        X_te_reg[col] = X_te_reg[col].astype("category")

    # 10a: Fixed targets
    print("  10a: Fixed targets (borked=0, tinkering=0.5, oob=1.0)")
    target_map = {0: 0.0, 1: 0.5, 2: 1.0}
    y_tr_reg = np.array([target_map[v] for v in y_train])
    y_te_reg = np.array([target_map[v] for v in y_test])

    reg_a = lgb.LGBMRegressor(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    reg_a.fit(X_tr_reg, y_tr_reg,
              eval_set=[(X_te_reg, y_te_reg)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=500)],
              categorical_feature=cat_cols_reg)
    y_pred_a = reg_a.predict(X_te_reg)
    mae_a = mean_absolute_error(y_te_reg, y_pred_a)
    print(f"    MAE={mae_a:.4f}, iter={reg_a.best_iteration_}")

    print(f"    {'thresholds':>15}  {'F1':>7}  {'borked R':>9}  {'borked P':>9}  {'oob R':>7}  {'oob P':>7}")
    for t_b, t_o in [(0.15, 0.75), (0.2, 0.7), (0.2, 0.65), (0.25, 0.65),
                     (0.25, 0.7), (0.3, 0.65), (0.3, 0.7)]:
        y_d = np.where(y_pred_a < t_b, 0, np.where(y_pred_a > t_o, 2, 1))
        f1_d = f1_score(y_test, y_d, average="macro", zero_division=0)
        br = (y_d[y_test == 0] == 0).mean()
        bp = (y_test[y_d == 0] == 0).mean() if (y_d == 0).any() else 0
        _or = (y_d[y_test == 2] == 2).mean()
        op = (y_test[y_d == 2] == 2).mean() if (y_d == 2).any() else 0
        print(f"    ({t_b:.2f},{t_o:.2f})       {f1_d:>7.4f}  {br:>9.4f}  {bp:>9.4f}  {_or:>7.4f}  {op:>7.4f}")

    # 10b: Game-based target (avg_verdict_score)
    print("\n  10b: Game-based target (avg_verdict_score)")
    y_tr_reg_b = avg_score_train
    # Eval still on true labels
    y_te_reg_b = y_te_reg

    reg_b = lgb.LGBMRegressor(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    reg_b.fit(X_tr_reg, y_tr_reg_b,
              eval_set=[(X_te_reg, y_te_reg_b)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=500)],
              categorical_feature=cat_cols_reg)
    y_pred_b = reg_b.predict(X_te_reg)
    mae_b = mean_absolute_error(y_te_reg_b, y_pred_b)
    print(f"    MAE={mae_b:.4f}, iter={reg_b.best_iteration_}")

    print(f"    {'thresholds':>15}  {'F1':>7}  {'borked R':>9}  {'borked P':>9}  {'oob R':>7}  {'oob P':>7}")
    for t_b, t_o in [(0.15, 0.75), (0.2, 0.7), (0.2, 0.65), (0.25, 0.65),
                     (0.25, 0.7), (0.3, 0.65), (0.3, 0.7)]:
        y_d = np.where(y_pred_b < t_b, 0, np.where(y_pred_b > t_o, 2, 1))
        f1_d = f1_score(y_test, y_d, average="macro", zero_division=0)
        br = (y_d[y_test == 0] == 0).mean()
        bp = (y_test[y_d == 0] == 0).mean() if (y_d == 0).any() else 0
        _or = (y_d[y_test == 2] == 2).mean()
        op = (y_test[y_d == 2] == 2).mean() if (y_d == 2).any() else 0
        print(f"    ({t_b:.2f},{t_o:.2f})       {f1_d:>7.4f}  {br:>9.4f}  {bp:>9.4f}  {_or:>7.4f}  {op:>7.4f}")

    # 10c: Regression score distribution analysis
    print("\n  10c: Score distribution analysis (10a model):")
    for cls_name, cls_val in [("borked", 0), ("tinkering", 1), ("oob", 2)]:
        scores = y_pred_a[y_test == cls_val]
        print(f"    {cls_name:12s}: mean={scores.mean():.3f} std={scores.std():.3f} "
              f"median={np.median(scores):.3f} [p10={np.percentile(scores,10):.3f}, "
              f"p90={np.percentile(scores,90):.3f}]")

    # ===== SUMMARY =====
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline cascade F1:  {r_base['f1']:.4f}")
    print(f"  Cleanlab issues:      {n_issues} ({pct_issues:.1f}%)")
    print(f"  Regression 10a MAE:   {mae_a:.4f}")
    print(f"  Regression 10b MAE:   {mae_b:.4f}")


if __name__ == "__main__":
    main()
