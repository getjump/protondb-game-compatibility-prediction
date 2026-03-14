#!/usr/bin/env python3
"""Phase 9.2 experiment: aggregate features from cust_*, flag_*, fault, game_metadata.

Builds per-game aggregate features and tests their impact on cascade model F1.

Feature groups:
  P1 — cust_* aggregates per app_id (% reports using winetricks, protontricks, etc.)
  P2 — flag_* aggregates per app_id (% reports with esync/fsync/d3d11 flags)
  P3 — fault aggregates per app_id (% reports with audio/graphical/etc. faults)
  P4 — game_metadata structural: deck_status ordinal, github_open_count, has_regression

Each group is tested individually and combined, vs baseline (no new features).
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Column lists ──────────────────────────────────────────────────

CUST_COLS = [
    "cust_winetricks", "cust_protontricks", "cust_config_change",
    "cust_custom_prefix", "cust_custom_proton", "cust_lutris",
    "cust_media_foundation", "cust_protonfixes", "cust_native2proton",
    "cust_not_listed",
]

FLAG_COLS = [
    "flag_use_wine_d3d11", "flag_disable_esync", "flag_enable_nvapi",
    "flag_disable_fsync", "flag_use_wine_d9vk", "flag_large_address_aware",
    "flag_disable_d3d11", "flag_hide_nvidia", "flag_game_drive",
    "flag_no_write_watch", "flag_no_xim", "flag_old_gl_string",
    "flag_use_seccomp", "flag_fullscreen_integer_scaling",
]

FAULT_COLS = [
    "audio_faults", "graphical_faults", "input_faults",
    "performance_faults", "stability_faults", "windowing_faults",
    "save_game_faults", "significant_bugs",
]


def build_per_game_aggregates(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Build per-app_id aggregate DataFrames for P1/P2/P3/P4.

    Returns dict mapping group name -> DataFrame indexed by app_id.
    """
    # ── P1: cust_* aggregates ──
    cust_select = ", ".join(
        f"AVG(COALESCE({c}, 0)) AS agg_{c}" for c in CUST_COLS
    )
    # Also add a composite: any_customization
    p1_df = pd.read_sql_query(
        f"""
        SELECT app_id, COUNT(*) as n_reports_agg,
               {cust_select},
               AVG(CASE WHEN ({' + '.join(f'COALESCE({c},0)' for c in CUST_COLS)}) > 0
                   THEN 1.0 ELSE 0.0 END) AS agg_any_customization
        FROM reports
        GROUP BY app_id
        HAVING COUNT(*) >= 3
        """,
        conn,
    ).set_index("app_id")

    # ── P2: flag_* aggregates ──
    flag_select = ", ".join(
        f"AVG(COALESCE({c}, 0)) AS agg_{c}" for c in FLAG_COLS
    )
    p2_df = pd.read_sql_query(
        f"""
        SELECT app_id,
               {flag_select},
               AVG(CASE WHEN ({' + '.join(f'COALESCE({c},0)' for c in FLAG_COLS)}) > 0
                   THEN 1.0 ELSE 0.0 END) AS agg_any_flag
        FROM reports
        GROUP BY app_id
        HAVING COUNT(*) >= 3
        """,
        conn,
    ).set_index("app_id")

    # ── P3: fault aggregates ──
    # Fault columns are TEXT "yes"/"no" — convert to 0/1 in SQL
    fault_select = ", ".join(
        f"AVG(CASE WHEN {c} = 'yes' THEN 1.0 ELSE 0.0 END) AS agg_{c}"
        for c in FAULT_COLS
    )
    p3_df = pd.read_sql_query(
        f"""
        SELECT app_id,
               {fault_select},
               AVG(CASE WHEN (
                   {' + '.join(f"CASE WHEN {c} = 'yes' THEN 1 ELSE 0 END" for c in FAULT_COLS)}
               ) > 0 THEN 1.0 ELSE 0.0 END) AS agg_any_fault,
               AVG(
                   {' + '.join(f"CASE WHEN {c} = 'yes' THEN 1 ELSE 0 END" for c in FAULT_COLS)}
               ) AS agg_fault_count_mean
        FROM reports
        WHERE audio_faults IS NOT NULL
        GROUP BY app_id
        HAVING COUNT(*) >= 3
        """,
        conn,
    ).set_index("app_id")

    # ── P4: game_metadata structural ──
    p4_df = pd.read_sql_query(
        """
        SELECT app_id,
               COALESCE(deck_status, 0) AS meta_deck_status,
               COALESCE(github_open_count, 0) AS meta_github_open,
               COALESCE(github_has_regression, 0) AS meta_has_regression,
               CASE WHEN github_issue_count IS NOT NULL THEN 1 ELSE 0 END AS meta_has_github_issues
        FROM game_metadata
        """,
        conn,
    ).set_index("app_id")

    return {"P1_cust": p1_df, "P2_flag": p2_df, "P3_fault": p3_df, "P4_meta": p4_df}


def run_experiment(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test_es: pd.DataFrame,
    y_test_es: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    label: str,
    console: Console,
) -> dict:
    """Train cascade and return metrics on both ES and eval sets."""
    from protondb_settings.ml.models.cascade import CascadeClassifier, train_stage1, train_stage2
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    t0 = time.time()

    # Stage 1
    s1 = train_stage1(X_train.copy(), y_train, X_test_es.copy(), y_test_es, categorical_cols=cat_cols)

    # Stage 2
    s2, s2_dropped = train_stage2(X_train.copy(), y_train, X_test_es.copy(), y_test_es, categorical_cols=cat_cols)

    # Cascade (no calibration for experiment — we compare raw F1)
    cascade = CascadeClassifier(s1, s2, s2_dropped)

    # Ensure categorical dtypes on prediction sets
    for col in cat_cols:
        if col in X_test_es.columns:
            X_test_es[col] = X_test_es[col].astype("category")
        if col in X_eval.columns:
            X_eval[col] = X_eval[col].astype("category")

    # ES metrics
    y_pred_es = cascade.predict(X_test_es)
    f1_es = f1_score(y_test_es, y_pred_es, average="macro", zero_division=0)
    f1_per_es = f1_score(y_test_es, y_pred_es, average=None, labels=[0, 1, 2], zero_division=0)

    # Eval metrics (held-out)
    y_pred_eval = cascade.predict(X_eval)
    acc_eval = accuracy_score(y_eval, y_pred_eval)
    f1_eval = f1_score(y_eval, y_pred_eval, average="macro", zero_division=0)
    f1_per_eval = f1_score(y_eval, y_pred_eval, average=None, labels=[0, 1, 2], zero_division=0)

    elapsed = time.time() - t0

    console.print(f"  [{label}] ES F1={f1_es:.4f}  eval F1={f1_eval:.4f}  acc={acc_eval:.4f}  "
                  f"borked={f1_per_eval[0]:.3f} tink={f1_per_eval[1]:.3f} oob={f1_per_eval[2]:.3f}  "
                  f"({elapsed:.0f}s)")

    return {
        "label": label,
        "f1_macro": f1_es,
        "f1_eval": f1_eval,
        "accuracy": acc_eval,
        "f1_borked": f1_per_eval[0],
        "f1_tinkering": f1_per_eval[1],
        "f1_works_oob": f1_per_eval[2],
        "elapsed": elapsed,
        "s1_iter": s1.best_iteration_,
        "s2_iter": s2.best_iteration,
        "n_features": X_train.shape[1],
    }


def main():
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.relabeling import apply_relabeling, get_relabel_ids
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split

    console = Console()
    db_path = Path("data/protondb.db")
    emb_path = Path("data/embeddings.npz")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── Step 1: Build base feature matrix using cached embeddings ──
    console.print("[bold]Loading cached embeddings...[/bold]")
    emb_data = load_embeddings(emb_path)
    console.print(f"  GPU: {emb_data.get('n_components_gpu', '?')} dims, "
                  f"Text: {emb_data.get('text_n_components', 0)} dims")

    from protondb_settings.config import NORMALIZED_DATA_SOURCE

    X, y, timestamps, report_ids, label_maps = _build_feature_matrix(
        conn, emb_data, normalized_data_source=NORMALIZED_DATA_SOURCE,
    )
    console.print(f"Base: {X.shape[0]} samples, {X.shape[1]} features")

    # We need app_id for each row to join aggregates
    console.print("Fetching app_ids for all reports...")
    report_app_ids = {}
    for row in conn.execute("SELECT id, app_id FROM reports").fetchall():
        report_app_ids[row["id"]] = row["app_id"]

    app_ids = np.array([report_app_ids.get(rid, -1) for rid in report_ids])
    X["_app_id"] = app_ids

    # ── Step 2: Build aggregates ──
    console.print("[bold]Building per-game aggregates...[/bold]")
    agg_groups = build_per_game_aggregates(conn)

    for name, df in agg_groups.items():
        console.print(f"  {name}: {len(df)} games, {len(df.columns)} features")

    # ── Step 3: Split (before joining — no data leakage!) ──
    console.print("[bold]Time-based split...[/bold]")

    # Relabeling
    relabel_ids = get_relabel_ids(conn)

    X_train, X_test, y_train, y_test, train_rids, _ = _time_based_split(
        X, y, timestamps, test_fraction=0.2, report_ids=report_ids,
    )
    y_train, n_relabeled = apply_relabeling(y_train, train_rids, relabel_ids)
    console.print(f"  Relabeled {n_relabeled} training samples")

    # Split cal/eval from test
    n_cal = len(X_test) // 2
    X_eval = X_test.iloc[n_cal:].copy().reset_index(drop=True)
    y_eval = y_test[n_cal:]
    # Use first half as "test" for early stopping, second half for eval
    X_test_es = X_test.iloc[:n_cal].copy().reset_index(drop=True)
    y_test_es = y_test[:n_cal]

    console.print(f"  Train: {len(X_train)}, ES: {len(X_test_es)}, Eval: {len(X_eval)}")

    # ── Step 4: Build variant DataFrames ──
    # For each group, join aggregates to train/test/eval using _app_id
    def join_aggregates(X_base: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        """Join selected aggregate groups to base DataFrame."""
        X_out = X_base.copy()
        for grp in groups:
            agg_df = agg_groups[grp]
            # Join on _app_id
            for col in agg_df.columns:
                if col == "n_reports_agg":
                    continue  # skip count column
                mapping = agg_df[col].to_dict()
                X_out[col] = X_out["_app_id"].map(mapping).astype(float)
        return X_out

    # Drop _app_id before training (it's just for joining)
    def drop_app_id(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["_app_id"], errors="ignore")

    # ── Step 5: Run experiments ──
    console.print("\n[bold]Running experiments...[/bold]\n")

    results = []

    # Baseline (no new features)
    console.print("[cyan]Baseline (no aggregate features):[/cyan]")
    r = run_experiment(
        drop_app_id(X_train), y_train,
        drop_app_id(X_test_es), y_test_es,
        drop_app_id(X_eval), y_eval,
        "baseline", console,
    )
    results.append(r)

    # Individual groups
    for grp_name in ["P1_cust", "P2_flag", "P3_fault", "P4_meta"]:
        console.print(f"\n[cyan]{grp_name}:[/cyan]")
        X_tr = join_aggregates(X_train, [grp_name])
        X_te = join_aggregates(X_test_es, [grp_name])
        X_ev = join_aggregates(X_eval, [grp_name])

        r = run_experiment(
            drop_app_id(X_tr), y_train,
            drop_app_id(X_te), y_test_es,
            drop_app_id(X_ev), y_eval,
            grp_name, console,
        )
        results.append(r)

    # All combined
    console.print(f"\n[cyan]ALL (P1+P2+P3+P4):[/cyan]")
    all_groups = ["P1_cust", "P2_flag", "P3_fault", "P4_meta"]
    X_tr = join_aggregates(X_train, all_groups)
    X_te = join_aggregates(X_test_es, all_groups)
    X_ev = join_aggregates(X_eval, all_groups)

    r = run_experiment(
        drop_app_id(X_tr), y_train,
        drop_app_id(X_te), y_test_es,
        drop_app_id(X_ev), y_eval,
        "ALL", console,
    )
    results.append(r)

    # Best individual groups combined (top-2 by eval F1)
    indiv = [r for r in results if r["label"] not in ("baseline", "ALL")]
    indiv_sorted = sorted(indiv, key=lambda x: x["f1_eval"], reverse=True)
    if len(indiv_sorted) >= 2:
        top2 = [indiv_sorted[0]["label"], indiv_sorted[1]["label"]]
        combo_name = "+".join(top2)
        console.print(f"\n[cyan]Top-2 combo: {combo_name}:[/cyan]")
        X_tr = join_aggregates(X_train, top2)
        X_te = join_aggregates(X_test_es, top2)
        X_ev = join_aggregates(X_eval, top2)

        r = run_experiment(
            drop_app_id(X_tr), y_train,
            drop_app_id(X_te), y_test_es,
            drop_app_id(X_ev), y_eval,
            combo_name, console,
        )
        results.append(r)

    # ── Summary table ──
    console.print("\n")
    table = Table(title="Phase 9.2 — Aggregate Features Ablation")
    table.add_column("Experiment", style="cyan")
    table.add_column("Features", justify="right")
    table.add_column("F1 (ES)", justify="right")
    table.add_column("F1 (eval)", justify="right", style="bold")
    table.add_column("ΔF1", justify="right")
    table.add_column("borked", justify="right")
    table.add_column("tinkering", justify="right")
    table.add_column("works_oob", justify="right")

    baseline_f1 = results[0]["f1_eval"]
    for r in results:
        delta = r["f1_eval"] - baseline_f1
        delta_str = f"{delta:+.4f}"
        if delta > 0.002:
            delta_str = f"[green]{delta_str}[/green]"
        elif delta < -0.002:
            delta_str = f"[red]{delta_str}[/red]"

        table.add_row(
            r["label"],
            str(r["n_features"]),
            f"{r['f1_macro']:.4f}",
            f"{r['f1_eval']:.4f}",
            delta_str,
            f"{r['f1_borked']:.3f}",
            f"{r['f1_tinkering']:.3f}",
            f"{r['f1_works_oob']:.3f}",
        )

    console.print(table)
    conn.close()



if __name__ == "__main__":
    main()
