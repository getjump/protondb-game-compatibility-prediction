#!/usr/bin/env python3
"""Find games that likely work on Linux but have unclear/negative community perception.

Criteria for "hidden gems":
  - Model predicts works_oob or tinkering (not borked)
  - Community reports are mixed or mostly negative
  - Game has enough reports to be interesting (≥3)
  - No blocking anticheat

Also finds the inverse: games community thinks work but model says borked.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_batch_features(
    app_ids: list[int],
    gpu_info: dict,
    emb_data: dict,
    variant: str = "official",
) -> pd.DataFrame:
    """Build feature matrix for many app_ids at once (same GPU)."""
    from protondb_settings.ml.features.encoding import extract_gpu_family
    from protondb_settings.ml.predict import _parse_nvidia_driver, _parse_mesa_driver

    gpu_raw = gpu_info.get("gpu_raw")
    gpu_family = extract_gpu_family(gpu_raw)
    vendor = gpu_info.get("vendor")

    nvidia_dv = _parse_nvidia_driver(gpu_info.get("driver_version")) if vendor == "nvidia" else None
    mesa_dv = _parse_mesa_driver(gpu_info.get("driver_version")) if vendor != "nvidia" else None

    gpu_lower = (gpu_raw or "").lower()
    is_steam_deck = 1 if ("vangogh" in gpu_lower or "van gogh" in gpu_lower) else 0

    # Embedding lookups
    gpu_families = emb_data.get("gpu_families", [])
    gpu_emb_matrix = emb_data.get("gpu_embeddings", np.array([]))
    gpu_family_to_idx = {f: i for i, f in enumerate(gpu_families)}

    game_ids = emb_data.get("game_ids", [])
    game_emb_matrix = emb_data.get("game_embeddings", np.array([]))
    game_id_to_idx = {int(g): i for i, g in enumerate(game_ids)}

    n_gpu_emb = gpu_emb_matrix.shape[1] if gpu_emb_matrix.ndim == 2 and gpu_emb_matrix.size > 0 else 0
    n_text_emb = emb_data.get("text_n_components", 0)

    # GPU embedding (same for all rows)
    gpu_emb_vals = [np.nan] * n_gpu_emb
    if gpu_family and gpu_family in gpu_family_to_idx and gpu_emb_matrix.size > 0:
        idx = gpu_family_to_idx[gpu_family]
        gpu_emb_vals = [float(gpu_emb_matrix[idx, d]) for d in range(n_gpu_emb)]

    # Static part (same for all rows)
    static = {
        "gpu_family": gpu_family,
        "nvidia_driver_version": nvidia_dv,
        "mesa_driver_version": mesa_dv,
        "is_apu": 0, "is_igpu": 0, "is_mobile": 0,
        "is_steam_deck": is_steam_deck,
        "variant": variant,
        "has_concluding_notes": 0, "concluding_notes_length": 0,
        "fault_notes_count": 0, "has_customization_notes": 0,
        "total_notes_length": 0,
        "mentions_crash": 0, "mentions_fix": 0, "mentions_perfect": 0,
        "mentions_proton_version": 0, "mentions_env_var": 0,
        "mentions_performance": 0,
        "sentiment_negative_words": 0, "sentiment_positive_words": 0,
    }
    for d in range(n_gpu_emb):
        static[f"gpu_emb_{d}"] = gpu_emb_vals[d]
    for d in range(n_text_emb):
        static[f"text_emb_{d}"] = np.nan

    # Per-game aggregates (Phase 9.2)
    game_agg_cust = emb_data.get("game_agg_cust")
    game_agg_flag = emb_data.get("game_agg_flag")
    agg_cust_cols = emb_data.get("game_agg_columns_cust", [])
    agg_flag_cols = emb_data.get("game_agg_columns_flag", [])

    # Build rows — game_emb and aggregates differ per game
    records = []
    for app_id in app_ids:
        row = dict(static)
        if app_id in game_id_to_idx and game_emb_matrix.size > 0:
            idx = game_id_to_idx[app_id]
            for d in range(n_gpu_emb):
                row[f"game_emb_{d}"] = float(game_emb_matrix[idx, d])
            # Aggregates use same index
            if game_agg_cust is not None:
                for j, col in enumerate(agg_cust_cols):
                    row[col] = float(game_agg_cust[idx, j])
                for j, col in enumerate(agg_flag_cols):
                    row[col] = float(game_agg_flag[idx, j])
        else:
            for d in range(n_gpu_emb):
                row[f"game_emb_{d}"] = np.nan
            if game_agg_cust is not None:
                for col in agg_cust_cols:
                    row[col] = np.nan
                for col in agg_flag_cols:
                    row[col] = np.nan
        records.append(row)

    X = pd.DataFrame(records)

    for col in ("nvidia_driver_version", "mesa_driver_version"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in X.columns:
        if X[col].dtype == object and col != "variant":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    if "variant" in X.columns:
        X["variant"] = X["variant"].astype("category")

    return X


def main():
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.predict import detect_gpu, _check_metadata_override
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    import joblib
    from rich.console import Console
    from rich.table import Table

    console = Console()
    db_path = Path("data/protondb.db")
    model_path = Path("data/model_cascade.pkl")
    emb_path = Path("data/embeddings.npz")

    if not model_path.exists() or not emb_path.exists():
        console.print("[red]Model or embeddings not found. Run 'ml train-cascade' first.[/red]")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    gpu_info = detect_gpu()
    console.print(f"GPU: {gpu_info.get('gpu_raw', 'unknown')}")

    cascade = joblib.load(model_path)
    emb_data = load_embeddings(emb_path)
    game_ids_set = {int(g) for g in emb_data.get("game_ids", [])}

    # Per-game report stats
    console.print("Loading report statistics...")
    rows = conn.execute("""
        SELECT app_id,
               COUNT(*) as total,
               SUM(CASE WHEN verdict = 'no' THEN 1 ELSE 0 END) as borked,
               SUM(CASE WHEN verdict = 'yes' THEN 1 ELSE 0 END) as works
        FROM reports
        GROUP BY app_id
        HAVING total >= 3
    """).fetchall()

    game_stats = {}
    for r in rows:
        app_id = r["app_id"]
        total = r["total"]
        game_stats[app_id] = {
            "total_reports": total,
            "borked_pct": r["borked"] / total,
            "works_pct": r["works"] / total,
        }

    console.print(f"Games with ≥3 reports: {len(game_stats)}")

    # Metadata
    meta_rows = conn.execute(
        "SELECT app_id, anticheat, anticheat_status, deck_status FROM game_metadata"
    ).fetchall()
    meta_lookup = {r["app_id"]: dict(r) for r in meta_rows}

    # Separate overridden vs model-predicted
    override_results = []
    model_app_ids = []

    for app_id in game_stats:
        meta = meta_lookup.get(app_id)
        override = _check_metadata_override(meta)
        if override:
            override_results.append((app_id, override))
        elif app_id in game_ids_set:
            model_app_ids.append(app_id)

    console.print(f"Overridden by metadata: {len(override_results)}")
    console.print(f"Model predictions needed: {len(model_app_ids)}")

    # Batch predict
    console.print("Building features...")
    X = _build_batch_features(model_app_ids, gpu_info, emb_data)

    # Align columns with model
    if hasattr(cascade, 'stage1') and hasattr(cascade.stage1, 'feature_name_'):
        expected_cols = cascade.stage1.feature_name_
        for col in expected_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[expected_cols]

    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    console.print("Running batch prediction...")
    result = cascade.predict_with_confidence(X)
    labels = {0: "borked", 1: "tinkering", 2: "works_oob"}

    # Collect all results
    all_results = []

    for app_id, override in override_results:
        meta = meta_lookup.get(app_id)
        all_results.append({
            **game_stats[app_id],
            "app_id": app_id,
            "prediction": override["prediction"],
            "confidence": override["confidence"],
            "p_borked": override["probabilities"]["borked"],
            "p_works_oob": override["probabilities"]["works_oob"],
            "deck_status": meta.get("deck_status") if meta else None,
            "anticheat": meta.get("anticheat") if meta else None,
        })

    for i, app_id in enumerate(model_app_ids):
        meta = meta_lookup.get(app_id)
        all_results.append({
            **game_stats[app_id],
            "app_id": app_id,
            "prediction": labels[int(result["prediction"][i])],
            "confidence": float(result["confidence"][i]),
            "p_borked": float(result["probabilities"][i, 0]),
            "p_works_oob": float(result["probabilities"][i, 2]),
            "deck_status": meta.get("deck_status") if meta else None,
            "anticheat": meta.get("anticheat") if meta else None,
        })

    console.print(f"Total: {len(all_results)} games\n")

    deck_labels = {1: "unsup", 2: "play", 3: "verified", None: "-"}

    # === Hidden Gems: model says works, community says borked/mixed ===
    gems = [
        r for r in all_results
        if r["prediction"] in ("works_oob", "tinkering")
        and r["confidence"] >= 0.6
        and r["borked_pct"] >= 0.4
        and r["total_reports"] >= 5
    ]
    gems.sort(key=lambda x: (-x["confidence"], -x["borked_pct"]))

    table = Table(title="Hidden Gems: model says WORKS but community reports are negative")
    table.add_column("App ID", style="cyan")
    table.add_column("Prediction", style="green")
    table.add_column("Conf")
    table.add_column("Reports")
    table.add_column("Borked%", style="red")
    table.add_column("Works%", style="green")
    table.add_column("Deck")
    table.add_column("Anticheat")

    for r in gems[:30]:
        table.add_row(
            str(r["app_id"]), r["prediction"], f"{r['confidence']:.0%}",
            str(r["total_reports"]), f"{r['borked_pct']:.0%}", f"{r['works_pct']:.0%}",
            deck_labels.get(r["deck_status"], "?"), r["anticheat"] or "-",
        )
    console.print(table)
    console.print(f"Total hidden gems: {len(gems)}\n")

    # === Opposite: community says works, model says borked ===
    opposite = [
        r for r in all_results
        if r["prediction"] == "borked"
        and r["confidence"] >= 0.6
        and r["works_pct"] >= 0.6
        and r["total_reports"] >= 5
    ]
    opposite.sort(key=lambda x: (-x["confidence"], -x["works_pct"]))

    table2 = Table(title="Suspicious: model says BORKED but community reports positive")
    table2.add_column("App ID", style="cyan")
    table2.add_column("Conf")
    table2.add_column("Reports")
    table2.add_column("Borked%", style="red")
    table2.add_column("Works%", style="green")
    table2.add_column("Deck")
    table2.add_column("Anticheat")

    for r in opposite[:30]:
        table2.add_row(
            str(r["app_id"]), f"{r['confidence']:.0%}",
            str(r["total_reports"]), f"{r['borked_pct']:.0%}", f"{r['works_pct']:.0%}",
            deck_labels.get(r["deck_status"], "?"), r["anticheat"] or "-",
        )
    console.print(table2)
    console.print(f"Total suspicious: {len(opposite)}\n")

    # === Works OOB but community unclear (mixed verdicts) ===
    unclear = [
        r for r in all_results
        if r["prediction"] == "works_oob"
        and r["p_works_oob"] >= 0.5
        and 0.3 <= r["works_pct"] <= 0.7
        and r["total_reports"] >= 10
    ]
    unclear.sort(key=lambda x: -x["p_works_oob"])

    table3 = Table(title="Likely Works OOB but mixed community verdicts (≥10 reports)")
    table3.add_column("App ID", style="cyan")
    table3.add_column("P(OOB)")
    table3.add_column("Reports")
    table3.add_column("Borked%", style="red")
    table3.add_column("Works%", style="green")
    table3.add_column("Deck")

    for r in unclear[:30]:
        table3.add_row(
            str(r["app_id"]), f"{r['p_works_oob']:.0%}",
            str(r["total_reports"]), f"{r['borked_pct']:.0%}", f"{r['works_pct']:.0%}",
            deck_labels.get(r["deck_status"], "?"),
        )
    console.print(table3)
    console.print(f"Total unclear but likely works: {len(unclear)}")

    conn.close()


if __name__ == "__main__":
    main()
