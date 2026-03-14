"""Game-level features from game_metadata table."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Per-report boolean columns for aggregate features (Phase 9.2)
CUST_COLUMNS = [
    "cust_winetricks", "cust_protontricks", "cust_config_change",
    "cust_custom_prefix", "cust_custom_proton", "cust_lutris",
    "cust_media_foundation", "cust_protonfixes", "cust_native2proton",
    "cust_not_listed",
]

FLAG_COLUMNS = [
    "flag_use_wine_d3d11", "flag_disable_esync", "flag_enable_nvapi",
    "flag_disable_fsync", "flag_use_wine_d9vk", "flag_large_address_aware",
    "flag_disable_d3d11", "flag_hide_nvidia", "flag_game_drive",
    "flag_no_write_watch", "flag_no_xim", "flag_old_gl_string",
    "flag_use_seccomp", "flag_fullscreen_integer_scaling",
]

AGG_CUST_COLS = [f"agg_{c}" for c in CUST_COLUMNS] + ["agg_any_customization"]
AGG_FLAG_COLS = [f"agg_{c}" for c in FLAG_COLUMNS] + ["agg_any_flag"]


def build_game_aggregates(conn: sqlite3.Connection) -> dict[int, dict[str, float]]:
    """Build per-game aggregate features from cust_* and flag_* columns.

    Phase 9.2: P1+P2 combined give +0.024 F1 improvement.
    - P1 (cust_*): % of reports per game using each customization type
    - P2 (flag_*): % of reports per game with each launch flag

    Returns dict mapping app_id -> {agg_cust_winetricks: 0.05, ...}.
    """
    cust_select = ", ".join(
        f"AVG(COALESCE({c}, 0)) AS agg_{c}" for c in CUST_COLUMNS
    )
    flag_select = ", ".join(
        f"AVG(COALESCE({c}, 0)) AS agg_{c}" for c in FLAG_COLUMNS
    )
    cust_any = " + ".join(f"COALESCE({c},0)" for c in CUST_COLUMNS)
    flag_any = " + ".join(f"COALESCE({c},0)" for c in FLAG_COLUMNS)

    query = f"""
        SELECT app_id,
               {cust_select},
               AVG(CASE WHEN ({cust_any}) > 0 THEN 1.0 ELSE 0.0 END) AS agg_any_customization,
               {flag_select},
               AVG(CASE WHEN ({flag_any}) > 0 THEN 1.0 ELSE 0.0 END) AS agg_any_flag
        FROM reports
        GROUP BY app_id
    """
    rows = conn.execute(query).fetchall()
    result = {}
    all_cols = AGG_CUST_COLS + AGG_FLAG_COLS
    for row in rows:
        app_id = row["app_id"]
        result[app_id] = {col: float(row[col] or 0.0) for col in all_cols}

    logger.info("Built game aggregates for %d games (%d features)",
                len(result), len(all_cols))
    return result


def game_aggregates_to_arrays(
    game_agg_lookup: dict[int, dict[str, float]],
    game_ids: list,
) -> dict[str, Any]:
    """Convert game aggregate lookup to aligned numpy arrays for npz storage.

    Aligns to the same game_ids ordering used for game embeddings.
    """
    n_cust = len(AGG_CUST_COLS)
    n_flag = len(AGG_FLAG_COLS)
    n_games = len(game_ids)

    agg_cust = np.full((n_games, n_cust), np.nan)
    agg_flag = np.full((n_games, n_flag), np.nan)

    for i, gid in enumerate(game_ids):
        agg = game_agg_lookup.get(int(gid))
        if agg:
            for j, col in enumerate(AGG_CUST_COLS):
                agg_cust[i, j] = agg.get(col, np.nan)
            for j, col in enumerate(AGG_FLAG_COLS):
                agg_flag[i, j] = agg.get(col, np.nan)

    return {
        "game_agg_cust": agg_cust,
        "game_agg_flag": agg_flag,
        "game_agg_columns_cust": AGG_CUST_COLS,
        "game_agg_columns_flag": AGG_FLAG_COLS,
    }


def _build_game_metadata_lookup(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    """Build a lookup dict from game_metadata table (app_id -> row).

    Only fetches columns still used by the ML pipeline.  Many game_metadata
    columns were dropped after ablation showed ΔF1 < 0.002 (see PLAN_ML_7.md).
    """
    rows = conn.execute(
        "SELECT app_id FROM game_metadata"
    ).fetchall()
    if not rows:
        return {}
    return {row["app_id"]: {} for row in rows}


def extract_game_features(
    app_id: int,
    metadata_lookup: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Extract game-level features for a given app_id.

    After Phase 7 ablation, all per-game metadata features were removed:
    - engine (+0.004 F1 when dropped — actively harmful, high cardinality noise)
    - graphics_api_* (5), drm (2), anticheat (1), genre (1) — ΔF1 < 0.002
    - release_year, has_linux_native, is_multiplayer — ΔF1 < 0.002

    Game-level signal now comes entirely from game_emb (SVD embeddings),
    which capture game "personality" much better (ΔF1 = −0.060 if removed).
    """
    # All game metadata features removed after ablation.
    # Game embeddings (game_emb_*) provide the game-level signal.
    return {}


def build_game_metadata_lookup(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    """Build game metadata lookup from the database."""
    return _build_game_metadata_lookup(conn)
