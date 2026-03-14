"""SVD embeddings from extended co-occurrence matrices."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from .encoding import extract_gpu_family, extract_gpu_vendor

logger = logging.getLogger(__name__)


def _verdict_score(verdict: str | None, verdict_oob: str | None) -> float | None:
    """Map verdict fields to a numeric score.

    - verdict_oob="yes" -> 1.0 (works out of box)
    - verdict_oob="no" + verdict="yes" -> 0.5 (needs tinkering)
    - verdict="no" -> 0.0 (borked)
    """
    if verdict_oob == "yes":
        return 1.0
    if verdict_oob == "no" and verdict == "yes":
        return 0.5
    if verdict == "no":
        return 0.0
    # verdict_oob is null, verdict is "yes" -> tinkering
    if verdict == "yes":
        return 0.5
    return None


def _resolve_gpu_family(
    gpu_raw: str | None,
    gpu_lookup: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve GPU family from normalization lookup or heuristic."""
    if not gpu_raw:
        return None
    norm = gpu_lookup.get(gpu_raw)
    if norm:
        vendor = norm.get("vendor", "").lower()
        if vendor == "unknown":
            return None
        return norm.get("family")
    # Fallback: heuristic
    vendor = extract_gpu_vendor(gpu_raw)
    if vendor is None:
        return None
    family = extract_gpu_family(gpu_raw)
    return family



def _select_n_components(
    matrix,
    target_variance: float = 0.90,
    min_d: int = 16,
    max_d: int = 64,
) -> int:
    """Select number of SVD components by explained variance."""
    if hasattr(matrix, "shape"):
        max_possible = min(matrix.shape) - 1
    else:
        max_possible = 64

    if max_possible < min_d:
        return max(1, max_possible)

    probe_n = min(100, max_possible)
    svd_probe = TruncatedSVD(n_components=probe_n, random_state=42)
    svd_probe.fit(matrix)
    cumulative_var = np.cumsum(svd_probe.explained_variance_ratio_)
    n = int(np.searchsorted(cumulative_var, target_variance) + 1)
    n = int(np.clip(n, min_d, min(max_d, max_possible)))
    logger.info(
        "Selected %d components (%.1f%% variance at that point)",
        n,
        cumulative_var[min(n - 1, len(cumulative_var) - 1)] * 100,
    )
    return n


def _build_extended_cooccurrence(
    conn: sqlite3.Connection,
    gpu_lookup: dict[str, dict[str, Any]],
) -> tuple[csr_matrix, list[str], list[int], list[str]]:
    """Build extended co-occurrence matrix: (gpu_family + variant + engine + deck) × game.

    Returns:
        matrix: sparse (n_axes, n_games)
        axes: list of axis names (e.g. "gpu:GeForce RTX 30", "var:ge", "eng:unity")
        game_ids: sorted list of app_ids
        gpu_families: list of GPU family names (for gpu_embeddings extraction)
    """
    logger.info("Building extended co-occurrence matrix...")

    rows = conn.execute(
        "SELECT app_id, gpu, variant, verdict, verdict_oob FROM reports "
        "WHERE gpu IS NOT NULL AND gpu != ''"
    ).fetchall()

    # Engine lookup from game_metadata
    engine_lookup: dict[int, str] = {}
    for r in conn.execute(
        "SELECT app_id, engine FROM game_metadata WHERE engine IS NOT NULL"
    ).fetchall():
        eng = r["engine"].strip().split(",")[0].strip().lower()
        if eng:
            engine_lookup[int(r["app_id"])] = eng

    # Aggregate: (axis_key, app_id) -> list of scores
    pair_scores: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        score = _verdict_score(row["verdict"], row["verdict_oob"])
        if score is None:
            continue
        app_id = int(row["app_id"])

        # GPU family axis
        gpu_family = _resolve_gpu_family(row["gpu"], gpu_lookup)
        if gpu_family:
            pair_scores.setdefault((f"gpu:{gpu_family}", app_id), []).append(score)

        # Variant axis
        variant = row["variant"]
        if variant:
            pair_scores.setdefault((f"var:{variant}", app_id), []).append(score)

        # Engine axis
        eng = engine_lookup.get(app_id)
        if eng:
            pair_scores.setdefault((f"eng:{eng}", app_id), []).append(score)

        # Steam Deck axis
        gpu_lower = (row["gpu"] or "").lower()
        is_deck = "vangogh" in gpu_lower or "van gogh" in gpu_lower
        deck_key = "deck:yes" if is_deck else "deck:no"
        pair_scores.setdefault((deck_key, app_id), []).append(score)

    if not pair_scores:
        logger.warning("No valid data for extended co-occurrence matrix")
        return csr_matrix((0, 0)), [], [], []

    axes_set = sorted({k[0] for k in pair_scores})
    games_set = sorted({k[1] for k in pair_scores})
    axis_idx = {a: i for i, a in enumerate(axes_set)}
    game_idx = {g: i for i, g in enumerate(games_set)}

    row_indices = []
    col_indices = []
    values = []
    for (axis, app_id), scores in pair_scores.items():
        row_indices.append(axis_idx[axis])
        col_indices.append(game_idx[app_id])
        values.append(np.mean(scores))

    matrix = csr_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(axes_set), len(games_set)),
    )

    gpu_count = sum(1 for a in axes_set if a.startswith("gpu:"))
    var_count = sum(1 for a in axes_set if a.startswith("var:"))
    eng_count = sum(1 for a in axes_set if a.startswith("eng:"))
    deck_count = sum(1 for a in axes_set if a.startswith("deck:"))
    logger.info(
        "Extended matrix: %d axes (%d gpu, %d variant, %d engine, %d deck) x %d games, %d entries",
        len(axes_set), gpu_count, var_count, eng_count, deck_count,
        len(games_set), len(values),
    )

    gpu_families = [a.replace("gpu:", "") for a in axes_set if a.startswith("gpu:")]
    return matrix, axes_set, games_set, gpu_families


def build_embeddings(
    conn: sqlite3.Connection,
    gpu_lookup: dict[str, dict[str, Any]],
    cpu_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build GPU and game embeddings via extended co-occurrence SVD.

    Uses a multi-axis co-occurrence matrix (gpu_family + variant + engine + deck)
    for richer game embeddings. GPU embeddings are extracted from the GPU-family
    rows of the left singular vectors.

    Returns a dict with:
        gpu_embeddings, game_embeddings,
        gpu_families, game_ids,
        n_components_gpu
    """
    matrix, axes, game_ids, gpu_families = _build_extended_cooccurrence(conn, gpu_lookup)

    gpu_embeddings = np.array([])
    game_embeddings = np.array([])
    n_components_gpu = 0

    if len(axes) > 1 and len(game_ids) > 1:
        n_components_gpu = _select_n_components(matrix)
        svd = TruncatedSVD(n_components=n_components_gpu, random_state=42)
        axes_embeddings = svd.fit_transform(matrix)
        game_embeddings = svd.components_.T  # (n_games, n_components)

        # Extract GPU-family rows from axes embeddings
        gpu_indices = [i for i, a in enumerate(axes) if a.startswith("gpu:")]
        if gpu_indices:
            gpu_embeddings = axes_embeddings[gpu_indices]

        logger.info(
            "GPU embeddings: %s, Game embeddings: %s",
            gpu_embeddings.shape,
            game_embeddings.shape,
        )
    else:
        logger.warning("Not enough data for SVD (axes=%d, games=%d)",
                        len(axes), len(game_ids))

    return {
        "gpu_embeddings": gpu_embeddings,
        "cpu_embeddings": np.array([]),
        "game_embeddings": game_embeddings,
        "gpu_families": gpu_families,
        "cpu_families": [],
        "game_ids": game_ids,
        "n_components_gpu": n_components_gpu,
        "n_components_cpu": 0,
    }


def build_text_embeddings(
    conn: sqlite3.Connection,
    n_components: int = 32,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Build sentence embeddings from notes_verdict via sentence-transformers + SVD.

    Returns dict with:
        text_embeddings: (n_reports, n_components) SVD-reduced embeddings
        text_report_ids: list of report IDs (aligned with rows)
        text_svd_components: SVD components for transforming new texts
        text_svd_mean: mean vector for centering
        text_n_components: number of SVD dimensions
        text_model_name: sentence-transformers model used
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Building text embeddings (model=%s, SVD=%d)...", model_name, n_components)

    # Fetch notes_verdict for all reports with valid targets
    rows = conn.execute(
        "SELECT id, notes_verdict, verdict, verdict_oob FROM reports"
    ).fetchall()

    report_ids = []
    texts = []
    for row in rows:
        # Only include reports that have valid targets
        v, v_oob = row["verdict"], row["verdict_oob"]
        if v_oob == "yes" or (v_oob == "no" and v == "yes") or v == "no" or v == "yes":
            report_ids.append(row["id"])
            texts.append(row["notes_verdict"] or "")

    logger.info("Encoding %d texts (%d non-empty)...",
                len(texts), sum(1 for t in texts if t.strip()))

    # Encode
    st_model = SentenceTransformer(model_name)
    non_empty_mask = np.array([bool(t.strip()) for t in texts])
    non_empty_texts = [t for t in texts if t.strip()]

    dim = st_model.get_sentence_embedding_dimension()
    raw_embeddings = np.zeros((len(texts), dim), dtype=np.float32)

    if non_empty_texts:
        encoded = st_model.encode(non_empty_texts, batch_size=256, show_progress_bar=True)
        raw_embeddings[non_empty_mask] = encoded

    logger.info("Encoded %d texts, coverage %.1f%%",
                len(non_empty_texts), non_empty_mask.mean() * 100)

    # SVD on non-empty embeddings
    non_empty_embs = raw_embeddings[non_empty_mask]
    mean_vec = non_empty_embs.mean(axis=0)
    centered = non_empty_embs - mean_vec

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(centered)
    explained = svd.explained_variance_ratio_.sum()
    logger.info("Text SVD: %d dims, explained variance: %.3f", n_components, explained)

    # Transform all (centered)
    all_centered = raw_embeddings - mean_vec
    reduced = svd.transform(all_centered)

    # Set empty texts to NaN
    reduced[~non_empty_mask] = np.nan

    return {
        "text_embeddings": reduced,
        "text_report_ids": report_ids,
        "text_svd_components": svd.components_,
        "text_svd_mean": mean_vec,
        "text_n_components": n_components,
        "text_model_name": model_name,
    }


def save_embeddings(emb_data: dict[str, Any], path: Path) -> None:
    """Save embeddings to .npz file."""
    save_dict = {}
    for key in ("gpu_embeddings", "cpu_embeddings", "game_embeddings"):
        arr = emb_data[key]
        if isinstance(arr, np.ndarray) and arr.size > 0:
            save_dict[key] = arr
        else:
            save_dict[key] = np.array([])

    for key in ("gpu_families", "cpu_families", "game_ids"):
        val = emb_data[key]
        if val:
            save_dict[key] = np.array(val)
        else:
            save_dict[key] = np.array([])

    # Text embeddings (optional)
    if "text_embeddings" in emb_data:
        save_dict["text_embeddings"] = emb_data["text_embeddings"]
        save_dict["text_report_ids"] = np.array(emb_data["text_report_ids"])
        save_dict["text_svd_components"] = emb_data["text_svd_components"]
        save_dict["text_svd_mean"] = emb_data["text_svd_mean"]

    # Game aggregates (Phase 9.2, optional)
    if "game_agg_cust" in emb_data:
        save_dict["game_agg_cust"] = emb_data["game_agg_cust"]
        save_dict["game_agg_flag"] = emb_data["game_agg_flag"]
        save_dict["game_agg_columns_cust"] = np.array(emb_data["game_agg_columns_cust"])
        save_dict["game_agg_columns_flag"] = np.array(emb_data["game_agg_columns_flag"])

    np.savez(path, **save_dict)
    logger.info("Saved embeddings to %s", path)


def load_embeddings(path: Path) -> dict[str, Any]:
    """Load embeddings from .npz file."""
    data = np.load(path, allow_pickle=True)
    gpu_emb = data["gpu_embeddings"]
    result = {
        "gpu_embeddings": gpu_emb,
        "cpu_embeddings": data["cpu_embeddings"],
        "game_embeddings": data["game_embeddings"],
        "gpu_families": list(data["gpu_families"]) if data["gpu_families"].size > 0 else [],
        "cpu_families": list(data["cpu_families"]) if data["cpu_families"].size > 0 else [],
        "game_ids": list(data["game_ids"]) if data["game_ids"].size > 0 else [],
        "n_components_gpu": gpu_emb.shape[1] if gpu_emb.ndim == 2 else 0,
        "n_components_cpu": 0,
    }

    # Text embeddings (optional, backwards compatible)
    if "text_embeddings" in data and data["text_embeddings"].size > 0:
        result["text_embeddings"] = data["text_embeddings"]
        result["text_report_ids"] = list(data["text_report_ids"])
        result["text_svd_components"] = data["text_svd_components"]
        result["text_svd_mean"] = data["text_svd_mean"]
        result["text_n_components"] = data["text_embeddings"].shape[1]

    # Game aggregates (Phase 9.2, optional, backwards compatible)
    if "game_agg_cust" in data and data["game_agg_cust"].size > 0:
        result["game_agg_cust"] = data["game_agg_cust"]
        result["game_agg_flag"] = data["game_agg_flag"]
        result["game_agg_columns_cust"] = list(data["game_agg_columns_cust"])
        result["game_agg_columns_flag"] = list(data["game_agg_columns_flag"])

    return result
