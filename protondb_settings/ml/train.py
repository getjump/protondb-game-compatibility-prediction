"""Main training orchestrator for the ML pipeline."""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Keyword patterns for text features (Group E) ─────────────────

_CRASH_RE = re.compile(
    r"\b(crash|crashes|crashing|segfault|sigsegv|sigabrt|freeze|freezes|freezing"
    r"|hang|hangs|hanging|won.?t\s+start|won.?t\s+launch|doesn.?t\s+start"
    r"|doesn.?t\s+launch|fail\s+to\s+start|fail\s+to\s+launch|broken|unplayable)\b",
    re.IGNORECASE,
)
_FIX_RE = re.compile(
    r"\b(fix|fixed|fixes|workaround|tweak|tweaked|solved|solution"
    r"|resolved|launch\s+option|protontricks|winetricks"
    r"|you\s+need\s+to|have\s+to|must\s+set|try\s+setting)\b",
    re.IGNORECASE,
)
_PERFECT_RE = re.compile(
    r"\b(perfect|flawless|no\s+issues|works?\s+great|works?\s+perfectly"
    r"|works?\s+fine|out\s+of\s+the\s+box|smooth|excellent|without\s+any\s+issue"
    r"|no\s+problems|runs?\s+great|runs?\s+perfectly|just\s+works)\b",
    re.IGNORECASE,
)
_PROTON_VER_RE = re.compile(
    r"\b(proton\s*\d|ge[-\s]?proton|proton[-\s]?ge|proton\s+experimental"
    r"|proton\s+hotfix|proton\s+[\d.]+)\b",
    re.IGNORECASE,
)
_ENV_VAR_RE = re.compile(r"\b[A-Z_]{3,}=[^\s]+")
_PERF_RE = re.compile(
    r"\b(lag|laggy|lagging|stutter|stutters|stuttering|fps|frame.?rate"
    r"|slow|sluggish|performance|choppy)\b",
    re.IGNORECASE,
)
_NEG_WORDS_RE = re.compile(
    r"\b(broken|unplayable|garbage|terrible|horrible|awful|worst|useless"
    r"|waste|disappointed|frustrating|unbearable|atrocious|dreadful)\b",
    re.IGNORECASE,
)
_POS_WORDS_RE = re.compile(
    r"\b(great|excellent|smooth|perfect|fantastic|amazing|wonderful"
    r"|awesome|brilliant|superb|stellar|flawless|solid|stable)\b",
    re.IGNORECASE,
)


def _has_re(pattern: re.Pattern, text: str | None) -> int:
    """Return 1 if pattern matches text, 0 otherwise."""
    if not isinstance(text, str):
        return 0
    return 1 if pattern.search(text) else 0


def _count_re(pattern: re.Pattern, text: str | None) -> int:
    """Count pattern matches in text."""
    if not isinstance(text, str):
        return 0
    return len(pattern.findall(text))


def _compute_target(verdict: str | None, verdict_oob: str | None) -> int | None:
    """Compute multi-class target: 0=borked, 1=tinkering, 2=works_oob."""
    if verdict_oob == "yes":
        return 2
    if verdict_oob == "no" and verdict == "yes":
        return 1
    if verdict == "no":
        return 0
    if verdict == "yes":
        return 1
    return None


    # _compute_aggregated_report_features removed after Phase 7 ablation.
    # All per-game aggregated features had ΔF1 < 0.002 (see PLAN_ML_7.md).
    # Game embeddings (SVD) capture game-level signal much better.


def _build_feature_matrix(
    conn: sqlite3.Connection,
    emb_data: dict[str, Any],
    progress_callback=None,
    normalized_data_source: str = "heuristic",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], Any]:
    """Build the full feature matrix from all data sources (vectorized).

    Returns:
        X: Feature DataFrame
        y: Target array
        timestamps: Unix timestamp array (for time-based split)
        report_ids: list of report IDs (aligned with rows)
        label_maps: Fitted LabelMaps
    """
    from .features.encoding import LabelMaps, extract_gpu_family, extract_gpu_vendor
    from .features.game import build_game_aggregates, AGG_CUST_COLS, AGG_FLAG_COLS
    from .features.hardware import build_hardware_lookups

    t0 = time.time()

    # Build lookups
    gpu_lookup, cpu_lookup, driver_lookup = build_hardware_lookups(conn, source=normalized_data_source)
    game_agg_lookup = build_game_aggregates(conn)

    # ── Step 1: Load reports into DataFrame via SQL ──────────────────
    logger.info("Fetching reports into DataFrame...")
    df = pd.read_sql_query(
        "SELECT id, app_id, timestamp, gpu, gpu_driver, battery_performance, variant, "
        "verdict, verdict_oob, "
        "concluding_notes, notes_verdict, notes_extra, notes_customizations, "
        "notes_audio_faults, notes_graphical_faults, notes_performance_faults, "
        "notes_stability_faults, notes_windowing_faults, notes_input_faults, "
        "notes_significant_bugs, notes_save_game_faults, notes_concluding_notes "
        "FROM reports",
        conn,
    )
    logger.info("Loaded %d reports in %.1fs", len(df), time.time() - t0)

    if progress_callback:
        progress_callback(0, 10)  # rough progress steps

    # ── Step 2: Compute targets (vectorized) ─────────────────────────
    verdict = df["verdict"]
    verdict_oob = df["verdict_oob"]

    y_series = pd.Series(np.full(len(df), -1, dtype=np.int32), index=df.index)
    y_series[verdict_oob == "yes"] = 2
    y_series[(verdict_oob == "no") & (verdict == "yes")] = 1
    y_series[(y_series == -1) & (verdict == "no")] = 0
    y_series[(y_series == -1) & (verdict == "yes")] = 1

    valid_mask = y_series >= 0
    df = df[valid_mask].reset_index(drop=True)
    y = y_series[valid_mask].values

    report_ids_out = df["id"].tolist()
    ts = df["timestamp"].fillna(0).astype(np.int64).values

    logger.info("Valid samples: %d (%.1f%%)", len(df), len(df) / max(1, len(valid_mask)) * 100)

    if progress_callback:
        progress_callback(1, 10)

    # ── Step 3: Hardware features (vectorized via lookup maps) ───────
    # GPU family: lookup table → Series.map, fallback to heuristic
    gpu_family_map = {raw: data["family"] for raw, data in gpu_lookup.items()
                      if data.get("vendor", "").lower() != "unknown"}
    df["gpu_family"] = df["gpu"].map(gpu_family_map)
    # Fallback for unmapped GPUs
    unmapped = df["gpu_family"].isna() & df["gpu"].notna()
    if unmapped.any():
        df.loc[unmapped, "gpu_family"] = df.loc[unmapped, "gpu"].apply(extract_gpu_family)

    # Driver features: split by vendor
    nv_map = {}
    mesa_map = {}
    for raw, data in driver_lookup.items():
        vendor = data.get("driver_vendor")
        major = data.get("driver_major")
        minor = data.get("driver_minor")
        if vendor == "nvidia" and major is not None:
            nv_map[raw] = major + (minor or 0) / 1000.0
        elif vendor == "mesa" and major is not None:
            mesa_map[raw] = major + (minor or 0) / 10.0
    df["nvidia_driver_version"] = df["gpu_driver"].map(nv_map)
    df["mesa_driver_version"] = df["gpu_driver"].map(mesa_map)

    # APU/iGPU/mobile from GPU lookup
    for flag in ("is_apu", "is_igpu", "is_mobile"):
        flag_map = {raw: data.get(flag, 0) or 0 for raw, data in gpu_lookup.items()}
        df[flag] = df["gpu"].map(flag_map).fillna(0).astype(int)

    # Steam Deck detection
    gpu_lower = df["gpu"].fillna("").str.lower()
    df["is_steam_deck"] = ((gpu_lower.str.contains("vangogh", na=False) |
                            gpu_lower.str.contains("van gogh", na=False) |
                            df["battery_performance"].notna())).astype(int)

    if progress_callback:
        progress_callback(3, 10)

    # ── Step 4: Text features (vectorized) ───────────────────────────
    fault_cols = [
        "notes_audio_faults", "notes_graphical_faults",
        "notes_performance_faults", "notes_stability_faults",
        "notes_windowing_faults", "notes_input_faults",
        "notes_significant_bugs", "notes_save_game_faults",
    ]
    text_cols = [
        "concluding_notes", "notes_verdict", "notes_extra",
        "notes_customizations", "notes_concluding_notes",
    ] + fault_cols

    # Concatenate all text fields into one (reduce with + is faster than apply)
    all_text = df[text_cols[0]].fillna("")
    for col in text_cols[1:]:
        all_text = all_text + " " + df[col].fillna("")
    all_text = all_text.str.strip().replace("", np.nan)

    # Group D: meta-features
    df["has_concluding_notes"] = df["concluding_notes"].notna().astype(int)
    df["concluding_notes_length"] = df["concluding_notes"].fillna("").str.len()
    # Count non-empty fault note fields (vectorized)
    df["fault_notes_count"] = sum(
        (df[col].notna() & (df[col].str.strip() != "")).astype(int)
        for col in fault_cols
    )
    df["has_customization_notes"] = df["notes_customizations"].notna().astype(int)
    df["total_notes_length"] = all_text.fillna("").str.len()

    # Group E: keyword regex features (vectorized via numpy apply)
    text_s = all_text.fillna("")
    text_arr = text_s.values
    df["mentions_crash"] = np.array([_has_re(_CRASH_RE, t) for t in text_arr], dtype=np.int8)
    df["mentions_fix"] = np.array([_has_re(_FIX_RE, t) for t in text_arr], dtype=np.int8)
    df["mentions_perfect"] = np.array([_has_re(_PERFECT_RE, t) for t in text_arr], dtype=np.int8)
    df["mentions_proton_version"] = np.array([_has_re(_PROTON_VER_RE, t) for t in text_arr], dtype=np.int8)
    df["mentions_env_var"] = np.array([_has_re(_ENV_VAR_RE, t) for t in text_arr], dtype=np.int8)
    df["mentions_performance"] = np.array([_has_re(_PERF_RE, t) for t in text_arr], dtype=np.int8)
    df["sentiment_negative_words"] = np.array([_count_re(_NEG_WORDS_RE, t) for t in text_arr], dtype=np.int16)
    df["sentiment_positive_words"] = np.array([_count_re(_POS_WORDS_RE, t) for t in text_arr], dtype=np.int16)

    if progress_callback:
        progress_callback(5, 10)

    # ── Step 5: Per-game aggregate features (Phase 9.2) ──────────────
    agg_cols = AGG_CUST_COLS + AGG_FLAG_COLS
    if game_agg_lookup:
        agg_df = pd.DataFrame.from_dict(game_agg_lookup, orient="index")
        agg_df.index.name = "app_id"
        agg_df = agg_df.reindex(columns=agg_cols)
        # Map via app_id
        agg_mapped = agg_df.reindex(df["app_id"].values)
        agg_mapped.index = df.index
        for col in agg_cols:
            df[col] = agg_mapped[col].values if col in agg_mapped.columns else np.nan

    if progress_callback:
        progress_callback(6, 10)

    # ── Step 6: Embedding features (vectorized via fancy indexing) ───
    n_gpu_emb = emb_data.get("n_components_gpu", 0)
    gpu_emb_matrix = emb_data.get("gpu_embeddings", np.array([]))
    game_emb_matrix = emb_data.get("game_embeddings", np.array([]))

    # GPU embeddings: gpu_family → index → embedding row
    emb_frames = []
    if n_gpu_emb > 0 and gpu_emb_matrix.size > 0:
        gpu_family_to_idx = {f: i for i, f in enumerate(emb_data.get("gpu_families", []))}
        gpu_idx = df["gpu_family"].map(gpu_family_to_idx)
        valid_gpu = gpu_idx.notna()
        gpu_emb_arr = np.full((len(df), n_gpu_emb), np.nan)
        if valid_gpu.any():
            gpu_emb_arr[valid_gpu.values] = gpu_emb_matrix[gpu_idx[valid_gpu].astype(int).values]
        emb_frames.append(pd.DataFrame(
            gpu_emb_arr, columns=[f"gpu_emb_{d}" for d in range(n_gpu_emb)], index=df.index,
        ))

    # Game embeddings: app_id → index → embedding row
    if n_gpu_emb > 0 and game_emb_matrix.size > 0:
        game_id_to_idx = {int(g): i for i, g in enumerate(emb_data.get("game_ids", []))}
        game_idx = df["app_id"].map(game_id_to_idx)
        valid_game = game_idx.notna()
        game_emb_arr = np.full((len(df), n_gpu_emb), np.nan)
        if valid_game.any():
            game_emb_arr[valid_game.values] = game_emb_matrix[game_idx[valid_game].astype(int).values]
        emb_frames.append(pd.DataFrame(
            game_emb_arr, columns=[f"game_emb_{d}" for d in range(n_gpu_emb)], index=df.index,
        ))

    if progress_callback:
        progress_callback(8, 10)

    # Text embeddings: report_id → index → embedding row
    n_text_emb = emb_data.get("text_n_components", 0)
    text_emb_matrix = emb_data.get("text_embeddings", np.array([]))
    if n_text_emb > 0 and text_emb_matrix.size > 0:
        text_report_ids = emb_data.get("text_report_ids", [])
        text_rid_to_idx = pd.Series(range(len(text_report_ids)), index=text_report_ids)
        text_idx = df["id"].map(text_rid_to_idx)
        valid_text = text_idx.notna()
        text_emb_arr = np.full((len(df), n_text_emb), np.nan)
        if valid_text.any():
            text_emb_arr[valid_text.values] = text_emb_matrix[text_idx[valid_text].astype(int).values]
        emb_frames.append(pd.DataFrame(
            text_emb_arr, columns=[f"text_emb_{d}" for d in range(n_text_emb)], index=df.index,
        ))

    # Join all embedding columns at once to avoid fragmentation
    if emb_frames:
        df = pd.concat([df] + emb_frames, axis=1)

    if progress_callback:
        progress_callback(9, 10)

    # ── Step 7: Build final X ────────────────────────────────────────
    # Select only feature columns (drop raw/intermediate columns)
    drop_cols = [
        "id", "app_id", "timestamp", "gpu", "gpu_driver", "battery_performance",
        "verdict", "verdict_oob",
        "concluding_notes", "notes_verdict", "notes_extra", "notes_customizations",
        "notes_audio_faults", "notes_graphical_faults", "notes_performance_faults",
        "notes_stability_faults", "notes_windowing_faults", "notes_input_faults",
        "notes_significant_bugs", "notes_save_game_faults", "notes_concluding_notes",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Coerce numeric columns
    for col in ("nvidia_driver_version", "mesa_driver_version"):
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    if progress_callback:
        progress_callback(10, 10)

    elapsed = time.time() - t0
    logger.info("Feature matrix: %d samples, %d features (%.1fs)", X.shape[0], X.shape[1], elapsed)

    # Fit label encoders on categorical columns
    label_maps = LabelMaps()
    cat_cols = ["gpu_family"]
    for col in cat_cols:
        if col in X.columns:
            label_maps.fit_column(col, X[col].tolist(), top_n=100)

    return X, y, ts, report_ids_out, label_maps


def _time_based_split(
    X: pd.DataFrame,
    y: np.ndarray,
    timestamps: np.ndarray,
    test_fraction: float = 0.2,
    report_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    """Split data by time: oldest reports for training, newest for testing.

    Returns:
        X_train, X_test, y_train, y_test, train_report_ids, test_report_ids
    """
    sorted_indices = np.argsort(timestamps)
    split_point = int(len(sorted_indices) * (1 - test_fraction))

    train_idx = sorted_indices[:split_point]
    test_idx = sorted_indices[split_point:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    if report_ids is not None:
        train_rids = [report_ids[i] for i in train_idx]
        test_rids = [report_ids[i] for i in test_idx]
    else:
        train_rids = []
        test_rids = []

    logger.info(
        "Time-based split: %d train, %d test (split at index %d)",
        len(X_train),
        len(X_test),
        split_point,
    )
    return X_train, X_test, y_train, y_test, train_rids, test_rids


def _register_pipeline_run(
    conn: sqlite3.Connection, step: str, status: str = "running", total: int = 0
) -> int:
    """Register a pipeline run in the pipeline_runs table."""
    cur = conn.execute(
        "INSERT INTO pipeline_runs (step, status, total_items, processed) VALUES (?, ?, ?, 0)",
        (step, status, total),
    )
    conn.commit()
    return cur.lastrowid


def _update_pipeline_run(
    conn: sqlite3.Connection, run_id: int, status: str, processed: int = 0, error: str | None = None
) -> None:
    """Update a pipeline run status."""
    conn.execute(
        "UPDATE pipeline_runs SET status=?, processed=?, finished_at=datetime('now'), error=? WHERE id=?",
        (status, processed, error, run_id),
    )
    conn.commit()


def train_pipeline(
    conn: sqlite3.Connection,
    output_dir: str | Path = "data/",
    test_fraction: float = 0.2,
    normalized_data_source: str | None = None,
) -> dict[str, Any]:
    """Run the full ML training pipeline.

    Steps:
    1. Build embeddings (SVD)
    2. Extract features (all sources)
    3. Train/test split by time
    4. Train LightGBM
    5. Evaluate (accuracy, F1, confusion matrix)
    6. Export model.pkl + embeddings.npz

    Returns evaluation results dict.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    from .evaluate import evaluate_model, print_results
    from .export import export_all
    from .features.embeddings import build_embeddings
    from .features.hardware import build_hardware_lookups
    from .models.classifier import train_classifier

    console = Console()
    output_dir = Path(output_dir)

    # Register pipeline run
    run_id = _register_pipeline_run(conn, "ml_train")

    try:
        start_time = time.time()

        # Step 1: Build embeddings
        console.print("\n[bold]Step 1/6: Building embeddings (SVD)...[/bold]")
        if normalized_data_source is None:
            from protondb_settings.config import NORMALIZED_DATA_SOURCE
            normalized_data_source = NORMALIZED_DATA_SOURCE
        console.print(f"  Normalized data source: [cyan]{normalized_data_source}[/cyan]")

        gpu_lookup, cpu_lookup, _driver_lookup = build_hardware_lookups(conn, source=normalized_data_source)
        emb_data = build_embeddings(conn, gpu_lookup, cpu_lookup)
        console.print(
            f"  GPU: {emb_data['n_components_gpu']} dims, "
            f"{len(emb_data['gpu_families'])} families"
        )
        console.print(
            f"  Games: {len(emb_data['game_ids'])} games with embeddings"
        )

        # Step 2: Extract features
        console.print("\n[bold]Step 2/6: Extracting features...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Building features", total=None)

            def _progress_cb(current, total):
                progress.update(task, completed=current, total=total)

            X, y, timestamps, _report_ids, label_maps = _build_feature_matrix(
                conn, emb_data, progress_callback=_progress_cb,
                normalized_data_source=normalized_data_source,
            )

        console.print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

        # Target distribution
        unique, counts = np.unique(y, return_counts=True)
        for val, cnt in zip(unique, counts):
            from .models.classifier import TARGET_NAMES
            console.print(f"  {TARGET_NAMES.get(val, val)}: {cnt} ({cnt/len(y)*100:.1f}%)")

        # Step 3: Train/test split
        console.print("\n[bold]Step 3/6: Time-based train/test split...[/bold]")
        X_train, X_test, y_train, y_test, _, _ = _time_based_split(X, y, timestamps, test_fraction)
        console.print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

        # Step 4: Train
        console.print("\n[bold]Step 4/6: Training LightGBM...[/bold]")
        model = train_classifier(X_train, y_train, X_test, y_test)
        console.print(f"  Best iteration: {model.best_iteration_}")

        # Step 5: Evaluate
        console.print("\n[bold]Step 5/6: Evaluating...[/bold]")
        results = evaluate_model(model, X_test, y_test, feature_names=list(X.columns))
        print_results(results)

        # Step 6: Export
        console.print("\n[bold]Step 6/6: Exporting artifacts...[/bold]")
        paths = export_all(model, emb_data, label_maps, output_dir)
        for name, path in paths.items():
            console.print(f"  {name}: {path}")

        elapsed = time.time() - start_time
        console.print(f"\n[green]Training complete in {elapsed:.1f}s[/green]")

        _update_pipeline_run(conn, run_id, "completed", processed=X.shape[0])
        results["elapsed_seconds"] = elapsed
        return results

    except Exception as e:
        _update_pipeline_run(conn, run_id, "failed", error=str(e))
        logger.error("Training pipeline failed: %s", e)
        raise


def train_cascade_pipeline(
    conn: sqlite3.Connection,
    output_dir: str | Path = "data/",
    test_fraction: float = 0.2,
    normalized_data_source: str | None = None,
    embeddings_path: str | Path | None = None,
    force_embeddings: bool = False,
    reuse_stage1: str | Path | None = None,
) -> dict[str, Any]:
    """Run cascade ML training: Stage 1 (borked/works) + Stage 2 (tinkering/oob).

    Args:
        embeddings_path: Path to existing embeddings.npz to reuse.
            If None, looks for {output_dir}/embeddings.npz.
        force_embeddings: If True, rebuild all embeddings from scratch
            even if cached file exists.
        reuse_stage1: Path to a saved Stage 1 model (.pkl) to skip
            Stage 1 training. Useful for experiments that only change
            Stage 2 (relabeling, tinkering/oob boundary).

    Returns evaluation results dict.
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    from .evaluate import evaluate_model, print_results
    from .export import export_all
    from .features.embeddings import build_embeddings, build_text_embeddings, load_embeddings, save_embeddings
    from .features.hardware import build_hardware_lookups
    from .models.cascade import CascadeClassifier, train_stage1, train_stage2
    from .relabeling import apply_relabeling, get_relabel_ids

    console = Console()
    output_dir = Path(output_dir)

    run_id = _register_pipeline_run(conn, "ml_train_cascade")

    try:
        start_time = time.time()

        if normalized_data_source is None:
            from protondb_settings.config import NORMALIZED_DATA_SOURCE
            normalized_data_source = NORMALIZED_DATA_SOURCE

        # Resolve cached embeddings path
        if embeddings_path is not None:
            cached_emb_path = Path(embeddings_path)
        else:
            cached_emb_path = output_dir / "embeddings.npz"

        cached_emb = None
        if not force_embeddings and cached_emb_path.exists():
            console.print(f"\n[bold]Loading cached embeddings from {cached_emb_path}...[/bold]")
            cached_emb = load_embeddings(cached_emb_path)

        # Step 1: HW/game embeddings (SVD) — reuse if cached
        if cached_emb is not None and cached_emb.get("n_components_gpu", 0) > 0:
            console.print("[bold]Step 1/9: HW/game embeddings — cached[/bold]")
            emb_data = cached_emb
            console.print(
                f"  GPU: {emb_data['n_components_gpu']} dims, "
                f"{len(emb_data['gpu_families'])} families, "
                f"Games: {len(emb_data['game_ids'])}"
            )
        else:
            console.print("\n[bold]Step 1/9: Building HW/game embeddings (SVD)...[/bold]")
            gpu_lookup, cpu_lookup, _driver_lookup = build_hardware_lookups(conn, source=normalized_data_source)
            emb_data = build_embeddings(conn, gpu_lookup, cpu_lookup)
            console.print(
                f"  GPU: {emb_data['n_components_gpu']} dims, "
                f"{len(emb_data['gpu_families'])} families, "
                f"Games: {len(emb_data['game_ids'])}"
            )

        # Step 2: Text embeddings — reuse if cached
        if cached_emb is not None and cached_emb.get("text_n_components", 0) > 0:
            console.print("[bold]Step 2/9: Text embeddings — cached[/bold]")
            # Merge text data into emb_data (if loaded from separate build)
            for key in ("text_embeddings", "text_report_ids", "text_svd_components",
                        "text_svd_mean", "text_n_components", "text_model_name"):
                if key in cached_emb:
                    emb_data[key] = cached_emb[key]
            n_text = emb_data.get("text_n_components", 0)
            n_reports = len(emb_data.get("text_report_ids", []))
            console.print(f"  Text: {n_text} dims, {n_reports} reports")
        else:
            console.print("\n[bold]Step 2/9: Building text embeddings (sentence-transformers)...[/bold]")
            text_emb_data = build_text_embeddings(conn, n_components=32)
            emb_data.update(text_emb_data)
            console.print(
                f"  Text: {text_emb_data['text_n_components']} dims, "
                f"{len(text_emb_data['text_report_ids'])} reports"
            )

        # Step 2b: Build per-game aggregates (Phase 9.2)
        from .features.game import build_game_aggregates, game_aggregates_to_arrays
        game_agg_lookup = build_game_aggregates(conn)
        agg_arrays = game_aggregates_to_arrays(game_agg_lookup, emb_data["game_ids"])
        emb_data.update(agg_arrays)
        console.print(f"  Game aggregates: {len(game_agg_lookup)} games, "
                      f"{len(agg_arrays['game_agg_columns_cust']) + len(agg_arrays['game_agg_columns_flag'])} features")

        # Step 3: Extract features
        console.print("\n[bold]Step 3/9: Extracting features...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Building features", total=None)

            def _progress_cb(current, total):
                progress.update(task, completed=current, total=total)

            X, y, timestamps, report_ids, label_maps = _build_feature_matrix(
                conn, emb_data, progress_callback=_progress_cb,
                normalized_data_source=normalized_data_source,
            )

        console.print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

        from .models.classifier import TARGET_NAMES
        unique, counts = np.unique(y, return_counts=True)
        for val, cnt in zip(unique, counts):
            console.print(f"  {TARGET_NAMES.get(val, val)}: {cnt} ({cnt/len(y)*100:.1f}%)")

        # Step 3b: IRT fitting + contributor-aware relabeling (Phase 12.8 + 13.2)
        console.print("\n[bold]Step 3b/9: IRT fitting (annotator-game decomposition)...[/bold]")
        from .irt import fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features

        irt_theta, irt_difficulty = fit_irt(conn)
        if irt_theta:
            console.print(f"  IRT: {len(irt_theta)} contributors, {len(irt_difficulty)} items")
        else:
            console.print("  [yellow]IRT: not enough contributor data, skipping[/yellow]")

        relabel_ids = get_relabel_ids(conn)

        # Step 4: Train/test/calibration split
        console.print("\n[bold]Step 4/9: Time-based train/test split...[/bold]")
        X_train, X_test, y_train, y_test, train_rids, _test_rids = _time_based_split(
            X, y, timestamps, test_fraction, report_ids=report_ids,
        )

        # Add IRT features (Phase 12.8: +0.030 F1)
        if irt_theta:
            console.print("  Adding IRT features...")
            X_train = add_irt_features(X_train, train_rids, conn, irt_theta, irt_difficulty)
            X_test = add_irt_features(X_test, _test_rids, conn, irt_theta, irt_difficulty)

        # Add error-targeted features (Phase 16.6: +0.006 F1 with class weighting)
        console.print("  Adding error-targeted features...")
        X_train = add_error_targeted_features(X_train, train_rids, conn)
        X_test = add_error_targeted_features(X_test, _test_rids, conn)

        # Contributor-aware relabeling (Phase 13.2: +0.015 F1)
        # Replaces Phase 8 blind relabeling + Cleanlab noise removal
        if irt_theta:
            y_train, n_relabeled = contributor_aware_relabel(
                y_train, train_rids, relabel_ids, conn, irt_theta,
                theta_threshold=1.83)
            console.print(f"  Contributor-aware relabel: {n_relabeled} tinkering → works_oob")
        else:
            # Fallback: Phase 8 + Cleanlab when no contributor data
            y_train, n_relabeled = apply_relabeling(y_train, train_rids, relabel_ids)
            console.print(f"  Fallback (Phase 8): relabeled {n_relabeled} tinkering → works_oob")

            from .noise import find_noisy_samples
            keep_mask = find_noisy_samples(X_train, y_train, frac_remove=0.03,
                                           cache_dir=output_dir, force=force_embeddings)
            n_removed = (~keep_mask).sum()
            X_train = X_train[keep_mask].reset_index(drop=True)
            y_train = y_train[keep_mask]
            train_rids = [rid for rid, keep in zip(train_rids, keep_mask) if keep]
            console.print(f"  Cleanlab: removed {n_removed} noisy samples")

        # Show post-cleaning distribution
        for val, name in [(0, "borked"), (1, "tinkering"), (2, "works_oob")]:
            n = (y_train == val).sum()
            console.print(f"  Train {name}: {n} ({n/len(y_train)*100:.1f}%)")

        # Reserve half of test for calibration
        n_cal = len(X_test) // 2
        X_cal = X_test.iloc[:n_cal].copy().reset_index(drop=True)
        y_cal = y_test[:n_cal]
        X_eval = X_test.iloc[n_cal:].copy().reset_index(drop=True)
        y_eval = y_test[n_cal:]
        eval_rids = _test_rids[n_cal:]

        # Preserve categorical dtypes
        from .models.classifier import CATEGORICAL_FEATURES as _CAT_FEATS
        for col in _CAT_FEATS:
            if col in X_cal.columns:
                X_cal[col] = X_cal[col].astype("category")
                X_eval[col] = X_eval[col].astype("category")

        console.print(f"  Train: {len(X_train)}, Calibration: {len(X_cal)}, Eval: {len(X_eval)}")

        # Step 4: Train Stage 1 (borked vs works)
        if reuse_stage1 is not None:
            console.print(f"\n[bold]Step 5/9: Reusing Stage 1 from {reuse_stage1}[/bold]")
            import joblib
            s1_model = joblib.load(reuse_stage1)
            console.print(f"  Loaded (best iteration: {s1_model.best_iteration_})")
        else:
            console.print("\n[bold]Step 5/9: Training Stage 1 (borked vs works)...[/bold]")
            s1_model = train_stage1(X_train, y_train, X_test, y_test)
            console.print(f"  Best iteration: {s1_model.best_iteration_}")

        y_s1_test = (y_test > 0).astype(int)
        y_s1_pred = s1_model.predict(X_test)
        borked_r = (y_s1_pred[y_s1_test == 0] == 0).mean()
        borked_p = (y_s1_test[y_s1_pred == 0] == 0).mean() if (y_s1_pred == 0).any() else 0
        console.print(f"  borked recall: {borked_r:.4f}, precision: {borked_p:.4f}")

        # Step 5: Train Stage 2 (tinkering vs works_oob, no report_age_days)
        console.print("\n[bold]Step 6/9: Training Stage 2 (tinkering vs works_oob)...[/bold]")
        s2_model, s2_dropped = train_stage2(X_train, y_train, X_test, y_test)
        console.print(f"  Best iteration: {s2_model.best_iteration}")
        console.print(f"  Dropped features: {s2_dropped}")

        # Step 6: Calibrate
        console.print("\n[bold]Step 7/9: Calibrating probabilities (isotonic)...[/bold]")
        cascade = CascadeClassifier(s1_model, s2_model, s2_dropped)
        cascade.fit_calibrators(X_cal, y_cal)

        # Compute ECE before/after
        from sklearn.metrics import brier_score_loss
        proba_raw = cascade.predict_proba(X_eval, calibrated=False)
        proba_cal = cascade.predict_proba(X_eval, calibrated=True)

        def _ece(y_true, y_proba, n_bins=10):
            y_pred = y_proba.argmax(axis=1)
            confs = y_proba.max(axis=1)
            accs = (y_pred == y_true).astype(float)
            ece = 0.0
            for i in range(n_bins):
                lo, hi = i / n_bins, (i + 1) / n_bins
                mask = (confs > lo) & (confs <= hi)
                if mask.sum() > 0:
                    ece += mask.sum() / len(y_true) * abs(confs[mask].mean() - accs[mask].mean())
            return ece

        ece_before = _ece(y_eval, proba_raw)
        ece_after = _ece(y_eval, proba_cal)
        console.print(f"  ECE: {ece_before:.4f} → {ece_after:.4f}")

        for cls in range(3):
            from .models.classifier import TARGET_NAMES
            y_bin = (y_eval == cls).astype(int)
            b_before = brier_score_loss(y_bin, proba_raw[:, cls])
            b_after = brier_score_loss(y_bin, proba_cal[:, cls])
            console.print(f"  Brier {TARGET_NAMES[cls]}: {b_before:.4f} → {b_after:.4f}")

        # Step 7: Evaluate cascade
        console.print("\n[bold]Step 8/9: Evaluating cascade...[/bold]")
        results = evaluate_model(cascade, X_eval, y_eval, feature_names=list(X.columns))
        print_results(results)

        # Confidence stats
        proba_full = cascade.predict_proba(X_eval)
        confidence = proba_full.max(axis=1)
        confident_mask = confidence >= 0.7
        n_confident = confident_mask.sum()
        if n_confident > 0:
            from sklearn.metrics import accuracy_score as _acc
            acc_conf = _acc(y_eval[confident_mask], proba_full[confident_mask].argmax(axis=1))
            console.print(f"\n  Confidence ≥ 0.7: {n_confident}/{len(y_eval)} "
                          f"({n_confident/len(y_eval)*100:.0f}%), accuracy={acc_conf:.4f}")

        results["ece_before"] = ece_before
        results["ece_after"] = ece_after

        # Per-game aggregated evaluation
        from .evaluate import evaluate_per_game, print_per_game_results
        per_game = evaluate_per_game(cascade, X_eval, y_eval, eval_rids, conn)
        print_per_game_results(per_game)
        results["per_game"] = per_game

        # Step 8: Export
        console.print("\n[bold]Step 9/9: Exporting artifacts...[/bold]")
        import joblib
        from .features.embeddings import save_embeddings

        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        cascade_path = output_dir / "model_cascade.pkl"
        joblib.dump(cascade, cascade_path)
        paths["cascade"] = cascade_path

        s1_path = output_dir / "model_stage1.pkl"
        joblib.dump(s1_model, s1_path)
        paths["stage1"] = s1_path

        s2_path = output_dir / "model_stage2.pkl"
        joblib.dump(s2_model, s2_path)
        paths["stage2"] = s2_path

        emb_path = output_dir / "embeddings.npz"
        save_embeddings(emb_data, emb_path)
        paths["embeddings"] = emb_path

        lm_path = output_dir / "label_maps.json"
        label_maps.save(lm_path)
        paths["label_maps"] = lm_path

        for name, path in paths.items():
            console.print(f"  {name}: {path}")

        elapsed = time.time() - start_time
        console.print(f"\n[green]Cascade training complete in {elapsed:.1f}s[/green]")

        _update_pipeline_run(conn, run_id, "completed", processed=X.shape[0])
        results["elapsed_seconds"] = elapsed
        return results

    except Exception as e:
        _update_pipeline_run(conn, run_id, "failed", error=str(e))
        logger.error("Cascade training pipeline failed: %s", e)
        raise
