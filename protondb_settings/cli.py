"""Click CLI — entry point for all commands."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()  # Load .env before any config imports that read os.environ

from protondb_settings.config import Config, DEFAULT_DB_PATH


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=DEFAULT_DB_PATH,
    envvar="PROTONDB_DB_PATH",
    help="Path to SQLite database file.",
)
@click.pass_context
def cli(ctx: click.Context, db: Path) -> None:
    """ProtonDB Recommended Settings — CLI interface."""
    ctx.ensure_object(Config)
    ctx.obj.db_path = db


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--port", type=int, default=None, help="HTTP port (default: 8080).")
@click.option("--host", type=str, default="0.0.0.0", help="Bind address.")
@pass_config
def serve(config: Config, port: int | None, host: str) -> None:
    """Start the FastAPI HTTP server."""
    import uvicorn

    from protondb_settings.api.app import create_app

    if port is not None:
        config.port = port

    app = create_app(config)
    uvicorn.run(app, host=host, port=config.port)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------


@cli.group()
@pass_config
def worker(config: Config) -> None:
    """Worker commands: check for updates, sync dump data."""


@worker.command("check")
@pass_config
def worker_check(config: Config) -> None:
    """Check GitHub for new ProtonDB data dump releases."""
    import asyncio

    from protondb_settings.worker.protondb import check_for_update

    asyncio.run(check_for_update(config.db_path))


@worker.command("sync")
@click.option(
    "--file",
    "local_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Import from a local JSON file instead of downloading.",
)
@pass_config
def worker_sync(config: Config, local_file: Path | None) -> None:
    """Download (or load local) dump and import into the database."""
    import asyncio

    from protondb_settings.worker.protondb import sync_dump

    asyncio.run(sync_dump(config.db_path, local_file=local_file))


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


@cli.group()
@pass_config
def preprocess(config: Config) -> None:
    """Preprocessing pipeline: cleaning, enrichment, LLM tasks."""


@preprocess.command("check")
@pass_config
def preprocess_check(config: Config) -> None:
    """Show status of all preprocessing steps."""
    import logging

    from rich.console import Console
    from rich.table import Table

    from protondb_settings.db.connection import get_connection
    from protondb_settings.db.migrations import ensure_schema

    logging.basicConfig(level=logging.WARNING)
    conn = get_connection(config.db_path)
    ensure_schema(conn)
    console = Console()

    table = Table(title="Preprocessing Status")
    table.add_column("Step", style="bold")
    table.add_column("Status")
    table.add_column("Pending")
    table.add_column("Last Run")

    # Data cleaning
    from protondb_settings.preprocessing.cleaning import get_pending_count as cleaning_pending

    cp = cleaning_pending(conn)
    _add_step_row(table, conn, "cleaning", "Data cleaning", cp)

    # Enrichment
    from protondb_settings.preprocessing.enrichment.main import (
        get_pending_count as enrichment_pending,
    )

    ep = enrichment_pending(conn)
    _add_step_row(table, conn, "enrichment", "Enrichment", ep)

    # GPU heuristic normalization
    from protondb_settings.preprocessing.normalize.gpu_heuristic import (
        get_pending_count as gpu_h_pending,
    )

    ghp = gpu_h_pending(conn)
    _add_step_row(table, conn, "normalize_gpu_heuristic", "GPU normalization (heuristic)", ghp)

    # CPU heuristic normalization
    from protondb_settings.preprocessing.normalize.cpu_heuristic import (
        get_pending_count as cpu_h_pending,
    )

    chp = cpu_h_pending(conn)
    _add_step_row(table, conn, "normalize_cpu_heuristic", "CPU normalization (heuristic)", chp)

    # GPU driver normalization
    from protondb_settings.preprocessing.normalize.gpu_driver_heuristic import (
        get_pending_count as drv_pending,
    )

    dp = drv_pending(conn)
    _add_step_row(table, conn, "normalize_gpu_driver", "GPU driver normalization", dp)

    # GPU normalization (LLM)
    from protondb_settings.preprocessing.normalize.gpu import (
        get_pending_count as gpu_pending,
    )

    gp = gpu_pending(conn)
    _add_step_row(table, conn, "normalize_gpu", "GPU normalization (LLM)", gp)

    # CPU normalization (LLM)
    from protondb_settings.preprocessing.normalize.cpu import (
        get_pending_count as cpu_pending,
    )

    cpp = cpu_pending(conn)
    _add_step_row(table, conn, "normalize_cpu", "CPU normalization (LLM)", cpp)

    # Launch options
    from protondb_settings.preprocessing.normalize.launch_options import (
        get_pending_count as lo_pending,
    )

    lp = lo_pending(conn)
    _add_step_row(table, conn, "parse_launch_options", "Launch options", lp)

    # Text extraction
    from protondb_settings.preprocessing.extract.extractor import (
        get_pending_count as extract_pending,
    )

    xp = extract_pending(conn)
    _add_step_row(table, conn, "extract", "Text extraction", xp)

    console.print(table)

    # AWACY stale check
    from protondb_settings.preprocessing.enrichment.sources.anticheat import check_awacy_stale

    if check_awacy_stale(conn):
        console.print("  [yellow]AreWeAntiCheatYet: stale or not fetched[/yellow]")
    else:
        console.print("  [green]AreWeAntiCheatYet: up to date[/green]")

    conn.close()


def _add_step_row(table, conn, step_name: str, display_name: str, pending: int) -> None:
    """Add a row to the status table for a given pipeline step."""
    row = conn.execute(
        "SELECT status, processed, total_items, finished_at, started_at "
        "FROM pipeline_runs WHERE step = ? ORDER BY started_at DESC LIMIT 1",
        (step_name,),
    ).fetchone()

    if row is None:
        status = "[dim]never run[/dim]"
        last_run = "-"
    else:
        s = row["status"]
        if s == "completed":
            status = f"[green]done ({row['processed']}/{row['total_items']})[/green]"
        elif s == "running":
            status = f"[yellow]running ({row['processed']}/{row['total_items']})[/yellow]"
        elif s == "failed":
            status = f"[red]failed ({row['processed']}/{row['total_items']})[/red]"
        else:
            status = s
        last_run = row["finished_at"] or row["started_at"]

    pending_str = str(pending) if pending > 0 else "[green]0[/green]"
    table.add_row(display_name, status, pending_str, last_run)


@preprocess.command("run")
@click.option("--step", type=click.Choice(["cleaning", "normalize", "enrichment"]), default=None,
              help="Run only a specific step.")
@click.option("--force", is_flag=True, help="Reset step data and start fresh.")
@click.option("--min-reports", type=int, default=1,
              help="Enrichment: only games with at least N reports.")
@click.option("--source", type=click.Choice(["steam", "pcgamingwiki", "protondb", "protondb_reports", "steam_pics", "anticheat"]),
              default=None, help="Enrichment: only run specific source.")
@click.option("--refresh-older-than", type=str, default=None,
              help="Enrichment: re-enrich entries older than Nd (e.g. '30d').")
@pass_config
def preprocess_run(
    config: Config,
    step: str | None,
    force: bool,
    min_reports: int,
    source: str | None,
    refresh_older_than: str | None,
) -> None:
    """Run preprocessing: cleaning + enrichment (no LLM required)."""
    import logging

    from rich.console import Console

    from protondb_settings.db.connection import get_connection
    from protondb_settings.db.migrations import ensure_schema

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = get_connection(config.db_path)
    ensure_schema(conn)
    console = Console()

    refresh_days = None
    if refresh_older_than:
        refresh_days = int(refresh_older_than.rstrip("d"))

    if step is None or step == "cleaning":
        console.print("\n[bold]Step 1: Data cleaning[/bold]")
        from protondb_settings.preprocessing.cleaning import clean_reports

        n = clean_reports(conn, force=force)
        console.print(f"  Processed {n} reports")

    if step is None or step == "normalize":
        console.print("\n[bold]Step 2: Heuristic normalization (GPU + CPU + driver)[/bold]")
        from protondb_settings.preprocessing.normalize.gpu_heuristic import normalize_gpus_heuristic
        from protondb_settings.preprocessing.normalize.cpu_heuristic import normalize_cpus_heuristic
        from protondb_settings.preprocessing.normalize.gpu_driver_heuristic import normalize_gpu_drivers

        n = normalize_gpus_heuristic(conn, force=force)
        console.print(f"  GPU: {n} strings")
        n = normalize_cpus_heuristic(conn, force=force)
        console.print(f"  CPU: {n} strings")
        n = normalize_gpu_drivers(conn, force=force)
        console.print(f"  GPU driver: {n} strings")

    if step is None or step == "enrichment":
        console.print("\n[bold]Step 3: Enrichment[/bold]")
        from protondb_settings.preprocessing.enrichment.main import run_enrichment

        try:
            n = run_enrichment(
                conn,
                min_reports=min_reports,
                source=source,
                force=force,
                refresh_older_than_days=refresh_days,
            )
            console.print(f"  Processed {n} app_ids")
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — progress saved[/yellow]")

    conn.close()


# ---------------------------------------------------------------------------
# preprocess llm
# ---------------------------------------------------------------------------


@preprocess.group("llm")
@click.option("--base-url", envvar="OPENAI_BASE_URL", default=None,
              help="LLM API base URL.")
@click.option("--model", envvar="MODEL", default=None,
              help="Model name/identifier.")
@click.option("--api-key", envvar="OPENAI_API_KEY", default=None,
              help="API key for the LLM provider.")
@click.option("--concurrency", type=int, default=None,
              help="Max concurrent LLM requests.")
@click.pass_context
def preprocess_llm(
    ctx: click.Context,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    concurrency: int | None,
) -> None:
    """LLM preprocessing tasks: normalization, parsing, extraction."""
    from protondb_settings.preprocessing.llm.client import LLMClient

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key
    if concurrency:
        kwargs["max_concurrency"] = concurrency

    from protondb_settings.preprocessing.interrupt import shutdown_requested

    ctx.ensure_object(dict)
    llm = LLMClient(**kwargs)
    llm._cancel = shutdown_requested
    ctx.obj["llm"] = llm


def _get_llm_conn(config: Config):
    """Helper to get DB connection and ensure schema for LLM commands."""
    import logging

    from protondb_settings.db.connection import get_connection
    from protondb_settings.db.migrations import ensure_schema

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = get_connection(config.db_path)
    ensure_schema(conn)
    return conn


def _run_llm_task(func, conn, llm, *, force: bool = False) -> int | None:
    """Run an LLM pipeline task with graceful interrupt handling.

    Installs signal + stdin watchers for Ctrl+C (including kitty protocol).
    Returns processed count, or None if interrupted.
    """
    from protondb_settings.preprocessing.interrupt import (
        install_handlers,
        restore_handlers,
        shutdown_requested,
    )

    install_handlers()
    try:
        n = func(conn, llm, force=force)
        if shutdown_requested.is_set():
            click.echo("\nInterrupted — progress saved")
            return None
        return n
    except KeyboardInterrupt:
        shutdown_requested.set()
        click.echo("\nInterrupted — progress saved")
        return None
    finally:
        restore_handlers()


@preprocess_llm.command("normalize-gpu")
@click.option("--force", is_flag=True, help="Delete existing and start fresh.")
@click.pass_context
def llm_normalize_gpu(ctx: click.Context, force: bool) -> None:
    """Normalize GPU strings via LLM."""
    config = ctx.find_root().ensure_object(Config)
    conn = _get_llm_conn(config)
    from protondb_settings.preprocessing.normalize.gpu import normalize_gpus

    n = _run_llm_task(normalize_gpus, conn, ctx.obj["llm"], force=force)
    if n is not None:
        click.echo(f"Processed {n} GPU strings")
    conn.close()


@preprocess_llm.command("normalize-cpu")
@click.option("--force", is_flag=True, help="Delete existing and start fresh.")
@click.pass_context
def llm_normalize_cpu(ctx: click.Context, force: bool) -> None:
    """Normalize CPU strings via LLM."""
    config = ctx.find_root().ensure_object(Config)
    conn = _get_llm_conn(config)
    from protondb_settings.preprocessing.normalize.cpu import normalize_cpus

    n = _run_llm_task(normalize_cpus, conn, ctx.obj["llm"], force=force)
    if n is not None:
        click.echo(f"Processed {n} CPU strings")
    conn.close()


@preprocess_llm.command("parse-launch-options")
@click.option("--force", is_flag=True, help="Delete existing and start fresh.")
@click.pass_context
def llm_parse_launch_options(ctx: click.Context, force: bool) -> None:
    """Parse launch options strings via LLM."""
    config = ctx.find_root().ensure_object(Config)
    conn = _get_llm_conn(config)
    from protondb_settings.preprocessing.normalize.launch_options import parse_launch_options

    n = _run_llm_task(parse_launch_options, conn, ctx.obj["llm"], force=force)
    if n is not None:
        click.echo(f"Processed {n} launch option strings")
    conn.close()


@preprocess_llm.command("extract")
@click.option("--force", is_flag=True, help="Delete existing and start fresh.")
@click.pass_context
def llm_extract(ctx: click.Context, force: bool) -> None:
    """Extract actions/observations from report text via LLM."""
    config = ctx.find_root().ensure_object(Config)
    conn = _get_llm_conn(config)
    from protondb_settings.preprocessing.extract.extractor import run_extraction

    n = _run_llm_task(run_extraction, conn, ctx.obj["llm"], force=force)
    if n is not None:
        click.echo(f"Processed {n} reports")
    conn.close()


@preprocess_llm.command("all")
@click.option("--force", is_flag=True, help="Delete existing and start fresh.")
@click.pass_context
def llm_all(ctx: click.Context, force: bool) -> None:
    """Run all LLM tasks sequentially: normalize-gpu, normalize-cpu, parse-launch-options, extract."""
    from rich.console import Console

    from protondb_settings.preprocessing.interrupt import (
        install_handlers,
        restore_handlers,
        shutdown_requested,
    )

    config = ctx.find_root().ensure_object(Config)
    conn = _get_llm_conn(config)
    llm = ctx.obj["llm"]
    console = Console()

    from protondb_settings.preprocessing.extract.extractor import run_extraction
    from protondb_settings.preprocessing.normalize.cpu import normalize_cpus
    from protondb_settings.preprocessing.normalize.gpu import normalize_gpus
    from protondb_settings.preprocessing.normalize.launch_options import parse_launch_options

    steps = [
        ("1/4: GPU normalization", "GPU strings", normalize_gpus),
        ("2/4: CPU normalization", "CPU strings", normalize_cpus),
        ("3/4: Launch options parsing", "launch option strings", parse_launch_options),
        ("4/4: Text extraction", "reports", run_extraction),
    ]

    install_handlers()
    try:
        for title, unit, func in steps:
            if shutdown_requested.is_set():
                break
            console.print(f"\n[bold]{title}[/bold]")
            try:
                n = func(conn, llm, force=force)
                console.print(f"  Processed {n} {unit}")
            except KeyboardInterrupt:
                shutdown_requested.set()

        if shutdown_requested.is_set():
            console.print("\n[yellow]Interrupted — progress saved[/yellow]")
    finally:
        restore_handlers()

    conn.close()


# ---------------------------------------------------------------------------
# ml
# ---------------------------------------------------------------------------


@cli.group()
@pass_config
def ml(config: Config) -> None:
    """ML pipeline: train and evaluate compatibility models."""


@ml.command("train")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Output directory for model artifacts (default: same as db dir).")
@click.option("--test-fraction", type=float, default=0.2,
              help="Fraction of data to use for testing.")
@click.option("--normalized-data", type=click.Choice(["heuristic", "llm"]),
              default=None, envvar="NORMALIZED_DATA_SOURCE",
              help="Source of normalized data: heuristic (default) or llm.")
@pass_config
def ml_train(config: Config, output_dir: Path | None, test_fraction: float, normalized_data: str | None) -> None:
    """Train the ML model: embeddings + LightGBM classifier."""
    import logging

    from protondb_settings.db.connection import get_connection
    from protondb_settings.db.migrations import ensure_schema
    from protondb_settings.ml.train import train_pipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = get_connection(config.db_path)
    ensure_schema(conn)

    if output_dir is None:
        output_dir = config.db_path.parent

    results = train_pipeline(
        conn, output_dir=output_dir, test_fraction=test_fraction,
        normalized_data_source=normalized_data,
    )
    conn.close()

    click.echo(f"\nAccuracy: {results['accuracy']:.4f}")
    click.echo(f"F1 (macro): {results['f1_macro']:.4f}")


@ml.command("train-cascade")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Output directory for model artifacts (default: same as db dir).")
@click.option("--test-fraction", type=float, default=0.2,
              help="Fraction of data to use for testing.")
@click.option("--normalized-data", type=click.Choice(["heuristic", "llm"]),
              default=None, envvar="NORMALIZED_DATA_SOURCE",
              help="Source of normalized data: heuristic (default) or llm.")
@click.option("--embeddings", "embeddings_path", type=click.Path(path_type=Path), default=None,
              help="Path to existing embeddings.npz to reuse (default: {output-dir}/embeddings.npz).")
@click.option("--force-embeddings", is_flag=True, default=False,
              help="Rebuild all embeddings from scratch, ignoring cache.")
@click.option("--reuse-stage1", type=click.Path(path_type=Path), default=None,
              help="Reuse saved Stage 1 model to skip retraining (for Stage 2 experiments).")
@pass_config
def ml_train_cascade(config: Config, output_dir: Path | None, test_fraction: float,
                     normalized_data: str | None, embeddings_path: Path | None,
                     force_embeddings: bool, reuse_stage1: Path | None) -> None:
    """Train two-stage cascade classifier (borked/works → tinkering/oob)."""
    import logging

    from protondb_settings.db.connection import get_connection
    from protondb_settings.db.migrations import ensure_schema
    from protondb_settings.ml.train import train_cascade_pipeline

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    conn = get_connection(config.db_path)
    ensure_schema(conn)

    if output_dir is None:
        output_dir = config.db_path.parent

    results = train_cascade_pipeline(
        conn, output_dir=output_dir, test_fraction=test_fraction,
        normalized_data_source=normalized_data,
        embeddings_path=embeddings_path,
        force_embeddings=force_embeddings,
        reuse_stage1=reuse_stage1,
    )
    conn.close()

    click.echo(f"\nCascade Accuracy: {results['accuracy']:.4f}")
    click.echo(f"Cascade F1 (macro): {results['f1_macro']:.4f}")


@ml.command("evaluate")
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to model.pkl (default: data/model.pkl).")
@pass_config
def ml_evaluate(config: Config, model_path: Path | None) -> None:
    """Evaluate an existing trained model."""
    import logging

    from protondb_settings.db.connection import get_connection
    from protondb_settings.db.migrations import ensure_schema
    from protondb_settings.ml.export import load_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if model_path is None:
        model_path = config.model_path

    if not model_path.exists():
        click.echo(f"Model not found at {model_path}. Run 'ml train' first.")
        return

    model = load_model(model_path)
    click.echo(f"Loaded model from {model_path}")
    click.echo(f"Model type: {type(model).__name__}")
    if hasattr(model, "n_estimators"):
        click.echo(f"Estimators: {model.best_iteration_}")
    if hasattr(model, "feature_name_"):
        click.echo(f"Features: {len(model.feature_name_)}")


@ml.command("eval")
@click.argument("app_id", type=int)
@click.option("--variant", type=str, default="official",
              help="Proton variant (official, ge, experimental, native, notListed, older).")
@click.option("--gpu", type=str, default=None,
              help="Override GPU string (default: auto-detect).")
@click.option("--model-dir", type=click.Path(exists=True, path_type=Path), default=None,
              help="Directory with model_cascade.pkl and embeddings.npz (default: data/).")
@pass_config
def ml_eval(config: Config, app_id: int, variant: str, gpu: str | None, model_dir: Path | None) -> None:
    """Predict compatibility for APP_ID on the current PC.

    Detects your GPU, loads the trained cascade model, and predicts
    whether the game will work out-of-the-box, need tinkering, or be borked.

    Examples:

        protondb-settings ml eval 730        # Counter-Strike 2

        protondb-settings ml eval 1245620 --variant ge   # Elden Ring with GE-Proton
    """
    import logging

    from rich.console import Console
    from rich.table import Table

    from protondb_settings.ml.predict import predict_for_app

    logging.basicConfig(level=logging.WARNING)
    console = Console()

    if model_dir is None:
        model_dir = config.db_path.parent

    model_path = model_dir / "model_cascade.pkl"
    emb_path = model_dir / "embeddings.npz"

    if not model_path.exists():
        console.print(f"[red]Model not found:[/red] {model_path}")
        console.print("Run 'protondb-settings ml train-cascade' first.")
        return
    if not emb_path.exists():
        console.print(f"[red]Embeddings not found:[/red] {emb_path}")
        return

    game_name = None

    result = predict_for_app(
        app_id, model_path, emb_path,
        variant=variant, gpu_override=gpu,
    )

    # Display header
    title = f"App {app_id}"
    if game_name:
        title = f"{game_name} ({app_id})"
    console.print(f"\n[bold]{title}[/bold]")

    # Prediction result with color
    pred = result["prediction"]
    color = {"borked": "red", "tinkering": "yellow", "works_oob": "green"}[pred]
    pred_display = {"borked": "Borked", "tinkering": "Needs Tinkering", "works_oob": "Works Out of the Box"}[pred]
    conf = result["confidence"]
    conf_label = "high" if result["is_confident"] else "low"
    console.print(f"  Prediction: [{color} bold]{pred_display}[/{color} bold]  (confidence: {conf:.0%} {conf_label})")

    # Probabilities table
    table = Table(show_header=False, box=None, padding=(0, 2))
    probs = result["probabilities"]
    for label, p in probs.items():
        bar_len = int(p * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        c = {"borked": "red", "tinkering": "yellow", "works_oob": "green"}[label]
        display = {"borked": "Borked", "tinkering": "Tinkering", "works_oob": "Works OOB"}[label]
        table.add_row(f"  {display:12s}", f"[{c}]{bar}[/{c}]", f"{p:.1%}")
    console.print(table)

    # Hardware info
    console.print(f"\n  GPU: {result['gpu'] or 'unknown'} (family: {result['gpu_family'] or 'unknown'})")
    if result["driver_version"]:
        console.print(f"  Driver: {result['driver_version']}")
    console.print(f"  Variant: {result['variant']}")

    if result.get("override"):
        console.print(f"\n  [dim]Override: {result['override_reason']}[/dim]")

    if not result["has_game_embedding"]:
        console.print("\n  [yellow]Warning: no game embedding — this game has few/no reports, prediction may be unreliable.[/yellow]")
