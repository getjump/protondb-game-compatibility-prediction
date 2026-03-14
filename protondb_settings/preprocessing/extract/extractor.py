"""Layer 2: LLM text extraction — orchestrates the three-layer extraction pipeline.

Flow per batch:
1. Build context + regex spotting (spotter.py) -> pre-extracted entities
2. LLM batch extraction with context -> raw actions + observations
3. Post-validation (validator.py) -> sanitized results
4. UPSERT into extracted_data (committed after each inner batch for durability)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from protondb_settings.config import (
    EXTRACT_BATCH_CLOUD,
    EXTRACT_BATCH_LOCAL,
)
from protondb_settings.preprocessing.extract.filter import get_extractable_reports
from protondb_settings.preprocessing.interrupt import shutdown_requested
from protondb_settings.preprocessing.extract.models import ExtractionResult
from protondb_settings.preprocessing.extract.spotter import format_spotted, spot_entities
from protondb_settings.preprocessing.extract.validator import validate_extraction
from protondb_settings.preprocessing.llm.client import LLMClient
from protondb_settings.preprocessing.llm.prompts.text_extract import (
    SYSTEM_PROMPT,
    build_context_from_report,
    format_prompt,
)
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import upsert_rows

log = logging.getLogger(__name__)


def _get_game_metadata(conn: sqlite3.Connection, app_id: int) -> dict[str, Any] | None:
    """Fetch game_metadata + game name for a given app_id."""
    row = conn.execute(
        "SELECT g.name, gm.engine, gm.graphics_apis, gm.anticheat "
        "FROM games g LEFT JOIN game_metadata gm ON g.app_id = gm.app_id "
        "WHERE g.app_id = ?",
        (app_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def _prepare_report(
    report: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> tuple[str, str] | None:
    """Build LLM prompt for a report. Returns (system_prompt, user_prompt) or None if empty."""
    ctx = build_context_from_report(report, metadata)

    # Early exit: no text to extract from
    if not ctx["combined_text"].strip():
        return None

    # Layer 1: Regex spotting on the combined text
    spotted = spot_entities(ctx["combined_text"])
    ctx["pre_extracted_entities"] = format_spotted(spotted)

    user_prompt = format_prompt(**ctx)
    return (SYSTEM_PROMPT, user_prompt)


def _result_to_row(report: dict[str, Any], result: ExtractionResult) -> dict:
    """Convert an ExtractionResult to a row dict for extracted_data."""
    return {
        "report_id": report["id"],
        "app_id": report["app_id"],
        "actions_json": (
            json.dumps([a.model_dump() for a in result.actions])
            if result.actions else None
        ),
        "observations_json": (
            json.dumps([o.model_dump() for o in result.observations])
            if result.observations else None
        ),
        "useful": int(result.useful),
    }


def _empty_row(report: dict[str, Any]) -> dict:
    """Create a row for a report with no extractable text."""
    return {
        "report_id": report["id"],
        "app_id": report["app_id"],
        "actions_json": None,
        "observations_json": None,
        "useful": 0,
    }


def get_pending_count(conn: sqlite3.Connection) -> int:
    """Return number of reports eligible for extraction."""
    from protondb_settings.preprocessing.extract.filter import get_extractable_count

    return get_extractable_count(conn)


def run_extraction(
    conn: sqlite3.Connection,
    llm: LLMClient,
    *,
    force: bool = False,
) -> int:
    """Run text extraction on all pending extractable reports.

    Durable: commits after each inner batch, isolates per-report errors,
    retries failed batch items individually.

    Returns the number of reports processed.
    """
    if force:
        log.info("Force extraction: deleting extracted_data")
        conn.execute("DELETE FROM extracted_data")
        conn.commit()

    reports = get_extractable_reports(conn)
    if not reports:
        log.info("Text extraction: nothing to do")
        return 0

    batch_size = EXTRACT_BATCH_LOCAL if llm.is_local else EXTRACT_BATCH_CLOUD
    log.info(
        "Text extraction: %d reports to process (batch_size=%d)",
        len(reports), batch_size,
    )

    # Cache metadata per app_id
    metadata_cache: dict[int, dict[str, Any] | None] = {}

    processed = 0
    with PipelineStep(conn, "extract", len(reports)) as step:
        for batch in chunked(reports, batch_size):
            if shutdown_requested.is_set():
                log.info("Text extraction: interrupted, stopping after %d processed", processed)
                break
            rows_to_insert: list[dict] = []

            # Prepare all prompts in the batch
            tasks: list[tuple[str, str]] = []
            task_reports: list[dict[str, Any]] = []

            for report in batch:
                app_id = report["app_id"]
                if app_id not in metadata_cache:
                    metadata_cache[app_id] = _get_game_metadata(conn, app_id)

                try:
                    prepared = _prepare_report(report, metadata_cache[app_id])
                except Exception:
                    log.warning("Extract: failed to prepare report %s", report.get("id"), exc_info=True)
                    rows_to_insert.append(_empty_row(report))
                    continue

                if prepared is None:
                    rows_to_insert.append(_empty_row(report))
                else:
                    tasks.append(prepared)
                    task_reports.append(report)

            # Batch LLM call for non-empty reports
            if tasks:
                if len(tasks) == 1:
                    raw_results = [llm.complete_json(
                        tasks[0][0], tasks[0][1],
                        schema=ExtractionResult, schema_name="text_extraction",
                    )]
                else:
                    raw_results = llm.batch_complete_json(
                        tasks,
                        schema=ExtractionResult, schema_name="text_extraction",
                    )

                # Collect failed indices for individual retry
                retry_indices: list[int] = []

                for i, (report, raw_result) in enumerate(zip(task_reports, raw_results)):
                    try:
                        if raw_result is None:
                            retry_indices.append(i)
                            continue
                        result = validate_extraction(raw_result)
                        rows_to_insert.append(_result_to_row(report, result))
                    except Exception:
                        log.warning(
                            "Extract: validation failed for report %s",
                            report.get("id"), exc_info=True,
                        )
                        retry_indices.append(i)

                # Retry failed items individually
                for i in retry_indices:
                    report = task_reports[i]
                    sys_prompt, user_prompt = tasks[i]
                    try:
                        raw_result = llm.complete_json(
                            sys_prompt, user_prompt,
                            schema=ExtractionResult, schema_name="text_extraction",
                        )
                        result = validate_extraction(raw_result)
                        rows_to_insert.append(_result_to_row(report, result))
                    except Exception:
                        log.warning(
                            "Extract: retry also failed for report %s, marking not useful",
                            report.get("id"), exc_info=True,
                        )
                        rows_to_insert.append(_empty_row(report))

            # Commit after each inner batch for durability
            if rows_to_insert:
                try:
                    upsert_rows(conn, "extracted_data", rows_to_insert, "report_id")
                    conn.commit()
                except Exception:
                    log.warning("Extract: batch upsert failed", exc_info=True)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
