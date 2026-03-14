"""Launch options parsing pipeline — all unique strings via LLM."""

from __future__ import annotations

import json
import logging
import sqlite3

from protondb_settings.config import (
    LAUNCH_OPT_BATCH_CLOUD,
    LAUNCH_OPT_BATCH_LOCAL,
)
from protondb_settings.preprocessing.interrupt import shutdown_requested
from protondb_settings.preprocessing.llm.client import LLMClient
from protondb_settings.preprocessing.llm.prompts.launch_parse import (
    SYSTEM_PROMPT,
    format_batch_prompt,
    format_single_prompt,
)
from protondb_settings.preprocessing.llm.schemas import (
    LaunchParseBatchResponse,
    LaunchParseResult,
)
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import get_pending_strings, upsert_rows

log = logging.getLogger(__name__)


def get_pending_launch_strings(conn: sqlite3.Connection) -> list[str]:
    """Get distinct launch options strings not yet in launch_options_parsed."""
    return get_pending_strings(
        conn, source_table="reports", source_column="launch_options",
        norm_table="launch_options_parsed",
    )


def get_pending_count(conn: sqlite3.Connection) -> int:
    """Return number of pending launch options strings."""
    return len(get_pending_launch_strings(conn))


def _parse_single_result(raw_string: str, result: dict | None) -> dict | None:
    """Parse a single LLM result into a launch_options_parsed row."""
    if result is None:
        return None

    env_vars = result.get("env_vars", [])
    wrappers = result.get("wrappers", [])
    game_args = result.get("game_args", [])
    unparsed = result.get("unparsed", "")

    return {
        "raw_string": raw_string,
        "env_vars_json": json.dumps(env_vars) if env_vars else None,
        "wrappers_json": json.dumps(wrappers) if wrappers else None,
        "game_args_json": json.dumps(game_args) if game_args else None,
        "unparsed": unparsed if unparsed else None,
    }


def _process_single(llm: LLMClient, raw_string: str) -> dict | None:
    """Process a single launch options string via LLM with retry (used as fallback)."""
    prompt = format_single_prompt(raw_string)
    result = llm.complete_json(SYSTEM_PROMPT, prompt, schema=LaunchParseResult)
    return _parse_single_result(raw_string, result)


def parse_launch_options(
    conn: sqlite3.Connection,
    llm: LLMClient,
    *,
    force: bool = False,
) -> int:
    """Parse all pending unique launch options strings via LLM.

    Durable: commits after each batch, retries failed batch items individually.
    Returns the number of strings processed.
    """
    if force:
        log.info("Force launch options: deleting launch_options_parsed")
        conn.execute("DELETE FROM launch_options_parsed")
        conn.commit()

    pending = get_pending_launch_strings(conn)
    if not pending:
        log.info("Launch options parsing: nothing to do")
        return 0

    batch_size = LAUNCH_OPT_BATCH_LOCAL if llm.is_local else LAUNCH_OPT_BATCH_CLOUD
    log.info(
        "Launch options parsing: %d strings to process (batch_size=%d)",
        len(pending), batch_size,
    )

    processed = 0
    with PipelineStep(conn, "parse_launch_options", len(pending)) as step:
        for batch in chunked(pending, batch_size):
            if shutdown_requested.is_set():
                log.info("Launch options: interrupted, stopping after %d processed", processed)
                break
            rows_to_insert: list[dict] = []
            failed_strings: list[str] = []

            if len(batch) == 1:
                row = _process_single(llm, batch[0])
                if row:
                    rows_to_insert.append(row)
                else:
                    log.warning("Launch options: LLM failed for %r", batch[0])
            else:
                prompt = format_batch_prompt(batch)
                result = llm.complete_json(
                    SYSTEM_PROMPT, prompt, max_tokens=8192,
                    schema=LaunchParseBatchResponse, schema_name="launch_parse_batch",
                )
                if result and isinstance(result, dict):
                    results_list = result.get("results", [])
                    for i, lo_str in enumerate(batch):
                        if i < len(results_list):
                            row = _parse_single_result(lo_str, results_list[i])
                            if row:
                                rows_to_insert.append(row)
                            else:
                                log.warning("Launch options: failed to parse result for %r", lo_str)
                                failed_strings.append(lo_str)
                        else:
                            failed_strings.append(lo_str)
                else:
                    log.warning("Launch options: LLM failed for batch of %d, retrying individually", len(batch))
                    failed_strings = list(batch)

            # Retry failed items individually
            for lo_str in failed_strings:
                try:
                    row = _process_single(llm, lo_str)
                    if row:
                        rows_to_insert.append(row)
                    else:
                        log.warning("Launch options: retry also failed for %r", lo_str)
                except Exception:
                    log.warning("Launch options: retry error for %r", lo_str, exc_info=True)

            # Commit after each batch for durability
            if rows_to_insert:
                try:
                    upsert_rows(conn, "launch_options_parsed", rows_to_insert, "raw_string")
                    conn.commit()
                except Exception:
                    log.warning("Launch options: batch upsert failed", exc_info=True)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
