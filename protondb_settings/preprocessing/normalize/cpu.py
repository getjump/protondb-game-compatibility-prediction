"""CPU normalization pipeline — processes unique CPU strings via LLM."""

from __future__ import annotations

import logging
import sqlite3

from protondb_settings.config import (
    CPU_NORM_BATCH_CLOUD,
    CPU_NORM_BATCH_LOCAL,
)
from protondb_settings.preprocessing.interrupt import shutdown_requested
from protondb_settings.preprocessing.llm.client import LLMClient
from protondb_settings.preprocessing.llm.prompts.cpu_normalize import (
    SYSTEM_PROMPT,
    format_batch_prompt,
    format_single_prompt,
)
from protondb_settings.preprocessing.llm.schemas import (
    CpuNormBatchResponse,
    CpuNormResult,
)
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import get_pending_strings, upsert_rows

log = logging.getLogger(__name__)


def get_pending_cpu_strings(conn: sqlite3.Connection) -> list[str]:
    """Get distinct CPU strings not yet in cpu_normalization."""
    return get_pending_strings(
        conn, source_table="reports", source_column="cpu",
        norm_table="cpu_normalization",
    )


def get_pending_count(conn: sqlite3.Connection) -> int:
    """Return number of pending CPU strings."""
    return len(get_pending_cpu_strings(conn))


def _parse_single_result(raw_string: str, result: dict | None) -> dict | None:
    """Parse a single LLM result into a cpu_normalization row."""
    if result is None:
        return None
    gen = result.get("generation")
    if gen is not None:
        try:
            gen = int(float(gen))
        except (ValueError, TypeError):
            gen = None

    return {
        "raw_string": raw_string,
        "vendor": str(result.get("vendor", "unknown")).lower(),
        "family": str(result.get("family", "unknown")).lower(),
        "model": str(result.get("model", "unknown")).lower(),
        "normalized_name": str(result.get("normalized_name", "unknown")),
        "generation": gen,
        "is_apu": int(bool(result.get("is_apu", False))),
    }


def _process_single(llm: LLMClient, raw_string: str) -> dict | None:
    """Process a single CPU string via LLM with retry (used as fallback)."""
    prompt = format_single_prompt(raw_string)
    result = llm.complete_json(SYSTEM_PROMPT, prompt, schema=CpuNormResult)
    return _parse_single_result(raw_string, result)


def normalize_cpus(
    conn: sqlite3.Connection,
    llm: LLMClient,
    *,
    force: bool = False,
) -> int:
    """Run CPU normalization on all pending unique CPU strings.

    Durable: commits after each batch, retries failed batch items individually.
    Returns the number of strings processed.
    """
    if force:
        log.info("Force CPU normalization: deleting cpu_normalization")
        conn.execute("DELETE FROM cpu_normalization")
        conn.commit()

    pending = get_pending_cpu_strings(conn)
    if not pending:
        log.info("CPU normalization: nothing to do")
        return 0

    batch_size = CPU_NORM_BATCH_LOCAL if llm.is_local else CPU_NORM_BATCH_CLOUD
    log.info(
        "CPU normalization: %d strings to process (batch_size=%d)",
        len(pending), batch_size,
    )

    processed = 0
    with PipelineStep(conn, "normalize_cpu", len(pending)) as step:
        for batch in chunked(pending, batch_size):
            if shutdown_requested.is_set():
                log.info("CPU normalization: interrupted, stopping after %d processed", processed)
                break
            rows_to_insert: list[dict] = []
            failed_strings: list[str] = []

            if len(batch) == 1:
                row = _process_single(llm, batch[0])
                if row:
                    rows_to_insert.append(row)
                else:
                    log.warning("CPU normalization: LLM failed for %r", batch[0])
            else:
                prompt = format_batch_prompt(batch)
                result = llm.complete_json(
                    SYSTEM_PROMPT, prompt, max_tokens=8192,
                    schema=CpuNormBatchResponse, schema_name="cpu_norm_batch",
                )
                if result and isinstance(result, dict):
                    results_list = result.get("results", [])
                    for i, cpu_str in enumerate(batch):
                        if i < len(results_list):
                            row = _parse_single_result(cpu_str, results_list[i])
                            if row:
                                rows_to_insert.append(row)
                            else:
                                log.warning("CPU normalization: failed to parse result for %r", cpu_str)
                                failed_strings.append(cpu_str)
                        else:
                            failed_strings.append(cpu_str)
                else:
                    log.warning("CPU normalization: LLM failed for batch of %d, retrying individually", len(batch))
                    failed_strings = list(batch)

            # Retry failed items individually
            for cpu_str in failed_strings:
                try:
                    row = _process_single(llm, cpu_str)
                    if row:
                        rows_to_insert.append(row)
                    else:
                        log.warning("CPU normalization: retry also failed for %r", cpu_str)
                except Exception:
                    log.warning("CPU normalization: retry error for %r", cpu_str, exc_info=True)

            # Commit after each batch for durability
            if rows_to_insert:
                try:
                    upsert_rows(conn, "cpu_normalization", rows_to_insert, "raw_string")
                    conn.commit()
                except Exception:
                    log.warning("CPU normalization: batch upsert failed", exc_info=True)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
