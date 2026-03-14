"""GPU normalization pipeline — processes unique GPU strings via LLM."""

from __future__ import annotations

import logging
import sqlite3

from protondb_settings.config import (
    GPU_NORM_BATCH_CLOUD,
    GPU_NORM_BATCH_LOCAL,
)
from protondb_settings.preprocessing.interrupt import shutdown_requested
from protondb_settings.preprocessing.llm.client import LLMClient
from protondb_settings.preprocessing.llm.prompts.gpu_normalize import (
    SYSTEM_PROMPT,
    format_batch_prompt,
    format_single_prompt,
)
from protondb_settings.preprocessing.llm.schemas import (
    GpuNormBatchResponse,
    GpuNormResult,
)
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import get_pending_strings, upsert_rows

log = logging.getLogger(__name__)


def get_pending_gpu_strings(conn: sqlite3.Connection) -> list[str]:
    """Get distinct GPU strings not yet in gpu_normalization."""
    return get_pending_strings(
        conn, source_table="reports", source_column="gpu",
        norm_table="gpu_normalization",
    )


def get_pending_count(conn: sqlite3.Connection) -> int:
    """Return number of pending GPU strings."""
    return len(get_pending_gpu_strings(conn))


def _parse_single_result(raw_string: str, result: dict | None) -> dict | None:
    """Parse a single LLM result into a gpu_normalization row."""
    if result is None:
        return None
    return {
        "raw_string": raw_string,
        "vendor": str(result.get("vendor", "unknown")).lower(),
        "family": str(result.get("family", "unknown")).lower(),
        "model": str(result.get("model", "unknown")).lower(),
        "normalized_name": str(result.get("normalized_name", "unknown")),
        "is_apu": int(bool(result.get("is_apu", False))),
        "is_igpu": int(bool(result.get("is_igpu", False))),
        "is_virtual": int(bool(result.get("is_virtual", False))),
    }


def _process_single(llm: LLMClient, raw_string: str) -> dict | None:
    """Process a single GPU string via LLM with retry (used as fallback)."""
    prompt = format_single_prompt(raw_string)
    result = llm.complete_json(SYSTEM_PROMPT, prompt, schema=GpuNormResult)
    return _parse_single_result(raw_string, result)


def normalize_gpus(
    conn: sqlite3.Connection,
    llm: LLMClient,
    *,
    force: bool = False,
) -> int:
    """Run GPU normalization on all pending unique GPU strings.

    Durable: commits after each batch, retries failed batch items individually.
    Returns the number of strings processed.
    """
    if force:
        log.info("Force GPU normalization: deleting gpu_normalization")
        conn.execute("DELETE FROM gpu_normalization")
        conn.commit()

    pending = get_pending_gpu_strings(conn)
    if not pending:
        log.info("GPU normalization: nothing to do")
        return 0

    batch_size = GPU_NORM_BATCH_LOCAL if llm.is_local else GPU_NORM_BATCH_CLOUD
    log.info(
        "GPU normalization: %d strings to process (batch_size=%d)",
        len(pending), batch_size,
    )

    processed = 0
    with PipelineStep(conn, "normalize_gpu", len(pending)) as step:
        for batch in chunked(pending, batch_size):
            if shutdown_requested.is_set():
                log.info("GPU normalization: interrupted, stopping after %d processed", processed)
                break
            rows_to_insert: list[dict] = []
            failed_strings: list[str] = []

            if len(batch) == 1:
                row = _process_single(llm, batch[0])
                if row:
                    rows_to_insert.append(row)
                else:
                    log.warning("GPU normalization: LLM failed for %r", batch[0])
            else:
                prompt = format_batch_prompt(batch)
                result = llm.complete_json(
                    SYSTEM_PROMPT, prompt, max_tokens=8192,
                    schema=GpuNormBatchResponse, schema_name="gpu_norm_batch",
                )
                if result and isinstance(result, dict):
                    results_list = result.get("results", [])
                    for i, gpu_str in enumerate(batch):
                        if i < len(results_list):
                            row = _parse_single_result(gpu_str, results_list[i])
                            if row:
                                rows_to_insert.append(row)
                            else:
                                log.warning("GPU normalization: failed to parse result for %r", gpu_str)
                                failed_strings.append(gpu_str)
                        else:
                            failed_strings.append(gpu_str)
                else:
                    log.warning("GPU normalization: LLM failed for batch of %d, retrying individually", len(batch))
                    failed_strings = list(batch)

            # Retry failed items individually
            for gpu_str in failed_strings:
                try:
                    row = _process_single(llm, gpu_str)
                    if row:
                        rows_to_insert.append(row)
                    else:
                        log.warning("GPU normalization: retry also failed for %r", gpu_str)
                except Exception:
                    log.warning("GPU normalization: retry error for %r", gpu_str, exc_info=True)

            # Commit after each batch for durability
            if rows_to_insert:
                try:
                    upsert_rows(conn, "gpu_normalization", rows_to_insert, "raw_string")
                    conn.commit()
                except Exception:
                    log.warning("GPU normalization: batch upsert failed", exc_info=True)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    return processed
