"""LLM-based verdict inference for reports without verdict_oob.

Phase 19.4: For 211K reports with verdict=yes but verdict_oob=NULL,
uses LLM to determine if the game worked out of the box or needed tinkering,
based on the report text and structured customization data.

Results stored in `inferred_verdicts` table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from protondb_settings.config import REPORT_TEXT_FIELDS
from protondb_settings.preprocessing.interrupt import shutdown_requested
from protondb_settings.preprocessing.llm.client import LLMClient
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked
from protondb_settings.preprocessing.store import upsert_rows

log = logging.getLogger(__name__)


class VerdictInference(BaseModel):
    """LLM inference result for verdict_oob."""
    verdict: str = Field(description="works_oob or tinkering")
    confidence: str = Field(description="high, medium, or low")
    reason: str = Field(description="Brief explanation")


VERDICT_SYSTEM_PROMPT = """\
You classify ProtonDB compatibility reports. Given a user's report about running \
a game on Linux with Proton/Wine, determine if the game worked "out of the box" \
or required "tinkering" (manual customization).

Return JSON only: {"verdict": "works_oob" or "tinkering", "confidence": "high/medium/low", "reason": "brief explanation"}

Key distinctions:
- works_oob: Game launched and played with NO manual intervention beyond clicking Play. \
Choosing a Proton version from Steam's dropdown is NOT tinkering.
- tinkering: User had to set environment variables, install dependencies via protontricks, \
modify config files, use custom launch options, apply workarounds, or do anything beyond \
the standard Steam Play experience.

IMPORTANT: Selecting GE-Proton or Proton Experimental from the Steam compatibility menu \
is borderline. If that was the ONLY thing the user did, classify as works_oob. If they \
also did other customizations, classify as tinkering."""


def _build_verdict_prompt(report: dict[str, Any]) -> str:
    """Build user prompt for verdict inference."""
    # Collect customization signals
    cust_fields = [
        ("cust_winetricks", "winetricks"),
        ("cust_protontricks", "protontricks"),
        ("cust_config_change", "config change"),
        ("cust_custom_prefix", "custom prefix"),
        ("cust_lutris", "lutris"),
        ("cust_media_foundation", "media foundation"),
        ("cust_protonfixes", "protonfixes"),
        ("cust_native2proton", "native2proton"),
    ]
    active_custs = [label for key, label in cust_fields if report.get(key)]

    flag_fields = [
        ("flag_use_wine_d3d11", "PROTON_USE_WINED3D"),
        ("flag_disable_esync", "PROTON_NO_ESYNC"),
        ("flag_enable_nvapi", "PROTON_ENABLE_NVAPI"),
        ("flag_disable_fsync", "PROTON_NO_FSYNC"),
        ("flag_large_address_aware", "LARGE_ADDRESS_AWARE"),
        ("flag_disable_d3d11", "PROTON_NO_D3D11"),
        ("flag_hide_nvidia", "PROTON_HIDE_NVIDIA_GPU"),
    ]
    active_flags = [label for key, label in flag_fields if report.get(key)]

    # Text
    texts = []
    for field in REPORT_TEXT_FIELDS:
        val = report.get(field)
        if val and isinstance(val, str) and len(val.strip()) > 3:
            texts.append(val.strip())
    combined_text = " | ".join(texts) if texts else "(no text)"

    proton = report.get("custom_proton_version") or report.get("proton_version") or "unknown"
    variant = report.get("variant") or "unknown"
    launch_opts = report.get("launch_options") or ""

    parts = [f"Proton: {proton} ({variant})"]
    if active_custs:
        parts.append(f"Customizations: {', '.join(active_custs)}")
    if active_flags:
        parts.append(f"Flags set: {', '.join(active_flags)}")
    if launch_opts.strip():
        parts.append(f"Launch options: {launch_opts.strip()}")
    parts.append(f"User notes: {combined_text[:500]}")

    return "\n".join(parts)


def get_pending_reports(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    """Get reports needing verdict inference: verdict=yes, verdict_oob=NULL, not yet inferred."""
    limit_clause = f"LIMIT {limit}" if limit else ""
    rows = conn.execute(f"""
        SELECT r.id, r.app_id, r.verdict, r.verdict_oob, r.variant,
               r.proton_version, r.custom_proton_version, r.launch_options,
               r.concluding_notes, r.notes_verdict, r.notes_extra,
               r.notes_customizations, r.notes_launch_flags,
               r.cust_winetricks, r.cust_protontricks, r.cust_config_change,
               r.cust_custom_prefix, r.cust_custom_proton, r.cust_lutris,
               r.cust_media_foundation, r.cust_protonfixes, r.cust_native2proton,
               r.flag_use_wine_d3d11, r.flag_disable_esync, r.flag_enable_nvapi,
               r.flag_disable_fsync, r.flag_large_address_aware,
               r.flag_disable_d3d11, r.flag_hide_nvidia
        FROM reports r
        WHERE r.verdict = 'yes' AND r.verdict_oob IS NULL
        AND r.id NOT IN (SELECT report_id FROM inferred_verdicts)
        AND r.concluding_notes IS NOT NULL AND LENGTH(TRIM(r.concluding_notes)) > 10
        ORDER BY r.timestamp ASC
        {limit_clause}
    """).fetchall()
    return [dict(r) for r in rows]


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Ensure inferred_verdicts table exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inferred_verdicts (
            report_id TEXT PRIMARY KEY REFERENCES reports(id),
            verdict TEXT NOT NULL,
            confidence TEXT,
            reason TEXT,
            inferred_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_inferred_verdict
        ON inferred_verdicts(verdict)
    """)


def get_pending_count(conn: sqlite3.Connection) -> int:
    """Count reports needing verdict inference."""
    _ensure_table(conn)
    row = conn.execute("""
        SELECT COUNT(*) FROM reports r
        WHERE r.verdict = 'yes' AND r.verdict_oob IS NULL
        AND r.id NOT IN (SELECT report_id FROM inferred_verdicts)
    """).fetchone()
    return row[0] if row else 0


def _run_rule_based_verdicts(conn: sqlite3.Connection) -> int:
    """Assign verdicts to reports WITHOUT text using rule-based logic.

    - No text + has customization flags → tinkering
    - No text + no customizations → works_oob
    """
    _ensure_table(conn)

    # No text + has custs/flags → tinkering
    tinkering_rows = conn.execute("""
        SELECT r.id FROM reports r
        WHERE r.verdict = 'yes' AND r.verdict_oob IS NULL
        AND r.id NOT IN (SELECT report_id FROM inferred_verdicts)
        AND (r.concluding_notes IS NULL OR LENGTH(TRIM(r.concluding_notes)) <= 10)
        AND (r.cust_protontricks=1 OR r.cust_winetricks=1 OR r.cust_config_change=1
             OR r.cust_custom_prefix=1 OR r.cust_lutris=1 OR r.cust_media_foundation=1
             OR r.cust_protonfixes=1
             OR r.flag_disable_esync=1 OR r.flag_enable_nvapi=1 OR r.flag_disable_fsync=1
             OR r.flag_use_wine_d3d11=1 OR r.flag_large_address_aware=1
             OR r.flag_disable_d3d11=1 OR r.flag_hide_nvidia=1)
    """).fetchall()

    for r in tinkering_rows:
        conn.execute(
            "INSERT OR IGNORE INTO inferred_verdicts (report_id, verdict, confidence, reason) "
            "VALUES (?, 'tinkering', 'high', 'rule: has customization flags, no text')",
            (r["id"],))

    # No text + no custs → works_oob
    oob_rows = conn.execute("""
        SELECT r.id FROM reports r
        WHERE r.verdict = 'yes' AND r.verdict_oob IS NULL
        AND r.id NOT IN (SELECT report_id FROM inferred_verdicts)
        AND (r.concluding_notes IS NULL OR LENGTH(TRIM(r.concluding_notes)) <= 10)
    """).fetchall()

    for r in oob_rows:
        conn.execute(
            "INSERT OR IGNORE INTO inferred_verdicts (report_id, verdict, confidence, reason) "
            "VALUES (?, 'works_oob', 'high', 'rule: no customizations, no text')",
            (r["id"],))

    conn.commit()
    n = len(tinkering_rows) + len(oob_rows)
    log.info("Rule-based verdicts: %d tinkering, %d works_oob", len(tinkering_rows), len(oob_rows))
    return n


def run_verdict_inference(
    conn: sqlite3.Connection,
    llm: LLMClient,
    *,
    force: bool = False,
    batch_size: int = 50,
) -> int:
    """Run verdict inference: rule-based for no-text reports, LLM for reports with text.

    Results go into `inferred_verdicts` table (not modifying original reports).
    """
    _ensure_table(conn)
    conn.commit()

    if force:
        log.info("Force inference: deleting inferred_verdicts")
        conn.execute("DELETE FROM inferred_verdicts")
        conn.commit()

    # Step 1: Rule-based for reports without text (instant, free)
    n_rules = _run_rule_based_verdicts(conn)

    total_pending = get_pending_count(conn)
    if total_pending == 0:
        log.info("Verdict inference: nothing to do (rule-based handled %d)", n_rules)
        return n_rules

    log.info("Verdict inference: %d reports to process via LLM (batch_size=%d)", total_pending, batch_size)

    processed = 0
    consecutive_failures = 0
    max_consecutive_failures = 3  # stop after 3 fully-failed batches (likely budget exhausted)

    with PipelineStep(conn, "verdict_inference", total_pending) as step:
        while not shutdown_requested.is_set():
            # Fetch next batch (not all at once — saves memory, respects resume)
            batch = get_pending_reports(conn, limit=batch_size)
            if not batch:
                break

            # Prepare prompts
            tasks = []
            for report in batch:
                prompt = _build_verdict_prompt(report)
                tasks.append((VERDICT_SYSTEM_PROMPT, prompt))

            # LLM call
            if len(tasks) == 1:
                raw_results = [llm.complete_json(
                    tasks[0][0], tasks[0][1],
                    schema=VerdictInference, schema_name="verdict_inference",
                    max_tokens=4096,
                )]
            else:
                raw_results = llm.batch_complete_json(
                    tasks,
                    schema=VerdictInference, schema_name="verdict_inference",
                    max_tokens=4096,
                )

            # Check if entire batch failed (likely API budget/auth issue)
            all_failed = all(r is None for r in raw_results)
            if all_failed:
                consecutive_failures += 1
                log.warning("Verdict inference: entire batch failed (%d/%d consecutive)",
                            consecutive_failures, max_consecutive_failures)
                if consecutive_failures >= max_consecutive_failures:
                    log.error("Verdict inference: stopping after %d consecutive failed batches "
                              "(likely API budget exhausted or auth error)", max_consecutive_failures)
                    break
                continue  # retry with next batch (same reports since nothing was saved)

            consecutive_failures = 0

            rows_to_insert = []
            for report, raw in zip(batch, raw_results):
                if raw and isinstance(raw, dict):
                    verdict = raw.get("verdict", "tinkering")
                    if verdict not in ("works_oob", "tinkering"):
                        verdict = "tinkering"
                    rows_to_insert.append({
                        "report_id": report["id"],
                        "verdict": verdict,
                        "confidence": raw.get("confidence", "low"),
                        "reason": raw.get("reason", ""),
                    })
                else:
                    # Individual LLM failure — skip, will retry on next run
                    log.debug("Verdict inference: skipping report %s (LLM returned None)",
                              report["id"][:16])

            if rows_to_insert:
                try:
                    upsert_rows(conn, "inferred_verdicts", rows_to_insert, "report_id")
                    conn.commit()
                except Exception:
                    log.warning("Verdict inference: batch upsert failed", exc_info=True)
                    try:
                        conn.rollback()
                    except Exception:
                        pass

            step.advance(len(rows_to_insert))
            step.sync_run()
            processed += len(rows_to_insert)

            if processed % 100 == 0 and processed > 0:
                log.info("Verdict inference: %d processed so far", processed)

    return processed
