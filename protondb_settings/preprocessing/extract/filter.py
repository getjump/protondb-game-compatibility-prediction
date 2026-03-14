"""Report filtering for text extraction — skip reports that won't yield useful data."""

from __future__ import annotations

import sqlite3
from typing import Any

from protondb_settings.config import REPORT_TEXT_FIELDS

MIN_TEXT_LENGTH = 10


def _build_filter_clauses() -> tuple[str, str]:
    """Build shared SQL filter clauses for extractable reports.

    Returns (text_conditions, skip_oob_clause).
    """
    text_conditions = " OR ".join(
        f"(r.{f} IS NOT NULL AND LENGTH(TRIM(r.{f})) >= {MIN_TEXT_LENGTH})"
        for f in REPORT_TEXT_FIELDS
    )

    skip_oob_clause = f"""
        NOT (
            r.verdict_oob = 'yes'
            AND COALESCE(r.audio_faults, '') != 'yes'
            AND COALESCE(r.graphical_faults, '') != 'yes'
            AND COALESCE(r.input_faults, '') != 'yes'
            AND COALESCE(r.performance_faults, '') != 'yes'
            AND COALESCE(r.stability_faults, '') != 'yes'
            AND COALESCE(r.windowing_faults, '') != 'yes'
            AND COALESCE(r.save_game_faults, '') != 'yes'
            AND COALESCE(r.significant_bugs, '') != 'yes'
            AND COALESCE(LENGTH(TRIM(r.concluding_notes)), 0) < {MIN_TEXT_LENGTH}
            AND COALESCE(LENGTH(TRIM(r.notes_extra)), 0) < {MIN_TEXT_LENGTH}
        )
    """

    return text_conditions, skip_oob_clause


def get_extractable_reports(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Get reports that:
    1. Are not already in extracted_data (implicit checkpoint)
    2. Have useful text content (not just "works fine" verdicts with no notes)

    Returns list of report dicts, ordered by app_id report count (most reports first).
    """
    text_conditions, skip_oob_clause = _build_filter_clauses()
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        WITH app_counts AS (
            SELECT app_id, COUNT(*) as cnt FROM reports GROUP BY app_id
        )
        SELECT r.*
        FROM reports r
        JOIN app_counts ac ON r.app_id = ac.app_id
        WHERE r.id NOT IN (SELECT report_id FROM extracted_data)
        AND ({text_conditions})
        AND {skip_oob_clause}
        ORDER BY ac.cnt DESC
        {limit_clause}
    """

    rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_extractable_count(conn: sqlite3.Connection) -> int:
    """Count reports eligible for extraction (same filters as get_extractable_reports)."""
    text_conditions, skip_oob_clause = _build_filter_clauses()

    query = f"""
        SELECT COUNT(*) FROM reports r
        WHERE r.id NOT IN (SELECT report_id FROM extracted_data)
        AND ({text_conditions})
        AND {skip_oob_clause}
    """
    row = conn.execute(query).fetchone()
    return row[0] if row else 0
