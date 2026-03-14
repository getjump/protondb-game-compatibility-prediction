"""Data cleaning step — processes raw report fields into normalized columns.

Cleans:
- ram -> ram_mb: extract integer from raw string, NULL if garbage
- proton_version: trim, "Default"|"" -> NULL
- kernel: extract version regex, NULL if no match

Checkpoint strategy: We use ram_mb as the checkpoint indicator.
- NULL means "not yet cleaned"
- A positive value means valid RAM in MB
- 0 means "cleaned but ram was garbage/unparseable"
Consumers should use ``ram_mb > 0`` to filter for valid values.
"""

from __future__ import annotations

import logging
import re
import sqlite3

from protondb_settings.config import CLEANING_BATCH_SIZE
from protondb_settings.preprocessing.pipeline import PipelineStep, chunked

log = logging.getLogger(__name__)

# ── regex patterns ──────────────────────────────────────────────────────

_RAM_RE = re.compile(r"(\d+)")
_KERNEL_RE = re.compile(r"(\d+\.\d+(?:\.\d+)*)")
_PROTON_VERSION_RE = re.compile(r"(\d[\d.\-]+\d)")

# Sentinel value for "cleaned but ram was garbage". Consumers filter ram_mb > 0.
_RAM_GARBAGE_SENTINEL = 0

# 2 TB in MB — no consumer hardware exceeds this
_MAX_RAM_MB = 2 * 1024 * 1024


def _parse_ram_mb(raw: str | None) -> int | None:
    """Extract megabytes as int from ram string like '16 GB' or '16384'.

    Returns None for garbage values.
    """
    if not raw:
        return None
    m = _RAM_RE.search(raw)
    if m is None:
        return None
    value = int(m.group(1))
    if value > _MAX_RAM_MB * 1024:
        return None  # absurd number, treat as garbage
    raw_lower = raw.lower()
    # If value looks like GB or MB based on unit markers
    if "gb" in raw_lower:
        result = value * 1024
    elif "mb" in raw_lower:
        result = value
    # Heuristic: if number <= 256, likely GB; if > 256 likely MB already
    elif value <= 256:
        result = value * 1024
    else:
        result = value
    return result if result <= _MAX_RAM_MB else None


def _clean_proton_version(raw: str | None) -> str | None:
    """Trim and normalize proton version.  'Default' / '' -> None."""
    if not raw:
        return None
    cleaned = raw.strip().replace("\n", " ").strip()
    if cleaned.lower() in ("default", ""):
        return None
    return cleaned


def _clean_kernel(raw: str | None) -> str | None:
    """Extract kernel version (e.g. 6.1.12) from raw string, None if no match."""
    if not raw:
        return None
    m = _KERNEL_RE.search(raw)
    return m.group(1) if m else None


# ── main entry point ────────────────────────────────────────────────────


def get_pending_count(conn: sqlite3.Connection) -> int:
    """Return number of reports needing cleaning (ram_mb IS NULL = not yet cleaned)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE ram_mb IS NULL"
    ).fetchone()
    return row[0] if row else 0


def clean_reports(conn: sqlite3.Connection, *, force: bool = False) -> int:
    """Run data cleaning on unprocessed reports.

    Individual row errors are logged and skipped — they never crash the pipeline.
    Returns the number of reports processed.
    """
    if force:
        log.info("Force cleaning: resetting ram_mb to NULL")
        conn.execute("UPDATE reports SET ram_mb = NULL")
        conn.commit()

    # Implicit checkpoint: ram_mb IS NULL means not yet cleaned
    rows = conn.execute(
        "SELECT id, ram, proton_version, kernel FROM reports "
        "WHERE ram_mb IS NULL"
    ).fetchall()

    total = len(rows)
    if total == 0:
        log.info("Data cleaning: nothing to do")
        return 0

    log.info("Data cleaning: %d reports to process", total)
    processed = 0
    errors = 0

    with PipelineStep(conn, "cleaning", total) as step:
        for batch in chunked(rows, CLEANING_BATCH_SIZE):
            for row in batch:
                try:
                    report_id = row["id"]
                    ram_mb = _parse_ram_mb(row["ram"])
                    proton = _clean_proton_version(row["proton_version"])
                    kernel = _clean_kernel(row["kernel"])

                    # Use sentinel 0 for garbage ram so checkpoint works
                    effective_ram_mb = ram_mb if ram_mb is not None else _RAM_GARBAGE_SENTINEL

                    conn.execute(
                        "UPDATE reports SET ram_mb = ?, proton_version = ?, kernel = ? "
                        "WHERE id = ?",
                        (effective_ram_mb, proton, kernel, report_id),
                    )
                except Exception:
                    errors += 1
                    log.debug("Cleaning: failed for report %s, skipping", row.get("id", "?"), exc_info=True)
                    # Mark as processed (sentinel) so we don't retry forever
                    try:
                        conn.execute(
                            "UPDATE reports SET ram_mb = ? WHERE id = ?",
                            (_RAM_GARBAGE_SENTINEL, row["id"]),
                        )
                    except Exception:
                        pass

            conn.commit()
            step.advance(len(batch))
            step.sync_run()
            processed += len(batch)

    if errors:
        log.warning("Cleaning completed with %d errors out of %d reports", errors, processed)

    return processed
