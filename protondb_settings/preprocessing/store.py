"""Generic SQLite helpers for preprocessing pipelines."""

from __future__ import annotations

import sqlite3
from typing import Any, Sequence


def upsert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: Sequence[dict[str, Any]],
    conflict_column: str,
) -> int:
    """Insert or update rows into *table*.

    Each dict in *rows* must have the same keys.  On conflict on
    *conflict_column* every other column is updated.

    Automatically chunks large batches to stay within SQLite's
    SQLITE_MAX_VARIABLE_NUMBER limit (default 32766).

    Returns the number of rows affected.
    """
    if not rows:
        return 0

    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    col_names = ", ".join(columns)
    update_cols = [c for c in columns if c != conflict_column]
    update_clause = ", ".join(f"{c} = excluded.{c}" for c in update_cols)

    sql = (
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict_column}) DO UPDATE SET {update_clause}"
    )

    # SQLite default SQLITE_MAX_VARIABLE_NUMBER = 32766
    # Chunk to stay safe: max_rows_per_chunk = 32766 // num_columns
    max_chunk = max(1, 32000 // len(columns))

    total = 0
    for i in range(0, len(rows), max_chunk):
        chunk = rows[i : i + max_chunk]
        values_list = [tuple(row[c] for c in columns) for row in chunk]
        conn.executemany(sql, values_list)
        total += len(values_list)

    return total


def get_pending_strings(
    conn: sqlite3.Connection,
    *,
    source_table: str,
    source_column: str,
    norm_table: str,
    norm_column: str = "raw_string",
) -> list[str]:
    """Get distinct non-empty strings from *source_table* not yet in *norm_table*.

    Common pattern for GPU/CPU/launch_options normalization.
    """
    rows = conn.execute(
        f"SELECT DISTINCT {source_column} FROM {source_table} "
        f"WHERE {source_column} IS NOT NULL AND {source_column} != '' "
        f"AND {source_column} NOT IN (SELECT {norm_column} FROM {norm_table})"
    ).fetchall()
    return [row[source_column] for row in rows]
