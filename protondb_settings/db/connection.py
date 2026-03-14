"""SQLite connection management with WAL mode and recommended PRAGMAs."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open (or create) a SQLite database with WAL mode and performance PRAGMAs.

    Returns a ``sqlite3.Connection`` configured for concurrent reads and
    batched writes.  The caller is responsible for closing it.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")  # 30s — needed for parallel enrichment + LLM
    conn.execute("PRAGMA cache_size=-64000")  # 64 MB
    return conn
