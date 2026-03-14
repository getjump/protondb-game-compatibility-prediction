"""GET /health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from protondb_settings.db.connection import get_connection

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    """Return service health including report statistics."""
    config = request.app.state.config
    conn = get_connection(config.db_path)
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM reports").fetchone()
        reports_count = row["cnt"] if row else 0

        last_import = None
        dump_tag = None
        for key in ("dump_imported_at", "dump_release_tag"):
            meta_row = conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
            if meta_row:
                if key == "dump_imported_at":
                    last_import = meta_row["value"]
                else:
                    dump_tag = meta_row["value"]
    except Exception:
        reports_count = 0
        last_import = None
        dump_tag = None
    finally:
        conn.close()

    return {
        "status": "ok",
        "version": "0.1.0",
        "reports_count": reports_count,
        "last_import": last_import,
        "dump_tag": dump_tag,
    }
