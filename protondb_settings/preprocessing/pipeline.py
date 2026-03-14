"""Pipeline step context manager with rich progress bar and pipeline_runs tracking."""

from __future__ import annotations

import logging
import sqlite3
import traceback
from types import TracebackType

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

log = logging.getLogger(__name__)

# Shared console so Progress and RichHandler coordinate output
_console = Console(stderr=True)


class PipelineStep:
    """Context manager for a preprocessing pipeline step.

    Features:
    - Rich progress bar showing step name, bar, N/M, elapsed, ETA
    - Records run in ``pipeline_runs`` table (start, advance, complete/fail)
    - Detects interrupted runs and logs a resume message
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        step_name: str,
        total: int,
        *,
        dump_tag: str | None = None,
    ) -> None:
        self.conn = conn
        self.step_name = step_name
        self.total = total
        self.dump_tag = dump_tag
        self.run_id: int | None = None
        self._processed = 0
        self._progress: Progress | None = None
        self._task_id = None
        self._log_handler: logging.Handler | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> PipelineStep:
        # Check for an interrupted (still 'running') run for this step
        row = self.conn.execute(
            "SELECT id, processed FROM pipeline_runs "
            "WHERE step = ? AND status = 'running' "
            "ORDER BY started_at DESC LIMIT 1",
            (self.step_name,),
        ).fetchone()

        if row is not None:
            self.run_id = row["id"]
            self._processed = row["processed"]
            log.info(
                "Resuming %s from item %d / %d (interrupted run #%d)",
                self.step_name,
                self._processed,
                self.total,
                self.run_id,
            )
            # Update total in case it changed
            self.conn.execute(
                "UPDATE pipeline_runs SET total_items = ? WHERE id = ?",
                (self.total, self.run_id),
            )
            self.conn.commit()
        else:
            cur = self.conn.execute(
                "INSERT INTO pipeline_runs (step, total_items, dump_tag) VALUES (?, ?, ?)",
                (self.step_name, self.total, self.dump_tag),
            )
            self.run_id = cur.lastrowid
            self.conn.commit()

        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA"),
            TimeRemainingColumn(),
            speed_estimate_period=120,  # smooth ETA over 2 minutes
            refresh_per_second=2,
            console=_console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.step_name, total=self.total, completed=self._processed
        )

        # Route all logging through Rich so logs scroll above the progress bar
        self._log_handler = RichHandler(
            console=self._progress.console,
            show_path=False,
            show_time=False,
            markup=True,
        )
        self._log_handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        # Stash existing handlers and replace with RichHandler
        self._prev_handlers = root.handlers[:]
        root.handlers = [self._log_handler]

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._progress is not None:
            self._progress.stop()

        # Restore original logging handlers
        if self._log_handler is not None:
            root = logging.getLogger()
            root.handlers = self._prev_handlers
            self._log_handler = None

        if exc_type is not None:
            if issubclass(exc_type, KeyboardInterrupt):
                # Graceful interruption — keep status='running' so next run resumes
                self.conn.execute(
                    "UPDATE pipeline_runs SET processed = ? WHERE id = ?",
                    (self._processed, self.run_id),
                )
                self.conn.commit()
                log.info(
                    "Step %s interrupted at %d/%d — will resume on next run",
                    self.step_name, self._processed, self.total,
                )
                return True  # suppress KeyboardInterrupt

            error_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.conn.execute(
                "UPDATE pipeline_runs SET status = 'failed', finished_at = datetime('now'), "
                "error = ? WHERE id = ?",
                (error_text, self.run_id),
            )
            self.conn.commit()
            log.error("Step %s failed: %s", self.step_name, exc_val)
            return False  # propagate

        # Check if loop exited due to shutdown_requested (graceful break, no exception)
        from protondb_settings.preprocessing.interrupt import shutdown_requested

        if shutdown_requested.is_set() and self._processed < self.total:
            self.conn.execute(
                "UPDATE pipeline_runs SET processed = ? WHERE id = ?",
                (self._processed, self.run_id),
            )
            self.conn.commit()
            log.info(
                "Step %s interrupted at %d/%d — will resume on next run",
                self.step_name, self._processed, self.total,
            )
            return False

        self.conn.execute(
            "UPDATE pipeline_runs SET status = 'completed', finished_at = datetime('now'), "
            "processed = ? WHERE id = ?",
            (self._processed, self.run_id),
        )
        self.conn.commit()
        log.info("Step %s completed (%d items)", self.step_name, self._processed)
        return False

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def advance(self, n: int = 1) -> None:
        """Advance the progress bar and update the pipeline_runs row."""
        self._processed += n
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=n)

    def sync_run(self) -> None:
        """Persist current processed count to pipeline_runs (call after each batch commit)."""
        self.conn.execute(
            "UPDATE pipeline_runs SET processed = ? WHERE id = ?",
            (self._processed, self.run_id),
        )
        self.conn.commit(
        )

    @property
    def processed(self) -> int:
        return self._processed


def chunked(iterable, size: int):
    """Yield successive chunks of *size* from *iterable* (works on lists)."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
