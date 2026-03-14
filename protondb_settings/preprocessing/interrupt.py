"""Graceful interruption support for long-running preprocessing pipelines.

Handles:
- SIGINT (normal Ctrl+C in regular terminals)
- SIGTERM (kill <pid> from another terminal)
- Kitty keyboard protocol escape sequences (Cursor/kitty terminals send
  CSI u sequences instead of SIGINT for Ctrl+C)
"""

from __future__ import annotations

import logging
import os
import select
import signal
import sys
import termios
import threading
import tty

log = logging.getLogger(__name__)

# Global event — set from any source to request graceful shutdown
shutdown_requested = threading.Event()

_original_sigint = None
_original_sigterm = None
_stdin_watcher: threading.Thread | None = None
_original_termios = None

# Kitty keyboard protocol sequences for Ctrl+C:
#   \x1b[99;5u   — standard CSI u (99 = 'c', 5 = Ctrl)
#   \x1b[1089;5u — extended/alternate encoding seen in some terminals
#   \x03         — plain ETX (normal Ctrl+C that somehow reached stdin)
_INTERRUPT_SEQUENCES = (b"\x1b[99;5u", b"\x1b[1089;5u", b"\x03")


def _signal_handler(signum: int, frame: object) -> None:
    """Handle SIGINT/SIGTERM by setting the shutdown event."""
    name = signal.Signals(signum).name
    log.info("Received %s — requesting graceful shutdown", name)
    shutdown_requested.set()


def _watch_stdin() -> None:
    """Background thread: watch stdin for kitty keyboard protocol Ctrl+C.

    Puts stdin into cbreak mode (non-canonical, no line buffering) so we can
    read escape sequences byte-by-byte as they arrive, without waiting for Enter.
    Original terminal settings are restored on exit.
    """
    global _original_termios

    try:
        fd = sys.stdin.fileno()
    except (ValueError, OSError):
        return  # stdin not a real fd (e.g. piped)

    # Switch to cbreak mode so bytes arrive immediately
    try:
        _original_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except termios.error:
        return  # not a real terminal

    buf = b""
    try:
        while not shutdown_requested.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.3)
                if not ready:
                    continue
                chunk = os.read(fd, 64)
                if not chunk:
                    break  # stdin closed
                buf += chunk
                # Check for any known interrupt sequence
                for seq in _INTERRUPT_SEQUENCES:
                    if seq in buf:
                        log.info("Detected Ctrl+C via stdin (%r) — requesting shutdown", seq)
                        shutdown_requested.set()
                        return
                # Keep only tail to avoid unbounded growth
                buf = buf[-32:]
            except (OSError, ValueError):
                break
    finally:
        # Restore terminal settings
        try:
            if _original_termios is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, _original_termios)
                _original_termios = None
        except (termios.error, OSError):
            pass


def install_handlers() -> None:
    """Install signal handlers and start stdin watcher.

    Call this once at the start of a long-running CLI command.
    Safe to call multiple times (idempotent).
    """
    global _original_sigint, _original_sigterm, _stdin_watcher

    shutdown_requested.clear()

    # Signal handlers (main thread only)
    if threading.current_thread() is threading.main_thread():
        _original_sigint = signal.getsignal(signal.SIGINT)
        _original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    # Stdin watcher for kitty protocol
    if (_stdin_watcher is None or not _stdin_watcher.is_alive()) and sys.stdin.isatty():
        _stdin_watcher = threading.Thread(target=_watch_stdin, daemon=True, name="stdin-watcher")
        _stdin_watcher.start()


def restore_handlers() -> None:
    """Restore original signal handlers and terminal settings."""
    global _original_sigint, _original_sigterm, _original_termios

    if threading.current_thread() is threading.main_thread():
        if _original_sigint is not None:
            signal.signal(signal.SIGINT, _original_sigint)
            _original_sigint = None
        if _original_sigterm is not None:
            signal.signal(signal.SIGTERM, _original_sigterm)
            _original_sigterm = None

    # Restore terminal in case watcher thread didn't clean up
    if _original_termios is not None:
        try:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, _original_termios)
            _original_termios = None
        except (termios.error, OSError, ValueError):
            pass
