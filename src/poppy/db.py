"""Shared SQLite connection setup for Poppy's local store.

Poppy opens the same ``~/.poppy`` database from several places that can run at
the same time — the long-lived MCP server, interactive CLI commands, and the
background auto-sync that fires after each write. With SQLite's default
rollback-journal mode a writer takes an exclusive lock that blocks every other
connection, and a contended connection that needs to upgrade its lock gets an
immediate ``sqlite3.OperationalError: database is locked`` (the busy handler is
deliberately skipped in those deadlock-prone cases). The result was spurious
lock errors during normal MCP + CLI + sync overlap.

WAL mode lets a writer and any number of readers proceed concurrently, and an
explicit busy timeout makes a genuinely contended writer wait instead of
failing instantly. ``synchronous = NORMAL`` is the recommended durability level
under WAL (safe across application crashes; only an OS/power loss can lose the
last commit, which is acceptable for a local cache that also syncs to Trags).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Comfortably longer than any single Poppy write/commit, while still surfacing a
# real deadlock rather than hanging forever.
BUSY_TIMEOUT_MS = 5000


def connect(db_path: Path | str, *, check_same_thread: bool = False) -> sqlite3.Connection:
    """Open a Poppy SQLite connection with WAL + a busy timeout applied.

    Drop-in replacement for ``sqlite3.connect(str(db_path), check_same_thread=...)``.
    Callers still set their own ``row_factory`` and run their schema/migrations.
    """
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=BUSY_TIMEOUT_MS / 1000,
    )
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn
