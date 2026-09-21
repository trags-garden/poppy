"""One spelling per instant for every timestamp this store keeps.

Poppy stores timestamps as ISO-8601 TEXT, and several paths compare that text
directly inside SQLite: the push watermark filter in :mod:`poppy.sync`, the TTL
expiry purge on both engines, Trash's seven-day window. A text comparison of two
ISO strings compares WALL CLOCK, so it only answers a question about TIME when
every value is spelled in the same offset. ``2026-07-01T12:00:00+02:00`` and
``2026-07-01T11:00:00+00:00`` sort one way as text and the opposite way in fact.

Rows carrying a non-UTC offset are real. An importer stamping local time, a
research harness, a hand-built row and a cloud row minted by another client all
reach ``memories`` without passing through Poppy's own clock. Before this module
such a row was stored as it came: a tombstone at ``11:00+00:00`` sorted BELOW a
parent last pushed at ``12:00+02:00``, so push skipped the newer deletion and the
cloud kept the redacted parent live; a memory expiring an hour from now expressed
at ``-12:00`` sorted below ``now`` and was purged on the spot.

Timestamp normalization applies to live rows and retained deletion records:

  * WRITE TIME is the rule. Every engine write puts ``source_timestamp``,
    ``created_at``, ``updated_at`` and ``expires_at`` through
    :func:`utc_iso` so writes share one spelling.
  * A ONE-OFF REWRITE brings the rows already there into that spelling.
    :func:`normalise_stored_timestamps` runs once per store, from the engine
    constructors after legacy cleanup.

Where a compare site can afford to, it ALSO parses rather than trusting the
spelling, so a row that arrives some other way (a 0.2.4 client sharing
``~/.poppy``, a direct ``sqlite3`` write) still sorts correctly: the TTL purge on
both engines decides per row in Python, and the push watermark is normalised on
both sides of its comparison. Trash's seven-day window still compares text in
SQL, which is correct because both sides of it are now canonical UTC. That is
belt-and-braces; the write-time rule is what keeps the store consistent.

The rewrite records completion in the same transaction as the changed rows,
so an interrupted migration retries on the next open.

"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def utc_iso(value: str | datetime | None) -> str | None:
    """Canonical UTC text; preserve unreadable or out-of-range legacy values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(value)
        except (ValueError, OverflowError):
            return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(timezone.utc).isoformat()
    except OverflowError:
        return value if isinstance(value, str) else value.isoformat()


def chunked(items: list[str], size: int | None = None) -> list[list[str]]:
    limit = size or 500
    return [items[i : i + limit] for i in range(0, len(items), limit)]


# Name-keyed log of one-off DATA migrations — the ones whose having-run is not
# visible in the schema. Schema-shaped migrations stay as they are: a column's
# presence is a better signal than a bookkeeping row, because it cannot disagree
# with the thing it describes.
MIGRATIONS_TABLE = "poppy_migrations"
MIGRATIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {MIGRATIONS_TABLE} (
    name TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

TIMESTAMP_MIGRATION = "utc_timestamp_text"

# Recorded, with a timestamp, when the rewrite changed a row in one of the two
# tables PUSH READS. A STABLE FACT about this store: written once, never cleared,
# never consumed.
#
# Why a re-push is needed at all. The watermark records how far push got under the
# OLD ordering. If a row failed under that ordering and was retried by being ABOVE
# the watermark as text, re-spelling it can move it BELOW: a row at
# ``11:00+02:00`` (09:00Z) sat above a watermark of ``10:00+00:00`` and was retried
# every sync; normalised to ``09:00+00:00`` it falls under the
# ``iso <= watermark`` filter and is never sent again, with push reporting no
# errors at all. Normalising the watermark cannot recover it: the watermark is one
# string and says nothing about WHICH rows above it had succeeded. So push makes
# one pass over everything instead. Upserts are idempotent and the server's
# freshness gate rejects stale rows, so the cost is the requests and nothing else.
#
# It is a FACT rather than a FLAG because a flag has to be consumed, and there is
# no way to consume one here that is correct. Clearing it needs its own write to
# this database, which cannot be atomic with the watermark save in
# ``sync_state.json``: an interrupt between the two loses the recovery with the row
# still unsent. And "the re-push happened" is not a property of the store at all —
# it is per REMOTE, since each URL keeps its own watermark, so a single flag let
# the first remote consume the recovery and every other one skip the row for ever.
#
# Instead each ``RemoteState`` records the stamp it has already re-pushed for
# (``repush_done_for``), and it is written by the SAME state save that records how
# far that cycle got. One write, so there is no window; per remote, so every remote
# recovers; and a save that never lands leaves both the watermark and the marker
# untouched, so the next push simply does the pass again.
REPUSH_MARKER = "utc_timestamp_text_repush"

# The tables push enumerates: ``memories`` (live rows) and ``ui_tombstones``
# (deletions). A rewrite confined to the legacy copy tables cannot have moved a
# push candidate, so it asks for no re-push.
PUSH_SOURCE_TABLES = ("memories", "ui_tombstones")

# Every (table, columns) pair holding an ISO-8601 instant as text, with the
# primary key ``id`` in all of them. Tables and columns are checked for existence
# before being touched: the same store is opened by clients of several versions,
# and ``ui_tombstones`` in particular is created by the UI sidecar rather than by
# either engine, so it can legitimately be absent here.
#
# ``ui_tombstones`` is named as a literal because :mod:`poppy.ui.tombstones`
# imports the engine side, and the dependency may not run the other way.
TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "memories": ("source_timestamp", "created_at", "updated_at", "expires_at"),
    "ui_tombstones": (
        "source_timestamp",
        "created_at",
        "updated_at",
        "tombstoned_at",
        "memory_expires_at",
    ),
    "closet_tombstones": ("tombstoned_at",),
    "closet_migration_backup": ("created_at", "updated_at", "migrated_at"),
    "legacy_closet_ids": ("legacy_updated_at", "announced_at"),
}


def migration_applied(conn: sqlite3.Connection, name: str) -> bool:
    """Whether the one-off data migration ``name`` has run on this store."""
    if not _table_exists(conn, MIGRATIONS_TABLE):
        return False
    return conn.execute(f"SELECT 1 FROM {MIGRATIONS_TABLE} WHERE name = ?", (name,)).fetchone() is not None


def record_migration(conn: sqlite3.Connection, name: str, *, when: datetime | None = None) -> None:
    """Record that ``name`` has run. Caller owns the transaction."""
    conn.execute(MIGRATIONS_DDL)
    conn.execute(
        f"INSERT OR REPLACE INTO {MIGRATIONS_TABLE} (name, applied_at) VALUES (?, ?)",
        (name, utc_iso(when or datetime.now(timezone.utc))),
    )


def repush_stamp(conn: sqlite3.Connection) -> str | None:
    """When the rewrite moved a push candidate in this store, or None if it never did.

    The value identifies the event, so each remote can record which one it has
    already re-pushed for. Read-only: nothing ever clears or rewrites it, which is
    what makes it safe to consult from any number of processes and remotes.
    """
    if not _table_exists(conn, MIGRATIONS_TABLE):
        return None
    row = conn.execute(f"SELECT applied_at FROM {MIGRATIONS_TABLE} WHERE name = ?", (REPUSH_MARKER,)).fetchone()
    return row[0] if row is not None else None


def expiry_passed(stamp: str | None, now: datetime) -> bool:
    """Whether a stored ``expires_at`` names an instant at or before ``now``.

    The TTL purge on both engines asks this per row instead of comparing the
    stored text with ``now`` in SQL. Writes normalise, so in a consistent store
    the two agree — but a row that arrived another way can carry any offset, and
    as TEXT ``2026-07-01T13:00:00-12:00`` (an hour in the future) sorts below
    ``2026-07-01T12:00:00+00:00``, so the SQL comparison hard-deleted live
    memories on the spot.

    A stamp that does not parse is NOT expired: a purge is irreversible, and a
    value no clock wrote is not evidence that the memory's life is over. A naive
    stamp is read as UTC, the same convention the rest of the store uses.
    """
    if not stamp:
        return False
    try:
        parsed = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= now


def _rewrite_table(conn: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> int:
    """Rewrite deviant timestamps in one table. Returns the number of rows changed."""
    if not _table_exists(conn, table):
        return 0
    present = [c for c in columns if c in _columns(conn, table)]
    if not present:
        return 0
    # Positional access throughout, so this works whether or not the caller set
    # ``row_factory``: both engines do, the migration tests may not.
    selected = ", ".join(present)
    rows = conn.execute(f"SELECT id, {selected} FROM {table}").fetchall()
    changed = 0
    for row in rows:
        updates: list[tuple[str, str]] = []
        for index, column in enumerate(present, start=1):
            raw = row[index]
            if raw is None:
                continue
            canonical = utc_iso(raw)
            # ``utc_iso`` returns an unparseable string unchanged, so a row
            # written by something that is not a clock is left exactly as it is
            # rather than being mangled or raising mid-migration.
            if canonical != raw:
                updates.append((column, canonical))  # type: ignore[arg-type]
        if not updates:
            continue
        assignments = ", ".join(f"{column} = ?" for column, _ in updates)
        conn.execute(
            f"UPDATE {table} SET {assignments} WHERE id = ?",
            [value for _, value in updates] + [row[0]],
        )
        changed += 1
    return changed


def normalise_stored_timestamps(conn: sqlite3.Connection) -> int:
    """Rewrite every stored timestamp into UTC text, once per store.

    Returns the number of rows changed (0 when the migration has already run).

    The rewrite and its record share ONE transaction, so a store can never be
    marked done without having been rewritten, and a crash in the middle leaves
    no record and simply retries on the next open. The write lock is claimed up
    front for the same reason the marker migration claims it: the read that
    decides what to rewrite and the writes that do it must not have another
    process's write between them.

    Idempotent by construction as well as by record: a value already in canonical
    spelling is not rewritten, so running the body twice changes nothing.
    """
    if migration_applied(conn, TIMESTAMP_MIGRATION):
        return 0

    own_txn = not conn.in_transaction
    if own_txn:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            # A connection shared by threads can have opened a transaction
            # between the check and this statement; that transaction protects the
            # sequence just as well, and its owner commits it. Any other failure
            # — above all "database is locked" after the busy timeout — must
            # abort: rewriting without the write lock is exactly the race this
            # claim exists to close.
            if "within a transaction" not in str(exc):
                raise
            own_txn = False

    try:
        # Re-read under the lock: another process may have migrated while we
        # waited on the busy timeout.
        if migration_applied(conn, TIMESTAMP_MIGRATION):
            if own_txn:
                conn.rollback()
            return 0
        changed = 0
        moved_a_push_candidate = False
        for table, columns in TIMESTAMP_COLUMNS.items():
            rewritten = _rewrite_table(conn, table, columns)
            changed += rewritten
            if rewritten and table in PUSH_SOURCE_TABLES:
                moved_a_push_candidate = True
        record_migration(conn, TIMESTAMP_MIGRATION)
        if moved_a_push_candidate:
            # Re-spelling a row can move it below a watermark recorded under the
            # old ordering, so every remote owes one full pass. Recorded in the
            # SAME transaction as the rewrite: a marker without the rewrite costs a
            # pointless pass, a rewrite without the marker loses a row for good.
            record_migration(conn, REPUSH_MARKER)
    except BaseException:
        if own_txn:
            conn.rollback()
        raise
    if own_txn:
        conn.commit()
    return changed


def later_stamp(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    try:
        left, right = (datetime.fromisoformat(a), datetime.fromisoformat(b))
    except (ValueError, OverflowError, TypeError):
        return a
    if left.tzinfo is None:
        left = left.replace(tzinfo=timezone.utc)
    if right.tzinfo is None:
        right = right.replace(tzinfo=timezone.utc)
    return a if left >= right else b
