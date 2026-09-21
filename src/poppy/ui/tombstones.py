"""Sidecar tombstone storage for the UI.

Soft-delete promise from the marketing copy: "Tombstoned — you can restore for 7
days." We honor it without touching the immutable RetrievalEngine ABC or Memory
dataclass: tombstones live in a separate SQLite table in the same DB file,
managed entirely by the UI layer. On soft-delete the row is snapshotted into
`ui_tombstones` and removed from the engine. On restore it's re-ingested.
Tombstones older than the TTL are purged on UI startup.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poppy.db import apply_row_factory
from poppy.db import connect as connect_db
from poppy.engine._legacy_copies import (
    BACKUP_DDL,
    COPY_CLAIM_DDL,
    COPY_CLAIM_TABLE,
    COPY_DELETION_DDL,
    announced_copy_claim,
    claim_proven_unmarked_copy,
    clear_copy_snapshot,
    grade_copy_snapshot,
    is_marked_copy,
    is_proven_unmarked_copy,
    mark_legacy_announced,
    pending_legacy_announcements,
    rearm_legacy_announcement,
    record_copy_deletions,
    refuse_restorable_copy_snapshot,
)
from poppy.engine._timestamps import chunked, repush_stamp, utc_iso
from poppy.models import Memory, Source

TTL_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS ui_tombstones (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    project TEXT,
    source_type TEXT NOT NULL,
    source_session_id TEXT,
    source_timestamp TEXT NOT NULL,
    confidence REAL NOT NULL,
    related_to TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tombstoned_at TEXT NOT NULL,
    superseded_by TEXT,
    memory_expires_at TEXT,
    token TEXT
);

-- IDs known through successful pushes, live pulls, or the legacy backfill.
CREATE TABLE IF NOT EXISTS sync_remote_memories (
    id TEXT NOT NULL,
    remote_url TEXT NOT NULL,
    PRIMARY KEY (id, remote_url)
);
"""

# Historical copy deletions carry only an id and a timestamp. Keeping them
# separate from Trash prevents redacted speaker text from becoming restorable.
# Both engines and the Trash store preserve these on-disk tables for old stores.
SCHEMA += COPY_DELETION_DDL + BACKUP_DDL + COPY_CLAIM_DDL


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add columns ui_tombstones gained after its first release.

    ``superseded_by`` came with lifecycle work; ``memory_expires_at`` is the
    tombstoned memory's own TTL. It is named apart from
    ``Tombstone.expires_at``, which is the restore-window deadline — two
    different clocks. Without the column a TTL'd memory came back permanent on
    restore and pushed ``expires_at: null`` to the cloud. ``token`` identifies
    one particular tombstone: ``tombstoned_at`` is a wall clock and two deletes
    of the same memory can land on the same value, so a conditional delete keyed
    on the timestamp can remove a replacement it never read. Rows written before
    the column get a token backfilled, so the conditional delete works for them
    too.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    for name in ("superseded_by", "memory_expires_at", "token"):
        if name in cols:
            continue
        try:
            conn.execute(f"ALTER TABLE ui_tombstones ADD COLUMN {name} TEXT")
        except Exception as exc:
            # The check and the ALTER are not atomic across processes: the UI,
            # the CLI and the autosync worker can open the store at the same
            # moment and all see the column missing. Whoever loses that race
            # gets "duplicate column name", which means the migration is done,
            # not that the store is broken.
            #
            # Matched on the message rather than on sqlite3.OperationalError: an
            # encrypted store runs on the SQLCipher driver, whose OperationalError
            # is a different class entirely, so a typed except would let the loser
            # abort startup on exactly the stores that can least afford it.
            if "duplicate column name" not in str(exc).lower():
                raise
    # Backfill tokens for rows written before the column existed, so the
    # conditional delete matches them too. Probed first: this runs on every store
    # open, and an unconditional UPDATE would take a write lock every time.
    if conn.execute("SELECT 1 FROM ui_tombstones WHERE token IS NULL LIMIT 1").fetchone() is not None:
        # randomblob gives each row its own token in one statement.
        conn.execute("UPDATE ui_tombstones SET token = lower(hex(randomblob(16))) WHERE token IS NULL")
        conn.commit()


@dataclass
class Tombstone:
    memory: Memory
    tombstoned_at: datetime
    superseded_by: str | None = None
    # Unique per write, unlike tombstoned_at: identifies THIS tombstone so a
    # conditional delete cannot remove a replacement written in the same tick.
    token: str | None = None
    sent_remotes: set[str] = field(default_factory=set)

    @property
    def expires_at(self) -> datetime:
        return self.tombstoned_at + timedelta(days=TTL_DAYS)


@dataclass(frozen=True)
class CopyDeletion:
    """A historical copy deletion, with no content to expose or restore."""

    id: str
    tombstoned_at: datetime


class TombstoneStore:
    """Manages the `ui_tombstones` sidecar table in the Poppy SQLite DB."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = connect_db(db_path, check_same_thread=False)
        apply_row_factory(self._conn)
        self._conn.executescript(SCHEMA)
        _migrate_columns(self._conn)
        self._migrate_sync_provenance()

    def _migrate_sync_provenance(self) -> None:
        """Backfill once, atomically with the sent_remotes column as the marker.

        Before this version every deletion was sent regardless of provenance.
        Treat every pre-upgrade memory and tombstone as known to every remote
        present at upgrade, reproducing main's behavior for old data without a
        regression. Watermarks cannot reconstruct which IDs reached the cloud.
        Per-ID provenance applies in full to memories created after upgrade and to stores
        with no remote at upgrade, whose known IDs start empty.

        Unreadable state aborts before the marker is added, allowing a retry after
        repair. Absent state means no remotes. File-backed DBs load state beside
        the resolved database path; in-memory DBs never load sync state.
        """
        from poppy.sync.state import SyncState, load

        def columns(table: str) -> set[str]:
            return {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}

        if "sent_remotes" in columns("ui_tombstones"):
            return
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            cols = columns("ui_tombstones")
            if "sent_remotes" in cols:  # Another opener completed the migration.
                return
            state = SyncState()
            if str(self._db_path) != ":memory:":
                try:
                    state = load(self._db_path.resolve().parent, strict=True)
                except (OSError, ValueError, TypeError, AttributeError) as exc:
                    raise RuntimeError(
                        f"Cannot migrate sync provenance: unable to load {self._db_path.parent / 'sync_state.json'}. "
                        "Repair the sync state and reopen the store; migration has not been marked complete."
                    ) from exc
            self._conn.execute("ALTER TABLE ui_tombstones ADD COLUMN sent_remotes TEXT NOT NULL DEFAULT '{}'")
            if "sync_evidence" in cols:
                # Preserve acknowledgements from prerelease stores, not the
                # pending/unseen guesses that this design replaces.
                self._conn.execute("""UPDATE ui_tombstones SET sent_remotes = (
                    SELECT json_group_object(key, 1) FROM json_each(sync_evidence) WHERE value = 'sent'
                )""")
                self._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sync_evidence")
            live_cols = columns("memories")
            for url in state.remotes:
                self._conn.execute(
                    "INSERT OR IGNORE INTO sync_remote_memories (id, remote_url) SELECT id, ? FROM ui_tombstones",
                    (url,),
                )
                if live_cols:
                    eligible = "WHERE COALESCE(is_closet, 0) = 0" if "is_closet" in live_cols else ""
                    self._conn.execute(
                        "INSERT OR IGNORE INTO sync_remote_memories (id, remote_url) "
                        f"SELECT id, ? FROM memories {eligible}",
                        (url,),
                    )
            # Never infer sent acknowledgements from legacy watermarks: older
            # clients could advance them incorrectly. Legacy deletions re-send
            # once; upserts are idempotent, as on main's watermark-reset replay.
            self._conn.execute("""INSERT OR IGNORE INTO sync_remote_memories (id, remote_url)
                SELECT t.id, sent.key FROM ui_tombstones t, json_each(t.sent_remotes) sent""")

    def add(
        self,
        memory: Memory,
        *,
        superseded_by: str | None = None,
        tombstoned_at: datetime | None = None,
        sent_to: str | None = None,
    ) -> Tombstone:
        """Record a soft-delete.

        ``tombstoned_at`` is WHEN THE DELETION HAPPENED. A deletion this device
        is making leaves it None and gets now. A deletion PULLED from another
        device must pass the incoming one: stamping receipt time turns a delete
        that happened yesterday into a brand-new one, and push then sends it back
        up as newer than anything written in between — overwriting an
        independent note another device wrote while the pull was in flight.
        Pass ``sent_to`` for a pulled deletion so creation and its
        acknowledgement are atomic; it must never be re-announced to its source.
        """
        # Stored as UTC text: `purge_expired` compares this column with the
        # push watermark as strings, and one instant must have one spelling.
        now = (tombstoned_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        token = uuid.uuid4().hex
        with self._lock:
            # Never backwards. A local forget at 14:00 followed by a pull of an
            # older cloud deletion at 12:00 must leave the record saying 14:00 —
            # a downgraded deletion, once pushed, loses to another device's 13:00
            # recreation and lets the forgotten content come back on the next
            # pull. And the STRICTLY OLDER deletion's snapshot is not written
            # either: it describes an earlier life of the id, so replacing the
            # snapshot would put that older text in Trash in place of the memory
            # the user actually deleted at 14:00, restorable, and on the wire.
            # An equal or newer deletion replaces the row as before.
            # Only a PULLED deletion can be the older one: a deletion this device
            # is making now always snapshots the row it is about to remove, or
            # a clock-skewed record from elsewhere would leave the memory the
            # user just forgot with no way back.
            existing = self._conn.execute(
                "SELECT tombstoned_at FROM ui_tombstones WHERE id = ?", (memory.id,)
            ).fetchone()
            if (
                tombstoned_at is not None
                and existing is not None
                and datetime.fromisoformat(existing["tombstoned_at"]) > now
            ):
                kept = self.get(memory.id)
                if kept is not None:
                    return kept
            sent_remotes = {sent_to.rstrip("/")} if sent_to else set()
            self._conn.execute(
                """INSERT OR REPLACE INTO ui_tombstones (
                    id, content, memory_type, project, source_type, source_session_id,
                    source_timestamp, confidence, related_to, created_at, updated_at, tombstoned_at,
                    superseded_by, memory_expires_at, token, sent_remotes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.project,
                    memory.source.type,
                    memory.source.session_id,
                    # The snapshot's own stamps are normalised too: `restore`
                    # writes them straight back into `memories`, so a row that
                    # went into Trash with a non-UTC offset would come out of it
                    # carrying one.
                    utc_iso(memory.source.timestamp),
                    memory.confidence,
                    json.dumps(memory.related_to),
                    utc_iso(memory.created_at),
                    utc_iso(memory.updated_at),
                    now.isoformat(),
                    superseded_by,
                    utc_iso(memory.expires_at),
                    token,
                    json.dumps(dict.fromkeys(sent_remotes, 1)),
                ),
            )
            self._conn.commit()
        return Tombstone(
            memory=memory, tombstoned_at=now, superseded_by=superseded_by, token=token, sent_remotes=sent_remotes
        )

    def known_ids(self, remote_url: str) -> set[str]:
        """IDs this remote is known to hold, independent of local edits/deletes."""
        with self._lock:
            return {
                row["id"]
                for row in self._conn.execute(
                    "SELECT id FROM sync_remote_memories WHERE remote_url = ?", (remote_url.rstrip("/"),)
                )
            }

    def note_remote_memories(self, memory_ids: set[str], remote_url: str) -> None:
        """Commit a push/pull's known IDs together, before its watermark."""
        if not memory_ids:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO sync_remote_memories (id, remote_url) VALUES (?, ?)",
                [(mid, remote_url.rstrip("/")) for mid in memory_ids],
            )

    def mark_sent(self, tombstones: list[Tombstone], remote_url: str) -> None:
        """Acknowledge a batch without clearing any concurrent replacements."""
        if not tombstones:
            return
        path = "$." + json.dumps(remote_url.rstrip("/"))
        with self._lock, self._conn:
            self._conn.executemany(
                "UPDATE ui_tombstones SET sent_remotes = json_set(sent_remotes, ?, 1) WHERE id = ? AND token = ?",
                [(path, ts.memory.id, ts.token) for ts in tombstones],
            )

    def add_copy_deletions(
        self,
        memory_ids: list[str],
        *,
        now: datetime | None = None,
        applying_remote_deletion: bool = False,
        authoritative: bool = False,
    ) -> list[CopyDeletion]:
        """Retain content-free deletion evidence for copies from older releases.

        Use the event's timestamp for a remote deletion so it cannot suppress a
        newer real memory at the same id. Local deletions use the current time.
        """
        if not memory_ids:
            return []
        ids = list(memory_ids)
        rows = []
        with self._lock:
            record_copy_deletions(
                self._conn,
                ids,
                when=now,
                applying_remote_deletion=applying_remote_deletion,
                authoritative=authoritative,
            )
            self._conn.commit()
            for batch in chunked(ids):
                placeholders = ",".join("?" * len(batch))
                rows.extend(
                    self._conn.execute(
                        f"SELECT id, tombstoned_at FROM closet_tombstones WHERE id IN ({placeholders})",
                        batch,
                    ).fetchall()
                )
        return [CopyDeletion(id=r["id"], tombstoned_at=datetime.fromisoformat(r["tombstoned_at"])) for r in rows]

    def claim_proven_unmarked_copy(self, memory_id: str) -> bool:
        """Prove an old unmarked copy and retain its deletion evidence if so."""
        with self._lock:
            claimed = claim_proven_unmarked_copy(self._conn, memory_id)
            self._conn.commit()
        return claimed

    def is_proven_unmarked_copy(self, memory_id: str) -> bool:
        """Whether a live unmarked row at ``memory_id`` is a PROVEN copy of its live parent."""
        with self._lock:
            return is_proven_unmarked_copy(self._conn, memory_id)

    def grade_copy_snapshot(self, memory: Memory) -> str | None:
        """Grade an INCOMING row against its live parent: a tier, or None.

        Read by pull twice over. Before a content-carrying tombstone becomes a Trash
        entry (only an old client deleting a copy row by hand produces one, and filing
        it put the speaker text in front of the user as restorable), and before a live
        row is suppressed on the strength of an announcement claim, which is an
        existence hint rather than a clock.
        """
        with self._lock:
            return grade_copy_snapshot(
                self._conn,
                memory.id,
                content=memory.content,
                related_to=memory.related_to,
                created_at=memory.created_at.isoformat(),
            )

    def claim_leaked_copy(self, memory_id: str, seen_at: datetime) -> None:
        """Retain a proven legacy copy claim at its observed timestamp.

        The claim protects old deletion records from premature Trash purge.
        Only proven copies qualify; a claim must never hide a real memory.
        """
        with self._lock:
            rearm_legacy_announcement(self._conn, memory_id, seen_at.isoformat())
            self._conn.commit()

    def refuse_restorable_copy_snapshot(self, memory_id: str) -> str | None:
        """The parent to name in a refusal if the STORED Trash entry is PROVEN a copy.

        Drops the entry and claims its cloud row on that proof, and answers None for
        everything else — a LIKELY entry restores, like any row short of proof. Called
        by ``restore`` only when nothing is live at the id.
        """
        with self._lock:
            parent = refuse_restorable_copy_snapshot(self._conn, memory_id)
            self._conn.commit()
        return parent

    def clear_copy_snapshot(self, memory_id: str) -> bool:
        """Drop a legacy Trash entry at a live marked copy's id if it holds the parent's text.

        Called by ``forget`` BEFORE the copy row is deleted: the row's back-reference
        is what names the parent whose derivation the entry is compared with.
        """
        with self._lock:
            cleared = clear_copy_snapshot(self._conn, memory_id)
            self._conn.commit()
        return cleared

    def get_copy_deletion(self, memory_id: str) -> CopyDeletion | None:
        """The recorded deletion of ``memory_id`` as a derived copy, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, tombstoned_at FROM closet_tombstones WHERE id = ?", (memory_id,)
            ).fetchone()
        if row is None:
            return None
        return CopyDeletion(id=row["id"], tombstoned_at=datetime.fromisoformat(row["tombstoned_at"]))

    def announced_copy_claim(self, memory_id: str) -> datetime | None:
        """Whether this store ever PROVED the id to be a leaked copy, and when it was seen.

        Outlives the local deletion record, which is what pull needs when a concurrent
        push purges that record while pull waits for the write gate. Pull treats the
        value as an existence hint and decides by evidence; the time is only ever used
        to raise a bar a deletion record already set.
        """
        with self._lock:
            stamp = announced_copy_claim(self._conn, memory_id)
        if stamp is None:
            return None
        try:
            claimed = datetime.fromisoformat(stamp)
        except (ValueError, OverflowError):
            # `utc_iso` passes a value it cannot parse through verbatim, so this
            # column is not guaranteed to hold a timestamp. An unusable claim simply
            # sets no bar.
            return None
        # Aware, always: these are compared with parsed row timestamps, and a naive
        # one would raise rather than answer.
        return claimed if claimed.tzinfo is not None else claimed.replace(tzinfo=timezone.utc)

    def has_copy_deletion(self, memory_id: str) -> bool:
        """Whether this device deleted ``memory_id`` as a derived per-speaker copy.

        Read by sync's pull: while the deletion is inside its retention window,
        an older cloud row for that id must not be re-ingested, or the redacted
        speaker text comes straight back as an ordinary memory. Past the window
        the row is gone and a resurrecting cloud copy is the pre-marker cleanup
        case.
        """
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM closet_tombstones WHERE id = ?", (memory_id,)).fetchone()
        return row is not None

    def pending_legacy_announcements(self) -> list[tuple[str, str | None]]:
        """Legacy claims whose deletion evidence still needs to be retained."""
        with self._lock:
            return pending_legacy_announcements(self._conn)

    def local_deletion_wins(self, memory_id: str, updated_at: datetime) -> bool:
        """Whether a recorded local deletion supersedes an incoming version of this id."""
        from poppy.sync.state import local_deletion_wins

        with self._lock:
            return local_deletion_wins(self._conn, memory_id, updated_at)

    def record_local_deletion(self, memory_id: str, deleted_at: datetime) -> None:
        """Record a deletion that is neither restorable nor pushed.

        Taken under the store lock like every other write on this connection: the
        daemon, the CLI and the hook share it, and an unlocked write can join
        another thread's open transaction and be rolled back with it.
        """
        from poppy.sync.state import record_local_deletion

        with self._lock:
            record_local_deletion(self._conn, memory_id, deleted_at)

    def local_deletion_at(self, memory_id: str) -> datetime | None:
        """When this store deleted ``memory_id`` locally, or None if it never did."""
        from poppy.sync.state import local_deletion_at

        with self._lock:
            return local_deletion_at(self._conn, memory_id)

    def clear_local_deletion(self, memory_id: str) -> None:
        """Forget the deletion record for an id a real memory has taken back."""
        from poppy.sync.state import clear_local_deletion

        with self._lock:
            clear_local_deletion(self._conn, memory_id)

    def repush_stamp(self) -> str | None:
        """When the UTC rewrite moved a push candidate here, or None if it never did.

        Read through the store because that is where the fact lives: it is a
        property of this SQLite file, and both tables it concerns are in here. What
        each REMOTE has done about it lives in ``sync_state.json`` instead, keyed by
        URL — nothing here is ever cleared.
        """
        with self._lock:
            return repush_stamp(self._conn)

    def mark_legacy_announced(
        self, memory_ids: list[str] | list[tuple[str, str | None]], *, when: datetime | None = None
    ) -> None:
        """Clear a legacy pending flag after its deletion evidence is settled.

        Pass ``(id, stamp)`` pairs to clear only the exact claim that was sent.
        """
        with self._lock:
            mark_legacy_announced(self._conn, memory_ids, when=when)
            self._conn.commit()

    def list_copy_deletions(self) -> list[CopyDeletion]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, tombstoned_at FROM closet_tombstones ORDER BY tombstoned_at DESC"
            ).fetchall()
        return [CopyDeletion(id=r["id"], tombstoned_at=datetime.fromisoformat(r["tombstoned_at"])) for r in rows]

    def remove(self, memory_id: str, *, token: str | None = None) -> bool:
        """Delete a tombstone; returns whether a row was actually removed.

        Pass the ``token`` of the tombstone you read to make the delete
        conditional on it still being that one. Sync's pull needs that: between
        reading a tombstone and clearing it, a concurrent local delete can
        replace it, and a delete by id alone would silently swallow that fresh
        deletion. The token, not ``tombstoned_at``, is what makes the condition
        safe — two deletes of one memory can share a wall-clock timestamp.
        """
        with self._lock:
            if token is None:
                cursor = self._conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (memory_id,))
            else:
                cursor = self._conn.execute(
                    "DELETE FROM ui_tombstones WHERE id = ? AND token = ?",
                    (memory_id, token),
                )
            self._conn.commit()
            return cursor.rowcount > 0

    def get(self, memory_id: str) -> Tombstone | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM ui_tombstones WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_tombstone(row)

    def get_public(self, memory_id: str) -> Tombstone | None:
        """Read Trash without exposing a snapshot of a hidden live row."""
        with self._lock:
            if is_marked_copy(self, memory_id):
                return None
            return self.get(memory_id)

    def list_public(self) -> list[Tombstone]:
        """Keep hidden live rows out of Trash and combined dashboard listings."""
        with self._lock:
            return [t for t in self.list_all() if not is_marked_copy(self, t.memory.id)]

    def list_all(self) -> list[Tombstone]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM ui_tombstones ORDER BY tombstoned_at DESC").fetchall()
        return [self._row_to_tombstone(r) for r in rows]

    def purge_expired(self, *, pushed_through: str | None, require_sent: bool = False) -> int:
        """Age out records past the restore window. Returns ui tombstones purged.

        A tombstone is the ONLY thing that carries a deletion to the cloud: push
        sends deletions as tombstones and nothing else. So age alone must never
        remove one — a laptop offline for a fortnight, an unconfigured remote or
        a revoked key all leave week-old tombstones that have never been sent,
        and dropping them leaves the cloud row live for the next pull to
        re-ingest. The forgotten memory comes back.

        ``pushed_through`` is the point up to which deletions no longer need to
        be kept. Two callers, two ways of arriving at it:

        * ``sync``, after push, passes the current time and sets ``require_sent``:
          known IDs must be acknowledged by every remote that knows them.
          Sent marks are independent of the live watermark; unknown IDs have
          no pending work.
        * a caller on a store with NO remote configured passes the current time.
          There is nowhere for a deletion to travel to, so nothing is waiting on
          it and the seven-day window applies on age alone — which is what keeps
          Trash from growing without bound for a user who never enables sync.

        The argument is required, with no default. Passing ``None`` touches
        neither deletion table, which is the safe answer for a caller that cannot
        tell which situation it is in — but it has to be said out loud, because a
        caller that silently got the no-op would believe it had aged Trash out.

        The migration's pre-images have nothing to push either way, so they
        always age out on the window alone.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=TTL_DAYS)).isoformat()
        # The purge bound is compared with stored UTC text, so it gets the same
        # spelling: `12:00+02:00` is 10:00Z, and as raw text it would purge an
        # 11:00Z record that has not been acknowledged.
        pushed_through = utc_iso(pushed_through)
        with self._lock:
            purged = 0
            if pushed_through is not None:
                cursor = self._conn.execute(
                    """DELETE FROM ui_tombstones WHERE tombstoned_at < ? AND tombstoned_at <= ?
                    AND (? = 0 OR NOT EXISTS (
                        SELECT 1 FROM sync_remote_memories known
                        WHERE known.id = ui_tombstones.id AND NOT EXISTS (
                            SELECT 1 FROM json_each(ui_tombstones.sent_remotes) sent
                            WHERE sent.key = known.remote_url)))""",
                    (cutoff, pushed_through, require_sent),
                )
                purged = cursor.rowcount
                # A historical copy deletion record makes pull skip the cloud's
                # stale copy of that id. Pending legacy claims preserve evidence
                # for old speaker snapshots even after the ordinary Trash window.
                # Store-open cleanup retires those claims together with recording
                # deletion evidence in the sync ledger.
                self._conn.execute(
                    f"""DELETE FROM closet_tombstones WHERE tombstoned_at < ? AND tombstoned_at <= ?
                    AND id NOT IN (SELECT id FROM {COPY_CLAIM_TABLE} WHERE announce_pending = 1)""",
                    (cutoff, pushed_through),
                )
            # Legacy claims outlive the Trash window. Migration pre-images have
            # no remote work and age out on the window alone.
            self._conn.execute("DELETE FROM closet_migration_backup WHERE migrated_at < ?", (cutoff,))
            self._conn.commit()
            return purged

    @staticmethod
    def _row_to_tombstone(row: sqlite3.Row) -> Tombstone:
        keys = row.keys()
        raw_expiry = row["memory_expires_at"] if "memory_expires_at" in keys else None
        memory = Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            source=Source(
                type=row["source_type"],
                session_id=row["source_session_id"],
                timestamp=datetime.fromisoformat(row["source_timestamp"]),
            ),
            project=row["project"],
            related_to=json.loads(row["related_to"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            confidence=row["confidence"],
            expires_at=datetime.fromisoformat(raw_expiry) if raw_expiry else None,
        )
        return Tombstone(
            memory=memory,
            tombstoned_at=datetime.fromisoformat(row["tombstoned_at"]),
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
            token=row["token"] if "token" in keys else None,
            sent_remotes=set(json.loads(row["sent_remotes"])),
        )
