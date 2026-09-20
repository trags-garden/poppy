"""Provenance marker for bloom's synthetic per-speaker closet rows.

Bloom expands a multi-speaker memory into one extra row per speaker, so a query
about Alice can score an Alice-only copy above the joint session::

    main memory:    id=D1,               content=[alice turns + bob turns]
    alice closet:   id=D1_closet_alice,  content=[alice turns only]

Those copies are *derived data*: they duplicate the parent's text, they must
disappear when the parent is redacted, and they must never be listed or synced
as memories in their own right. So every write path needs to answer "is this row
a closet?" — and answer it for rows of ANY origin, because ids reach the store
from the cloud, from the web API, from importers and from external harnesses,
none of which use the local id generators.

**Closet identity is a positive provenance marker, set when bloom CREATES the
row** — the ``memories.is_closet`` column. It is never inferred from the id, from
a ``_closet_`` substring, from ``related_to`` structure, or from a ``mem_``
prefix. Each of those three inferential tests shipped and then broke on an input
that merely shared the shape: an externally-minted parent id such as
``sess-2026-01`` produces closets no id test recognises, so a redaction left the
speaker text live and recallable. A marker set at creation is decidable for every
parent-id origin, and a real memory called ``customer_closet_notes`` is never
marked, so it is never hidden and never deleted.

Two rules follow from that, and every path obeys them:

  * the marker is set ONLY by bloom's closet synthesis; every other write of a
    row clears it, so a real note written over a closet's id becomes a real
    memory rather than inheriting derived-data status;
  * a marked closet is LOCAL-ONLY derived data. Sync never touches one in
    either direction: it is not pushed, and an incoming row for a marked id is
    not applied. That decision reads the local marker, never the id's shape, so
    a real cloud memory that merely looks like a closet syncs normally — unless
    this device happens to derive a copy at exactly that id.

    That collision is worth stating in full. If a real cloud memory's id is
    exactly one this device derives a copy at, the incoming row is skipped and
    the watermark advances past it, so THIS device never sees that memory; it
    stays in the cloud and on every other device. Nothing is sent for the local
    copy when it is later redacted: its deletion record is local-only, and the
    cloud is told only about ids PROVEN to be leaked copies, stamped with the
    leaked row's own timestamp, so a newer real memory at such an id wins the
    server's freshness comparison and survives. Reaching even the skip requires
    a cloud memory whose id is ``<a memory this device holds>_closet_<a speaker
    in it>``, so it cannot happen between ids Poppy mints; it is possible for
    ids supplied by the web API, an importer, or a research harness.

The one place inference cannot be avoided is data written before the column
existed — a store being migrated, or a copy pulled from an account an older
client synced. There is no marker in it to read, so the evidence is GRADED and
each grade earns only what it can support (see :func:`classify_closet_row`):

  * PROVEN — the row's text is exactly what the parent derives right now. Mark
    it, align its derived fields, and announce the cloud's copy for deletion.
    Announcing is the only irreversible act here, and it is the only grade that
    may do it.
  * LIKELY — closet-shaped, same ``created_at``, the parent derives the id, but
    the text differs. Snapshot, mark, and KEEP the text. Hidden and covered by
    the parent's redaction, never rewritten and never announced, so a wrong
    guess costs visibility rather than data.
  * ORPHAN — shape matches but no parent confirms it. Left entirely alone and
    reported for a human.

An id alone never justifies touching a memory, and a store that never ran the
default engine is never graded at all: nothing in it ever derived a copy.

This module holds the marker, the pure-text closet derivation it depends on, and
the SQL helpers both engines share. It imports no ML runtime and no engine
module, so ``seed`` — the dependency-free fallback — can use it directly.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

# Separator between a parent id and the speaker slug in a closet id. Part of the
# id SCHEME, not an identity test: it builds ids and scopes deletes to one
# parent's children, and every such statement ALSO requires the marker.
CLOSET_SEPARATOR = "_closet_"

MARKER_COLUMN = "is_closet"

# Predicates over the marker column. ``COALESCE`` covers the window between
# ``ALTER TABLE`` and a backfill on a store another process is migrating, and any
# row written through SQL that omits the column.
# The marker is a STATE, not a flag, because two kinds of row are copies and
# only one of them may be regenerated.
CLOSET_DERIVED = 1  # this engine derived it; rebuilt freely from the parent
CLOSET_ADOPTED_UNVERIFIED = 2  # adopted on inference; its text is NOT ours to rewrite

IS_CLOSET_SQL = f"COALESCE({MARKER_COLUMN}, 0) >= 1"
NOT_CLOSET_SQL = f"COALESCE({MARKER_COLUMN}, 0) = 0"
# Only the rows this engine derived. Re-derivation deletes and rewrites these;
# an adopted-unverified row is left exactly where it is, because rewriting it
# would destroy the text the LIKELY grade deliberately preserved.
IS_DERIVED_CLOSET_SQL = f"COALESCE({MARKER_COLUMN}, 0) = {CLOSET_DERIVED}"

# Written into ``memory_embeddings.model_id`` when a vector no longer describes
# the text of its row. Any value that is not a live engine's model_id would do; a
# named sentinel makes the reason legible in the table and in doctor output.
# Re-exported by ``engine.seed`` as ``SEED_INVALIDATED_MODEL_ID``.
STALE_VECTOR_MODEL_ID = "seed-invalidated"


# --- id helpers -----------------------------------------------------------


def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters so ``value`` matches literally under ESCAPE '\\'."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _closet_like_pattern(memory_id: str) -> str:
    """LIKE pattern matching a memory's synthetic ``_closet_<speaker>`` rows.

    The id and the literal ``_closet_`` separator are escaped so only the
    trailing speaker slug stays a wildcard. Ids reach delete/ingest from
    ``sync pull`` unvalidated; one containing ``%`` or ``_`` would otherwise
    over-match and silently touch unrelated memories. Use with ``ESCAPE '\\'``.

    This SCOPES a statement to one parent's children. It never decides that a row
    IS a closet — every query built on it also requires ``IS_CLOSET_SQL``.
    """
    return _escape_like(f"{memory_id}{CLOSET_SEPARATOR}") + "%"


def closet_id(parent_id: str, slug: str) -> str:
    return f"{parent_id}{CLOSET_SEPARATOR}{slug}"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "x"


def _disambiguate_slugs(speakers: list[str]) -> dict[str, str]:
    """Map each speaker to a unique closet slug.

    Two speakers whose names slugify identically (``O'Brien`` vs ``O Brien`` both
    -> ``o_brien``) would collide on one ``_closet_<slug>`` id, and the second
    ``INSERT OR REPLACE`` would silently drop the first speaker's closet. Append a
    numeric suffix to break ties. Stable across re-ingest because the speaker
    order is deterministic (first-seen order in the session).
    """
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for sp in speakers:
        base = _slug(sp)
        slug = base
        n = 2
        while slug in used:
            slug = f"{base}_{n}"
            n += 1
        used.add(slug)
        mapping[sp] = slug
    return mapping


# --- content derivation ---------------------------------------------------

# The opening of every per-speaker enrichment. Stable across the whole published
# history of the writer. NOT usable as an identity test: a closet pulled from the
# cloud by a client at or below 0.3.0 is re-enriched as an ordinary memory and
# loses it (see ``is_closet_shaped``).
CLOSET_PREAMBLE_TEMPLATE = "{speaker}'s contributions"


def _parse_turns(content: str) -> list:
    try:
        turns = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return []
    return turns if isinstance(turns, list) else []


def speaker_of(turn: object) -> str | None:
    """The speaker name on a turn, or None if the turn is not speaker-shaped.

    ``speaker`` is whatever JSON happened to be stored, so it can be a list, a
    dict, a number or absent. Anything but a non-empty string is "not a turn we
    recognise" — never an error. A raise here would abort the marker migration
    mid-transaction and leave the store unopenable, since the classification
    walks every pre-existing row.
    """
    if not isinstance(turn, dict):
        return None
    speaker = turn.get("speaker")
    return speaker if isinstance(speaker, str) and speaker else None


def speakers_of(content: str) -> list[str]:
    """Distinct speakers in first-seen order, or [] when the content is not turns."""
    speakers: list[str] = []
    seen: set[str] = set()
    for turn in _parse_turns(content):
        sp = speaker_of(turn)
        if sp is not None and sp not in seen:
            speakers.append(sp)
            seen.add(sp)
    return speakers


def _format_date(session_timestamp: str | None) -> str:
    if not session_timestamp:
        return ""
    try:
        return datetime.fromisoformat(session_timestamp).strftime("%B %d, %Y at %I:%M %p")
    except ValueError:
        return session_timestamp


def _enrich_closet_content(
    content: str, session_timestamp: str | None, target_speaker: str, other_speakers: list[str]
) -> tuple[str, str]:
    """Return (closet_raw_json, closet_enriched_text) for one speaker's turns.

    Returns ('', '') if the speaker has no turns in this session.
    """
    speaker_turns = [t for t in _parse_turns(content) if speaker_of(t) == target_speaker]
    if not speaker_turns:
        return "", ""

    date_str = _format_date(session_timestamp)

    # Preamble names both the speaker and the counterpart(s) so cross-speaker
    # context is not lost entirely — the closet is Alice's contributions, but
    # names Bob so queries mentioning Bob can still match.
    # Every branch opens with CLOSET_PREAMBLE_TEMPLATE, which the migration's
    # shape test matches on. Built from the same constant so the writer and that
    # test cannot drift apart.
    opening = CLOSET_PREAMBLE_TEMPLATE.format(speaker=target_speaker)
    if other_speakers:
        if len(other_speakers) == 1:
            partner = other_speakers[0]
        else:
            partner = ", ".join(other_speakers[:-1]) + f", and {other_speakers[-1]}"
        if date_str:
            preamble = f"{opening} to a conversation with {partner} on {date_str}."
        else:
            preamble = f"{opening} to a conversation with {partner}."
    else:
        if date_str:
            preamble = f"{opening} on {date_str}."
        else:
            preamble = f"{opening}."

    lines = [preamble, ""]
    for turn in speaker_turns:
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        if dia_id and text:
            lines.append(f"{dia_id} {target_speaker}: {text}")
        elif text:
            lines.append(f"{target_speaker}: {text}")

    return json.dumps(speaker_turns), "\n".join(lines)


def derive_closets(content: str, session_timestamp: str | None) -> list[tuple[str, str, str]]:
    """The closets bloom builds from one parent: ``(slug, raw_json, enriched_text)``.

    Empty for anything that is not a turn list with at least two speakers — a
    single-speaker session would produce a closet identical to the parent.

    Single source of truth for closet content: ``bloom.ingest`` writes exactly
    this, and the one-time migration re-derives from it to decide which
    pre-marker rows are provably closets.
    """
    speakers = speakers_of(content)
    if len(speakers) < 2:
        return []
    slug_map = _disambiguate_slugs(speakers)
    out: list[tuple[str, str, str]] = []
    for sp in speakers:
        others = [o for o in speakers if o != sp]
        raw, enriched = _enrich_closet_content(content, session_timestamp, sp, others)
        if raw:
            out.append((slug_map[sp], raw, enriched))
    return out


# --- schema introspection -------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    # Positional access (PRAGMA returns cid, name, type, ...) so this works
    # whether or not the caller set row_factory = sqlite3.Row.
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def has_marker(conn: sqlite3.Connection) -> bool:
    """Whether ``memories`` carries the closet provenance column."""
    return MARKER_COLUMN in _columns(conn, "memories")


# --- side tables: closet deletions and the migration backup ---------------

# Both DDLs live here rather than in ``poppy.ui.tombstones`` because the engines
# are what write these tables, and the marker migration runs long before any
# TombstoneStore is constructed. ``ui.tombstones`` imports them so the two never
# drift; the direction (ui -> engine) is the right way round.

# Deletions of marked closets: an id and a time, never a memory snapshot.
CLOSET_TOMBSTONE_TABLE = "closet_tombstones"
CLOSET_TOMBSTONE_DDL = f"""
CREATE TABLE IF NOT EXISTS {CLOSET_TOMBSTONE_TABLE} (
    id TEXT PRIMARY KEY,
    tombstoned_at TEXT NOT NULL,
    -- 1 once THIS device deleted the copy. That time is a floor: a deletion we
    -- merely heard about, however it is dated, may never move the record below
    -- it. See `record_closet_tombstones`.
    is_local INTEGER NOT NULL DEFAULT 0
);
"""

# Ids of copies that existed BEFORE the marker migration ran, and so may sit in
# the cloud as ordinary memories: a 0.2.4 client's ``list_all`` had no marker to
# exclude them by, so it pushed them.
#
# This table is an ANNOUNCEMENT QUEUE, not a deletion log. Its job is to tell the
# cloud, exactly once per leaked id, that the row up there is derived data that
# should not exist. That is deliberately separate from ``closet_tombstones``,
# which is a LOCAL record of when a copy was deleted here and exists to make
# pull's skip and freshness comparisons correct. The two have different clocks
# and different lifetimes, and riding one on the other lost announcements
# whenever the local record was rewritten.
#
# Only leaked ids are ever announced. A copy created after the fix is never
# pushed, so the cloud has nothing to delete, and announcing its removal would be
# dangerous rather than merely noisy: the id space is shared, so a real cloud
# memory whose id happens to equal one this device derives a copy at would be
# deleted FOR EVERY DEVICE.
#
# BEST-EFFORT BY DESIGN. An announcement carries the leaked row's own old
# timestamp (that is what makes it a safe ownership claim, see
# `closet_tombstone_to_wire`), and the tombstone the server stores keeps it — so
# a device whose pull watermark is already past that point never sees the
# deletion and keeps its own copy. Stamping applied deletions forward
# server-side was tried and reverted, because it lost client-time ordering for
# offline restores. The authoritative cloud erasure is the human-run
# server-side cleanup (trags #134, migration 033), whose tombstones are stamped
# forward; this lane converges what it can without a person in the loop.
LEGACY_CLOSET_TABLE = "legacy_closet_ids"
LEGACY_CLOSET_DDL = f"""
CREATE TABLE IF NOT EXISTS {LEGACY_CLOSET_TABLE} (
    id TEXT PRIMARY KEY,
    -- 1 until the cloud has accepted the deletion. Cleared only on a 2xx, so a
    -- failed or offline sync retries rather than dropping the announcement.
    announce_pending INTEGER NOT NULL DEFAULT 1,
    announced_at TEXT,
    -- The copy row's OWN updated_at as it stood before the migration touched it:
    -- exactly the value a 0.2.4 client put on the wire for that row. Captured
    -- here because the rebuild overwrites it with the parent's current one, and
    -- a purge removes the row entirely. It is what the announcement is stamped
    -- with, so the server's freshness gate decides whether we still own the
    -- remote row (see `closet_tombstone_to_wire`).
    legacy_updated_at TEXT
);
"""

# Pre-images of every row the one-time marker migration rebuilt or purged.
CLOSET_BACKUP_TABLE = "closet_migration_backup"
CLOSET_BACKUP_DDL = f"""
CREATE TABLE IF NOT EXISTS {CLOSET_BACKUP_TABLE} (
    id TEXT PRIMARY KEY,
    content TEXT,
    enriched_content TEXT,
    related_to TEXT,
    created_at TEXT,
    updated_at TEXT,
    action TEXT NOT NULL,
    migrated_at TEXT NOT NULL
);
"""

# SQLite refuses a statement with more than 32766 host parameters, and a store
# with tens of thousands of closets would hit that inside the migration — where
# the exception rolls back the ALTER as well, leaving the store permanently
# unopenable by either engine. Every `IN (?,?,...)` built from a row set is
# chunked below this. Module-level so tests can shrink it.
SQL_PARAM_CHUNK = 500


def chunked(items: list[str], size: int | None = None) -> list[list[str]]:
    """Split ``items`` into runs small enough for one parameterised statement."""
    limit = size or SQL_PARAM_CHUNK
    return [items[i : i + limit] for i in range(0, len(items), limit)]


def ensure_closet_side_tables(conn: sqlite3.Connection) -> None:
    """Create the closet side tables. Called once per store open, by both engines.

    Kept out of the write helpers below: they run on the ingest hot path, and a
    ``CREATE TABLE IF NOT EXISTS`` per memory written is a needless statement.
    """
    conn.execute(CLOSET_TOMBSTONE_DDL)
    conn.execute(CLOSET_BACKUP_DDL)
    conn.execute(LEGACY_CLOSET_DDL)
    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a column
    # added later needs its own idempotent step.
    if "is_local" not in _columns(conn, CLOSET_TOMBSTONE_TABLE):
        conn.execute(f"ALTER TABLE {CLOSET_TOMBSTONE_TABLE} ADD COLUMN is_local INTEGER NOT NULL DEFAULT 0")


def record_legacy_closet_ids(conn: sqlite3.Connection, entries: list[tuple[str, str | None]]) -> None:
    """Queue leaked copy ids for a one-off cloud announcement.

    Called by the one-time migration for every id it marks or purges: those are
    exactly the copies that existed before the marker and could therefore have
    been pushed by an older client. Queued immediately rather than on the
    parent's next write — a leaked copy is stale text in the cloud from the
    moment this store knows about it, and waiting for a redaction that may never
    come leaves it there. The set is bounded by what the store held at migration
    and each id is sent once.

    Each entry carries the row's stored ``updated_at`` as it was BEFORE the
    migration rewrote or removed it. That is the value a 0.2.4 client put on the
    wire for the row, and stamping the announcement with it is what lets the
    server decide whether the remote row is still the one we are talking about.
    """
    if not entries:
        return
    conn.executemany(
        f"INSERT OR IGNORE INTO {LEGACY_CLOSET_TABLE} (id, announce_pending, legacy_updated_at) VALUES (?, 1, ?)",
        [(mid, utc_iso(stamp)) for mid, stamp in entries],
    )


def utc_iso(value: str | datetime | None) -> str | None:
    """One spelling per instant, so the SQL ``MAX``/``MIN`` on these columns compare time.

    The claims and deletion records below are compared as TEXT inside SQLite.
    ``2026-07-01T12:00:00+02:00`` and ``2026-07-01T11:00:00+00:00`` sort the
    wrong way round as strings, and an announcement that keeps the older instant
    is answered ``stale_ignored`` and consumed, leaving the leaked text live with
    nothing pending. Everything stored here is therefore normalised to UTC; a
    string that does not parse is stored as it came.

    NEVER RAISES. This runs on the ingest hot path and inside the one-time
    migrations, where an exception is not a rejected value but a store that
    cannot be opened: the migration rolls back and re-raises, so every later open
    fails the same way and the CLI is dead. Two inputs get that treatment and are
    passed through verbatim instead:

      * a string that is not a timestamp at all (``ValueError``);
      * a timestamp at the edge of the representable range whose UTC form is
        outside it (``OverflowError``) — ``9999-12-31T23:59:59-05:00`` is year
        10000 in UTC, and ``0001-01-01T00:00:00+02:00`` is year 0. Those values
        reach the store: ``fromisoformat`` accepts them, so an importer or an
        ``--expires-at`` before this fix could store one. Rejecting such a value
        is the WRITE BOUNDARY's job (``lifecycle.parse_expires_at``); here the
        only safe answer is to leave the row exactly as it is.
    """
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


def later_stamp(a: str | None, b: str | None) -> str | None:
    """The later of two stored stamps, compared as INSTANTS and returned as stored.

    A claim has to be at least as new as the cloud row it names, or the server's
    freshness gate answers ``stale_ignored`` and the announcement is consumed with the
    text still up there. A Trash entry carries two candidates — when the row was last
    written and when it was deleted — and either can be the later one, because a 0.2.4
    client wrote both. Compared as TEXT they sort by offset rather than by time, which
    is this file's oldest recurring bug; an unparseable value simply loses.
    """
    if a is None:
        return b
    if b is None:
        return a
    try:
        left, right = datetime.fromisoformat(a), datetime.fromisoformat(b)
    except (ValueError, OverflowError, TypeError):
        return a
    if left.tzinfo is None:
        left = left.replace(tzinfo=timezone.utc)
    if right.tzinfo is None:
        right = right.replace(tzinfo=timezone.utc)
    return a if left >= right else b


def rearm_legacy_announcement(conn: sqlite3.Connection, memory_id: str, updated_at: str) -> None:
    """Queue, re-arm and advance the claim for a leaked id. TIER A EVIDENCE ONLY.

    Called when a live cloud row is seen that IS our derivation of that id, which
    proves two things at once: the cloud still holds the leak, and this is the
    timestamp it currently carries.

    Three effects, all needed:

      * queue it, if the migration never saw this id (a store created after the
        fix pulling from an account an older client synced);
      * RE-ARM it, if we already announced successfully and an old client has
        republished the copy since. Without this the row stays
        ``announce_pending = 0`` for ever and the text lives on in the cloud;
      * ADVANCE the claim to the later timestamp, because an announcement stamped
        with an older sighting loses the server's freshness comparison, is
        answered ``stale_ignored``, and clears the queue with the text still up
        there.

    Only Tier A may do any of this. The timestamp is an ownership claim, and one
    taken from a row we have not PROVEN is ours can tie with — and therefore
    overwrite — a real memory another device wrote at that id.
    """
    conn.execute(
        f"INSERT INTO {LEGACY_CLOSET_TABLE} (id, announce_pending, announced_at, legacy_updated_at) "
        "VALUES (?, 1, NULL, ?) ON CONFLICT(id) DO UPDATE SET "
        "announce_pending = 1, announced_at = NULL, legacy_updated_at = "
        f"MAX(COALESCE({LEGACY_CLOSET_TABLE}.legacy_updated_at, excluded.legacy_updated_at), "
        "excluded.legacy_updated_at)",
        (memory_id, utc_iso(updated_at)),
    )


def pending_legacy_announcements(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    """(id, the row's pre-migration updated_at) the cloud has not confirmed deleting."""
    if not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return []
    rows = conn.execute(
        f"SELECT id, legacy_updated_at FROM {LEGACY_CLOSET_TABLE} WHERE announce_pending = 1 ORDER BY id"
    )
    return [(row[0], row[1]) for row in rows]


def announced_copy_claim(conn: sqlite3.Connection, memory_id: str) -> str | None:
    """The claim stamp queued for an id this store PROVED was a leaked copy, or None.

    Survives the local deletion record. That record is what makes pull skip the
    cloud's stale copy, and it ages out seven days after the deletion once the
    announcement has landed — while the cloud row itself only disappears when the
    server applies that delete. A pull holding a row captured before it waited for
    the write gate can therefore find the record already purged by a concurrent
    push and ingest the copy as an ordinary memory.

    The queue entry outlives the record and says the same thing more narrowly: THIS
    id was proven to be a per-speaker copy, and this is the timestamp the cloud row
    carried. Announced or still pending — the row stays either way, and only
    reclaiming the id for a real memory removes it, which is exactly when the
    suppression must stop.
    """
    if not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return None
    row = conn.execute(f"SELECT legacy_updated_at FROM {LEGACY_CLOSET_TABLE} WHERE id = ?", (memory_id,)).fetchone()
    return row[0] if row is not None else None


def mark_legacy_announced(
    conn: sqlite3.Connection,
    memory_ids: list[str] | list[tuple[str, str | None]],
    *,
    when: datetime | None = None,
) -> None:
    """Record that the cloud ACCEPTED the deletion of these leaked ids.

    Only ever called after a 2xx: an announcement that failed, or never left an
    offline device, stays pending so the next sync retries it.
    """
    if not memory_ids or not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return
    stamp = (when or datetime.now(timezone.utc)).isoformat()
    for entry in memory_ids:
        if isinstance(entry, tuple):
            # Acknowledge the exact claim that was sent. Between the snapshot
            # push took and the server's answer, a pull can see the copy
            # republished with a newer timestamp and re-arm the announcement
            # with it; the old claim is answered stale_ignored, and clearing by
            # id alone would consume the new one with the text still up there.
            memory_id, sent = entry
            conn.execute(
                f"UPDATE {LEGACY_CLOSET_TABLE} SET announce_pending = 0, announced_at = ? "
                "WHERE id = ? AND COALESCE(legacy_updated_at, '') = COALESCE(?, '')",
                (stamp, memory_id, utc_iso(sent)),
            )
        else:
            conn.execute(
                f"UPDATE {LEGACY_CLOSET_TABLE} SET announce_pending = 0, announced_at = ? WHERE id = ?",
                (stamp, entry),
            )


def record_closet_tombstones(
    conn: sqlite3.Connection,
    memory_ids: list[str],
    *,
    when: datetime | None = None,
    applying_remote_deletion: bool = False,
    authoritative: bool = False,
) -> None:
    """Record LOCALLY that marked copies were removed. Ids and a time, nothing else.

    This is not what tells the cloud. It exists so pull can skip an incoming row
    for an id this device has just deleted as derived data, and so the freshness
    comparison against a later recreation of that id is right. Announcing a
    LEAKED id to the cloud is a separate lane with a separate clock — see
    ``LEGACY_CLOSET_TABLE`` — because the two were conflated and every rewrite of
    this record silently dropped the announcement.

    ``when`` is WHEN THE DELETION HAPPENED, not when this device heard about it.
    For a deletion pulled from another device that is the incoming row's
    timestamp: stamping receipt time instead makes the record look newer than a
    recreation of the id that actually came after it, so the freshness rule in
    sync would suppress the recreation.

    Idempotent, and the two kinds of write settle conflicts differently.

    A deletion HEARD ABOUT keeps the EARLIEST sighting. The same one reaches
    here by more than one route — a cascade from the parent, then the pulled row
    itself — and letting a later write push the record forward would make
    re-pulling it suppress ever-newer recreations of the id.

    ``applying_remote_deletion`` is the cascade from a pulled parent tombstone:
    the engine has just stamped those copies as local, because clearing a
    parent's copies is the same code whoever asked, and this corrects the record
    to the deletion's own clock.

    A deletion MADE HERE is a floor, and nothing lowers it. Without that, a
    forget at 14:00 followed by a pull of an older cloud deletion at 12:00 left
    the record saying 12:00: push then sent the downgraded deletion, it lost to
    another device's 13:00 recreation, and the next pull brought the forgotten
    text back. "Earliest wins" is about not refreshing a record; it was never
    meant to let a stranger's older event overwrite our own.

    ``authoritative`` is the exception both of those rules need, and it RAISES:
    the record becomes ``max(existing, incoming)``, over a local floor as well.
    It is for a deletion that IS its own event rather than another sighting of
    one — a server-side cleanup tombstone for a copy id (trags migration 033),
    which is stamped forward and speaks for the account. Without it a cleanup at
    t3 left a record still saying t1, and after a pull-watermark reset the cloud's
    t2 copy of that id beat the record and was ingested as an ordinary memory.
    Raising is safe in the direction that matters: a record can only ever suppress
    an OLDER incoming row, so a later stamp suppresses more, never less. Two
    sightings of the same deletion carry the same stamp, so nothing moves.
    """
    if not memory_ids:
        return
    stamp = utc_iso(when or datetime.now(timezone.utc))
    if authoritative and when is not None:
        # Capped at NOW. The deletion this raises from is a placeholder another client
        # announced, and its timestamp is whatever that client wrote: an arbitrary
        # future stamp would otherwise push the record — including a local floor —
        # years ahead and suppress every honest row at that id until it arrived. A
        # cleanup we cannot date is still a cleanup; dating it now is the most this
        # device can verify.
        capped = when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)
        stamp = utc_iso(min(capped, datetime.now(timezone.utc)))
        conn.executemany(
            f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) "
            f"ON CONFLICT(id) DO UPDATE SET tombstoned_at = "
            f"MAX({CLOSET_TOMBSTONE_TABLE}.tombstoned_at, excluded.tombstoned_at)",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    if applying_remote_deletion:
        # The engine stamped these moments ago as local, because deleting a
        # parent's copies is the same operation whoever asked for it. This says
        # the ask came from another device, so the record takes that deletion's
        # clock and stops being a floor.
        conn.executemany(
            f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) "
            f"ON CONFLICT(id) DO UPDATE SET tombstoned_at = excluded.tombstoned_at, is_local = 0",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    if when is None:
        conn.executemany(
            f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 1) "
            f"ON CONFLICT(id) DO UPDATE SET is_local = 1, tombstoned_at = "
            f"MAX({CLOSET_TOMBSTONE_TABLE}.tombstoned_at, excluded.tombstoned_at)",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    conn.executemany(
        f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) "
        f"ON CONFLICT(id) DO UPDATE SET tombstoned_at = CASE "
        f"WHEN {CLOSET_TOMBSTONE_TABLE}.is_local = 1 THEN {CLOSET_TOMBSTONE_TABLE}.tombstoned_at "
        f"ELSE MIN({CLOSET_TOMBSTONE_TABLE}.tombstoned_at, excluded.tombstoned_at) END",
        [(mid, stamp) for mid in memory_ids],
    )


def clear_closet_tombstones(conn: sqlite3.Connection, memory_ids: list[str]) -> None:
    """Release ids that have been RECLAIMED by a real memory.

    A copy's id can legitimately become an ordinary memory's: the copy is
    forgotten, and a later write puts an independent note at that id. From then
    on it is a real memory's id, and BOTH obligations attached to it have to go.

      * the local deletion record, or pull keeps skipping incoming updates and
        the new memory silently stops syncing;
      * any pending cloud announcement, or push sends the new note live and then
        a NEWER deletion for the same id — and the next pull removes the note
        the user just wrote.

    Called from every non-closet write of a row, on both engines, which is the
    one point where an id stops being derived data.
    """
    if not memory_ids:
        return
    ids = list(memory_ids)
    clear_closet_deletion_records(conn, ids)
    if not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return
    for batch in chunked(ids):
        placeholders = ",".join("?" * len(batch))
        conn.execute(f"DELETE FROM {LEGACY_CLOSET_TABLE} WHERE id IN ({placeholders})", batch)


def clear_closet_deletion_records(conn: sqlite3.Connection, memory_ids: list[str]) -> None:
    """Drop the LOCAL deletion records for these ids, leaving any announcement alone.

    Used when a copy comes back as a copy — a restore regenerates it, so the id
    is live derived data again and the old "deleted at T" record is a lie. Left
    in place it would win the freshness comparison against a cloud row written
    AFTER T, and a leaked copy stamped in between would be ingested live.
    Pull's skip does not need it: a live copy is caught by its marker.

    Deliberately does NOT touch the announcement queue. A restore does not
    un-leak the copy an older client pushed, so the cloud still owes that delete.
    """
    if not memory_ids or not _table_exists(conn, CLOSET_TOMBSTONE_TABLE):
        return
    for batch in chunked(list(memory_ids)):
        placeholders = ",".join("?" * len(batch))
        conn.execute(f"DELETE FROM {CLOSET_TOMBSTONE_TABLE} WHERE id IN ({placeholders})", batch)


def back_up_migrated_rows(conn: sqlite3.Connection, ids: list[str], action: str) -> None:
    """Snapshot rows the one-time marker migration is about to rewrite or delete.

    The migration's classification is conjunctive and deliberate, but it is still
    inference over data that carries no marker, so a row it misreads would
    otherwise be destroyed with no way back. This makes that case recoverable for
    the same seven days a deletion is: the pre-image is copied here first, and
    ``TombstoneStore.purge_expired`` ages it out on the tombstone clock.

    Local only. Never synced, never listed, never retrieved — nothing reads this
    table but a human with the recovery query:

        sqlite3 ~/.poppy/memories.db \
          "SELECT id, action, content FROM closet_migration_backup;"

    ``action`` is ``rebuilt`` or ``purged``.
    """
    if not ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    has_enriched = "enriched_content" in _columns(conn, "memories")
    enriched_col = "enriched_content" if has_enriched else "NULL"
    for batch in chunked(ids):
        placeholders = ",".join("?" * len(batch))
        conn.execute(
            f"INSERT OR REPLACE INTO {CLOSET_BACKUP_TABLE}"
            " (id, content, enriched_content, related_to, created_at, updated_at, action, migrated_at)"
            f" SELECT id, content, {enriched_col}, related_to, created_at, updated_at, ?, ?"
            f" FROM memories WHERE id IN ({placeholders})",
            [action, now, *batch],
        )


# --- runtime helpers ------------------------------------------------------


def is_marked_closet_row(conn: sqlite3.Connection, memory_id: str) -> bool:
    """Whether the stored row for ``memory_id`` is itself a marked closet."""
    return conn.execute(f"SELECT 1 FROM memories WHERE id = ? AND {IS_CLOSET_SQL}", (memory_id,)).fetchone() is not None


def is_marked_closet(engine: object, memory_id: str) -> bool:
    """Whether an engine reports ``memory_id`` as one of its own derived copies.

    Duck-typed rather than added to the ``RetrievalEngine`` ABC, which is a
    deliberately stable boundary. An engine with no closet expansion answers
    False, which is correct for it.

    No error tolerance: if an engine raises here the caller must fail loudly
    rather than quietly treat a derived copy as an ordinary memory.
    """
    check = getattr(engine, "is_closet_row", None)
    if check is None:
        return False
    return bool(check(memory_id))


def note_leaked_cloud_copy(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    content: str,
    related_to: list[str],
    created_at: str,
    updated_at: str,
) -> bool:
    """React to a live cloud row at an id this device already derives a copy at.

    Reached when pull skips such a row. Three cases, and the line between them is
    what may touch the announcement's TIMESTAMP — because that timestamp is an
    ownership claim, and a claim taken from a row we have not proven is ours can
    tie with, and therefore overwrite, a real memory another device wrote there.

    PROVEN (shape checks out AND the text is our derivation) — the cloud is
        holding our leak, pushed by a client at or below 0.2.4. Queue it, re-arm
        it if we already announced and an old client has republished since, and
        advance the claim to the newer timestamp. This is the ONLY case that may
        write one.

    LIKELY (shape checks out, text differs) — it descends from our parent, but we
        have not proven the cloud row is the leak rather than something another
        device wrote and then edited at that id. Touch nothing, and that is the
        right answer either way: if it really is our untouched leak, the queued
        timestamp still matches it and the delete lands on the tie; if it is
        someone else's edited row, its timestamp has moved past ours, our
        announcement is answered ``stale_ignored``, and their row survives.

    RECLAIM (fails our parent's provenance test) — the id belongs to a real
        memory now. Cancel any queued announcement; sending it would tie with
        that row and delete it everywhere. The local copy is untouched — this is
        about the cloud, not about us.

    Returns whether an announcement is queued for the id afterwards.
    """
    tier, _parent, _raw, _enriched, _meta = classify_closet_row(
        conn, memory_id, content=content, related_raw=json.dumps(related_to), created_at=created_at
    )

    if tier == TIER_PROVEN:
        rearm_legacy_announcement(conn, memory_id, updated_at)
        return True

    if tier == TIER_LIKELY:
        # Shape says it descends from our parent, but the text is not our
        # derivation, so we have NOT proven the cloud row is the leak rather than
        # something another device wrote and edited at that id. Touch nothing.
        #
        # Leaving an existing announcement alone is the right answer either way.
        # If the cloud row really is our untouched leak, its timestamp still
        # matches what we queued and the delete lands on the tie. If it is
        # someone else's edited row, its timestamp has moved past ours, our
        # announcement is answered `stale_ignored`, and their row survives —
        # which is exactly what should happen. Copying their timestamp into our
        # claim would have made it tie, and replaced their text with the
        # placeholder.
        return _table_exists(conn, LEGACY_CLOSET_TABLE) and bool(
            conn.execute(
                f"SELECT 1 FROM {LEGACY_CLOSET_TABLE} WHERE id = ? AND announce_pending = 1", (memory_id,)
            ).fetchone()
        )

    # Fails our parent's provenance test: the id belongs to a real memory now.
    # Cancel any queued announcement — sending it would be stamped with that
    # row's own timestamp, tie, and delete it everywhere. The local copy is left
    # exactly as it is; this is about the cloud, not about us.
    if _table_exists(conn, LEGACY_CLOSET_TABLE):
        conn.execute(f"DELETE FROM {LEGACY_CLOSET_TABLE} WHERE id = ?", (memory_id,))
    return False


def marked_closet_ids(conn: sqlite3.Connection, parent_id: str, *, derived_only: bool = False) -> list[str]:
    """Ids of the closets bloom marked as derived from ``parent_id``.

    Two conditions, both required: the id sits under the escaped
    ``<parent_id>_closet_`` prefix, AND the row carries the provenance marker. A
    real memory is never marked, so it can never be returned here however its id
    is spelled.
    """
    predicate = IS_DERIVED_CLOSET_SQL if derived_only else IS_CLOSET_SQL
    rows = conn.execute(
        f"SELECT id, related_to FROM memories WHERE id LIKE ? ESCAPE '\\' AND {predicate} ORDER BY id",
        (_closet_like_pattern(parent_id),),
    ).fetchall()
    return [row[0] for row in rows if _owned_by(parent_id, row[1])]


def adopted_closet_ids(conn: sqlite3.Connection, parent_id: str) -> list[str]:
    """Ids under ``parent_id`` that were adopted on inference rather than derived."""
    rows = conn.execute(
        f"SELECT id, related_to FROM memories WHERE id LIKE ? ESCAPE '\\' AND {MARKER_COLUMN} = ? ORDER BY id",
        (_closet_like_pattern(parent_id), CLOSET_ADOPTED_UNVERIFIED),
    ).fetchall()
    return [row[0] for row in rows if _owned_by(parent_id, row[1])]


def _owned_by(parent_id: str, related_raw: str | None) -> bool:
    """Whether a marked row's back-reference names exactly ``parent_id``.

    The prefix query is a coarse net: ``<parent>_closet_%`` also catches the
    copies of ANOTHER parent whose own id begins with that prefix (``p`` and
    ``p_closet_notes``, say). A parent's redaction must take only its own
    copies, so the back-reference every copy carries decides ownership.
    """
    try:
        return json.loads(related_raw or "[]") == [parent_id]
    except (TypeError, ValueError):
        return False


def _clear_unmarked_copies(
    conn: sqlite3.Connection,
    parent_id: str,
    *,
    has_embeddings: bool,
    tombstone: bool,
    already: set[str],
    event_ts: datetime | None,
) -> list[str]:
    """Grade UNMARKED rows at this parent's derived ids and clear the ones that are copies.

    The parent row must still be present: its text is what the grading compares
    against. Callers that delete the parent do this first.

    PROVEN — its text is what the parent derives, so the cloud holding it is our
        leak. Deleted, recorded, and queued for announcement with the timestamp
        the row carries.
    LIKELY — the shape of a copy, but text the parent does not derive. LEFT
        ALONE. The action has to be bounded by the evidence, and on the only
        store that reaches here with such a row unmarked (one that never ran
        bloom, so nothing adopted it at ingest) the row is as likely a memory
        the user wrote at that id as a copy of a parent since edited. Deleting
        it was the seed-only exemption the migration makes, undone at every
        forget. ``poppy doctor`` counts it for explicit repair.
    otherwise — a real memory that happens to hold the id. Untouched.
    """
    parent = conn.execute("SELECT content, source_timestamp FROM memories WHERE id = ?", (parent_id,)).fetchone()
    if parent is None:
        return []

    cleared: list[str] = []
    announce: list[tuple[str, str | None]] = []
    for slug, _raw, _enriched in derive_closets(parent[0], parent[1]):
        cid = closet_id(parent_id, slug)
        if cid in already:
            continue
        row = conn.execute(
            f"SELECT content, related_to, created_at, updated_at FROM memories WHERE id = ? AND {NOT_CLOSET_SQL}",
            (cid,),
        ).fetchone()
        if row is None:
            continue
        try:
            tier, _p, _r, _e, _m = classify_closet_row(conn, cid, content=row[0], related_raw=row[1], created_at=row[2])
        except Exception:
            continue
        if tier != TIER_PROVEN:
            continue
        cleared.append(cid)
        announce.append((cid, row[3]))

    if not cleared:
        return []
    back_up_migrated_rows(conn, cleared, "cleared")
    for batch in chunked(cleared):
        placeholders = ",".join("?" * len(batch))
        if has_embeddings:
            conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
    # ALWAYS, whatever `tombstone` says. That flag decides whether the LOCAL
    # deletion record is written — a question about this device's Trash and
    # pull-skip. Whether the cloud is still holding our leaked text is a
    # different question with a different answer, and gating it here meant a
    # content edit under the default engine cleared a proven copy locally and
    # left the cloud's copy of the secret alive indefinitely.
    record_legacy_closet_ids(conn, announce)
    return cleared


def delete_marked_closets(
    conn: sqlite3.Connection,
    parent_id: str,
    *,
    has_embeddings: bool,
    tombstone: bool = True,
    derived_only: bool = False,
    event_ts: datetime | None = None,
) -> list[str]:
    """Delete ``parent_id``'s marked closets — rows and vectors. Returns their ids.

    Closets are derived data: bloom regenerates them from the parent on its next
    ingest, they are excluded from ``list_all`` so they never sync as memories,
    and their text is a copy of the parent's. Clearing them on a redaction is
    therefore lossless locally and is what makes a forget or a content edit
    actually remove the per-speaker copies.

    Recording the content-free tombstones happens HERE rather than at each call
    site. This is the one place a marked closet row is removed on behalf of its
    parent — by a forget, a delete, a content edit, a supersede or a pulled
    deletion — so doing it here is what makes it impossible for a new redaction
    path to be added and silently leave a cloud copy live.

    ``tombstone=False`` is for TTL expiry, which is not a redaction: the cloud
    row carries the same ``expires_at`` and ages out on its own, so announcing
    the deletion would be noise.

    ``event_ts`` is when the operation that caused this HAPPENED, for one driven
    by a pulled row. Every deletion record written during the call takes it, for
    graded-unmarked ids as well as marked ones, and stops being a local floor.
    Left None by a local operation, which gets this device's clock. Passing it in
    is what makes the timestamps right in ONE place, rather than sync correcting
    afterwards the subset of ids it happened to enumerate.

    ``derived_only=True`` is for a METADATA-ONLY parent write, where this engine
    regenerates its own copies from a body that has not changed. An
    adopted-unverified row is not ours to rewrite, and nothing about the memory's
    text has moved, so it is left alone.

    Every other caller passes False, and that includes a CONTENT-changing edit as
    well as forget, supersede and delete. Redaction beats preservation: the text
    of the memory has changed or gone, so no copy of the old text may survive,
    whatever grade adopted it. Adopted rows are snapshotted first — they might be
    a curated split rather than a copy, and the snapshot is the only way back.
    """
    if not conn.in_transaction:
        # Claim the write lock BEFORE enumerating. SQLite only takes it at the
        # first DELETE below, so between reading the ids and removing them
        # another process could restore a real memory at one of them, and the
        # DELETE by id would then take that memory with it. Under BEGIN
        # IMMEDIATE the other writer waits on the busy timeout instead. The
        # caller's own commit ends the transaction, as it does for the DELETEs.
        # A connection shared by threads without a lock can open one between
        # the check and this statement; that transaction protects the sequence
        # just as well, so THAT refusal is not an error. Any other failure,
        # above all "database is locked" after the busy timeout, must abort:
        # enumerating without the lock is exactly the window this claim exists
        # to close, and a redaction that went on regardless has deleted a
        # memory another writer committed at one of these ids in the meantime.
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "within a transaction" not in str(exc):
                raise
    ids = marked_closet_ids(conn, parent_id, derived_only=derived_only)
    if not derived_only:
        # Copies that were never marked are still copies. A store that only ever
        # ran the fallback engine never synthesises, so it never adopts one at
        # ingest, and pull's notice only looks at ids already marked — so a
        # legacy copy pulled from an account an older client synced sat there
        # unmarked, survived its parent's redaction, stayed recallable, and was
        # pushed live with the text. Graded HERE the redaction path needs no
        # help from either.
        #
        # Same grading as everywhere else, so a real memory that merely holds a
        # derived id is left exactly where it is.
        ids = ids + _clear_unmarked_copies(
            conn,
            parent_id,
            has_embeddings=has_embeddings,
            tombstone=tombstone,
            already=set(ids),
            event_ts=event_ts,
        )
    # Trash entries holding the text being redacted go on every redaction,
    # whether or not a live copy still exists (an older client may have sent
    # both copies to Trash, leaving snapshots and no rows) and whatever the
    # caller decided about local deletion records (a content edit passes
    # ``tombstone=False`` because it rewrites most ids; the snapshot of the OLD
    # text is not rewritten by anything).
    # The text of each copy row this call is clearing, read while the rows are
    # still here. A snapshot holding exactly that text is a snapshot of the text
    # being redacted even when the parent's own text has drifted past it, which
    # is the one case the derivation comparison cannot see.
    snapshots = (
        []
        if derived_only
        else _clear_copy_snapshots(conn, parent_id, already=set(ids), row_texts=_row_texts(conn, ids))
    )
    if not ids and not snapshots:
        return []
    if ids and not derived_only:
        # Only the inferred ones: a copy this engine derived is reproducible from
        # its parent, so there is nothing to preserve.
        back_up_migrated_rows(conn, adopted_closet_ids(conn, parent_id), "cleared")
    for batch in chunked(ids):
        placeholders = ",".join("?" * len(batch))
        if has_embeddings:
            conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
    ids = ids + snapshots
    if tombstone:
        record_closet_tombstones(conn, ids, when=event_ts, applying_remote_deletion=event_ts is not None)
    return ids


def claim_proven_unmarked_copy(conn: sqlite3.Connection, memory_id: str) -> bool:
    """Whether an UNMARKED live row is, on full proof, a copy — and if so, claim it.

    A store that never ran bloom never adopts a pulled copy, so a leaked copy
    sits there unmarked and the user can ask to forget it by id. Treating it as
    an ordinary memory would snapshot its speaker text into Trash and push that
    as the body of a tombstone, which another device then files as restorable.
    PROVEN here means the full provenance test AND text equal to what the live
    parent derives, the same bar as everywhere else; on that proof the cloud
    row is announced for deletion with the row's own timestamp. Anything less
    is left to the ordinary path: a real memory's Trash entry is its way back.
    """
    stamp = _proven_unmarked_copy_stamp(conn, memory_id)
    if stamp is None:
        return False
    rearm_legacy_announcement(conn, memory_id, stamp)
    return True


def is_proven_unmarked_copy(conn: sqlite3.Connection, memory_id: str) -> bool:
    """Whether an UNMARKED live row is, on full proof, a copy of its live parent. Pure check."""
    return _proven_unmarked_copy_stamp(conn, memory_id) is not None


def _proven_unmarked_copy_stamp(conn: sqlite3.Connection, memory_id: str) -> str | None:
    """The row's ``updated_at`` if it is an unmarked PROVEN copy, else None."""
    row = conn.execute(
        f"SELECT content, related_to, created_at, updated_at FROM memories WHERE id = ? AND {NOT_CLOSET_SQL}",
        (memory_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        tier, _p, _r, _e, _m = classify_closet_row(
            conn, memory_id, content=row[0], related_raw=row[1], created_at=row[2]
        )
    except Exception:
        return None
    return row[3] if tier == TIER_PROVEN else None


def clear_copy_snapshot(conn: sqlite3.Connection, copy_id: str) -> bool:
    """Drop the Trash entry at a marked copy's own id when it holds the parent's text.

    Forgetting a copy directly records a content-free deletion and snapshots
    nothing — but an older client may already have left a snapshot of that copy
    in Trash, and deleting the live copy while that stays would leave the text
    restorable and pushable. Same bar and same claim as the parent's redaction.
    The copy row must still be present (its back-reference names the parent).
    The row may be marked or an unmarked PROVEN copy: the caller has already
    decided it is a copy, and the snapshot itself is held to the full
    provenance test below, so the row's marker adds nothing here.

    The row's OWN text counts as well as the parent's current derivation: an
    adopted copy is deliberately never rewritten, so once the parent's text has
    drifted the snapshot of that copy matches the row we are deleting and
    nothing else.
    """
    row = conn.execute("SELECT related_to, content FROM memories WHERE id = ?", (copy_id,)).fetchone()
    if row is None:
        return False
    try:
        parents = json.loads(row[0] or "[]")
    except (TypeError, ValueError):
        return False
    if len(parents) != 1:
        return False
    return copy_id in _clear_copy_snapshots(conn, parents[0], already=set(), only=copy_id, row_texts={copy_id: row[1]})


def grade_copy_snapshot(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    content: str,
    related_to: list[str],
    created_at: str,
) -> str | None:
    """Grade a Trash SNAPSHOT the way a live row is graded: a tier, or None.

    A ``ui_tombstones`` entry is a memory's only way back, so what is filed there
    has to be judged by the same evidence as everything else. Two routes put a
    per-speaker copy's text in front of the user as restorable:

      * a content-carrying cloud tombstone for a copy id, which only an old
        (0.2.4) client deleting a copy row by hand produces. Pull filed it with
        no local row to compare against, and after the parent was forgotten
        ``restore`` brought the speaker text back as an ordinary memory and push
        sent it live;
      * a snapshot an older build wrote locally, still sitting there when the
        upgrade arrives.

    Returns ``TIER_PROVEN`` (text IS the parent's derivation, so the row is the copy
    and the caller may drop the entry and claim its cloud row), ``TIER_LIKELY``
    (provenance holds, text differs — marked or counted, never destroyed, and a Trash
    entry at that grade still restores), or None for anything else, including an
    orphan whose parent is gone. Grading is worth doing while the parent is still
    HERE: once it is forgotten there is nothing left to prove anything against. Used
    on an incoming soft-delete's snapshot and on an incoming LIVE row alike — the
    question and the evidence are the same.
    """
    try:
        tier, _p, _raw, _enr, _meta = classify_closet_row(
            conn, memory_id, content=content, related_raw=json.dumps(list(related_to)), created_at=created_at
        )
    except Exception:
        return None
    return tier if tier in (TIER_PROVEN, TIER_LIKELY) else None


def refuse_restorable_copy_snapshot(conn: sqlite3.Connection, memory_id: str) -> str | None:
    """Grade the STORED Trash entry at ``memory_id``; on full proof drop it.

    ``restore``'s own guard reads the LIVE row's marker, so an entry left by a
    0.2.4 client that forgot a copy while keeping its parent passed straight
    through: there is no live row to be marked, and the restore created an
    unmarked memory holding the speaker text that the next push uploaded live.
    The snapshot's own provenance is what decides here instead.

    PROVEN ONLY. The entry is removed and its cloud row claimed with the entry's own
    deletion time — the same action, and the same claim, as the parent's redaction —
    and the parent id is returned so the caller can refuse. A LIKELY entry RESTORES:
    it is closet-shaped with the parent's creation instant but the text is NOT the
    parent's derivation, and on the store that reaches here with such an entry (one
    that never ran bloom, so nothing adopted it) that is as likely a memory the user
    wrote as a stale copy. ``refuse_if_derived_copy`` and ``_clear_unmarked_copies``
    draw the line in the same place: anything short of proof is the user's. The
    restored row keeps the snapshot's ``related_to``, so a later redaction of the
    parent still grades and reaches it.

    Grading never raises here, as it never does on any other path: a malformed legacy
    entry is not a copy, and a restore must not become a traceback.
    """
    if not _table_exists(conn, "ui_tombstones"):
        return None
    row = conn.execute(
        "SELECT content, related_to, created_at, tombstoned_at, updated_at FROM ui_tombstones WHERE id = ?",
        (memory_id,),
    ).fetchone()
    if row is None:
        return None
    try:
        related = json.loads(row[1] or "[]")
        tier, parent_id, _raw, _enr, _meta = classify_closet_row(
            conn, memory_id, content=row[0], related_raw=row[1], created_at=row[2]
        )
    except Exception:
        return None
    if tier != TIER_PROVEN:
        return None
    conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (memory_id,))
    rearm_legacy_announcement(conn, memory_id, later_stamp(row[3], row[4]))  # type: ignore[arg-type]
    return parent_id or (related[0] if related else None)


def _row_texts(conn: sqlite3.Connection, memory_ids: list[str]) -> dict[str, str]:
    """The stored text of each live row named, for rows that are still present."""
    out: dict[str, str] = {}
    for batch in chunked(list(memory_ids)):
        placeholders = ",".join("?" * len(batch))
        for row in conn.execute(f"SELECT id, content FROM memories WHERE id IN ({placeholders})", batch):
            out[row[0]] = row[1]
    return out


def _clear_copy_snapshots(
    conn: sqlite3.Connection,
    parent_id: str,
    *,
    already: set[str],
    only: str | None = None,
    row_texts: dict[str, str] | None = None,
) -> list[str]:
    """Drop Trash entries that hold a copy of the text this redaction removes.

    An older client listed the per-speaker copies as memories, so a user could
    send one to Trash; the copy was then re-derived and the snapshot stayed. A
    parent's forget cleared the live copy and left that snapshot: restorable
    (the text came back as an ordinary, syncable memory once the live marker
    was gone), and pushed as a tombstone carrying the speaker text, whose newer
    timestamp then beat the content-free announcement.

    Only a snapshot that passes the same PROVEN bar as everywhere else goes:
    the full provenance test (back-reference to this parent, one speaker whose
    slug is the id's, the parent's creation instant) AND text that IS the
    parent's derivation. Matching text alone is not enough — an independent
    memory can hold the same projected turns — and its Trash entry is the only
    way back for it. The snapshot's deletion time becomes the claim the
    announcement is stamped with, so the cloud's copy of that snapshot is the
    one the deletion reaches. The parent row must still be present.

    ``row_texts`` maps a copy id this redaction is CLEARING to that row's own
    text, and it counts as the same proof as the derivation. An adopted copy is
    deliberately never rewritten, so once the parent's text has drifted its
    snapshot matches neither the parent's current derivation nor anything else —
    and the row beside it has already been positively identified as a copy of
    this parent, which is what makes the snapshot a snapshot of redacted text
    rather than a memory of the user's. The provenance test still
    has to pass, so a real memory's Trash entry is as safe as before.
    """
    if not _table_exists(conn, "ui_tombstones"):
        return []
    parent = conn.execute(
        "SELECT content, source_timestamp, created_at FROM memories WHERE id = ?", (parent_id,)
    ).fetchone()
    if parent is None:
        return []
    texts = dict(row_texts or {})
    candidates = [
        (closet_id(parent_id, slug), slug, raw) for slug, raw, _enriched in derive_closets(parent[0], parent[1])
    ]
    # A parent edited down past a speaker no longer derives that slug, so the
    # loop above would never reach a copy row still sitting at it. The ids being
    # cleared are known by the caller, and their prefix is this parent's, so the
    # slug is the rest of the id.
    derived_ids = {cid for cid, _slug, _raw in candidates}
    prefix = parent_id + CLOSET_SEPARATOR
    for cid in sorted(texts):
        if cid in derived_ids or not cid.startswith(prefix):
            continue
        candidates.append((cid, cid[len(prefix) :], None))
    cleared: list[str] = []
    for cid, slug, raw in candidates:
        if only is not None and cid != only:
            continue
        row = conn.execute(
            "SELECT content, related_to, created_at, tombstoned_at, updated_at FROM ui_tombstones WHERE id = ?",
            (cid,),
        ).fetchone()
        if row is None:
            continue
        is_derivation = raw is not None and row[0] == raw
        if not is_derivation and row[0] != texts.get(cid):
            continue
        if not is_closet_shaped(
            parent_id, slug, content=row[0], related_raw=row[1], created_at=row[2], parent_created_at=parent[2]
        ):
            continue
        if not is_derivation:
            # Cleared on the strength of the copy row beside it, so that row's
            # pre-image is the only remaining copy of this text — kept HERE rather
            # than left to the caller. The parent's redaction backs the same row up
            # a moment later (idempotently), but a DIRECT forget of the copy comes
            # through ``clear_copy_snapshot`` and never did, so once the adoption's
            # own pre-image had aged out the entry went with nothing behind it.
            back_up_migrated_rows(conn, [cid], "cleared")
        conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (cid,))
        if is_derivation:
            # PROVEN: the snapshot IS what this parent derives, so the cloud's
            # copy of it is ours to delete. The other arm rests on the grade that
            # adopted the row beside it, which is inference — and an announcement
            # deletes the cloud's row for every device, so it never rides on one.
            # The text is not lost either way: the pre-image above is kept for
            # exactly as long as a deletion is.
            rearm_legacy_announcement(conn, cid, later_stamp(row[3], row[4]))  # type: ignore[arg-type]
        if cid not in already:
            cleared.append(cid)
    return cleared


# --- one-time migration ---------------------------------------------------


def _parent_splits(memory_id: str) -> list[tuple[str, str]]:
    """Every ``(parent_id, slug)`` split of an id at a ``_closet_`` separator.

    All of them, not just the first: a parent id may itself contain the
    separator, and a speaker slug may too.
    """
    out: list[tuple[str, str]] = []
    start = 0
    while True:
        idx = memory_id.find(CLOSET_SEPARATOR, start)
        if idx < 0:
            return out
        out.append((memory_id[:idx], memory_id[idx + len(CLOSET_SEPARATOR) :]))
        start = idx + 1


def is_closet_shaped(
    parent_id: str,
    slug: str,
    *,
    content: str,
    related_raw: str | None,
    created_at: str | None = None,
    parent_created_at: str | None = None,
) -> bool:
    """Whether a pre-marker row carries the shape only bloom's closet writer produces.

    There is no marker in pre-migration data to read, so this is the one
    inferential test in the fix, and every migration decision that TOUCHES a row
    requires it — the mark-and-rebuild step as well as the orphan purge. Being
    re-derivable from a live parent is NOT sufficient on its own: an id is just
    an id, and a real memory that happens to sit at ``<parent>_closet_<speaker>``
    would otherwise have its body overwritten with that speaker's turns and be
    hidden from every list.

    A row is treated as a legacy closet only if ALL of:

      * the id splits as ``<parent_id>_closet_<slug>``;
      * ``related_to`` is exactly ``[parent_id]`` — the back-reference bloom
        writes, not merely a list containing it;
      * the content parses as a non-empty JSON list of turn objects that ALL
        carry the same single speaker — bloom's per-speaker projection;
      * that speaker's slug is ``slug``, or ``slug`` is its disambiguated
        ``<base>_<n>`` form;
      * when the parent ROW is still present — whether or not it still derives
        this id — ``created_at`` equals the parent's.

    That last one gates the orphan purge as well as the rebuild. A parent edited
    down to prose before this fix derives nothing, so its genuine stale copies
    reach the purge branch; they still carry its ``created_at``, while a
    hand-crafted row that merely points at it typically does not. Only a parent
    that is entirely gone leaves the other four properties standing alone,
    because then there is nothing left to compare against.

    ``created_at`` is the only parent-relative field safe to require. A pre-fix
    fallback-engine edit of the parent rewrites its type, project, source,
    confidence, expiry and ``updated_at`` without touching the copies, and
    rebuilding those stale copies is the whole point — but nothing ever rewrites
    ``created_at``.

    ``enriched_content`` deliberately proves NOTHING here, tempting as it looks.
    Closets ARE synced by clients at or below 0.3.0, and a second device that
    pulls one runs it through ``ClosetHybridEngine.ingest`` as an ordinary
    memory, which rewrites the enrichment with the FULL-session preamble
    ("Conversation on ... between Alice and Bob") instead of the per-speaker one.
    A genuine legacy closet can therefore arrive with any enrichment at all, and
    requiring the per-speaker preamble left exactly those rows unmarked, listed,
    and alive through their parent's redaction.

    This test alone is never enough to REWRITE, DELETE or ANNOUNCE a row. It is
    the LIKELY grade in :func:`classify_closet_row`, which earns marking and a
    snapshot and nothing else. A hand-crafted row matching all of the above is
    indistinguishable from a copy and will be hidden — with its text intact, a
    pre-image kept, and no announcement — which is a cost measured in visibility
    rather than in data. Reaching even that means choosing an id of the form
    ``<an existing memory>_closet_<x>``, pointing ``related_to`` at exactly that
    memory, storing a single-speaker turn list whose speaker slugifies to ``x``,
    and sharing that memory's ``created_at``. Every realistic false positive — a
    hand-titled ``customer_closet_notes``, or one holding the only copy of a
    contract — fails on the content shape alone.

    The cost is deliberate in the other direction too: a genuine pre-fix copy
    whose body is somehow not a turn list is left alone. Not destroying a real
    memory is the hard requirement; catching every copy is not.
    """
    try:
        related = json.loads(related_raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return False
    if related != [parent_id]:
        return False

    turns = _parse_turns(content)
    if not turns:
        return False
    speakers: set[str] = set()
    for turn in turns:
        speaker = speaker_of(turn)
        if speaker is None:
            return False
        speakers.add(speaker)
    if len(speakers) != 1:
        return False

    base = _slug(next(iter(speakers)))
    if not (slug == base or re.fullmatch(rf"{re.escape(base)}_\d+", slug)):
        return False

    if parent_created_at is not None and not _same_instant(created_at, parent_created_at):
        return False

    return True


def _same_instant(a: str | None, b: str | None) -> bool:
    """Whether two stored ``created_at`` strings name the same moment.

    Compared as datetimes, not text: a parent imported with a ``+02:00`` offset
    and its copy coming back from the server normalised to ``+00:00`` are the
    same instant spelled two ways, and a text comparison would grade the copy as
    a stranger, leaving it unmarked for the parent's redaction to miss. Strings
    that do not parse fall back to text equality.
    """
    if a == b:
        return True
    if a is None or b is None:
        return False
    try:
        return datetime.fromisoformat(a) == datetime.fromisoformat(b)
    except ValueError:
        return False


# The parent columns a copy inherits, in the order bloom's insert supplies them.
_PARENT_FIELDS = (
    "memory_type",
    "project",
    "source_type",
    "source_session_id",
    "source_timestamp",
    "created_at",
    "updated_at",
    "confidence",
    "expires_at",
)


# What the evidence supports for one unmarked row sitting at a derived id. The
# tiers exist because the consequences are not symmetric: marking a row hides it
# and makes a redaction reach it, rewriting it destroys text, and ANNOUNCING it
# deletes the cloud's copy for every device. Only the strongest evidence earns
# the last of those.
TIER_PROVEN = "proven"  # content IS what the parent derives right now
TIER_LIKELY = "likely"  # closet-shaped and the parent derives the id, text differs
TIER_ORPHAN = "orphan"  # closet-shaped but no parent row to check against
TIER_NONE = "none"  # a real memory; never touched


@dataclass(frozen=True)
class _Adopt:
    """A row the evidence says is a per-speaker copy, and how strong that is."""

    memory_id: str
    parent_id: str
    tier: str
    raw: str
    enriched: str
    # Parent metadata keyed by _PARENT_FIELDS. Written for TIER_PROVEN only: a
    # copy inherits its parent's project and type at creation, so a pre-fix
    # fallback edit that moved the parent to a private project left the copy
    # under the old public one, and leaving that stale keeps the parent's
    # private turns retrievable under the old filter.
    parent_meta: dict[str, object]


@dataclass(frozen=True)
class _BackfillPlan:
    proven: list[_Adopt]
    likely: list[_Adopt]
    # Counted for `poppy doctor`, never acted on.
    orphans: list[str]
    # id -> the row's stored updated_at, read BEFORE anything was rewritten.
    stored_updated_at: dict[str, str]

    def __bool__(self) -> bool:
        return bool(self.proven or self.likely)


def classify_closet_row(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    content: str,
    related_raw: str | None,
    created_at: str | None,
) -> tuple[str, str | None, str, str, dict[str, object]]:
    """Grade the evidence that ``memory_id`` is one of bloom's per-speaker copies.

    Returns ``(tier, parent_id, derived_raw, derived_enriched, parent_meta)``.

    THE TIERS, and why they are graded rather than a single yes/no:

    TIER_PROVEN — the row's content is EXACTLY what ``derive_closets`` produces
        for this id from the parent as it stands now. Not an inference: the
        parent, the id and the text all agree. Safe to mark, safe to align the
        derived fields (they already say the same thing), and safe to ANNOUNCE,
        which is the only tier that may be, because an announcement deletes the
        cloud's row for every device. The residual is a byte-identical
        hand-crafted row, which is not a thing that happens.

    TIER_LIKELY — the row passes the full conjunctive shape test, shares its
        parent's ``created_at``, and the live parent derives this exact id, but
        the TEXT DIFFERS. That is a stale pre-fix copy, or a curated split an
        importer wrote. Strong, but still inference, so it earns the reversible
        half only: snapshot, mark (hidden from list and sync, and cleared when
        the parent is redacted, so the promise holds), and KEEP the content. No
        rewrite, no deletion, no announcement. If the guess is wrong the row is
        hidden with its text intact and recoverable, never destroyed and never
        propagated.

    TIER_ORPHAN — shape matches but there is no parent row to check against, so
        neither the id derivation nor ``created_at`` can be confirmed. Left
        entirely alone and counted for a human-confirmed repair.

    TIER_NONE — everything else: a real memory that merely holds the id.
    """
    for parent_id, slug in _parent_splits(memory_id):
        parent = conn.execute(
            f"SELECT content, {', '.join(_PARENT_FIELDS)} FROM memories WHERE id = ?", (parent_id,)
        ).fetchone()
        if parent is None:
            # No parent to check against. Shape alone is all there is, and shape
            # alone has never been enough to touch a row.
            if is_closet_shaped(parent_id, slug, content=content, related_raw=related_raw):
                return (TIER_ORPHAN, parent_id, "", "", {})
            continue

        meta = dict(zip(_PARENT_FIELDS, parent[1:]))
        derived = {
            closet_id(parent_id, s): (raw, enriched)
            for s, raw, enriched in derive_closets(parent[0], meta["source_timestamp"])
        }
        if memory_id not in derived:
            # The parent is here but no longer accounts for this id — edited down
            # to prose, or a speaker removed, both by the pre-fix fallback bug.
            # Same evidence problem as a missing parent: nothing confirms the
            # derivation, so nothing is done and it is reported instead.
            if is_closet_shaped(
                parent_id,
                slug,
                content=content,
                related_raw=related_raw,
                created_at=created_at,
                parent_created_at=meta["created_at"],
            ):
                return (TIER_ORPHAN, parent_id, "", "", {})
            continue
        raw, enriched = derived[memory_id]
        # Provenance FIRST, in both grades. Matching text strengthens the case;
        # it never substitutes for the fields that say where the row came from.
        # An independent record that happens to hold exactly the turns this
        # parent projects — pointing at its own source, created weeks earlier, in
        # another project — is not a copy of anything, and grading it PROVEN on
        # the text alone got its cloud row announced for deletion.
        if not is_closet_shaped(
            parent_id,
            slug,
            content=content,
            related_raw=related_raw,
            created_at=created_at,
            parent_created_at=meta["created_at"],
        ):
            continue
        if content == raw:
            return (TIER_PROVEN, parent_id, raw, enriched, meta)
        return (TIER_LIKELY, parent_id, raw, enriched, meta)
    return (TIER_NONE, None, "", "", {})


def _plan_backfill(conn: sqlite3.Connection) -> _BackfillPlan:
    """Grade every unmarked row whose id contains the separator. READ ONLY.

    Grading is per row and never raises: a malformed legacy row is TIER_NONE,
    not an exception that would roll back the whole ALTER and leave the store
    unopenable. Tolerance is confined to this read-only classification; no write
    path gains any.
    """
    # Unmarked candidates only. On the first run every row is unmarked (the
    # column was just added with DEFAULT 0), so this is a no-op there; on a
    # re-run it scopes the work to rows an older client wrote since, and stops
    # already-classified rows being rewritten on every open.
    candidates = conn.execute(
        "SELECT id, content, related_to, created_at, updated_at "
        f"FROM memories WHERE instr(id, ?) > 0 AND {NOT_CLOSET_SQL}",
        (CLOSET_SEPARATOR,),
    ).fetchall()
    if not candidates:
        return _BackfillPlan(proven=[], likely=[], orphans=[], stored_updated_at={})

    proven: list[_Adopt] = []
    likely: list[_Adopt] = []
    orphans: list[str] = []
    stored_updated_at: dict[str, str] = {}
    for row in candidates:
        row_id, content, related_raw, row_created_at = row[0], row[1], row[2], row[3]
        try:
            tier, parent_id, raw, enriched, meta = classify_closet_row(
                conn, row_id, content=content, related_raw=related_raw, created_at=row_created_at
            )
        except Exception:
            tier, parent_id, raw, enriched, meta = (TIER_NONE, None, "", "", {})

        if tier == TIER_ORPHAN:
            orphans.append(row_id)
            continue
        if tier == TIER_NONE:
            continue

        stored_updated_at[row_id] = row[4]
        item = _Adopt(
            memory_id=row_id,
            parent_id=parent_id,  # type: ignore[arg-type]
            tier=tier,
            raw=raw,
            enriched=enriched,
            parent_meta=meta,
        )
        (proven if tier == TIER_PROVEN else likely).append(item)

    return _BackfillPlan(proven=proven, likely=likely, orphans=orphans, stored_updated_at=stored_updated_at)


def _apply_backfill(conn: sqlite3.Connection, plan: _BackfillPlan) -> None:
    """Write a plan produced by :func:`_plan_backfill`. Caller holds the write lock.

    Each tier gets only what its evidence supports. Orphans get nothing at all.
    """
    has_enriched = "enriched_content" in _columns(conn, "memories")

    for item in plan.proven:
        # The text already says what the parent derives, so nothing is destroyed
        # by aligning the row with it. The inherited fields ARE written: a copy
        # takes its parent's project and type at creation, and leaving a stale
        # public project on a copy of a now-private parent keeps the parent's
        # turns retrievable under the old filter.
        assignments = [f"{MARKER_COLUMN} = {CLOSET_DERIVED}", "related_to = ?"]
        params: list[object] = [json.dumps([item.parent_id])]
        if has_enriched:
            assignments.append("enriched_content = ?")
            params.append(item.enriched)
        for field in _PARENT_FIELDS:
            assignments.append(f"{field} = ?")
            params.append(item.parent_meta.get(field))
        params.append(item.memory_id)
        conn.execute(f"UPDATE memories SET {', '.join(assignments)} WHERE id = ?", params)

    for item in plan.likely:
        # Marked and nothing else. Hiding it and making a redaction reach it is
        # what the promise needs; rewriting its text is not, and would destroy a
        # curated split if the inference is wrong. The pre-image is kept anyway,
        # so even the marker is reversible.
        back_up_migrated_rows(conn, [item.memory_id], "adopted")
        conn.execute(
            f"UPDATE memories SET {MARKER_COLUMN} = ? WHERE id = ?",
            (CLOSET_ADOPTED_UNVERIFIED, item.memory_id),
        )

    # ONLY the proven tier is announced. An announcement deletes the cloud's row
    # for every device, so it is the one action that must not rest on inference:
    # a wrong guess there is unrecoverable everywhere, which is exactly what the
    # local pre-image cannot help with. Likely-tier copies leave their cloud rows
    # to the human-confirmed server-side cleanup.
    record_legacy_closet_ids(
        conn, [(item.memory_id, plan.stored_updated_at.get(item.memory_id)) for item in plan.proven]
    )


def count_unmarked_derivable_copies(conn: sqlite3.Connection) -> int:
    """Copies an older Poppy wrote that this one has not marked. COUNTS, never acts.

    Mixed installs are real — a 0.2.4 pipx alongside a 0.3.0 uv, both pointed at
    ``~/.poppy`` — and the older client writes unmarked copies into a store this
    one already migrated.

    They are reported rather than repaired: marking them means running the
    migration's inference outside the migration, and doing that on every store
    open turned a one-time pass over pre-marker data into a standing rule applied
    to rows that arrived long after, including from the cloud.

    Only rows the parent PROVES (content is what it derives) or strongly implies
    are counted, so a real memory whose id merely looks like a copy is never
    reported.
    """
    plan = _plan_backfill(conn)
    return len(plan.proven) + len(plan.likely)


def count_adopted_pending_rebuild(conn: sqlite3.Connection) -> int:
    """Copies adopted on inference, whose text this engine has not rewritten.

    The likely tier: hidden and covered by the parent's redaction, but never
    regenerated, because the text could be a curated split rather than a stale
    copy. Reported so the explicit repair has something to act on, and
    so a split hidden by a wrong guess is visible rather than silent.
    """
    if MARKER_COLUMN not in _columns(conn, "memories"):
        return 0
    return conn.execute(
        f"SELECT COUNT(*) FROM memories WHERE {MARKER_COLUMN} = ?", (CLOSET_ADOPTED_UNVERIFIED,)
    ).fetchone()[0]


def count_orphan_shaped_rows(conn: sqlite3.Connection) -> int:
    """Copy-shaped rows with no parent row to check against. COUNTS, never acts.

    They can only come from the pre-fix fallback-engine edit/delete bug, and
    staying recallable is the status quo rather than a regression, so purging
    them on a guess is not worth the one case where the guess is wrong. Reported
    for the human-confirmed repair.
    """
    return len(_plan_backfill(conn).orphans)


def migrate_closet_marker(conn: sqlite3.Connection, *, had_bloom_schema: bool) -> None:
    """Idempotently add ``memories.is_closet`` and classify pre-marker rows.

    The ALTER and the backfill share ONE transaction, so the column can never
    exist without having been classified: a crash in between would otherwise
    leave every legacy closet permanently unmarked — listed, synced, and immune
    to redaction — with nothing that would ever revisit it.

    The column's presence is what makes this run-once, so the state is re-read
    inside the write lock: another process may have migrated while we waited.

    Run ONCE, and only here. The column having been absent is what proves this
    store predates the marker, which is the only thing that justifies grading
    provenance from data that carries none. A store that already has the column
    is left alone: copies an older Poppy writes into it afterwards stay unmarked,
    and are REPORTED by ``poppy doctor`` (see
    :func:`count_unmarked_derivable_copies`) rather than silently reclassified.
    Doing that reclassification on every open turned this migration into a
    standing rule and destroyed rows that had merely arrived from the cloud.

    ``had_bloom_schema`` says whether the store had ever been written by the
    default engine BEFORE this open. A store that only ever ran ``seed`` cannot
    contain a per-speaker copy, because nothing in it ever derived one — so
    nothing in it is classified, whatever the ids look like. The caller has to
    supply this: bloom adds its own schema during construction, so by the time
    this runs the store looks like a bloom store either way.
    """
    if has_marker(conn):
        return

    # The caller may already hold a transaction, so that what this marks and what
    # the caller does with those marks commit together. Joining it rather than
    # refusing is what lets the copy cleanup run in the same breath: a crash
    # between the two would otherwise leave rows marked and queued for an upload
    # lane this version no longer drains.
    own_txn = not conn.in_transaction
    if own_txn:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "within a transaction" not in str(exc):
                raise
            own_txn = False
    try:
        if has_marker(conn):
            if own_txn:
                conn.rollback()
            return
        conn.execute(f"ALTER TABLE memories ADD COLUMN {MARKER_COLUMN} INTEGER NOT NULL DEFAULT 0")
        if had_bloom_schema:
            _apply_backfill(conn, _plan_backfill(conn))
        if own_txn:
            conn.commit()
    except Exception:
        if own_txn:
            conn.rollback()
        raise
