"""Embedding-channel migration helpers.

The shared ``memory_embeddings`` table can hold missing, invalidated, or older
model vectors. These helpers restore the full text and embedding together.

The CLI command ``poppy migrate-engine`` exposes selective re-embedding so a
user can scope the work to a project, memory type, or time window — useful
when most of the corpus is archival and not worth the re-embed cost. The
daemon backfills use the same migration path as the manual command.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from poppy.db import apply_row_factory
from poppy.db import connect as connect_db
from poppy.engine._timestamps import migration_applied, record_migration
from poppy.engine.interface import RetrievalEngine

BACKFILL_LIMIT = 32
ENRICHMENT_REPAIR = "full_content_embeddings_v1"


@dataclass
class StaleStats:
    """How much of ``memory_embeddings`` needs re-embedding under ``model_id``.

    ``compatible`` is rows already tagged with the active engine's model_id.
    ``stale`` is rows tagged with a different model_id. ``unknown`` is rows
    from pre-engine-fingerprinting DBs whose model_id column is NULL — we can't tell what
    produced them, so we treat them as stale by default. ``missing`` is live
    memories with no embedding row at all: the FTS-only engines write rows
    without vectors, and before this was counted such a memory was invisible
    to both doctor and migrate-engine (every query joined *through*
    memory_embeddings), so it stayed permanently FTS-only under bloom.
    """

    compatible: int
    stale: int
    unknown: int
    orphans: int = 0
    missing: int = 0

    @property
    def needs_migration(self) -> int:
        # Orphans deliberately excluded: ``migrate-engine`` can't re-embed an
        # embedding whose ``memories`` row no longer exists, so flagging them
        # as "needs migration" would produce a WARN the user can never clear.
        # ``missing`` IS included: those rows exist and can be embedded, they
        # just never were.
        return self.stale + self.unknown + self.missing


def stale_stats(db_path: Path, model_id: str | None) -> StaleStats:
    """Count embeddings by compatibility with ``model_id``.

    Returns zeros if the DB or memory_embeddings table doesn't exist, or if
    the engine doesn't use embeddings (``model_id is None``).

    Counts are scoped to rows whose ``memories`` row still exists — orphans
    (deleted memory, embedding row left behind) are tallied separately so
    the user-facing ``needs_migration`` number matches what
    ``migrate-engine`` can actually act on.
    """
    if not db_path.exists() or model_id is None:
        return StaleStats(compatible=0, stale=0, unknown=0)
    conn = connect_db(db_path)
    try:
        apply_row_factory(conn)
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
        ).fetchone()
        if not table:
            return StaleStats(0, 0, 0)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(memory_embeddings)")}
        # An anti-join against ``memories`` catches embeddings whose memory
        # was deleted without the cascading row removal. They show up as
        # ``orphans`` and don't inflate the migration target count.
        orphans = conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings e LEFT JOIN memories m ON m.id = e.id WHERE m.id IS NULL"
        ).fetchone()[0]
        if "model_id" not in cols:
            # Pre-migration DB. Live embeddings are all "unknown"; orphans
            # are still tracked separately.
            live_total = conn.execute(
                "SELECT COUNT(*) FROM memory_embeddings e JOIN memories m ON m.id = e.id"
            ).fetchone()[0]
            missing = conn.execute(
                "SELECT COUNT(*) FROM memories m LEFT JOIN memory_embeddings e ON m.id = e.id WHERE e.id IS NULL"
            ).fetchone()[0]
            return StaleStats(compatible=0, stale=0, unknown=live_total, orphans=orphans, missing=missing)
        compatible = conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings e JOIN memories m ON m.id = e.id WHERE e.model_id = ?",
            (model_id,),
        ).fetchone()[0]
        unknown = conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings e JOIN memories m ON m.id = e.id WHERE e.model_id IS NULL"
        ).fetchone()[0]
        live_total = conn.execute("SELECT COUNT(*) FROM memory_embeddings e JOIN memories m ON m.id = e.id").fetchone()[
            0
        ]
        stale = live_total - compatible - unknown
        # Anti-join the other way: memories that never got a vector.
        missing = conn.execute(
            "SELECT COUNT(*) FROM memories m LEFT JOIN memory_embeddings e ON m.id = e.id WHERE e.id IS NULL"
        ).fetchone()[0]
        return StaleStats(
            compatible=compatible,
            stale=stale,
            unknown=unknown,
            orphans=orphans,
            missing=missing,
        )
    finally:
        conn.close()


def sweep_orphans(db_path: Path) -> int:
    """Delete ``memory_embeddings`` rows whose memory was already removed.

    Returns rowcount. Called automatically by ``migrate-engine`` so a routine
    re-embed run also keeps the embeddings table tidy. Safe to call on a
    clean DB — it's a no-op when there are no orphans.
    """
    if not db_path.exists():
        return 0
    conn = connect_db(db_path)
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
        ).fetchone()
        if not table:
            return 0
        cursor = conn.execute(
            "DELETE FROM memory_embeddings "
            "WHERE id IN ("
            "  SELECT e.id FROM memory_embeddings e "
            "  LEFT JOIN memories m ON m.id = e.id "
            "  WHERE m.id IS NULL"
            ")"
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


@dataclass
class MigrateFilters:
    """Selective scoping for ``poppy migrate-engine``.

    All filters are AND-combined. ``since_days`` constrains by ``created_at``
    relative to now. ``include_compatible`` re-embeds rows already tagged with
    the active model_id too — useful for a forced full rebuild after a model
    weight upgrade where the name stayed the same.
    """

    project: str | None = None
    memory_type: str | None = None
    since_days: int | None = None
    include_compatible: bool = False


def _build_where_clause(model_id: str, filters: MigrateFilters) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if not filters.include_compatible:
        # ``e.id IS NULL`` is the LEFT JOIN miss: a memory with no vector at
        # all. It is a migration target for the same reason a mismatched
        # model_id is — bloom cannot score it until something embeds it.
        clauses.append("(e.id IS NULL OR e.model_id IS NULL OR e.model_id != ?)")
        params.append(model_id)
    if filters.project:
        clauses.append("m.project = ?")
        params.append(filters.project)
    if filters.memory_type:
        clauses.append("m.memory_type = ?")
        params.append(filters.memory_type)
    if filters.since_days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=filters.since_days)).isoformat()
        clauses.append("m.created_at >= ?")
        params.append(cutoff)
    where = " AND ".join(clauses) if clauses else "1=1"
    return where, params


def count_targets(db_path: Path, model_id: str, filters: MigrateFilters) -> int:
    """Count rows matching the migration filters, without re-embedding any."""
    if not db_path.exists():
        return 0
    conn = connect_db(db_path)
    try:
        # An FTS-only DB (e.g. created by SeedEngine) has no
        # memory_embeddings — nothing to migrate from that direction.
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
        ).fetchone()
        if not table:
            return 0
        apply_row_factory(conn)
        where, params = _build_where_clause(model_id, filters)
        sql = f"SELECT COUNT(*) FROM memories m LEFT JOIN memory_embeddings e ON m.id = e.id WHERE {where}"
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def list_targets(
    db_path: Path, model_id: str, filters: MigrateFilters, *, limit: int | None = None
) -> list[tuple[str, str]]:
    """Return ``[(memory_id, content), ...]`` to re-embed.

    Materializes the result so the read cursor closes before callers acquire
    a write connection — SQLite serializes writers, so an open read cursor on
    the same DB blocks UPDATEs and produces ``database is locked``.

    Content is a raw snapshot from the canonical column shared by every engine.
    ``migrate`` re-reads it with the source timestamp and uses the active
    engine's ``embed_text_for`` method to prepare embedding text.
    """
    if not db_path.exists():
        return []
    conn = connect_db(db_path)
    try:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
        ).fetchone()
        if not table:
            return []
        apply_row_factory(conn)
        where, params = _build_where_clause(model_id, filters)
        sql = (
            "SELECT m.id, m.content FROM memories m "
            "LEFT JOIN memory_embeddings e ON m.id = e.id "
            f"WHERE {where} "
            "ORDER BY m.created_at"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [(row["id"], row["content"]) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def migrate(
    engine: RetrievalEngine,
    db_path: Path,
    filters: MigrateFilters,
    on_progress: Callable[[int, int], None] | None = None,
    limit: int | None = None,
) -> int:
    """Re-embed every row matching ``filters`` using ``engine``'s bi-encoder.

    Returns the number of rows actually re-embedded. Callers should refuse to
    call this when ``engine.model_id is None`` (FTS-only engines have no
    bi-encoder to call).

    Per-row commit so an interrupted run is resumable: anything written stays
    written, and the next call picks up where this one stopped because the
    just-finished rows now match the active model_id and drop out of the
    target set.
    """
    if engine.model_id is None:
        raise ValueError("engine does not use embeddings; models must be available before migration.")
    embedder = getattr(engine, "_embed", None)
    if embedder is None:
        raise RuntimeError(
            f"engine {type(engine).__name__} declares model_id={engine.model_id!r} "
            "but exposes no _embed(text) method; migration cannot proceed."
        )
    targets = (
        list_targets(db_path, engine.model_id, filters)
        if limit is None
        else list_targets(db_path, engine.model_id, filters, limit=limit)
    )
    total = len(targets)
    if total == 0:
        return 0
    write_conn = connect_db(db_path)
    try:
        done = 0
        for memory_id, _snapshot in targets:
            # Re-read the content instead of embedding the snapshot list_targets
            # took: another writer (a seed edit) may have changed it since, and
            # embedding stale text while tagging the row with the active model_id
            # would leave a wrong vector that doctor then reports as compatible.
            # A row deleted in that window is skipped.
            row = write_conn.execute(
                "SELECT content, source_timestamp FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                continue
            content, source_timestamp = row
            text = engine.embed_text_for(content, source_timestamp)
            vec = embedder(text)
            blob = vec.tobytes()
            # Embedding can be slow, so it runs outside the lock; the write then
            # re-verifies the raw content and timestamp under BEGIN IMMEDIATE.
            # It only stores the vector if neither changed during the embed. A
            # concurrent seed edit that landed meanwhile has already invalidated
            # the row's model_id, so skipping here leaves it a target for the
            # next pass rather than overwriting that invalidation with a vector
            # for the old text.
            write_conn.execute("BEGIN IMMEDIATE")
            try:
                current = write_conn.execute(
                    "SELECT content, source_timestamp FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                if current is None or current[0] != content or current[1] != source_timestamp:
                    write_conn.rollback()
                    continue
                # INSERT OR REPLACE, not UPDATE: a target may have no embedding
                # row yet (written by an FTS-only engine), and an UPDATE would
                # silently affect nothing while still counting as migrated.
                write_conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
                    (memory_id, blob, engine.model_id),
                )
                # The AFTER UPDATE trigger replaces the FTS row with this
                # enrichment, keeping both retrieval channels equal to ingest.
                write_conn.execute("UPDATE memories SET enriched_content = ? WHERE id = ?", (text, memory_id))
                write_conn.commit()
            except Exception:
                write_conn.rollback()
                raise
            done += 1
            if on_progress:
                on_progress(done, total)
        return done
    finally:
        write_conn.close()


def _repair_enrichment(engine: RetrievalEngine, db_path: Path, limit: int) -> None:
    """Invalidate old raw-text vectors, scanning at most ``limit`` rows per pass."""
    conn = connect_db(db_path)
    try:
        if migration_applied(conn, ENRICHMENT_REPAIR):
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            if migration_applied(conn, ENRICHMENT_REPAIR):
                conn.commit()
                return
            # The migration log records completion; a separate cursor lets a
            # large store finish the one-time scan over several bounded passes.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS poppy_backfill_state (name TEXT PRIMARY KEY, last_rowid INTEGER NOT NULL)"
            )
            cursor = conn.execute(
                "SELECT last_rowid FROM poppy_backfill_state WHERE name = ?", (ENRICHMENT_REPAIR,)
            ).fetchone()
            rows = conn.execute(
                "SELECT rowid, id, content, source_timestamp, enriched_content FROM memories "
                "WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (cursor[0] if cursor else 0, limit),
            ).fetchall()
            for _, memory_id, content, source_timestamp, enriched in rows:
                if enriched != engine.embed_text_for(content, source_timestamp):
                    # Persist the target before advancing the cursor. If an
                    # embed fails or a concurrent edit is skipped, the normal
                    # stale-vector query will retry it even after the scan ends.
                    conn.execute(
                        "UPDATE memory_embeddings SET model_id = 'enrichment-invalidated' "
                        "WHERE id = ? AND model_id = ?",
                        (memory_id, engine.model_id),
                    )
            if len(rows) < limit:
                record_migration(conn, ENRICHMENT_REPAIR)
                conn.execute("DELETE FROM poppy_backfill_state WHERE name = ?", (ENRICHMENT_REPAIR,))
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO poppy_backfill_state (name, last_rowid) VALUES (?, ?)",
                    (ENRICHMENT_REPAIR, rows[-1][0]),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    finally:
        conn.close()


def backfill(engine: RetrievalEngine, db_path: Path, *, limit: int = BACKFILL_LIMIT) -> int:
    """Restore a bounded batch of memories, including a resumable one-time text repair."""
    if engine.model_id is None or limit <= 0:
        return 0
    _repair_enrichment(engine, db_path, limit)
    return migrate(engine, db_path, MigrateFilters(), limit=limit)
