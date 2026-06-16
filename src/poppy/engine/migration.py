"""Embedding-channel migration helpers.

When a user switches engines, the shared ``memory_embeddings`` table can hold
BLOBs that were produced by a different bi-encoder. Reading them through the
new engine's cosine math produces noise. These helpers identify and re-embed
those rows.

The CLI command ``poppy migrate-engine`` exposes selective re-embedding so a
user can scope the work to a project, memory type, or time window — useful
when most of the corpus is archival and not worth the re-embed cost. The
``engines use`` and ``doctor`` commands call ``stale_count`` to surface drift.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from poppy.db import connect as connect_db
from poppy.engine.interface import RetrievalEngine


@dataclass
class StaleStats:
    """How much of ``memory_embeddings`` needs re-embedding under ``model_id``.

    ``compatible`` is rows already tagged with the active engine's model_id.
    ``stale`` is rows tagged with a different model_id. ``unknown`` is rows
    from pre-engine-fingerprinting DBs whose model_id column is NULL — we can't tell what
    produced them, so we treat them as stale by default.
    """

    compatible: int
    stale: int
    unknown: int
    orphans: int = 0

    @property
    def needs_migration(self) -> int:
        # Orphans deliberately excluded: ``migrate-engine`` can't re-embed an
        # embedding whose ``memories`` row no longer exists, so flagging them
        # as "needs migration" would produce a WARN the user can never clear.
        return self.stale + self.unknown


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
        conn.row_factory = sqlite3.Row
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
            return StaleStats(compatible=0, stale=0, unknown=live_total, orphans=orphans)
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
        return StaleStats(
            compatible=compatible,
            stale=stale,
            unknown=unknown,
            orphans=orphans,
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
        clauses.append("(e.model_id IS NULL OR e.model_id != ?)")
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
        conn.row_factory = sqlite3.Row
        where, params = _build_where_clause(model_id, filters)
        sql = f"SELECT COUNT(*) FROM memory_embeddings e JOIN memories m ON m.id = e.id WHERE {where}"
        return conn.execute(sql, params).fetchone()[0]
    finally:
        conn.close()


def list_targets(db_path: Path, model_id: str, filters: MigrateFilters) -> list[tuple[str, str]]:
    """Return ``[(memory_id, content), ...]`` to re-embed.

    Materializes the result so the read cursor closes before callers acquire
    a write connection — SQLite serializes writers, so an open read cursor on
    the same DB blocks UPDATEs and produces ``database is locked``.

    The active engine's content extractor is the engine's responsibility once
    we hand back IDs — we read ``memories.content`` which is the canonical
    column shared by every engine. Engines like ``bloom`` that index
    enriched_content compute that derivation at re-embed time from content.
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
        conn.row_factory = sqlite3.Row
        where, params = _build_where_clause(model_id, filters)
        sql = (
            "SELECT m.id, m.content FROM memory_embeddings e "
            "JOIN memories m ON m.id = e.id "
            f"WHERE {where} "
            "ORDER BY m.created_at"
        )
        return [(row["id"], row["content"]) for row in conn.execute(sql, params)]
    finally:
        conn.close()


def migrate(
    engine: RetrievalEngine,
    db_path: Path,
    filters: MigrateFilters,
    on_progress: Callable[[int, int], None] | None = None,
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
        raise ValueError(
            "engine does not use embeddings; nothing to migrate. Switch to an engine with a bi-encoder first."
        )
    embedder = getattr(engine, "_embed", None)
    if embedder is None:
        raise RuntimeError(
            f"engine {type(engine).__name__} declares model_id={engine.model_id!r} "
            "but exposes no _embed(text) method; migration cannot proceed."
        )
    targets = list_targets(db_path, engine.model_id, filters)
    total = len(targets)
    if total == 0:
        return 0
    write_conn = connect_db(db_path)
    try:
        done = 0
        for memory_id, content in targets:
            vec = embedder(content)
            blob = vec.tobytes()
            write_conn.execute(
                "UPDATE memory_embeddings SET embedding = ?, model_id = ? WHERE id = ?",
                (blob, engine.model_id, memory_id),
            )
            write_conn.commit()
            done += 1
            if on_progress:
                on_progress(done, total)
        return done
    finally:
        write_conn.close()
