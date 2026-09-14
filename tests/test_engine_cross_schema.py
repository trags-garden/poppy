"""Cross-engine schema compatibility: seed writing to a bloom-created store.

``bloom`` owns a richer ``memories`` schema than ``seed``: a NOT NULL
``enriched_content`` column, plus FTS triggers that index that column instead
of ``content``. ``seed`` is the documented offline fallback, so a store first
written by bloom must stay writable after ``poppy engines use seed`` — and the
rows seed writes must stay keyword-searchable under both engines.

Encoders are injected fakes: no ONNX, no model downloads.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine import seed as seed_mod
from poppy.engine._closet_engine import SCHEMA as BLOOM_SCHEMA
from poppy.engine._closet_marker import _closet_like_pattern
from poppy.engine.bloom import BloomEngine
from poppy.engine.migration import MigrateFilters, count_targets, list_targets, stale_stats
from poppy.engine.migration import migrate as run_migrate
from poppy.engine.seed import SEED_INVALIDATED_MODEL_ID, SeedEngine
from poppy.models import Memory, Source

TURNS_JSON = '[{"speaker":"Alice","dia_id":"D1","text":"hello"},{"speaker":"Bob","dia_id":"D2","text":"world"}]'


class _FakeBiEncoder:
    def embed(self, texts):
        for t in texts:
            h = hash(t) & 0xFFFF
            yield np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeCrossEncoder:
    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


def _bloom(db_path: Path) -> BloomEngine:
    return BloomEngine(db_path=db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())


def _memory(mid: str, content: str, *, project: str = "p1") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=content,
        memory_type="fact",
        source=Source(type="cli", session_id=None, timestamp=now),
        project=project,
        related_to=[],
        created_at=now,
        updated_at=now,
        confidence=1.0,
    )


def _fts_text(db_path: Path, memory_id: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT content FROM memory_fts WHERE id = ?", (memory_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _enriched(db_path: Path, memory_id: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT enriched_content FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_seed_ingests_into_bloom_schema_store(tmp_path: Path) -> None:
    """Regression: seed INSERT hit
    'NOT NULL constraint failed: memories.enriched_content' on any store bloom
    had written first.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("bloom1", "bloom wrote this body"))

    seed = SeedEngine(db_path=db)
    assert seed.ingest(_memory("seed1", "seed wrote this body")) == "seed1"

    got = seed.get("seed1")
    assert got is not None
    assert got.content == "seed wrote this body"
    # Mirrored into the column bloom's triggers index, so the row is not a
    # search black hole under either engine.
    assert _enriched(db, "seed1") == "seed wrote this body"
    assert _fts_text(db, "seed1") == "seed wrote this body"


def test_seed_ingests_on_bloom_schema_applied_directly(tmp_path: Path) -> None:
    """Same contract against the raw bloom schema, with no engine in between."""
    db = tmp_path / "memories.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(BLOOM_SCHEMA)
    conn.commit()
    conn.close()

    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("seed1", "offline fallback body"))
    assert _enriched(db, "seed1") == "offline fallback body"


def test_seed_written_row_is_searchable_under_both_engines(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("bloom1", "bloom wrote this body"))

    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("seed1", "capacitor discharge notes"))

    assert [r.memory.id for r in seed.retrieve("capacitor discharge", limit=5)] == ["seed1"]
    # Switching back to bloom must find the same row (FTS-only: seed writes no
    # embedding, so it reaches RRF through the FTS channel alone).
    reopened = _bloom(db)
    assert "seed1" in {r.memory.id for r in reopened.retrieve("capacitor discharge", limit=5)}


def test_seed_update_on_bloom_store_reindexes_enriched_content(tmp_path: Path) -> None:
    """Updating through seed must refresh enriched_content too.

    The FTS triggers reindex from enriched_content on UPDATE, so leaving
    bloom's stale enrichment behind would keep the old text searchable and
    hide the new one.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("m1", TURNS_JSON))
    assert "Alice" in (_fts_text(db, "m1") or "")

    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("m1", "rewritten by seed"))

    assert seed.get("m1").content == "rewritten by seed"
    assert _enriched(db, "m1") == "rewritten by seed"
    assert _fts_text(db, "m1") == "rewritten by seed"
    assert [r.memory.id for r in seed.retrieve("rewritten by seed", limit=5)] == ["m1"]


def test_bloom_still_writes_after_seed_touched_the_store(tmp_path: Path) -> None:
    """Switching back to bloom keeps working, and seed's rows survive it."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("bloom1", "bloom wrote this body"))
    SeedEngine(db_path=db).ingest(_memory("seed1", "seed wrote this body"))

    engine = _bloom(db)
    engine.ingest(_memory("bloom2", TURNS_JSON))
    assert engine.get("bloom2") is not None
    assert engine.get("seed1").content == "seed wrote this body"
    # Bloom's enrichment path is unaffected by the seed row alongside it.
    assert "Conversation" in (_fts_text(db, "bloom2") or "")


def test_seed_store_still_upgrades_to_bloom(tmp_path: Path) -> None:
    """Reverse direction: a seed-only store keeps migrating cleanly."""
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_memory("legacy", "legacy body"))

    engine = _bloom(db)
    assert engine.get("legacy").content == "legacy body"
    assert _enriched(db, "legacy") == "legacy body"
    engine.ingest(_memory("new", "new body"))
    assert engine.get("new") is not None


def _closet_ids(db_path: Path, parent_id: str, table: str = "memories") -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            f"SELECT id FROM {table} WHERE id LIKE ? ESCAPE '\\'",  # noqa: S608 - table is a literal above
            (_closet_like_pattern(parent_id),),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def _embedding_ids(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT id FROM memory_embeddings ORDER BY id")]
    finally:
        conn.close()


def test_migration_cannot_interleave_between_probe_and_write(tmp_path: Path) -> None:
    """Cross-process TOCTOU: the schema probe and the write must be one atomic step.

    Without the write lock, a bloom process can ALTER enriched_content in
    after the probe says "legacy" and before the INSERT runs. The migrated
    column is nullable, so the legacy INSERT succeeds and stores NULL — and
    the rewired FTS triggers then index nothing for that row, making it
    unrecallable by content under both engines.

    The interleave is attempted from a genuinely separate connection with a
    short busy timeout, so the assertion is deterministic rather than timing
    dependent: either it is blocked, or the bug is present.
    """
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)  # seed-only store: the probe will say "legacy"
    race: dict[str, object] = {}
    real_probe = seed_mod._has_enriched_content

    def racing_probe(conn):
        result = real_probe(conn)
        if not result and "attempted" not in race:
            race["attempted"] = True
            other = sqlite3.connect(str(db), timeout=0.5)
            try:
                seed_mod._migrate_enriched_content(other)
                race["migrated"] = True
            except sqlite3.OperationalError as exc:
                race["blocked"] = str(exc)
            finally:
                other.close()
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(seed_mod, "_has_enriched_content", racing_probe)
        seed.ingest(_memory("m1", "racy body"))

    assert race.get("attempted") is True, "probe never ran; the race was not exercised"
    assert "migrated" not in race, "a concurrent migration slipped between the probe and the write"
    assert "locked" in str(race.get("blocked", "")).lower()

    # The migration lands afterwards instead, which backfills our row rather
    # than stranding it: still non-NULL and still searchable.
    conn = sqlite3.connect(str(db))
    seed_mod._migrate_enriched_content(conn)
    conn.close()
    assert _enriched(db, "m1") == "racy body"
    assert [r.memory.id for r in SeedEngine(db_path=db).retrieve("racy body", limit=5)] == ["m1"]


def test_seed_delete_on_seed_only_store_still_works(tmp_path: Path) -> None:
    """The closet cleanup must not touch a store with no memory_embeddings table."""
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("m1", "plain body"))
    assert seed.delete("m1") is True
    assert seed.get("m1") is None


def _multi_speaker(mid: str, phrase: str) -> Memory:
    turns = f'[{{"speaker":"Alice","dia_id":"D1","text":"{phrase}"}},{{"speaker":"Bob","dia_id":"D2","text":"hi"}}]'
    return _memory(mid, turns)


def test_seed_metadata_only_edit_preserves_bloom_derived_data(tmp_path: Path) -> None:
    """A metadata-only re-ingest must not tear down bloom's derived data.

    `poppy edit m1 --project p2` replays the memory through ingest with
    identical content. Clearing closets, the parent vector and the enrichment
    there destroys accurate data that nothing rebuilds: migrate-engine and
    doctor reach rows through a join on memory_embeddings, so a row left
    without one is invisible to both.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "hello"))
    closets_before = _closet_ids(db, "m1")
    embeddings_before = _embedding_ids(db)
    enriched_before = _enriched(db, "m1")
    assert len(closets_before) == 2
    assert len(embeddings_before) == 3

    seed = SeedEngine(db_path=db)
    same_content = seed.get("m1").content
    edited = _memory("m1", same_content, project="p2")  # project changes, content does not
    seed.ingest(edited)

    assert seed.get("m1").project == "p2"  # the edit did land
    assert _closet_ids(db, "m1") == closets_before
    assert _embedding_ids(db) == embeddings_before
    assert _enriched(db, "m1") == enriched_before
    assert "Conversation" in (enriched_before or "")
    # Bloom still finds it by a term that exists only in the enrichment
    # preamble, never in the raw JSON content.
    assert "m1" in {r.memory.id for r in _bloom(db).retrieve("Conversation", limit=10)}


def test_seed_metadata_only_edit_on_seed_only_store(tmp_path: Path) -> None:
    """The gate is a no-op where there is no bloom-derived data to protect."""
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("m1", "plain body"))
    seed.ingest(_memory("m1", "plain body", project="p2"))

    got = seed.get("m1")
    assert got.project == "p2"
    assert got.content == "plain body"
    assert [r.memory.id for r in seed.retrieve("plain body", limit=5)] == ["m1"]


def _model_ids(db_path: Path) -> dict[str, str | None]:
    conn = sqlite3.connect(str(db_path))
    try:
        return dict(conn.execute("SELECT id, model_id FROM memory_embeddings"))
    finally:
        conn.close()


def test_retagged_row_is_a_migrate_engine_target_and_doctor_stale_count(tmp_path: Path) -> None:
    """The repair path exists end to end: doctor counts it, migrate-engine fixes it."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("m1", "alpha body"))
    assert stale_stats(db, BloomEngine.model_id).needs_migration == 0

    SeedEngine(db_path=db).ingest(_memory("m1", "beta body"))

    stats = stale_stats(db, BloomEngine.model_id)
    assert stats.stale == 1  # what `poppy doctor` reports
    assert stats.needs_migration == 1
    assert stats.orphans == 0
    filters = MigrateFilters()
    assert count_targets(db, BloomEngine.model_id, filters) == 1
    assert [mid for mid, _ in list_targets(db, BloomEngine.model_id, filters)] == ["m1"]

    # And running the migration actually repairs it, re-embedding from the new content.
    assert run_migrate(_bloom(db), db, filters) == 1
    assert _model_ids(db)["m1"] == BloomEngine.model_id
    assert stale_stats(db, BloomEngine.model_id).needs_migration == 0


def test_seed_metadata_only_edit_does_not_retag(tmp_path: Path) -> None:
    """No content change, no invalidation: the vector still describes the row."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("m1", "alpha body"))

    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("m1", seed.get("m1").content, project="p2"))

    assert seed.get("m1").project == "p2"
    assert _model_ids(db)["m1"] == BloomEngine.model_id
    assert stale_stats(db, BloomEngine.model_id).needs_migration == 0


def test_seed_content_edit_retags_the_parent_vector_only(tmp_path: Path) -> None:
    """The retag covers this row's vector and nothing else."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("m1", "alpha body"))

    SeedEngine(db_path=db).ingest(_memory("m1", "beta body"))

    assert _embedding_ids(db) == ["m1"]  # retagged, not deleted
    assert _model_ids(db)["m1"] == SEED_INVALIDATED_MODEL_ID


def test_bloom_reingest_rebuilds_everything_after_a_seed_edit(tmp_path: Path) -> None:
    """Bloom's own ingest is the repair for closet staleness."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "obsoletephoenix"))
    SeedEngine(db_path=db).ingest(_memory("m1", "current replacement"))

    engine = _bloom(db)
    engine.ingest(_memory("m1", "current replacement"))

    assert _closet_ids(db, "m1") == []  # plain text has no speakers to expand
    assert _model_ids(db) == {"m1": BloomEngine.model_id}
    assert not [r for r in engine.retrieve("obsoletephoenix", limit=10) if "obsoletephoenix" in r.memory.content]


def test_seed_written_rows_are_visible_to_doctor_and_migrate_engine(tmp_path: Path) -> None:
    """Rows seed inserts have no vector, and must not be invisible because of it.

    Every migration query used to run *through* memory_embeddings with an
    inner join, so a memory with no embedding row was in no count and no
    target list: doctor reported healthy and migrate-engine found nothing,
    leaving the memory permanently FTS-only under bloom. Before the crash fix
    the insert failed loudly, so this hole only opened once seed could write.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("bloom1", "bloom wrote this"))
    assert stale_stats(db, BloomEngine.model_id).needs_migration == 0

    seed = SeedEngine(db_path=db)
    for i in range(3):
        seed.ingest(_memory(f"seed{i}", f"seed wrote this {i}"))

    stats = stale_stats(db, BloomEngine.model_id)
    assert stats.missing == 3  # what doctor now warns about
    assert stats.needs_migration == 3
    assert stats.compatible == 1
    filters = MigrateFilters()
    assert count_targets(db, BloomEngine.model_id, filters) == 3
    assert [mid for mid, _ in list_targets(db, BloomEngine.model_id, filters)] == ["seed0", "seed1", "seed2"]

    # migrate-engine embeds them for real: an UPDATE would have matched no row.
    assert run_migrate(_bloom(db), db, filters) == 3
    assert sorted(_embedding_ids(db)) == ["bloom1", "seed0", "seed1", "seed2"]
    assert set(_model_ids(db).values()) == {BloomEngine.model_id}
    assert stale_stats(db, BloomEngine.model_id).needs_migration == 0

    # And they now reach bloom's vector channel, not just FTS.
    assert "seed1" in {r.memory.id for r in _bloom(db).retrieve("seed wrote this 1", limit=10)}


def test_migrate_targets_respect_filters_for_embedding_less_rows(tmp_path: Path) -> None:
    """The new anti-join must not bypass the project/type scoping."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("bloom1", "bloom wrote this"))
    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("keep", "in scope", project="p1"))
    seed.ingest(_memory("skip", "out of scope", project="other"))

    filters = MigrateFilters(project="p1")
    assert [mid for mid, _ in list_targets(db, BloomEngine.model_id, filters)] == ["keep"]
    assert count_targets(db, BloomEngine.model_id, filters) == 1


def test_seed_content_edit_invalidates_vector_on_half_migrated_store(tmp_path):
    """A store with memory_embeddings but no enriched_content (bloom died mid-migration):
    a seed content edit must still invalidate the parent vector so migrate/doctor can repair it."""
    import sqlite3

    from poppy.engine.seed import SeedEngine

    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_memory("m1", "alpha secret"))
    # Attach a memory_embeddings row tagged with a live-looking model_id, but do
    # NOT add enriched_content (the half-migrated shape).
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL, model_id TEXT)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
        ("m1", np.array([1.0], dtype=np.float32).tobytes(), "real-bi-encoder-v1"),
    )
    conn.commit()
    assert "enriched_content" not in {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    conn.close()

    SeedEngine(db_path=db).ingest(_memory("m1", "beta replacement"))

    conn = sqlite3.connect(str(db))
    model_id = conn.execute("SELECT model_id FROM memory_embeddings WHERE id = ?", ("m1",)).fetchone()[0]
    conn.close()
    assert model_id != "real-bi-encoder-v1", "the stale vector was not invalidated"


def test_seed_delete_removes_the_parent_embedding_so_a_reinsert_cannot_inherit_it(tmp_path):
    """Seed delete drops the parent vector so a later insert cannot inherit a deleted memory's embedding."""
    import sqlite3

    from poppy.engine.seed import SeedEngine

    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("m1", "the deleted text"))
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT 1 FROM memory_embeddings WHERE id = ?", ("m1",)).fetchone()
    conn.close()

    assert SeedEngine(db_path=db).delete("m1") is True

    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT 1 FROM memory_embeddings WHERE id = ?", ("m1",)).fetchone() is None
    conn.close()

    # A fresh seed insert of the same id starts with no inherited vector.
    SeedEngine(db_path=db).ingest(_memory("m1", "unrelated new text"))
    conn = sqlite3.connect(str(db))
    assert conn.execute("SELECT 1 FROM memory_embeddings WHERE id = ?", ("m1",)).fetchone() is None
    conn.close()


def test_seed_content_edit_clears_the_bloom_speaker_copies(tmp_path: Path) -> None:
    """A redaction on the fallback engine must not leave the old text in a closet.

    Previously, seed rewrote only the parent row, so the
    default engine's per-speaker copies kept the removed text and came back the
    moment the user switched engines back.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "obsoletephoenix"))
    assert len(_closet_ids(db, "m1")) == 2

    SeedEngine(db_path=db).ingest(_memory("m1", "current replacement"))

    assert _closet_ids(db, "m1") == []
    assert _closet_ids(db, "m1", table="memory_embeddings") == []
    assert _model_ids(db)["m1"] == SEED_INVALIDATED_MODEL_ID
    # Nothing left holding the redacted phrase, under either engine.
    assert not _bloom(db).retrieve("obsoletephoenix", limit=10)
    assert not SeedEngine(db_path=db).retrieve("obsoletephoenix", limit=10)


def test_seed_delete_clears_the_bloom_speaker_copies(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "obsoletephoenix"))
    assert len(_closet_ids(db, "m1")) == 2

    assert SeedEngine(db_path=db).delete("m1") is True

    assert _closet_ids(db, "m1") == []
    assert _closet_ids(db, "m1", table="memory_embeddings") == []
    assert not SeedEngine(db_path=db).retrieve("obsoletephoenix", limit=10)


def test_seed_sweeps_an_orphaned_closet_on_a_repeat_delete(tmp_path: Path) -> None:
    """A closet whose parent is already gone is still reachable by a delete of that id."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "obsoletephoenix"))
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = ?", ("m1",))  # parent lost, closets orphaned
    conn.commit()
    conn.close()
    assert len(_closet_ids(db, "m1")) == 2

    seed = SeedEngine(db_path=db)
    assert seed.delete("m1") is False  # no parent row to remove
    assert _closet_ids(db, "m1") == []  # but the orphans are swept


def test_seed_list_all_hides_closets_so_they_never_reach_sync(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "hello"))
    assert len(_closet_ids(db, "m1")) == 2

    assert [m.id for m in SeedEngine(db_path=db).list_all()] == ["m1"]
    assert [m.id for m in _bloom(db).list_all()] == ["m1"]


def test_seed_metadata_only_edit_still_preserves_the_speaker_copies(tmp_path: Path) -> None:
    """Only a CHANGED body is a redaction. A project edit must keep the closets."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_multi_speaker("m1", "hello"))
    closets_before = _closet_ids(db, "m1")
    assert len(closets_before) == 2

    seed = SeedEngine(db_path=db)
    same_content = seed.get("m1").content
    seed.ingest(_memory("m1", same_content, project="p2"))

    assert seed.get("m1").project == "p2"
    assert _closet_ids(db, "m1") == closets_before
