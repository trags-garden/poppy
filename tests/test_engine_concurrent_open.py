"""Opening one store from several processes at once, while it still owes upgrades.

Both engines write at construction: they run ``executescript(SCHEMA)`` and then
the open-time migrations (``expires_at``, ``enriched_content``, the embedding
``model_id`` column, the timestamp rewrite). Each is idempotent on its own, but
"check, then ALTER" is not atomic across processes: two openers can both see a
column missing and both try to add it, and the loser gets ``duplicate column
name`` or, once the writers pile up, ``database is locked``.

The realistic trigger is the first open after an upgrade, when the CLI, the MCP
server, the daemon and the dashboard all reach a store that has not applied its
new migrations yet and start together.

So the store built here is *upgrade-shaped*: the schema an older Poppy left
behind, missing the columns the migrations add, with rows whose timestamps still
need rewriting. Then N real processes open it at the same instant, held at a
file barrier so they collide inside the constructor instead of queueing politely
behind each other's interpreter startup. Threads would not do: the race is
between processes, and one process's lock and shared connection hide it.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import poppy
from poppy.engine.seed import SCHEMA as SEED_SCHEMA

# Enough openers to lose the race reliably on an unfixed tree, without making CI
# wait on a crowd of interpreter startups.
PROCESSES = 8

# How long a child may wait at the barrier, and the parent for a whole round.
BARRIER_TIMEOUT_S = 60.0
ROUND_TIMEOUT_S = 120.0

# The ``memories`` table as Poppy left it before the lifecycle work added
# ``expires_at`` -- derived from the live schema so the fixture cannot drift away
# from what the migration actually looks for.
_LEGACY_SCHEMA = SEED_SCHEMA.replace(",\n    expires_at TEXT\n", "\n")

# Present but without ``model_id``, so bloom's embedding migration has an ALTER
# to run too. Same shape the bloom schema creates today.
_LEGACY_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


def _legacy_rows() -> list[tuple]:
    """Rows whose timestamps are naive local text, so the rewrite has work to do."""
    rows = []
    for i in range(25):
        stamp = f"2026-01-0{(i % 9) + 1} 09:00:00"
        rows.append((f"m{i}", f"legacy memory {i}", "fact", "proj", "cli", None, stamp, 1.0, "[]", stamp, stamp))
    return rows


# Runs in a child interpreter. argv: db, engine, ready file, go file, timeout.
#
# Every child announces itself and then spins on the go file, so the parent can
# release all of them within a few milliseconds of each other. The reads at the
# end prove the migrations this child depends on actually landed, not merely
# that the constructor returned.
_CHILD = r"""
import sys, time
from pathlib import Path

db, engine, ready, go = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])

if engine == "seed":
    from poppy.engine.seed import SeedEngine as Engine
else:
    from poppy.engine._hybrid import HybridEngine

    class Engine(HybridEngine):
        # Tags the BLOBs this engine would write. The constructor wants it set;
        # no embedding runtime is touched because nothing is embedded here.
        model_id = "test-model"

ready.write_text("up")
deadline = time.monotonic() + float(sys.argv[5])
while not go.exists():
    if time.monotonic() > deadline:
        raise SystemExit("barrier timed out")
    time.sleep(0.002)

eng = Engine(db)
try:
    eng._conn.execute("SELECT expires_at FROM memories LIMIT 1").fetchone()
    if engine != "seed":
        eng._conn.execute("SELECT enriched_content FROM memories LIMIT 1").fetchone()
        eng._conn.execute("SELECT model_id FROM memory_embeddings LIMIT 1").fetchone()
finally:
    eng._conn.close()
"""


def _build_upgrade_shaped_store(db_path: Path) -> None:
    """Write a store in the state an older Poppy would have left it in."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_LEGACY_SCHEMA)
        conn.executescript(_LEGACY_EMBEDDINGS)
        conn.executemany(
            "INSERT INTO memories (id, content, memory_type, project, source_type,"
            " source_session_id, source_timestamp, confidence, related_to, created_at,"
            " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _legacy_rows(),
        )
        conn.commit()
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    finally:
        conn.close()
    assert "expires_at" not in cols, "fixture store is not upgrade-shaped: expires_at is already present"
    assert "enriched_content" not in cols


def _child_env(store_dir: Path) -> dict[str, str]:
    """A child environment pinned to the scratch store, never the real one."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(poppy.__file__).resolve().parents[1])
    env["POPPY_DIR"] = str(store_dir)
    env["HOME"] = str(store_dir)
    return env


def open_store_concurrently(store_dir: Path, engine: str, processes: int = PROCESSES) -> None:
    """Open one upgrade-shaped store from ``processes`` real processes at once.

    Raises ``AssertionError`` naming every child that failed, with the last line
    of its stderr, so a regression reads as the error it actually was.
    """
    db_path = store_dir / "memories.db"
    _build_upgrade_shaped_store(db_path)

    go = store_dir / "go"
    children = []
    for i in range(processes):
        ready = store_dir / f"ready-{i}"
        argv = [sys.executable, "-c", _CHILD, str(db_path), engine, str(ready), str(go), str(BARRIER_TIMEOUT_S)]
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(store_dir),
        )
        children.append((ready, proc))

    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    for ready, proc in children:
        while not ready.exists():
            assert proc.poll() is None, f"a child died before the barrier: {proc.communicate()[1]}"
            assert time.monotonic() < deadline, "a child never reached the barrier"
            time.sleep(0.002)

    go.write_text("go")

    failures = []
    for _, proc in children:
        try:
            _, err = proc.communicate(timeout=ROUND_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            failures.append("a child hung while opening the store")
            continue
        if proc.returncode != 0:
            failures.append(err.strip().splitlines()[-1] if err.strip() else f"exit {proc.returncode}")

    assert not failures, (
        f"{len(failures)}/{processes} {engine} openers failed on an upgrade-shaped store: " + " | ".join(failures)
    )


@pytest.mark.parametrize("engine", ["seed", "bloom"])
def test_concurrent_open_of_an_upgrade_shaped_store(tmp_path, engine):
    """Every opener gets through the open-time schema and migration block."""
    store_dir = tmp_path / engine
    store_dir.mkdir()
    open_store_concurrently(store_dir, engine)
