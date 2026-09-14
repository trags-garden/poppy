"""The UI must survive another process opening and writing the same store.

`poppy ui` keeps its reader and its tombstone store open for the life of the
process. A second Poppy process (`poppy list`, `poppy remember`) opening and
closing its own connection used to unlink the store's -wal/-shm out from under
those long-lived connections, after which every UI read failed with
"disk I/O error" / "database disk image is malformed" until the UI was restarted.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from poppy.db import connect
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source

# The only part of `poppy list` that matters here: a separate process that opens
# the store and closes it. Stdlib-only so the round trip is an interpreter start
# rather than a Poppy import, which keeps the concurrency loop below quick.
_SQLITE_PROBE = (
    "import sqlite3,sys;"
    "c=sqlite3.connect(sys.argv[1]);"
    "c.execute('PRAGMA journal_mode = WAL');"
    "c.execute('SELECT COUNT(*) FROM memories').fetchone();"
    "c.close()"
)

# Stands in for `poppy list` followed by `poppy remember`: a real second process
# that opens a Poppy connection and closes it, then opens one and writes.
_OTHER_PROCESS = """
import sys
from datetime import datetime, timezone
from pathlib import Path

from poppy.db import connect
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source

db = Path(sys.argv[1])

reader = connect(db)                      # `poppy list`
reader.execute("SELECT COUNT(*) FROM memories").fetchone()
reader.close()

writer = SeedEngine(db_path=db)           # `poppy remember`
now = datetime.now(timezone.utc)
writer.ingest(
    Memory(
        id="from-other-process",
        content="written by another process",
        memory_type="fact",
        source=Source(type="manual", session_id=None, timestamp=now),
        project=None,
        related_to=[],
        created_at=now,
        updated_at=now,
    )
)
writer._conn.close()
"""


def _ingest(engine: SeedEngine, mid: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    memory = Memory(
        id=mid,
        content=content,
        memory_type="fact",
        source=Source(type="manual", session_id=None, timestamp=now),
        project=None,
        related_to=[],
        created_at=now,
        updated_at=now,
    )
    engine.ingest(memory)
    return memory


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The UI wired to SeedEngine for both reader and writer, so no models load.

    Reader and writer are separate connections, and `create_app` opens a third
    for the tombstone store -- the same multi-connection shape the real UI has,
    which is what the bug needed.
    """
    db_path = tmp_path / "memories.db"
    reader = SeedEngine(db_path=db_path)
    writer = SeedEngine(db_path=db_path)

    from poppy.ui import server as ui_server

    monkeypatch.setattr(ui_server, "get_fast_engine", lambda _d: reader)
    monkeypatch.setattr(ui_server, "get_engine", lambda _d: writer)
    app = ui_server.create_app(poppy_dir=tmp_path)
    # TrustedHostMiddleware only accepts loopback hosts.
    return TestClient(app, base_url="http://localhost")


def _run_other_process(db_path: Path, poppy_dir: Path) -> None:
    env = dict(os.environ)
    env.update(POPPY_DIR=str(poppy_dir), POPPY_SUPPRESS_HOOKS="1", POPPY_TELEMETRY_OFF="1")
    result = subprocess.run(
        [sys.executable, "-c", _OTHER_PROCESS, str(db_path)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_reader_survives_a_second_process_writing_the_store(app_client: TestClient, tmp_path: Path) -> None:
    """Stats and restore keep working after another process touches the store.

    Seeded well past a single database page: a stale wal-index only misdirects a
    read once the store spans enough pages for the reader to consult it, which is
    why a three-row store hid this bug.
    """
    db_path = tmp_path / "memories.db"
    seeded = SeedEngine(db_path=db_path)
    for i in range(60):
        _ingest(seeded, f"m{i}", f"probe memory {i} " + "padding " * 20)

    assert app_client.get("/api/stats").status_code == 200

    _run_other_process(db_path, tmp_path)

    # The UI's own write is what creates the fresh -wal/-shm pair that the stale
    # readers used to choke on, so delete before asserting on the reads.
    assert app_client.delete("/api/memories/m0").status_code == 200
    assert app_client.post("/api/memories/m0/restore").status_code == 200
    assert app_client.get("/api/stats").status_code == 200
    assert app_client.get("/api/memories").status_code == 200


def test_second_connection_does_not_unlock_the_store(tmp_path: Path) -> None:
    """Opening a second connection must not let another process delete the sidecars.

    The root cause: `connect` sniffs the file header with a plain open/close, and
    closing any descriptor on an inode drops every POSIX lock the process holds on
    it -- so the second connect silently released the first connection's lock.
    """
    db_path = tmp_path / "memories.db"
    first = SeedEngine(db_path=db_path)
    _ingest(first, "m0", "probe memory")
    second = SeedEngine(db_path=db_path)
    assert second.stats().memory_count == 1

    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    assert wal.exists() and shm.exists()

    _run_other_process(db_path, tmp_path)

    assert wal.exists(), "another process checkpointed and unlinked the -wal under a live connection"
    assert shm.exists(), "another process unlinked the -shm under a live connection"


def test_concurrent_connects_never_unlock_the_store(tmp_path: Path) -> None:
    """Threads racing to connect from a cold cache must not unlock the store.

    Caching the sniff is not enough on its own: with nothing registered yet, every
    racing thread sniffs, and the slowest one's close lands after a faster one's
    connection is already open and holding locks. Each round below opens several
    connections at once, keeps them all live, and lets a separate process try to
    checkpoint -- which it can only do if this process has lost its locks.
    """
    db_path = tmp_path / "memories.db"
    seeded = SeedEngine(db_path=db_path)
    for i in range(5):
        _ingest(seeded, f"m{i}", f"probe memory {i} " + "padding " * 20)
    seeded._conn.close()  # go cold: nothing registered, so every thread will sniff

    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    lost: list[int] = []

    for round_no in range(15):
        barrier = threading.Barrier(8)
        opened: list[sqlite3.Connection] = []
        failures: list[str] = []
        guard = threading.Lock()

        def worker() -> None:
            try:
                barrier.wait()
                conn = connect(db_path)
                conn.execute("SELECT COUNT(*) FROM memories").fetchone()
                with guard:
                    opened.append(conn)
            except Exception as exc:  # noqa: BLE001 - reported through `failures`
                with guard:
                    failures.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures, failures

        # Every connection is open and has read, so the sidecars must survive.
        subprocess.run([sys.executable, "-c", _SQLITE_PROBE, str(db_path)], check=True, capture_output=True)
        if not wal.exists() or not shm.exists():
            lost.append(round_no)

        for conn in opened:
            conn.close()

    assert not lost, f"store was left unlocked in rounds {lost}: another process unlinked -wal/-shm"
