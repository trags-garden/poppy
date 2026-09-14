"""The shared SQLite connect helper must enable WAL + a busy timeout."""

from poppy.db import BUSY_TIMEOUT_MS, connect


def test_connect_enables_wal_and_busy_timeout(tmp_path):
    conn = connect(tmp_path / "x.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
        # NORMAL (=1) is the recommended durability level under WAL.
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    finally:
        conn.close()


def test_connect_accepts_str_path(tmp_path):
    conn = connect(str(tmp_path / "y.db"))
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_concurrent_reader_during_open_writer(tmp_path):
    """In rollback-journal mode this raised 'database is locked'; WAL allows it."""
    db = tmp_path / "z.db"
    writer = connect(db)
    reader = connect(db)
    try:
        writer.execute("CREATE TABLE t (x INTEGER)")
        writer.commit()
        writer.execute("BEGIN")
        writer.execute("INSERT INTO t VALUES (1)")  # uncommitted; writer holds lock
        # Reader sees the last committed state without erroring.
        assert reader.execute("SELECT count(*) FROM t").fetchone()[0] == 0
        writer.commit()
        assert reader.execute("SELECT count(*) FROM t").fetchone()[0] == 1
    finally:
        writer.close()
        reader.close()
