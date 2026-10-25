"""Tests for the capture journal's host/source field."""

from __future__ import annotations

import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import SimpleNamespace

import pytest

from poppy.capture import _state, journal


def test_record_removes_legacy_previews_and_stores_only_types(tmp_path):
    text = "distinctive private capture text"
    path = tmp_path / journal.JOURNAL_FILENAME
    legacy = {"session_id": "old", "count": 1, "items": [{"type": "fact", "preview": text}]}
    path.write_text(json.dumps(legacy) + "\n")
    path.chmod(0o644)

    rec = journal.record(
        tmp_path,
        session_id="new",
        project="poppy",
        count=1,
        items=[SimpleNamespace(memory_type="decision", content=text)],
    )

    assert text not in path.read_text()
    assert "preview" not in path.read_text()
    assert rec.items == [{"type": "decision"}]
    rows = journal.read_all(tmp_path)
    assert rows[0] == {**legacy, "items": [{"type": "fact"}]}
    assert rows[1]["count"] == 1
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_failed_atomic_record_preserves_previous_journal(tmp_path, monkeypatch):
    journal.record(tmp_path, session_id="old", project=None, count=1, items=[])
    path = tmp_path / journal.JOURNAL_FILENAME
    original = path.read_bytes()

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("poppy.paths.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        journal.record(tmp_path, session_id="new", project=None, count=2, items=[])

    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path, tmp_path / journal.LOCK_FILENAME}


def test_strip_previews_tolerates_malformed_records(tmp_path):
    path = tmp_path / journal.JOURNAL_FILENAME
    legacy = {"count": 1, "items": [None, {"type": "fact", "preview": "forgotten text"}]}
    path.write_text('broken line\n[]\n{"items": null}\n' + json.dumps(legacy) + "\n")

    journal.strip_previews(tmp_path)

    assert "forgotten text" not in path.read_text()
    assert journal.read_all(tmp_path) == [{"items": []}, {"count": 1, "items": [{"type": "fact"}]}]


@pytest.mark.skipif(_state.fcntl is None, reason="flock requires POSIX")
@pytest.mark.parametrize("rewrite", ["strip", "record"])
def test_record_during_rewrite_preserves_both_records(tmp_path, monkeypatch, rewrite):
    path = tmp_path / journal.JOURNAL_FILENAME
    legacy = {"session_id": "old", "count": 1, "items": [{"type": "fact", "preview": "private text"}]}
    path.write_text(json.dumps(legacy) + "\n")
    real_read_all = journal.read_all
    main_thread = threading.current_thread()
    started = threading.Event()
    captures = []

    def capture():
        started.set()
        journal.record(tmp_path, session_id="new", project=None, count=1, items=[])

    with ThreadPoolExecutor(max_workers=1) as pool:

        def read_during_capture(poppy_dir):
            rows = real_read_all(poppy_dir)
            if threading.current_thread() is main_thread:
                future = pool.submit(capture)
                captures.append(future)
                assert started.wait(5)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.2)
            return rows

        monkeypatch.setattr(journal, "read_all", read_during_capture)
        if rewrite == "strip":
            journal.strip_previews(tmp_path)
        else:
            journal.record(tmp_path, session_id="middle", project=None, count=1, items=[])
        captures[0].result(timeout=5)

    rows = real_read_all(tmp_path)
    expected = ["old", "new"] if rewrite == "strip" else ["old", "middle", "new"]
    assert [row["session_id"] for row in rows] == expected
    assert "preview" not in path.read_text()


def test_record_round_trips_source(tmp_path: Path) -> None:
    journal.record(tmp_path, session_id="s1", project="poppy", count=2, items=[], source="codex")
    rec = journal.read_last(tmp_path)
    assert rec is not None
    assert rec.source == "codex"


def test_read_last_source_filter_skips_legacy_and_other_hosts(tmp_path: Path) -> None:
    # A legacy line written before the source field existed (no "source" key).
    legacy = {"ts": "2026-07-01T00:00:00+00:00", "session_id": "old", "project": None, "count": 1, "items": []}
    path = tmp_path / journal.JOURNAL_FILENAME
    path.write_text(json.dumps(legacy) + "\n")
    # A claude-code capture is the most recent, but is not a codex record.
    journal.record(tmp_path, session_id="cc", project="poppy", count=5, items=[], source="claude-code")

    # Legacy lines still parse for the unfiltered read.
    assert journal.read_last(tmp_path).source == "claude-code"
    # No codex record yet: the host-filtered read returns None (does not read the
    # legacy line or the claude-code capture).
    assert journal.read_last(tmp_path, source="codex") is None

    journal.record(tmp_path, session_id="cx", project="poppy", count=3, items=[], source="codex")
    codex_rec = journal.read_last(tmp_path, source="codex")
    assert codex_rec is not None and codex_rec.count == 3
