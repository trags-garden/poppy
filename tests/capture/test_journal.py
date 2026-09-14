"""Tests for the capture journal's host/source field."""

from __future__ import annotations

import json
from pathlib import Path

from poppy.capture import journal


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
