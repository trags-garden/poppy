"""Tests for LLM-assisted conflict detection on remember writes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from poppy.capture.reconciler import (
    AUTO_SUPERSEDE_THRESHOLD,
    CONFLICT_LLM_TIMEOUT_S,
    detect_conflicts,
    find_candidates,
    parse_llm_response,
    pick_auto_supersede,
)
from poppy.config import PoppyConfig
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source


def _mk(mid: str, content: str, *, project: str | None = "poppy", memory_type: str = "decision") -> Memory:
    n = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=content,
        memory_type=memory_type,
        source=Source(type="manual", session_id=None, timestamp=n),
        project=project,
        related_to=[],
        created_at=n,
        updated_at=n,
    )


@pytest.fixture
def engine(tmp_path: Path) -> SeedEngine:
    return SeedEngine(db_path=tmp_path / "memories.db")


def test_find_candidates_returns_same_project_and_type(engine: SeedEngine) -> None:
    engine.ingest(_mk("a", "use all-MiniLM for embeddings"))
    engine.ingest(_mk("b", "use bge-small for embeddings", project="other"))  # wrong project
    engine.ingest(_mk("c", "test runner is pytest", memory_type="fact"))  # wrong type

    new = _mk("new", "use bge-large for embeddings instead")
    cands = find_candidates(engine, new, top_k=5, min_score=0.0)
    cand_ids = {c.memory.id for c in cands}
    assert "a" in cand_ids
    assert "b" not in cand_ids
    assert "c" not in cand_ids


def test_find_candidates_excludes_self(engine: SeedEngine) -> None:
    new = _mk("self", "use all-MiniLM")
    engine.ingest(new)

    cands = find_candidates(engine, new, top_k=5, min_score=0.0)
    assert all(c.memory.id != "self" for c in cands)


def test_detect_conflicts_short_circuits_when_no_candidates(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty engine = no candidates = no LLM call."""
    called = []

    def fake_call_llm(prompt, *, transcript_path, cfg, **kwargs):
        called.append(prompt)
        return []

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    new = _mk("new", "any content")
    cfg = PoppyConfig()

    conflicts = detect_conflicts(engine, new, cfg=cfg)
    assert conflicts == []
    assert called == [], "call_llm must not run when no candidates exist"


def test_detect_conflicts_filters_to_known_ids(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM may hallucinate ids; we drop entries that aren't actual candidates."""
    engine.ingest(_mk("a", "use all-MiniLM"))

    def fake_call_llm(prompt, *, transcript_path, cfg, **kwargs):
        # Returns one real id and one bogus id.
        return [
            {"id": "a", "confidence": 0.9, "reason": "explicit replacement"},
            {"id": "ghost", "confidence": 0.95, "reason": "made up"},
        ]

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    new = _mk("new", "use bge-large")
    cfg = PoppyConfig()

    conflicts = detect_conflicts(engine, new, cfg=cfg)
    assert [c.memory.id for c in conflicts] == ["a"]
    assert conflicts[0].confidence == 0.9


@pytest.mark.parametrize("backend", ["host", "remote", "fallback"])
@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "Verdicts: {}"])
def test_detect_conflicts_parses_backend_text(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, backend: str, wrapper: str
) -> None:
    """A verdict survives the real parser, from either backend, however it is wrapped.

    Stubs the backend at the subprocess / HTTP boundary and hands back a raw
    string, so the production parse path runs for real. Stubbing the layer above
    it would hide a parser that drops every verdict.
    """
    engine.ingest(_mk("a", "use all-MiniLM"))
    raw = wrapper.format(
        json.dumps(
            [
                {"id": "a", "confidence": 0.9, "reason": "explicit replacement"},
                {"id": "ghost", "confidence": 0.95, "reason": "unknown memory"},
                "invalid entry",
            ]
        )
    )
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/claude" if backend != "remote" and name == "claude" else None
    )
    calls = []

    def host(prompt, *, cli, timeout_s, **kwargs):
        assert cli == "claude"
        # A verdict is a small prompt; it must not book the whole-transcript budget.
        assert timeout_s == CONFLICT_LLM_TIMEOUT_S
        calls.append("host")
        return raw if backend == "host" else "invalid JSON"

    def remote(prompt, **kwargs):
        calls.append("remote")
        return raw

    monkeypatch.setattr("poppy.consolidation.call_host_cli", host)
    monkeypatch.setattr("poppy.consolidation.call_openai_compat", remote)
    cfg = PoppyConfig(consolidate_model="test-model", consolidate_api_key="test-key")
    conflicts = detect_conflicts(engine, _mk("new", "use bge-large"), cfg=cfg)

    assert [c.memory.id for c in conflicts] == ["a"]
    assert conflicts[0].confidence == 0.9
    assert conflicts[0].reason == "explicit replacement"
    assert calls == {"host": ["host"], "remote": ["remote"], "fallback": ["host", "remote"]}[backend]


def test_conflict_verdict_budget_fits_inside_the_capture_lock() -> None:
    """A whole capture batch must finish before its lock is treated as abandoned.

    A capture worker holds the per-session lock across one extraction plus a
    verdict for every candidate it extracted. If that can outlast the stale-lock
    window, a second worker steals the lock from a live one and both read the
    same turns.
    """
    from poppy.capture.lock import LOCK_TTL_S
    from poppy.consolidation import HOST_CLI_TIMEOUT_S

    default_batch = 5  # resolved_consolidate_settings().max_items
    assert HOST_CLI_TIMEOUT_S + default_batch * CONFLICT_LLM_TIMEOUT_S < LOCK_TTL_S


def test_detect_conflicts_clamps_confidence(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    engine.ingest(_mk("a", "use all-MiniLM"))

    def fake_call_llm(prompt, *, transcript_path, cfg, **kwargs):
        return [{"id": "a", "confidence": 1.7, "reason": "high"}]

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    new = _mk("new", "use bge-large")
    conflicts = detect_conflicts(engine, new, cfg=PoppyConfig())
    assert conflicts[0].confidence == 1.0


def test_detect_conflicts_handles_invalid_confidence(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    engine.ingest(_mk("a", "use all-MiniLM"))

    def fake_call_llm(prompt, *, transcript_path, cfg, **kwargs):
        return [{"id": "a", "confidence": "high", "reason": "bad type"}]

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    new = _mk("new", "use bge-large")
    conflicts = detect_conflicts(engine, new, cfg=PoppyConfig())
    assert conflicts == []


def test_pick_auto_supersede_requires_unique_high_confidence() -> None:
    from poppy.capture.reconciler import Conflict

    a = Conflict(memory=_mk("a", "x"), confidence=0.9, reason="r")
    b = Conflict(memory=_mk("b", "y"), confidence=0.92, reason="r")
    low = Conflict(memory=_mk("c", "z"), confidence=0.5, reason="r")

    # Two above threshold → ambiguous, return None.
    assert pick_auto_supersede([a, b]) is None
    # Single above threshold → that one.
    picked = pick_auto_supersede([a, low])
    assert picked is not None and picked.memory.id == "a"
    # All below → None.
    assert pick_auto_supersede([low]) is None
    # Empty → None.
    assert pick_auto_supersede([]) is None


def test_threshold_is_at_least_85_percent() -> None:
    """Auto-supersede must remain conservative — guard against drift."""
    assert AUTO_SUPERSEDE_THRESHOLD >= 0.80


def test_parse_llm_response_strips_markdown_fence() -> None:
    raw = '```json\n[{"id": "a", "confidence": 0.9}]\n```'
    parsed = parse_llm_response(raw)
    assert parsed == [{"id": "a", "confidence": 0.9}]


def test_parse_llm_response_extracts_array_from_prose() -> None:
    raw = 'I think the answer is [{"id": "a", "confidence": 0.9}] based on...'
    parsed = parse_llm_response(raw)
    assert parsed == [{"id": "a", "confidence": 0.9}]


def test_parse_llm_response_returns_empty_on_garbage() -> None:
    assert parse_llm_response("not json at all") == []
    assert parse_llm_response("") == []


def test_remember_check_conflicts_does_not_write(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI --check-conflicts must run detection and exit without ingest."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    # Pre-populate.
    engine.ingest(_mk("existing", "use all-MiniLM"))

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.cli.main._get_engine", lambda: engine)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(
        "poppy.consolidation.call_host_cli",
        lambda prompt, *, cli, **kwargs: '[{"id": "existing", "confidence": 0.9, "reason": "explicit replacement"}]',
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "remember",
            "use bge-large",
            "--type",
            "decision",
            "--project",
            "poppy",
            "--check-conflicts",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "conflict candidate" in result.output
    assert "existing" in result.output
    assert "conf=0.90" in result.output
    assert "explicit replacement" in result.output
    # Engine still has only the original — nothing written.
    all_mems = engine.list_all(limit=50)
    assert {m.id for m in all_mems} == {"existing"}


def test_remember_auto_supersede_replaces_existing(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI --auto-supersede tombstones the existing high-confidence match."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    engine.ingest(_mk("existing", "use all-MiniLM"))

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.cli.main._get_engine", lambda: engine)
    monkeypatch.setattr("poppy.cli.main._get_poppy_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg, **kwargs: [
            {"id": "existing", "confidence": 0.91, "reason": "replaces"}
        ],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["remember", "use bge-large now", "--type", "decision", "--project", "poppy", "--auto-supersede"],
    )
    assert result.exit_code == 0, result.output
    assert "auto-supersede" in result.output
    assert "supersedes existing" in result.output

    # Existing is gone from the engine.
    assert engine.get("existing") is None
    # And there's a new memory pointing back at it.
    all_mems = engine.list_all(limit=50)
    assert len(all_mems) == 1
    assert "existing" in all_mems[0].related_to


def test_remember_default_off_skips_detection(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """auto_supersede=off (default) means zero LLM calls on plain remember."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    engine.ingest(_mk("existing", "use all-MiniLM"))

    called = []

    def boom(prompt, *, transcript_path, cfg, **kwargs):
        called.append(prompt)
        return []

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.cli.main._get_engine", lambda: engine)
    monkeypatch.setattr("poppy.cli.main._get_poppy_dir", lambda: tmp_path)
    monkeypatch.setattr("poppy.consolidation.call_llm", boom)

    runner = CliRunner()
    result = runner.invoke(cli, ["remember", "use bge-large now", "--type", "decision"])
    assert result.exit_code == 0, result.output
    assert called == [], "default-off mode must never invoke the LLM"
