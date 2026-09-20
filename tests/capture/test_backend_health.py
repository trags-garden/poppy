"""Backend health: a host CLI that is on PATH but fails every extraction.

The regression: with consent granted and a logged-out `claude` on PATH, every
capture failed while the SessionStart banner said "active" and `poppy doctor`
said OK. The end-to-end test below stubs `claude` with a script that exits 1 and
drives the real SessionEnd capture path, then asserts the banner degrades to
INACTIVE naming the backend and the doctor's `auto-capture` line is a WARN
carrying the stderr tail.

Two follow-ups from the review of that fix are covered here too: failures are
counted per CLI, so two broken backends used in alternation cannot hide each
other from the threshold, and concurrent writers publish through temporary files
of their own, so neither can splice bytes into the other's record.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.capture import health
from poppy.capture.banner import is_user_facing_status, render_banner
from poppy.capture.policy import CaptureStatus, evaluate, is_capture_enabled
from poppy.capture.reconciler import detect_conflicts
from poppy.capture.redaction import MASK
from poppy.cli.hooks import _session_banner
from poppy.cli.main import cli
from poppy.config import PoppyConfig
from poppy.consolidation import call_llm, consolidate_stop_event
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source

STUB_STDERR = "Invalid API key. Please run /login"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("POPPY_CONSOLIDATE", "POPPY_CONSOLIDATE_MODEL", "POPPY_CONSOLIDATE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- health record ---------------------------------------------------------


def test_failures_accumulate_until_the_threshold(tmp_path: Path) -> None:
    for _ in range(health.FAILURE_THRESHOLD - 1):
        health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)
        assert not health.backend_failing(tmp_path)
    health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)
    assert health.backend_failing(tmp_path)
    assert health.load(tmp_path).cli == "claude"


def test_success_clears_the_record(tmp_path: Path) -> None:
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)
    health.record_success(poppy_dir=tmp_path)
    assert health.load(tmp_path).consecutive_failures == 0
    assert not health.backend_failing(tmp_path)


def test_each_cli_keeps_its_own_count(tmp_path: Path) -> None:
    """A second backend failing must not erase the first one's history."""
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)
    health.record_failure("codex", "exited rc=1", poppy_dir=tmp_path)
    recorded = health.load(tmp_path)
    assert recorded.clis["claude"].consecutive_failures == health.FAILURE_THRESHOLD
    assert recorded.clis["codex"].consecutive_failures == 1
    # The warning names the backend that actually crossed the threshold.
    assert recorded.failing
    assert recorded.cli == "claude"


def test_success_clears_only_that_cli(tmp_path: Path) -> None:
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)
        health.record_failure("codex", "exited rc=1", poppy_dir=tmp_path)
    health.record_success("claude", poppy_dir=tmp_path)
    recorded = health.load(tmp_path)
    assert "claude" not in recorded.clis
    assert recorded.cli == "codex"
    assert recorded.failing


def test_a_non_numeric_count_reads_healthy(tmp_path: Path) -> None:
    """A hand-corrupted record must not escape as a ValueError (load never raises)."""
    health.health_path(tmp_path).write_text(json.dumps({"clis": {"claude": {"consecutive_failures": "lots"}}}))
    assert health.load(tmp_path).consecutive_failures == 0
    assert health.backend_failing(tmp_path) is False


def test_non_utf8_bytes_read_healthy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UnicodeDecodeError is a ValueError, not an OSError — it used to propagate."""
    health.health_path(tmp_path).write_bytes(b"\xff\xfe not utf-8 at all")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    assert health.load(tmp_path).consecutive_failures == 0
    assert health.backend_failing(tmp_path) is False
    # The corruption stops at load(): the policy the hooks and doctor call is fine.
    assert evaluate(PoppyConfig(consent="granted"), host_cli=True) is CaptureStatus.ACTIVE
    # And the next failure simply starts a fresh record over the garbage.
    health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)
    assert health.load(tmp_path).consecutive_failures == 1


def test_concurrent_writers_never_tear_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two writers publishing at once must not splice bytes into each other's record.

    The interleaving that a shared temp name could not survive: writer A stages
    its record and stalls at the publish, writer B stages and publishes a longer
    one, then A publishes. Sharing one `.tmp` path put A's bytes inside the file
    B had just published, leaving trailing garbage that read back as no failures
    at all.
    """
    real_replace = os.replace
    staged: list[str] = []
    a_staged = threading.Event()
    b_published = threading.Event()

    def scheduled_replace(src, dst):  # type: ignore[no-untyped-def]
        staged.append(Path(src).name)
        if threading.current_thread().name == "writer-A":
            a_staged.set()
            assert b_published.wait(5)
        real_replace(src, dst)

    monkeypatch.setattr(health.os, "replace", scheduled_replace)
    writer_a = threading.Thread(
        name="writer-A",
        target=lambda: health.record_failure("claude", "short", poppy_dir=tmp_path),
    )
    writer_a.start()
    assert a_staged.wait(5)
    health.record_failure("codex", "a much longer error " * 20, poppy_dir=tmp_path)
    b_published.set()
    writer_a.join(5)
    assert not writer_a.is_alive()

    # Each writer staged through a temporary file of its own.
    assert len(set(staged)) == 2, staged
    # Whoever published last, the file is whole: parseable, with one record intact.
    raw = health.health_path(tmp_path).read_text()
    json.loads(raw)  # no trailing garbage from the other writer
    recorded = health.load(tmp_path)
    assert set(recorded.clis) in ({"claude"}, {"codex"})
    assert recorded.last_error in ("short", "a much longer error " * 20)
    assert recorded.consecutive_failures == 1


def test_a_failed_publish_leaves_no_temp_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(src, dst):  # type: ignore[no-untyped-def]
        raise OSError("no space left on device")

    monkeypatch.setattr(health.os, "replace", boom)
    health.record_failure("claude", "exited rc=1", poppy_dir=tmp_path)  # never raises
    assert list(tmp_path.glob("*.tmp")) == []
    assert not health.health_path(tmp_path).exists()


def test_missing_file_reads_healthy(tmp_path: Path) -> None:
    assert health.load(tmp_path).consecutive_failures == 0
    assert not health.backend_failing(tmp_path)


# --- policy + banner -------------------------------------------------------


def test_failing_backend_is_not_active_but_still_runs() -> None:
    cfg = PoppyConfig(consent="granted")
    assert evaluate(cfg, host_cli=True, backend_broken=True) is CaptureStatus.WARN_BACKEND_BROKEN
    # Capture keeps firing so a re-login can clear the failure record by itself.
    assert is_capture_enabled(cfg, host_cli=True, backend_broken=True) is True


def test_forced_env_still_reports_a_broken_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """POPPY_CONSOLIDATE=1 forces capture on; it cannot make a logged-out CLI work."""
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    cfg = PoppyConfig()  # consent still pending — the override implies it
    assert evaluate(cfg, backend_broken=False) is CaptureStatus.FORCED_ENV
    assert evaluate(cfg, backend_broken=True) is CaptureStatus.WARN_BACKEND_BROKEN
    # Forced on stays forced on: the operator is told, not overruled.
    assert is_capture_enabled(cfg, backend_broken=True) is True


def test_healthy_backend_is_still_active() -> None:
    cfg = PoppyConfig(consent="granted")
    assert evaluate(cfg, host_cli=True, backend_broken=False) is CaptureStatus.ACTIVE


def test_banner_names_the_failing_backend() -> None:
    out = render_banner(
        CaptureStatus.WARN_BACKEND_BROKEN,
        project="poppy",
        memory_count=5,
        last_session_count=None,
        backend_cli="claude",
    )
    assert out is not None
    assert "INACTIVE" in out
    assert "`claude`" in out
    assert is_user_facing_status(CaptureStatus.WARN_BACKEND_BROKEN)


# --- end to end: a stubbed `claude` that exits 1 ---------------------------


def _bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private bin directory at the front of PATH, for host-CLI stubs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _stub(bin_dir: Path, name: str, *, stdout: str = "", stderr: str = "", code: int = 1, sleep_s: float = 0) -> None:
    """Write an executable host-CLI stub with a fixed exit code and output.

    ``sleep_s`` stalls the stub so the caller's timeout is what ends the call.
    """
    script = bin_dir / name
    body = "#!/bin/sh\n"
    if sleep_s:
        body += f"sleep {sleep_s}\n"
    if stdout:
        body += f"cat <<'OUT'\n{stdout}\nOUT\n"
    if stderr:
        body += f'echo "{stderr}" >&2\n'
    script.write_text(body + f"exit {code}\n")
    script.chmod(0o755)


def _stub_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a `claude` on PATH that fails every extraction, like a logged-out CLI."""
    _stub(_bin_dir(tmp_path, monkeypatch), "claude", stderr=STUB_STDERR)


def _transcript(tmp_path: Path) -> Path:
    """A Claude Code transcript (its path is what selects the `claude` backend)."""
    session_dir = tmp_path / ".claude" / "projects" / "proj"
    session_dir.mkdir(parents=True)
    rows = []
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"turn {i}: a substantive discussion of the deployment architecture and schema decision {i}"
        rows.append({"type": role, "message": {"role": role, "content": text}})
    path = session_dir / "session.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def _codex_transcript(tmp_path: Path) -> Path:
    """A Codex rollout — the `session_meta` envelope is what selects `codex`."""
    session_dir = tmp_path / ".codex" / "sessions"
    session_dir.mkdir(parents=True)
    rows: list[dict] = [{"type": "session_meta", "payload": {"id": "codex-session"}}]
    for i in range(4):
        role = "user" if i % 2 == 0 else "assistant"
        rows.append({"type": "response_item", "payload": {"type": "message", "role": role, "content": []}})
    path = session_dir / "rollout.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_broken_claude_degrades_banner_and_doctor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    poppy_dir = tmp_path / "poppy"
    poppy_dir.mkdir()
    (poppy_dir / "config.json").write_text(json.dumps({"consent": "granted", "engine": "seed"}))
    monkeypatch.setenv("POPPY_DIR", str(poppy_dir))
    monkeypatch.setattr("poppy.consolidation.get_engine", lambda *_a, **_k: SeedEngine(db_path=poppy_dir / "m.db"))
    _stub_claude(tmp_path, monkeypatch)
    payload = {
        "session_id": "sess-broken",
        "transcript_path": str(_transcript(tmp_path)),
        "cwd": str(tmp_path / "proj"),
    }

    # Before the threshold the banner still reads active — one failure is not proof.
    assert consolidate_stop_event(payload) == 0
    banner, _user_facing, status = _session_banner(poppy_dir, project="poppy", memory_count=0, engine_ok=True)
    assert status is CaptureStatus.ACTIVE
    assert "active" in (banner or "").lower()

    for _ in range(health.FAILURE_THRESHOLD - 1):
        assert consolidate_stop_event(payload) == 0

    banner, user_facing, status = _session_banner(poppy_dir, project="poppy", memory_count=0, engine_ok=True)
    assert status is CaptureStatus.WARN_BACKEND_BROKEN
    assert banner is not None
    assert "INACTIVE" in banner
    assert "`claude`" in banner
    assert user_facing is True

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={"POPPY_DIR": str(poppy_dir), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude-config")},
    )
    assert result.exit_code == 0, result.output
    assert "[!] auto-capture: WARN" in result.output
    assert STUB_STDERR in result.output
    assert "rc=1" in result.output

    # A working backend heals the state: the next success clears the record.
    health.record_success(poppy_dir=poppy_dir)
    _banner, _user_facing, status = _session_banner(poppy_dir, project="poppy", memory_count=0, engine_ok=True)
    assert status is CaptureStatus.ACTIVE


# --- end to end: two broken backends, and exit-0 that is not success -------


def _consented_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A POPPY_DIR with consent granted, pointed at by the environment."""
    poppy_dir = tmp_path / "poppy"
    poppy_dir.mkdir()
    (poppy_dir / "config.json").write_text(json.dumps({"consent": "granted", "engine": "seed"}))
    monkeypatch.setenv("POPPY_DIR", str(poppy_dir))
    return poppy_dir


def test_alternating_broken_backends_still_trip_the_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two broken host CLIs used in turn must not hide each other from the threshold.

    A single shared counter restarted on every switch between backends, so a
    developer with both `claude` and `codex` logged out could fail forever and
    never see the warning. Per-CLI counting is what closes it.
    """
    poppy_dir = _consented_dir(tmp_path, monkeypatch)
    bin_dir = _bin_dir(tmp_path, monkeypatch)
    _stub(bin_dir, "claude", stderr=STUB_STDERR)
    _stub(bin_dir, "codex", stderr=STUB_STDERR)
    transcripts = {"claude": str(_transcript(tmp_path)), "codex": str(_codex_transcript(tmp_path))}
    cfg = PoppyConfig(consent="granted")

    for _ in range(health.FAILURE_THRESHOLD):
        for name in ("claude", "codex"):
            assert call_llm("extract memories", transcript_path=transcripts[name], cfg=cfg) == []

    recorded = health.load(poppy_dir)
    assert recorded.failing, recorded.clis
    assert evaluate(cfg) is CaptureStatus.WARN_BACKEND_BROKEN
    assert recorded.clis["claude"].consecutive_failures == health.FAILURE_THRESHOLD
    assert recorded.clis["codex"].consecutive_failures == health.FAILURE_THRESHOLD

    banner, user_facing, status = _session_banner(poppy_dir, project="poppy", memory_count=0, engine_ok=True)
    assert status is CaptureStatus.WARN_BACKEND_BROKEN
    assert "INACTIVE" in (banner or "")
    assert user_facing is True


def _decision(mid: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=content,
        memory_type="decision",
        source=Source(type="manual", session_id=None, timestamp=now),
        project="poppy",
        related_to=[],
        created_at=now,
        updated_at=now,
    )


def test_a_slow_conflict_verdict_does_not_mark_the_backend_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verdict's timeout says nothing about whether extraction works.

    A verdict runs on a far shorter budget than an extraction, so a CLI that is
    merely slow to start times out here while extracting perfectly well. One
    capture pass asks for a verdict per candidate, so counting those would reach
    the threshold on its own and tell someone whose capture is fine that every
    extraction is failing.
    """
    poppy_dir = _consented_dir(tmp_path, monkeypatch)
    _stub(_bin_dir(tmp_path, monkeypatch), "claude", sleep_s=30, stdout="[]", code=0)
    monkeypatch.setattr("poppy.capture.reconciler.CONFLICT_LLM_TIMEOUT_S", 0.05)
    engine = SeedEngine(db_path=poppy_dir / "m.db")
    engine.ingest(_decision("existing", "We deploy the recall worker to Railway in Amsterdam."))
    cfg = PoppyConfig(consent="granted")

    for _ in range(health.FAILURE_THRESHOLD):
        candidate = _decision("new", "We deploy the recall worker to Railway in Frankfurt.")
        assert detect_conflicts(engine, candidate, cfg=cfg) == []

    assert health.load(poppy_dir).clis == {}, "a verdict must not touch the extraction health record"
    assert evaluate(cfg) is CaptureStatus.ACTIVE
    banner, _user_facing, status = _session_banner(poppy_dir, project="poppy", memory_count=1, engine_ok=True)
    assert status is CaptureStatus.ACTIVE
    assert "INACTIVE" not in (banner or "")

    # The same CLI timing out on a real extraction is still counted, so opting
    # the verdict out has not blinded the record to a genuinely broken backend.
    transcript = str(_transcript(tmp_path))
    for _ in range(health.FAILURE_THRESHOLD):
        assert call_llm("extract memories", transcript_path=transcript, cfg=cfg, host_timeout_s=0.05) == []
    assert health.load(poppy_dir).failing
    assert evaluate(cfg) is CaptureStatus.WARN_BACKEND_BROKEN


def test_an_exit_zero_error_page_does_not_clear_the_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exiting 0 is not proof the backend worked — only a parseable array is."""
    poppy_dir = _consented_dir(tmp_path, monkeypatch)
    _stub(_bin_dir(tmp_path, monkeypatch), "claude", stdout="Error: your session has expired, run /login", code=0)
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", "exited rc=1", poppy_dir=poppy_dir)

    transcript = str(_transcript(tmp_path))
    assert call_llm("extract memories", transcript_path=transcript, cfg=PoppyConfig(consent="granted")) == []
    assert health.load(poppy_dir).failing


def test_an_empty_array_counts_as_a_working_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty array is a real answer from a healthy CLI, so it clears the count."""
    poppy_dir = _consented_dir(tmp_path, monkeypatch)
    _stub(_bin_dir(tmp_path, monkeypatch), "claude", stdout="[]", code=0)
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", "exited rc=1", poppy_dir=poppy_dir)

    transcript = str(_transcript(tmp_path))
    assert call_llm("extract memories", transcript_path=transcript, cfg=PoppyConfig(consent="granted")) == []
    assert health.load(poppy_dir).clis == {}


# --- doctor: a shareable WARN, and one that survives the env override ------

SECRET_TAIL = "exited rc=1 stderr_tail='rejected key sk-abcDEF0123456789abcDEF0123'"


def _doctor(poppy_dir: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    return CliRunner().invoke(
        cli,
        ["doctor"],
        env={"POPPY_DIR": str(poppy_dir), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude-config")},
    )


def test_doctor_redacts_the_stderr_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An auth failure can echo the credential back; doctor output gets pasted into bug reports."""
    poppy_dir = _consented_dir(tmp_path, monkeypatch)
    _stub_claude(tmp_path, monkeypatch)
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", SECRET_TAIL, poppy_dir=poppy_dir)

    result = _doctor(poppy_dir, tmp_path)
    assert result.exit_code == 0, result.output
    assert "[!] auto-capture: WARN" in result.output
    assert "sk-abcDEF0123456789abcDEF0123" not in result.output
    assert MASK in result.output
    assert "rejected key" in result.output  # masked, not dropped


def test_forced_env_still_warns_in_doctor_and_the_banner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """POPPY_CONSOLIDATE=1 used to report OK over a backend that failed every run."""
    poppy_dir = tmp_path / "poppy"
    poppy_dir.mkdir()
    monkeypatch.setenv("POPPY_DIR", str(poppy_dir))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")  # forced on, consent never recorded
    _stub_claude(tmp_path, monkeypatch)
    for _ in range(health.FAILURE_THRESHOLD):
        health.record_failure("claude", "exited rc=1", poppy_dir=poppy_dir)

    banner, user_facing, status = _session_banner(poppy_dir, project="poppy", memory_count=0, engine_ok=True)
    assert status is CaptureStatus.WARN_BACKEND_BROKEN
    assert "INACTIVE" in (banner or "")
    assert "`claude`" in (banner or "")
    assert user_facing is True

    result = _doctor(poppy_dir, tmp_path)
    assert result.exit_code == 0, result.output
    assert "[!] auto-capture: WARN" in result.output
