"""The SessionStart status banner + last-session journal aggregation.

The banner render is a pure function, so the active / INACTIVE / consent-pending
wording is asserted directly. ``last_session_count`` is exercised against a temp
journal so the banner's "M captured last session" is grounded in real records.
"""

from __future__ import annotations

from pathlib import Path

from poppy.capture import journal
from poppy.capture.banner import render_banner
from poppy.capture.policy import CaptureStatus


def _render(status: CaptureStatus, **kw) -> str | None:
    base = {"project": "poppy", "memory_count": 5, "last_session_count": 3}
    base.update(kw)
    return render_banner(status, **base)


def test_active_banner_shows_counts() -> None:
    out = _render(CaptureStatus.ACTIVE)
    assert out is not None
    assert "active" in out.lower()
    assert "5 memories for this project" in out
    assert "3 memories captured last session" in out


def test_active_banner_omits_capture_clause_when_never_captured() -> None:
    out = _render(CaptureStatus.ACTIVE, last_session_count=None)
    assert out is not None
    assert "active" in out.lower()
    assert "5 memories for this project" in out
    assert "captured last session" not in out  # nothing captured yet → no clause


def test_active_banner_singular_grammar() -> None:
    out = _render(CaptureStatus.ACTIVE, memory_count=1, last_session_count=1)
    assert "1 memory for this project" in out
    assert "1 memory captured last session" in out


def test_forced_env_renders_active() -> None:
    assert "active" in _render(CaptureStatus.FORCED_ENV).lower()


def test_no_project_scope_wording() -> None:
    out = _render(CaptureStatus.ACTIVE, project=None)
    assert "all projects" in out


def test_consent_pending_points_to_enable_and_never_says_zero_captured() -> None:
    out = _render(CaptureStatus.INERT_PENDING, last_session_count=None)
    assert out is not None
    assert "pending your consent" in out
    assert "poppy consent --enable" in out
    # AC: pending must not render a misleading "0 captured".
    assert "0 " not in out
    assert "captured last session" not in out


def test_no_backend_is_loud_inactive() -> None:
    out = _render(CaptureStatus.DISABLED_NO_BACKEND)
    assert out is not None
    assert "INACTIVE" in out
    assert "no extraction backend" in out


def test_remote_only_is_loud_inactive() -> None:
    out = _render(CaptureStatus.WARN_REMOTE_ONLY)
    assert out is not None
    assert "INACTIVE" in out
    assert "auto-spend" in out


def test_engine_failure_is_loud_inactive_regardless_of_status() -> None:
    out = _render(CaptureStatus.ACTIVE, engine_ok=False)
    assert out is not None
    assert "INACTIVE" in out
    assert "engine" in out.lower()


def test_opt_out_is_silent() -> None:
    assert _render(CaptureStatus.DISABLED_OPT_OUT) is None


def test_env_off_is_silent() -> None:
    assert _render(CaptureStatus.DISABLED_ENV) is None


# --- last_session_count aggregation ---------------------------------------


def test_last_session_count_none_when_empty(tmp_path: Path) -> None:
    assert journal.last_session_count(tmp_path) is None


def test_last_session_count_sums_only_latest_session(tmp_path: Path) -> None:
    # Two captures in an older session, then two in the most recent one.
    journal.record(tmp_path, session_id="old", project="p", count=1, items=[])
    journal.record(tmp_path, session_id="old", project="p", count=2, items=[])
    journal.record(tmp_path, session_id="new", project="p", count=4, items=[])
    journal.record(tmp_path, session_id="new", project="p", count=1, items=[])
    # Latest session is "new": 4 + 1, the older "old" records are not counted.
    assert journal.last_session_count(tmp_path) == 5


def test_last_session_count_single_record(tmp_path: Path) -> None:
    journal.record(tmp_path, session_id="s", project=None, count=3, items=[])
    assert journal.last_session_count(tmp_path) == 3
