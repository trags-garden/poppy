"""Unit tests for poppy.telemetry: file mutations, consent, and precedence.

Network is never hit because we never call `capture()` with a working SDK
(the PostHog import is shimmed out per test) and the autouse conftest fixture
sets POPPY_TELEMETRY_OFF=1 unless a test removes it. The tests focus on the
deterministic state of `~/.poppy/analytics.json` and `~/.poppy/config.json`.
"""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import poppy
from poppy import telemetry


@pytest.fixture
def telemetry_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the suite-wide POPPY_TELEMETRY_OFF=1 guard for telemetry-on tests."""
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)


@pytest.fixture
def telemetry_on(tmp_path: Path, telemetry_allowed) -> None:
    """Record opt-in without creating analytics state before capture tests."""
    from poppy.config import load_config, save_config

    cfg = load_config(tmp_path)
    cfg.telemetry_enabled = True
    save_config(cfg)


class FakeClient:
    """Stand-in for posthog.Posthog matching the real capture() signature."""

    instances: list["FakeClient"] = []

    def __init__(self, *args, **kwargs):
        self.captured: list[tuple[str, str | None, dict | None]] = []
        self.shut_down = False
        FakeClient.instances.append(self)

    def capture(self, event, distinct_id=None, properties=None):
        self.captured.append((event, distinct_id, properties))

    # posthog-python 6+ signature: no timeout parameter.
    def shutdown(self):
        self.shut_down = True


@pytest.fixture
def fresh_client_state():
    """Reset the module-level client cache around a test."""
    FakeClient.instances = []
    old = telemetry._state["client"]
    old_host = telemetry._state.get("host")
    telemetry._state["client"] = None
    telemetry._state["host"] = None
    yield
    telemetry._state["client"] = old
    telemetry._state["host"] = old_host


def _capture_once_worker(poppy_dir: str, barrier, results) -> None:
    """Call capture_once concurrently without initializing the PostHog client."""
    import os

    from poppy import telemetry as worker_telemetry

    os.environ.pop("POPPY_TELEMETRY_OFF", None)
    worker_telemetry.capture = lambda *_args, **_kwargs: None
    barrier.wait()
    results.put(worker_telemetry.capture_once(Path(poppy_dir), "closed_loop"))


def test_first_run_creates_analytics_with_device_id(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    assert not (tmp_path / "analytics.json").exists()

    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "memory_write", {"memory_type": "fact"})

    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["telemetry"] == "on"
    assert len(data["device_id"]) >= 16
    assert "created_at" in data


def test_an_interrupted_analytics_save_keeps_the_previous_file(tmp_path: Path, monkeypatch) -> None:
    """A torn analytics.json reads as empty, minting a new device id; a failed save must leave the old file."""
    telemetry._save(tmp_path, {"device_id": "kept", "first_run_notice_shown": True})

    def killed_before_rename(*_args, **_kwargs):
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", killed_before_rename)
    with pytest.raises(OSError):
        telemetry._save(tmp_path, {"device_id": "new"})
    monkeypatch.undo()

    assert telemetry._load(tmp_path) == {"device_id": "kept", "first_run_notice_shown": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["analytics.json"]


def test_an_unpersisted_analytics_save_stays_silent(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch
) -> None:
    """A disk error while saving analytics.json never reaches the CLI; the milestone just reports not emitted."""
    real_fsync = os.fsync

    def failing_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "Input/output error")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing_directory_fsync)
    with patch("posthog.Posthog", FakeClient):
        assert telemetry.capture_once(tmp_path, "closed_loop") is False
        telemetry.capture(tmp_path, "memory_write")

    assert all(not client.captured for client in FakeClient.instances)


def test_telemetry_off_writes_nothing(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    telemetry.set_enabled(tmp_path, False)

    with patch("posthog.Posthog") as fake_ph:
        telemetry.capture(tmp_path, "memory_write", {"memory_type": "fact"})

    # Off should never even import/initialize the client.
    fake_ph.assert_not_called()
    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["telemetry"] == "off"


def test_get_device_id_returns_none_when_off(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, False)
    assert telemetry.get_device_id(tmp_path) is None


def test_get_device_id_returns_uuid_when_on(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, True)
    device_id = telemetry.get_device_id(tmp_path)
    assert device_id is not None
    assert len(device_id) >= 16


def test_capture_once_emits_only_the_first_time(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    """A funnel milestone emits once per device and latches thereafter."""
    with patch("posthog.Posthog", FakeClient):
        first = telemetry.capture_once(tmp_path, "closed_loop", {"channel": "session_start"})
        second = telemetry.capture_once(tmp_path, "closed_loop", {"channel": "recall_prompt"})

    assert first is True
    assert second is False  # latched — no second emit
    events = [e for c in FakeClient.instances for (e, _d, _p) in c.captured]
    assert events.count("closed_loop") == 1

    # The latch persists in analytics.json under the milestones map.
    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["milestones"]["closed_loop"] is True


def test_capture_once_emits_exactly_once_across_processes(tmp_path: Path, telemetry_on) -> None:
    """Concurrent hook processes share one atomic milestone latch."""
    pytest.importorskip("fcntl")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [context.Process(target=_capture_once_worker, args=(str(tmp_path), barrier, results)) for _ in range(2)]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(results.get(timeout=1) for _ in processes) == [False, True]
    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["milestones"]["closed_loop"] is True


def test_capture_once_is_per_event(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    """Distinct milestones each fire once — they don't share the latch."""
    with patch("posthog.Posthog", FakeClient):
        assert telemetry.capture_once(tmp_path, "setup_completed", {"agent": "claude-code"}) is True
        assert telemetry.capture_once(tmp_path, "consent_granted", {"via": "setup_prompt"}) is True
        assert telemetry.capture_once(tmp_path, "setup_completed", {"agent": "cursor"}) is False

    # (cli_install is emitted once on the first-ever capture; ignore it here.)
    events = sorted(e for c in FakeClient.instances for (e, _d, _p) in c.captured if e != "cli_install")
    assert events == ["consent_granted", "setup_completed"]


def test_capture_once_no_op_when_off(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    telemetry.set_enabled(tmp_path, False)
    with patch("posthog.Posthog") as fake_ph:
        emitted = telemetry.capture_once(tmp_path, "closed_loop", {"channel": "session_start"})
    assert emitted is False
    fake_ph.assert_not_called()
    data = json.loads((tmp_path / "analytics.json").read_text())
    assert "milestones" not in data  # nothing latched when telemetry is off


def test_capture_never_raises_on_sdk_failure(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, True)

    class BrokenClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated SDK failure")

    # Reset module-level cache so we re-attempt init under the patched SDK.
    telemetry._state["client"] = None

    with patch("posthog.Posthog", BrokenClient):
        # Must not raise.
        telemetry.capture(tmp_path, "memory_write", {})


def test_set_enabled_preserves_device_id(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, True)
    first_id = telemetry.get_device_id(tmp_path)
    telemetry.set_enabled(tmp_path, False)
    telemetry.set_enabled(tmp_path, True)
    assert telemetry.get_device_id(tmp_path) == first_id


# ---------------------------------------------------------------------------
# Config.json flag, precedence, status, first-run consent
# ---------------------------------------------------------------------------


def test_default_is_off_until_answered(tmp_path: Path, telemetry_allowed) -> None:
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is False
    assert reason == "not answered yet"


def test_env_var_always_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)
    telemetry.set_enabled(tmp_path, True)
    assert telemetry.is_enabled(tmp_path) is True

    monkeypatch.setenv("POPPY_TELEMETRY_OFF", "1")
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is False
    assert "POPPY_TELEMETRY_OFF" in reason


def test_set_enabled_persists_flag_in_config_json(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, False)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is False
    assert telemetry.is_enabled(tmp_path) is False

    telemetry.set_enabled(tmp_path, True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is True
    assert telemetry.is_enabled(tmp_path) is True


def test_legacy_analytics_opt_out_still_respected(tmp_path: Path, telemetry_allowed) -> None:
    # Older installs persisted only {"telemetry": "off"} in analytics.json.
    (tmp_path / "analytics.json").write_text(json.dumps({"telemetry": "off"}))
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is False
    assert "analytics.json" in reason


def test_corrupt_config_json_falls_back_to_off(tmp_path: Path, telemetry_allowed) -> None:
    (tmp_path / "config.json").write_text("{not json")
    assert telemetry.is_enabled(tmp_path) is False


@pytest.mark.parametrize("legacy_data", [{}, {"telemetry": "on", "device_id": "old", "first_run_notice_shown": True}])
def test_unanswered_consent_never_initializes_client_or_latches_events(
    tmp_path, telemetry_allowed, fresh_client_state, legacy_data, capsys
):
    if legacy_data:
        telemetry._save(tmp_path, legacy_data)
    with patch("posthog.Posthog") as client:
        telemetry.capture(tmp_path, "memory_write")
        assert telemetry.capture_once(tmp_path, "closed_loop") is False
    client.assert_not_called()
    assert telemetry.get_device_id(tmp_path) is None
    assert telemetry._load(tmp_path) == legacy_data
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_prompt_persistence_failure_stays_off(tmp_path, telemetry_allowed, monkeypatch):
    monkeypatch.setattr(telemetry, "_a_person_is_watching", lambda: True)
    with patch("click.confirm", return_value=True), patch("poppy.config.save_config", side_effect=OSError("disk full")):
        telemetry.maybe_prompt_for_consent(tmp_path)
    assert telemetry.status(tmp_path) == (False, "not answered yet")


def _scrub_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the agent and CI markers the suite itself is probably running under."""
    for name in telemetry._NON_INTERACTIVE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(telemetry._NON_INTERACTIVE_ENV_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


def test_a_clean_terminal_is_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_streams_are_a_terminal", lambda: True)
    _scrub_agent_env(monkeypatch)
    assert telemetry._a_person_is_watching() is True


@pytest.mark.parametrize("var", sorted(telemetry._NON_INTERACTIVE_ENV_VARS) + ["CLAUDE_CODE_ENTRYPOINT"])
def test_an_agent_or_runner_is_not_a_person(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    """These all allocate a terminal nobody can type into."""
    monkeypatch.setattr(telemetry, "_streams_are_a_terminal", lambda: True)
    _scrub_agent_env(monkeypatch)
    monkeypatch.setenv(var, "1")
    assert telemetry._a_person_is_watching() is False


def test_a_dumb_terminal_is_not_a_person(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_streams_are_a_terminal", lambda: True)
    _scrub_agent_env(monkeypatch)
    monkeypatch.setenv("TERM", "dumb")
    assert telemetry._a_person_is_watching() is False


def test_an_empty_agent_variable_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-empty variable is how a shell says "not set"."""
    monkeypatch.setattr(telemetry, "_streams_are_a_terminal", lambda: True)
    _scrub_agent_env(monkeypatch)
    monkeypatch.setenv("CI", "")
    assert telemetry._a_person_is_watching() is True


@pytest.mark.parametrize(
    ("stored", "answered"),
    [
        (True, True),
        (False, True),
        ("false", False),
        ("off", False),
        ("true", False),
        ("", False),
        (None, False),
        (0, False),
        (1, False),
        ([], False),
    ],
)
def test_only_a_real_boolean_counts_as_an_answer(tmp_path: Path, telemetry_allowed, stored, answered) -> None:
    """A hand-edited `"off"` used to read as an opt-in: bool() makes every non-empty string true."""
    (tmp_path / "config.json").write_text(json.dumps({"telemetry_enabled": stored}))

    enabled, reason = telemetry.status(tmp_path)

    assert telemetry.is_unanswered(tmp_path) is not answered
    if answered:
        assert enabled is stored
    else:
        # Never on, and never mistaken for a decline the user made.
        assert enabled is False
        assert reason == "not answered yet"


def test_an_unreadable_config_cannot_record_a_choice(tmp_path: Path, telemetry_allowed) -> None:
    (tmp_path / "config.json").write_text("{not json")

    with pytest.raises(telemetry.TelemetryChoiceError) as excinfo:
        telemetry.set_enabled(tmp_path, True)

    assert "config.json" in str(excinfo.value)
    assert telemetry.is_unanswered(tmp_path) is True


def test_a_failed_legacy_mirror_still_records_the_choice(tmp_path: Path, telemetry_allowed) -> None:
    """analytics.json is a compatibility mirror; losing it does not unmake an answer."""
    (tmp_path / "analytics.json").mkdir()

    telemetry.set_enabled(tmp_path, False)  # must not raise

    from poppy.config import load_config

    assert load_config(tmp_path).telemetry_enabled is False
    assert telemetry.status(tmp_path) == (False, f"set in {tmp_path / 'config.json'}")


def test_cli_install_reports_resolved_version(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "recall_call", {"query_length": 3})

    assert len(FakeClient.instances) == 1
    events = FakeClient.instances[0].captured
    install_events = [e for e in events if e[0] == "cli_install"]
    assert len(install_events) == 1
    props = install_events[0][2]
    assert props["version"] == poppy.__version__


def test_answering_the_question_still_reports_the_install(
    tmp_path: Path, telemetry_allowed, fresh_client_state
) -> None:
    """Saying yes mints the device id, which must not swallow the install event."""
    telemetry.set_enabled(tmp_path, True)

    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "recall_call", {"query_length": 3})
        telemetry.capture(tmp_path, "recall_call", {"query_length": 4})

    events = [event for (event, _, _) in FakeClient.instances[0].captured]
    assert events.count("cli_install") == 1


def test_an_install_from_before_the_latch_is_not_recounted(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    """Upgrades carry a device id and no latch: their install was already reported."""
    telemetry._save(tmp_path, {"device_id": "existing", "telemetry": "on"})

    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "recall_call", {"query_length": 3})

    events = [event for (event, _, _) in FakeClient.instances[0].captured]
    assert "cli_install" not in events


class _ExplodingShutdownClient(FakeClient):
    def shutdown(self):
        raise RuntimeError("flush failed")


class _StalledShutdownClient(FakeClient):
    """An SDK draining a backed-up queue against a dead network."""

    def shutdown(self):
        import time

        time.sleep(5)


@pytest.mark.parametrize("client_cls", [FakeClient, _ExplodingShutdownClient])
def test_shutdown_never_raises_and_flushes(fresh_client_state, client_cls) -> None:
    """PostHog 6+ has shutdown(self); passing timeout= made the SDK print a traceback."""
    client = client_cls()
    telemetry._state["client"] = client
    telemetry._shutdown()
    if client_cls is FakeClient:
        assert client.shut_down is True


def test_shutdown_skips_flush_when_a_thread_cannot_start(fresh_client_state, monkeypatch) -> None:
    """CPython 3.12.0-3.12.2 raise on thread creation at exit; skip the flush, never block."""
    client = FakeClient()
    telemetry._state["client"] = client

    def _boom(*_a, **_k):
        raise RuntimeError("can't create new thread at interpreter shutdown")

    monkeypatch.setattr(telemetry.threading, "Thread", _boom)
    telemetry._shutdown()  # must return without raising and without an unbounded inline flush
    assert client.shut_down is False


def test_shutdown_is_bounded_even_when_the_sdk_stalls(fresh_client_state) -> None:
    import time

    telemetry._state["client"] = _StalledShutdownClient()
    started = time.monotonic()
    telemetry._shutdown()
    assert time.monotonic() - started < telemetry._FLUSH_TIMEOUT_S + 1.0


def test_loopback_host_override_reaches_the_client(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch
) -> None:
    """The CI telemetry-on smoke points the SDK at a local sink instead of production."""
    monkeypatch.setenv(telemetry._HOST_ENV, "http://127.0.0.1:9")
    seen: dict = {}

    class _Recording(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.update(kwargs)

    with patch("posthog.Posthog", _Recording):
        telemetry.capture(tmp_path, "cli_remember")
    assert seen.get("host") == "http://127.0.0.1:9"
    # Bounded, single-attempt delivery so an offline exit never stalls.
    assert seen.get("max_retries") == 0
    assert seen.get("timeout") == telemetry._REQUEST_TIMEOUT_S


@pytest.mark.parametrize(
    "value",
    [
        "http://example.invalid",
        "https://eu.i.posthog.com.attacker.test",
        "127.0.0.1:9",
        "not a url",
        # Parser-disagreement bypass: urlsplit reads hostname 127.0.0.1 but
        # requests would send to attacker.example.
        "http://attacker.example\\@127.0.0.1",
        "http://user@127.0.0.1",
        "ftp://127.0.0.1",
        "http://127.0.0.1:99999",
        "",  # present but empty (e.g. POPPY_TELEMETRY_HOST="${SINK:-}") must fail closed, not hit production
    ],
)
def test_non_loopback_host_override_turns_telemetry_off(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch, value
) -> None:
    """A mistyped or hostile override must fail closed, never fall back to production."""
    monkeypatch.setenv(telemetry._HOST_ENV, value)
    assert telemetry._host() is None
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "cli_remember")
    assert telemetry._state["client"] is None
    assert FakeClient.instances == []

    # The rejected override must not consume first-run state: analytics.json is
    # untouched, so cli_install still fires once the override is removed.
    assert not (tmp_path / "analytics.json").exists()
    monkeypatch.delenv(telemetry._HOST_ENV)
    telemetry._state["client"] = None
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "cli_remember")
    emitted = [event for inst in FakeClient.instances for (event, _d, _p) in inst.captured]
    assert "cli_install" in emitted


@pytest.mark.parametrize("value", ["http://127.0.0.1:9", "http://127.0.0.1", "https://[::1]:8000"])
def test_clean_loopback_override_is_accepted(monkeypatch, value) -> None:
    monkeypatch.setenv(telemetry._HOST_ENV, value)
    assert telemetry._host() == value


@pytest.mark.parametrize("value", ["https://example.invalid", "http://127.0.0.1:99999", "http://user@127.0.0.1"])
def test_bad_host_override_reports_off_everywhere(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch, value
) -> None:
    """A set-but-unusable override disables telemetry in status, the prompt and capture."""
    import io
    from contextlib import redirect_stderr

    monkeypatch.setenv(telemetry._HOST_ENV, value)
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is False and telemetry._HOST_ENV in reason
    buf = io.StringIO()
    with redirect_stderr(buf):
        telemetry.maybe_prompt_for_consent(tmp_path)
    assert buf.getvalue() == ""
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "cli_remember")
    assert FakeClient.instances == []
    assert not (tmp_path / "analytics.json").exists()


def test_override_delivery_session_does_not_follow_redirects(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch
) -> None:
    """A loopback sink that 307s must not be able to forward the payload off-box."""
    pr = pytest.importorskip("posthog.request")
    monkeypatch.setenv(telemetry._HOST_ENV, "http://127.0.0.1:9")
    pr._session.max_redirects = 30  # requests default, so the assertion proves our code set it
    pr._session.trust_env = True
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "cli_remember")
    assert pr._session.max_redirects == 0
    assert pr._session.trust_env is False  # no proxy env, no redirect: cannot leave loopback

    # The SDK rebuilds its sessions in a fork child; our at-fork hook must re-confine.
    pr._session.max_redirects = 30
    pr._session.trust_env = True
    telemetry._reconfine_override_session_in_child()
    assert pr._session.max_redirects == 0 and pr._session.trust_env is False


def test_production_delivery_session_is_left_alone(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch
) -> None:
    """Redirect hardening applies only to the override path, never to production delivery.

    Patches the SDK so no real client is built and no event egresses to the real
    project; the FakeClient path still exercises the host==production branch that
    skips the redirect hardening.
    """
    pr = pytest.importorskip("posthog.request")
    monkeypatch.delenv(telemetry._HOST_ENV, raising=False)
    pr._session.max_redirects = 30
    pr._session.trust_env = True
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "cli_remember")
    assert pr._session.max_redirects == 30
    assert pr._session.trust_env is True


def test_client_is_rebuilt_when_the_resolved_host_changes(
    tmp_path: Path, telemetry_on, fresh_client_state, monkeypatch
) -> None:
    """A cached client must not keep hitting the old host after the override changes."""
    hosts: list = []

    class _Recording(FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            hosts.append(kwargs.get("host"))

    monkeypatch.delenv(telemetry._HOST_ENV, raising=False)
    with patch("posthog.Posthog", _Recording):
        telemetry.capture(tmp_path, "cli_remember")
        monkeypatch.setenv(telemetry._HOST_ENV, "http://127.0.0.1:9")
        telemetry.capture(tmp_path, "cli_remember")
    assert hosts == [telemetry._POSTHOG_HOST, "http://127.0.0.1:9"]


def test_valid_loopback_override_keeps_status_on(tmp_path: Path, telemetry_on, monkeypatch) -> None:
    monkeypatch.setenv(telemetry._HOST_ENV, "http://127.0.0.1:9")
    enabled, _ = telemetry.status(tmp_path)
    assert enabled is True


def test_real_sdk_exit_hook_name_is_one_we_disown() -> None:
    """Guard the ("join", "_atexit") list against a rename in a future posthog release."""
    import inspect

    posthog = pytest.importorskip("posthog")
    init_src = inspect.getsource(posthog.Posthog.__init__)
    registered = [line.strip() for line in init_src.splitlines() if "atexit.register" in line]
    assert registered, "the SDK no longer registers an exit hook; drop _disown_sdk_exit_hook or update this test"
    assert all("self.join" in line or "self._atexit" in line for line in registered), registered


def test_sdk_logger_is_silenced_even_with_a_root_handler(
    tmp_path: Path, telemetry_on, fresh_client_state, capsys: pytest.CaptureFixture
) -> None:
    import logging

    log = logging.getLogger("posthog")
    before_handlers, before_propagate = list(log.handlers), log.propagate
    root_handler = logging.StreamHandler(sys.stderr)
    logging.getLogger().addHandler(root_handler)
    try:
        with patch("posthog.Posthog", FakeClient):
            telemetry.capture(tmp_path, "cli_remember")
        assert any(isinstance(h, logging.NullHandler) for h in log.handlers)
        log.error("error uploading: connection refused")
        assert "error uploading" not in capsys.readouterr().err
    finally:
        logging.getLogger().removeHandler(root_handler)
        log.handlers[:] = before_handlers
        log.propagate = before_propagate


def test_sdk_exit_hook_is_unregistered(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    """Only poppy's bounded flush may run at exit; the SDK's join has no deadline."""

    class _WithJoin(FakeClient):
        def join(self):
            pass

    with patch("posthog.Posthog", _WithJoin), patch("poppy.telemetry.atexit.unregister") as unregister:
        telemetry.capture(tmp_path, "cli_remember")
    client = telemetry._state["client"]
    unregister.assert_called_once_with(client.join)
