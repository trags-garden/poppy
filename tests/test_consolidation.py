import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import poppy.consolidation
from poppy.capture import health
from poppy.config import PoppyConfig, resolved_consolidate_settings
from poppy.consolidation import (
    MAX_RESPONSE_BYTES,
    OpenAICompatError,
    _compact_event_id,
    _http_client,
    call_host_cli,
    call_llm,
    call_openai_compat,
    consolidate_compact_event,
    consolidate_stop_event,
    detect_host_cli,
    format_transcript,
    get_engine,
    get_poppy_dir,
    is_enabled,
    parse_json_array,
)
from poppy.models import Filters

CURSOR_FIXTURE = Path(__file__).parent / "fixtures" / "cursor" / "run-shell.jsonl"


@pytest.fixture
def remote_backend(monkeypatch):
    monkeypatch.setattr("poppy.consolidation.detect_host_cli", lambda _: None)
    monkeypatch.setitem(sys.modules, "openai", None)
    for name in ("POPPY_CONSOLIDATE_MODEL", "POPPY_CONSOLIDATE_BASE_URL", "POPPY_CONSOLIDATE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return PoppyConfig(consolidate_model="test-model", consolidate_api_key="test-key")


def mock_endpoint(monkeypatch, respond):
    """Serve the fallback from a mock transport, keeping the real client stack.

    Everything above the socket stays production code: the same client, the
    same streaming read, the same deadline and size cap.
    """
    seen = {}

    real = poppy.consolidation._http_client

    def factory(*, timeout_s, trust_env):
        seen["timeout_s"] = timeout_s
        seen["trust_env"] = trust_env
        # The real builder, so timeouts, redirect policy and the proxy
        # decision are the production ones; only the socket is replaced.
        return real(timeout_s=timeout_s, trust_env=trust_env, transport=httpx.MockTransport(respond))

    monkeypatch.setattr("poppy.consolidation._http_client", factory)
    return seen


class DripStream(httpx.SyncByteStream):
    """A body that arrives in chunks, each well inside the per-read timeout."""

    def __init__(self, chunk: bytes, count: int, pause_s: float = 0.0):
        self.chunk, self.count, self.pause_s = chunk, count, pause_s

    def __iter__(self):
        for _ in range(self.count):
            if self.pause_s:
                time.sleep(self.pause_s)
            yield self.chunk


@pytest.mark.parametrize("base_url", [None, "https://llm.test/v1", "https://llm.test/prefix/v1/"])
def test_openai_compat_uses_httpx_without_sdk(monkeypatch, remote_backend, base_url):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    mock_endpoint(monkeypatch, respond)
    assert (
        call_openai_compat("extract", model="test-model", base_url=base_url, api_key="test-key", max_tokens=42) == "[]"
    )
    (request,) = requests
    assert str(request.url) == (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "extract"}],
        "temperature": 0.2,
        "max_tokens": 42,
    }
    timeout = request.extensions["timeout"]
    assert timeout["read"] == 120 and timeout["write"] == 120
    # Reaching the endpoint is capped short so a stalled connect cannot spend
    # the budget before a byte is read.
    assert timeout["connect"] == 5 and timeout["pool"] == 5


def test_openai_compat_falls_back_to_the_openai_base_url_env(monkeypatch):
    """A shell already pointed at a compatible server keeps working unconfigured."""
    for name in ("POPPY_CONSOLIDATE_BASE_URL", "POPPY_CONSOLIDATE_MODEL", "POPPY_CONSOLIDATE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    assert resolved_consolidate_settings(PoppyConfig()).base_url == "https://env.test/v1"
    explicit = PoppyConfig(consolidate_base_url="https://config.test/v1")
    assert resolved_consolidate_settings(explicit).base_url == "https://config.test/v1"


def test_openai_compat_does_not_follow_a_redirect(monkeypatch, remote_backend):
    """A redirect can point at any host, and the Authorization header must not go there."""
    seen = []

    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://attacker.test/v1/chat/completions"})

    mock_endpoint(monkeypatch, respond)
    with pytest.raises(OpenAICompatError) as excinfo:
        call_openai_compat("extract", model="test-model", base_url="https://llm.test/v1", api_key="test-key")
    assert seen == ["https://llm.test/v1/chat/completions"]
    assert "HTTP 302" in str(excinfo.value)
    assert "attacker.test" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("endpoint", "remote"),
    [
        ("https://llm.test/v1", True),
        ("http://localhost:11434/v1", False),
        ("http://127.0.0.1:1234/v1", False),
        ("http://127.7.7.7:1234/v1", False),
        ("http://ollama.localhost/v1", False),
        # A fully qualified name may carry the root dot, and the unspecified
        # addresses reach this machine too.
        ("http://localhost.:11434/v1", False),
        ("http://LocalHost./v1", False),
        ("http://127.0.0.1./v1", False),
        ("http://0.0.0.0:11434/v1", False),
        ("http://[::]:11434/v1", False),
        ("http://[::1]:11434/v1", False),
        ("http://192.168.1.9:11434/v1", True),
        ("http://llm.test/v1", True),
        ("http://localhost.evil.test/v1", True),
    ],
)
def test_openai_compat_treats_a_local_model_server_differently(monkeypatch, capsys, endpoint, remote):
    """A model server on this machine needs no TLS, and must never go via a proxy.

    An environment proxy is how a remote request often gets out of a corporate
    network, but for a server on this machine it is both wrong and a way for
    whoever set the variable to collect the key.
    """

    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    seen = mock_endpoint(monkeypatch, respond)
    call_openai_compat("extract", model="test-model", base_url=endpoint, api_key="test-key")
    assert seen["trust_env"] is remote
    stderr = capsys.readouterr().err
    assert ("sent in clear text" in stderr) is (remote and endpoint.startswith("http://"))
    assert "test-key" not in stderr


def test_the_real_client_refuses_redirects_and_is_built_per_endpoint():
    """The mock transport replaces the socket, so pin the real client's own defaults."""
    with _http_client(timeout_s=12, trust_env=False) as client:
        assert client.follow_redirects is False
        assert client.trust_env is False
        assert client.timeout.read == 12
        assert client.timeout.connect == 5 and client.timeout.pool == 5
    with _http_client(timeout_s=12, trust_env=True) as client:
        assert client.trust_env is True
    # A budget shorter than the cap must not be extended by it.
    with _http_client(timeout_s=1.5, trust_env=False) as client:
        assert client.timeout.connect == 1.5


@pytest.mark.parametrize(
    ("base_url", "detail"),
    [
        ("http://[invalid/v1", "configured endpoint is not a valid URL"),
        ("ftp://llm.test/v1", "configured endpoint is not an http or https URL"),
        ("not-a-url", "configured endpoint is not an http or https URL"),
        ("https:///v1", "configured endpoint is not an http or https URL"),
    ],
)
def test_a_malformed_endpoint_is_a_named_failure_not_a_traceback(monkeypatch, capsys, remote_backend, base_url, detail):
    """A configured URL is user input, and a capture pass must survive a bad one."""
    monkeypatch.setattr("poppy.consolidation._http_client", _unreachable_client)
    with pytest.raises(OpenAICompatError) as excinfo:
        call_openai_compat("extract", model="test-model", base_url=base_url, api_key="test-key")
    assert str(excinfo.value) == detail
    cfg = PoppyConfig(consolidate_model="m", consolidate_api_key="test-key", consolidate_base_url=base_url)
    assert call_llm("extract", transcript_path=None, cfg=cfg) == []
    assert f"openai-compat {detail}" in capsys.readouterr().err


def _unreachable_client(*, timeout_s, trust_env):  # pragma: no cover - no request should be built
    raise AssertionError("a malformed endpoint must fail before any request is made")


def test_a_slow_drip_response_stops_at_the_deadline(monkeypatch):
    """An httpx timeout is per read, so a drip inside it would never trip it.

    Each chunk arrives comfortably within the per-read timeout, so only a real
    wall-clock deadline ends this. Without one the capture lock outlives its
    TTL and another worker steals it from a running pass.
    """

    def respond(request):
        return httpx.Response(200, stream=DripStream(b"x" * 8, count=10_000, pause_s=0.002))

    mock_endpoint(monkeypatch, respond)
    started = time.monotonic()
    with pytest.raises(OpenAICompatError) as excinfo:
        call_openai_compat("extract", model="m", base_url="https://llm.test/v1", api_key="test-key", timeout_s=0.25)
    elapsed = time.monotonic() - started
    assert str(excinfo.value) == "timed out reading the response"
    assert elapsed < 5, f"the deadline did not bound the read ({elapsed:.1f}s)"


def test_an_unbounded_response_body_stops_at_the_size_cap(monkeypatch):
    """A server streaming without end must not be buffered into memory."""
    chunk = b"y" * (1024 * 1024)
    count = (MAX_RESPONSE_BYTES // len(chunk)) + 4

    def respond(request):
        return httpx.Response(200, stream=DripStream(chunk, count=count))

    mock_endpoint(monkeypatch, respond)
    with pytest.raises(OpenAICompatError) as excinfo:
        call_openai_compat("extract", model="m", base_url="https://llm.test/v1", api_key="test-key", timeout_s=30)
    assert str(excinfo.value) == "response larger than 8MB"


@pytest.mark.parametrize("record_health", [True, False])
@pytest.mark.parametrize("api_key", ["sk-live-5f3a9c1e77b24d0e", "an-arbitrary-self-hosted-key"])
def test_a_key_echoed_back_by_the_endpoint_discards_the_answer(monkeypatch, capsys, api_key, record_health):
    """An endpoint that quotes the credential back is not one to keep an answer from.

    Response text becomes memories, which sync, and its opening is logged, so
    the whole answer is dropped rather than edited: a key that reads like
    ordinary words would otherwise be rewritten out of genuine memories.
    """
    detail = "response echoed the configured API key and was discarded"
    health.record_failure("claude", "previous failure")
    before = health.load()

    def respond(request):
        content = json.dumps([{"type": "fact", "content": f"The configured key is {api_key} and it works."}])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_endpoint(monkeypatch, respond)
    with pytest.raises(OpenAICompatError) as excinfo:
        call_openai_compat("extract", model="m", base_url="https://llm.test/v1", api_key=api_key)
    assert str(excinfo.value) == detail
    assert api_key not in str(excinfo.value)

    cfg = PoppyConfig(consolidate_model="m", consolidate_api_key=api_key, consolidate_base_url="https://llm.test/v1")
    monkeypatch.setattr("poppy.consolidation.detect_host_cli", lambda _: None)
    assert call_llm("extract", transcript_path=None, cfg=cfg, record_health=record_health) == []
    stderr = capsys.readouterr().err
    assert f"openai-compat {detail}" in stderr
    assert api_key not in stderr, "no snippet of the echoing response may be logged"
    if record_health:
        recorded = health.load().clis["openai-compat"].last_error
        assert recorded == detail and api_key not in recorded
    else:
        assert health.load() == before


def test_an_ordinary_word_configured_as_a_key_does_not_corrupt_memories(monkeypatch):
    """Rewriting the key out of the text would silently damage a real memory.

    A key long enough to be treated as a secret can still read like ordinary
    prose, so the answer is discarded whole rather than edited in place.
    """
    api_key = "deployment-notes"

    def respond(request):
        content = json.dumps([{"type": "decision", "content": f"Keep the {api_key} in the runbook."}])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_endpoint(monkeypatch, respond)
    with pytest.raises(OpenAICompatError):
        call_openai_compat("extract", model="m", base_url="https://llm.test/v1", api_key=api_key)


def test_a_short_placeholder_key_never_discards_an_answer(monkeypatch):
    """Local servers take a dummy key, and seeing that word in an answer means nothing."""

    def respond(request):
        content = json.dumps([{"type": "fact", "content": "We run ollama for local models."}])
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_endpoint(monkeypatch, respond)
    text = call_openai_compat("extract", model="m", base_url="http://localhost:11434/v1", api_key="ollama")
    assert "We run ollama for local models." in text


def test_parse_json_array_drops_items_whose_fields_are_not_text():
    """One malformed item from the model must not end the whole capture pass."""
    raw = json.dumps(
        [
            {"type": "fact", "content": 123},
            {"type": "fact", "content": {"nested": "object"}},
            {"type": "fact", "content": None},
            {"type": 7, "content": "A numeric type falls back to fact."},
            {"type": "decision", "content": "We keep the good item."},
        ]
    )
    assert parse_json_array(raw) == [
        {"type": "fact", "content": "A numeric type falls back to fact."},
        {"type": "decision", "content": "We keep the good item."},
    ]


@pytest.mark.parametrize("record_health", [True, False])
@pytest.mark.parametrize("items", [[], [{"type": "fact", "content": "Use pytest."}]])
def test_openai_compat_success_health_and_timeout(monkeypatch, remote_backend, record_health, items):
    health.record_failure("openai-compat", "HTTP 401: authentication failed")
    before = health.load()

    def respond(request):
        # The budget covers the whole call, so the fallback gets what is left of it.
        assert all(0 < value <= 20 for value in request.extensions["timeout"].values())
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(items)}}]})

    mock_endpoint(monkeypatch, respond)
    assert (
        call_llm("extract", transcript_path=None, cfg=remote_backend, host_timeout_s=20, record_health=record_health)
        == items
    )
    assert health.load() == (health.BackendHealth() if record_health else before)


@pytest.mark.parametrize("record_health", [True, False])
@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        (401, "HTTP 401: authentication failed"),
        (403, "HTTP 403: access denied"),
        (429, "HTTP 429: rate limit exceeded"),
        (500, "HTTP 500: server error"),
        (httpx.ConnectError, "connection failed (ConnectError)"),
        (httpx.ReadTimeout, "timed out after 20s (ReadTimeout)"),
        ("invalid-json", "response is not valid JSON"),
        ("missing-content", "response has no text in choices[0].message.content"),
        ("null-content", "response has no text in choices[0].message.content"),
        ("blank-content", "response has no text in choices[0].message.content"),
    ],
)
def test_openai_compat_failure_is_specific(monkeypatch, capsys, remote_backend, record_health, failure, detail):
    health.record_failure("claude", "previous failure")
    before = health.load()

    def respond(request):
        if isinstance(failure, type):
            raise failure("private endpoint and credentials", request=request)
        if isinstance(failure, int):
            return httpx.Response(failure, json={"error": {"message": "private endpoint and credentials"}})
        if failure == "invalid-json":
            return httpx.Response(200, text="private endpoint and credentials")
        if failure in ("null-content", "blank-content"):
            content = None if failure == "null-content" else "   "
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        return httpx.Response(200, json={"choices": []})

    mock_endpoint(monkeypatch, respond)
    assert (
        call_llm("extract", transcript_path=None, cfg=remote_backend, host_timeout_s=20, record_health=record_health)
        == []
    )
    stderr = capsys.readouterr().err
    assert f"openai-compat {detail}" in stderr
    assert "private endpoint and credentials" not in stderr
    assert "returned no text" not in stderr
    assert "test-key" not in stderr
    if record_health:
        recorded = health.load().clis["openai-compat"].last_error
        assert recorded == detail
        assert "test-key" not in recorded
    else:
        assert health.load() == before


def test_openai_compat_output_we_cannot_parse_does_not_mark_the_backend_healthy(monkeypatch, capsys, remote_backend):
    """Text back is not proof the fallback works: a server can return an error page as prose."""
    health.record_failure("openai-compat", "HTTP 500: server error")
    before = health.load()

    def respond(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "Sorry, I cannot help with that."}}]})

    mock_endpoint(monkeypatch, respond)
    assert call_llm("extract", transcript_path=None, cfg=remote_backend) == []
    assert "openai-compat output yielded 0 items" in capsys.readouterr().err
    assert health.load() == before


@pytest.mark.parametrize("cli_is_slow", [True, False])
def test_the_fallback_gets_what_is_left_of_the_budget(monkeypatch, capsys, remote_backend, cli_is_slow):
    """Both backends share one budget, so a capture pass cannot outlive its lock.

    A host CLI that burns the whole budget leaves nothing to call the fallback
    with; one that fails immediately leaves almost all of it.
    """
    budget = 0.2
    monkeypatch.setattr("poppy.consolidation.detect_host_cli", lambda _: "claude")
    monkeypatch.setattr("poppy.consolidation.MIN_FALLBACK_TIMEOUT_S", 0.01)
    timeouts = []

    def cli(prompt, *, cli, timeout_s, record_health=True):
        if cli_is_slow:
            time.sleep(timeout_s)
        return None

    def respond(request):
        timeouts.append(max(request.extensions["timeout"].values()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}}]})

    monkeypatch.setattr("poppy.consolidation.call_host_cli", cli)
    mock_endpoint(monkeypatch, respond)
    assert call_llm("extract", transcript_path=None, cfg=remote_backend, host_timeout_s=budget) == []
    stderr = capsys.readouterr().err
    if cli_is_slow:
        assert timeouts == [], "the fallback must not start a request the budget cannot cover"
        assert f"openai-compat skipped, claude used the {budget}s budget" in stderr
    else:
        assert timeouts and timeouts[0] <= budget
        assert "skipped" not in stderr


def test_is_enabled_respects_env(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    cfg = PoppyConfig(poppy_dir=tmp_path)
    assert is_enabled(cfg) is False
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    assert is_enabled(cfg) is True


def test_is_enabled_respects_config(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    # A legacy consolidate-enabled true is grandfathered as consent, and
    # default-on then needs a free host-CLI backend present.
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: True)
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_enabled=True)
    assert is_enabled(cfg) is True
    # An explicit opt-out stays off even with a backend present.
    assert is_enabled(PoppyConfig(poppy_dir=tmp_path, consent="denied")) is False


def test_is_enabled_respects_per_project_off_switch(tmp_path, monkeypatch):
    """A deny-listed project is off even when global capture is on."""
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: True)
    cfg = PoppyConfig(poppy_dir=tmp_path, consent="granted", disabled_projects=["client-secret"])
    assert is_enabled(cfg, project="client-secret") is False  # gated by the deny-list
    assert is_enabled(cfg, project="my-oss") is True  # other projects unaffected
    assert is_enabled(cfg) is True  # no project context → global state


def test_parse_json_array_strips_fences():
    raw = '```json\n[{"type": "decision", "content": "use ruff"}]\n```'
    assert parse_json_array(raw) == [{"type": "decision", "content": "use ruff"}]


def test_parse_json_array_handles_malformed():
    assert parse_json_array("not json at all") == []
    assert parse_json_array("[]") == []
    assert parse_json_array('[{"type": "fact"}]') == []  # missing content


def test_parse_json_array_normalizes_unknown_type():
    assert parse_json_array('[{"type": "weird", "content": "x"}]') == [{"type": "fact", "content": "x"}]


def test_format_transcript_respects_char_budget():
    messages = [{"role": "user", "text": "x" * 5000}, {"role": "assistant", "text": "y" * 5000}]
    out = format_transcript(messages, char_budget=2000)
    # First message gets truncated to 1500 chars; budget stops the second.
    assert len(out) <= 2000 + 200
    assert "USER:" in out


def test_detect_host_cli_prefers_claude_for_claude_transcripts(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "claude" else None)
    assert detect_host_cli("/Users/x/.claude/projects/abc/session.jsonl") == "claude"


def test_detect_host_cli_uses_cursor_agent_for_cursor_transcripts(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "cursor-agent" else None)
    assert detect_host_cli(str(CURSOR_FIXTURE)) == "cursor-agent"


def test_detect_host_cli_returns_none_when_no_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert detect_host_cli("/some/path.jsonl") is None


@pytest.mark.parametrize("transcript_path", [None, ""])
@pytest.mark.parametrize("cli_name", ["claude", "cursor-agent", "codex", "gemini", None])
def test_detect_host_cli_without_transcript(monkeypatch, transcript_path, cli_name):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == cli_name else None)
    assert detect_host_cli(transcript_path) == cli_name


def test_detect_host_cli_without_transcript_prefers_claude(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    assert detect_host_cli(None) == "claude"


@pytest.mark.parametrize("cli_name", ["claude", "cursor-agent", "codex", "gemini"])
def test_call_host_cli_suppresses_hooks_in_child_environment(monkeypatch, cli_name):
    calls = []

    class Result:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr("poppy.consolidation.subprocess.run", lambda *args, **kwargs: calls.append(kwargs) or Result())

    assert call_host_cli("extract", cli=cli_name) == "[]"
    assert calls[0]["env"]["POPPY_SUPPRESS_HOOKS"] == "1"


def _host_cli_stub(monkeypatch, stdout: str, *, returncode: int = 0):
    """Make `claude` the detected host CLI and hand its process the given output.

    Both stand-ins are bound onto ``poppy.consolidation`` itself rather than onto
    the shared ``shutil`` / ``subprocess`` module objects, so no other importer
    sees a stubbed PATH lookup or a stubbed process spawn.
    """
    result = SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(
        "poppy.consolidation.shutil",
        SimpleNamespace(which=lambda name: f"/usr/bin/{name}" if name == "claude" else None),
    )
    monkeypatch.setattr(
        "poppy.consolidation.subprocess",
        SimpleNamespace(run=lambda *args, **kwargs: result, TimeoutExpired=subprocess.TimeoutExpired),
    )


@pytest.mark.parametrize("stdout", ["[]", "```json\n[]\n```", "  [ ]  "])
def test_call_llm_never_spends_on_remote_when_host_cli_returns_empty(tmp_path, monkeypatch, stdout):
    """An empty array from the host CLI ends the extraction; it is not a failure.

    Most capture windows hold nothing durable, so `[]` is the common answer. Falling
    through to the openai-compat backend would auto-spend on a paid remote model on
    the majority of capture cycles for anyone with both backends configured.
    """
    _host_cli_stub(monkeypatch, stdout)
    remote_calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_openai_compat",
        lambda prompt, **kwargs: remote_calls.append(kwargs) or '[{"type": "fact", "content": "paid"}]',
    )
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_model="gpt-4o-mini", consolidate_api_key="sk-test")

    assert call_llm("extract", transcript_path="/Users/x/.claude/projects/abc/session.jsonl", cfg=cfg) == []
    assert remote_calls == []


@pytest.mark.parametrize(
    "stdout",
    [
        '["Use PostgreSQL for persistent storage."]',  # bare strings, not objects
        '[{"type": "decision", "text": "Use PostgreSQL."}]',  # no "content" key
        '[{"type": "fact", "content": "   "}]',  # blank content
    ],
)
def test_call_llm_falls_back_to_remote_when_host_array_is_unusable(tmp_path, monkeypatch, capsys, stdout):
    """A non-empty array the parser cannot use is a broken answer, not "nothing durable".

    Only a genuinely empty array means the host CLI found nothing. When its output
    shape drifts from the prompt the array is non-empty but yields no memories, and
    the remote fallback has to stay reachable or the drift captures zero memories
    every cycle with a log line that claims there was nothing to keep.
    """
    _host_cli_stub(monkeypatch, stdout)
    remote_calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_openai_compat",
        lambda prompt, **kwargs: remote_calls.append(kwargs) or '[{"type": "decision", "content": "from remote"}]',
    )
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_model="gpt-4o-mini", consolidate_api_key="sk-test")

    items = call_llm("extract", transcript_path="/Users/x/.claude/projects/abc/session.jsonl", cfg=cfg)
    assert items == [{"type": "decision", "content": "from remote"}]
    assert len(remote_calls) == 1
    assert "unparseable as JSON array" in capsys.readouterr().err


def test_call_llm_still_falls_back_to_remote_when_host_cli_fails(tmp_path, monkeypatch):
    """The fallback survives for the case it exists for: no usable host-CLI answer."""
    _host_cli_stub(monkeypatch, "", returncode=1)
    remote_calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_openai_compat",
        lambda prompt, **kwargs: remote_calls.append(kwargs) or '[{"type": "fact", "content": "from remote"}]',
    )
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_model="gpt-4o-mini", consolidate_api_key="sk-test")

    items = call_llm("extract", transcript_path="/Users/x/.claude/projects/abc/session.jsonl", cfg=cfg)
    assert items == [{"type": "fact", "content": "from remote"}]
    assert len(remote_calls) == 1


def test_compact_event_id_is_stable_per_summary():
    a = _compact_event_id("sess1", "summary X")
    b = _compact_event_id("sess1", "summary X")
    c = _compact_event_id("sess1", "summary Y")
    assert a == b
    assert a != c
    assert a.startswith("sess1:compact:")


def test_consolidate_compact_event_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    payload = {
        "session_id": "s1",
        "compact_summary": "We decided to use ruff for formatting.",
        "cwd": "/tmp/poppy",
    }
    assert consolidate_compact_event(payload) == 0


def test_consolidate_compact_event_skips_without_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    assert consolidate_compact_event({"session_id": "s1", "cwd": "/tmp/poppy"}) == 0
    assert consolidate_compact_event({"compact_summary": "x" * 100, "cwd": "/tmp/poppy"}) == 0


def test_consolidate_compact_event_stores_and_is_idempotent(tmp_path, monkeypatch):
    """Stub the LLM, run twice with the same payload, expect 1 fire only."""
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))

    fake_items = [{"type": "decision", "content": "Use ruff for formatting."}]
    calls = {"n": 0}

    def fake_call_llm(prompt, *, transcript_path, cfg):
        calls["n"] += 1
        return fake_items

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)

    payload = {
        "session_id": "s1",
        "compact_summary": "User and assistant agreed to adopt ruff for code formatting.",
        "cwd": str(tmp_path / "poppy"),
        "transcript_path": "/dev/null",
    }
    n1 = consolidate_compact_event(payload)
    n2 = consolidate_compact_event(payload)
    assert n1 == 1
    assert n2 == 0  # idempotent re-fire
    assert calls["n"] == 1  # second call short-circuited before LLM


def test_consolidate_compact_event_distinct_summaries_both_fire(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))

    # Distinct content per fire so the dedup-on-capture reconciler does
    # not skip the second as a duplicate — this test guards the per-compact-event
    # idempotency key, not the no-dedup behaviour the reconciler replaces.
    contents = [
        "The build uses Cargo for the Rust workspace crates.",
        "Deployments run on Kubernetes via Helm charts in us-east-1.",
    ]
    calls = {"n": 0}

    def fake_call_llm(prompt, *, transcript_path, cfg):
        out = [{"type": "fact", "content": contents[calls["n"] % len(contents)]}]
        calls["n"] += 1
        return out

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)

    base = {"session_id": "s1", "cwd": str(tmp_path / "poppy"), "transcript_path": "/dev/null"}
    n1 = consolidate_compact_event({**base, "compact_summary": "first compact"})
    n2 = consolidate_compact_event({**base, "compact_summary": "second compact"})
    assert n1 == 1
    assert n2 == 1


def test_capture_scopes_project_by_marker_not_basename(tmp_path, monkeypatch):
    """Auto-capture must scope memories with the shared marker-walking resolver,
    not the naive ``Path(cwd).name``.

    Regression for the mis-scoping bug: a capture in a directory that is not a
    project root (``~/code/personal``, ``~/scratch``) was tagged with the bare
    basename, so the memory never resurfaced under recall's marker-walked scope.
    """
    from poppy.capture import journal

    store = tmp_path / "store"
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(store))
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"type": "fact", "content": "The store is postgres."}],
    )

    # A bare directory with no project marker anywhere up the chain: the project
    # tag must be None, not "personal".
    bare = tmp_path / "code" / "personal"
    bare.mkdir(parents=True)
    consolidate_compact_event(
        {"session_id": "bare", "compact_summary": "one", "cwd": str(bare), "transcript_path": "/dev/null"}
    )
    assert journal.read_all(store)[-1]["project"] is None

    # A directory carrying a marker is tagged with that directory's name.
    marked = tmp_path / "code" / "myrepo"
    (marked / "src").mkdir(parents=True)
    (marked / "pyproject.toml").write_text("")
    consolidate_compact_event(
        {"session_id": "marked", "compact_summary": "two", "cwd": str(marked / "src"), "transcript_path": "/dev/null"}
    )
    assert journal.read_all(store)[-1]["project"] == "myrepo"


def test_consolidate_stop_event_stamps_source_from_host_cli(tmp_path, monkeypatch):
    """A Codex session's captured memory is stamped `codex`, not `claude-code`.

    The bug hardcoded ``source_type="claude-code"`` at every capture site even
    though ``detect_host_cli`` already classifies the host CLI. The transcript
    uses Codex's rollout shape and only ``codex`` resolves on PATH, so provenance
    must follow the real client without relying on the directory name.
    """
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    # Only `codex` is on PATH — detect_host_cli must classify this as codex.
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "codex" else None)

    # A real Codex rollout shape under a Codex path.
    codex_dir = tmp_path / ".codex" / "sessions"
    codex_dir.mkdir(parents=True)
    transcript = codex_dir / "rollout.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"type": "session_meta", "id": "codex-sess-1"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "How should we format Python?"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Use ruff for formatting and linting."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "And for the package manager?"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Standardize on uv."}],
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows))

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"type": "decision", "content": "Use ruff for Python formatting."}],
    )

    # A marked project dir so project_from_cwd tags it (it returns None without a marker).
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text("")

    payload = {
        "session_id": "codex-sess-1",
        "transcript_path": str(transcript),
        "cwd": str(proj),
    }
    stored = consolidate_stop_event(payload)
    assert stored == 1

    engine = get_engine(get_poppy_dir())
    memories = engine.list_all(filters=Filters(project="myproj"), limit=10)
    assert len(memories) == 1
    assert memories[0].source.type == "codex"
