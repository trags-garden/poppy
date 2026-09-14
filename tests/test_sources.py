"""Source-app provenance vocabulary + normalization."""

from types import SimpleNamespace

import pytest

from poppy.sources import (
    DEFAULT_CAPTURE_SOURCE,
    GENERIC_AGENT,
    TRANSPORT_SENTINEL,
    client_name_from_context,
    normalize_client,
    resolve_source,
    source_from_capture_host,
    source_from_host_cli,
)

# ---------- normalize_client ----------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("claude-code", "claude-code"),
        ("Claude Code", "claude-code"),  # slug
        ("claudecode", "claude-code"),  # alias
        ("Cursor", "cursor"),
        ("codex-cli", "codex"),
        ("gemini-cli", "gemini"),
        ("Gemini CLI", "gemini"),
        ("Goose", "goose"),
        ("Windsurf", "windsurf"),
        ("vscode", "vscode"),
        ("Visual Studio Code", "vscode"),  # alias
        ("Claude Desktop", "claude-desktop"),  # slug happens to match vocab
        ("Some Unknown Client", "some-unknown-client"),  # honest slug, not dropped
    ],
)
def test_normalize_known_and_slug(raw, expected):
    assert normalize_client(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", TRANSPORT_SENTINEL, "MCP", "  mcp "])
def test_normalize_unidentified_is_none(raw):
    # Empty / missing / the transport string itself are NOT real source apps.
    assert normalize_client(raw) is None


def test_normalize_never_returns_transport_string():
    # Even if a client literally reports "mcp", we must not emit it as a source.
    assert normalize_client("mcp") != TRANSPORT_SENTINEL


# ---------- resolve_source (precedence) ----------


def test_explicit_configured_source_wins():
    # poppy setup <client> wrote --source deliberately; it is authoritative.
    assert resolve_source(client_name="cursor", configured="claude-code") == "claude-code"


def test_clientinfo_used_when_configured_is_sentinel():
    # Generic `poppy serve` (no --source) -> configured is the "mcp" sentinel ->
    # fall back to the live clientInfo. This is the bug the sentinel fallback fixes.
    assert resolve_source(client_name="Claude Code", configured=TRANSPORT_SENTINEL) == "claude-code"


def test_clientinfo_used_when_configured_missing():
    assert resolve_source(client_name="Cursor", configured=None) == "cursor"


def test_configured_used_when_client_unidentified():
    assert resolve_source(client_name=None, configured="claude-code") == "claude-code"


def test_falls_back_to_agent_never_mcp():
    # Nothing identifies the writer: a generic agent, never the transport string.
    got = resolve_source(client_name=None, configured=TRANSPORT_SENTINEL)
    assert got == GENERIC_AGENT
    assert got != TRANSPORT_SENTINEL


# ---------- client_name_from_context (defensive read) ----------


def test_client_name_from_full_context():
    ctx = SimpleNamespace(
        session=SimpleNamespace(client_params=SimpleNamespace(clientInfo=SimpleNamespace(name="claude-code")))
    )
    assert client_name_from_context(ctx) == "claude-code"


@pytest.mark.parametrize(
    "ctx",
    [
        None,
        SimpleNamespace(),  # no session
        SimpleNamespace(session=SimpleNamespace()),  # no client_params
        SimpleNamespace(session=SimpleNamespace(client_params=None)),  # not initialized
        SimpleNamespace(session=SimpleNamespace(client_params=SimpleNamespace(clientInfo=None))),
    ],
)
def test_client_name_from_context_is_defensive(ctx):
    assert client_name_from_context(ctx) is None


# ---------- source_from_host_cli ----------


@pytest.mark.parametrize(
    "cli,expected",
    [
        ("claude", "claude-code"),  # executable is `claude`; source app is `claude-code`
        ("cursor-agent", "cursor"),
        ("codex", "codex"),
        ("gemini", "gemini"),
        ("CODEX", "codex"),  # case-insensitive
    ],
)
def test_source_from_host_cli_maps_known(cli, expected):
    assert source_from_host_cli(cli) == expected


def test_source_from_host_cli_falls_back_when_unclassified():
    # No host CLI detected (no transcript path / nothing on PATH) -> historical stamp.
    assert source_from_host_cli(None) == DEFAULT_CAPTURE_SOURCE
    assert source_from_host_cli("") == DEFAULT_CAPTURE_SOURCE


def test_source_from_host_cli_slugifies_unknown():
    # An unknown-but-identifying CLI keeps its real provenance rather than being dropped.
    assert source_from_host_cli("Some Agent") == "some-agent"


def test_cursor_transcript_host_maps_to_cursor_source():
    assert source_from_capture_host("cursor", fallback_cli="claude") == "cursor"
