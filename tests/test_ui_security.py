"""Security regression tests for the local UI and MCP output bounds.

Covers stored-content escaping at every render sink (the server-side Today
summary and, statically, the app.js memory_type sinks), plus the memory_type enum
guard at the MCP write boundary. The UI answers only allowlisted Host headers,
and MCP recall, context, and recall_full bound their output. CSP and nosniff
headers are present on every UI response.

Setup-config safety lives in test_setup_claude_code.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source

# A payload that is dangerous only if rendered as live HTML.
SCRIPT_PAYLOAD = "<script>alert(1)</script>"
IMG_PAYLOAD = '<img src=x onerror="alert(1)">'


def _ingest(
    engine: SeedEngine,
    mid: str,
    content: str,
    *,
    memory_type: str = "fact",
    project: str | None = None,
    source_type: str = "manual",
) -> Memory:
    n = datetime.now(timezone.utc)
    m = Memory(
        id=mid,
        content=content,
        memory_type=memory_type,
        source=Source(type=source_type, session_id=None, timestamp=n),
        project=project,
        related_to=[],
        created_at=n,
        updated_at=n,
    )
    engine.ingest(m)
    return m


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db_path)
    fast_engine = SeedEngine(db_path=db_path)

    from poppy.ui import server as ui_server

    monkeypatch.setattr(ui_server, "get_fast_engine", lambda _d: fast_engine)
    monkeypatch.setattr(ui_server, "get_engine", lambda _d: engine)
    app = ui_server.create_app(poppy_dir=tmp_path)
    # TrustedHostMiddleware only accepts loopback hosts; TestClient's default
    # Host is "testserver", so bind the client to a loopback name.
    return TestClient(app, base_url="http://localhost")


# --- The server-rendered Today summary escapes user values ---


def test_today_summary_escapes_project(app_client: TestClient, tmp_path: Path) -> None:
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "x1", "captured a fact", project=SCRIPT_PAYLOAD)

    summary = app_client.get("/api/today").json()["summary"]
    assert SCRIPT_PAYLOAD not in summary
    assert "&lt;script&gt;" in summary


def test_today_summary_escapes_source(app_client: TestClient, tmp_path: Path) -> None:
    db = SeedEngine(db_path=tmp_path / "memories.db")
    # A single source makes the "from <source>" clause fire.
    _ingest(db, "x1", "captured a fact", source_type=IMG_PAYLOAD)

    summary = app_client.get("/api/today").json()["summary"]
    assert "<img" not in summary
    assert "&lt;img" in summary


def test_today_summary_escapes_type(app_client: TestClient, tmp_path: Path) -> None:
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "x1", "captured a fact", memory_type=IMG_PAYLOAD)

    summary = app_client.get("/api/today").json()["summary"]
    assert "<img" not in summary


# --- app.js escapes the memory_type sinks (JS-side regression net) ---


def test_app_js_escapes_memory_type_sinks() -> None:
    app_js = (Path(__file__).parent.parent / "src" / "poppy" / "ui" / "static" / "app.js").read_text()
    # No raw ${m.memory_type} interpolation remains; every sink is escaped.
    assert "${m.memory_type}" not in app_js
    assert "escapeHtml(m.memory_type)" in app_js


# --- CSP + nosniff headers are present on every response ---


def test_csp_and_nosniff_headers_present(app_client: TestClient) -> None:
    resp = app_client.get("/api/facets")
    assert resp.status_code == 200
    csp = resp.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert resp.headers.get("x-content-type-options") == "nosniff"


def test_csp_disallows_inline_script_and_style(app_client: TestClient) -> None:
    # Regression net: the UI has no inline <script>/<style>, so neither
    # directive should ever regain 'unsafe-inline'.
    resp = app_client.get("/api/facets")
    csp = resp.headers.get("content-security-policy", "")
    assert "'unsafe-inline'" not in csp
    assert "script-src 'self'" in csp
    assert "style-src 'self'" in csp


# --- Host allowlist ---


def test_untrusted_host_rejected(app_client: TestClient) -> None:
    resp = app_client.get("/api/facets", headers={"host": "attacker.example.com"})
    assert resp.status_code == 400


def test_loopback_host_with_port_accepted(app_client: TestClient) -> None:
    resp = app_client.get("/api/facets", headers={"host": "127.0.0.1:7800"})
    assert resp.status_code == 200


def test_allow_remote_disables_host_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db_path)
    from poppy.ui import server as ui_server

    monkeypatch.setattr(ui_server, "get_fast_engine", lambda _d: engine)
    monkeypatch.setattr(ui_server, "get_engine", lambda _d: engine)
    app = ui_server.create_app(poppy_dir=tmp_path, allowed_hosts=["*"])
    client = TestClient(app, base_url="http://anything.example.com")
    assert client.get("/api/facets").status_code == 200


# --- memory_type enum normalization at the MCP write boundary ---


@pytest.fixture
def mcp_server(tmp_path: Path):
    from poppy.mcp_server.server import PoppyMcpServer

    return PoppyMcpServer(poppy_dir=tmp_path, engine=SeedEngine(db_path=tmp_path / "memories.db"))


@pytest.mark.asyncio
async def test_remember_normalizes_unknown_type(mcp_server) -> None:
    r = await mcp_server.handle_remember(content="x", memory_type=IMG_PAYLOAD)
    stored = mcp_server._engine.get(r["id"])
    assert stored.memory_type == "fact"


@pytest.mark.asyncio
async def test_remember_preserves_canonical_type(mcp_server) -> None:
    # "summary" and "context" are valid even though the CLI's --type choice omits
    # them; normalization must not clobber them.
    for mtype in ("summary", "context", "decision"):
        r = await mcp_server.handle_remember(content=f"a {mtype}", memory_type=mtype)
        assert mcp_server._engine.get(r["id"]).memory_type == mtype


@pytest.mark.asyncio
async def test_edit_normalizes_unknown_type(mcp_server) -> None:
    r = await mcp_server.handle_remember(content="x", memory_type="fact")
    await mcp_server.handle_edit(id=r["id"], memory_type=SCRIPT_PAYLOAD)
    assert mcp_server._engine.get(r["id"]).memory_type == "fact"


# --- MCP output bounds ---


@pytest.mark.asyncio
async def test_recall_clamps_oversized_limit(mcp_server) -> None:
    from poppy.mcp_server.server import MAX_RECALL_LIMIT

    for i in range(MAX_RECALL_LIMIT + 20):
        await mcp_server.handle_remember(content=f"alpha memory number {i}", memory_type="fact")
    result = await mcp_server.handle_recall(query="alpha", limit=100_000)
    assert len(result["memories"]) <= MAX_RECALL_LIMIT


@pytest.mark.asyncio
async def test_context_clamps_limit_and_truncates_content(mcp_server) -> None:
    from poppy.mcp_server.server import CONTEXT_CONTENT_BUDGET, MAX_RECALL_LIMIT

    long_content = "y" * (CONTEXT_CONTENT_BUDGET + 500)
    for _ in range(MAX_RECALL_LIMIT + 10):
        await mcp_server.handle_remember(content=long_content, memory_type="fact", project="p")
    result = await mcp_server.handle_context(project="p", limit=100_000)
    lines = result["context"].split("\n")
    assert len(lines) <= MAX_RECALL_LIMIT
    # Each line is "- [fact] <content>" with content bounded to the budget.
    assert all(len(line) <= CONTEXT_CONTENT_BUDGET + 32 for line in lines)


@pytest.mark.asyncio
async def test_recall_full_caps_batch_size(mcp_server) -> None:
    from poppy.mcp_server.server import MAX_RECALL_FULL_IDS

    ids = []
    for i in range(MAX_RECALL_FULL_IDS + 30):
        r = await mcp_server.handle_remember(content=f"fact {i}", memory_type="fact")
        ids.append(r["id"])
    result = await mcp_server.handle_recall_full(ids=ids)
    assert len(result["memories"]) <= MAX_RECALL_FULL_IDS


# --- sync pull deliberately keeps freeform memory_type ---


def test_sync_pull_preserves_freeform_type() -> None:
    """wire_to_memory must NOT coerce memory_type — freeform is by design on the
    Trags side; render-time escaping is the guard, not normalization."""
    from poppy.sync.serializer import wire_to_memory

    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": "mem_freeform",
        "content": "c",
        "memory_type": "custom-cloud-type",
        "source_type": "trags-sync",
        "created_at": now,
        "updated_at": now,
        "source_timestamp": now,
        "confidence": 1.0,
        "related_to": [],
    }
    mem = wire_to_memory(row)
    assert mem.memory_type == "custom-cloud-type"
