from types import SimpleNamespace

import pytest

from poppy.mcp_server.server import PoppyMcpServer


def _stub_ctx(name):
    """A FastMCP-Context-shaped stub carrying clientInfo.name."""
    return SimpleNamespace(
        session=SimpleNamespace(client_params=SimpleNamespace(clientInfo=SimpleNamespace(name=name)))
    )


@pytest.fixture
def server(tmp_path):
    return PoppyMcpServer(poppy_dir=tmp_path)


@pytest.mark.asyncio
async def test_remember(server):
    result = await server.handle_remember(
        content="use Pydantic for validation",
        memory_type="preference",
        project="trags-apps",
        related_to=None,
    )
    assert "id" in result
    assert result["id"].startswith("mem_")


@pytest.mark.asyncio
async def test_recall(server):
    await server.handle_remember(content="always use Pydantic validation on FastAPI", memory_type="preference")
    await server.handle_remember(content="prefer PostgreSQL for complex queries", memory_type="preference")

    result = await server.handle_recall(query="Pydantic validation")
    assert len(result["memories"]) > 0
    assert "Pydantic" in result["memories"][0]["content"]


@pytest.mark.asyncio
async def test_recall_empty(server):
    result = await server.handle_recall(query="nonexistent")
    assert len(result["memories"]) == 0


@pytest.mark.asyncio
async def test_forget(server):
    remember_result = await server.handle_remember(content="temporary fact", memory_type="fact")
    mem_id = remember_result["id"]

    forget_result = await server.handle_forget(id=mem_id)
    assert forget_result["deleted"] is True

    recall_result = await server.handle_recall(query="temporary fact")
    assert len(recall_result["memories"]) == 0


@pytest.mark.asyncio
async def test_forget_nonexistent(server):
    result = await server.handle_forget(id="mem_nonexistent")
    assert result["deleted"] is False


@pytest.mark.asyncio
async def test_consolidate(server):
    result = await server.handle_consolidate(
        session_summary="worked on FastAPI endpoints",
        facts=["chose Pydantic for validation", "decided to use async handlers"],
    )
    # Summary + 2 facts
    assert len(result["memory_ids"]) == 3


@pytest.mark.asyncio
async def test_consolidate_facts_only(server):
    result = await server.handle_consolidate(
        session_summary="",
        facts=["chose Pydantic for validation"],
    )
    assert len(result["memory_ids"]) == 1


@pytest.mark.asyncio
async def test_recall_index_then_full(server):
    await server.handle_remember(content="prefer asyncpg over psycopg", memory_type="preference")
    await server.handle_remember(content="DB pool size should be 20", memory_type="decision")

    index = await server.handle_recall_index(query="database pool")
    assert len(index["results"]) > 0
    for r in index["results"]:
        assert "snippet" in r and "id" in r and "score" in r

    ids = [r["id"] for r in index["results"]]
    full = await server.handle_recall_full(ids=ids)
    assert len(full["memories"]) == len(ids)
    assert all("content" in m for m in full["memories"])


@pytest.mark.asyncio
async def test_context(server):
    await server.handle_remember(content="use Pydantic for validation", memory_type="preference", project="trags")
    await server.handle_remember(content="chose FastAPI", memory_type="decision", project="trags")

    result = await server.handle_context(project="trags")
    assert "Pydantic" in result["context"]
    assert "FastAPI" in result["context"]


@pytest.mark.asyncio
async def test_context_empty(server):
    result = await server.handle_context(project="nonexistent")
    assert result["context"] == ""


# ---------- MCP-side conflict detection ----------


@pytest.fixture
def fast_server(tmp_path):
    """MCP server backed by the lightweight SeedEngine (no model download)."""
    from poppy.engine.seed import SeedEngine

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    return PoppyMcpServer(poppy_dir=tmp_path, engine=engine)


@pytest.mark.asyncio
async def test_handle_remember_check_conflicts_dry_run(fast_server, monkeypatch):
    """check_conflicts=True must return candidates and write nothing."""
    await fast_server.handle_remember(content="use all-MiniLM", memory_type="decision", project="poppy")

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [
            {"id": fast_server._engine.list_all(limit=5)[0].id, "confidence": 0.91, "reason": "replaces"}
        ],
    )

    before = len(fast_server._engine.list_all(limit=10))
    result = await fast_server.handle_remember(
        content="use bge-large now",
        memory_type="decision",
        project="poppy",
        check_conflicts=True,
    )
    after = len(fast_server._engine.list_all(limit=10))

    assert result["wrote"] is False
    cs = result["conflicts"]
    assert len(cs) == 1
    assert cs[0]["confidence"] == 0.91
    assert cs[0]["reason"] == "replaces"
    assert cs[0]["content"] == "use all-MiniLM"
    assert before == after, "check_conflicts must not write"


@pytest.mark.asyncio
async def test_handle_remember_auto_supersede_path(fast_server, monkeypatch):
    """auto_supersede=True with a single high-confidence conflict triggers supersede."""
    first = await fast_server.handle_remember(content="use all-MiniLM", memory_type="decision", project="poppy")
    old_id = first["id"]

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"id": old_id, "confidence": 0.91, "reason": "replaces"}],
    )

    result = await fast_server.handle_remember(
        content="use bge-large now",
        memory_type="decision",
        project="poppy",
        auto_supersede=True,
    )
    assert result.get("supersedes") == old_id
    assert result.get("tombstoned") is True
    # And the conflicts payload is included alongside the supersede result.
    assert any(c["id"] == old_id for c in result.get("conflicts", []))
    # Old memory is gone from the engine.
    assert fast_server._engine.get(old_id) is None


# ---------- MCP write-path provenance (source app) ----------


@pytest.mark.asyncio
async def test_remember_stamps_explicit_source(fast_server):
    """A resolved source app is threaded onto the stored memory's Source.type."""
    result = await fast_server.handle_remember(content="x", memory_type="fact", source="cursor")
    mem = fast_server._engine.get(result["id"])
    assert mem.source.type == "cursor"


@pytest.mark.asyncio
async def test_remember_default_never_stamps_transport_string(fast_server):
    """With no client resolved and the default 'mcp' sentinel, never emit 'mcp'."""
    result = await fast_server.handle_remember(content="x", memory_type="fact")
    mem = fast_server._engine.get(result["id"])
    assert mem.source.type == "agent"
    assert mem.source.type != "mcp"


@pytest.mark.asyncio
async def test_consolidate_threads_source(fast_server):
    """consolidate-via-MCP must carry the same source app, not inherit 'mcp'."""
    result = await fast_server.handle_consolidate(
        session_summary="did work",
        facts=["learned a thing"],
        source="claude-code",
    )
    for mem_id in result["memory_ids"]:
        assert fast_server._engine.get(mem_id).source.type == "claude-code"


def test_server_resolve_source_reads_clientinfo(fast_server):
    """resolve_source maps a connecting client's clientInfo.name end to end."""
    assert fast_server.resolve_source(_stub_ctx("Claude Code")) == "claude-code"


def test_server_resolve_source_unidentified_is_agent(fast_server):
    assert fast_server.resolve_source(_stub_ctx(None)) == "agent"


def test_server_resolve_source_honors_configured(tmp_path):
    """An explicit --source (poppy setup) wins over a divergent live clientInfo."""
    from poppy.engine.seed import SeedEngine

    server = PoppyMcpServer(
        poppy_dir=tmp_path, engine=SeedEngine(db_path=tmp_path / "memories.db"), source="claude-code"
    )
    assert server.resolve_source(_stub_ctx("cursor")) == "claude-code"


@pytest.mark.asyncio
async def test_handle_remember_default_off_skips_llm(fast_server, monkeypatch):
    """Default mode (off) must never call the LLM."""
    await fast_server.handle_remember(content="existing", memory_type="fact", project="poppy")

    called: list[str] = []

    def boom(prompt, *, transcript_path, cfg):
        called.append(prompt)
        return []

    monkeypatch.setattr("poppy.consolidation.call_llm", boom)

    result = await fast_server.handle_remember(content="another fact", memory_type="fact", project="poppy")
    assert "id" in result
    assert "conflicts" not in result
    assert called == []


@pytest.mark.asyncio
async def test_handle_recall_since_filters_results(fast_server):
    """`since` is wired into Filters (PR #3 review): old rows drop out."""
    import datetime as dt
    import uuid

    from poppy.models import Memory, Source

    def _seed(content: str, created_at: dt.datetime) -> None:
        fast_server._engine.ingest(
            Memory(
                id=f"mem_{uuid.uuid4().hex[:12]}",
                content=content,
                memory_type="fact",
                source=Source(type="manual", session_id=None, timestamp=created_at),
                project=None,
                related_to=[],
                created_at=created_at,
                updated_at=created_at,
            )
        )

    _seed("alpha old entry", dt.datetime(2026, 5, 20, 10, 0, tzinfo=dt.UTC))
    _seed("alpha new entry", dt.datetime(2026, 6, 5, 10, 0, tzinfo=dt.UTC))

    result = await fast_server.handle_recall(query="alpha", since="2026-06-01")
    contents = [m["content"] for m in result["memories"]]
    assert "alpha new entry" in contents
    assert "alpha old entry" not in contents

    # Without since, both surface.
    result = await fast_server.handle_recall(query="alpha")
    assert len(result["memories"]) == 2


@pytest.mark.asyncio
async def test_handle_recall_since_invalid_returns_error(fast_server):
    """Invalid since input surfaces an error instead of being silently ignored."""
    result = await fast_server.handle_recall(query="anything", since="not-a-date")
    assert "error" in result
    assert "invalid since value" in result["error"]


@pytest.mark.asyncio
async def test_handle_recall_since_error_is_flag_free(fast_server):
    """The MCP-facing error names the tool parameter, never the CLI flag spelling
    ("--since"), which means nothing to an MCP client (PR #4 review)."""
    result = await fast_server.handle_recall(query="anything", since="not-a-date")
    assert "--since" not in result["error"]
    assert "since" in result["error"]
