import datetime as _dt
import uuid as _uuid
from types import SimpleNamespace

import pytest

from poppy.mcp_server.assembly import assemble_recall_context, lifecycle_markers
from poppy.mcp_server.server import PoppyMcpServer
from poppy.models import Memory, Source


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


@pytest.mark.asyncio
async def test_handle_remember_check_with_supersedes_is_dry_run(fast_server):
    """check_conflicts=True + explicit supersedes must report wrote=False AND not
    tombstone the target (the dry-run once fell through to a
    destructive supersede)."""
    first = await fast_server.handle_remember(content="old fact", memory_type="fact")
    target_id = first["id"]

    result = await fast_server.handle_remember(
        content="new fact", memory_type="fact", supersedes=target_id, check_conflicts=True
    )
    assert result["wrote"] is False
    assert "id" not in result
    assert fast_server._engine.get(target_id) is not None, "dry run must not tombstone the target"


@pytest.mark.asyncio
async def test_handle_remember_emits_memory_write(fast_server, monkeypatch):
    """MCP writes emit the same content-free memory_write event as the CLI now
    that telemetry lives in the shared flow (closes the MCP telemetry gap)."""
    events: list = []
    monkeypatch.setattr(
        "poppy.telemetry.capture",
        lambda poppy_dir, event, properties=None: events.append((event, properties or {})),
    )
    await fast_server.handle_remember(content="x", memory_type="fact", project="poppy", source="cursor")
    assert ("memory_write", {"memory_type": "fact", "has_project": True, "source": "cursor"}) in events


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_name,expected",
    [
        ("cursor", "cursor"),
        ("Visual Studio Code", "vscode"),
        ("codex", "codex"),
        ("gemini-cli", "gemini"),
    ],
)
async def test_clientinfo_attribution_remember_recall_round_trip(fast_server, client_name, expected):
    source = fast_server.resolve_source(_stub_ctx(client_name))
    content = f"{expected} native daemon attribution marker"

    written = await fast_server.handle_remember(content=content, memory_type="fact", source=source)
    recalled = await fast_server.handle_recall(query=content)

    assert any(memory["id"] == written["id"] for memory in recalled["memories"])
    assert fast_server._engine.get(written["id"]).source.type == expected


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


# ---------- Enriched recall context assembly ----------


def _ingest(server, content, *, project=None, created_at=None, expires_at=None, memory_type="fact"):
    created_at = created_at or _dt.datetime(2026, 6, 1, 12, 0, tzinfo=_dt.UTC)
    mem = Memory(
        id=f"mem_{_uuid.uuid4().hex[:12]}",
        content=content,
        memory_type=memory_type,
        source=Source(type="manual", session_id=None, timestamp=created_at),
        project=project,
        related_to=[],
        created_at=created_at,
        updated_at=created_at,
        expires_at=expires_at,
    )
    server._engine.ingest(mem)
    return mem


def test_lifecycle_markers_expired_and_expiring():
    now = _dt.datetime(2026, 6, 1, 12, 0, tzinfo=_dt.UTC)
    # Past expiry -> expired.
    assert lifecycle_markers(expires_at=now - _dt.timedelta(days=1), superseded=False, now=now) == ["expired"]
    # Within the 7-day window -> expires <date>.
    soon = now + _dt.timedelta(days=3)
    assert lifecycle_markers(expires_at=soon, superseded=False, now=now) == [f"expires {soon.date().isoformat()}"]
    # Far future -> no marker.
    assert lifecycle_markers(expires_at=now + _dt.timedelta(days=90), superseded=False, now=now) == []
    # No expiry, not superseded -> nothing.
    assert lifecycle_markers(expires_at=None, superseded=False, now=now) == []
    # Superseded is listed first, ahead of any expiry marker.
    assert lifecycle_markers(expires_at=now - _dt.timedelta(days=1), superseded=True, now=now) == [
        "superseded",
        "expired",
    ]


def test_assemble_recall_context_empty():
    assert assemble_recall_context([]) == "No memories found."


def test_assemble_recall_context_groups_by_project_and_orders():
    mems = [
        {
            "id": "mem_a",
            "content": "high",
            "memory_type": "fact",
            "project": "trags",
            "created_at": "2026-06-01T12:00:00+00:00",
            "score": 0.9,
            "markers": [],
        },
        {
            "id": "mem_b",
            "content": "mid",
            "memory_type": "fact",
            "project": None,
            "created_at": "2026-06-02T12:00:00+00:00",
            "score": 0.5,
            "markers": [],
        },
        {
            "id": "mem_c",
            "content": "low",
            "memory_type": "fact",
            "project": "trags",
            "created_at": "2026-06-03T12:00:00+00:00",
            "score": 0.3,
            "markers": [],
        },
    ]
    out = assemble_recall_context(mems)
    # Grouped under project headers, including the no-project fallback.
    assert "## trags" in out
    assert "## (no project)" in out
    # Date anchor + score are surfaced per memory.
    assert "2026-06-01" in out
    assert "score 0.900" in out
    # Ids preserved for follow-up.
    assert "(id: mem_a)" in out
    # Within the trags group, the higher score renders before the lower.
    assert out.index("(id: mem_a)") < out.index("(id: mem_c)")


def test_assemble_recall_context_truncates_to_budget():
    long = "x" * 100
    mems = [
        {
            "id": "mem_a",
            "content": long,
            "memory_type": "fact",
            "project": None,
            "created_at": "2026-06-01T12:00:00+00:00",
            "score": 0.5,
            "markers": [],
        }
    ]
    out = assemble_recall_context(mems, char_budget=20)
    assert "…" in out
    assert long not in out


def test_assemble_recall_context_renders_markers():
    mems = [
        {
            "id": "mem_a",
            "content": "c",
            "memory_type": "fact",
            "project": None,
            "created_at": "2026-06-01T12:00:00+00:00",
            "score": 0.5,
            "markers": ["superseded", "expired"],
        }
    ]
    out = assemble_recall_context(mems)
    assert "⚠ superseded, expired" in out


@pytest.mark.asyncio
async def test_handle_recall_enriched_fields(fast_server):
    _ingest(fast_server, "postgres pooling connection tips")
    result = await fast_server.handle_recall(query="postgres pooling")
    assert result["memories"]
    m = result["memories"][0]
    for key in ("id", "content", "memory_type", "project", "created_at", "score", "expires_at", "markers"):
        assert key in m
    assert isinstance(m["score"], (int, float))
    assert m["markers"] == []
    assert m["expires_at"] is None


@pytest.mark.asyncio
async def test_handle_recall_expiring_marker(fast_server):
    now = _dt.datetime.now(_dt.UTC)
    _ingest(fast_server, "rotate staging token soon", expires_at=now + _dt.timedelta(days=3))
    result = await fast_server.handle_recall(query="rotate staging token")
    assert result["memories"]
    markers = result["memories"][0]["markers"]
    assert any(mk.startswith("expires ") for mk in markers)


@pytest.mark.asyncio
async def test_handle_recall_expired_marker(fast_server):
    now = _dt.datetime.now(_dt.UTC)
    _ingest(fast_server, "old expired api key note", expires_at=now - _dt.timedelta(days=1))
    result = await fast_server.handle_recall(query="expired api key", include_expired=True)
    assert result["memories"], "include_expired should surface the expired memory"
    assert "expired" in result["memories"][0]["markers"]


@pytest.mark.asyncio
async def test_handle_recall_superseded_marker(fast_server):
    from poppy.ui.tombstones import TombstoneStore

    mem = _ingest(fast_server, "deploy step manual once")
    # Record the live memory as superseded in the sidecar (without deleting it
    # from the engine) so recall still surfaces it and the marker fires.
    TombstoneStore(fast_server._poppy_dir / "memories.db").add(mem, superseded_by="mem_newer")
    result = await fast_server.handle_recall(query="deploy step manual")
    assert result["memories"]
    assert "superseded" in result["memories"][0]["markers"]


@pytest.mark.asyncio
async def test_handle_recall_deterministic_order(fast_server):
    _ingest(fast_server, "alpha beta gamma one")
    _ingest(fast_server, "alpha beta gamma two")
    _ingest(fast_server, "alpha beta gamma three")
    result = await fast_server.handle_recall(query="alpha beta gamma")
    scores = [m["score"] for m in result["memories"]]
    assert scores == sorted(scores, reverse=True)


# ---------- Recall relevance floor + abstention ----------


def _set_recall_floor(poppy_dir, value):
    from poppy.config import load_config, save_config

    cfg = load_config(poppy_dir)
    cfg.recall_min_score = value
    save_config(cfg)


@pytest.mark.asyncio
async def test_recall_floor_off_returns_all(fast_server):
    for i in range(3):
        _ingest(fast_server, f"alpha beta gamma item {i}")
    result = await fast_server.handle_recall(query="alpha beta gamma")
    assert len(result["memories"]) == 3


@pytest.mark.asyncio
async def test_recall_floor_suppresses_everything_when_nothing_clears(fast_server):
    # No CE score maps to a probability >= 0.99, so a near-max floor abstains
    # entirely even for otherwise-matching memories.
    for i in range(3):
        _ingest(fast_server, f"alpha beta gamma item {i}")
    _set_recall_floor(fast_server._poppy_dir, 0.99)
    result = await fast_server.handle_recall(query="alpha beta gamma")
    assert result["memories"] == []


@pytest.mark.asyncio
async def test_recall_floor_keeps_clearly_relevant(fast_server):
    # A low floor keeps genuine matches (their probability is > 0.5).
    _ingest(fast_server, "postgres pooling connection tips")
    _set_recall_floor(fast_server._poppy_dir, 0.3)
    result = await fast_server.handle_recall(query="postgres pooling")
    assert len(result["memories"]) >= 1


@pytest.mark.asyncio
async def test_recall_floor_empty_store_returns_empty(fast_server):
    _set_recall_floor(fast_server._poppy_dir, 0.5)
    result = await fast_server.handle_recall(query="nothing here")
    assert result["memories"] == []


@pytest.mark.asyncio
async def test_recall_index_respects_floor(fast_server):
    for i in range(3):
        _ingest(fast_server, f"alpha beta gamma item {i}")
    # Off -> all surface.
    idx = await fast_server.handle_recall_index(query="alpha beta gamma")
    assert len(idx["results"]) == 3
    # Near-max floor -> abstain.
    _set_recall_floor(fast_server._poppy_dir, 0.99)
    idx = await fast_server.handle_recall_index(query="alpha beta gamma")
    assert idx["results"] == []
