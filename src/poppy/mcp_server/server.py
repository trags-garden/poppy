import datetime
from collections.abc import Callable
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters, normalize_memory_type
from poppy.paths import ensure_poppy_dir
from poppy.runtime import get_engine
from poppy.sources import client_name_from_context
from poppy.sources import resolve_source as _resolve_source
from poppy.write_flow import RememberResult


def _conflicts_payload(result: RememberResult) -> list[dict]:
    """Serialize a remember flow's conflicts for the MCP JSON contract.

    A detection error surfaces as a single ``{"error": ...}`` entry (the write
    still proceeded); otherwise one entry per detected conflict.
    """
    if result.conflict_error:
        return [{"error": result.conflict_error}]
    return [
        {
            "id": c.memory.id,
            "confidence": c.confidence,
            "reason": c.reason,
            "content": c.memory.content,
        }
        for c in result.conflicts
    ]


# Output bounds: a single tool call must not dump the whole store into
# the calling agent's context. `recall` per-memory content is already budgeted
# by the assembly layer; these clamp the surfaces that layer doesn't:
# the result count on every search, the batch size on `recall_full`, and the
# per-memory content on `context` (which otherwise emits full content uncapped).
MAX_RECALL_LIMIT = 50
MAX_RECALL_FULL_IDS = 100
CONTEXT_CONTENT_BUDGET = 500


def _clamp_limit(limit: int) -> int:
    """Bound a caller-supplied result count to [1, MAX_RECALL_LIMIT]."""
    if limit < 1:
        return 1
    return min(limit, MAX_RECALL_LIMIT)


def _truncate_content(content: str, budget: int = CONTEXT_CONTENT_BUDGET) -> str:
    if len(content) > budget:
        return content[: budget - 1].rstrip() + "…"
    return content


class PoppyMcpServer:
    """Core logic for Poppy MCP tools. Used by both MCP server and tests."""

    def __init__(
        self,
        poppy_dir: Path,
        engine: RetrievalEngine | None = None,
        source: str = "mcp",
        *,
        engine_error_sink: Callable[[str], None] | None = None,
    ) -> None:
        ensure_poppy_dir(poppy_dir)
        self._poppy_dir = poppy_dir
        engine_errors: list[str] = []
        if engine is not None:
            self._engine = engine
        elif engine_error_sink is None:
            self._engine = get_engine(poppy_dir)
        else:

            def capture_engine_error(message: str) -> None:
                engine_errors.append(message)
                engine_error_sink(message)

            self._engine = get_engine(poppy_dir, error_sink=capture_engine_error)
        self._engine_error = "; ".join(engine_errors) or None
        # The configured source app. Each editor's `poppy setup` passes its own
        # (e.g. "claude-code"); the "mcp" sentinel means "unset" and defers to the
        # live clientInfo at write time (see `resolve_source`).
        self._source = source or "mcp"

    def resolve_source(self, ctx: object) -> str:
        """Resolve the source app to stamp for a memory written over `ctx`.

        Prefers an explicit configured `--source`, else the connecting client's
        normalized `clientInfo.name`, else the generic "agent" — never "mcp".
        """
        return _resolve_source(client_name=client_name_from_context(ctx), configured=self._source)

    async def handle_remember(
        self,
        content: str,
        memory_type: str = "fact",
        project: str | None = None,
        related_to: list[str] | None = None,
        ttl: str | None = None,
        expires_at: str | None = None,
        supersedes: str | None = None,
        check_conflicts: bool = False,
        auto_supersede: bool = False,
        source: str | None = None,
    ) -> dict:
        from poppy.write_flow import remember as _remember

        # `source` is the resolved source app from the MCP tool wrapper. Direct
        # callers (tests, importers) omit it; fall back to the configured source,
        # which still avoids ever stamping the transport string "mcp".
        resolved_source = source if source is not None else _resolve_source(client_name=None, configured=self._source)
        try:
            result = _remember(
                self._engine,
                self._poppy_dir,
                content=content,
                memory_type=normalize_memory_type(memory_type),
                project=project,
                related_to=related_to,
                ttl=ttl,
                expires_at=expires_at,
                supersedes=supersedes,
                check_conflicts=check_conflicts,
                auto_supersede=auto_supersede,
                source=resolved_source,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        except KeyError as exc:
            return {"error": str(exc)}

        conflicts_payload = _conflicts_payload(result)
        if result.mode == "check":
            # Dry-run: never write.
            return {"conflicts": conflicts_payload, "wrote": False}

        if result.superseded_id:
            out = {"id": result.memory.id, "supersedes": result.superseded_id, "tombstoned": True}
        else:
            out = {"id": result.memory.id}
        if conflicts_payload:
            out["conflicts"] = conflicts_payload
        return out

    async def handle_edit(
        self,
        id: str,
        content: str | None = None,
        memory_type: str | None = None,
        project: str | None = None,
        clear_project: bool = False,
        ttl: str | None = None,
        expires_at: str | None = None,
        clear_expiry: bool = False,
    ) -> dict:
        from poppy.lifecycle import edit_memory, resolve_expiry

        try:
            new_expiry = resolve_expiry(ttl, expires_at)
        except ValueError as exc:
            return {"error": str(exc)}
        # None means "leave the type unchanged"; normalize only an explicit value.
        if memory_type is not None:
            memory_type = normalize_memory_type(memory_type)
        try:
            result = edit_memory(
                self._engine,
                id,
                content=content,
                memory_type=memory_type,
                project=project,
                project_unset=clear_project,
                expires_at=new_expiry,
                clear_expiry=clear_expiry,
                poppy_dir=self._poppy_dir,
            )
        except KeyError as exc:
            return {"error": str(exc)}
        except ValueError as exc:
            return {"error": str(exc)}
        if result.changed:
            from poppy.sync.auto import trigger as _trigger_autosync

            _trigger_autosync(self._poppy_dir)
        return {
            "id": result.memory.id,
            "changed": result.changed,
            "expires_at": result.memory.expires_at.isoformat() if result.memory.expires_at else None,
        }

    async def handle_recall(
        self,
        query: str,
        project: str | None = None,
        memory_type: str | None = None,
        since: str | None = None,
        limit: int = 10,
        include_expired: bool = False,
    ) -> dict:
        # `since` goes through the same parse path as the CLI's --since
        # (ISO date/datetime or a duration like "7d"), normalized to UTC so
        # engine string comparison against stored '+00:00' timestamps is
        # correct.
        since_dt = None
        if since is not None:
            from poppy.lifecycle import parse_since

            try:
                since_dt = parse_since(since).astimezone(datetime.UTC)
            except ValueError as exc:
                # parse_since speaks CLI ("--since"); strip the flag spelling so
                # the MCP-facing error names the tool parameter, not a CLI flag.
                return {"error": str(exc).replace("--since", "since")}
        limit = _clamp_limit(limit)
        filters = Filters(project=project, memory_type=memory_type, since=since_dt, include_expired=include_expired)
        results = self._engine.retrieve(query, filters=filters, limit=limit)

        # Relevance floor / abstention: drop candidates whose
        # cross-encoder score doesn't clear the configured probability floor, so
        # a weak-match query returns nothing rather than padding to top-k. Floor
        # 0.0 (default) keeps everything — an exact no-op. Post-retrieve filter
        # only; engine ranking is untouched.
        from poppy.config import load_config
        from poppy.relevance import clears_floor

        min_score = load_config(self._poppy_dir).recall_min_score
        results = [r for r in results if clears_floor(r.score, min_score)]

        # Enrich each result with the signals the calling agent needs to reason
        # over the context (score, date anchor, staleness markers). The engine
        # is untouched — this is assembly only.
        from poppy.mcp_server.assembly import lifecycle_markers, lookup_superseded

        now = datetime.datetime.now(datetime.UTC)
        superseded_ids = lookup_superseded(self._poppy_dir, [r.memory.id for r in results])
        memories = [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "memory_type": r.memory.memory_type,
                "project": r.memory.project,
                "confidence": r.memory.confidence,
                "created_at": r.memory.created_at.isoformat(),
                "score": round(r.score, 3),
                "expires_at": r.memory.expires_at.isoformat() if r.memory.expires_at else None,
                "markers": lifecycle_markers(
                    expires_at=r.memory.expires_at,
                    superseded=r.memory.id in superseded_ids,
                    now=now,
                ),
            }
            for r in results
        ]

        # Deterministic order: score desc, newest first on ties, then id — so
        # the same store + query always assembles the same context. Stable
        # multi-key sort applied least-significant key first.
        memories.sort(key=lambda d: d["id"])
        memories.sort(key=lambda d: d["created_at"], reverse=True)
        memories.sort(key=lambda d: d["score"], reverse=True)
        return {"memories": memories}

    async def handle_recall_index(
        self,
        query: str,
        project: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Return a compact index (id + snippet + score). Use before recall_full."""
        limit = _clamp_limit(limit)
        filters = Filters(project=project, memory_type=memory_type)
        results = self._engine.retrieve(query, filters=filters, limit=limit)

        # Same relevance floor as handle_recall; 0.0 default is a no-op.
        from poppy.config import load_config
        from poppy.relevance import clears_floor

        min_score = load_config(self._poppy_dir).recall_min_score
        results = [r for r in results if clears_floor(r.score, min_score)]
        return {
            "results": [
                {
                    "id": r.memory.id,
                    "type": r.memory.memory_type,
                    "snippet": r.memory.content[:120],
                    "score": round(r.score, 3),
                    "created_at": r.memory.created_at.isoformat(),
                }
                for r in results
            ]
        }

    async def handle_recall_full(self, ids: list[str]) -> dict:
        """Fetch full memory content for a batch of IDs."""
        # Full content is intentionally uncapped here (this is the "fetch full"
        # path), so bound the batch size instead.
        ids = ids[:MAX_RECALL_FULL_IDS]
        memories = []
        for mem_id in ids:
            m = self._engine.get_public(mem_id)
            if m is None:
                continue
            memories.append(
                {
                    "id": m.id,
                    "content": m.content,
                    "memory_type": m.memory_type,
                    "project": m.project,
                    "confidence": m.confidence,
                    "created_at": m.created_at.isoformat(),
                    "related_to": m.related_to,
                }
            )
        return {"memories": memories}

    async def handle_forget(self, id: str) -> dict:
        # The core flow tombstones-before-delete (restorable and sync-visible)
        # and triggers autosync on success.
        from poppy.write_flow import forget as _forget

        result = _forget(self._engine, self._poppy_dir, id)
        return {"deleted": result.deleted}

    async def handle_consolidate(
        self,
        session_summary: str,
        facts: list[str],
        project: str | None = None,
        source: str | None = None,
    ) -> dict:
        memory_ids = []
        if session_summary:
            r = await self.handle_remember(
                content=session_summary, memory_type="summary", project=project, source=source
            )
            memory_ids.append(r["id"])
        for fact in facts:
            r = await self.handle_remember(content=fact, memory_type="fact", project=project, source=source)
            memory_ids.append(r["id"])
        return {"memory_ids": memory_ids}

    async def handle_context(self, project: str | None = None, limit: int = 20) -> dict:
        """Top-N most recent memories scoped to a project, ranked by recency."""
        limit = _clamp_limit(limit)
        filters = Filters(project=project)
        memories = self._engine.list_all(filters=filters, limit=limit)
        if not memories:
            return {"context": ""}
        lines = [f"- [{m.memory_type}] {_truncate_content(m.content)}" for m in memories]
        return {"context": "\n".join(lines)}


def create_mcp_server(
    poppy_dir: Path | None = None,
    source: str = "mcp",
    engine: RetrievalEngine | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    json_response: bool = False,
    stateless_http: bool = False,
    engine_error_sink: Callable[[str], None] | None = None,
) -> FastMCP:
    """Create and configure the MCP server with Poppy tools."""
    poppy_dir = poppy_dir or Path.home() / ".poppy"
    handler = PoppyMcpServer(
        poppy_dir=poppy_dir,
        engine=engine,
        source=source,
        engine_error_sink=engine_error_sink,
    )

    mcp = FastMCP(
        "poppy",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        json_response=json_response,
        stateless_http=stateless_http,
    )
    # Status introspection shares this already-created engine and never constructs
    # a second holder or forces model loading.
    mcp._poppy_engine = handler._engine
    mcp._poppy_engine_error = handler._engine_error

    @mcp.tool()
    async def remember(
        ctx: Context,
        content: str,
        memory_type: str = "fact",
        project: str | None = None,
        related_to: list[str] | None = None,
        ttl: str | None = None,
        expires_at: str | None = None,
        supersedes: str | None = None,
        check_conflicts: bool = False,
        auto_supersede: bool = False,
    ) -> str:
        """Store a memory. Use for facts, decisions, preferences, or lessons learned during coding.

        Args:
            content: The memory to store
            memory_type: One of: fact, decision, preference, lesson, summary
            project: Project name (optional, for scoping)
            related_to: IDs of related memories (optional)
            ttl: Expire after duration (e.g. "30d", "12h", "1w3d") — for time-sensitive facts
            expires_at: ISO-8601 datetime to expire at (mutually exclusive with ttl)
            supersedes: ID of a memory this one replaces — old is tombstoned (restorable for 7 days)
            check_conflicts: Dry-run LLM conflict detection only — returns candidates, writes nothing
            auto_supersede: If a high-confidence conflict (>=0.85) is found, supersede that memory
        """
        result = await handler.handle_remember(
            content,
            memory_type,
            project,
            related_to,
            ttl=ttl,
            expires_at=expires_at,
            supersedes=supersedes,
            check_conflicts=check_conflicts,
            auto_supersede=auto_supersede,
            source=handler.resolve_source(ctx),
        )
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("wrote") is False:
            # check_conflicts dry-run — surface the candidates only.
            cs = result.get("conflicts") or []
            if not cs:
                return "No conflict candidates. (Nothing was written.)"
            lines = [f"{len(cs)} conflict candidate(s):"]
            for c in cs:
                lines.append(f"  {c['id']}  conf={c.get('confidence', 0):.2f}  {c.get('reason') or '—'}")
            return "\n".join(lines)
        base = (
            f"Remembered: {result['id']} (supersedes {result['supersedes']})"
            if result.get("supersedes")
            else f"Remembered: {result['id']}"
        )
        cs = result.get("conflicts") or []
        if cs:
            tail = "; ".join(f"{c['id']}@{c.get('confidence', 0):.2f}" for c in cs[:3])
            base += f"\n  ⚠ may supersede: {tail}"
        return base

    @mcp.tool()
    async def edit(
        id: str,
        content: str | None = None,
        memory_type: str | None = None,
        project: str | None = None,
        clear_project: bool = False,
        ttl: str | None = None,
        expires_at: str | None = None,
        clear_expiry: bool = False,
    ) -> str:
        """Edit a memory in place. Preserves id, created_at, and source. Bumps updated_at.

        Pass only the fields you want to change. clear_project / clear_expiry explicitly null those fields.

        Args:
            id: Memory ID to edit
            content: New content (optional)
            memory_type: New type: fact, decision, preference, lesson (optional)
            project: New project name (optional)
            clear_project: Set project to null
            ttl: Reset expiry to a duration (e.g. "30d")
            expires_at: Reset expiry to an ISO-8601 datetime
            clear_expiry: Remove expiry — make memory permanent
        """
        result = await handler.handle_edit(
            id,
            content=content,
            memory_type=memory_type,
            project=project,
            clear_project=clear_project,
            ttl=ttl,
            expires_at=expires_at,
            clear_expiry=clear_expiry,
        )
        if "error" in result:
            return f"Error: {result['error']}"
        if not result["changed"]:
            return f"No changes for {id}."
        msg = f"Updated {id}"
        if result.get("expires_at"):
            msg += f" (expires {result['expires_at']})"
        return msg

    @mcp.tool()
    async def recall(
        query: str,
        project: str | None = None,
        memory_type: str | None = None,
        since: str | None = None,
        limit: int = 10,
    ) -> str:
        """Search memories by natural language query.

        Args:
            query: What to search for
            project: Scope to a specific project (optional)
            memory_type: Filter by type: fact, decision, preference, lesson (optional)
            since: Only memories created at or after this point: ISO date (2026-06-01) or duration (7d, 1w3d)
            limit: Max results (default 10)
        """
        result = await handler.handle_recall(query, project, memory_type, since=since, limit=limit)
        if "error" in result:
            return f"Error: {result['error']}"
        if not result["memories"]:
            # Nothing in the store, or nothing cleared the relevance floor
            # — abstain rather than pad with weak matches.
            return "No relevant memories."
        from poppy.mcp_server.assembly import assemble_recall_context

        return assemble_recall_context(result["memories"])

    @mcp.tool()
    async def recall_index(
        query: str,
        project: str | None = None,
        memory_type: str | None = None,
        limit: int = 20,
    ) -> str:
        """Cheap first pass — returns IDs + snippets only. Pair with recall_full to fetch full content for relevant IDs.

        Args:
            query: What to search for
            project: Scope to a specific project (optional)
            memory_type: Filter by type (optional)
            limit: Max results (default 20)
        """
        result = await handler.handle_recall_index(query, project, memory_type, limit=limit)
        if not result["results"]:
            return "No memories found."
        lines = []
        for r in result["results"]:
            lines.append(f"{r['id']} [{r['type']}] ({r['score']}): {r['snippet']}")
        return "\n".join(lines)

    @mcp.tool()
    async def recall_full(ids: list[str]) -> str:
        """Fetch full content for a batch of memory IDs returned by recall_index. Always batch.

        Args:
            ids: Memory IDs from a prior recall_index call
        """
        result = await handler.handle_recall_full(ids)
        if not result["memories"]:
            return "No memories found for given IDs."
        lines = []
        for m in result["memories"]:
            project_tag = f" [{m['project']}]" if m["project"] else ""
            lines.append(f"{m['id']} [{m['memory_type']}]{project_tag}\n  {m['content']}")
        return "\n\n".join(lines)

    @mcp.tool()
    async def forget(id: str) -> str:
        """Delete a memory by ID.

        Args:
            id: The memory ID to delete
        """
        result = await handler.handle_forget(id)
        if result["deleted"]:
            return f"Forgotten: {id}"
        return f"Memory {id} not found."

    @mcp.tool()
    async def consolidate(
        ctx: Context,
        session_summary: str,
        facts: list[str],
        project: str | None = None,
    ) -> str:
        """Store learnings from a coding session. Call at session end to persist what you learned.

        Args:
            session_summary: Brief summary of what was done in the session (stored as a 'summary' memory)
            facts: List of facts, decisions, preferences, or lessons learned
            project: Project name (optional, for scoping)
        """
        result = await handler.handle_consolidate(
            session_summary, facts, project=project, source=handler.resolve_source(ctx)
        )
        return f"Stored {len(result['memory_ids'])} memories from session."

    @mcp.tool()
    async def context(project: str | None = None, limit: int = 20) -> str:
        """Get the top-N most recent memories for a project. Useful for session-start priming.

        Args:
            project: Scope to a specific project (optional)
            limit: Max memories to return (default 20)
        """
        result = await handler.handle_context(project, limit=limit)
        if not result["context"]:
            return "No memories stored yet."
        return result["context"]

    return mcp
