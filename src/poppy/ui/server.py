"""FastAPI server for the Poppy memory management UI.

Bound to localhost only. Reads/writes go through the active RetrievalEngine
(the same one MCP and CLI use, per runtime.get_engine), so all surfaces stay
in sync. Soft-deletes are recorded in the sidecar TombstoneStore.
"""

from __future__ import annotations

import html
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters, Memory
from poppy.paths import ensure_poppy_dir
from poppy.runtime import get_engine, get_fast_engine, get_poppy_dir

from .tombstones import TTL_DAYS, Tombstone, TombstoneStore

STATIC_DIR = Path(__file__).parent / "static"

# Hostnames the UI answers to by default. TrustedHostMiddleware compares the
# Host header (port-stripped) against this list, so a DNS-rebinding attacker who
# resolves a name they control to 127.0.0.1 is rejected before reaching the
# delete/edit API. `poppy ui --allow-remote` widens this to ["*"] deliberately.
LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "::1", "[::1]"]

# Content-Security-Policy for the local UI. Scripts and styles both load from
# external files (no inline <script>/<style>), so both are locked to
# same-origin with no 'unsafe-inline'. This turns any stored-content injection
# that slips past escaping into inert text rather than an executing script,
# and blocks exfiltration to third-party origins.
CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'; "
    "object-src 'none'"
)


class MemoryOut(BaseModel):
    id: str
    content: str
    memory_type: str
    project: str | None
    source_type: str
    source_session_id: str | None
    source_timestamp: str
    confidence: float
    related_to: list[str]
    created_at: str
    updated_at: str
    score: float | None = None
    tombstoned: bool = False
    tombstoned_at: str | None = None
    # `expires_at` is the memory's own TTL (or None for permanent memories).
    # `tombstone_expires_at` is when a tombstone auto-purges (set only when
    # tombstoned=True). The two are unrelated.
    expires_at: str | None = None
    tombstone_expires_at: str | None = None
    # Reverse pointer: when this memory was tombstoned by a supersede, the new
    # memory's id lives here. None for plain deletions and active memories.
    superseded_by: str | None = None

    @classmethod
    def from_memory(cls, m: Memory, score: float | None = None) -> "MemoryOut":
        return cls(
            id=m.id,
            content=m.content,
            memory_type=m.memory_type,
            project=m.project,
            source_type=m.source.type,
            source_session_id=m.source.session_id,
            source_timestamp=m.source.timestamp.isoformat(),
            confidence=m.confidence,
            related_to=m.related_to,
            created_at=m.created_at.isoformat(),
            updated_at=m.updated_at.isoformat(),
            expires_at=m.expires_at.isoformat() if m.expires_at else None,
            score=score,
        )

    @classmethod
    def from_tombstone(cls, t: Tombstone) -> "MemoryOut":
        out = cls.from_memory(t.memory)
        out.tombstoned = True
        out.tombstoned_at = t.tombstoned_at.isoformat()
        out.tombstone_expires_at = t.expires_at.isoformat()
        out.superseded_by = t.superseded_by
        return out


class MemoryPatch(BaseModel):
    content: str | None = None
    memory_type: str | None = None
    project: str | None = Field(default=None)
    clear_project: bool = False
    ttl: str | None = None
    expires_at: str | None = None
    clear_expiry: bool = False


class SupersedeBody(BaseModel):
    content: str
    memory_type: str = "fact"
    project: str | None = None
    ttl: str | None = None
    expires_at: str | None = None


class _Context:
    """Holds engines and the tombstone store.

    Reads use the fast (FTS5-only) engine — instant startup, no model download.
    Writes lazily instantiate the heavy engine so the UI stays in sync with the
    BloomEngine's embedding index that MCP/CLI rely on. First write pays the
    model-load cost; subsequent writes are instant.
    """

    def __init__(
        self,
        poppy_dir: Path,
        fast_engine: RetrievalEngine,
        tombstones: TombstoneStore,
    ) -> None:
        self.poppy_dir = poppy_dir
        self.fast_engine = fast_engine
        self.tombstones = tombstones
        self._writer: RetrievalEngine | None = None

    @property
    def reader(self) -> RetrievalEngine:
        return self.fast_engine

    def writer(self) -> RetrievalEngine:
        if self._writer is None:
            # Heavy engine: BGE-small bi-encoder + ms-marco cross-encoder.
            # Loading it triggers a one-off ~160MB model download on first ever run.
            self._writer = get_engine(self.poppy_dir)
        return self._writer


def create_app(poppy_dir: Path | None = None, allowed_hosts: list[str] | None = None) -> FastAPI:
    poppy_dir = poppy_dir or get_poppy_dir()
    ensure_poppy_dir(poppy_dir)
    db_path = poppy_dir / "memories.db"

    # Reads use the fast (FTS5-only, no models) engine — instant startup.
    # Writes lazily instantiate the BloomEngine so the embedding index that
    # MCP/CLI rely on stays in sync. First write pays the model-load cost.
    from poppy.sync import remote_is_configured

    fast_engine = get_fast_engine(poppy_dir)
    tombstones = TombstoneStore(db_path)
    # With a remote configured, a week-old tombstone may still be waiting to be
    # pushed, and dropping it would leave the cloud row live for the next pull to
    # re-ingest the forgotten memory — so only `sync`, which knows what it sent,
    # may purge one. With no remote there is nowhere for it to travel, nothing is
    # waiting on it, and the seven-day window applies on age alone; otherwise a
    # user who never enables sync would keep every deletion for ever.
    purged = tombstones.purge_expired(
        pushed_through=None if remote_is_configured(poppy_dir) else datetime.now(timezone.utc).isoformat()
    )
    if purged:
        print(f"[poppy ui] purged {purged} expired tombstone(s)")

    ctx = _Context(poppy_dir=poppy_dir, fast_engine=fast_engine, tombstones=tombstones)

    def _autosync() -> None:
        """Queue a background sync after a dashboard write, matching the CLI and
        MCP paths. Without this, dashboard edits/deletes sit unpushed until some
        other sync fires — including tombstones, so a dashboard delete would not
        propagate to other devices."""
        from poppy.sync.auto import trigger as _trigger_autosync

        _trigger_autosync(ctx.poppy_dir)

    app = FastAPI(title="Poppy", docs_url=None, redoc_url=None, openapi_url=None)

    # Reject non-allowlisted Host headers (DNS-rebinding defense). `["*"]` (set
    # by `poppy ui --allow-remote`) disables the check for deliberate exposure.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or LOOPBACK_HOSTS)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP_POLICY)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.get("/api/memories")
    def list_memories(
        q: str | None = None,
        type: str | None = None,
        project: str | None = None,
        source: str | None = None,
        scope: str = "active",
        limit: int = 200,
        include_expired: bool = False,
    ) -> dict[str, Any]:
        """List memories. scope=active|tombstoned|all."""
        if scope == "tombstoned":
            tombs = ctx.tombstones.list_public()
            items = [MemoryOut.from_tombstone(t) for t in tombs]
            items = _filter_items(items, type=type, project=project, source=source, q=q)
            return {"items": [i.model_dump() for i in items], "scope": scope}

        filters = Filters(project=project, memory_type=type, include_expired=include_expired)
        if q:
            # Use the fast FTS5 engine for live search; latency matters for typing.
            scored = ctx.fast_engine.retrieve(q, filters=filters, limit=limit)
            items = [MemoryOut.from_memory(s.memory, score=s.score) for s in scored]
        else:
            mems = ctx.reader.list_all(filters=filters, limit=limit)
            items = [MemoryOut.from_memory(m) for m in mems]

        if source:
            items = [i for i in items if i.source_type == source]

        if scope == "all":
            tombs = ctx.tombstones.list_public()
            tomb_items = _filter_items(
                [MemoryOut.from_tombstone(t) for t in tombs],
                type=type,
                project=project,
                source=source,
                q=q,
            )
            items = items + tomb_items

        return {"items": [i.model_dump() for i in items], "scope": scope}

    @app.get("/api/memories/{memory_id}")
    def get_memory(memory_id: str) -> dict[str, Any]:
        m = ctx.reader.get_public(memory_id)
        if m is not None:
            return MemoryOut.from_memory(m).model_dump()
        t = ctx.tombstones.get_public(memory_id)
        if t is not None:
            return MemoryOut.from_tombstone(t).model_dump()
        raise HTTPException(status_code=404, detail="Memory not found")

    @app.patch("/api/memories/{memory_id}")
    def patch_memory(memory_id: str, patch: MemoryPatch) -> dict[str, Any]:
        from poppy.lifecycle import edit_memory, resolve_expiry

        if patch.clear_expiry and (patch.ttl or patch.expires_at):
            raise HTTPException(
                status_code=400,
                detail="clear_expiry cannot be combined with ttl or expires_at",
            )
        if patch.clear_project and patch.project is not None:
            raise HTTPException(
                status_code=400,
                detail="clear_project cannot be combined with project",
            )
        try:
            new_expiry = resolve_expiry(patch.ttl, patch.expires_at)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            result = edit_memory(
                ctx.writer(),
                memory_id,
                content=patch.content,
                memory_type=patch.memory_type,
                project=patch.project,
                project_unset=patch.clear_project,
                expires_at=new_expiry,
                clear_expiry=patch.clear_expiry,
                poppy_dir=ctx.poppy_dir,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _autosync()
        return MemoryOut.from_memory(result.memory).model_dump()

    @app.post("/api/memories/{memory_id}/supersede")
    def supersede_endpoint(memory_id: str, body: SupersedeBody) -> dict[str, Any]:
        """Tombstone the old memory and ingest a new one in its place."""
        from poppy.lifecycle import resolve_expiry, supersede_memory
        from poppy.models import Source
        from poppy.write_flow import make_memory_id

        try:
            new_expiry = resolve_expiry(body.ttl, body.expires_at)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        now = datetime.now(timezone.utc)
        new_memory = Memory(
            id=make_memory_id(),
            content=body.content,
            memory_type=body.memory_type,
            source=Source(type="ui", session_id=None, timestamp=now),
            project=body.project,
            related_to=[],
            created_at=now,
            updated_at=now,
            expires_at=new_expiry,
        )
        try:
            result = supersede_memory(ctx.writer(), new_memory, memory_id, poppy_dir=ctx.poppy_dir)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _autosync()
        return {
            "new": MemoryOut.from_memory(new_memory).model_dump(),
            "supersedes": result.old_id,
            "tombstoned": True,
        }

    @app.delete("/api/memories/{memory_id}")
    def delete_memory(memory_id: str) -> dict[str, Any]:
        from poppy.write_flow import forget

        # Existence is read through the fast FTS reader; the delete goes through
        # the heavy write engine so the embedding index MCP/CLI rely on stays in
        # sync. The shared flow tombstones-before-delete and triggers autosync,
        # matching the CLI and MCP paths.
        result = forget(
            ctx.writer(),
            ctx.poppy_dir,
            memory_id,
            reader=ctx.reader,
            tombstones=ctx.tombstones,
        )
        if result.deleted and result.tombstone is None:
            # A content-free legacy deletion reports no snapshot to restore and
            # no memory to render, by design. Answered before the check below so
            # that a delete which did happen is not reported as a miss.
            return {"ok": True, "restorable": False}
        if result.memory is None:
            if result.already_tombstoned:
                # Idempotent: the row is already gone.
                return {"ok": True, "already_tombstoned": True}
            raise HTTPException(status_code=404, detail="Memory not found")
        ts = result.tombstone
        if ts is None:
            # A deletion that lost a race cannot promise a restore window.
            return {"ok": True, "restorable": False}
        return {
            "ok": True,
            "tombstoned_at": ts.tombstoned_at.isoformat(),
            "expires_at": ts.expires_at.isoformat(),
            "ttl_days": TTL_DAYS,
        }

    @app.post("/api/memories/{memory_id}/restore")
    def restore_memory(memory_id: str) -> dict[str, Any]:
        from poppy.write_flow import restore

        # The shared flow re-ingests with a fresh updated_at and queues autosync,
        # so the restore is visible to push instead of sitting below the
        # watermark and getting re-deleted by the next pull.
        try:
            result = restore(ctx.writer(), ctx.poppy_dir, memory_id, tombstones=ctx.tombstones)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result.found:
            raise HTTPException(status_code=404, detail="Tombstone not found")
        if result.expired:
            # The memory outlived its own TTL while deleted, so it is left to
            # expire rather than resurrected permanently. Its tombstone
            # stays until the restore window closes, so nothing is destroyed here.
            raise HTTPException(status_code=410, detail="memory's TTL elapsed while it was deleted")
        # A raced restore still restored the row, so it reports success like any
        # other; the stale tombstone beside it is resolved by the next sync.
        return MemoryOut.from_memory(result.memory).model_dump()

    @app.get("/api/facets")
    def facets() -> dict[str, Any]:
        # Cheap full scan — these tables are small (single-user developer memory).
        all_memories = ctx.reader.list_all(limit=100_000)
        types = Counter(m.memory_type for m in all_memories)
        projects = Counter(m.project for m in all_memories if m.project)
        sources = Counter(m.source.type for m in all_memories)
        tombstone_count = len(ctx.tombstones.list_public())
        return {
            "types": dict(types),
            "projects": dict(projects),
            "sources": dict(sources),
            "totals": {
                "active": len(all_memories),
                "tombstoned": tombstone_count,
            },
        }

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        s = ctx.reader.stats()
        all_memories = ctx.reader.list_all(limit=100_000)
        # Activity: counts per ISO date for the last 30 days.
        today = datetime.now(timezone.utc).date()
        activity: Counter[str] = Counter()
        for m in all_memories:
            d = m.created_at.date()
            delta = (today - d).days
            if 0 <= delta < 30:
                activity[d.isoformat()] += 1
        return {
            "engine": {
                "name": s.engine_name,
                "version": s.engine_version,
                "memory_count": s.memory_count,
                "storage_bytes": s.storage_bytes,
            },
            "tombstoned": len(ctx.tombstones.list_public()),
            "activity": dict(sorted(activity.items())),
            "ttl_days": TTL_DAYS,
        }

    @app.get("/api/today")
    def today() -> dict[str, Any]:
        all_memories = ctx.reader.list_all(limit=100_000)
        now = datetime.now(timezone.utc)
        today_d = now.date()

        def is_today(m: Memory) -> bool:
            return m.created_at.date() == today_d

        def is_this_week(m: Memory) -> bool:
            delta = (today_d - m.created_at.date()).days
            return 0 <= delta < 7

        todays = [m for m in all_memories if is_today(m)]
        weeks = [m for m in all_memories if is_this_week(m)]

        types_today = Counter(m.memory_type for m in todays)
        projects_today = Counter(m.project for m in todays if m.project)
        sources_today = Counter(m.source.type for m in todays)

        projects_week = Counter(m.project for m in weeks if m.project)

        # 7-day activity, oldest → newest
        activity_7d: list[dict[str, Any]] = []
        for delta in range(6, -1, -1):
            d = today_d - timedelta(days=delta)
            n = sum(1 for m in all_memories if m.created_at.date() == d)
            activity_7d.append({"date": d.isoformat(), "count": n})

        # Quiet observations: projects that have memories but none recently
        last_seen_by_project: dict[str, datetime] = {}
        for m in all_memories:
            if not m.project:
                continue
            existing = last_seen_by_project.get(m.project)
            if existing is None or m.created_at > existing:
                last_seen_by_project[m.project] = m.created_at
        quiet: list[dict[str, Any]] = []
        for proj, last in last_seen_by_project.items():
            days = (today_d - last.date()).days
            if days >= 14:
                quiet.append({"project": proj, "days_quiet": days})
        quiet.sort(key=lambda x: x["days_quiet"], reverse=True)
        quiet = quiet[:5]

        # Editorial summary line
        summary = _today_summary(
            todays_count=len(todays),
            types=types_today,
            projects=projects_today,
            sources=sources_today,
        )

        # Today's feed, oldest → newest, with a coarse "band" for the timeline rail
        feed: list[dict[str, Any]] = []
        for m in sorted(todays, key=lambda x: x.created_at):
            hour = m.created_at.astimezone().hour
            if hour < 12:
                band = "Morning"
            elif hour < 17:
                band = "Midday"
            elif hour < 21:
                band = "Afternoon"
            else:
                band = "Evening"
            feed.append({**MemoryOut.from_memory(m).model_dump(), "band": band})

        return {
            "date": today_d.isoformat(),
            "weekday": now.strftime("%A").upper(),
            "month": now.strftime("%b").upper(),
            "day": now.day,
            "iso_week": now.isocalendar().week,
            "summary": summary,
            "totals": {
                "today": len(todays),
                "this_week": len(weeks),
                "all_time": len(all_memories),
            },
            "today_breakdown": {
                "types": dict(types_today),
                "projects": dict(projects_today),
                "sources": dict(sources_today),
            },
            "top_projects_week": projects_week.most_common(5),
            "activity_7d": activity_7d,
            "quiet": quiet,
            "feed": feed,
        }

    # --- Static frontend ---
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


def _today_summary(
    *,
    todays_count: int,
    types: Counter,
    projects: Counter,
    sources: Counter,
) -> str:
    """A short editorial line for the Today hero. Generated from the day's data."""
    if todays_count == 0:
        return "Today is quiet. Nothing captured yet. Your agent's been forgetful."
    # The summary is rendered as HTML in the Today hero; the <em>/<strong> tags
    # are intentional, but every interpolated value (project/source/type) comes
    # from stored memories and is attacker-controllable, so escape each one so
    # only the literal markup below reaches the browser.
    top_type, _ = types.most_common(1)[0]
    parts = [f"Today Poppy stored <em>{todays_count}</em> memor" + ("y" if todays_count == 1 else "ies")]
    if projects:
        top_proj, _ = projects.most_common(1)[0]
        parts.append(f"mostly under <strong>{html.escape(str(top_proj))}</strong>")
    if len(sources) == 1:
        only_src = next(iter(sources))
        parts.append(f"from <strong>{html.escape(str(only_src))}</strong>")
    parts.append(f"Flavor of the day: <em>{html.escape(str(top_type))}</em>.")
    return ", ".join(parts[:-1]) + ". " + parts[-1]


def _filter_items(
    items: list[MemoryOut],
    *,
    type: str | None = None,
    project: str | None = None,
    source: str | None = None,
    q: str | None = None,
) -> list[MemoryOut]:
    out = items
    if type:
        out = [i for i in out if i.memory_type == type]
    if project:
        out = [i for i in out if i.project == project]
    if source:
        out = [i for i in out if i.source_type == source]
    if q:
        ql = q.lower()
        out = [i for i in out if ql in i.content.lower()]
    return out


def run(
    host: str = "127.0.0.1",
    port: int = 7800,
    poppy_dir: Path | None = None,
    allow_remote: bool = False,
) -> None:
    import uvicorn

    # With --allow-remote the operator has opted into LAN exposure, so drop the
    # Host allowlist (the browser's Host header will be the machine's LAN
    # IP/hostname, not a loopback name). Otherwise keep the loopback allowlist.
    allowed_hosts = ["*"] if allow_remote else None
    app = create_app(poppy_dir=poppy_dir, allowed_hosts=allowed_hosts)
    uvicorn.run(app, host=host, port=port, log_level="warning")
