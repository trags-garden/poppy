import datetime
import json
import os
import uuid
from pathlib import Path

import click

from poppy import __version__, telemetry
from poppy.config import load_config, save_config
from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters, Memory, Source
from poppy.runtime import get_engine as _runtime_get_engine
from poppy.runtime import get_poppy_dir as _runtime_get_poppy_dir


def _get_poppy_dir() -> Path:
    return _runtime_get_poppy_dir()


def _get_engine() -> RetrievalEngine:
    return _runtime_get_engine(_get_poppy_dir())


def _make_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _parse_since_option(since: str | None) -> datetime.datetime | None:
    """Parse a --since CLI value into a datetime, or None if not given.

    The result is normalized to UTC: engines compare it against created_at
    values stored as '+00:00' ISO strings, and SQLite string comparison is
    only correct when both sides share the same UTC offset.

    Raises click.BadParameter (usage error, exit code 2) on invalid input.
    """
    if since is None:
        return None
    from poppy.lifecycle import parse_since

    try:
        return parse_since(since).astimezone(datetime.timezone.utc)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc


@click.group()
@click.version_option(__version__, "--version", prog_name="poppy")
@click.pass_context
def cli(ctx: click.Context):
    """Poppy -- remember what matters."""
    # One-time telemetry disclosure (stderr only, never when telemetry is off,
    # never twice, never raises). Skipped for `poppy telemetry ...` itself:
    # the user is already looking at the switch.
    if ctx.invoked_subcommand != "telemetry":
        telemetry.maybe_print_first_run_notice(_get_poppy_dir())


@cli.command()
@click.argument("content")
@click.option("--type", "memory_type", default="fact", type=click.Choice(["fact", "decision", "preference", "lesson"]))
@click.option("--project", default=None, help="Project name (auto-detected from cwd if not set)")
@click.option("--ttl", default=None, help="Expire after duration (e.g. 30d, 12h, 1w3d)")
@click.option("--expires-at", default=None, help="Expire at ISO-8601 datetime (mutually exclusive with --ttl)")
@click.option("--supersedes", default=None, help="ID of a memory this one replaces (tombstones the old)")
@click.option(
    "--check-conflicts",
    is_flag=True,
    help="Run LLM conflict detection and print candidates without writing",
)
@click.option(
    "--auto-supersede",
    is_flag=True,
    help="If a high-confidence conflict is found, supersede it instead of plain ingest",
)
def remember(
    content: str,
    memory_type: str,
    project: str | None,
    ttl: str | None,
    expires_at: str | None,
    supersedes: str | None,
    check_conflicts: bool,
    auto_supersede: bool,
):
    """Store a memory."""
    from poppy.config import load_config
    from poppy.lifecycle import resolve_expiry, supersede_memory

    try:
        expiry = resolve_expiry(ttl, expires_at)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    engine = _get_engine()
    now = datetime.datetime.now(datetime.UTC)
    memory = Memory(
        id=_make_id(),
        content=content,
        memory_type=memory_type,
        source=Source(type="manual", session_id=None, timestamp=now),
        project=project,
        related_to=[],
        created_at=now,
        updated_at=now,
        confidence=1.0,
        expires_at=expiry,
    )

    # Resolve which auto-supersede mode to run.
    cfg = load_config(_get_poppy_dir())
    if check_conflicts:
        mode = "check"
    elif auto_supersede:
        mode = "auto"
    else:
        mode = cfg.auto_supersede  # off | suggest | auto

    conflicts = []
    if mode != "off" and not supersedes:
        from poppy.conflict_detection import detect_conflicts, pick_auto_supersede

        try:
            conflicts = detect_conflicts(engine, memory, cfg=cfg)
        except Exception as exc:
            click.echo(f"  (conflict detection failed: {exc})", err=True)
            conflicts = []

        if mode == "check":
            if not conflicts:
                click.echo("No conflict candidates found. (Nothing was written.)")
            else:
                click.echo(f"{len(conflicts)} conflict candidate(s):")
                for c in conflicts:
                    click.echo(f"  {c.memory.id}  conf={c.confidence:.2f}  {c.reason or '—'}")
                    click.echo(f"    {c.memory.content[:80]}")
            return

        if mode == "auto":
            picked = pick_auto_supersede(conflicts)
            if picked is not None:
                supersedes = picked.memory.id
                click.echo(f"  auto-supersede: {picked.memory.id} (confidence {picked.confidence:.2f})")

    if supersedes:
        try:
            result = supersede_memory(engine, memory, supersedes, poppy_dir=_get_poppy_dir())
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Remembered ({memory_type}): {content[:80]}")
        click.echo(f"  supersedes {result.old_id} (tombstoned, restorable for 7 days)")
    else:
        engine.ingest(memory)
        click.echo(f"Remembered ({memory_type}): {content[:80]}")

    # Privacy: send whether a project was set, never the project name itself
    # (project labels can carry client or codename information).
    telemetry.capture(
        _get_poppy_dir(),
        "memory_write",
        {"memory_type": memory_type, "has_project": memory.project is not None, "source": memory.source.type},
    )

    from poppy.sync.auto import trigger as _trigger_autosync

    _trigger_autosync(_get_poppy_dir())

    # Suggest hint after a normal write.
    if mode in ("suggest", "auto") and conflicts and not supersedes:
        click.echo(f"  ⚠ may supersede {len(conflicts)} memor{'y' if len(conflicts) == 1 else 'ies'}:")
        for c in conflicts[:3]:
            click.echo(f"    {c.memory.id}  conf={c.confidence:.2f}  {c.memory.content[:60]}")
        click.echo("  Re-run with --auto-supersede or `poppy remember --supersedes <id>`.")

    if expiry is not None:
        click.echo(f"  expires at {expiry.isoformat()}")


@cli.command()
@click.argument("query")
@click.option("--project", default=None, help="Only memories tagged with this project.")
@click.option("--type", "memory_type", default=None, help="Only memories of this type, e.g. fact or decision.")
@click.option(
    "--since",
    default=None,
    help="Only memories created at or after this point: ISO date (2026-06-01) or duration for the last N (7d, 1w3d).",
)
@click.option("--limit", default=10, type=int, help="Maximum number of results (default: 10).")
@click.option("--json", "as_json", is_flag=True, help="Print results as JSON instead of formatted text.")
@click.option("--include-expired", is_flag=True, help="Include memories whose TTL has passed.")
def recall(
    query: str,
    project: str | None,
    memory_type: str | None,
    since: str | None,
    limit: int,
    as_json: bool,
    include_expired: bool,
):
    """Search memories, ranked by relevance to QUERY."""
    since_dt = _parse_since_option(since)
    engine = _get_engine()
    filters_active = any(v is not None for v in (project, memory_type, since_dt))
    filters = Filters(project=project, memory_type=memory_type, since=since_dt, include_expired=include_expired)
    results = engine.retrieve(query, filters=filters, limit=limit)

    # Report the engine that actually served the query. This used to be the
    # hardcoded pre-registry name "local_bloom"; the active engine is whatever
    # the registry resolved (default: bloom, falling back to sprout/seed).
    telemetry.capture(
        _get_poppy_dir(),
        "recall_call",
        {
            "query_length": len(query),
            "result_count": len(results),
            "engine": engine.stats().engine_name,
        },
    )

    if not results:
        # Distinguish "the query matched nothing" from "the filters excluded
        # everything" so a --since/--project/--type miss is not misread as an
        # empty store (PR #3 review).
        if filters_active:
            click.echo("No memories match the given filters (--since/--project/--type).")
        else:
            click.echo("No memories found.")
        return

    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "id": r.memory.id,
                        "content": r.memory.content,
                        "type": r.memory.memory_type,
                        "project": r.memory.project,
                        "score": round(r.score, 3),
                        "created_at": r.memory.created_at.isoformat(),
                    }
                    for r in results
                ],
                indent=2,
            )
        )
        return

    for r in results:
        project_tag = f" [{r.memory.project}]" if r.memory.project else ""
        date_str = r.memory.created_at.strftime("%Y-%m-%d")
        click.echo(f"  {r.memory.content}")
        click.echo(f"    {r.memory.memory_type}{project_tag} | {date_str} | score: {r.score:.2f}")
        click.echo()


@cli.command("list")
@click.option("--project", default=None, help="Only memories tagged with this project.")
@click.option("--type", "memory_type", default=None, help="Only memories of this type, e.g. fact or decision.")
@click.option(
    "--since",
    default=None,
    help="Only memories created at or after this point: ISO date (2026-06-01) or duration for the last N (7d, 1w3d).",
)
@click.option("--limit", default=50, type=int, help="Maximum number of results (default: 50).")
@click.option("--json", "as_json", is_flag=True, help="Print results as JSON instead of formatted text.")
@click.option("--include-expired", is_flag=True, help="Include memories whose TTL has passed.")
def list_memories(
    project: str | None,
    memory_type: str | None,
    since: str | None,
    limit: int,
    as_json: bool,
    include_expired: bool,
):
    """List all memories, newest first."""
    since_dt = _parse_since_option(since)
    engine = _get_engine()
    filters_active = any(v is not None for v in (project, memory_type, since_dt))
    filters = Filters(project=project, memory_type=memory_type, since=since_dt, include_expired=include_expired)
    memories = engine.list_all(filters=filters, limit=limit)

    if as_json:
        rows = [
            {
                "id": m.id,
                "content": m.content,
                "type": m.memory_type,
                "project": m.project,
                "created_at": m.created_at.isoformat(),
            }
            for m in memories
        ]
        click.echo(json.dumps(rows, indent=2))
        return

    if not memories:
        # An empty result under active filters does not mean an empty store:
        # say which it is, so a --since/--project/--type miss is not misread
        # as "nothing stored" (PR #3 review).
        if filters_active:
            click.echo("No memories match the given filters (--since/--project/--type).")
        else:
            click.echo("No memories stored yet.")
        return

    for m in memories:
        project_tag = f" [{m.project}]" if m.project else ""
        date_str = m.created_at.strftime("%Y-%m-%d")
        click.echo(f"  {m.content}")
        click.echo(f"    {m.memory_type}{project_tag} | {date_str}")
        click.echo()


@cli.command()
@click.argument("memory_id")
@click.option("--yes", is_flag=True, help="Skip confirmation")
def forget(memory_id: str, yes: bool):
    """Delete a memory by ID."""
    engine = _get_engine()
    mem = engine.get(memory_id)
    if mem is None:
        click.echo(f"Memory {memory_id} not found.")
        return

    if not yes:
        click.echo(f"  {mem.content}")
        if not click.confirm("Forget this memory?"):
            click.echo("Cancelled.")
            return

    engine.delete(memory_id)
    click.echo(f"Forgotten: {memory_id}")

    from poppy.sync.auto import trigger as _trigger_autosync

    _trigger_autosync(_get_poppy_dir())


@cli.command()
@click.argument("memory_id")
@click.option("--content", default=None, help="New content")
@click.option(
    "--type",
    "memory_type",
    default=None,
    type=click.Choice(["fact", "decision", "preference", "lesson"]),
)
@click.option("--project", default=None, help="Set project (use --no-project to clear)")
@click.option("--no-project", is_flag=True, help="Clear project")
@click.option("--ttl", default=None, help="Reset TTL (e.g. 30d)")
@click.option("--expires-at", default=None, help="Reset expiry to ISO-8601 datetime")
@click.option("--no-expiry", is_flag=True, help="Clear expiry — make memory permanent")
def edit(
    memory_id: str,
    content: str | None,
    memory_type: str | None,
    project: str | None,
    no_project: bool,
    ttl: str | None,
    expires_at: str | None,
    no_expiry: bool,
):
    """Edit a memory in place. Preserves id, created_at, and source."""
    from poppy.lifecycle import edit_memory, resolve_expiry

    if no_expiry and (ttl or expires_at):
        raise click.BadParameter("--no-expiry cannot be combined with --ttl or --expires-at")
    if project and no_project:
        raise click.BadParameter("--project and --no-project are mutually exclusive")

    try:
        new_expiry = resolve_expiry(ttl, expires_at)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    engine = _get_engine()
    try:
        result = edit_memory(
            engine,
            memory_id,
            content=content,
            memory_type=memory_type,
            project=project,
            project_unset=no_project,
            expires_at=new_expiry,
            clear_expiry=no_expiry,
        )
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    if not result.changed:
        click.echo(f"No changes for {memory_id}.")
        return
    click.echo(f"Updated {memory_id}: {result.memory.content[:80]}")
    if result.memory.expires_at is not None:
        click.echo(f"  expires at {result.memory.expires_at.isoformat()}")

    from poppy.sync.auto import trigger as _trigger_autosync

    _trigger_autosync(_get_poppy_dir())


@cli.command()
@click.option("--yes", is_flag=True, help="Purge instead of preview")
def expire(yes: bool):
    """List memories whose TTL has passed; --yes to purge them."""
    engine = _get_engine()
    expired = [m for m in engine.list_all(filters=Filters(include_expired=True), limit=10_000) if m.expires_at]
    expired = [m for m in expired if m.expires_at and m.expires_at <= datetime.datetime.now(datetime.UTC)]
    if not expired:
        click.echo("No expired memories.")
        return
    for m in expired:
        click.echo(f"  {m.id}  {m.content[:70]}  (expired {m.expires_at.isoformat()})")
    if not yes:
        click.echo(f"\n{len(expired)} memory(s) expired. Re-run with --yes to purge.")
        return
    purge = getattr(engine, "purge_expired", None)
    if purge is None:
        # Engines without purge_expired: fall back to per-id delete.
        n = sum(1 for m in expired if engine.delete(m.id))
    else:
        n = purge()
    click.echo(f"Purged {n} memory(s).")
    if n > 0:
        from poppy.sync.auto import trigger as _trigger_autosync

        _trigger_autosync(_get_poppy_dir())


@cli.command()
def stats():
    """Show memory stats."""
    engine = _get_engine()
    s = engine.stats()
    click.echo(f"  Memories: {s.memory_count}")
    click.echo(f"  Engine:   {s.engine_name} v{s.engine_version}")
    click.echo(f"  Storage:  {s.storage_bytes / 1024:.1f} KB")


@cli.group(invoke_without_command=True)
@click.pass_context
def engines(ctx: click.Context):
    """List or switch the active retrieval engine.

    With no subcommand, prints the catalog: name, description, dep status
    (✓/✗), and ★ on the active engine.
    """
    if ctx.invoked_subcommand is not None:
        return
    from poppy.engine.registry import list_engines

    active = load_config(_get_poppy_dir()).engine
    rows = list_engines()
    name_w = max(len(e.name) for e in rows)
    for e in rows:
        marker = "★" if e.name == active else " "
        ok = "✓" if e.deps_ok else "✗"
        tag = "  (built-in)" if e.builtin else ""
        click.echo(f" {marker} {ok}  {e.name:<{name_w}}{tag}  {e.description}")
        if not e.deps_ok and e.deps_error:
            click.echo(f"          missing: {e.deps_error}")
    click.echo()
    click.echo(f"Active: {active}.  Switch with `poppy engines use <name>`.")


@engines.command("use")
@click.argument("name")
@click.option(
    "--migrate",
    is_flag=True,
    help="Re-embed existing memories with the new engine's bi-encoder before returning.",
)
def engines_use(name: str, migrate: bool):
    """Switch the active retrieval engine (writes to ~/.poppy/config.json).

    Unless --migrate is given, embeddings produced by the previous engine
    remain in memory_embeddings tagged with their original model_id. The new
    engine ignores them for the cosine channel (FTS-only fallback for those
    rows) until you run ``poppy migrate-engine``.
    """
    from poppy.engine.migration import stale_stats
    from poppy.engine.registry import resolve_engine

    poppy_dir = _get_poppy_dir()
    cfg = load_config(poppy_dir)
    try:
        cfg.set("engine", name)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    save_config(cfg)
    click.echo(f"Set engine = {cfg.engine}")

    # Probe the new engine to learn its model_id, then count stale rows.
    db_path = poppy_dir / "memories.db"
    try:
        new_engine = resolve_engine(cfg.engine, db_path)
    except (ImportError, ValueError) as exc:
        click.echo(f"⚠  engine {cfg.engine!r} unavailable here ({exc}); falling back at runtime.")
        return

    stats = stale_stats(db_path, new_engine.model_id)
    if stats.needs_migration > 0:
        click.echo(
            f"⚠  {stats.needs_migration} memories indexed under a different model "
            f"({stats.stale} stale, {stats.unknown} pre-tagging) will use FTS-only "
            f"recall until re-embedded."
        )
        if migrate:
            click.echo(f"Re-embedding with {new_engine.model_id} ...")
            from poppy.engine.migration import MigrateFilters
            from poppy.engine.migration import migrate as run_migrate

            with click.progressbar(length=stats.needs_migration, label="Migrating") as bar:

                def _tick(done: int, total: int) -> None:
                    bar.update(1)

                run_migrate(new_engine, db_path, MigrateFilters(), on_progress=_tick)
            click.echo("Done.")
        else:
            click.echo("Run `poppy migrate-engine` to re-embed (or rerun with --migrate).")
    click.echo("Restart any running MCP server / Claude Code session to pick up the change.")


@cli.command("migrate-engine")
@click.option("--project", default=None, help="Re-embed only memories in this project.")
@click.option("--memory-type", default=None, help="Re-embed only memories of this type.")
@click.option(
    "--since",
    default=None,
    help="Only memories created within the last N days (e.g. 7).",
    type=int,
)
@click.option(
    "--all",
    "include_compatible",
    is_flag=True,
    help="Re-embed every row, including ones already tagged with the active model.",
)
@click.option("--dry-run", is_flag=True, help="Show the count, do not re-embed.")
def migrate_engine(
    project: str | None,
    memory_type: str | None,
    since: int | None,
    include_compatible: bool,
    dry_run: bool,
):
    """Re-embed memories so their vectors match the active engine.

    Without filters: re-embeds every row whose model_id differs from the
    active engine's bi-encoder (or is NULL from a legacy DB). With filters:
    only the matching subset. Idempotent and resumable — each row is
    committed individually, so Ctrl+C and rerun continues where it left off.
    """
    from poppy.engine.migration import (
        MigrateFilters,
        count_targets,
        sweep_orphans,
    )
    from poppy.engine.migration import (
        migrate as run_migrate,
    )

    poppy_dir = _get_poppy_dir()
    engine = _runtime_get_engine(poppy_dir)
    if engine.model_id is None:
        raise click.ClickException(
            f"Active engine ({type(engine).__name__}) does not use embeddings; "
            "switch to an embedding-based engine first (e.g. `poppy engines use bloom`)."
        )

    filters = MigrateFilters(
        project=project,
        memory_type=memory_type,
        since_days=since,
        include_compatible=include_compatible,
    )
    db_path = poppy_dir / "memories.db"
    target = count_targets(db_path, engine.model_id, filters)
    click.echo(f"Target: {target} memories under model_id={engine.model_id!r}")
    if dry_run:
        return
    if target > 0:
        with click.progressbar(length=target, label="Re-embedding") as bar:

            def _tick(done: int, total: int) -> None:
                bar.update(1)

            done = run_migrate(engine, db_path, filters, on_progress=_tick)
        click.echo(f"Re-embedded {done} memories.")
    # Routine housekeeping: clear out embeddings whose ``memories`` row was
    # deleted without cascade. Done after the re-embed pass so it doesn't
    # touch anything the user might've cared about mid-run.
    swept = sweep_orphans(db_path)
    if swept:
        click.echo(f"Swept {swept} orphan embeddings (memory rows already deleted).")


@cli.group()
def config():
    """Manage Poppy configuration."""
    pass


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a config value."""
    poppy_dir = _get_poppy_dir()
    if key == "telemetry":
        normalized = value.strip().lower()
        if normalized in {"on", "true", "1", "yes"}:
            telemetry.set_enabled(poppy_dir, True)
            click.echo("Set telemetry = on")
            return
        if normalized in {"off", "false", "0", "no"}:
            telemetry.set_enabled(poppy_dir, False)
            click.echo("Set telemetry = off")
            return
        raise click.BadParameter("telemetry must be on or off")

    cfg = load_config(poppy_dir=poppy_dir)
    cfg.set(key, value)
    save_config(cfg)
    click.echo(f"Set {key} = {value}")


@cli.group("telemetry", invoke_without_command=True)
@click.pass_context
def telemetry_group(ctx: click.Context):
    """Show or change anonymous usage telemetry (status | on | off)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(telemetry_status)


@telemetry_group.command("status")
def telemetry_status():
    """Show whether telemetry is on, and why."""
    enabled, reason = telemetry.status(_get_poppy_dir())
    click.echo(f"Telemetry: {'on' if enabled else 'off'} ({reason})")
    if enabled:
        click.echo("Anonymous usage events only. Memory content, queries, and project names are never sent.")
        click.echo("Turn off with: poppy telemetry off")


@telemetry_group.command("on")
def telemetry_on():
    """Enable anonymous usage telemetry (persists in ~/.poppy/config.json)."""
    telemetry.set_enabled(_get_poppy_dir(), True)
    click.echo("Telemetry is on. Anonymous usage events only; memory content is never sent.")
    if os.environ.get("POPPY_TELEMETRY_OFF") == "1":
        click.echo("Note: POPPY_TELEMETRY_OFF=1 is set in your environment and keeps telemetry off until you unset it.")


@telemetry_group.command("off")
def telemetry_off():
    """Disable anonymous usage telemetry (persists in ~/.poppy/config.json)."""
    telemetry.set_enabled(_get_poppy_dir(), False)
    click.echo("Telemetry is off.")


@cli.command()
@click.option(
    "--source",
    default=lambda: os.environ.get("POPPY_MCP_SOURCE", "mcp"),
    help="Source label recorded for memories from this server (e.g. the client name).",
)
def serve(source: str):
    """Start the Poppy MCP server."""
    from poppy.mcp_server.server import create_mcp_server

    poppy_dir = _get_poppy_dir()
    mcp = create_mcp_server(poppy_dir=poppy_dir, source=source)
    # MCP stdio transport requires a clean JSON-RPC stream on stdout — any
    # banner text breaks strict clients (Claude desktop). Log to stderr.
    click.echo(f"Starting Poppy MCP server (storage: {poppy_dir})", err=True)
    mcp.run(transport="stdio")


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host (default: localhost only)")
@click.option("--port", default=7800, type=int, help="Bind port (default: 7800)")
@click.option("--no-open", is_flag=True, help="Don't open the browser on launch")
def ui(host: str, port: int, no_open: bool):
    """Browse and manage memories in a local web UI."""
    import threading
    import webbrowser

    # Silence HF Hub / transformers noise that surfaces when the BloomEngine lazy-loads
    # its bi-encoder + cross-encoder on the first edit. The UI doesn't load these at
    # startup anymore (reads use the FTS5-only fast engine), so users only encounter
    # this if they edit/delete/restore.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from poppy.ui.server import run as run_ui

    poppy_dir = _get_poppy_dir()
    url = f"http://{host}:{port}"
    click.echo(f"Poppy UI · {url}  ·  storage: {poppy_dir}")
    click.echo("  ctrl-c to stop")

    if not no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        run_ui(host=host, port=port, poppy_dir=poppy_dir)
    except KeyboardInterrupt:
        click.echo("\nstopped.")


@cli.command("consent")
@click.option("--enable", "action", flag_value="enable", help="Grant consent and enable automatic capture.")
@click.option("--disable", "action", flag_value="disable", help="Opt out of automatic capture (persists).")
@click.option("--status", "action", flag_value="status", default=True, help="Show consent and capture status.")
def consent(action: str):
    """Manage consent for automatic memory capture (ADR-0002).

    Auto-capture is enabled by default but stays inert until you record a
    one-time consent. Both granting and opting out persist permanently.
    """
    from poppy.capture import policy

    config = load_config(_get_poppy_dir())
    if action == "enable":
        config.consent = "granted"
        save_config(config)
        click.echo("✓ Automatic capture consent granted.")
        click.echo(policy.status_message(policy.evaluate(config)))
    elif action == "disable":
        config.consent = "denied"
        save_config(config)
        click.echo("Automatic capture disabled. This choice persists across upgrades.")
    else:
        click.echo(f"Consent: {policy.effective_consent(config).value}")
        click.echo(policy.status_message(policy.evaluate(config)))


def _maybe_prompt_consent(config, *, assume_yes: bool) -> None:
    """Record auto-capture consent during setup (ADR-0002).

    On a TTY, ask y/n and persist the answer. Non-interactively, leave consent
    pending (the SessionStart notice + ``poppy consent --enable`` carry it).
    Never re-prompts a user who already granted (incl. grandfathered) or opted out.
    """
    import sys

    from poppy.capture import policy

    if policy.effective_consent(config) is not policy.Consent.PENDING:
        return
    if assume_yes:
        config.consent = "granted"
        save_config(config)
        click.echo("✓ Automatic capture enabled.")
        return
    if sys.stdin.isatty():
        granted = click.confirm(
            "\nPoppy can automatically remember decisions and lessons from your sessions so "
            "they're there next time. Extraction runs locally using your own Claude CLI; your "
            "conversation never leaves your machine. Enable automatic capture?",
            default=True,
        )
        config.consent = "granted" if granted else "denied"
        save_config(config)
        click.echo(
            "✓ Automatic capture enabled."
            if granted
            else "Automatic capture left off (enable later with `poppy consent --enable`)."
        )
    else:
        click.echo("\nAutomatic capture is pending your consent. Run `poppy consent --enable` to turn it on.")


@cli.group()
def setup():
    """Set up integrations."""
    pass


def _print_install_paths(paths: dict, client: str) -> None:
    for label, path in paths.items():
        click.echo(f"  {label}: {path}")
    click.echo(f"\nPoppy is ready. Restart {client} to activate.")


def _install_simple_client(client: str) -> None:
    """Install Poppy MCP server (+ AGENTS.md primer when applicable) for a non-Claude-Code client."""
    from poppy.setup.claude_code import install_for_client

    paths = install_for_client(client=client)
    _print_install_paths(paths, client)
    telemetry.capture(_get_poppy_dir(), "agent_setup", {"agent": client})
    if client == "pi":
        click.echo(
            "\nPi reads MCP servers via the pi-mcp-adapter extension. If you "
            "haven't already installed it, run:\n"
            "  pi install npm:pi-mcp-adapter\n"
            "Then restart pi."
        )


@setup.command("claude-code")
@click.option(
    "--hooks/--no-hooks",
    default=True,
    help="Install lifecycle hooks. Default: enabled.",
)
@click.option(
    "--claude-md/--no-claude-md",
    default=True,
    help="Add a managed CLAUDE.md block describing Poppy tools.",
)
@click.option("--yes", is_flag=True, help="Grant auto-capture consent without prompting (non-interactive installs).")
def setup_claude_code(hooks: bool, claude_md: bool, yes: bool):
    """Install Poppy into Claude Code (MCP + hooks + CLAUDE.md primer)."""
    from poppy.setup.claude_code import install_for_client

    claude_dir = Path(os.environ["CLAUDE_CONFIG_DIR"]) if os.environ.get("CLAUDE_CONFIG_DIR") else None

    paths = install_for_client(
        client="claude-code",
        claude_config_dir=claude_dir,
        install_hooks=hooks,
        install_claude_md=claude_md,
    )
    _print_install_paths(paths, "claude-code")
    telemetry.capture(_get_poppy_dir(), "agent_setup", {"agent": "claude-code"})

    # Consent for automatic capture (ADR-0002): ask once on a TTY,
    # otherwise leave pending for the SessionStart notice + `poppy consent`.
    _maybe_prompt_consent(load_config(_get_poppy_dir()), assume_yes=yes)


@setup.command("copilot-cli")
def setup_copilot_cli():
    """Install Poppy into GitHub Copilot CLI (MCP + AGENTS.md primer)."""
    _install_simple_client("copilot-cli")


@setup.command("pi")
def setup_pi():
    """Install Poppy into Pi (MCP via pi-mcp-adapter + AGENTS.md primer)."""
    _install_simple_client("pi")


@setup.command("cursor")
def setup_cursor():
    """Install Poppy into Cursor (MCP only)."""
    _install_simple_client("cursor")


@setup.command("windsurf")
def setup_windsurf():
    """Install Poppy into Windsurf (MCP only)."""
    _install_simple_client("windsurf")


@setup.command("codex")
def setup_codex():
    """Install Poppy into Codex (MCP only)."""
    _install_simple_client("codex")


@setup.command("goose")
def setup_goose():
    """Install Poppy into Goose (MCP extension + .goosehints primer).

    Goose (Block's open-source agent) reads MCP servers from the
    ``extensions:`` block of ``~/.config/goose/config.yaml`` and global
    agent hints from ``~/.config/goose/.goosehints``.
    """
    from poppy.setup.goose import install_for_goose

    paths = install_for_goose()
    for label, path in paths.items():
        click.echo(f"  {label}: {path}")
    click.echo(
        "\nPoppy is registered as a Goose extension. Run `goose session` to "
        "start a session — Poppy will be available via the standard MCP tool surface."
    )
    telemetry.capture(_get_poppy_dir(), "agent_setup", {"agent": "goose"})


@setup.command("hermes-agent")
def setup_hermes_agent():
    """Install Poppy as a Hermes Agent (Nous Research) memory provider plugin.

    Hermes doesn't speak MCP — instead it loads memory providers from
    `~/.hermes/plugins/<name>/`. This drops the Poppy plugin + sets
    `memory.provider: poppy` in `~/.hermes/config.yaml`.
    """
    from poppy.setup.hermes import install_for_hermes

    paths = install_for_hermes()
    for label, path in paths.items():
        click.echo(f"  {label}: {path}")
    click.echo(
        "\nPoppy is the active hermes memory provider. Run `hermes memory status` "
        "to verify, then start a hermes session — it will call poppy_recall before "
        "each turn and consolidate at session end."
    )
    telemetry.capture(_get_poppy_dir(), "agent_setup", {"agent": "hermes-agent"})


@setup.command("claude-desktop")
@click.option(
    "--print-instructions",
    is_flag=True,
    help="Print the primer for Claude desktop's Instructions for Claude (no install).",
)
@click.option(
    "--print-import-prompt",
    is_flag=True,
    help="Print a prompt to paste into a Claude chat to backfill stored memories into Poppy.",
)
def setup_claude_desktop(print_instructions: bool, print_import_prompt: bool):
    """Register the Poppy MCP server in the Claude desktop app.

    Backs up the existing config to `<config>.pre-poppy.bak` on first run.
    Restart the Claude desktop app afterwards to load the server.

    Two helper flags (no install when used):
      --print-instructions   Primer for Settings → General → Profile →
                             Instructions for Claude, so the desktop agent
                             knows when to call remember/recall.
      --print-import-prompt  Paste this into an existing Claude chat (with
                             the poppy connector enabled) to dump Claude's
                             stored memories into Poppy.
    """
    from poppy.setup.claude_code import CLAUDE_IMPORT_PROMPT, CLAUDE_MD_BODY, install_for_client

    if print_instructions:
        click.echo(
            "Copy the block below into Claude desktop's Settings → General → "
            "Profile → 'Instructions for Claude' (or a Project's instructions):\n"
        )
        click.echo("---8<---")
        click.echo(CLAUDE_MD_BODY)
        click.echo("---8<---")
        return

    if print_import_prompt:
        click.echo(
            "Open a Claude desktop chat with the poppy connector enabled, then "
            "paste the prompt below. Claude will read its stored memories and "
            "ingest each entry into Poppy via the remember tool:\n"
        )
        click.echo("---8<---")
        click.echo(CLAUDE_IMPORT_PROMPT)
        click.echo("---8<---")
        return

    paths = install_for_client(client="claude-desktop")
    if "backup" in paths:
        click.echo(f"  backup: {paths['backup']}")
    click.echo(f"  MCP config: {paths['MCP config']}")
    telemetry.capture(_get_poppy_dir(), "agent_setup", {"agent": "claude-desktop"})
    click.echo(
        "\nPoppy is ready. Restart the Claude desktop app to activate.\n"
        "Tip:\n"
        "  poppy setup claude-desktop --print-instructions     # paste into Instructions for Claude\n"
        "  poppy setup claude-desktop --print-import-prompt    # paste into a chat to backfill memories"
    )


@setup.command("trags")
@click.option(
    "--api-url",
    default=None,
    help="Override trags-api-url (defaults to the configured value or https://trags.ai).",
)
def setup_trags(api_url: str | None):
    """One-command device-code onboarding for Trags cloud sync.

    Opens your browser to authorize this machine, then writes the returned
    API key to ~/.poppy/config.json. After this you can run `poppy sync push`.
    """
    from poppy.setup.trags import run_device_code_flow

    run_device_code_flow(api_url)
    telemetry.capture(_get_poppy_dir(), "agent_setup", {"agent": "trags"})


from poppy.cli.hooks import hook as _hook_group  # noqa: E402

cli.add_command(_hook_group)


@cli.group("build")
def build_group():
    """Build distributable artifacts."""
    pass


@build_group.command("mcpb")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("dist"),
    help="Where to write the .mcpb file (default: ./dist).",
)
def build_mcpb_cmd(output_dir: Path):
    """Build a Claude Desktop Extension bundle (.mcpb) from this checkout."""
    from poppy.build_mcpb import build_mcpb

    repo_root = Path(__file__).resolve().parents[3]
    try:
        produced = build_mcpb(repo_root=repo_root, output_dir=output_dir)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    size_kb = produced.stat().st_size / 1024
    click.echo(f"  built: {produced}")
    click.echo(f"  size:  {size_kb:.1f} KB")
    click.echo("\nDouble-click the .mcpb file to install it in Claude Desktop.")


@cli.group("import")
def import_group():
    """Import memories from external sources."""
    pass


@import_group.command("claude-memories")
@click.option("--dry-run", is_flag=True, help="Show what would be imported without writing.")
@click.option(
    "--projects-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override the Claude Code projects directory (default: ~/.claude/projects).",
)
def import_claude_memories(dry_run: bool, projects_dir: Path | None):
    """Import curated auto-memory files from ~/.claude/projects/<slug>/memory/."""
    from poppy.integrations.claude_memory_import import (
        default_claude_projects_dir,
    )
    from poppy.integrations.claude_memory_import import (
        import_claude_memories as _do_import,
    )

    target = projects_dir or default_claude_projects_dir()
    if not target.is_dir():
        click.echo(f"No Claude Code projects directory found at {target}", err=True)
        raise click.Abort()

    engine = _get_engine()
    result = _do_import(engine, projects_dir=target, dry_run=dry_run)

    verb = "Would import" if dry_run else "Imported"
    click.echo(f"{verb}: {result.imported}  skipped (already present): {result.skipped}  failed: {result.failed}")
    if dry_run and result.paths_imported:
        for path in result.paths_imported[:20]:
            click.echo(f"  + {path}")
        if len(result.paths_imported) > 20:
            click.echo(f"  ... and {len(result.paths_imported) - 20} more")
    if not dry_run and result.imported > 0:
        from poppy.sync.auto import trigger as _trigger_autosync

        _trigger_autosync(_get_poppy_dir())


@import_group.command("hermes-memories")
@click.option("--dry-run", is_flag=True, help="Show what would be imported without writing.")
@click.option(
    "--memories-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override the hermes memories directory (default: ~/.hermes/memories).",
)
def import_hermes_memories_cmd(dry_run: bool, memories_dir: Path | None):
    """Import paragraphs from ~/.hermes/memories/{MEMORY,USER}.md."""
    from poppy.integrations.hermes_memory_import import (
        default_hermes_memories_dir,
        import_hermes_memories,
    )

    target = memories_dir or default_hermes_memories_dir()
    if not target.is_dir():
        click.echo(f"No hermes memories directory found at {target}", err=True)
        raise click.Abort()

    engine = _get_engine()
    result = import_hermes_memories(engine, memories_dir=target, dry_run=dry_run)

    verb = "Would import" if dry_run else "Imported"
    count = len(result.paths_imported) if dry_run else result.imported
    click.echo(f"{verb}: {count}  skipped (already present): {result.skipped}  failed: {result.failed}")
    if dry_run and result.paths_imported:
        for entry in result.paths_imported[:20]:
            click.echo(f"  + {entry}")
        if len(result.paths_imported) > 20:
            click.echo(f"  ... and {len(result.paths_imported) - 20} more")
    if not dry_run and result.imported > 0:
        from poppy.sync.auto import trigger as _trigger_autosync

        _trigger_autosync(_get_poppy_dir())


@cli.group("sync")
def sync_group():
    """Sync memories to/from a Trags instance."""
    pass


def _sync_client():
    """Build a Trags HTTP client from poppy config. Raises Abort if not set."""
    from poppy.config import load_config
    from poppy.sync import TragsClient

    cfg = load_config(_get_poppy_dir())
    if not cfg.trags_api_key:
        click.echo(
            "Trags API key not configured. Set it with:\n"
            "  poppy config set trags-api-key <usr_xxxxx>\n"
            "  poppy config set trags-api-url <https://your-trags-host>   # optional, defaults to https://trags.ai",
            err=True,
        )
        raise click.Abort()
    return TragsClient(base_url=cfg.trags_api_url, api_key=cfg.trags_api_key), cfg.trags_api_url


def _sync_tombstones():
    from poppy.ui.tombstones import TombstoneStore

    return TombstoneStore(_get_poppy_dir() / "memories.db")


def _print_push(res) -> None:
    click.echo(
        f"  push: {res.sent_live} live, {res.sent_tombstones} tombstones, {res.skipped} skipped, {res.errors} errors"
    )


def _print_pull(res) -> None:
    click.echo(
        f"  pull: {res.applied_live} live, {res.applied_tombstones} tombstones, "
        f"{res.skipped_stale} skipped (local newer), {res.errors} errors"
    )


@sync_group.command("push")
@click.option("--dry-run", is_flag=True, help="Show what would be sent without writing to Trags.")
def sync_push(dry_run: bool):
    """Send local memories + tombstones to Trags (since the last push watermark)."""
    from poppy.sync import load, push

    client, _ = _sync_client()
    with client:
        engine = _get_engine()
        state = load(_get_poppy_dir())
        res = push(
            engine=engine,
            tombstones=_sync_tombstones(),
            client=client,
            state=state,
            poppy_dir=_get_poppy_dir(),
            dry_run=dry_run,
        )
    _print_push(res)


@sync_group.command("pull")
@click.option("--dry-run", is_flag=True, help="Show what would be applied without touching local DB.")
def sync_pull(dry_run: bool):
    """Apply Trags rows newer than our last pull watermark."""
    from poppy.sync import load, pull

    client, _ = _sync_client()
    with client:
        engine = _get_engine()
        state = load(_get_poppy_dir())
        res = pull(
            engine=engine,
            tombstones=_sync_tombstones(),
            client=client,
            state=state,
            poppy_dir=_get_poppy_dir(),
            dry_run=dry_run,
        )
    _print_pull(res)


@sync_group.command("status")
def sync_status():
    """Show watermarks and last-sync time."""
    from poppy.config import load_config
    from poppy.sync import remote_state_for

    cfg = load_config(_get_poppy_dir())
    if not cfg.trags_api_key:
        click.echo("Trags not configured. Run `poppy config set trags-api-key <usr_xxxxx>`.")
        return
    rs = remote_state_for(_get_poppy_dir(), cfg.trags_api_url)
    pending = (_get_poppy_dir() / "sync.pending").exists()
    click.echo(f"  url:             {cfg.trags_api_url}")
    click.echo(f"  auto-sync:       {cfg.auto_sync}{'  (pending)' if pending else ''}")
    click.echo(f"  last pulled at:  {rs.last_pulled_at or '—'}")
    click.echo(f"  last pushed at:  {rs.last_pushed_at or '—'}")
    click.echo(f"  last synced at:  {rs.last_synced_at or '—'}")
    click.echo(f"  pushed (total):  {rs.pushed_count}")
    click.echo(f"  pulled (total):  {rs.pulled_count}")


@sync_group.command("run")
@click.option("--dry-run", is_flag=True, help="Show what would happen without writing anywhere.")
def sync_run(dry_run: bool):
    """Pull then push — full bidirectional sync."""
    from poppy.sync import sync as do_sync

    client, _ = _sync_client()
    with client:
        engine = _get_engine()
        res = do_sync(
            engine=engine,
            tombstones=_sync_tombstones(),
            client=client,
            poppy_dir=_get_poppy_dir(),
            dry_run=dry_run,
        )
    _print_pull(res.pull)
    _print_push(res.push)


@sync_group.command("_auto-worker", hidden=True)
def sync_auto_worker():
    """Internal: detached worker spawned by auto-sync triggers. Do not invoke directly."""
    from poppy.sync.auto import run_worker

    run_worker(_get_poppy_dir())


@cli.command()
def doctor():
    """Verify the Poppy installation: engine, storage, MCP config, hooks."""
    import shutil

    from poppy.setup.claude_code import (
        get_claude_config_dir,
        get_claude_desktop_config_path,
        is_hook_installed,
        is_mcp_installed,
        managed_claude_md_present,
    )

    _ = get_claude_desktop_config_path  # used below — pin against ruff auto-strip

    ok = True

    def line(label: str, status: str, detail: str = "", hint: str = "") -> None:
        nonlocal ok
        if status == "FAIL":
            ok = False
        marker = {"OK": "✓", "WARN": "!", "FAIL": "✗"}.get(status, "·")
        bits = [f"  [{marker}] {label}: {status}"]
        if detail:
            bits.append(f" — {detail}")
        if hint and status != "OK":
            bits.append(f" — {hint}")
        click.echo("".join(bits))

    poppy_bin = shutil.which("poppy")
    line("poppy executable", "OK" if poppy_bin else "FAIL", poppy_bin or "not on PATH")

    poppy_dir = _get_poppy_dir()
    line("storage dir", "OK" if poppy_dir.exists() else "WARN", str(poppy_dir))

    try:
        engine = _get_engine()
        s = engine.stats()
        # Surface two drift signals:
        #   1. Configured engine != active engine — get_engine fell back due to
        #      missing deps or an unknown name.
        #   2. memory_embeddings rows tagged with a model_id that doesn't match
        #      the active engine's bi-encoder (or with NULL legacy rows). Those
        #      rows silently degrade to FTS-only recall.
        from poppy.config import load_config as _load_config
        from poppy.engine.migration import stale_stats as _stale_stats

        configured = _load_config(poppy_dir).engine
        engine_msg = f"{s.engine_name} v{s.engine_version} ({s.memory_count} memories)"
        if configured != s.engine_name:
            line(
                "engine",
                "WARN",
                f"configured={configured!r}, active={s.engine_name!r}",
                hint="install the configured engine's deps or `poppy engines use <name>`",
            )
        else:
            line("engine", "OK", engine_msg)
        if engine.model_id is not None:
            drift = _stale_stats(poppy_dir / "memories.db", engine.model_id)
            if drift.needs_migration > 0:
                line(
                    "embedding index",
                    "WARN",
                    f"{drift.needs_migration} memories on a stale model_id (compatible={drift.compatible})",
                    hint="run `poppy migrate-engine` to re-embed",
                )
    except Exception as exc:
        line("engine", "FAIL", str(exc))

    claude_dir = get_claude_config_dir()
    line("claude config dir", "OK" if claude_dir.exists() else "WARN", str(claude_dir))

    line(
        "MCP server registered",
        "OK" if is_mcp_installed(claude_dir) else "WARN",
        hint="run `poppy setup claude-code` to install",
    )
    for event in ("SessionStart", "UserPromptSubmit", "PreToolUse", "SessionEnd"):
        line(
            f"{event} hook",
            "OK" if is_hook_installed(claude_dir, event) else "WARN",
            hint="run `poppy setup claude-code` to install",
        )
    line(
        "CLAUDE.md block",
        "OK" if managed_claude_md_present(claude_dir) else "WARN",
        hint="run `poppy setup claude-code --claude-md` to install",
    )

    desktop_path = get_claude_desktop_config_path()
    if desktop_path.exists():
        try:
            desktop_settings = json.loads(desktop_path.read_text())
        except (OSError, json.JSONDecodeError):
            desktop_settings = {}
        registered = "poppy" in desktop_settings.get("mcpServers", {})
        line(
            "Claude desktop MCP",
            "OK" if registered else "WARN",
            str(desktop_path),
            hint="run `poppy setup claude-desktop` to install" if not registered else "",
        )
    else:
        line(
            "Claude desktop MCP",
            "WARN",
            f"no config at {desktop_path}",
            hint="install the Claude desktop app, or run `poppy setup claude-desktop` once it's installed",
        )

    # Other MCP clients (informational — only flag WARN if config exists but
    # poppy isn't registered; absent client → silent).
    from poppy.setup.claude_code import _client_settings_path

    for client_id, label, install_cmd in (
        ("cursor", "Cursor MCP", "poppy setup cursor"),
        ("windsurf", "Windsurf MCP", "poppy setup windsurf"),
        ("codex", "Codex MCP", "poppy setup codex"),
        ("copilot-cli", "Copilot CLI MCP", "poppy setup copilot-cli"),
        ("pi", "Pi MCP", "poppy setup pi"),
    ):
        path = _client_settings_path(client_id, claude_dir)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            data = {}
        registered = "poppy" in data.get("mcpServers", {})
        line(
            label,
            "OK" if registered else "WARN",
            str(path),
            hint=f"run `{install_cmd}` to install" if not registered else "",
        )

    # Primer (AGENTS.md) presence for Copilot CLI / Pi — only flag WARN if MCP
    # is registered but the agent has no primer telling it to use Poppy tools.
    from poppy.setup.claude_code import _client_primer_path, managed_primer_present

    for client_id, label, install_cmd in (
        ("copilot-cli", "Copilot CLI primer", "poppy setup copilot-cli"),
        ("pi", "Pi primer", "poppy setup pi"),
    ):
        primer = _client_primer_path(client_id)
        if primer is None:
            continue
        mcp_path = _client_settings_path(client_id, claude_dir)
        if not mcp_path.exists():
            continue
        line(
            label,
            "OK" if managed_primer_present(primer) else "WARN",
            str(primer),
            hint=f"run `{install_cmd}` to install" if not managed_primer_present(primer) else "",
        )

    # Goose (Block) — YAML-config MCP client. Only flag when the user has a
    # goose config directory present; absent → silent.
    from poppy.setup.goose import get_goose_config_dir, is_goose_installed

    goose_dir = get_goose_config_dir()
    if goose_dir.exists():
        line(
            "Goose MCP",
            "OK" if is_goose_installed(goose_dir) else "WARN",
            str(goose_dir / "config.yaml"),
            hint="run `poppy setup goose` to install" if not is_goose_installed(goose_dir) else "",
        )
        goose_hints = goose_dir / ".goosehints"
        line(
            "Goose primer",
            "OK" if managed_primer_present(goose_hints) else "WARN",
            str(goose_hints),
            hint="run `poppy setup goose` to install" if not managed_primer_present(goose_hints) else "",
        )

    # Hermes Agent (Nous Research) — plugin-based, not MCP. Show status only
    # when ~/.hermes/ exists (i.e. the user has hermes installed).
    from poppy.setup.hermes import HERMES_PLUGIN_NAME, get_hermes_home, is_hermes_installed

    _ = HERMES_PLUGIN_NAME  # pin against ruff auto-strip
    hermes_home = get_hermes_home()
    if hermes_home.exists():
        plugin_dir = hermes_home / "plugins" / "poppy"
        line(
            "Hermes Agent plugin",
            "OK" if is_hermes_installed(hermes_home) else "WARN",
            str(plugin_dir),
            hint="run `poppy setup hermes-agent` to install" if not is_hermes_installed(hermes_home) else "",
        )
        hermes_primer = hermes_home / "AGENTS.md"
        line(
            "Hermes primer",
            "OK" if managed_primer_present(hermes_primer) else "WARN",
            str(hermes_primer),
            hint="run `poppy setup hermes-agent` to install" if not managed_primer_present(hermes_primer) else "",
        )

    # Pi requires the pi-mcp-adapter extension to consume the MCP config.
    pi_settings_path = Path.home() / ".pi" / "agent" / "settings.json"
    if pi_settings_path.exists():
        try:
            pi_settings = json.loads(pi_settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            pi_settings = {}
        adapter_installed = any("pi-mcp-adapter" in p for p in pi_settings.get("packages", []))
        line(
            "Pi MCP adapter",
            "OK" if adapter_installed else "WARN",
            hint="run `pi install npm:pi-mcp-adapter` to bridge MCP servers into Pi" if not adapter_installed else "",
        )

    # Auto-capture status: report the granular consent + backend
    # state from the ConsolidationPolicy, not a bare enabled/disabled bool, so
    # "consent pending" and the silent-breakage cases (remote-only, no backend)
    # are distinguishable and each hint points at the real fix.
    from poppy.capture import journal as _journal
    from poppy.capture.policy import CaptureStatus, status_message
    from poppy.capture.policy import evaluate as _evaluate_capture
    from poppy.config import load_config

    cfg = load_config(_get_poppy_dir())
    cap_status = _evaluate_capture(cfg)
    capture_doctor = {
        CaptureStatus.ACTIVE: ("OK", ""),
        CaptureStatus.FORCED_ENV: ("OK", ""),
        CaptureStatus.INERT_PENDING: ("WARN", "run `poppy consent --enable` to turn it on"),
        # A deliberate off-state is not a problem the doctor should flag.
        CaptureStatus.DISABLED_OPT_OUT: ("OK", ""),
        CaptureStatus.DISABLED_ENV: ("OK", ""),
        CaptureStatus.WARN_REMOTE_ONLY: ("WARN", "install a host CLI (claude/codex/gemini) for free local capture"),
        CaptureStatus.DISABLED_NO_BACKEND: ("WARN", "install the Claude CLI for free local capture"),
    }
    status_kind, status_hint = capture_doctor.get(cap_status, ("WARN", ""))
    line("auto-capture", status_kind, status_message(cap_status), hint=status_hint)

    # Last-capture freshness: proof the loop has actually run. Reads the
    # local capture journal; informational, never a failure. Includes the total
    # journal record count so growth over time is visible.
    journal_count = len(_journal.read_all(_get_poppy_dir()))
    last_capture = _journal.read_last(_get_poppy_dir())
    if last_capture is not None:
        scope = last_capture.project or "all projects"
        line(
            "last capture",
            "OK",
            f"{last_capture.count} stored for {scope} at {last_capture.ts} ({journal_count} journaled total)",
        )
    else:
        line("last capture", "OK", "no captures recorded yet")

    # Capture watermark/lock state: per-session progress + any lock
    # files. A fresh lock means a worker is in flight; one older than LOCK_TTL_S
    # belonged to a crashed worker and will be auto-stolen on the next fire.
    import time as _time

    from poppy.capture import _state as _capture_state
    from poppy.capture.lock import LOCK_TTL_S

    tracked_sessions = len(_capture_state.load(_get_poppy_dir()))
    locks = sorted(_get_poppy_dir().glob("capture-*.lock"))
    stale_locks = []
    for lock_path in locks:
        try:
            if _time.time() - lock_path.stat().st_mtime > LOCK_TTL_S:
                stale_locks.append(lock_path)
        except OSError:
            continue
    if stale_locks:
        line(
            "capture state",
            "WARN",
            f"{tracked_sessions} session(s) tracked, {len(stale_locks)} stale lock(s)",
            hint="stale locks are auto-stolen on the next capture; safe to delete",
        )
    else:
        in_flight = f", {len(locks)} capture in flight" if locks else ""
        line("capture state", "OK", f"{tracked_sessions} session(s) tracked{in_flight}")

    if not ok:
        raise SystemExit(1)


def main() -> None:
    """Console-script entrypoint.

    Wraps the click group so expected operational failures (e.g. retrieval
    models unavailable offline with a cold cache) render as a one-line
    actionable message instead of a raw traceback. Imported lazily from
    ``poppy.errors`` (which is import-cheap, no ML deps) so a deps-missing
    install still runs.
    """
    from poppy.errors import ModelUnavailableError

    try:
        cli()
    except ModelUnavailableError as exc:
        click.echo(f"poppy: {exc}", err=True)
        raise SystemExit(1) from None
