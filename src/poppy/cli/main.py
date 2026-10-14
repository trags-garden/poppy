import datetime
import errno
import json
import os
from pathlib import Path

import click

from poppy import __version__, telemetry, update_check
from poppy.config import PoppyConfig, config_key, load_config, parse_setting, save_config
from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters
from poppy.runtime import get_engine as _runtime_get_engine
from poppy.runtime import get_poppy_dir as _runtime_get_poppy_dir


def _get_poppy_dir() -> Path:
    return _runtime_get_poppy_dir()


_ENGINE_FALLBACK_NOTICE_FILE = ".engine_fallback_notice"
_ENGINE_FALLBACK_REASON_LIMIT = 160


def _fallback_reason(configured: str, description: str) -> str:
    """Extract a short user-facing reason from the runtime's diagnostic."""
    reason = description.removeprefix(f"{configured}: ")
    if ": " in reason:
        reason = reason.split(": ", 1)[1]
    reason = reason.splitlines()[0].strip() or "unknown error"
    if len(reason) > _ENGINE_FALLBACK_REASON_LIMIT:
        reason = reason[: _ENGINE_FALLBACK_REASON_LIMIT - 3].rstrip() + "..."
    return reason


def _maybe_print_engine_fallback_notice(
    poppy_dir: Path,
    *,
    configured: str,
    active: str,
    error: str,
) -> None:
    """Print and persist the daily fallback notice without affecting the command."""
    notice_path = poppy_dir / _ENGINE_FALLBACK_NOTICE_FILE
    reason = _fallback_reason(configured, error)
    # Throttle on the (date, configured->active, reason) triple rather than the
    # date alone: an identical (pair, reason) repeat stays silent for the day,
    # while a change in either the substitution pair or the reason re-notifies,
    # so a new actionable failure is never hidden behind an earlier notice.
    state = f"{datetime.date.today().isoformat()} {configured}->{active} {reason}"
    try:
        if notice_path.read_text(encoding="utf-8").strip() == state:
            return
    except (OSError, UnicodeError):
        pass

    # Persistence is best-effort: engine fallback correctness must never depend
    # on whether this advisory notice can update its throttle file.
    try:
        notice_path.write_text(state + "\n", encoding="utf-8")
    except OSError:
        pass

    click.echo(
        f"configured engine {configured!r} unavailable ({reason}), using {active!r}",
        err=True,
    )


def _get_engine() -> RetrievalEngine:
    poppy_dir = _get_poppy_dir()
    configured = load_config(poppy_dir).engine
    errors: list[str] = []
    engine = _runtime_get_engine(poppy_dir, error_sink=errors.append)
    active = getattr(engine, "_engine_name", configured)
    if errors and active != configured:
        _maybe_print_engine_fallback_notice(
            poppy_dir,
            configured=configured,
            active=active,
            error=errors[0],
        )
    return engine


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


# Top-level command groups that must never stop to ask the telemetry question.
# Hooks, the MCP server and the daemon run with nobody watching and inherit the
# terminal of whatever started them, so the interactivity check alone would not
# hold. `telemetry` is excluded because the user is already standing at the
# switch. (`hook` is defined in its own module with plain click classes, so its
# subcommands cannot reach the question at all; it is named here so the rule
# stays true if that ever changes.)
_NO_CONSENT_PROMPT_COMMANDS = frozenset({"daemon", "hook", "serve", "telemetry"})
# Parameters that mean "do not stop to ask me": `--json` output has to stay
# machine-readable, and `--yes` is how an unattended install says it will not
# be answering questions.
_NO_CONSENT_PROMPT_PARAMS = ("as_json", "yes")


def _invocation_path(ctx: click.Context) -> tuple[str, ...]:
    """Subcommand names below the root group, e.g. ("sync", "push")."""
    names: list[str] = []
    while ctx.parent is not None:
        names.append(ctx.info_name or "")
        ctx = ctx.parent
    return tuple(reversed(names))


def _may_ask_about_telemetry(ctx: click.Context) -> bool:
    """Whether this fully parsed invocation is one a person could answer during."""
    path = _invocation_path(ctx)
    if path[:1] and path[0] in _NO_CONSENT_PROMPT_COMMANDS:
        return False
    # Detached workers are the hidden underscore-prefixed commands Poppy spawns
    # for itself (`sync _auto-worker`, the capture workers). Keying on the
    # naming convention means a new worker is covered the day it is added.
    if any(name.startswith("_") for name in path):
        return False
    return not any(ctx.params.get(name) for name in _NO_CONSENT_PROMPT_PARAMS)


class _AskAboutTelemetryCommand(click.Command):
    """A command that asks the one-time telemetry question before it runs.

    The question lives here, not in the root group callback, because click runs
    a group callback before it builds the subcommand's context: asking there
    meant `poppy recall --help` and plain usage errors stopped for a question
    first. `Command.invoke` runs after every argument has been parsed and
    validated, so help output and usage errors are already handled by click,
    and the parsed parameters can be read instead of guessed from argv.
    """

    def invoke(self, ctx: click.Context):
        if _may_ask_about_telemetry(ctx):
            telemetry.maybe_prompt_for_consent(_get_poppy_dir())
        return super().invoke(ctx)


class _PoppyGroup(click.Group):
    """Root group whose commands, and nested groups' commands, ask before running."""

    command_class = _AskAboutTelemetryCommand
    group_class = type  # click sentinel: subgroups are this same class


@click.group(cls=_PoppyGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-v", "--version", prog_name="poppy")
def cli():
    """Poppy -- remember what matters."""


@cli.result_callback()
@click.pass_context
def _update_notice_epilogue(ctx: click.Context, result):
    """Surface a cached update once after interactive human commands."""
    if ctx.invoked_subcommand not in {"doctor", "hook", "serve"}:
        update_check.maybe_print_update_notice(_get_poppy_dir())
    return result


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
    from poppy.lifecycle import resolve_expiry
    from poppy.write_flow import remember as _remember

    # Validate the expiry flags before constructing the engine so a bad --ttl
    # fails fast (cheap BadParameter) instead of paying the model-load cost first.
    try:
        resolve_expiry(ttl, expires_at)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    try:
        result = _remember(
            _get_engine(),
            _get_poppy_dir(),
            content=content,
            memory_type=memory_type,
            project=project,
            ttl=ttl,
            expires_at=expires_at,
            supersedes=supersedes,
            check_conflicts=check_conflicts,
            auto_supersede=auto_supersede,
            source="manual",
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc

    if result.conflict_error:
        click.echo(f"  ({result.conflict_error})", err=True)

    if result.mode == "check":
        if not result.conflicts:
            click.echo("No conflict candidates found. (Nothing was written.)")
        else:
            click.echo(f"{len(result.conflicts)} conflict candidate(s):")
            for c in result.conflicts:
                click.echo(f"  {c.memory.id}  conf={c.confidence:.2f}  {c.reason or 'No reason provided'}")
                click.echo(f"    {c.memory.content[:80]}")
        return

    if result.superseded_id:
        picked = next((c for c in result.conflicts if c.memory.id == result.superseded_id), None)
        if picked is not None:
            click.echo(f"  auto-supersede: {picked.memory.id} (confidence {picked.confidence:.2f})")
        click.echo(f"Remembered ({memory_type}): {content[:80]}")
        click.echo(f"  id: {result.memory.id}")
        click.echo(f"  supersedes {result.superseded_id} (tombstoned, restorable for 7 days)")
    else:
        click.echo(f"Remembered ({memory_type}): {content[:80]}")
        click.echo(f"  id: {result.memory.id}")

    # Suggest hint after a normal write.
    if result.mode in ("suggest", "auto") and result.conflicts and not result.superseded_id:
        n = len(result.conflicts)
        click.echo(f"  ⚠ may supersede {n} memor{'y' if n == 1 else 'ies'}:")
        for c in result.conflicts[:3]:
            click.echo(f"    {c.memory.id}  conf={c.confidence:.2f}  {c.memory.content[:60]}")
        click.echo("  Re-run with --auto-supersede or `poppy remember --supersedes <id>`.")

    if result.memory.expires_at is not None:
        click.echo(f"  expires at {result.memory.expires_at.isoformat()}")


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
    # the registry resolved (default: bloom, falling back to seed).
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
        # empty store.
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
        click.echo(f"    {r.memory.memory_type}{project_tag} | {date_str} | score: {r.score:.2f} | {r.memory.id}")
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
        # as "nothing stored".
        if filters_active:
            click.echo("No memories match the given filters (--since/--project/--type).")
        else:
            click.echo("No memories stored yet.")
        return

    for m in memories:
        project_tag = f" [{m.project}]" if m.project else ""
        date_str = m.created_at.strftime("%Y-%m-%d")
        click.echo(f"  {m.content}")
        click.echo(f"    {m.memory_type}{project_tag} | {date_str} | {m.id}")
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

    # The core flow tombstones-before-delete (restorable + sync-visible)
    # and triggers autosync on success. Runs only after the confirmation gate,
    # so a cancelled forget records nothing.
    from poppy.write_flow import forget as _forget

    _forget(engine, _get_poppy_dir(), memory_id)
    click.echo(f"Forgotten: {memory_id}")
    if mem.expires_at is None or mem.expires_at > datetime.datetime.now(datetime.timezone.utc):
        click.echo(f"Restore within 7 days with: poppy restore {memory_id}")


@cli.command()
@click.argument("memory_id")
def restore(memory_id: str):
    """Restore a memory from Trash within 7 days."""
    from poppy.write_flow import restore as _restore

    engine = _get_engine()
    try:
        result = _restore(engine, _get_poppy_dir(), memory_id)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not result.found:
        raise click.ClickException(f"Memory {memory_id} is not in Trash.")
    if result.expired:
        raise click.ClickException(f"Memory {memory_id} has passed its expiry and cannot be restored.")
    click.echo(f"Restored: {memory_id}")


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
@click.option("--no-expiry", is_flag=True, help="Clear expiry: make memory permanent")
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
            poppy_dir=_get_poppy_dir(),
        )
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
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
            f"⚠  {stats.needs_migration} memories are not embedded for this engine "
            f"({stats.stale} stale, {stats.unknown} pre-tagging, {stats.missing} never embedded) "
            f"and will use FTS-only recall until re-embedded."
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
    only the matching subset. Idempotent and resumable: each row is
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
    engine = _get_engine()
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


def _trags_key_keychain_failure(poppy_dir: Path) -> str:
    """Classify why a store's Trags key is not in the OS keychain.

    Prefers what this process's own save actually did (recorded by save_config)
    over a fresh probe: probing writes and reads a *different* entry afterwards,
    so it reports a healthy session for the very attempt the backend refused.
    A process that never wrote the key -- `poppy doctor` -- has
    nothing recorded and probes this session instead: the same classification,
    measured now. Callers map the outcome to their own wording.
    """
    from poppy import keychain  # noqa: PLC0415
    from poppy.config import trags_key_keychain_outcome  # noqa: PLC0415

    return trags_key_keychain_outcome(poppy_dir) or keychain.probe()


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
    # Parse and validate BEFORE any I/O: load_config can migrate a plaintext key
    # into the keychain and rewrite config.json as a side effect, so a bad value
    # must fail here without touching the file.
    try:
        parsed = parse_setting(key, value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    entry = config_key(key)
    # A self-persisting key applies its own write (telemetry -> telemetry.set_enabled,
    # which mirrors analytics.json and latches the first-run notice); everything
    # else applies to a freshly loaded config and is saved once here.
    if entry.self_persist:
        try:
            entry.apply(PoppyConfig(poppy_dir=poppy_dir), parsed)
        except telemetry.TelemetryChoiceError as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        cfg = load_config(poppy_dir=poppy_dir)
        entry.apply(cfg, parsed)
        save_config(cfg)
    if key == "trags-api-key":
        # Never echo the secret back to the terminal / scrollback.
        from poppy import keychain
        from poppy.config import TRAGS_API_KEY_ENV, trags_api_key_location, trags_key_keychain_outcome

        key_location = trags_api_key_location(poppy_dir)
        if value == "":
            if os.environ.get(TRAGS_API_KEY_ENV):
                click.echo(f"Note: {TRAGS_API_KEY_ENV} is set in your environment and overrides the stored key.")
            # Two ways to learn the secret outlived the clear: the delete this
            # session attempted was refused, or the entry still reads back. The
            # first is what actually happened; the second is the backstop for a
            # refusal the delete could not see.
            refused = trags_key_keychain_outcome(poppy_dir) == keychain.DELETE_REFUSED
            if refused or key_location == "keychain":
                still = "the key is still active" if key_location == "keychain" else "the key may still be active"
                raise click.ClickException(
                    f"Could not remove trags-api-key from the OS keychain; {still}. "
                    "Re-run from an interactive login session or remove the 'poppy-memory' item "
                    "in your keychain manager."
                )
            if key_location == "keychain-unreadable":
                click.echo(
                    "Cleared trags-api-key from config.json. This session cannot read the OS keychain, "
                    "so a keychain copy may survive; re-run this command from an interactive login "
                    "session to be sure."
                )
            else:
                click.echo("Cleared trags-api-key.")
            return
        if key_location == "keychain":
            click.echo("Set trags-api-key (stored in the OS keychain, not config.json).")
        elif key_location == "config-file":
            # Name the step that actually failed rather than assuming read-back:
            # a backend can be present and readable and still refuse the write.
            reason = {
                keychain.PROBE_NO_BACKEND: "No OS keychain backend is available here.",
                keychain.PROBE_WRITE_REJECTED: "The OS keychain rejected the write.",
                keychain.PROBE_READBACK_FAILED: "The OS keychain could not be read back in this session.",
            }.get(
                _trags_key_keychain_failure(poppy_dir),
                "The OS keychain could not be used for the key in this session.",
            )
            click.echo(
                f"Set trags-api-key in {poppy_dir / 'config.json'} (plaintext; the file is 0600). "
                f"{reason} On a headless host, prefer setting "
                f"{TRAGS_API_KEY_ENV} in the environment over storing the key on disk."
            )
        else:
            # The write went through, but reading the store back says neither
            # keychain nor file, so do not claim either one holds the key.
            click.echo(
                "Set trags-api-key. This session could not confirm where the key was stored; "
                "run `poppy doctor` from an interactive login session to check."
            )
        if os.environ.get(TRAGS_API_KEY_ENV):
            click.echo(f"Note: {TRAGS_API_KEY_ENV} is set in your environment and overrides the stored key.")
        return
    shown = entry.display(parsed) if entry.display is not None else value
    click.echo(f"Set {key} = {shown}")


@cli.group("redaction")
def redaction_group():
    """Manage custom secret redaction (add | remove | list)."""
    pass


def _redaction_entry(literal: str | None, env_var: str | None) -> tuple[str, str]:
    """Resolve the exactly-one input shape shared by add and remove."""
    if literal is not None and env_var is not None:
        raise click.UsageError("Provide either a literal or --env NAME, not both.")
    if literal is None and env_var is None:
        raise click.UsageError("Provide a literal or --env NAME.")
    kind, value = ("environment variable", env_var) if env_var is not None else ("literal", literal)
    assert value is not None
    value = value.strip()
    if not value:
        raise click.BadParameter(f"{kind} must not be blank")
    return kind, value


def _redaction_detail(kind: str, value: str) -> str:
    """Describe a redaction entry for CLI output without leaking a secret literal.

    Configured literals are themselves secrets, so they must never reach terminal
    scrollback, CI logs, or shared screens; report them by character count only
    (the same shape `redaction list` uses). Environment-variable names are not
    secret, so they stay visible.
    """
    if kind == "literal":
        return f" ({len(value)} characters)"
    return f": {value}"


@redaction_group.command("add")
@click.argument("literal", required=False)
@click.option("--env", "env_var", metavar="NAME", help="Redact the current value of this environment variable.")
def redaction_add(literal: str | None, env_var: str | None):
    """Add a literal or environment-variable redaction."""
    from poppy.capture.redaction import MIN_CUSTOM_SECRET_LENGTH, valid_env_var_name

    kind, value = _redaction_entry(literal, env_var)
    if kind == "literal" and len(value) < MIN_CUSTOM_SECRET_LENGTH:
        raise click.BadParameter(
            f"literal must be at least {MIN_CUSTOM_SECRET_LENGTH} characters after trimming",
            param_hint="literal",
        )
    if kind == "environment variable" and not valid_env_var_name(value):
        raise click.BadParameter(
            "environment variable name must match [A-Za-z_][A-Za-z0-9_]*",
            param_hint="--env",
        )

    config = load_config(_get_poppy_dir())
    entries = config.redaction_env_vars if env_var is not None else config.redaction_literals
    if value in entries:
        click.echo(f"Custom {kind} redaction already exists{_redaction_detail(kind, value)}.")
        return
    entries.append(value)
    save_config(config)
    click.echo(f"Added custom {kind} redaction{_redaction_detail(kind, value)}.")


@redaction_group.command("remove")
@click.argument("literal", required=False)
@click.option("--env", "env_var", metavar="NAME", help="Remove this environment-variable redaction.")
def redaction_remove(literal: str | None, env_var: str | None):
    """Remove a literal or environment-variable redaction."""
    kind, value = _redaction_entry(literal, env_var)
    config = load_config(_get_poppy_dir())
    entries = config.redaction_env_vars if env_var is not None else config.redaction_literals
    if value not in entries:
        click.echo(f"No custom {kind} redaction found{_redaction_detail(kind, value)}.")
        return
    entries.remove(value)
    save_config(config)
    click.echo(f"Removed custom {kind} redaction{_redaction_detail(kind, value)}.")


@redaction_group.command("list")
@click.option("--show", "show_values", is_flag=True, help="Print literal values instead of hiding them.")
def redaction_list(show_values: bool):
    """List built-in protection and configured custom entries."""
    config = load_config(_get_poppy_dir())
    click.echo("Built-in protection: 6 secret pattern families, always on")
    click.echo("  private-key blocks, connection-string passwords, bearer tokens")
    click.echo("  AWS access key ids, OpenAI-style keys, GitHub tokens")
    click.echo(f"Custom literals ({len(config.redaction_literals)}):")
    for literal in config.redaction_literals:
        # The literals are themselves secrets: keep them out of terminal
        # scrollback, CI logs, and shared screens unless explicitly requested.
        click.echo(f"  {literal}" if show_values else f"  ({len(literal)} characters, hidden; use --show)")
    if not config.redaction_literals:
        click.echo("  none")
    click.echo(f"Environment variables ({len(config.redaction_env_vars)}):")
    for name in config.redaction_env_vars:
        click.echo(f"  {name}")
    if not config.redaction_env_vars:
        click.echo("  none")


@cli.group("encrypt", invoke_without_command=True)
@click.pass_context
def encrypt_group(ctx: click.Context):
    """Encrypt the local store at rest (status | enable | disable | repair)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(encrypt_status)


# What `poppy encrypt status` prints for each EncryptionStatus.keychain_state.
# "refused" is spelled out instead of folded into "unavailable": a session that
# may read the keychain but not write it still has its key, and must not be told
# the backend is missing on the line right above "key in keychain: yes".
_KEYCHAIN_STATE_LINES = {
    "usable": "available",
    "refused": "present, but this session cannot store and read back an entry",
    "unavailable": "unavailable",
    "not-probed": "not checked (the key comes from the environment)",
}


@encrypt_group.command("status")
@click.option("--deep", is_flag=True, help="Actually open the store (read-only) to detect bit-rot behind a marker.")
def encrypt_status(deep: bool):
    """Show whether the local store is encrypted at rest."""
    from poppy import encryption

    st = encryption.status(_get_poppy_dir(), deep=deep)
    click.echo(f"Encryption: {'on' if st.enabled else 'off'}")
    click.echo(f"  store: {st.db_path}")
    click.echo(f"  encrypted on disk: {'yes' if st.encrypted_on_disk else 'no'}")
    if st.inconsistent:
        click.echo(f"  warning: inconsistent state ({st.state}); run `poppy encrypt repair`")
    if st.state == "corrupt":
        click.echo("  warning: the store file is not plaintext and does not open under the key (corrupt or foreign)")
    elif st.state == "unreadable":
        click.echo("  warning: the store could not be read (permission or read-only filesystem), not corruption")
    elif st.state == "unknown":
        click.echo("  warning: the store file is not plaintext and cannot be verified without the extra and key")
    if not st.deps_installed:
        click.echo("  dependencies: not installed")
        for hint_line in encryption.INSTALL_HINT.splitlines():
            click.echo(f"    {hint_line}")
    else:
        click.echo("  dependencies: installed")
        if st.env_key_malformed:
            click.echo(f"  key source: {encryption.ENV_KEY} is set but MALFORMED (not 64 hex); it shadows the")
            click.echo("    keychain. Fix or unset it, then retry.")
        elif st.env_key_active:
            click.echo(f"  key source: {encryption.ENV_KEY} environment variable")
        else:
            click.echo(f"  keychain backend: {_KEYCHAIN_STATE_LINES.get(st.keychain_state, st.keychain_state)}")
            click.echo(f"  key in keychain: {'yes' if st.key_present else 'no'}")
        if st.enabled and not st.key_present and not st.env_key_malformed:
            click.echo("  warning: the store is encrypted but no key is available here; it cannot be opened.")
    if st.residue:
        names = ", ".join(p.name for p in st.residue)
        click.echo(f"  residue: leftover temp files ({names}); run `poppy encrypt repair` to remove")
    if st.stray:
        names = ", ".join(p.name for p in st.stray)
        click.echo(f"  quarantined sidecars set aside (not replayed): {names}")


@encrypt_group.command("enable")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def encrypt_enable(yes: bool):
    """Encrypt the local store, migrating existing memories in place.

    Generates a random 256-bit key, stores it in the OS keychain, and rewrites
    the store as a SQLCipher database. This is local encryption at rest, not
    zero-knowledge or end-to-end encryption: anything running as your user that
    can read the keychain can decrypt the store.
    """
    from poppy import encryption

    poppy_dir = _get_poppy_dir()
    if not encryption.dependencies_installed():
        raise click.ClickException(f"Encryption needs optional dependencies.\n{encryption.INSTALL_HINT}")
    state = encryption.store_state(poppy_dir)
    if state in ("encrypted", "encrypted-no-sentinel"):
        click.echo("The local store is already encrypted.")
        if state == "encrypted-no-sentinel":
            for action in encryption.repair(poppy_dir):
                click.echo(f"  {action}")
        return
    if not yes:
        click.echo("This encrypts the local store at rest with a key in your OS keychain (or POPPY_DB_KEY if set).")
        click.echo("Stop `poppy serve` (MCP), `poppy ui`, and any capture or sync workers first.")
        click.echo("If the key is lost, the store cannot be recovered.")
        click.confirm("Encrypt the local store now?", abort=True)
    try:
        result = encryption.enable(poppy_dir)
    except encryption.EncryptionError as exc:
        raise click.ClickException(str(exc)) from exc
    if result.created_empty:
        click.echo("Encryption enabled. A new encrypted store was created.")
    else:
        click.echo(f"Encryption enabled. Migrated {result.migrated_rows} memories into the encrypted store.")


@encrypt_group.command("disable")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def encrypt_disable(yes: bool):
    """Decrypt the local store back to plaintext and remove the key."""
    from poppy import encryption

    poppy_dir = _get_poppy_dir()
    if not encryption.dependencies_installed():
        raise click.ClickException(f"Decryption needs optional dependencies.\n{encryption.INSTALL_HINT}")
    state = encryption.store_state(poppy_dir)
    if state in ("plaintext", "absent"):
        click.echo("The local store is not encrypted.")
        return
    if not yes:
        click.echo("This rewrites the store as plaintext and removes the stored encryption key.")
        click.echo("Stop `poppy serve` (MCP), `poppy ui`, and any capture or sync workers first.")
        click.confirm("Decrypt the local store now?", abort=True)
    try:
        rows = encryption.disable(poppy_dir)
    except encryption.EncryptionError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Encryption disabled. The store is plaintext again ({rows} memories).")


@encrypt_group.command("repair")
def encrypt_repair():
    """Reconcile an inconsistent encryption state and remove leftover temp files.

    Uses the on-disk store as the source of truth: a plaintext store loses a
    stale encryption marker, an encrypted store regains a missing one, and any
    leftover migration temp files (including a decrypted copy) are removed.
    """
    from poppy import encryption

    try:
        actions = encryption.repair(_get_poppy_dir())
    except encryption.EncryptionError as exc:
        raise click.ClickException(str(exc)) from exc
    for action in actions:
        click.echo(f"  {action}")


@cli.group("telemetry", invoke_without_command=True)
@click.pass_context
def telemetry_group(ctx: click.Context):
    """Show or change anonymous usage telemetry (status | on | off)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(telemetry_status)


@telemetry_group.command("status")
def telemetry_status():
    """Show whether telemetry is on, and why."""
    poppy_dir = _get_poppy_dir()
    enabled, reason = telemetry.status(poppy_dir)
    click.echo(f"Telemetry: {'on' if enabled else 'off'} ({reason})")
    if enabled:
        click.echo("Anonymous usage events only. Memory content, queries, and project names are never sent.")
        click.echo("Turn off with: poppy telemetry off")
    elif telemetry.is_unanswered(poppy_dir):
        click.echo("Nothing is sent until you answer. Turn it on with: poppy telemetry on")


@telemetry_group.command("on")
def telemetry_on():
    """Enable anonymous usage telemetry (persists in ~/.poppy/config.json)."""
    poppy_dir = _get_poppy_dir()
    try:
        telemetry.set_enabled(poppy_dir, True)
    except telemetry.TelemetryChoiceError as exc:
        raise click.ClickException(str(exc)) from exc
    # Report the effective state, not just the persisted flag: an environment
    # override can keep telemetry off even after enabling it in config, and the
    # confirmation must match `poppy telemetry status`.
    enabled, reason = telemetry.status(poppy_dir)
    if enabled:
        click.echo("Telemetry is on. Anonymous usage events only; memory content is never sent.")
    else:
        click.echo(f"Telemetry is enabled in config, but currently off: {reason}.")


@telemetry_group.command("off")
def telemetry_off():
    """Disable anonymous usage telemetry (persists in ~/.poppy/config.json)."""
    try:
        telemetry.set_enabled(_get_poppy_dir(), False)
    except telemetry.TelemetryChoiceError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Telemetry is off.")


@cli.command()
@click.option(
    "--source",
    default=lambda: os.environ.get("POPPY_MCP_SOURCE", "mcp"),
    help="Source label recorded for memories from this server (e.g. the client name).",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http", "streamable-http"]),
    default="stdio",
    show_default=True,
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=7679, envvar="POPPY_DAEMON_PORT", show_default=True)
@click.option("--no-daemon", is_flag=True, help="Run the stdio server in this process.")
def serve(source: str, transport: str, host: str, port: int, no_daemon: bool):
    """Start the Poppy MCP server."""
    poppy_dir = _get_poppy_dir()
    if transport == "stdio":
        if no_daemon or os.environ.get("POPPY_SERVE_NO_DAEMON") == "1":
            _run_in_process_stdio(poppy_dir, source)
            return

        import time

        from poppy.mcp_server.lifecycle import LifecycleError, agent_path, kickstart_agent, probe_status

        status, _error = probe_status(poppy_dir, port)
        installed = False
        if status is None:
            try:
                installed = agent_path().is_file()
            except RuntimeError:
                pass
            if installed:
                try:
                    kickstart_agent()
                except (LifecycleError, OSError, RuntimeError):
                    pass
                timeout = _positive_float_env("POPPY_DAEMON_POLL_TIMEOUT", 5.0)
                interval = _positive_float_env("POPPY_DAEMON_POLL_INTERVAL", 0.1)
                deadline = time.monotonic() + timeout
                while True:
                    status, _error = probe_status(poppy_dir, port)
                    if status is not None or time.monotonic() >= deadline:
                        break
                    time.sleep(min(interval, max(0.0, deadline - time.monotonic())))

        if status is None:
            detail = " after kickstart" if installed else ""
            click.echo(f"Poppy daemon unavailable{detail}; using in-process server.", err=True)
            _run_in_process_stdio(poppy_dir, source)
            return

        daemon_version = status.get("version")
        if daemon_version and daemon_version != __version__:
            click.echo(
                f"Warning: Poppy CLI {__version__} is forwarding to daemon {daemon_version}.",
                err=True,
            )
        try:
            from poppy.mcp_server.forwarder import run_forwarder
        except ImportError:
            click.echo("MCP SDK too old for daemon forwarding; using in-process server.", err=True)
            _run_in_process_stdio(poppy_dir, source)
            return

        exit_code = run_forwarder(poppy_dir, port)
        if exit_code:
            raise click.exceptions.Exit(exit_code)
        return

    from poppy import writers
    from poppy.mcp_server.auth import BearerAuthMiddleware, ensure_daemon_token
    from poppy.mcp_server.daemon import (
        ALREADY_RUNNING_MESSAGE,
        bind_socket,
        create_http_app,
        daemon_lock,
        run_server,
    )
    from poppy.mcp_server.server import create_mcp_server

    if source != "mcp":
        click.echo(
            "Warning: --source is ignored for HTTP transport; per-client clientInfo attribution "
            "takes precedence in a shared daemon.",
            err=True,
        )

    with daemon_lock(poppy_dir) as acquired:
        if not acquired:
            click.echo(ALREADY_RUNNING_MESSAGE, err=True)
            return
        try:
            listener = bind_socket(host, port)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                click.echo(ALREADY_RUNNING_MESSAGE, err=True)
                return
            raise

        with listener, writers.registered(poppy_dir, "daemon"):
            engine_errors: list[str] = []
            mcp = create_mcp_server(
                poppy_dir=poppy_dir,
                source="mcp",
                host=host,
                port=port,
                json_response=True,
                stateless_http=False,
                engine_error_sink=engine_errors.append,
            )
            mcp_app = mcp.streamable_http_app()
            engine = getattr(mcp, "_poppy_engine", object())
            engine_error = "; ".join(engine_errors) or None
            app = BearerAuthMiddleware(
                create_http_app(
                    mcp_app,
                    engine=engine,
                    engine_error=engine_error,
                    poppy_dir=poppy_dir,
                    port=port,
                ),
                # This process holds the daemon run-lock and is about to serve, so
                # a deliberate POPPY_DAEMON_TOKEN here rotates the stored token.
                token=ensure_daemon_token(poppy_dir, rotate_from_env=True),
                path=(mcp.settings.streamable_http_path, "/status"),
            )
            click.echo(f"Starting Poppy MCP daemon at http://{host}:{port}/mcp (storage: {poppy_dir})", err=True)
            run_server(app, listener, host, port)


def _positive_float_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _run_in_process_stdio(poppy_dir: Path, source: str) -> None:
    from poppy import writers
    from poppy.mcp_server.server import create_mcp_server

    # Correctness is the DB gate the engine's connection holds (a migration cannot
    # run while it is open). Registration supplies a friendly name if that gate
    # blocks a migration; it does not refuse migrations itself.
    with writers.registered(poppy_dir, "serve"):
        mcp = create_mcp_server(poppy_dir=poppy_dir, source=source)
        # MCP stdio requires a clean JSON-RPC stream on stdout. Log to stderr.
        click.echo(f"Starting Poppy MCP server (storage: {poppy_dir})", err=True)
        mcp.run(transport="stdio")


@cli.group("daemon")
def daemon_group():
    """Manage the shared Poppy MCP daemon."""


@daemon_group.command("run")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=7679, envvar="POPPY_DAEMON_PORT", show_default=True)
@click.pass_context
def daemon_run(ctx: click.Context, host: str, port: int):
    """Run the daemon in the foreground."""
    ctx.invoke(serve, source="mcp", transport="http", host=host, port=port)


@daemon_group.command("status")
def daemon_status():
    """Show service installation, lock, and HTTP health."""
    from poppy.mcp_server.lifecycle import LifecycleError, inspect_daemon

    try:
        state = inspect_daemon(_get_poppy_dir())
    except LifecycleError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Agent installed: {'yes' if state.installed else 'no'}")
    lock_detail = f"yes (PID {state.pid})" if state.lock_held and state.pid else "yes" if state.lock_held else "no"
    click.echo(f"Lock held: {lock_detail}")
    click.echo(f"HTTP reachable: {'yes' if state.reachable else 'no'}")
    if state.status:
        click.echo(f"Daemon version: {state.status.get('version', 'unknown')}")


@daemon_group.command("install")
def daemon_install():
    """Install and load the per-user OS service."""
    from poppy.mcp_server.lifecycle import LifecycleError, install_agent

    try:
        path, _ = install_agent(_get_poppy_dir())
    except (LifecycleError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Installed Poppy daemon agent: {path}")


@daemon_group.command("uninstall")
def daemon_uninstall():
    """Unload and remove the per-user OS service."""
    from poppy.mcp_server.lifecycle import LifecycleError, uninstall_agent

    try:
        path, _ = uninstall_agent()
    except (LifecycleError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Uninstalled Poppy daemon agent: {path}")


@daemon_group.command("start")
def daemon_start():
    """Start the installed service or detached fallback daemon."""
    from poppy.mcp_server.lifecycle import LifecycleError, start_daemon

    try:
        click.echo(start_daemon(_get_poppy_dir()))
    except LifecycleError as exc:
        raise click.ClickException(str(exc)) from exc


@daemon_group.command("stop")
def daemon_stop():
    """Stop the installed service or detached fallback daemon."""
    from poppy.mcp_server.lifecycle import LifecycleError, stop_daemon

    try:
        click.echo(stop_daemon(_get_poppy_dir()))
    except LifecycleError as exc:
        raise click.ClickException(str(exc)) from exc


@daemon_group.command("restart")
def daemon_restart():
    """Stop and start the Poppy daemon."""
    from poppy.mcp_server.lifecycle import LifecycleError, start_daemon, stop_daemon

    try:
        click.echo(stop_daemon(_get_poppy_dir()))
        click.echo(start_daemon(_get_poppy_dir()))
    except LifecycleError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind host (default: localhost only)")
@click.option("--port", default=7800, type=int, help="Bind port (default: 7800)")
@click.option("--no-open", is_flag=True, help="Don't open the browser on launch")
@click.option(
    "--allow-remote",
    is_flag=True,
    help="Permit a non-loopback --host (exposes the unauthenticated delete/edit API to your network).",
)
def ui(host: str, port: int, no_open: bool, allow_remote: bool):
    """Browse and manage memories in a local web UI."""
    import threading
    import webbrowser

    from poppy.ui.server import LOOPBACK_HOSTS

    # The UI has no auth: binding a non-loopback host publishes the delete/edit
    # API to anyone who can reach it. Refuse unless the user explicitly opts in.
    if host not in LOOPBACK_HOSTS and not allow_remote:
        raise click.UsageError(
            f"Refusing to bind non-loopback host {host!r}: the Poppy UI has no authentication, "
            "so this would expose your memories' delete/edit API to the network. "
            "Re-run with --allow-remote if you really intend this."
        )
    if host not in LOOPBACK_HOSTS:
        click.echo(f"⚠  Binding {host}: the UI's delete/edit API is now reachable, unauthenticated, from your network.")

    # Silence HF Hub / transformers noise that surfaces when the BloomEngine lazy-loads
    # its bi-encoder + cross-encoder on the first edit. The UI doesn't load these at
    # startup anymore (reads use the FTS5-only fast engine), so users only encounter
    # this if they edit/delete/restore.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from poppy import writers
    from poppy.ui.server import run as run_ui

    poppy_dir = _get_poppy_dir()
    url = f"http://{host}:{port}"
    click.echo(f"Poppy UI · {url}  ·  storage: {poppy_dir}")
    click.echo("  ctrl-c to stop")

    if not no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        # Register as a live writer so `poppy encrypt` refuses to migrate under us.
        with writers.registered(poppy_dir, "ui"):
            run_ui(host=host, port=port, poppy_dir=poppy_dir, allow_remote=allow_remote)
    except KeyboardInterrupt:
        click.echo("\nstopped.")


_CONSENT_DISCLOSURE = (
    "\nPoppy can automatically remember decisions and lessons from your sessions so "
    "they're there next time. Extraction runs through your own coding-agent CLI on "
    "your existing login, so no new third party sees your conversation. Memories "
    "stay on your machine unless you turn on sync. Enable automatic capture?"
)


def _consent_disclosure() -> str:
    """Return the frozen disclosure copy shared by every consent grant flow."""
    return _CONSENT_DISCLOSURE


def _set_consent(config, *, granted: bool) -> None:
    """Persist the consent evidence record through one mutation path."""
    config.consent = "granted" if granted else "denied"
    save_config(config)


@cli.group("autocapture", invoke_without_command=True)
@click.pass_context
def autocapture_group(ctx: click.Context):
    """Show or change automatic capture (status | on | off)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(autocapture_status)


@autocapture_group.command("status")
@click.option("--project", default=None, help="Project name (auto-detected from the current directory if not set).")
def autocapture_status(project: str | None):
    """Show the resolved effect plus project and global consent layers."""
    from poppy.capture import policy
    from poppy.project import project_from_cwd

    project = project or project_from_cwd(os.getcwd())
    config = load_config(_get_poppy_dir())
    resolved = policy.evaluate(config, project=project)
    click.echo(f"Resolved: {policy.status_message(resolved)}")
    if project:
        project_state = "off" if config.is_project_disabled(project) else "on"
        click.echo(f"Project '{project}': {project_state}.")
    else:
        click.echo("Project: no project detected.")
    click.echo(f"Global consent: {policy.effective_consent(config).value}.")


def _validate_autocapture_scope(*, is_global: bool, project: str | None) -> None:
    if is_global and project:
        raise click.UsageError("--global and --project NAME are mutually exclusive.")


@autocapture_group.command("on")
@click.option("--global", "is_global", is_flag=True, help="Enable automatic capture everywhere.")
@click.option("--project", default=None, help="Project name (auto-detected from the current directory if not set).")
@click.option("--yes", is_flag=True, help="Skip the global consent confirmation prompt.")
def autocapture_on(is_global: bool, project: str | None, yes: bool):
    """Enable automatic capture for one project or everywhere."""
    from poppy.capture import policy
    from poppy.project import project_from_cwd

    _validate_autocapture_scope(is_global=is_global, project=project)
    poppy_dir = _get_poppy_dir()
    config = load_config(poppy_dir)

    if is_global:
        # The disclosure copy is frozen consent evidence and must be surfaced even
        # when --yes skips the interactive confirmation.
        click.echo(_consent_disclosure())
        if not yes:
            prompt = (
                "You previously opted out of automatic capture. Re-enable it everywhere?"
                if policy.effective_consent(config) is policy.Consent.DENIED
                else "Enable automatic capture everywhere?"
            )
            click.confirm(prompt, default=True, abort=True)
        _set_consent(config, granted=True)
        telemetry.capture_once(poppy_dir, "consent_granted", {"via": "consent_cmd"})
        click.echo("✓ Automatic capture consent granted.")
        click.echo(policy.status_message(policy.evaluate(config)))
        return

    project = project or project_from_cwd(os.getcwd())
    if not project:
        raise click.UsageError(
            "Could not identify a project for the current directory. Run this inside a "
            "project (a repo with a CLAUDE.md, pyproject.toml, package.json, .git, ...) "
            "or pass --project NAME."
        )
    changed = config.enable_project(project)
    if changed:
        save_config(config)
    click.echo(
        f"Automatic capture is back on for '{project}'."
        if changed
        else f"Automatic capture was already on for '{project}'."
    )
    if policy.effective_consent(config) is not policy.Consent.GRANTED:
        click.echo("Run `poppy autocapture on --global` to enable capture everywhere.")


@autocapture_group.command("off")
@click.option("--global", "is_global", is_flag=True, help="Disable automatic capture everywhere.")
@click.option("--project", default=None, help="Project name (auto-detected from the current directory if not set).")
def autocapture_off(is_global: bool, project: str | None):
    """Disable automatic capture for one project or everywhere."""
    import sys

    from poppy.project import project_from_cwd

    _validate_autocapture_scope(is_global=is_global, project=project)
    config = load_config(_get_poppy_dir())

    if is_global:
        _set_consent(config, granted=False)
        click.echo("Automatic capture disabled. This choice persists across upgrades.")
        return

    detected_project = project or project_from_cwd(os.getcwd())
    if not detected_project:
        raise click.UsageError(
            "Could not identify a project for the current directory. Run this inside a "
            "project (a repo with a CLAUDE.md, pyproject.toml, package.json, .git, ...) "
            "or pass --project NAME."
        )

    if project is None:
        # Consent withdrawal must not fail open: non-interactive callers need to
        # state whether they mean the current project or every project.
        if not sys.stdin.isatty():
            raise click.UsageError("Choose a scope explicitly with --global or --project NAME.")
        scope = click.prompt(
            "Turn automatic capture off for",
            type=click.Choice(["just this project", "everywhere"], case_sensitive=False),
        )
        if scope == "everywhere":
            _set_consent(config, granted=False)
            click.echo("Automatic capture disabled. This choice persists across upgrades.")
            return

    changed = config.disable_project(detected_project)
    if changed:
        save_config(config)
    click.echo(
        f"Automatic capture is off for '{detected_project}'."
        if changed
        else f"Automatic capture was already off for '{detected_project}'."
    )
    click.echo("Recall still works here. Run `poppy autocapture on` to re-enable capturing.")


@cli.command("consent")
@click.option("--enable", "action", flag_value="enable", help="Grant consent and enable automatic capture.")
@click.option("--disable", "action", flag_value="disable", help="Opt out of automatic capture (persists).")
@click.option("--status", "action", flag_value="status", default=True, help="Show consent and capture status.")
def consent(action: str):
    """Manage consent for automatic memory capture.

    Auto-capture is enabled by default but stays inert until you record a
    one-time consent. Both granting and opting out persist permanently.
    Prefer ``poppy autocapture`` for the unified project and global controls.
    """
    from poppy.capture import policy

    config = load_config(_get_poppy_dir())
    if action == "enable":
        click.echo(_consent_disclosure())
        _set_consent(config, granted=True)
        telemetry.capture_once(_get_poppy_dir(), "consent_granted", {"via": "consent_cmd"})
        click.echo("✓ Automatic capture consent granted.")
        click.echo(policy.status_message(policy.evaluate(config)))
    elif action == "disable":
        _set_consent(config, granted=False)
        click.echo("Automatic capture disabled. This choice persists across upgrades.")
    else:
        click.echo(f"Consent: {policy.effective_consent(config).value}")
        click.echo(policy.status_message(policy.evaluate(config)))


@cli.command("capture", hidden=True)
@click.option("--off", "action", flag_value="off", help="Turn automatic capture off for this project.")
@click.option("--on", "action", flag_value="on", help="Turn automatic capture back on for this project.")
@click.option("--status", "action", flag_value="status", default=True, help="Show capture status for this project.")
@click.option("--project", default=None, help="Project name (auto-detected from the current directory if not set).")
def capture(action: str, project: str | None):
    """Turn automatic capture on or off for the current project.

    A per-project scoping control on top of the global consent: keep a
    sensitive repo out of memory without turning capture off everywhere. Global
    consent is unchanged; recall keeps working in a disabled project, only
    automatic capture stops. Run inside the repo, or pass ``--project``.
    """
    from poppy.capture import policy
    from poppy.project import project_from_cwd

    project = project or project_from_cwd(os.getcwd())
    if action in ("off", "on") and not project:
        raise click.UsageError(
            "Could not identify a project for the current directory. Run this inside a "
            "project (a repo with a CLAUDE.md, pyproject.toml, package.json, .git, ...) "
            "or pass --project NAME."
        )

    config = load_config(_get_poppy_dir())
    if action == "off":
        changed = config.disable_project(project)
        if changed:
            save_config(config)
        click.echo(
            f"Automatic capture is off for '{project}'."
            if changed
            else f"Automatic capture was already off for '{project}'."
        )
        legacy_on_command = " ".join(("poppy", "capture --on"))
        click.echo(f"Recall still works here. Run `{legacy_on_command}` to re-enable capturing.")
    elif action == "on":
        changed = config.enable_project(project)
        if changed:
            save_config(config)
        click.echo(
            f"Automatic capture is back on for '{project}'."
            if changed
            else f"Automatic capture was already on for '{project}'."
        )
    else:
        scope = f"'{project}'" if project else "the current directory (no project detected)"
        status = policy.evaluate(config, project=project)
        state = "off" if config.is_project_disabled(project) else "on"
        click.echo(f"Capture for {scope}: {state}.")
        click.echo(policy.status_message(status))


def _maybe_prompt_consent(config, *, assume_yes: bool) -> None:
    """Record auto-capture consent during setup under the consent and
    default-on precedence rule.

    On a TTY, ask y/n and persist the answer. Non-interactively, leave consent
    pending (the SessionStart notice + ``poppy autocapture on --global`` carry it).
    Never re-prompts a user who already granted (incl. grandfathered) or opted out.
    """
    import sys

    from poppy.capture import policy

    if policy.effective_consent(config) is not policy.Consent.PENDING:
        return
    if assume_yes:
        click.echo(_consent_disclosure())
        _set_consent(config, granted=True)
        telemetry.capture_once(_get_poppy_dir(), "consent_granted", {"via": "setup_yes"})
        click.echo("✓ Automatic capture enabled.")
        return
    if sys.stdin.isatty():
        granted = click.confirm(_consent_disclosure(), default=True)
        _set_consent(config, granted=granted)
        if granted:
            telemetry.capture_once(_get_poppy_dir(), "consent_granted", {"via": "setup_prompt"})
        click.echo(
            "✓ Automatic capture enabled."
            if granted
            else "Automatic capture left off (enable later with `poppy autocapture on --global`)."
        )
    else:
        telemetry.capture_once(_get_poppy_dir(), "consent_pending_shown", {"channel": "setup"})
        click.echo("\nAutomatic capture is pending your consent. Run `poppy autocapture on --global` to turn it on.")


@cli.group()
def setup():
    """Set up integrations."""
    pass


def _print_install_paths(paths: dict, client: str) -> None:
    for label, path in paths.items():
        click.echo(f"  {label}: {path}")
    _warn_if_poppy_unresolved()
    click.echo(f"\nPoppy is ready. Restart {client} to activate.")


def _warn_if_poppy_unresolved() -> None:
    from poppy.setup.claude_code import get_poppy_executable

    if get_poppy_executable() == "poppy":
        click.echo(
            "Warning: poppy could not be resolved to an absolute path; the client may not find poppy on its PATH.",
            err=True,
        )


def _install_or_abort(*, daemon_mode: bool = False, **kwargs) -> dict:
    """Run install_for_client, turning a corrupt-config abort into a clean CLI error."""
    from poppy.setup.claude_code import CorruptConfigError, install_for_client, validate_client_config

    try:
        if daemon_mode:
            validate_client_config(
                client=kwargs["client"],
                claude_config_dir=kwargs.get("claude_config_dir"),
                install_hooks=kwargs.get("install_hooks", True),
            )
            kwargs.update(_daemon_setup_kwargs())
        return install_for_client(**kwargs)
    except CorruptConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _record_agent_setup(client: str) -> None:
    """Emit setup telemetry: the per-install ``agent_setup`` event plus the
    once-per-device ``setup_completed`` funnel milestone. Content-free."""
    poppy_dir = _get_poppy_dir()
    telemetry.capture(poppy_dir, "agent_setup", {"agent": client})
    telemetry.capture_once(poppy_dir, "setup_completed", {"agent": client})


def _daemon_setup_kwargs() -> dict:
    """Bootstrap the shared daemon before any client configuration is changed."""
    from poppy.mcp_server.auth import ensure_daemon_token
    from poppy.mcp_server.lifecycle import (
        LifecycleError,
        daemon_port,
        install_agent,
        start_daemon,
        wait_for_daemon_token,
    )

    poppy_dir = _get_poppy_dir()
    try:
        # Ensure a token EXISTS (seed mode: fills an absent file, never clobbers a
        # live one) so a fresh machine has something for the service to serve, but
        # do NOT bake this value: the daemon may rotate it on startup from its own
        # env (POPPY_DAEMON_TOKEN in a service drop-in). We bake what it serves.
        ensure_daemon_token(poppy_dir)
        install_agent(poppy_dir)
        click.echo(start_daemon(poppy_dir))
    except (LifecycleError, RuntimeError) as exc:
        raise click.ClickException(
            f"{exc}\nThe daemon agent files may remain installed; the client config was not changed."
        ) from exc
    # Poll the authenticated /status endpoint until the daemon is ready AND any
    # startup rotation has settled, then bake exactly the token it serves. A bare
    # read here would race an async service rotation and bake a stale token.
    # start_daemon can return early on a lock already held by the
    # just-started service, so readiness must be confirmed separately.
    token = wait_for_daemon_token(poppy_dir)
    if not token:
        raise click.ClickException(
            "The Poppy daemon did not become ready with a token; the client config was not changed."
        )
    return {"daemon": True, "daemon_port": daemon_port(), "daemon_token": token}


def _install_simple_client(client: str, *, daemon_mode: bool = False) -> None:
    """Install Poppy MCP server and the client's global primer when applicable."""
    paths = _install_or_abort(client=client, daemon_mode=daemon_mode)
    _print_install_paths(paths, client)
    _record_agent_setup(client)
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
@click.option("--daemon", "daemon_mode", is_flag=True, help="Use the shared authenticated HTTP daemon.")
def setup_claude_code(hooks: bool, claude_md: bool, yes: bool, daemon_mode: bool):
    """Install Poppy into Claude Code (MCP + hooks + CLAUDE.md primer)."""
    claude_dir = Path(os.environ["CLAUDE_CONFIG_DIR"]) if os.environ.get("CLAUDE_CONFIG_DIR") else None

    paths = _install_or_abort(
        client="claude-code",
        claude_config_dir=claude_dir,
        install_hooks=hooks,
        install_claude_md=claude_md,
        daemon_mode=daemon_mode,
    )
    _print_install_paths(paths, "claude-code")
    _record_agent_setup("claude-code")

    # Consent and default-on precedence: ask once on a TTY,
    # otherwise leave pending for the SessionStart notice + `poppy consent`.
    _maybe_prompt_consent(load_config(_get_poppy_dir()), assume_yes=yes)


@setup.command("copilot-cli")
def setup_copilot_cli():
    """Install Poppy into GitHub Copilot CLI (MCP + instructions primer)."""
    _install_simple_client("copilot-cli")


@setup.command("pi")
def setup_pi():
    """Install Poppy into Pi (MCP via pi-mcp-adapter + AGENTS.md primer)."""
    _install_simple_client("pi")


@setup.command("cursor")
@click.option(
    "--hooks/--no-hooks",
    default=True,
    help="Install lifecycle hooks. Default: enabled.",
)
@click.option("--daemon", "daemon_mode", is_flag=True, help="Use the shared authenticated HTTP daemon.")
def setup_cursor(hooks: bool, daemon_mode: bool):
    """Install Poppy into Cursor (MCP + hooks)."""
    paths = _install_or_abort(client="cursor", install_hooks=hooks, daemon_mode=daemon_mode)
    _print_install_paths(paths, "cursor")
    _record_agent_setup("cursor")
    if hooks:
        from poppy.setup.claude_code import cursor_unknown_hook_events

        unknown_events = cursor_unknown_hook_events(paths["Cursor hooks"])
        if unknown_events:
            click.echo(
                "\nWarning: Cursor will ignore the entire hooks file until these unrecognized event keys "
                f"are removed: {', '.join(unknown_events)}",
                err=True,
            )
        else:
            click.echo(
                f"\nCursor hooks are installed at {paths['Cursor hooks']}.\n"
                "  Capture starts immediately once automatic capture is enabled. No hook trust step is required."
            )


@setup.command("windsurf")
def setup_windsurf():
    """Install Poppy into Windsurf (MCP only)."""
    _install_simple_client("windsurf")


@setup.command("codex")
@click.option(
    "--hooks/--no-hooks",
    default=True,
    help="Install lifecycle hooks. Default: enabled.",
)
@click.option("--daemon", "daemon_mode", is_flag=True, help="Use the shared authenticated HTTP daemon.")
def setup_codex(hooks: bool, daemon_mode: bool):
    """Install Poppy into Codex (MCP + hooks + AGENTS.md primer)."""
    paths = _install_or_abort(client="codex", install_hooks=hooks, daemon_mode=daemon_mode)
    _print_install_paths(paths, "codex")
    _record_agent_setup("codex")
    if hooks:
        click.echo(
            "\nCodex hook trust is required before capture starts:\n"
            "  Complete one-time interactive approval: run Codex in the TUI once and approve the Poppy hooks.\n"
            "  Headless `codex exec` silently skips the hooks until they are trusted.\n"
            "  Editing a hook command later changes its hash and requires approval again."
        )


@setup.command("gemini")
@click.option("--daemon", "daemon_mode", is_flag=True, help="Use the shared authenticated HTTP daemon.")
def setup_gemini(daemon_mode: bool):
    """Install Poppy into Gemini CLI (MCP only)."""
    _install_simple_client("gemini", daemon_mode=daemon_mode)


@setup.command("vscode")
@click.option("--daemon", "daemon_mode", is_flag=True, help="Use the shared authenticated HTTP daemon.")
def setup_vscode(daemon_mode: bool):
    """Install Poppy into VS Code (MCP only)."""
    _install_simple_client("vscode", daemon_mode=daemon_mode)


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
    _warn_if_poppy_unresolved()
    click.echo(
        "\nPoppy is registered as a Goose extension. Run `goose session` to "
        "start a session. Poppy will be available via the standard MCP tool surface."
    )
    _record_agent_setup("goose")


@setup.command("hermes-agent")
def setup_hermes_agent():
    """Install Poppy as a Hermes Agent (Nous Research) memory provider plugin.

    Hermes doesn't speak MCP. Instead, it loads memory providers from
    `~/.hermes/plugins/<name>/`. This drops the Poppy plugin + sets
    `memory.provider: poppy` in `~/.hermes/config.yaml`.
    """
    from poppy.setup.hermes import install_for_hermes

    paths = install_for_hermes()
    for label, path in paths.items():
        click.echo(f"  {label}: {path}")
    click.echo(
        "\nPoppy is the active hermes memory provider. Run `hermes memory status` "
        "to verify, then start a hermes session. It will recall relevant memories "
        "before each turn."
    )
    _record_agent_setup("hermes-agent")


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
    from poppy.setup.claude_code import CLAUDE_IMPORT_PROMPT, CLAUDE_MD_BODY

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

    paths = _install_or_abort(client="claude-desktop")
    if "backup" in paths:
        click.echo(f"  backup: {paths['backup']}")
    click.echo(f"  MCP config: {paths['MCP config']}")
    if "MSIX backup" in paths:
        click.echo(f"  MSIX backup: {paths['MSIX backup']}")
    if "MSIX MCP config" in paths:
        click.echo(f"  MSIX MCP config: {paths['MSIX MCP config']}")
        click.echo(
            "\nNote: Claude Desktop from the Microsoft Store/MSIX reads a virtualized config. "
            "Poppy wrote both config files due to this upstream issue: "
            "https://github.com/anthropics/claude-code/issues/26073"
        )
    _warn_if_poppy_unresolved()
    _record_agent_setup("claude-desktop")
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
    help="Override trags-api-url (defaults to the configured value or https://api.trags.ai).",
)
def setup_trags(api_url: str | None):
    """One-command device-code onboarding for Trags cloud sync.

    Opens your browser to authorize this machine, then writes the returned
    API key to ~/.poppy/config.json. After this you can run `poppy sync push`.
    """
    from poppy.setup.trags import run_device_code_flow

    run_device_code_flow(api_url)
    _record_agent_setup("trags")


from poppy.cli.hooks import hook as _hook_group  # noqa: E402

cli.add_command(_hook_group)


@cli.group("build")
def build_group():
    """Build distributable artifacts."""
    pass


@build_group.command(
    "mcpb",
    short_help="Build a .mcpb bundle. Needs a source checkout.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("dist"),
    help="Where to write the .mcpb file (default: ./dist).",
)
def build_mcpb_cmd(output_dir: Path):
    """Build a Claude Desktop Extension bundle (.mcpb) from a source checkout.

    This is a maintainer tool, not a supported way to install Poppy. It reads
    `pyproject.toml` and `mcpb/` from the repository, so it cannot run from a
    Poppy installed with pipx or pip. To use Poppy with the Claude desktop app,
    run `poppy setup claude-desktop` instead.
    """
    from poppy.build_mcpb import build_mcpb

    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "pyproject.toml").is_file() or not (repo_root / "mcpb" / "manifest.json").is_file():
        raise click.ClickException(
            "poppy build mcpb only works from a source checkout. Use `poppy setup claude-desktop` instead."
        )
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


def _sync_client():
    """Build a Trags HTTP client from poppy config. Raises Abort if not set."""
    from poppy.config import load_config, resolve_trags_api_key
    from poppy.sync import TragsClient

    cfg = load_config(_get_poppy_dir())
    api_key = resolve_trags_api_key(cfg)
    if not api_key:
        click.echo(
            "Trags API key not configured. Set it with:\n"
            "  poppy config set trags-api-key <usr_xxxxx>\n"
            "  poppy config set trags-api-url <https://your-trags-host>   # optional, defaults to https://api.trags.ai",
            err=True,
        )
        raise click.Abort()
    return TragsClient(base_url=cfg.trags_api_url, api_key=api_key), cfg.trags_api_url


def _sync_tombstones():
    from poppy.tombstones import TombstoneStore

    return TombstoneStore(_get_poppy_dir() / "memories.db")


def _abort_on_auth_error(exc, url: str, *, record: bool = True) -> None:
    """Turn a 401 from Trags into a clean message + Abort, not a raw traceback.

    The failure is also written to the sync state's auth slot, in the same words
    the auto-sync worker uses, so `poppy sync status` shows a `last error:` line
    after a manual `sync push|pull|run` too. Before this the CLI printed the 401
    and forgot it, and three consecutive rejections in a row left `status`
    looking healthy.

    ``record=False`` for a ``--dry-run``, which must not write anything.
    """
    if record:
        from poppy.sync.client import auth_error_message
        from poppy.sync.state import record_error

        record_error(_get_poppy_dir(), url, auth_error_message(exc), source="auth")
    click.echo(
        f"Trags rejected the API key (401 Unauthorized): {exc}\n"
        "The key may be revoked or wrong. Refresh it with `poppy setup trags` "
        "(or `poppy config set trags-api-key <usr_xxxxx>`).",
        err=True,
    )
    raise click.Abort()


def _abort_on_quota_error(exc) -> None:
    """Turn a 402 quota cap into a clean message + Abort, not a silent success.

    `push()` already froze the watermark and recorded the error in the sync
    state (so `poppy sync status` shows a `last error:` line); here we surface
    the server's upgrade prompt once and exit non-zero.
    """
    click.echo(str(exc), err=True)
    raise click.Abort()


def _abort_on_transport_error(exc) -> None:
    """Turn an unreachable Trags host into one offline line, not a traceback.

    The line says only what this run can prove: that the host gave no usable
    answer, and what the run had already done when it went away. A sync that
    applied pulled rows (a pulled TOMBSTONE deletes a local memory) or sent part
    of its backlog before the host died must never be summarised as "nothing
    happened" — the user would have no reason to go looking for the memory
    inside the tombstone-retention window. A lost response proves even less: the
    server may have accepted the write whose reply never came back, so that case
    is reported as unknown rather than guessed either way.

    A `--dry-run`'s counts get the conditional they describe. Both sync loops
    count what they WOULD do so the counter lines can show it, and reporting
    those as completed work is the same lie the other way round. A dry run also
    issues nothing but reads, so no request of its own could have landed however
    the connection failed, and it says nothing about an unknown outcome.

    Still one line and still `click.Abort` — so click's own `Aborted!` marker
    follows it, exactly as the auth and quota handlers above already do.
    """
    simulated = getattr(exc, "simulated", False)
    pulled = getattr(exc, "applied_pulled", 0)
    pushed = getattr(exc, "sent_pushed", 0)
    unknown = getattr(exc, "outcome_unknown", False) and not simulated
    did = []
    if pulled:
        did.append(f"applied {pulled} pulled change{'' if pulled == 1 else 's'}")
    if pushed:
        did.append(f"sent {pushed} row{'' if pushed == 1 else 's'}")
    parts = [f"{exc}."]
    if did:
        lead = "This run would have" if simulated else "This run"
        parts.append(f"{lead} {' and '.join(did)} before the host stopped answering.")
    elif not unknown:
        parts.append("Nothing was sent and nothing changed locally.")
    if unknown:
        parts.append("The last request may still have completed.")
    parts.append("The next sync retries.")
    click.echo(" ".join(parts), err=True)
    raise click.Abort()


def _fail_if_unresolved_error(url) -> None:
    """Exit non-zero if a sync command didn't itself raise but left an error
    unresolved — a no-op (zero-request) push, or a pull-origin failure a lone
    push can't clear. Without this the command looks healthy while a row stays
    stuck; `poppy sync status` would be the only hint.
    """
    from poppy.sync import remote_state_for

    rs = remote_state_for(_get_poppy_dir(), url)
    if rs.errors:
        for kind, message in rs.errors.items():
            click.echo(f"sync incomplete, {kind}: {message}", err=True)
        raise click.Abort()


def _print_push(res) -> None:
    click.echo(
        f"  push: {res.sent_live} live, {res.sent_tombstones} tombstones, "
        f"{res.skipped} skipped, {res.rejected} rejected, {res.errors} errors"
    )


def _print_pull(res) -> None:
    click.echo(
        f"  pull: {res.applied_live} live, {res.applied_tombstones} tombstones, "
        f"{res.skipped_stale} skipped (local newer), {res.errors} errors"
    )


@sync_group.command("push")
@click.option("--dry-run", is_flag=True, help="Preview what would be sent; store upgrades still run on open.")
def sync_push(dry_run: bool):
    """Send local memories + tombstones to Trags (since the last push watermark)."""
    from poppy.sync import TragsAuthError, TragsQuotaError, TragsTransportError, load, push

    client, url = _sync_client()
    try:
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
        if not dry_run:
            _fail_if_unresolved_error(url)
    except TragsQuotaError as exc:
        _abort_on_quota_error(exc)
    except TragsAuthError as exc:
        _abort_on_auth_error(exc, url, record=not dry_run)
    except TragsTransportError as exc:
        _abort_on_transport_error(exc)


@sync_group.command("pull")
@click.option("--dry-run", is_flag=True, help="Preview incoming changes; store upgrades still run on open.")
def sync_pull(dry_run: bool):
    """Apply Trags rows newer than our last pull watermark."""
    from poppy.sync import TragsAuthError, TragsTransportError, load, pull

    client, url = _sync_client()
    try:
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
        if not dry_run:
            _fail_if_unresolved_error(url)
    except TragsAuthError as exc:
        _abort_on_auth_error(exc, url, record=not dry_run)
    except TragsTransportError as exc:
        _abort_on_transport_error(exc)


@sync_group.command("status")
def sync_status():
    """Show watermarks and last-sync time."""
    from poppy.config import load_config, resolve_trags_api_key, trags_api_key_location
    from poppy.sync import remote_state_for

    cfg = load_config(_get_poppy_dir())
    if not resolve_trags_api_key(cfg):
        if trags_api_key_location(cfg.poppy_dir) == "keychain-unreadable":
            click.echo(
                "This session cannot read the OS keychain, so it cannot tell whether a Trags key is stored there. "
                "Run from an interactive login session, set POPPY_TRAGS_API_KEY for background jobs, "
                "or run `poppy setup trags` if you have not configured Trags yet."
            )
        else:
            click.echo(
                "Trags not configured. Run `poppy setup trags` to connect this machine "
                "(or `poppy config set trags-api-key <usr_xxxxx>` if you already have a key)."
            )
        return
    rs = remote_state_for(_get_poppy_dir(), cfg.trags_api_url)
    pending = (_get_poppy_dir() / "sync.pending").exists()
    click.echo(f"  url:             {cfg.trags_api_url}")
    click.echo(f"  auto-sync:       {cfg.auto_sync}{'  (pending)' if pending else ''}")
    click.echo(f"  last pulled at:  {rs.last_pulled_at or 'not yet'}")
    click.echo(f"  last pushed at:  {rs.last_pushed_at or 'not yet'}")
    click.echo(f"  last synced at:  {rs.last_synced_at or 'not yet'}")
    click.echo(f"  pushed (total):  {rs.pushed_count}")
    click.echo(f"  pulled (total):  {rs.pulled_count}")
    # One line per outstanding origin — a push failure and a pull failure can
    # both be unresolved at once.
    for kind, message in rs.errors.items():
        click.echo(f"  last error ({kind}): {message}")


@sync_group.command("run")
@click.option("--dry-run", is_flag=True, help="Preview sync changes; store upgrades still run on open.")
def sync_run(dry_run: bool):
    """Pull then push: full bidirectional sync."""
    from poppy.sync import TragsAuthError, TragsQuotaError, TragsTransportError
    from poppy.sync import sync as do_sync

    client, url = _sync_client()
    try:
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
        if not dry_run:
            _fail_if_unresolved_error(url)
    except TragsQuotaError as exc:
        _abort_on_quota_error(exc)
    except TragsAuthError as exc:
        _abort_on_auth_error(exc, url, record=not dry_run)
    except TragsTransportError as exc:
        _abort_on_transport_error(exc)


@sync_group.command("_auto-worker", hidden=True)
def sync_auto_worker():
    """Internal: detached worker spawned by auto-sync triggers. Do not invoke directly."""
    from poppy.sync.auto import run_worker

    run_worker(_get_poppy_dir())


@cli.command()
def doctor():
    """Verify the Poppy installation: engine, storage, MCP config, hooks."""
    from poppy.cli.doctor import run_doctor

    run_doctor()


def main() -> None:
    """Console-script entrypoint.

    Wraps the click group so expected operational failures (e.g. retrieval
    models unavailable offline with a cold cache) render as a one-line
    actionable message instead of a raw traceback. Imported lazily from
    ``poppy.errors`` (which is import-cheap, no ML deps) so a deps-missing
    install still runs.
    """
    from poppy.errors import PoppyError

    try:
        cli()
    except PoppyError as exc:
        # EncryptionError includes KeychainUnavailable and DependencyMissing, so
        # a keychain failure or a missing extra while opening an encrypted store
        # (recall/list/hooks) renders as one line instead of a raw traceback.
        click.echo(f"poppy: {exc}", err=True)
        raise SystemExit(1) from None
