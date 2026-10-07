# Poppy architecture

A map of the code for people reading or changing it. It covers what each part
owns, how the four main operations travel through the code, which processes
run, what lives in the data directory, which way imports point, and what each
lock guarantees. Paths are relative to `src/poppy/`. A hop written as
`file.py:name` names a function or class in that file; open the file for the
details, which this page does not repeat.

## Module map

| Part | Owns |
|---|---|
| [`cli/main.py`](../src/poppy/cli/main.py) | Every user command (the `cli` click group) and the `main` entry point, which prints any `PoppyError` as one line and exits 1. |
| [`cli/hooks.py`](../src/poppy/cli/hooks.py) | Hook entry points for Claude Code, Cursor and Codex, and the detached capture workers they start. |
| [`cli/doctor.py`](../src/poppy/cli/doctor.py) | `poppy doctor`: one function per check, walked by `run_doctor`. |
| [`write_flow.py`](../src/poppy/write_flow.py) | The shared write core: `remember`, `forget`, `restore`. CLI, MCP and the dashboard are thin adapters over it. |
| [`lifecycle.py`](../src/poppy/lifecycle.py) | Edit, supersede, and time-to-live parsing. |
| [`engine/`](../src/poppy/engine/) | Retrieval. `interface.py` is the `RetrievalEngine` contract, `registry.py` maps names to engines, `seed.py` is keyword search on SQLite FTS5, `bloom.py` is a thin subclass of `_hybrid.py` (keywords plus vectors, then a cross-encoder rerank). Model loading lives in `_fastembed_loader.py`, `_model_holder.py` and `_model_cache.py`; `_daemon_models.py` borrows the daemon's loaded models over HTTP; `migration.py` re-embeds when the engine changes. |
| [`runtime.py`](../src/poppy/runtime.py) | `get_engine` builds the configured engine and falls back from `bloom` to `seed`; `get_fast_engine` returns a model-free `seed` engine for hooks. |
| [`tombstones.py`](../src/poppy/tombstones.py) | The Trash store (seven-day soft delete) and sync bookkeeping: which server knows which memory, and which deletions were sent. `ui/tombstones.py` is an alias kept for old imports. |
| [`capture/`](../src/poppy/capture/) | Building blocks of automatic capture: consent policy, transcript readers, window and watermark, turn cadence, the per-session lock, the reconciler (dedupe against the store before writing), the orchestrator, journal, backend health, secret redaction, and the session banner. |
| [`consolidation.py`](../src/poppy/consolidation.py) | The capture entry points (`consolidate_capture_event`, `consolidate_compact_event`, `consolidate_stop_event`) and the LLM backends behind `call_llm`: the host coding-agent CLI first, then an OpenAI-compatible endpoint. |
| [`sync/`](../src/poppy/sync/) | Cloud sync. `__init__.py` holds `sync`, `pull` and `push`; `client.py` is the HTTP client; `serializer.py` the wire format; `state.py` the sync state file; `auto.py` the trigger and background worker. |
| [`mcp_server/`](../src/poppy/mcp_server/) | `server.py` defines the MCP tools; `daemon.py` the shared HTTP daemon, which also serves model inference; `auth.py` its token; `forwarder.py` bridges a stdio client to the daemon; `lifecycle.py` installs, starts and stops it with launchd or systemd; `assembly.py` formats recall context. |
| [`ui/`](../src/poppy/ui/) | The local web dashboard (`server.py` plus `static/`). |
| [`setup/`](../src/poppy/setup/) | Writing Poppy into other tools' config. `claude_code.py` serves most clients despite its name; `goose.py`; `hermes.py` installs the plugin file `hermes_plugin.py.txt`; `trags.py` is cloud onboarding. |
| [`integrations/`](../src/poppy/integrations/) | One-off importers for Claude Code auto-memory files and Hermes memories. |
| Core helpers | `models.py` (memory dataclasses), `errors.py` (`PoppyError` and its families, cheap to import), `paths.py` (data directory creation and `write_text_atomic`, stdlib only), `db.py` (SQLite connections, gates, `write_txn`), `config.py` (`config.json` through a key registry), `keychain.py` and `encryption.py` (optional encryption at rest), `writers.py` (live writer registry), `telemetry.py`, `update_check.py`, `sources.py` (which client wrote a memory), `project.py` (project name from a directory), `relevance.py` (score floor for recall), `build_mcpb.py` (Claude Desktop bundle). |

## Main paths

**remember (CLI).** `cli/main.py:remember` → `write_flow.py:remember` →
optional conflict check `capture/reconciler.py:detect_conflicts` → either
`lifecycle.py:supersede_memory` or `engine.ingest` (for `bloom`,
`engine/_hybrid.py:HybridEngine.ingest`) → `write_flow.py:_emit_memory_write`
(telemetry) → `write_flow.py:_trigger_autosync` → `sync/auto.py:trigger`.

**remember (MCP).** The `remember` tool in `mcp_server/server.py:create_mcp_server`
→ `mcp_server/server.py:PoppyMcpServer.handle_remember` → the same
`write_flow.py:remember`. The dashboard reaches `write_flow` the same way for
forget and restore.

**recall.** `cli/main.py:recall` → `cli/main.py:_get_engine` →
`runtime.py:get_engine` → `engine/registry.py:resolve_engine` →
`engine/bloom.py:BloomEngine` → `engine/_hybrid.py:HybridEngine.retrieve` →
`HybridEngine._hybrid_retrieve` (keyword search plus vectors, merged) →
`BloomEngine._rerank` → models from `engine/_fastembed_loader.py:FastembedModels`
in process, or from the daemon through `engine/_daemon_models.py:DaemonModels`.
MCP recall enters at `PoppyMcpServer.handle_recall`. Hooks recall through
`runtime.py:get_fast_engine` so they never load a model.

**capture (mid-session).** `cli/hooks.py:user_prompt_submit` →
`cli/hooks.py:_maybe_fire_capture` (consent, cadence) →
`cli/hooks.py:_spawn_detached_worker` starts `poppy hook _capture-worker` →
`cli/hooks.py:_capture_worker` → `capture/orchestrator.py:run_capture_worker`,
given `consolidation.py:consolidate_capture_event` →
`capture/lock.py:single_flight` → `capture/watermark.py:get_watermark` and
`capture/window.py:read_window` → `consolidation.py:_orchestrator` →
`capture/orchestrator.py:CaptureOrchestrator.run` →
`consolidation.py:call_llm` → `capture/reconciler.py:reconcile_and_ingest` →
`engine.ingest` → `capture/watermark.py:set_watermark` (only after a
successful extract and reconcile) → back in `run_capture_worker`,
`sync/auto.py:trigger` if anything was stored. The
end-of-session and post-compaction backstops follow the same shape through
`_session_end_worker` and `_post_compact_worker`. Note that the flow starts in
`cli/`, passes through `consolidation.py`, and only its middle lives in
`capture/`.

**sync (after any write).** `sync/auto.py:trigger` touches `sync.pending` and
`sync/auto.py:_spawn_worker` starts `poppy sync _auto-worker` →
`cli/main.py:sync_auto_worker` → `sync/auto.py:run_worker` (takes `sync.lock`)
→ `sync/auto.py:_drain_pending` → `sync/auto.py:_do_sync` →
`sync/__init__.py:sync` → `pull` (each row applied by `_apply_pulled_row` under
the write gate) then `push` → `sync/client.py:TragsClient`. Wire conversion is
`sync/serializer.py:memory_to_wire`, `tombstone_to_wire` and `wire_to_memory`;
deletions come from `tombstones.py:TombstoneStore`; watermarks and errors are
saved with `sync/state.py:mutate_remote`. The manual `poppy sync run`
(`cli/main.py:sync_run`) calls the same `sync/__init__.py:sync` directly.

## Process model

Several kinds of Poppy process can run at once on one machine, all against the
same data directory (`~/.poppy`, or `POPPY_DIR`):

| Process | Started by | Code |
|---|---|---|
| CLI command | the user | `cli/main.py:main` |
| Hook (short, must exit fast) | Claude Code, Cursor, Codex | `cli/hooks.py` |
| Capture workers (three kinds, detached) | hooks, via `cli/hooks.py:_spawn_detached_worker` | `cli/hooks.py:_capture_worker`, `_post_compact_worker`, `_session_end_worker` |
| Auto-sync worker (detached) | any write, via `sync/auto.py:_spawn_worker` | `sync/auto.py:run_worker` |
| Daemon: shared MCP over HTTP on loopback, plus model inference for other processes | a launchd or systemd service, or `poppy daemon start` (`mcp_server/lifecycle.py:start_daemon`) | `cli/main.py:serve` with HTTP transport, `mcp_server/daemon.py:create_http_app` |
| stdio MCP server: a forwarder to the daemon, or an in-process server when no daemon answers | each MCP client running `poppy serve` | `cli/main.py:serve`, `mcp_server/forwarder.py:run_forwarder`, `mcp_server/server.py:create_mcp_server` |
| Dashboard | `poppy ui` | `ui/server.py:create_app` |
| Claude Desktop bundle | Claude Desktop | [`mcpb/server/main.py`](../mcpb/server/main.py) at the repo root, which builds the same server with `create_mcp_server` |
| Hermes plugin | Hermes Agent, inside its own Python; it runs `poppy` commands rather than importing Poppy | `setup/hermes_plugin.py.txt`, installed by `setup/hermes.py:install_for_hermes` |

Background processes are started in two styles today: `sys.argv[0]` in
`cli/hooks.py:_spawn_detached_worker` and `sync/auto.py:_spawn_worker`, and
`mcp_server/lifecycle.py:resolve_poppy_executable` for the daemon.

## Data directory

Everything below lives in the data directory. Writers are named by module.

| File | What it is | Written by |
|---|---|---|
| `memories.db` (+ `-wal`, `-shm`) | Memories, full-text index, embeddings, Trash (`ui_tombstones` table) and sync provenance (`sync_remote_memories` table) | the engines (`engine/seed.py`, `engine/_hybrid.py`), `tombstones.py`, opened through `db.py:connect` |
| `.poppy-encrypted` | Marker that the store is encrypted | `encryption.py` |
| `memories.db.enc-tmp`, `memories.db.plain-tmp`, `*.stray-*` | Short-lived files during an encryption migration, and sidecars set aside by `poppy encrypt repair` | `encryption.py`, `db.py:quarantine_sidecars` |
| `db.gate`, `write.gate` | Lock files, see below | `db.py` |
| `writers/<surface>.<pid>.lock` | One lock file per live long-running process | `writers.py:registered` |
| `config.json` | Settings, written whole | `config.py:save_config` |
| `analytics.json`, `analytics.json.lock` | Device id and telemetry bookkeeping | `telemetry.py` |
| `update_check.json`, `update_check.json.lock` | Cached result of the update check | `update_check.py` |
| `sync_state.json`, `sync_state.json.lock` | Sync watermarks and last errors, per server URL | `sync/state.py` |
| `sync.pending`, `sync.lock` | "A sync is owed" flag, and the background worker's lock | `sync/auto.py` |
| `sync-worker.log`, `sync-worker.raw.log` | Worker log, and its raw stdout and stderr | `sync/auto.py` |
| `capture_state.json`, `capture_state.json.lock` | Per-session capture watermark and turn counters | `capture/_state.py`, through `capture/watermark.py` and `capture/cadence.py` |
| `capture-<session>.flock` | Per-session capture lock (older versions used `capture-<session>.lock`) | `capture/lock.py` |
| `capture_journal.jsonl` | What each capture stored, read by the session banner and `poppy doctor` | `capture/journal.py` |
| `capture_health.json` | Health of the capture LLM backends | `capture/health.py` |
| `capture-worker.log`, `postcompact-worker.log`, `sessionend-worker.log` | Output of the detached capture workers | `cli/hooks.py:_spawn_detached_worker` |
| `postcompact-debug.log`, `sessionend-debug.log` | Scrubbed hook payloads for `poppy hook replay-compact` and `replay-session-end` | `cli/hooks.py` |
| `daemon.token`, `.daemon.token.lock` | Bearer token for the daemon, and its lock | `mcp_server/auth.py` |
| `daemon.lock` | The running daemon's lock, holding its pid | `mcp_server/daemon.py:daemon_lock` |
| `logs/daemon.log` | Daemon output when started by a service or `daemon start` | `mcp_server/lifecycle.py` |

Outside the data directory: downloaded models are cached under
`POPPY_FASTEMBED_CACHE` if set, else `$XDG_CACHE_HOME/fastembed`, else
`~/.cache/fastembed` (`engine/_model_cache.py:fastembed_cache_dir`). The
encryption key lives only in the OS keychain (`keychain.py`). The Trags key goes
to the keychain too, but falls back to `config.json` when the keychain write
cannot be read back (`config.py:_save_trags_api_key`), and
`consolidate-api-key` is always stored in `config.json`, so treat that file as
secret. `setup/` edits each client's own config
files.

## Import direction

Intended layers, lowest first. A module may import from its own layer or any
layer below it.

1. `models`, `errors`, and the leaf helpers `sources`, `project`, `relevance`
2. `paths` / `db` (with `writers`)
3. `config` (with `telemetry`, `update_check`)
4. `keychain` / `engine` / `lifecycle` (with `encryption`, `runtime`)
5. `write_flow`
6. the Trash store (`tombstones`) / `capture` (with `consolidation`)
7. `sync` / `mcp_server`
8. `ui`
9. `setup` (with `integrations`)
10. `cli` (with `build_mcpb`)

Lower layers do not import higher ones; the remaining exceptions are being
removed. Today they are:

- `engine/_daemon_models.py` imports `mcp_server/auth.py:load_daemon_token` at
  the top of the file, to read the daemon token. This is the only exception at
  module level.
- `config.py` imports `engine/registry.py` (engine name checks) and
  `keychain.py`, inside functions.
- `db.py:connect` imports `encryption.py` inside the function, for an encrypted
  store.
- `lifecycle.py` imports the Trash store inside `_supersede_memory`.
- `write_flow.py` imports `capture/reconciler.py`, the Trash store and
  `sync/auto.py` inside functions.
- `tombstones.py` imports `sync/state.py` inside a function.
- `capture/orchestrator.py:run_capture_worker` imports `sync/auto.py` inside
  the function.

Many other imports sit inside functions on purpose, to keep hooks and the CLI
fast to start; `errors.py` and `paths.py` stay cheap to import for the same
reason.

## Locks and what they guarantee

Locks here are advisory `flock` locks unless stated. Poppy supports macOS and
Linux, where `flock` is always available.

| Lock | File | Protects | If it can't be taken |
|---|---|---|---|
| SQLite write transaction (`db.py:write_txn`) | `memories.db` | A multi-statement write runs as one transaction: `BEGIN IMMEDIATE` takes the write lock up front, and any exception rolls back. Engines also hold an in-process lock around it. | Waits up to the 5 s busy timeout, then raises `database is locked`; nothing is written. |
| Encryption gate (`db.py:acquire_shared_gate`, `db.py:exclusive_gate`) | `db.gate` | Every connection holds it shared; `poppy encrypt` takes it exclusive, so a migration never runs under an open connection. | A new connection waits about 10 s, then raises `EncryptionError`. A migration waits about 5 s, then refuses and names the live writers. |
| Write gate (`db.py:write_gate`) | `write.gate` | Orders multi-step read-decide-write sequences across processes: forget, restore, edit, supersede, each pulled row, and store upgrades at open. | Gives up after about 10 s and runs the sequence anyway; callers' own checks still apply. |
| Writer registry (`writers.py:registered`) | `writers/<surface>.<pid>.lock` | Lets an encryption migration see which long-running processes are live (MCP server, daemon, dashboard, sync and capture workers). The kernel drops the lock when a process dies. | Registration failure is ignored; the process still runs. `writers.py:live_writers` removes files whose owner is gone. |
| Background sync lock (`sync/auto.py:run_worker`) | `sync.lock` | At most one auto-sync worker at a time. A manual `poppy sync` does not take it. | The new worker exits; the running one sees `sync.pending`. When a worker stops because nothing was pending, it checks `sync.pending` once more after releasing the lock and drains it, so a write that lands as it finishes is not stranded. After an auth stop, a quota stop, an exhausted round budget or a re-armed failure, `sync.pending` waits for the next trigger. |
| Sync state lock (`sync/state.py:state_lock`) | `sync_state.json.lock` | Read-modify-write of `sync_state.json` by any sync process, manual or background. | Blocks until free. If `flock` itself errors, the update runs unlocked. |
| Capture state lock (`capture/_state.py:update`) | `capture_state.json.lock` | Read-modify-write of watermarks and turn counters from concurrent hook processes. | Blocks until free. |
| Telemetry lock (`telemetry.py:_analytics_file_lock`) | `analytics.json.lock` | Updates to `analytics.json`. | Blocks until free. |
| Update check lock (`update_check.py:_cache_file_lock`) | `update_check.json.lock` | Cache refreshes and the once-only update notice. Also takes an in-process lock. | Blocks until free. |
| Daemon token lock (`mcp_server/auth.py:ensure_daemon_token`) | `.daemon.token.lock` | Creating or repairing `daemon.token`, so `daemon run`, `daemon start` and `setup --daemon` agree on one token. | Blocks until free. |
| Daemon lock (`mcp_server/daemon.py:daemon_lock`) | `daemon.lock` | One daemon per data directory. | `poppy daemon run` prints that the daemon is already running and exits. |
| Per-session capture lock (`capture/lock.py:single_flight`) | `capture-<session>.flock` | One capture worker per session at a time. The kernel releases it if the worker dies, so it has no time-to-live. | Retries for 0.1 s, then the worker skips this round; the watermark catches up next time. A `capture-<session>.lock` file left by an older version counts as held for 300 s. On a filesystem that refuses locks, capture runs without it. |
| `config.json` | none | Nothing today. `config.py:save_config` replaces the file atomically, so readers never see a torn file, but two concurrent read-modify-writes can lose one update. | Not applicable. |

`capture_journal.jsonl` and `capture_health.json` have no lock either;
`capture/health.py` writes through a temporary file unique to each writer.

## Glossary

- **Capture.** Automatic extraction of memories from a coding session by an
  LLM. The code calls it **consolidation** (`consolidation.py`,
  `consolidate_*` settings). Two other things share the word: the MCP tool
  `consolidate`, which stores a session summary and facts the agent passes in,
  and `RetrievalEngine.consolidate`, an engine method that does nothing in
  either engine.
- **Tombstone.** A Trash entry: a soft-deleted or superseded memory kept for
  seven days so it can be restored, and so the deletion can sync. Stored by
  `tombstones.py:TombstoneStore`.
- **Watermark.** A "processed up to here" marker. Capture keeps one per session
  (the last turn captured, in `capture_state.json`), advanced only after a
  successful extract and reconcile (an empty extract leaves it in place). Sync keeps one pull and one push timestamp per server (in
  `sync_state.json`).
- **Daemon.** One long-running Poppy process that serves MCP over HTTP to all
  clients and holds the retrieval models, so other processes can borrow them
  instead of loading their own.
- **Hook.** A short command a coding agent runs at a session event (session
  start, each prompt, before a tool call, compaction, session end). Hooks
  recall into the agent's context and start capture workers.
- **Writer registry.** The `writers/` folder: each live long-running process
  holds a lock on its own file there, so an encryption migration can refuse to
  run under it.
- **Engine.** A swappable retrieval backend behind `RetrievalEngine`. `seed` is
  keyword search only and needs no models. `bloom`, the default, combines
  keyword and vector search and reranks with a cross-encoder.
- **Write gate.** The `write.gate` lock that orders multi-step writes across
  processes; see the locks table.
- **Backstop.** The end-of-session and post-compaction captures that pick up
  turns the mid-session capture did not reach.
