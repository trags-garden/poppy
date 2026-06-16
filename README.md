<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme-header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/readme-header-light.svg">
  <img src="https://raw.githubusercontent.com/trags-garden/poppy/main/docs/assets/readme-header-light.svg" alt="poppy: remember what matters" width="640">
</picture>

Poppy is a developer memory CLI and MCP server. Your AI coding agent stores
decisions, preferences, and lessons in a local SQLite store, then recalls
them when they're relevant. No cloud, no API keys, no per-query cost.

```bash
curl -fsSL https://raw.githubusercontent.com/trags-garden/poppy/main/install.sh | bash
```

Then wire up your agent:

```bash
poppy setup claude-code   # or: claude-desktop, cursor, windsurf, codex,
                          #     copilot-cli, pi, goose, hermes-agent
```

## What it does

- **`remember`** stores a fact, decision, preference, or lesson.
- **`recall`** pulls the few memories most relevant to a query.
- **MCP server**: agents call `remember`/`recall`/`forget`/`consolidate`
  directly. Works with Claude Code, Claude Desktop, Cursor, Windsurf, Codex,
  Copilot CLI, Pi, Goose, and Hermes Agent.
- **Hooks**: Claude Code's `SessionStart` and `UserPromptSubmit` hooks
  surface relevant memories as additional context for every prompt.
- **Zero-touch capture**: with your one-time consent, Poppy extracts
  durable memories from your sessions automatically, locally, using the
  coding-agent CLI you already have. See
  [Zero-touch capture](#zero-touch-capture).
- **Interchangeable engines**: retrieval runs on a swappable local engine;
  the default, `bloom`, tracks the current best by benchmark. See
  [Engines](#engines).

```bash
poppy remember "we always use uv for python deps" --type preference
poppy recall "python package manager"
#   we always use uv for python deps
#     preference | 2026-05-25 | score: 0.91
```

## Zero-touch capture

With the Claude Code hooks installed, Poppy can also extract durable
memories from your sessions automatically: decisions, preferences, and
lessons that would help a future session, written one sentence each.

**Nothing is captured without your consent.** Capture ships enabled by
default but stays inert until you record a one-time consent:

```bash
poppy consent --enable     # turn automatic capture on
poppy consent --disable    # opt out (persists across upgrades)
poppy consent --status     # show the current consent and capture state
```

`poppy setup claude-code` asks once on a TTY (pass `--yes` for
non-interactive installs). Until you decide, every session start shows a
short notice and nothing runs. Opting out persists permanently; a later
default change can never silently re-enable capture you turned off.

What fires when, once consent is granted:

- **Every 3rd prompt**, a detached background worker reads only the
  transcript turns added since the last capture and extracts memories
  from that window. It never blocks your prompt, captures no turn twice,
  and stops at a per-session soft cap.
- **At session end**, a backstop pass flushes whatever the mid-session
  loop has not captured yet.
- **After context compaction**, the compact summary is consolidated the
  same way.

Extraction runs locally through the same coding-agent CLI you already use
(`claude`, `codex`, or `gemini`), on your existing login. Your
conversation never leaves your machine, and Poppy never auto-spends on a
paid remote model: if only an API-key backend is configured, capture
stays off and tells you why.

Before anything is written, a reconciler dedups each candidate against
the store: clear duplicates are skipped, clear updates supersede the old
memory (tombstoned and restorable for 7 days), and everything uncertain
is added rather than merged.

What is stored, all locally under `~/.poppy/`:

- the extracted memories themselves, in `memories.db`, each tagged with
  the source app and session id;
- a capture journal (`capture_journal.jsonl`) recording what each capture
  stored, so the session banner and `poppy doctor` can show it;
- per-session progress state (`capture_state.json`). The journal and
  state files are per-device and are never synced anywhere, not even to
  Trags.

Every session start prints a one-line status banner (active with counts,
a loud INACTIVE when something is broken, or the consent nudge), and
`poppy doctor` reports the full capture state. `POPPY_CONSOLIDATE=1/0`
remains an explicit environment override in both directions.

## Engines

Retrieval runs on one of three interchangeable engines. Switch any time with
`poppy engines use <name>` (`--migrate` re-embeds your existing memories for
the new engine):

| Name | What it is |
|---|---|
| `bloom`  | Default. Hybrid (FTS5 + bge-small embeddings + RRF) into a cross-encoder rerank, with per-speaker content expansion. ~600 MB ML deps. |
| `sprout` | Mid tier. Same two-stage architecture, a lighter bi-encoder, no expansion. |
| `seed`   | FTS5 only. No ML deps, no model downloads. The universal floor. |

How the three engines compare on retrieval quality, latency, and footprint:
see [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md).

`bloom` is the **evolving default champion**: a future release can promote a
better local engine into `bloom`, [chosen by benchmark](benchmarks/BENCHMARKS.md)
rather than by hand, so your default retrieval improves over time. `sprout` and `seed` are **stable
anchors** that don't change under you. No engine's architecture ever changes
silently. Every change ships in a release with notes, and `poppy stats` always
shows the exact architecture you're running as `engine vX.Y.Z`.

Because a new champion can use a different embedding model, your stored vectors
may need refreshing: run `poppy migrate-engine` after switching engines, or if
recall quality dips after a `bloom` upgrade. If an upgrade ever changes behavior
you depend on, switch to `sprout` or `seed` for a setup that stays put.

## Commands

| Command | Flags | What it does |
|---|---|---|
| `poppy remember CONTENT` | `--type`, `--project`, `--ttl`, `--expires-at`, `--supersedes`, `--check-conflicts`, `--auto-supersede` | Store a memory. |
| `poppy recall QUERY` | `--project`, `--type`, `--since`, `--limit`, `--json`, `--include-expired` | Search memories, ranked by relevance to the query. |
| `poppy list` | `--project`, `--type`, `--since`, `--limit`, `--json`, `--include-expired` | List all memories, newest first. |
| `poppy edit MEMORY_ID` | `--content`, `--type`, `--project`, `--no-project`, `--ttl`, `--expires-at`, `--no-expiry` | Edit a memory in place. |
| `poppy forget MEMORY_ID` | `--yes` | Delete a memory by ID. |
| `poppy expire` | `--yes` | List memories whose TTL has passed; `--yes` purges them. |
| `poppy stats` | | Show memory stats. |
| `poppy engines` | | List the engine catalog with the active engine starred. |
| `poppy engines use NAME` | `--migrate` | Switch the active retrieval engine. |
| `poppy migrate-engine` | `--project`, `--memory-type`, `--since`, `--all`, `--dry-run` | Re-embed memories so their vectors match the active engine. |
| `poppy config set KEY VALUE` | | Set a config value. |
| `poppy consent` | `--enable`, `--disable`, `--status` | Manage consent for automatic capture. See [Zero-touch capture](#zero-touch-capture). |
| `poppy telemetry` | `status`, `on`, `off` | Show or change anonymous usage telemetry. See [Telemetry](#telemetry). |
| `poppy serve` | | Start the Poppy MCP server (stdio). |
| `poppy ui` | `--host`, `--port`, `--no-open` | Browse and manage memories in a local web UI. |
| `poppy setup claude-code` | `--hooks/--no-hooks`, `--claude-md/--no-claude-md`, `--yes` | Install Poppy into Claude Code (MCP + hooks + CLAUDE.md primer). |
| `poppy setup claude-desktop` | `--print-instructions`, `--print-import-prompt` | Register the Poppy MCP server in the Claude desktop app. |
| `poppy setup cursor` | | Install Poppy into Cursor (MCP only). |
| `poppy setup windsurf` | | Install Poppy into Windsurf (MCP only). |
| `poppy setup codex` | | Install Poppy into Codex (MCP only). |
| `poppy setup copilot-cli` | | Install Poppy into GitHub Copilot CLI (MCP + AGENTS.md primer). |
| `poppy setup pi` | | Install Poppy into Pi (MCP via pi-mcp-adapter + AGENTS.md primer). |
| `poppy setup goose` | | Install Poppy into Goose (MCP extension + .goosehints primer). |
| `poppy setup hermes-agent` | | Install Poppy as a Hermes Agent memory provider plugin. |
| `poppy setup trags` | `--api-url` | One-command device-code onboarding for Trags cloud sync. |
| `poppy sync push` | `--dry-run` | Send local memories + tombstones to Trags. |
| `poppy sync pull` | `--dry-run` | Apply Trags rows newer than the last pull watermark. |
| `poppy sync run` | `--dry-run` | Pull then push, full bidirectional sync. |
| `poppy sync status` | | Show watermarks and last-sync time. |
| `poppy import claude-memories` | `--dry-run`, `--projects-dir` | Import auto-memory files from `~/.claude/projects/<slug>/memory/`. |
| `poppy import hermes-memories` | `--dry-run`, `--memories-dir` | Import paragraphs from `~/.hermes/memories/{MEMORY,USER}.md`. |
| `poppy build mcpb` | `--output-dir` | Build a Claude Desktop Extension bundle (.mcpb). |
| `poppy hook ...` | | Claude Code hook entrypoints (`session-start`, `user-prompt-submit`, `pre-tool-use`, `post-compact`, `session-end`, `stop`, `replay-compact`, `replay-session-end`). Invoked by the hooks that `poppy setup claude-code` installs. |
| `poppy doctor` | | Verify the installation: engine, storage, MCP config, hooks. |

Run `poppy COMMAND --help` for full flag descriptions.

## Storage

Memories live at `~/.poppy/memories.db` (set `POPPY_DIR` to override). The
schema is a single SQLite file with FTS5 plus an embeddings table; back it
up the same way you'd back up any SQLite DB.

## Sync to Trags (optional)

If you want your memories searchable across machines and inside the Trags
web app, run `poppy setup trags` to onboard. Sync is opt-in and additive;
your local store remains the source of truth.

## Browse memories

```bash
poppy ui    # opens a local web UI at http://127.0.0.1:7800
```

## Telemetry

Poppy sends a small number of anonymous usage events to PostHog (EU region)
so we can see which features get used. The distinct ID is a random UUID
generated on your machine and stored in `~/.poppy/analytics.json`; it is not
derived from hardware, accounts, or anything identifying.

This is the complete list of events and the properties they carry:

| Event | When | Properties |
|---|---|---|
| `cli_install` | once, before the first event from a machine | `version` (Poppy version), `python_version`, `platform` (`darwin`, `linux`, `win32`) |
| `memory_write` | `poppy remember` | `memory_type` (`fact`, `decision`, `preference`, `lesson`), `has_project` (true/false), `source` (`manual`) |
| `recall_call` | `poppy recall` | `query_length` (character count), `result_count`, `engine` (engine name) |
| `agent_setup` | `poppy setup <client>` | `agent` (client name, for example `claude-code`) |

Never sent: memory content, recall query text, project names, file paths,
config values, API keys, or anything else you type into Poppy. Properties are
counts, lengths, version strings, and fixed enum values only. If you onboard
to Trags with `poppy setup trags`, the random device UUID is shared with the
Trags server so it can link this machine's telemetry to your account.

The first time the CLI runs with telemetry on, it prints a one-line notice to
stderr. Turn telemetry off any time; the choice persists in
`~/.poppy/config.json`:

```bash
poppy telemetry off       # or: poppy telemetry status | on
```

Setting `POPPY_TELEMETRY_OFF=1` in the environment also turns telemetry off
and overrides everything else. When telemetry is off, Poppy makes no
telemetry network calls at all.

## Hacking on it

```bash
git clone https://github.com/trags-garden/poppy
cd poppy
uv sync
uv run pytest -v
uv run poppy --help
```

## License

GNU AGPL-3.0-or-later. See [LICENSE](LICENSE). Poppy's source is open, but the
project does not accept external contributions; see
[CONTRIBUTING](CONTRIBUTING.md). Bug reports are welcome via GitHub issues;
for security vulnerabilities use [SECURITY](SECURITY.md) instead.
