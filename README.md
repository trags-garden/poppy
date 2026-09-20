<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://trags.ai/images/poppy/readme-header-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://trags.ai/images/poppy/readme-header-light.svg">
  <img src="https://trags.ai/images/poppy/readme-header-light.svg" alt="poppy: remember what matters" width="640">
</picture>

Poppy is a developer memory CLI and MCP server. Your AI coding agent stores
decisions, preferences, and lessons in a local SQLite store, then recalls
them when they're relevant. No cloud, no API keys, no per-query cost.

```bash
curl -fsSL https://trags.ai/install/poppy | sh
```

Then wire up your agent:

```bash
poppy setup claude-code   # or: claude-desktop, cursor, vscode, windsurf, codex,
                          #     gemini, copilot-cli, pi, goose, hermes-agent
```

## What it does

- **`remember`** stores a fact, decision, preference, or lesson.
- **`recall`** pulls the few memories most relevant to a query.
- **MCP server**: agents call `remember`/`recall`/`forget`/`consolidate`
  directly. Works with Claude Code, Claude Desktop, Cursor, VS Code, Windsurf,
  Codex, Gemini CLI, Copilot CLI, Pi, and Goose. Hermes Agent gets a memory
  provider plugin instead.
- **Hooks**: Claude Code, Cursor, and Codex hooks surface relevant memories and
  capture new ones. Claude Code and Cursor use end-of-session and compaction
  backstops; Codex uses cadence-only capture.
- **Zero-touch capture**: with your one-time consent, Poppy extracts
  durable memories from your sessions automatically, locally, using the
  coding-agent CLI you already have. See
  [Zero-touch capture](#zero-touch-capture).
- **Interchangeable engines**: retrieval runs on a swappable local engine;
  the default, `bloom`, is a hybrid two-stage engine, with a zero-dependency
  full-text fallback (`seed`). See [Engines](#engines).

```bash
poppy remember "we always use uv for python deps" --type preference
poppy recall "python package manager"
#   we always use uv for python deps
#     preference | 2026-05-25 | score: 0.91
```

## Supported clients

`poppy setup <client>` wires Poppy into a coding agent. Eleven clients have an
integration today:

- **MCP server plus zero-touch capture hooks**: Claude Code, Cursor, Codex.
- **MCP server**: Claude Desktop, VS Code, Windsurf, Gemini CLI, GitHub Copilot
  CLI, Pi, Goose. Most also get a short primer file so the agent knows the
  memory tools are there.
- **Memory provider plugin** that calls the Poppy CLI: Hermes Agent.

What supported means here: each integration is verified against its client at
release time, and maintained best-effort between releases. These clients change
on their own schedule, so when one of them breaks Poppy, that is a bug worth
reporting as a
[GitHub issue](https://github.com/trags-garden/poppy/issues/new/choose) rather
than something that holds the next release back.

`poppy doctor` checks the integrations installed on your machine.

## Zero-touch capture

With the Claude Code, Cursor, or Codex hooks installed, Poppy can also extract durable
memories from your sessions automatically: decisions, preferences, and
lessons that would help a future session, written one sentence each.

**Nothing is captured without your consent.** Capture ships enabled by
default but stays inert until you record a one-time consent. If you already
had `consolidate-enabled true` set from an earlier version, that setting
counts as your recorded consent and capture stays active with no new prompt.
Setting `POPPY_CONSOLIDATE=1` overrides the consent check entirely and turns
capture on regardless. The unified command reports the resolved effect plus
the project and global layers:

```bash
poppy autocapture                 # show resolved, project, and global status
poppy autocapture on --global     # turn automatic capture on everywhere
poppy autocapture off --global    # opt out everywhere (persists across upgrades)
```

`poppy setup claude-code` asks once on a TTY (pass `--yes` for
non-interactive installs). Until you decide, every session start shows a
short notice and nothing runs. Opting out persists permanently; a later
default change can never silently re-enable capture you turned off.

What fires when, once automatic capture is enabled globally:

- **Every 3rd prompt**, a detached background worker reads only the
  transcript turns added since the last capture and extracts memories
  from that window. It never blocks your prompt, captures no turn twice,
  and stops at a per-session soft cap in Claude Code and Cursor. Codex has no
  soft cap because it has no end-of-session flush.
- **At session end in Claude Code and Cursor**, a backstop pass flushes whatever
  the mid-session loop has not captured yet. Codex `Stop` fires after each turn,
  so Poppy uses it only as a liveness signal and never as a flush.
- **Around context compaction**, Claude Code consolidates the compact summary;
  Cursor flushes the live transcript at `preCompact` before compaction begins.

Extraction runs locally through the same coding-agent CLI you already use
(`claude`, `cursor-agent`, `codex`, or `gemini`), on your existing login. Your
conversation never leaves your machine, and Poppy never auto-spends on a
paid remote model: if only an API-key backend is configured, capture
stays off and tells you why.

Before anything is written, a reconciler dedups each candidate against
the store: clear duplicates are skipped, clear updates supersede the old
memory (tombstoned and restorable for 7 days), and everything uncertain
is added rather than merged.

### Turning capture off for one project

Consent is global and one-time, but you can keep a single sensitive repo
(employer or client code) out of memory without turning capture off
everywhere:

```bash
poppy autocapture off        # in the repo: choose this project or everywhere
poppy autocapture on         # capture this project again
poppy autocapture status     # show the resolved, project, and global state
poppy autocapture off --project NAME   # target a project by name instead
```

The project is auto-detected from the current directory (a repo with a
`.git`, `CLAUDE.md`, `pyproject.toml`, `package.json`, ...); pass
`--project` if you're outside it. Disabled projects are stored as a
deny-list in `~/.poppy/config.json` (`disabled_projects`), and the switch
sits above consent: a disabled project stays out of capture no matter
what the global consent says. Only automatic capture stops. Recall keeps
working in that project, and `poppy remember` still writes there, so you
stay in control of what gets stored. Session start shows a one-line
reminder that capture is off for the project.

### Secret redaction

An extraction model can copy a credential out of your transcript into a
memory ("the staging DB is `postgres://admin:hunter2@db`"). Before a
captured candidate becomes a memory, Poppy masks secret-shaped tokens in
place with `[REDACTED]`. This runs at the earliest seam shared by all
three capture paths (mid-session, session end, post-compact), so the raw
value never reaches the store, the capture journal, or Trags sync.

Masked today: private-key blocks (`-----BEGIN ... PRIVATE KEY-----`),
passwords in connection-string URLs, `Bearer` tokens, AWS access key ids
(`AKIA...` / `ASIA...`), OpenAI-style `sk-` keys, and GitHub tokens
(`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`). The memory itself is always
kept, with only the secret masked, so a lesson does not get thrown away
over one token.

#### Customizing redaction

Add exact values or environment-variable names when your project uses secrets
that do not match the built-in shapes:

```bash
poppy redaction add "company-specific-secret"
poppy redaction add --env INTERNAL_API_TOKEN
poppy redaction list
poppy redaction remove "company-specific-secret"
poppy redaction remove --env INTERNAL_API_TOKEN
```

Literal entries are case-sensitive and match anywhere in captured content.
Environment-variable entries look up the variable's current value when Poppy
redacts a capture; `poppy redaction list` shows only the variable name, never
its value. Configured literals are secrets themselves, so `list` hides their
values too unless you pass `--show`. Literals and resolved environment values
must be at least four characters long. The six built-in pattern families always stay on, and custom
matches are masked in place with the same `[REDACTED]` marker.

This is a deliberately narrow deny-list of high-signal shapes, not a
general secret scanner: it will not catch every credential your session
touches, so treat it as a safety net rather than a guarantee. If a repo
is sensitive enough that this matters, turn capture off for that project
(above). Redaction applies to automatic capture only; a memory you write
yourself with `poppy remember` is stored exactly as you typed it.

What is stored, all locally under `~/.poppy/`:

- the extracted memories themselves, in `memories.db`, each tagged with
  the source app and session id;
- a capture journal (`capture_journal.jsonl`) recording what each capture
  stored, so the session banner and `poppy doctor` can show it;
- per-session progress state (`capture_state.json`). The journal and
  state files are per-device and are never synced anywhere, not even to
  Trags.

Every session start prints a one-line status banner (active with counts,
a loud INACTIVE when something is broken, or an `autocapture` consent nudge), and
`poppy doctor` reports the full capture state. `POPPY_CONSOLIDATE=1/0`
remains an explicit environment override in both directions; setting it to
`1` bypasses the consent check entirely.

## Engines

Retrieval runs on one of two engines. Switch any time with
`poppy engines use <name>` (`--migrate` re-embeds your existing memories for
the new engine):

| Name | What it is |
|---|---|
| `bloom` | Default. Hybrid (FTS5 + bge-small embeddings + RRF) into a cross-encoder rerank, with per-speaker content expansion. Runs on local ONNX models, so a plain `pip install poppy-memory` has everything it needs. |
| `seed`  | FTS5 only. No ML deps, no model downloads. The floor, and the automatic fallback when `bloom` cannot start. |

`bloom` is where retrieval work lands, so its architecture can change between
releases; `seed` is a stable anchor that doesn't change under you. No engine
changes silently: every change ships in a release with notes, and `poppy stats`
shows the exact architecture you're running as `engine vX.Y.Z`.

Because a release can change the embedding model, your stored vectors may need
refreshing: run `poppy migrate-engine` after switching engines, or if recall
quality dips after a `bloom` upgrade. Memories whose vectors don't match the
active engine stay searchable by keyword in the meantime. Nothing is lost, and
`poppy doctor` tells you how many are waiting.

## Commands

| Command | Flags | What it does |
|---|---|---|
| `poppy remember CONTENT` | `--type`, `--project`, `--ttl`, `--expires-at`, `--supersedes`, `--check-conflicts`, `--auto-supersede` | Store a memory. |
| `poppy recall QUERY` | `--project`, `--type`, `--since`, `--limit`, `--json`, `--include-expired` | Search memories, ranked by relevance to the query. |
| `poppy list` | `--project`, `--type`, `--since`, `--limit`, `--json`, `--include-expired` | List all memories, newest first. |
| `poppy edit MEMORY_ID` | `--content`, `--type`, `--project`, `--no-project`, `--ttl`, `--expires-at`, `--no-expiry` | Edit a memory in place. |
| `poppy forget MEMORY_ID` | `--yes` | Soft delete a memory by ID, restorable for 7 days. |
| `poppy expire` | `--yes` | List memories whose TTL has passed; `--yes` purges them. |
| `poppy stats` | | Show memory stats. |
| `poppy engines` | | List the engine catalog with the active engine starred. |
| `poppy engines use NAME` | `--migrate` | Switch the active retrieval engine. |
| `poppy migrate-engine` | `--project`, `--memory-type`, `--since`, `--all`, `--dry-run` | Re-embed memories so their vectors match the active engine. |
| `poppy config set KEY VALUE` | | Set a config value. |
| `poppy autocapture` | `status`, `on`, `off`, `--global`, `--project`, `--yes` | Show or change automatic capture at project or global scope. See [Zero-touch capture](#zero-touch-capture). |
| `poppy consent` | `--enable`, `--disable`, `--status` | Legacy visible consent command; still works while callers migrate to `poppy autocapture`. |
| `poppy capture` | `--off`, `--on`, `--status`, `--project` | Deprecated hidden alias for the per-project controls; still works while callers migrate. |
| `poppy telemetry` | `status`, `on`, `off` | Show or change anonymous usage telemetry. See [Telemetry](#telemetry). |
| `poppy encrypt` | `status`, `enable`, `disable`, `repair` | Turn on encryption at rest for the local store. See [Encryption at rest](#encryption-at-rest-optional). |
| `poppy serve` | | Start the Poppy MCP server (stdio). |
| `poppy ui` | `--host`, `--port`, `--no-open` | Browse and manage memories in a local web UI. |
| `poppy setup claude-code` | `--hooks/--no-hooks`, `--claude-md/--no-claude-md`, `--yes`, `--daemon` | Install Poppy into Claude Code (MCP + hooks + CLAUDE.md primer). |
| `poppy setup claude-desktop` | `--print-instructions`, `--print-import-prompt` | Register the Poppy MCP server in the Claude desktop app. |
| `poppy setup cursor` | `--hooks/--no-hooks`, `--daemon` | Install Poppy into Cursor (MCP + zero-touch capture hooks). Honors `$CURSOR_HOME`. |
| `poppy setup vscode` | `--daemon` | Install Poppy into VS Code (MCP only). |
| `poppy setup windsurf` | | Install Poppy into Windsurf (MCP only). |
| `poppy setup codex` | `--hooks/--no-hooks`, `--daemon` | Install Poppy into Codex (MCP + cadence capture hooks + AGENTS.md primer). Honors `$CODEX_HOME`. |
| `poppy setup gemini` | `--daemon` | Install Poppy into Gemini CLI (MCP only). |
| `poppy setup copilot-cli` | | Install Poppy into GitHub Copilot CLI (MCP via `$COPILOT_HOME/mcp-config.json` + `$COPILOT_HOME/copilot-instructions.md` primer; defaults to `~/.copilot`). |
| `poppy setup pi` | | Install Poppy into Pi (MCP via pi-mcp-adapter + AGENTS.md primer in `$PI_CODING_AGENT_DIR`; defaults to `~/.pi/agent`). |
| `poppy setup goose` | | Install Poppy into Goose (MCP extension + .goosehints primer). |
| `poppy setup hermes-agent` | | Install Poppy as a Hermes Agent memory provider plugin with SOUL.md guidance. |
| `poppy setup trags` | `--api-url` | One-command device-code onboarding for Trags cloud sync. |
| `poppy sync push` | `--dry-run` | Send local memories + tombstones to Trags. |
| `poppy sync pull` | `--dry-run` | Apply Trags rows newer than the last pull watermark. |
| `poppy sync run` | `--dry-run` | Pull then push, full bidirectional sync. |
| `poppy sync status` | | Show watermarks and last-sync time. |
| `poppy import claude-memories` | `--dry-run`, `--projects-dir` | Import auto-memory files from `~/.claude/projects/<slug>/memory/`. |
| `poppy import hermes-memories` | `--dry-run`, `--memories-dir` | Import paragraphs from `~/.hermes/memories/{MEMORY,USER}.md`. |
| `poppy hook ...` | | Claude Code, Cursor, and Codex hook entrypoints (`session-start`, `user-prompt-submit`, `pre-tool-use`, `post-compact`, `session-end`, `stop`, `replay-compact`, `replay-session-end`). Invoked by Poppy-installed hooks. |
| `poppy doctor` | | Verify the installation: engine, storage, MCP config, hooks. |

Cursor hooks are installed globally at `~/.cursor/hooks.json` by
`poppy setup cursor`. Capture starts immediately once automatic capture is
enabled; Cursor has no hook trust step. Hook execution also depends on Cursor's
server-side `enable_execute_hook_exec` account flag. If hooks are installed but
capture stays silent, `poppy doctor` reports that flag as a possible cause.
Cursor resumes append to the same transcript and keep the same session id. A
pure Q&A resume that runs no tool can emit no hooks, including `sessionEnd`; a
future idle-session sweep will cover that accepted tail-loss edge case. Cursor
transcripts omit tool outputs, so automatic capture sees the user and assistant
text but cannot use command or tool results as evidence. In headless
`cursor-agent -p` runs, `beforeSubmitPrompt` never fires, so capture is
backstop-only: there is no every-third-prompt cadence or prompt-time behavior,
but `sessionEnd` still flushes the remaining transcript.

Codex capture has a one-time trust gate. After `poppy setup codex`, run Codex
interactively in the TUI once and approve the Poppy hooks. Headless `codex exec`
silently skips untrusted hooks. Hook trust is tied to the command hash, so editing
a hook command later requires approval again. Poppy never adds a trust-bypass
flag to Codex commands.

Run `poppy COMMAND --help` for full flag descriptions.

## Storage

Memories live at `~/.poppy/memories.db` (set `POPPY_DIR` to override). The
schema is a single SQLite file with FTS5 plus an embeddings table; back it
up the same way you'd back up any SQLite DB.

Set `POPPY_MODEL_IDLE_S` to change the 600-second model idle-unload timeout; zero or an invalid value disables unloading.

## Encryption at rest (optional)

By default the local store is a plaintext SQLite file, protected only by file
permissions (`~/.poppy` is `0700`). You can encrypt it at rest with a key held
in your operating system's keychain:

```bash
# The installer (curl -fsSL https://trags.ai/install/poppy | sh) uses pipx, so inject the extra dep there:
pipx inject poppy-memory sqlcipher3
# (plain pip installs instead: pip install 'poppy-memory[encryption]')
poppy encrypt enable                      # migrates the existing store in place
poppy encrypt status                      # show state
poppy encrypt disable                     # decrypt back to plaintext
poppy encrypt repair                      # reconcile an inconsistent state, if ever needed
```

`enable` generates a random 256-bit key, stores it in the OS keychain (macOS
Keychain, Linux Secret Service, or Windows Credential Locker), and rewrites the
store as a [SQLCipher](https://www.zetetic.net/sqlcipher/) database. Every page
is encrypted, including the FTS5 search index and the embedding vectors, and the
`-wal` sidecar's frame contents are encrypted too (the `-shm` sidecar is a
plaintext wal-index that holds no database content). Search and recall keep
working locally with no change to the CLI, MCP server, or web UI. Existing
plaintext stores are migrated the first time you run `enable`; new writes go
straight to the encrypted store. Stop `poppy serve`, `poppy ui`, and any capture
or sync workers before migrating; `enable`/`disable` refuse if they detect a
live writer rather than risk losing a write.

What this protects against: theft of the disk or of an unencrypted backup, and
casual inspection of the store file. What it does not do: this is local
encryption at rest, not zero-knowledge or end-to-end encryption. The key lives
in your keychain, so anything running as your user account that can read the
keychain can decrypt the store. Cloud sync is unaffected and is a separate
concern (memories synced to Trags are covered by that service's own encryption).

If the keychain entry is lost, the store cannot be recovered, so keep a backup
strategy that accounts for the key. On a headless host with no keychain, set the
`POPPY_DB_KEY` environment variable to a 64-hex-character key instead; note that
a key in an environment variable is weaker than one in the keychain, because it
is visible to anything that can read the process environment.

## Sync to Trags (optional)

If you want your memories searchable across machines and inside the Trags
web app, run `poppy setup trags` to onboard. Sync is opt-in and additive;
your local store remains the source of truth.

Forgetting a memory is a soft delete: the record sits in Trash for 7 days and
`poppy restore` brings it back. Deletions travel only for memories the remote is
known to hold, from successful uploads, live pulls, or a one-time upgrade
backfill treating every existing memory and tombstone as known to each remote
in sync state. This preserves the previous behavior of sending every deletion.
Unreadable sync state stops the migration until repaired; absent state means
no remotes and an empty backfill. Knowledge is
tracked per memory and remote and checked at push time; unrelated uploads never
unlock a deletion. Sent marks keep pulled deletions from being re-announced and
retry failed sends without changing the live-upload watermark. Deletions carry
the full memory snapshot, including local edits, for Trash restore elsewhere.
A pull that sees a memory heals a lost upload response on the next push.
Residuals: an upload whose response was lost stays unknown until a pull sees
the row (incremental pull cannot see it below the pull cursor). A deletion
carries the local snapshot, including edits never pushed. Pre-upgrade memories
are treated as known, so the guarantee applies in full only to memories created
after upgrade or in stores that had no remote at upgrade.

The Trags API key goes into your operating system's keychain only when Poppy can
write it and read it back in the same process; otherwise it stays in plaintext
in `config.json` (which stays `0600`). Keys are resolved in this order: the
`POPPY_TRAGS_API_KEY` environment variable, then a plaintext key in `config.json`,
then the OS keychain. An existing plaintext key is moved into the keychain
automatically and scrubbed from the file the first time a Poppy process can
write it and read it back from the keychain. On macOS, processes started outside
a login session (launchd Background agents, cron, editor background jobs) can
often write the login keychain but not read it, so set `POPPY_TRAGS_API_KEY`
for those processes; captures they make are still pushed by the next sync a
login-session process triggers, such as the daemon. This is
local secret hygiene, not zero-knowledge: anything running as your user that
can read the keychain can read the key.

## Browse memories

```bash
poppy ui    # opens a local web UI at http://127.0.0.1:7800
```

## Telemetry

Telemetry is off until you turn it on. The first time you run Poppy in a
terminal it asks once whether to send anonymous usage events; the default is
no, so pressing Enter declines. Until you answer yes, Poppy sends nothing.

Saying yes sends a small number of events to PostHog (EU region) so we can see
which features get used. The distinct ID is a random UUID generated on your
machine then and stored in `~/.poppy/analytics.json`; it is not derived from
hardware, accounts, or anything identifying. Decline and no ID is generated at
all.

This is the complete list of events and the properties they carry. Every event
Poppy can emit is in this table; there are no others.

| Event | When | Properties |
|---|---|---|
| `cli_install` | once, before the first event from a machine | `version` (Poppy version), `python_version`, `platform` (`darwin`, `linux`, `win32`) |
| `memory_write` | every stored memory (`poppy remember`, the MCP `remember` tool, and automatic capture) | `memory_type` (`fact`, `decision`, `preference`, `lesson`, `summary`, `context`), `has_project` (true/false, never the project name), `source` (source app: `manual`, `claude-code`, `cursor`, ...) |
| `recall_call` | `poppy recall` | `query_length` (character count, never the query text), `result_count`, `engine` (engine name) |
| `agent_setup` | every `poppy setup <client>` | `agent` (client name, for example `claude-code`) |
| `setup_completed` | once per machine, on the first `poppy setup <client>` | `agent` (client name) |
| `consent_granted` | once per machine, when you turn automatic capture on | `via` (`consent_cmd`, `setup_yes`, or `setup_prompt`) |
| `consent_pending_shown` | once per machine, when the consent nudge is first shown | `channel` (`session_start` or `setup`) |
| `first_autocapture_stored` | once per machine, the first time automatic capture stores a memory | `trigger` (`mid_session`, `post_compact`, or `session_end`) |
| `closed_loop` | once per machine, the first time a memory captured in an earlier session is recalled back into a session | `channel` (`session_start`, `recall_prompt`, or `recall_file`), `injected_count`, `loop_count` (counts only) |

The last five are one-time funnel milestones: each fires at most once per
machine, ever, and then never again. The first four can fire repeatedly.

Never sent: memory content, recall query text, project names, file paths,
session ids, config values, API keys, or anything else you type into Poppy.
Properties are counts, lengths, version strings, and fixed enum values only.
If you onboard to Trags with `poppy setup trags`, the random device UUID is
shared with the Trags server so it can link this machine's telemetry to your
account.

The question is asked on stderr, and only when someone can actually answer
it: stdin, stdout and stderr all have to be a terminal, and Poppy must see no
sign of a coding agent or a CI runner driving that terminal. Hooks, the MCP
server (`poppy serve`), daemon commands, detached workers, `--json` output,
`--help` and any piped or redirected run are never interrupted by it, and
nothing is sent from any of them while the question is unanswered.

Your answer persists in `~/.poppy/config.json`, and either answer counts, so
you are asked once. Change your mind at any time, or answer without waiting
to be asked:

```bash
poppy telemetry status    # on, off, or not answered yet
poppy telemetry on
poppy telemetry off
```

Setting `POPPY_TELEMETRY_OFF=1` in the environment also turns telemetry off
and overrides everything else. When telemetry is off, Poppy makes no
telemetry network calls at all.

### Update check

`poppy doctor` may send one anonymous HTTP request to PyPI at most once every
24 hours to check the latest `poppy-memory` version. The request contains only
the installed version in its User-Agent and no identifiers. Normal commands
never make this request; they can only show a once-per-version hint already
stored in the local cache. Disable the check with `POPPY_UPDATE_CHECK_OFF=1` or
`poppy config set update_check off`. Turning telemetry off, with
`poppy telemetry off` or `POPPY_TELEMETRY_OFF=1`, also turns it off.

So does leaving the telemetry question unanswered. A fresh install makes no
network request of any kind, to PyPI or to PostHog, until you have said yes.

## Hacking on it

The full source ships in every release on PyPI:

```bash
pip download poppy-memory --no-binary :all: --no-deps -d .
tar xzf poppy_memory-*.tar.gz && cd poppy_memory-*/
uv sync
uv run pytest -v
uv run poppy --help
```

## License

Apache-2.0; the `LICENSE` file ships with the package. Poppy's source is
open, and you are free to read, run, self-host, fork, and modify it under
those terms. Poppy is maintained by the Trags team. We are not accepting pull
requests right now. Bug reports and feature requests are welcome as
[GitHub issues](https://github.com/trags-garden/poppy/issues/new/choose), or by
email at feedback@trags.ai. For security vulnerabilities, email
security@trags.ai instead of filing publicly.
