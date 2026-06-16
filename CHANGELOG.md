# Changelog

All notable changes to Poppy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-14

First public release of `poppy-memory`.

### Added

- Core memory CLI: `remember`, `recall`, `list`, `edit`, `forget`, `expire`,
  `stats`, with types (fact, decision, preference, lesson), projects, TTLs,
  supersede chains, and LLM conflict detection (`--check-conflicts`,
  `--auto-supersede`).
- Three retrieval engines with a floating default tier: `bloom` (hybrid FTS5 +
  bge-small embeddings + cross-encoder rerank), `sprout` (lighter bi-encoder),
  `seed` (FTS5 only, zero ML deps). Switch with `poppy engines use`, re-embed
  with `poppy migrate-engine`.
- MCP server (`poppy serve`) plus one-command setup for Claude Code (MCP,
  lifecycle hooks, CLAUDE.md primer), Claude Desktop, Cursor, Windsurf, Codex,
  Copilot CLI, Pi, Goose, and Hermes Agent. `poppy doctor` verifies the whole
  installation.
- Claude Desktop Extension bundle builder (`poppy build mcpb`); the build
  stamps the pyproject version into the bundle manifest.
- Optional Trags cloud sync: device-code onboarding (`poppy setup trags`),
  `poppy sync push|pull|run|status`, and auto-sync after local writes.
- Imports from existing stores: `poppy import claude-memories` and
  `poppy import hermes-memories`.
- Local web UI (`poppy ui`) for browsing, editing, and restoring memories.
- `--since` filter on `recall`/`list` accepting ISO dates or durations such as
  `7d` and `1w3d`.
- Telemetry disclosure and first-class opt-out: `poppy telemetry
  status|on|off` persisting to `~/.poppy/config.json`, a one-time first-run
  notice on stderr, and a README section listing every event and property
  sent. `POPPY_TELEMETRY_OFF=1` always wins.
- `poppy --version`, single-sourced from `pyproject.toml`.
- Zero-touch capture: consent-gated automatic memory extraction from coding
  sessions. Mid-session captures fire every Nth prompt under a single-flight
  lock with a per-session soft cap; a SessionEnd backstop and PostCompact
  re-extraction cover whatever the cadence missed. An incremental transcript
  window (watermark-based, compaction-safe) feeds extraction, and a
  dedup-on-capture reconciler decides ADD / SUPERSEDE / SKIP before anything
  is written. Extraction runs locally through the user's own host CLI
  (claude/codex/gemini); a remote-only backend is never auto-spent.
- `poppy consent --enable/--disable/--status` plus a one-time consent prompt
  in `poppy setup claude-code` (and a `--yes` flag): nothing is captured
  until consent is granted, and an opt-out persists.
- SessionStart status banner (active / inactive / consent pending) and
  expanded `poppy doctor` capture reporting: granular consent and backend
  status, last-capture freshness, journal count, and watermark/lock state.
- Source-app provenance on MCP writes: memories record the real client app
  (for example `claude-code` or `cursor`) from `poppy serve --source` or the
  connecting client's `clientInfo`, never the bare transport string `mcp`.
- Legacy engine names in an existing `config.json` keep working:
  `speaker_closet` and `best` map silently to `bloom`, `baseline` to `seed`.
- `since` filter on the MCP `recall` tool, accepting the same values as the
  CLI `--since` (ISO date or a duration such as `7d`).
- Offline-safe retrieval model loading: already-downloaded models load with
  `local_files_only`, so recall keeps working with the network down; a cold
  cache with no network exits with a one-line actionable error instead of a
  HuggingFace traceback; a one-time stderr notice announces the first-run
  model download.

### Fixed

- `--since` on `recall`/`list` was accepted but silently ignored; it now
  filters and rejects invalid values with a usage error.
- `recall`/`list` no longer print the misleading "No memories found." when
  memories exist but active `--since`/`--project`/`--type` filters excluded
  them; they say the filters matched nothing instead.
- Recall telemetry reported a stale hardcoded engine name instead of the
  engine that actually served the query.
- The `memory_write` telemetry event sent the raw project name; it now sends
  only a `has_project` boolean.

### Security

- License set to AGPL-3.0-or-later.
- `~/.poppy/config.json` (holds API keys) is written 0600 inside a 0700
  directory, atomically, so secrets never land world-readable.
- The `poppy setup trags` device-code flow encrypts the returned API key to an
  ephemeral RSA keypair, so the key never crosses the wire in plaintext.
- Runtime SQLite connections enable WAL and a busy timeout, preventing
  lock-related corruption under concurrent hook/CLI access.
