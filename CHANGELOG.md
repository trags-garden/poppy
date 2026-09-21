# Changelog

All notable changes to Poppy are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-09-13

### Changed

- `poppy doctor` no longer reports per-speaker copy counts or the retired
  migration diagnostics.
- `poppy sync push|pull|run --dry-run` no longer refuses to run on a store that
  has not been upgraded yet. Opening a store always applies its pending one-time
  upgrades, as every other command already did, so a dry run now previews the
  sync instead of stopping. It still sends nothing and receives nothing.
- Anonymous usage telemetry is off until you turn it on. Poppy asks once, on
  the first run in a terminal, and the default is no; `poppy telemetry on|off`
  answers it too. Nothing is sent, and the daily PyPI update check does not
  run either, while the question is unanswered, so a fresh install makes no
  network request until you say yes. The question is only asked when someone
  can answer it: hooks, the MCP server, daemon commands, detached workers,
  `--json` output, `--help`, any piped or redirected run, and any terminal
  driven by a coding agent or a CI runner are never interrupted by it.
  `poppy telemetry status` reports "not answered yet" separately from a choice
  you made.
- Removed experimental local ONNX model-file overrides. Bloom always uses its
  pinned models, including cached models when offline.
- The local web UI now runs the same design system as the trags.ai dashboard
  (the anthotype: one chlorophyll pigment on a warm cream sheet). Palette,
  type scale, topbar, facet rail, list rows, cards, buttons and empty states
  are the hosted dashboard's, and the small-type tier lifts to an 11px floor
  on a phone. Poppy keeps its own name and mark in the masthead.
- The facet rail is reachable on a phone: a Filters sheet opens it, and
  picking a memory opens the detail over the list with a way back.
- The chrome link now points at trags.ai instead of the source repository.
- A conversation transcript is now stored as one memory. Earlier versions also
  derived a separate hidden copy of each speaker's turns; nothing derives those
  any more.
- Opening a store with this version removes the per speaker copies that an
  earlier version derived from memories whose text was a conversation
  transcript, along with their search entries. The memory each copy came from
  is left untouched, nothing you wrote yourself is removed, and none of the
  copies are uploaded on the way out. A copy that an earlier version had
  already uploaded stays out of this store and never comes back to it, and it
  never appears in Trash as something to restore. A row the cleanup cannot read
  is left alone and stays visible, and the store still opens.

### Fixed

- Removed em dashes from user-facing CLI, dashboard, setup prompt, and MCP
  copy. `poppy doctor` status lines now read `label: OK, detail` where they
  used a dash.
- Exact-id reads and edits now treat hidden copies retained from earlier
  releases as missing, including MCP recall, CLI confirmations, and dashboard
  details and Trash. Internal cleanup still removes them without exposing text.
- A Trash entry holding a retained per-speaker copy's text no longer becomes
  readable, restorable and syncable the moment that copy is deleted. It is
  removed along with the copy, on all three paths that remove one: deleting the
  copy by id, redacting the memory it came from, and the one-time cleanup on
  first open. Its text stays recoverable for the usual Trash window unless
  something recoverable is already held for that id, in which case the earlier
  one is kept and the entry is removed without one. `poppy doctor` reports how
  many recoverable rows are held and until when.
- The OpenAI-compatible backend works on a default install. It called the
  endpoint through an SDK that Poppy does not depend on, so configuring a model
  and an API key produced empty consolidation and conflict results behind one
  generic log line. It now uses the HTTP client Poppy already ships, and names
  what actually went wrong (connection refused, authentication failed, rate
  limited, an unreadable response) in the worker log and in `poppy doctor`,
  never writing the credential to either. A host CLI and this fallback now
  share one time budget per call rather than taking one each, keeping a capture
  pass inside its lock. Short conflict checks still leave extraction-backend
  health unchanged. In `poppy doctor` a single failed call reads as
  information and only a run of them warns, matching how a host CLI is
  reported.
- Hardened that same path against a hostile or broken endpoint. Requests do not
  follow redirects, so the key is never replayed to a host the endpoint names.
  A model server on this machine bypasses any proxy set in the environment,
  which would otherwise be handed the key; a plaintext endpoint anywhere else
  warns that the key crosses the network in clear text. An answer that quotes
  the configured key back is discarded whole, rather than edited, so neither
  the credential nor a rewritten version of your own words can be stored. The
  response is read under a wall-clock deadline and an 8MB cap, and connecting
  gets a short cap of its own, so a server that stalls, drips bytes or never
  stops cannot outlast the capture lock or fill memory. A malformed endpoint
  URL, and model output with a number where text belongs, are now reported
  instead of ending the capture pass with a traceback.
- Conflict detection now reaches a backend and keeps the answer it gets back, so
  `remember --check-conflicts` reports real candidates and auto-supersede
  replaces a contradicted memory instead of quietly storing both. It picks a
  coding-agent CLI from PATH when there is no session transcript to name one,
  and no longer discards every verdict while parsing the response. A verdict
  gets a short timeout of its own, so a slow or wedged CLI cannot hold up a
  background capture pass, and a verdict that runs out of time is logged without
  counting against the backend, so a slow CLI cannot make a working install
  report itself as inactive.
- A client config that receives the loopback daemon's token now ends up readable
  only by its owner (mode 0600), even when it was group- or world-readable
  before, so no other user on the machine can read the token and talk to the
  memory daemon. Symlinked dotfiles configs and rotated backups are covered too.
  Setup also narrows backups left behind by earlier versions, which copied the
  config's own permissions and so can still hold a working token in a readable
  file. Setup reports each change it makes, and refuses to write the token at
  all to a file whose permissions it cannot narrow. Configs that carry no token
  keep the permissions you gave them.
- `poppy setup` now refuses unparseable or structurally invalid client configs
  before changing files or starting the daemon, preserving Claude Code settings
  instead of silently replacing user permissions, environment, model and hooks
  with Poppy's hooks.
  Cursor's existing hooks backup and replacement behavior is preserved for
  malformed JSON; non-UTF-8 configs are refused without changes.
- `poppy telemetry off` now also disables the daily PyPI update check in
  `poppy doctor`, honoring the persistent telemetry opt-out.
- A hand-edited `"telemetry_enabled": "false"` or `"off"` in `config.json` no
  longer reads as an opt-in. Only a real JSON boolean counts as an answer now;
  anything else leaves the question unanswered, which is off.
- Trags API keys stay in config.json as a fallback unless an OS keychain write
  can be read back in the same session, preventing background jobs from losing
  access to their configured key. `poppy doctor` now reports the key source, and
  both doctor and `poppy sync status` explain unreadable keychains and how to
  configure access for background jobs. A plaintext key in config.json now takes
  priority over the keychain because it is only there when the keychain could
  not be verified.
- `poppy encrypt enable` now reads the new encryption key back from the OS
  keychain before it touches the store, and refuses with a clear message when
  the key cannot be verified. A session that is allowed to write to the
  keychain but not read from it (a background launchd or cron job) can no
  longer create an encrypted store whose key it cannot resolve.
- `poppy encrypt` now checks that the OS keychain really works in this session
  by writing and reading a throwaway entry, so a locked login keychain (ssh,
  launchd, background jobs) makes `encrypt enable` stop with one line pointing
  at `POPPY_DB_KEY` instead of a raw keychain error code. `encrypt status` no
  longer calls such a keychain available: it reports it as present but unusable
  in this session, which is a different thing from no keychain at all. The check
  is skipped when `POPPY_DB_KEY` is set, so `encrypt status` and `poppy doctor`
  still reach no credential store at all on a headless or CI machine.
  `encrypt disable` with no key now prints the same "no usable key was found"
  line as recall and remember, and names the keychain error when there was one,
  so a merely locked keychain is not mistaken for a store encrypted on another
  machine. The wrong-key error states the store path and the reason only once.
- `poppy doctor` now warns when the Trags API key sits in plaintext in
  config.json, names the reason the keychain was not used and reports the
  file's real permissions, pointing at `chmod 600` when it is readable beyond
  the owner. The warning fires even with `POPPY_TRAGS_API_KEY` set, because the
  environment decides which key is used, not whether one sits on disk, so an
  upgrade that quietly fell back to the file no longer goes unnoticed.
- When the Trags key falls back to plaintext config.json, `poppy config set
  trags-api-key` now names the step that actually failed (no backend, a
  rejected write, or a write that could not be read back), `poppy doctor`
  explains the same fallback the same way, and both print the real config file
  path instead of a literal `~/.poppy/config.json`.
- `poppy config set trags-api-key ""` no longer reports a clean clear when the
  OS keychain refuses the delete. Poppy now checks whether the entry survived
  the delete and says the key is still active, naming the keychain item to
  remove.
- `poppy sync push|pull|run` now records a rejected API key the way the
  auto-sync worker does, so `poppy sync status` shows a `last error:` line
  naming the 401 instead of looking healthy. A push with nothing new to send
  also makes one cheap authenticated request, so a revoked key or a deleted
  account is reported rather than counted as a clean sync. A transient failure
  of that check (a 5xx, an offline laptop) is recorded on its own and cleared by
  the next healthy sync instead of wedging every later push, and a full
  `sync run` reuses the pull's request rather than making a second one.
- `poppy sync` no longer re-applies a deletion it already holds. A tombstone
  this device pushed comes back on the next pull, and re-recording it made every
  later `poppy sync run` report a tombstone applied on a store that had already
  converged. A deletion carrying something the store has not seen, such as a
  corrected body or a different supersede target, still applies, and `--dry-run`
  now reports the same outcome as the real run.
- Deletions travel only for memories the remote is known to hold. Per-ID,
  per-remote provenance is recorded on successful uploads and live pulls, with
  a one-time upgrade backfill treating every existing memory and tombstone as
  known to each remote in sync state, preserving the old deletion behavior.
  Unreadable sync state stops migration until repaired; absent state means no
  remotes and an empty backfill. Legacy deletions re-send once, without trusting
  old watermarks as acknowledgements. Unrelated uploads cannot
  unlock a deletion; sent marks retry failures without changing the live-upload
  watermark, and pulled deletions are not re-announced. Dry runs include IDs
  their pull would discover. Deletions carry the full snapshot, including local
  edits, for Trash restore elsewhere. Residuals: an upload whose response was
  lost stays unknown until a pull sees the row (incremental pull cannot see it
  below the pull cursor). A deletion carries the local snapshot, including
  edits never pushed. Pre-upgrade memories are treated as known, so the guarantee
  applies in full only to memories created after upgrade or in stores that had
  no remote at upgrade.
- A memory whose `updated_at` is in the future (a skewed clock, a bad import) no
  longer stops `poppy sync` from pushing anything else. The push watermark now
  stays on writes that have already happened, so later memories, edits and
  restores still reach the cloud. A watermark an earlier client already moved
  into the future cannot be recognised once the clock passes it, so the first
  push after this upgrade re-sends the whole store once per remote and recovers
  anything that was skipped. Every upload is an idempotent update the server's
  freshness gate settles, and later pushes are incremental again.
- A sync against an unreachable Trags host now prints one offline line and exits
  non-zero instead of dumping an httpx traceback. The line names the host and
  what that run had already done before it stopped answering, so a sync that
  applied pulled changes or sent part of its backlog is never reported as
  "nothing happened"; a lost response is reported as an unknown outcome rather
  than guessed either way. Watermarks stay frozen so every pending row retries,
  a dead host stops the push after three failed rows rather than working through
  the whole backlog, an offline `--dry-run` exits non-zero with the same
  explanation instead of a bare error counter, and the auto-sync worker still
  retries with backoff and keeps its pending flag when the pull is the only
  thing a cycle had to do.
- Auto-capture no longer sends a conversation to a configured remote model when
  the local host CLI already answered that there was nothing worth keeping. An
  empty array from the local extractor now ends the run, so the paid backend
  stays reserved for the case where the local CLI could not answer at all: no
  CLI on PATH, a crash or timeout, or output that is not a usable array.
- A host CLI that is installed but broken (for example a `claude` that is on
  PATH but logged out) no longer reports as healthy. After three failed
  extractions in a row from the same backend the session banner turns INACTIVE
  and names it, and `poppy doctor` warns on the `auto-capture` line with the
  last error from the CLI, with any credential in it masked. Each backend is
  counted separately, so two broken CLIs used in turn cannot hide each other,
  and the next successful capture through a backend clears its warning.
- Editing or re-ingesting a memory on the `bloom` engine no longer leaves the
  old text behind in the search index, where it stayed findable after the
  memory itself had been rewritten or redacted.
- The README now documents that an existing `consolidate-enabled true` setting
  from an earlier version counts as recorded consent, and that
  `POPPY_CONSOLIDATE=1` bypasses the consent check entirely.
- The 0.3.0 Upgrading section now says to upgrade every device on an account
  in the same sitting with auto-sync off, to back up `~/.poppy` first, and to
  never run an older Poppy against a store 0.3.0 has already opened.
- `poppy sync status` on an unconfigured store now offers `poppy setup trags`
  first, and the `poppy setup trags` success line prints the config file it
  actually wrote, which matters when `POPPY_DIR` moves the store.
- `poppy encrypt repair` now escapes the store path in the read-only plaintext
  check it runs before deleting a key. A `?` or `#` in the data directory could
  make a corrupt store pass that check (and leave a stray empty database next to
  it), and a `%` could make a healthy store fail it.
- `sync_state.json` and `analytics.json` are now written atomically (temp file,
  fsync, rename) and created owner-only. A crash, kill or full disk mid-save
  used to leave a truncated file that loaded as empty state, silently resetting
  every remote's sync watermarks and the telemetry device id.

### Upgrading

The first `poppy sync` after this upgrade may re-send every memory once per
configured remote while the push watermark recovers; the server keeps the
newest version of each memory, so nothing is overwritten and nothing is
duplicated. Devices that share a Trags account should be upgraded together,
as the 0.3.0 Upgrading note says.

## [0.3.0] - 2026-09-12

### Changed

- **Poppy now ships two engines: `bloom` and `seed`.** `bloom` is the hybrid
  retrieval engine: FTS5 + bge-small embeddings + RRF into a cross-encoder
  rerank, with per-speaker content expansion. `seed` is the FTS5-only floor
  and automatic fallback. Picking a retrieval engine used to mean choosing among
  four; there is now one ML engine and one floor.
- `bloom` runs on local ONNX models, so the default engine works on a plain
  `pip install poppy-memory` with no extra selected. The default engine's
  dependencies can no longer be missing on a fresh install; the first semantic
  query still performs a one-time model download.
- The engine formerly named `petal` **is** the new `bloom`: same architecture,
  same models, same `model_id`. A config that says `petal` keeps working
  silently, and a store indexed by `petal` needs no re-embedding.

### Removed

- The torch-backed `bloom` and `sprout` engines, and with them the `[torch]`
  install extra and its `ml` alias. `sentence-transformers` and torch are no
  longer dependencies of poppy-memory in any configuration. These engines live
  on in the poppy-lab experiment repo; if a future evaluation shows the torch
  stack materially better, it can come back with published results behind it.

### Fixed

- Restoring a deleted memory from the dashboard now reaches the cloud. The
  restored row kept its original `updated_at`, which sits below sync's push
  watermark, so the cloud row stayed deleted and the next pull deleted the
  memory again locally.
- Deleting a memory with a TTL and restoring it keeps its expiry. The tombstone
  table never stored `expires_at`, so a restored memory came back permanent and
  pushed a null expiry to the cloud. A memory whose TTL ran out while it was
  deleted is no longer brought back at all: the dashboard reports that it
  expired, rather than resurrecting data that was scheduled to disappear. Its
  tombstone is left to age out of the restore window on its own, so nothing is
  destroyed by the attempt.

### Security

- The daemon's MCP endpoint now requires a bearer token however the daemon is
  started. `poppy daemon run`/`daemon start` previously served `/mcp` (every MCP
  tool, including `forget`) and `/status` (which exposed the store path) with no
  authentication unless a token had been created by `setup <client> --daemon`,
  leaving them open to any local process. The daemon now mints a token on start,
  and an existing token is never overwritten by a transient `POPPY_DAEMON_TOKEN`.
- Forgetting, deleting, or editing the text of a multi-speaker memory now
  removes the per-speaker copies the default engine derives from it, including
  any it adopted from an older version. An edit that only changes metadata, such
  as the project, keeps them. Those copies duplicate
  the memory's text, and previously survived a redaction: the "forgotten" text
  stayed recallable, and synced to the cloud as live rows of its own. They are
  now identified by a marker the engine writes when it creates them, so the
  removal is reliable whatever the memory's id looks like, and a real memory
  whose id merely resembles one is never hidden or deleted. Existing stores are
  upgraded on first open, and how much is done to each copy depends on how
  certain it is: one whose text still matches its memory is simply marked, one
  whose text has drifted is marked but left untouched otherwise, and one whose
  memory is gone is not touched at all. `poppy doctor` reports the last two so
  they can be repaired deliberately rather than on a guess. A store that has only
  ever used the fallback engine is left alone entirely, since nothing in it ever
  made a copy. Marked copies are local to each
  device and stay out of sync in both directions, so nothing arriving from the
  cloud can turn one back into an ordinary memory or delete one this device
  still needs, and a deletion sends the cloud a removal that carries no memory
  text at all.
- A memory whose id happens to match one the engine would derive keeps that id
  and is left alone: it is not overwritten, hidden, deleted or synced as a copy.
  Ids Poppy mints cannot collide that way, so this only arises for ids supplied
  through the web API, an importer or a research harness. The one consequence is
  that if this device already derives a copy at an id, a cloud memory arriving
  later under the same id is not pulled down here; the local copy keeps the id.
- Per-speaker copies that an older Poppy had uploaded are cleaned out of the
  cloud on the first sync after upgrading: each is announced once as deleted,
  carrying no memory text, and retried until the server accepts it. Copies
  created from this release on are purely local, so nothing is ever sent for
  them. This clean-up is best effort. The deletion carries the copy's original
  timestamp, which is what stops it from overwriting a newer memory another
  device wrote at the same id, and the side effect is that a device which has
  already synced past that point will not see it. Removing such copies for good
  across every device is done from the server side.
- A store shared with an older Poppy (for example a 0.2.4 install beside this
  one, both using `~/.poppy`) can end up holding per-speaker copies this version
  has not marked. Those are listed and synced like ordinary memories and a
  redaction will not reach them. `poppy doctor` reports how many there are; the
  fix is to use one Poppy version against a store.
- A deletion pulled from another device is dated from when it happened there,
  not from when this device received it, and a deletion carrying the same
  timestamp as the local memory now applies, matching the server's rule. The
  seven-day Trash window is therefore measured from the deletion itself, so a
  device that was offline longer than that sees such deletions age out on its
  next sync rather than gaining a fresh window.
- Deleted memories now stay in Trash until sync confirms the deletion reached
  the cloud, so a device that is offline or has a rejected API key keeps them
  past the seven-day window rather than dropping a deletion that never
  propagated. A store with no cloud account configured is unaffected: nothing is
  waiting on those deletions, so they age out on the window as before.

### Upgrading

- **If you sync across more than one device on the same Trags account:**
  upgrade all of them in the same sitting, with auto-sync off until every
  device is on 0.3.0. Back up `~/.poppy` before you start. Do not run an
  older Poppy against a store this version has opened, since old versions do
  not use the per-speaker markers, deletion timestamps, or key storage this
  release relies on. The cross-device clean-up of old per-speaker copies
  described above is best effort and finishes server-side.
- **If your engine was `petal` or `seed`:** nothing to do. Your stored vectors
  are unchanged.
- **If your memories were embedded by the torch `bloom` (including via the
  `best` or `speaker_closet` aliases) or by `sprout`:** the name now resolves to
  the ONNX `bloom` (with a one-line notice for `sprout`). Those memories were
  embedded by a different bi-encoder, so they stay searchable by keyword but sit
  out the vector channel until you re-embed them:

  ```
  poppy migrate-engine
  ```

  `poppy doctor` reports how many memories are waiting. Nothing is deleted and
  no memory becomes unreachable in the meantime.

## [0.2.4] - 2026-07-23

### Added

- `poppy setup cursor` now installs native global Cursor hooks plus a Cursor
  transcript adapter for cadence capture, pre-compaction flushes, and a real
  session-end backstop. `poppy doctor` checks hook validity and Cursor capture
  liveness, including the server-side account flag when installed hooks stay silent.
- `poppy setup codex` now installs merge-safe Codex hooks and a rollout
  transcript adapter for cadence-only zero-touch capture. Setup and
  `poppy doctor` explain Codex's one-time interactive hook trust gate.
- Custom automatic-capture redaction with `poppy redaction add`, `remove`, and
  `list`. Exact literals and the current values of named environment variables
  are masked in place alongside the six built-in secret pattern families.

### Changed

- The Trags API key is now stored in the OS keychain on a standard install, not
  only when the `encryption` extra is present. `keyring` moved from the
  `encryption` extra into the base dependencies, so sync-only installs keep the
  key in the OS credential store instead of a plaintext `config.json`. Headless
  hosts with no usable keychain backend still fall back to `config.json` (0600),
  and `POPPY_TRAGS_API_KEY` still overrides both.
- Configuration keys are now driven by a single internal registry. Invalid
  `poppy config set` values are rejected with a clean error before anything is
  written, instead of a traceback, and `telemetry` now appears in the valid
  settings listing.

### Fixed

- `poppy setup claude-desktop` now resolves the config file at
  `$XDG_CONFIG_HOME/Claude/claude_desktop_config.json` (default
  `~/.config/Claude/`) on Linux, matching the first-party apt build of Claude
  Desktop. Previously it fell back to the macOS path on every non-Windows
  platform, which pointed at a location the Linux app never reads.
- Human-facing CLI commands now report when an unavailable configured retrieval
  engine falls back to another engine, throttled to once per day, while hooks,
  MCP serving, and the daemon remain quiet. Torch engine dependency checks now
  perform a real import and include the underlying import failure in diagnostics.

## [0.2.3] - 2026-07-18

### Added

- Poppy can now run as a single shared background daemon, so many concurrent
  editor and CLI sessions reuse one resident set of retrieval models instead of
  each loading its own. `poppy serve` gains `--transport http` and
  `--transport streamable-http` (the default stays `stdio` and is byte-for-byte
  unchanged), listening on `127.0.0.1:7679` with optional bearer-token
  authentication. A `poppy daemon` command group (`run`, `start`, `stop`,
  `restart`, `status`, `install`, `uninstall`) installs and controls the daemon
  as a launchd user agent on macOS or a systemd user unit on Linux; installing
  and starting are idempotent, so re-running setup is safe. An authenticated
  `/status` endpoint reports version, uptime, the active engine, and model
  residency. `poppy setup <client> --daemon` points a client at the daemon and
  mints a `0600` token, supported for Claude Code, Cursor, VS Code, Codex, and
  Gemini. A plain stdio `poppy serve` self-heals onto a running daemon (starting
  an installed one if needed and forwarding transparently) and falls back to an
  in-process server when no daemon is reachable, so it never hard-fails. Capture
  workers reuse the daemon's resident models instead of cold-loading the
  embedding stack on every event. The daemon is opt-in: plain stdio setup
  remains the default.
- Added an anonymous, daily PyPI update check to `poppy doctor` and a cached,
  once-per-version upgrade hint for interactive CLI commands. Both have a
  dedicated opt-out and honor the umbrella telemetry opt-out.
- Experimental, opt-in ONNX model-file override seam in the fastembed loader.
  `POPPY_ONNX_BI_MODEL_DIR`/`POPPY_ONNX_BI_MODEL_FILE` and
  `POPPY_ONNX_CE_MODEL_DIR`/`POPPY_ONNX_CE_MODEL_FILE` run alternate local ONNX
  bi-encoder and cross-encoder files for latency probes. Left unset, the stock
  model pins are byte-identical; invalid override paths fail closed with an
  actionable error rather than silently reverting to the stock model. The
  shipped model pins are unchanged, and an overridden bi-encoder reuses the
  existing embedding keyspace, so this is valid for probing only, not for
  shipping a model swap.

### Changed

- `poppy autocapture` is now the primary automatic-capture control, combining
  global consent and per-project scope. `poppy consent` remains visible and
  compatible for now, while `poppy capture` remains as a hidden deprecated
  alias. Both aliases will be retained for at least one minor release.

### Fixed

- `poppy serve` no longer balloons memory when many sessions run at once.
  Retrieval models load lazily on first use instead of at startup and unload
  after an idle period (`POPPY_MODEL_IDLE_S`, default 600 seconds), and the ONNX
  cross-encoder reranks in smaller batches; in measurements this cut resident
  memory and steady recall latency with identical scores. Listing or
  constructing an engine no longer imports torch, and a present-but-broken
  torch install now surfaces an actionable error naming
  `poppy engines use petal` instead of a raw traceback.
- The `torch` and `ml` install extras now cap `tokenizers` below 0.23. A fresh
  resolve had begun pulling tokenizers 0.23.1, which transformers rejects at
  import time, breaking the bloom engine on a clean install of the extra.
- A failed engine initialization no longer leaves an open write transaction
  behind. Every engine constructor now rolls back and closes its database
  connection on any failure path, so a partway-through init (for example a
  broken torch import after a startup migration) can no longer strand a lock and
  wedge every later command with "database is locked". The daemon's `/status`
  also gains an `engine_error` field, so a degraded engine is visible to health
  checks instead of reporting healthy.

## [0.2.2] - 2026-07-15

### Changed

- Relicensed from AGPL-3.0-or-later to Apache-2.0. The LICENSE file, package
  metadata, and MCP bundle manifest now carry the Apache-2.0 terms.
- The local browse UI adopts the herbarium treatment (paper chrome, forest
  data-ink) to match the hosted Trags dashboard.
- The documented telemetry event table is now enforced by a test: the emit
  call sites in code, the canonical event list, and the README table must
  agree, so the "complete list" claim cannot silently drift.
- The README now documents all four shipped engines: `petal` (the
  ONNX/fastembed architecture, no torch) had been missing even though it is
  what a slim base install runs and the automatic fallback when the torch
  engines are unavailable.
- Internal consolidation with no behavior change: the three capture entry
  points now share one orchestrator, Remember/Forget flows share one write
  core, reconcile decisions run behind a single seam, and transcript reading
  goes through a per-client adapter.

### Fixed

- Closed-loop telemetry no longer counts bulk-imported memories, and funnel
  milestone latches are now atomic across concurrent Poppy processes.
- The README install line now uses the canonical install URL
  (`https://trags.ai/install/poppy`) instead of the raw GitHub URL.
- The README now documents two shipped capture features that were undocumented:
  the per-project capture off switch (`poppy capture --off/--on/--status`, a
  deny-list layered on top of the global consent) and secret redaction, which
  masks secret-shaped tokens in automatically captured memories before they are
  stored, journalled, or synced. `poppy capture` is now in the command table.
- The README telemetry section now lists every event Poppy emits. It claimed to
  be a complete list while documenting 4 of the 9 events; the 5 one-time funnel
  milestones (`setup_completed`, `consent_granted`, `consent_pending_shown`,
  `first_autocapture_stored`, `closed_loop`) were undisclosed, and the
  `memory_write` row understated where it fires. No telemetry behavior changed;
  the properties were always content-free.
- `poppy setup codex` now registers the MCP server in `~/.codex/config.toml`
  and installs an AGENTS.md primer; the previous setup wrote configs Codex
  never read.
- copilot-cli and pi primers are written to the files those clients actually
  read, with environment overrides for nonstandard install paths.
- The hermes primer moved to SOUL.md and now references the plugin's real
  `poppy_*` tool names.
- Goose setup now tags captured memories with `goose` provenance and honors
  `GOOSE_PATH_ROOT`. Existing installs must re-run `poppy setup goose` to update
  the managed extension entry.
- MCP configs register poppy by absolute path, so GUI-launched clients
  (including the Claude Desktop MSIX install on Windows) work without shell
  PATH. Existing installs: re-run `poppy setup <client>` to refresh entries.
- Claude Code setup refreshes the PreToolUse hook matcher and installs the
  PostCompact hook that earlier versions promised but never wrote.
- CLI help no longer leaks internal tracker ids, and `-v` works as a version
  alias.

## [0.2.1] - 2026-07-06

### Added

- Optional encryption at rest for the local store (`poppy encrypt enable` /
  `status` / `disable` / `repair`), installable via the
  `poppy-memory[encryption]` extra. Encrypts the whole SQLite database with
  SQLCipher (content, FTS5 index, embedding vectors, and the `-wal` sidecar's
  frame contents; the `-shm` wal-index holds no database content), with a random
  256-bit key held in the OS keychain (macOS Keychain, Linux Secret Service,
  Windows Credential Locker). Existing plaintext stores are migrated in place
  under an exclusive lock with before/after verification, and migration refuses
  when a live writer is detected; the CLI, MCP server, and web UI work
  unchanged. Off by default. This is local encryption at rest, not
  zero-knowledge or end-to-end encryption. Set `POPPY_DB_KEY` to supply the key
  on a headless host with no keychain.
- Per-project auto-capture off switch: `poppy capture --off` / `--on` keeps a
  sensitive repo out of automatic capture without changing global consent or
  breaking recall in that project.
- Secret redaction of captured content: auto-captured memories are scanned for
  credentials and secrets and masked before they are written to the store.
- `-h` as an alias for `--help` on every command.
- The Trags API key is now stored in the OS keychain instead of a plaintext
  config file.
- Configurable recall relevance floor with abstention, so low-confidence
  matches can be withheld rather than returned.
- Activation-funnel telemetry: at-most-once milestone events (setup → consent →
  first auto-capture → closed loop) to measure onboarding, respecting the same
  opt-out as all other telemetry.

### Changed

- SessionStart hook injection is now bounded. The project memory snapshot is
  selected by recency, truncated per memory (300 chars), and held under a total
  block budget (~2 KB), with a marker line when memories are dropped to fit.
  Previously a single long memory could balloon the injected block to ~17 KB.
- Consent-pending and INACTIVE capture banners now surface through the hook
  `systemMessage` channel so Claude Code renders them to the developer; the
  reassuring ACTIVE line stays agent-context only.
- The default retrieval backend is now a slim ONNX/fastembed petal engine;
  heavyweight PyTorch is moved to the `poppy-memory[torch]` extra so the base
  install ships without large ML dependencies.
- Auto-captured memories are stamped with the source app of the host CLI.

### Fixed

- The `poppy consent` help text no longer references a dead pre-migration issue
  id.
- `poppy forget` now tombstones deletes on both the CLI and the MCP server, and
  sync pull consults tombstones, so a later sync can no longer resurrect a
  deleted memory.
- Correctness and hardening pass across the CLI.

### Security

- Web UI: escaped a stored-XSS vector, added a Content-Security-Policy and Host
  allowlist, and bounded MCP output size.

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
