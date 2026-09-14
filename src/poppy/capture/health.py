"""Extraction-backend health — consecutive host-CLI failures.

A host CLI on PATH proves the binary exists, not that it works: a logged-out
``claude`` exits non-zero on every extraction, so every capture fails while the
SessionStart banner still says "active" and ``poppy doctor`` still says OK. The
extraction backend therefore records what actually happened — each hard host-CLI
failure (non-zero exit, timeout, spawn error) increments a counter and stores the
failing CLI plus its stderr tail; the next successful extraction clears it. After
``FAILURE_THRESHOLD`` consecutive failures the ConsolidationPolicy reports
``WARN_BACKEND_BROKEN``, which the banner and doctor render.

Failures are counted **per CLI**, never globally. A developer whose ``claude``
and ``codex`` are both logged out alternates between them from one session to
the next, and a single shared counter restarts on every switch, so two broken
backends hide each other and the warning never fires. Each CLI keeps its own
count, any CLI that reaches the threshold trips the warning and is the one
named, and a working extraction clears only that CLI's count.

The record is a small JSON file next to the capture state and journal — a
per-device cache of capture progress, so it is NOT synced. Every writer
publishes through a temporary file of its own and ``os.replace``, so a reader
only ever sees a whole record. The writes are unlocked, so two capture workers
failing at once can still lose an increment, which delays the warning by one
capture.

``WARN_BACKEND_BROKEN`` stays an *enabled* status: capture keeps firing while the
backend is broken, so a re-login heals the state on the next successful capture
instead of latching the warning on forever.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from poppy.paths import ensure_poppy_dir

HEALTH_FILENAME = "capture_health.json"

# Consecutive failures before the backend counts as broken. Low enough to catch a
# logged-out CLI inside one session, high enough to ride out a one-off timeout.
FAILURE_THRESHOLD = 3

# Cap on the stored stderr tail so a chatty backend can never grow the file.
MAX_ERROR_CHARS = 500

# Cap on how many CLIs the record tracks. Only four host CLIs are ever detected,
# so this is a bound on a file that must stay small rather than a live concern;
# the least recently failing entries are the ones dropped.
MAX_TRACKED_CLIS = 8


@dataclass(frozen=True)
class CliHealth:
    """One host CLI's current run of consecutive failures."""

    consecutive_failures: int = 0
    last_error: str = ""
    last_failure_at: str = ""


@dataclass
class BackendHealth:
    """Last known state of the extraction backend, one entry per failing CLI.

    ``cli`` / ``consecutive_failures`` / ``last_error`` report on the single
    backend worth telling the developer about, so the banner and doctor can stay
    one-line: the CLI that crossed the threshold, or the latest one to fail.
    """

    clis: dict[str, CliHealth] = field(default_factory=dict)

    @property
    def failing_cli(self) -> str | None:
        """The CLI whose consecutive failures reached ``FAILURE_THRESHOLD``, if any.

        With several broken at once the worst run wins, most recent failure
        breaking the tie, so the warning names one backend to go and fix.
        """
        broken = [(name, e) for name, e in self.clis.items() if e.consecutive_failures >= FAILURE_THRESHOLD]
        if not broken:
            return None
        return max(broken, key=lambda item: (item[1].consecutive_failures, item[1].last_failure_at))[0]

    @property
    def failing(self) -> bool:
        """Whether any backend has failed ``FAILURE_THRESHOLD`` times running."""
        return self.failing_cli is not None

    @property
    def cli(self) -> str | None:
        """The backend this record speaks for: the failing one, else the latest to fail."""
        failing = self.failing_cli
        if failing is not None:
            return failing
        if not self.clis:
            return None
        return max(self.clis.items(), key=lambda item: item[1].last_failure_at)[0]

    def _reported(self) -> CliHealth:
        name = self.cli
        return self.clis[name] if name is not None else CliHealth()

    @property
    def consecutive_failures(self) -> int:
        return self._reported().consecutive_failures

    @property
    def last_error(self) -> str:
        return self._reported().last_error

    @property
    def last_failure_at(self) -> str:
        return self._reported().last_failure_at


def health_path(poppy_dir: Path) -> Path:
    return poppy_dir / HEALTH_FILENAME


def _resolve(poppy_dir: Path | None) -> Path:
    if poppy_dir is not None:
        return poppy_dir
    from poppy.runtime import get_poppy_dir

    return get_poppy_dir()


def load(poppy_dir: Path | None = None) -> BackendHealth:
    """Read the recorded backend health; an absent or unreadable file reads healthy.

    Never raises, for any content the file might hold. Beyond a missing file, the
    record can be hand-edited or corrupted from outside Poppy: non-UTF-8 bytes
    raise ``UnicodeDecodeError`` and a non-numeric count raises ``ValueError`` or
    ``TypeError``, none of which are ``OSError``, so without this they would
    escape through ``evaluate()`` into the capture hooks and ``poppy doctor``.
    A record that cannot be read is treated as no record at all.
    """
    try:
        data = json.loads(health_path(_resolve(poppy_dir)).read_text())
        if not isinstance(data, dict) or not isinstance(data.get("clis"), dict):
            return BackendHealth()
        clis = {
            str(name): CliHealth(
                consecutive_failures=int(entry.get("consecutive_failures", 0) or 0),
                last_error=str(entry.get("last_error", "") or ""),
                last_failure_at=str(entry.get("last_failure_at", "") or ""),
            )
            for name, entry in data["clis"].items()
            if isinstance(entry, dict)
        }
    except (OSError, ValueError, TypeError):
        return BackendHealth()
    return BackendHealth(clis=clis)


def _pruned(clis: dict[str, CliHealth]) -> dict[str, CliHealth]:
    """Trim the record to the ``MAX_TRACKED_CLIS`` most recently failing backends."""
    if len(clis) <= MAX_TRACKED_CLIS:
        return clis
    newest = sorted(clis.items(), key=lambda item: item[1].last_failure_at, reverse=True)
    return dict(newest[:MAX_TRACKED_CLIS])


def _save(poppy_dir: Path, health: BackendHealth) -> None:
    """Publish the record through a temporary file unique to this writer.

    ``os.replace`` is atomic, but only over a source no one else is touching: a
    temporary path shared by every writer puts two capture workers into the same
    file, so one can publish what the other is still writing and leave a torn
    record on disk — which then reads as healthy.
    """
    ensure_poppy_dir(poppy_dir)
    path = health_path(poppy_dir)
    payload = json.dumps(
        {
            "version": 1,
            "clis": {
                name: {
                    "consecutive_failures": entry.consecutive_failures,
                    "last_error": entry.last_error,
                    "last_failure_at": entry.last_failure_at,
                }
                for name, entry in health.clis.items()
            },
        },
        indent=2,
    )
    fd, tmp_name = tempfile.mkstemp(dir=poppy_dir, prefix=f"{HEALTH_FILENAME}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    finally:
        # A successful replace already moved the temp away; this is what keeps a
        # failed write or replace from leaving a stray .tmp behind in ~/.poppy.
        tmp.unlink(missing_ok=True)


def record_failure(cli: str, detail: str, *, poppy_dir: Path | None = None) -> None:
    """Count one hard failure of ``cli`` and store ``detail`` (its stderr tail).

    Only ``cli``'s own count moves. Another backend failing in between must not
    erase this one's history — that is what let two broken CLIs hide each other
    from the threshold indefinitely.

    Never raises: recording health must not be able to break a capture.
    """
    try:
        resolved = _resolve(poppy_dir)
        clis = dict(load(resolved).clis)
        previous = clis.get(cli)
        clis[cli] = CliHealth(
            consecutive_failures=(previous.consecutive_failures if previous else 0) + 1,
            last_error=detail[:MAX_ERROR_CHARS],
            last_failure_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        _save(resolved, BackendHealth(clis=_pruned(clis)))
    except OSError:
        pass


def record_success(cli: str | None = None, *, poppy_dir: Path | None = None) -> None:
    """Clear the failure record after a working extraction. Never raises.

    ``cli`` clears that backend's count and nothing else: one host CLI working
    says nothing about another that is still logged out. ``None`` means
    extraction as a whole succeeded without a host CLI — the configured remote
    fallback — which clears every recorded failure.
    """
    try:
        resolved = _resolve(poppy_dir)
        current = load(resolved).clis
        if not current:
            return
        if cli is None:
            _save(resolved, BackendHealth())
            return
        if cli not in current:
            return
        _save(resolved, BackendHealth(clis={n: e for n, e in current.items() if n != cli}))
    except OSError:
        pass


def backend_failing(poppy_dir: Path | None = None) -> bool:
    """Whether any extraction backend has failed ``FAILURE_THRESHOLD`` times running."""
    return load(poppy_dir).failing
