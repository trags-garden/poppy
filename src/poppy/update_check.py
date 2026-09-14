"""Anonymous, cached update checks for the human-facing CLI.

Only :func:`check` may contact PyPI. Normal CLI commands use
:func:`consume_update_notice`, which reads the cache and latches a notice
without performing network I/O. Every public operation is best-effort: an
update check must never break a Poppy command.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
import tempfile
import threading
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Mapping, TextIO

from poppy.paths import ensure_poppy_dir

PYPI_URL = "https://pypi.org/pypi/poppy-memory/json"
CACHE_FILENAME = "update_check.json"
CHECK_INTERVAL = datetime.timedelta(hours=24)
_VERSION_RE = re.compile(r"\d+(?:\.\d+)*")
_thread_lock = threading.Lock()

Fetcher = Callable[[], Mapping[str, object] | str | None]


@dataclass(frozen=True)
class CheckResult:
    """The installed/latest versions and whether the check was permitted."""

    installed_version: str
    latest_version: str | None
    enabled: bool = True
    disabled_reason: str | None = None
    refreshed: bool = False

    @property
    def update_available(self) -> bool:
        return is_newer(self.latest_version, self.installed_version)


def installed_version() -> str:
    """Return the installed distribution version with a source-tree fallback."""
    try:
        return metadata.version("poppy-memory")
    except Exception:
        try:
            from poppy import __version__

            return __version__
        except Exception:
            return "unknown"


def _parse_version(value: str | None) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not _VERSION_RE.fullmatch(value):
        return None
    return tuple(int(part) for part in value.split("."))


def is_newer(candidate: str | None, current: str | None) -> bool:
    """Compare clean dot-separated integer versions, padding missing parts."""
    candidate_parts = _parse_version(candidate)
    current_parts = _parse_version(current)
    if candidate_parts is None or current_parts is None:
        return False
    width = max(len(candidate_parts), len(current_parts))
    padded_candidate = candidate_parts + (0,) * (width - len(candidate_parts))
    padded_current = current_parts + (0,) * (width - len(current_parts))
    return padded_candidate > padded_current


def upgrade_command(prefix: str | None = None) -> str:
    """Return the appropriate upgrade command for pipx or pip installs."""
    active_prefix = (prefix or sys.prefix).replace("\\", "/")
    if "/pipx/venvs/" in active_prefix:
        return "pipx upgrade poppy-memory"
    return "pip install -U poppy-memory"


def status(poppy_dir: Path) -> tuple[bool, str | None]:
    """Return whether update checks are enabled and an opt-out reason."""
    if os.environ.get("POPPY_UPDATE_CHECK_OFF") == "1":
        return False, "POPPY_UPDATE_CHECK_OFF=1"
    if os.environ.get("POPPY_TELEMETRY_OFF") == "1":
        return False, "POPPY_TELEMETRY_OFF=1"
    try:
        from poppy.config import load_config
        from poppy.telemetry import status as telemetry_status

        if not load_config(poppy_dir).update_check:
            return False, f"update_check is off in {poppy_dir / 'config.json'}"
        telemetry_enabled, reason = telemetry_status(poppy_dir)
        if not telemetry_enabled:
            return False, f"telemetry is off: {reason}"
    except Exception:
        return False, "configuration could not be read"
    return True, None


def _cache_path(poppy_dir: Path) -> Path:
    return poppy_dir / CACHE_FILENAME


def _empty_cache() -> dict[str, str | None]:
    return {"checked_at": None, "latest_version": None, "notified_version": None}


def _load_cache(poppy_dir: Path) -> dict[str, str | None]:
    try:
        raw = json.loads(_cache_path(poppy_dir).read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return _empty_cache()
    if not isinstance(raw, dict):
        return _empty_cache()
    return {
        "checked_at": raw.get("checked_at") if isinstance(raw.get("checked_at"), str) else None,
        "latest_version": raw.get("latest_version") if isinstance(raw.get("latest_version"), str) else None,
        "notified_version": (raw.get("notified_version") if isinstance(raw.get("notified_version"), str) else None),
    }


def _save_cache(poppy_dir: Path, data: Mapping[str, str | None]) -> None:
    ensure_poppy_dir(poppy_dir)
    payload = json.dumps(
        {
            "checked_at": data.get("checked_at"),
            "latest_version": data.get("latest_version"),
            "notified_version": data.get("notified_version"),
        },
        indent=2,
    )
    fd, tmp_name = tempfile.mkstemp(dir=str(poppy_dir), prefix=".update-check-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as file:
            file.write(payload)
        os.replace(tmp_name, _cache_path(poppy_dir))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@contextmanager
def _cache_file_lock(poppy_dir: Path):
    """Serialize cache refreshes and notice latches across processes."""
    with _thread_lock:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback
            yield
            return

        ensure_poppy_dir(poppy_dir)
        lock_path = _cache_path(poppy_dir).with_suffix(".json.lock")
        with lock_path.open("a") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _as_utc(now: datetime.datetime | None) -> datetime.datetime:
    value = now or datetime.datetime.now(datetime.UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=datetime.UTC)
    return value.astimezone(datetime.UTC)


def _is_fresh(checked_at: str | None, now: datetime.datetime) -> bool:
    if checked_at is None:
        return False
    try:
        checked = datetime.datetime.fromisoformat(checked_at)
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=datetime.UTC)
        else:
            checked = checked.astimezone(datetime.UTC)
    except ValueError:
        return False
    age = now - checked
    return datetime.timedelta(0) <= age < CHECK_INTERVAL


def _production_fetcher(version: str) -> Mapping[str, object]:
    request = urllib.request.Request(
        PYPI_URL,
        headers={"User-Agent": f"poppy-memory/{version} (update-check)"},
    )
    with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310
        payload = json.load(response)
    return payload if isinstance(payload, dict) else {}


def _latest_from_payload(payload: Mapping[str, object] | str | None) -> str | None:
    if isinstance(payload, str):
        version = payload
    elif isinstance(payload, Mapping):
        info = payload.get("info")
        if not isinstance(info, Mapping):
            return None
        version = info.get("version")
    else:
        return None
    if not isinstance(version, str) or _parse_version(version) is None:
        return None
    return version.strip()


def check(
    poppy_dir: Path,
    fetcher: Fetcher | None = None,
    now: datetime.datetime | None = None,
) -> CheckResult:
    """Refresh the daily PyPI cache when due, swallowing every failure."""
    installed = installed_version()
    enabled, reason = status(poppy_dir)
    if not enabled:
        return CheckResult(installed, None, enabled=False, disabled_reason=reason)

    current_time = _as_utc(now)
    try:
        with _cache_file_lock(poppy_dir):
            cache = _load_cache(poppy_dir)
            if _is_fresh(cache["checked_at"], current_time):
                return CheckResult(installed, cache["latest_version"])

            cache["checked_at"] = current_time.isoformat()
            try:
                payload = fetcher() if fetcher is not None else _production_fetcher(installed)
                cache["latest_version"] = _latest_from_payload(payload)
            except Exception:
                pass
            _save_cache(poppy_dir, cache)
            return CheckResult(installed, cache["latest_version"], refreshed=True)
    except Exception:
        return CheckResult(installed, None)


def consume_update_notice(poppy_dir: Path) -> str | None:
    """Return and latch a cached update notice, without any network access."""
    enabled, _ = status(poppy_dir)
    if not enabled:
        return None
    try:
        with _cache_file_lock(poppy_dir):
            cache = _load_cache(poppy_dir)
            latest = cache["latest_version"]
            installed = installed_version()
            if not is_newer(latest, installed) or cache["notified_version"] == latest:
                return None
            cache["notified_version"] = latest
            _save_cache(poppy_dir, cache)
        return f"poppy {latest} is available (you have {installed}). Upgrade: {upgrade_command()}"
    except Exception:
        return None


def maybe_print_update_notice(poppy_dir: Path, stderr: TextIO | None = None) -> None:
    """Print one cached update notice only for an interactive stderr."""
    try:
        enabled, _ = status(poppy_dir)
        if not enabled:
            return
        stream = stderr or sys.stderr
        if not stream.isatty():
            return
        notice = consume_update_notice(poppy_dir)
        if notice:
            print(notice, file=stream)
    except Exception:
        pass
