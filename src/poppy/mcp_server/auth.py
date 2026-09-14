"""Optional static bearer authentication for the daemon MCP endpoint."""

from __future__ import annotations

import contextlib
import hmac
import os
import secrets
import tempfile
from collections.abc import Iterable
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

DAEMON_TOKEN_FILENAME = "daemon.token"
DAEMON_TOKEN_LOCK_FILENAME = ".daemon.token.lock"


def _fcntl():  # pragma: no cover - trivial import shim
    try:
        import fcntl

        return fcntl
    except ImportError:  # Windows
        return None


def _write_token_atomically(poppy_dir: Path, token_path: Path, token: str) -> None:
    """Write ``token`` to ``token_path`` via a secure temp file + atomic replace.

    ``tempfile.mkstemp`` gives an unpredictable name created with ``O_EXCL``, so
    a pre-planted symlink at a guessable temp path cannot redirect the write to
    a victim file. ``os.replace`` publishes it atomically.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(poppy_dir), prefix=f"{DAEMON_TOKEN_FILENAME}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(f"{token}\n")
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, token_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _mint_token(poppy_dir: Path, token_path: Path) -> str:
    """Create the token file if absent and return its token, race-safe on any OS.

    The advisory lock (:func:`_token_lock`) is a no-op on Windows, so the create
    itself uses ``os.link`` as the election: the temp file is written in full
    first, then linked into place, which is atomic and fails if another process
    already linked one. The loser reads that already-complete file, so two
    concurrent first-runs never diverge even without the lock.
    """
    existing = _read_token(token_path)
    if existing:
        return existing
    token = secrets.token_hex(32)
    fd, tmp_name = tempfile.mkstemp(dir=str(poppy_dir), prefix=f"{DAEMON_TOKEN_FILENAME}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(f"{token}\n")
        os.chmod(tmp_path, 0o600)
        try:
            os.link(tmp_path, token_path)
            return token
        except FileExistsError:
            # Another process linked first (the election loser reads its token).
            existing = _read_token(token_path)
            if existing:
                return existing
            # The file exists but is empty (touch / crash / ENOSPC). Repair it.
            os.replace(tmp_path, token_path)
            tmp_path = None  # consumed by os.replace
            return token
        except OSError:
            # Filesystems without hard-link support (exFAT, some network/virtual
            # mounts) raise a non-FileExistsError OSError from os.link, which
            # would otherwise escape `serve` and crash `daemon run` with a
            # traceback. The caller holds _token_lock on POSIX, so a re-read +
            # atomic replace is race-safe here; os.link stays the election on the
            # Windows/no-flock path above.
            existing = _read_token(token_path)
            if existing:
                return existing
            os.replace(tmp_path, token_path)
            tmp_path = None  # consumed by os.replace
            return token
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def bearer_token_matches(authorization: str, token: str | None) -> bool:
    """Return whether ``authorization`` contains the configured bearer token."""
    if not token:
        # An empty configured token must never authenticate: `Bearer ` would
        # otherwise match it via compare_digest("", "").
        return False
    scheme, separator, supplied_token = authorization.partition(" ")
    return bool(separator) and scheme.lower() == "bearer" and hmac.compare_digest(supplied_token, token)


def load_daemon_token(poppy_dir: Path) -> str | None:
    """Load the daemon token: the on-disk file wins, the env var only fills an absent file.

    The daemon and its clients (`daemon status`, the stdio forwarder, daemon
    inference) are separate processes with independent environments, so the only
    value they can both agree on is the shared ``daemon.token`` file. A
    per-process ``POPPY_DAEMON_TOKEN`` override would desync a client from what
    the daemon serves — the file is authoritative here exactly as it is in
    ``ensure_daemon_token``. The env token is only a fallback
    for an absent/empty file (which ``ensure_daemon_token`` then persists), so
    first boot still works. Both values are stripped so a padded env
    matches the stripped token the daemon serves and stores.
    """
    token_path = poppy_dir / DAEMON_TOKEN_FILENAME
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if token:
        return token
    env_token = (os.environ.get("POPPY_DAEMON_TOKEN") or "").strip()
    return env_token or None


def _read_token(token_path: Path) -> str:
    # Only a MISSING file means "no token yet". A file that exists but cannot be
    # read (e.g. mode 000) must NOT be treated as absent: minting a new token
    # would silently discard the real one and break every existing client.
    # Let that error propagate so startup fails loudly instead.
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


@contextlib.contextmanager
def _token_lock(poppy_dir: Path):
    """Hold an exclusive advisory lock across a read-or-write of the token file.

    Serialises `daemon run`, `daemon start` and `setup --daemon` on the same
    store so they never mint, repair or persist the token concurrently and end
    up with the daemon holding one value while disk/clients hold another.
    A no-op where fcntl is unavailable (Windows).
    """
    poppy_dir.mkdir(parents=True, exist_ok=True)
    fcntl = _fcntl()
    lock_fd = None
    if fcntl is not None:
        lock_fd = os.open(poppy_dir / DAEMON_TOKEN_LOCK_FILENAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(lock_fd)


def ensure_daemon_token(poppy_dir: Path, *, rotate_from_env: bool = False) -> str:
    """Return the daemon token, creating a mode-0600 token file when needed.

    Two roles share the store's ``daemon.token`` file with different, correct
    precedence for ``POPPY_DAEMON_TOKEN``:

    ``rotate_from_env=True`` — the DAEMON serve path (`daemon run`). MUST be
    called only once this process is actually going to serve (i.e. after the
    daemon run-lock is acquired) so a start that loses the lock to an already
    running daemon never rewrites the file. Here ``POPPY_DAEMON_TOKEN`` is a
    deliberate operator rotation: it is PERSISTED (atomic overwrite) and served,
    so a leaked credential can be revoked by restarting the daemon with a new env
    value and every client follows the new file. With no env token an existing
    file token is served; with neither, one is minted and persisted.

    ``rotate_from_env=False`` (default) — SEED mode (`setup <client> --daemon`,
    `daemon start` kwargs). An on-disk token is AUTHORITATIVE and returned
    untouched: a transient env (e.g. ``POPPY_DAEMON_TOKEN=x poppy setup ...``)
    must never overwrite a live daemon's credential, which would 401 every baked
    client unrecoverably. Only an ABSENT file is seeded — from the env token when
    set, so a fresh machine bakes exactly what the env-less service daemon later
    reads; otherwise, from a freshly minted one.

    The env token is stripped so the persisted/served value matches what file
    readers strip to. An empty file (a `touch`, or a crash/ENOSPC) is repaired
    rather than returned as "" — an empty token both authenticates a bare
    `Bearer ` and reads back as None for every client.
    """
    token_path = poppy_dir / DAEMON_TOKEN_FILENAME
    env_token = (os.environ.get("POPPY_DAEMON_TOKEN") or "").strip()

    if rotate_from_env and env_token:
        # Serving daemon + a deliberate operator rotation: overwrite and serve so
        # the file (and therefore every client) reflects the new credential.
        with _token_lock(poppy_dir):
            if _read_token(token_path) != env_token:
                _write_token_atomically(poppy_dir, token_path, env_token)
        return env_token

    with _token_lock(poppy_dir):
        # The WHOLE seed decision — read + return, not just the write — runs under
        # the lock so it is mutually exclusive with a concurrent daemon
        # rotate-overwrite (also locked): setup reads either the full pre-rotation
        # or the full post-rotation token, never a stale value mid-overwrite that
        # would bake a token the daemon no longer serves. The lock serialises this
        # on POSIX; _mint_token's os.link election keeps the mint race-safe even
        # where the lock is a no-op (Windows). Residual: setup and a deliberate
        # rotation firing at the same instant is last-writer-wins on two admin
        # actions — re-run setup after a rotation; we do not couple setup to the
        # daemon lifecycle.
        existing = _read_token(token_path)
        if existing:
            return existing
        if env_token:
            # Seed an absent file so an env-less service daemon reads the same
            # value the caller bakes into client configs (first boot).
            _write_token_atomically(poppy_dir, token_path, env_token)
            return env_token
        return _mint_token(poppy_dir, token_path)


class BearerAuthMiddleware:
    """Require a configured bearer token for requests below one ASGI path."""

    def __init__(self, app: ASGIApp, token: str | None, path: str | Iterable[str] = "/mcp") -> None:
        self.app = app
        self.token = token
        paths = (path,) if isinstance(path, str) else path
        self.paths = tuple(item.rstrip("/") or "/" for item in paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.token is None or scope["type"] != "http" or not self._is_protected(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        authorization = Headers(scope=scope).get("authorization", "")
        if bearer_token_matches(authorization, self.token):
            await self.app(scope, receive, send)
            return

        response = JSONResponse({"error": "unauthorized"}, status_code=401)
        await response(scope, receive, send)

    def _is_protected(self, path: str) -> bool:
        return any(path == protected or path.startswith(f"{protected}/") for protected in self.paths)
