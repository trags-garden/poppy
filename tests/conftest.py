"""Shared test fixtures.

Every test runs with POPPY_DIR pointed at a per-test temp directory and with
POPPY_TELEMETRY_OFF=1, so the suite can never read or write the real
``~/.poppy`` and never makes telemetry network calls. A few CLI tests used to
fall through to ``Path.home() / ".poppy"`` when they only set client-specific
env vars; this fixture closes that hole for good.

Tests that exercise telemetry-on behavior opt back in explicitly, either via
``monkeypatch.delenv("POPPY_TELEMETRY_OFF")`` or CliRunner's
``env={"POPPY_TELEMETRY_OFF": None}``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_poppy_env(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    poppy_dir = tmp_path_factory.mktemp("poppy-dir")
    monkeypatch.setenv("POPPY_DIR", str(poppy_dir))
    monkeypatch.setenv("POPPY_TELEMETRY_OFF", "1")
    monkeypatch.delenv("POPPY_TELEMETRY_HOST", raising=False)


@pytest.fixture(autouse=True)
def _isolate_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the test suite touch the real OS keychain.

    Installs a fresh in-memory keyring backend per test and clears the
    POPPY_DB_KEY env override so encryption tests start from a known state.
    ``keyring`` is a base dependency, so the backend swap runs everywhere; the
    ``ImportError`` guard below stays only for an ancient env that predates
    keyring being a base dep, where it degrades to a no-op.
    """
    monkeypatch.delenv("POPPY_DB_KEY", raising=False)
    monkeypatch.delenv("POPPY_TRAGS_API_KEY", raising=False)
    # The Trags-key keychain read is cached per process; clear it so a value
    # cached in one test can never leak into the next.
    from poppy.config import _reset_trags_key_cache

    _reset_trags_key_cache()
    try:
        import keyring
        from keyring.backend import KeyringBackend
    except ImportError:
        yield
        return

    class _MemoryKeyring(KeyringBackend):
        priority = 1

        def __init__(self) -> None:
            super().__init__()
            self._store: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, username: str) -> str | None:
            return self._store.get((service, username))

        def set_password(self, service: str, username: str, password: str) -> None:
            self._store[(service, username)] = password

        def delete_password(self, service: str, username: str) -> None:
            if (service, username) not in self._store:
                from keyring.errors import PasswordDeleteError

                raise PasswordDeleteError("not found")
            del self._store[(service, username)]

    previous = keyring.get_keyring()
    keyring.set_keyring(_MemoryKeyring())
    try:
        yield
    finally:
        keyring.set_keyring(previous)


@pytest.fixture
def synced_state():
    """Build state for a remote with prior upload evidence, without filtering rows."""
    from poppy.sync.state import PUSH_WATERMARK_VERSION, RemoteState, SyncState

    def make(url="https://trags.test", *, last_pushed_at="2020-01-01T00:00:00+00:00"):
        return SyncState(
            remotes={url: RemoteState(last_pushed_at=last_pushed_at, push_watermark_v=PUSH_WATERMARK_VERSION)}
        )

    return make
