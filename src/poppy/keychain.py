"""OS keychain access for Poppy secrets.

Poppy keeps the local-store encryption key in the operating system's own
credential store, reached through the ``keyring`` library:

* macOS: Keychain
* Linux: Secret Service (GNOME Keyring / KWallet, over D-Bus)
* Windows: Credential Locker

The key is never written to ``config.json`` or any other file Poppy owns. This
is local encryption at rest with the key in the OS keychain. It is not
zero-knowledge or end-to-end encryption: anything running as your user account
that can read the keychain can decrypt the store.

``keyring`` is a base dependency, so keychain storage works on every install
(including sync-only). The import stays lazy anyway: it keeps the cold-import
path light, and it lets Poppy degrade to config.json on any ancient env that
somehow lacks the module.
"""

from __future__ import annotations

import contextlib
import os

from poppy.errors import KeychainUnavailable

# Namespaces every Poppy credential under one service so the OS credential
# manager groups them and a user can find/remove them by name.
SERVICE = "poppy-memory"

__all__ = [
    "SERVICE",
    "KeychainUnavailable",
    "PROBE_OK",
    "PROBE_NO_BACKEND",
    "PROBE_WRITE_REJECTED",
    "PROBE_READBACK_FAILED",
    "DELETE_REFUSED",
    "available",
    "probe",
    "writable",
    "get_secret",
    "set_secret",
    "delete_secret",
]

# What a session can actually do with the keychain, as classified by ``probe``.
# Callers use these to name the real failure instead of guessing at one.
PROBE_OK = "ok"
PROBE_NO_BACKEND = "no-backend"
PROBE_WRITE_REJECTED = "write-rejected"
PROBE_READBACK_FAILED = "readback-failed"
# Not a probe outcome: what a caller records when ``delete_secret`` could not
# remove an entry, so it can report the secret that may have survived.
DELETE_REFUSED = "delete-refused"

# Throwaway entry used by ``writable`` to prove this session may actually use
# the credential store. Namespaced under SERVICE like everything else, and one
# fixed account name rather than one per process: a process killed between the
# write and the delete strands its entry, and nothing enumerates or sweeps
# them, so per-process names would accumulate one dead entry per killed pid.
# Under a single name the next probe overwrites and deletes whatever was
# stranded, bounding the litter at one inert entry. The value carries the pid so
# a probe running concurrently in another process is recognisable as one.
_PROBE_ACCOUNT = "session-probe"
_PROBE_VALUE_PREFIX = "poppy-probe:"


def _keyring():
    try:
        import keyring  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        from poppy.encryption import INSTALL_HINT  # noqa: PLC0415

        raise KeychainUnavailable(f"the 'keyring' package is not installed. {INSTALL_HINT}") from exc
    return keyring


def available() -> bool:
    """Whether a real (non-failing) keyring backend is present.

    ``keyring`` always imports and always returns *a* backend; on a headless box
    with no Secret Service it returns the ``fail`` backend that raises on use.
    Detect that case up front so callers can surface a clear message instead of
    an opaque error mid-operation.
    """
    try:
        keyring = _keyring()
        from keyring.backends import fail  # noqa: PLC0415
    except Exception:
        return False
    try:
        return not isinstance(keyring.get_keyring(), fail.Keyring)
    except Exception:
        return False


def probe() -> str:
    """Try the keychain in this session and report which step fails.

    ``available`` only proves a real backend loaded. On macOS the real Keychain
    backend loads in an ssh, launchd or background session too, but every call
    is then refused with errSecInteractionNotAllowed (-25308) because the login
    keychain is locked to that session. The only honest check is to try: write a
    throwaway entry, read it back, delete it. Returns one of ``PROBE_OK``,
    ``PROBE_NO_BACKEND``, ``PROBE_WRITE_REJECTED`` or ``PROBE_READBACK_FAILED``
    so a caller reporting a fallback can name the failure it actually hit.

    A probe, not a lock: it says what happened just now, promises nothing about
    the next call, and threads or processes probing at once share the one entry.
    A session that may not use the keychain is refused with an exception, never
    with somebody else's value, so reading back any probe value still proves the
    round-trip. Only an entry deleted between our write and our read is
    ambiguous, and that is retried once.
    """
    if not available():
        return PROBE_NO_BACKEND
    value = f"{_PROBE_VALUE_PREFIX}{os.getpid()}"
    for _ in range(2):
        try:
            try:
                set_secret(_PROBE_ACCOUNT, value)
            except KeychainUnavailable:
                return PROBE_WRITE_REJECTED
            try:
                read_back = get_secret(_PROBE_ACCOUNT)
            except KeychainUnavailable:
                return PROBE_READBACK_FAILED
        finally:
            # The probe entry is ours to clean up; a keychain that refuses the
            # delete is already reported by the outcome we return.
            with contextlib.suppress(KeychainUnavailable):
                delete_secret(_PROBE_ACCOUNT)
        if read_back is not None:
            return PROBE_OK if read_back.startswith(_PROBE_VALUE_PREFIX) else PROBE_READBACK_FAILED
        # Read back empty: a concurrent probe's delete landed in between. Retry
        # once. A backend that accepts writes and stores nothing reads back
        # empty again, and is reported as unusable rather than papered over.
    return PROBE_READBACK_FAILED


def writable() -> bool:
    """Whether this session can actually store and read back a keychain entry.

    Callers that are about to depend on the keychain (enabling encryption,
    reporting encryption status) use this; ``probe`` is the same check for
    callers that need to say *why* it failed.
    """
    return probe() == PROBE_OK


def get_secret(account: str) -> str | None:
    """Read a secret, or ``None`` if it is not set."""
    keyring = _keyring()
    try:
        return keyring.get_password(SERVICE, account)
    except Exception as exc:
        raise KeychainUnavailable(f"could not read from the OS keychain: {exc}") from exc


def set_secret(account: str, value: str) -> None:
    keyring = _keyring()
    try:
        keyring.set_password(SERVICE, account, value)
    except Exception as exc:
        raise KeychainUnavailable(f"could not write to the OS keychain: {exc}") from exc


def delete_secret(account: str) -> None:
    """Remove a secret. A missing secret is not an error.

    ``keyring`` raises the same ``PasswordDeleteError`` whether the entry was
    never there and whether the backend refused the delete (a login keychain
    locked to this session, -25308), so tell the two apart by asking whether the
    entry survived. A delete the backend *accepted* is checked the same way: a
    backend can answer the call and keep the entry (a locked Secret Service
    collection), and only a read-back proves the removal. Raises
    ``KeychainUnavailable`` when the entry may still be there: a caller that
    silently treats a refused delete as a clear leaves a live secret behind.
    """
    keyring = _keyring()
    failure: Exception | None = None
    try:
        keyring.delete_password(SERVICE, account)
    except Exception as exc:
        failure = exc
    try:
        survived = keyring.get_password(SERVICE, account) is not None
    except Exception:
        # Cannot read, so cannot prove anything either way: trust a delete the
        # backend accepted, distrust one it refused.
        survived = failure is not None
    if not survived:
        return  # Gone, or never there: deleting nothing is a success from our side.
    if failure is None:
        raise KeychainUnavailable("could not delete from the OS keychain: the entry is still readable afterwards")
    raise KeychainUnavailable(f"could not delete from the OS keychain: {failure}") from failure
