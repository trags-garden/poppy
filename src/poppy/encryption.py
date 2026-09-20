"""Encryption at rest for the local Poppy store.

Poppy's local store (``$POPPY_DIR/memories.db``: SQLite + FTS5 + embedding
BLOBs) is plaintext by default. This module adds opt-in whole-database
encryption using SQLCipher, with a random 256-bit key held in the OS keychain.

Why whole-database (SQLCipher) rather than encrypting individual columns:

* The FTS5 index and the embedding BLOBs are *derived from* content. Encrypting
  only the ``content`` column would still leave the searchable FTS5 inverted
  index and the (partially invertible) embedding vectors sitting in plaintext on
  disk, so it would not honestly be "encrypted at rest".
* SQLCipher encrypts every page, including the FTS5 shadow tables and the
  embedding BLOBs. The ``-wal`` sidecar's frame contents are encrypted too. The
  ``-shm`` sidecar is a plaintext wal-index that holds no database content.
  Recall (FTS5 MATCH + vector dot products + rerank) runs unchanged because
  decryption is transparent at the connection layer, which is the single
  chokepoint in ``poppy.db.connect``.
* The key is a high-entropy random 256-bit value, passed to SQLCipher as a raw
  key. That skips the expensive PBKDF2 pass a passphrase would trigger on every
  open, keeping short-lived CLI / hook / sync processes cheap (sub-millisecond
  open cost).

Honesty note (binding): this is local encryption at rest, key in the OS
keychain. It is deliberately not described as zero-knowledge or end-to-end
encryption anywhere, because anything that runs as your user and can read the
keychain can decrypt the store.

Dependencies (``sqlcipher3``, ``keyring``) are optional and only imported on the
encrypted path; a plaintext install never pulls them in.
"""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from poppy import keychain
from poppy.db import (
    BUSY_TIMEOUT_MS,
    ENCRYPTION_SENTINEL,
    exclusive_gate,
    quarantine_sidecars,
    stray_sidecars,
)

# Re-exported so callers can keep using ``encryption.EncryptionError`` etc. The
# base classes live in poppy.errors so poppy.keychain can subclass
# KeychainUnavailable from EncryptionError without a circular import.
from poppy.errors import DependencyMissing, EncryptionError, KeychainUnavailable, StorePermissionError
from poppy.paths import ensure_poppy_dir

__all__ = [
    "EncryptionError",
    "DependencyMissing",
    "KeychainUnavailable",
    "EncryptionStatus",
    "EnableResult",
    "ENV_KEY",
    "INSTALL_HINT",
    "generate_key",
    "resolve_key",
    "database_is_encrypted",
    "is_enabled",
    "store_state",
    "connect_encrypted",
    "open_without_sentinel",
    "enable",
    "disable",
    "repair",
    "find_residue",
    "cleanup_residue",
    "status",
    "dependencies_installed",
]

DB_FILENAME = "memories.db"

# Environment override for the raw key (64 hex chars). A deliberate, weaker
# escape hatch for headless / CI hosts with no OS keychain, and the seam the
# test suite uses. When set it takes precedence over the keychain. The key is
# then visible to anything that can read the process environment, so it is
# documented as the lesser option.
ENV_KEY = "POPPY_DB_KEY"

# Install guidance for the optional extra. Most installs use pipx, so plain
# `pip install` would land the deps in the wrong interpreter; lead with the pipx form.
INSTALL_HINT = (
    "Install the encryption extra. If you installed Poppy with pipx (the default):\n"
    "  pipx inject poppy-memory sqlcipher3\n"
    "otherwise, in the environment that owns the `poppy` command:\n"
    "  pip install 'poppy-memory[encryption]'"
)

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

# Temp files migration leaves behind if it is killed mid-flight. ``.plain-tmp``
# is a decrypted copy, so it is treated as sensitive residue to be cleaned up.
_TMP_SUFFIXES = (".enc-tmp", ".plain-tmp")


@dataclass
class EncryptionStatus:
    enabled: bool
    encrypted_on_disk: bool
    deps_installed: bool
    # How usable the OS keychain is in this session, as observed rather than
    # guessed: ``usable`` (a throwaway probe entry was written and read back),
    # ``refused`` (a real backend loaded but the probe was refused: a locked
    # login keychain, an ssh or launchd session), ``unavailable`` (no real
    # backend loaded) or ``not-probed`` (POPPY_DB_KEY is active or the extra is
    # missing, so the keychain is not this session's key source and was left
    # alone). ``refused`` is not folded into ``unavailable``: a session may be
    # allowed to read the keychain and not to write it, and still have its key.
    keychain_state: str
    key_present: bool
    db_path: Path
    env_key_active: bool
    state: str
    inconsistent: bool
    env_key_malformed: bool = False
    residue: list[Path] = field(default_factory=list)
    stray: list[Path] = field(default_factory=list)


@dataclass
class EnableResult:
    migrated_rows: int
    created_empty: bool


# --- optional dependency + key helpers -------------------------------------


def _import_sqlcipher():
    try:
        from sqlcipher3 import dbapi2  # noqa: PLC0415
    except ImportError as exc:
        raise DependencyMissing(f"the 'sqlcipher3' package is not installed. {INSTALL_HINT}") from exc
    return dbapi2


def dependencies_installed() -> bool:
    try:
        import keyring  # noqa: F401, PLC0415
        import sqlcipher3  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _require_deps(verb: str) -> None:
    if not dependencies_installed():
        raise DependencyMissing(f"{verb} needs the optional encryption dependencies. {INSTALL_HINT}")


def _db_path(poppy_dir: Path) -> Path:
    return poppy_dir / DB_FILENAME


def _key_account(poppy_dir: Path) -> str:
    """Keychain account for a store, unique per resolved store directory.

    Keying on the absolute path lets several stores (real store, test dirs,
    per-project dirs) coexist in one keychain without clobbering each other.
    """
    return f"db-key:{poppy_dir.resolve()}"


def generate_key() -> str:
    """A fresh raw 256-bit key as 64 lowercase hex characters."""
    return secrets.token_hex(32)


def _normalize_key(value: str | None) -> str | None:
    """Canonicalize a key: strip surrounding whitespace, lowercase, validate.

    Returns the 64-hex-char canonical form, or ``None`` if the input cannot be
    made into a valid raw key. Hand-restored keys often carry a trailing newline
    or uppercase hex (SQLCipher treats hex case-insensitively); normalizing here
    means such a key still opens the store instead of being silently rejected.
    """
    if value is None:
        return None
    v = value.strip().lower()
    return v if _HEX64.match(v) else None


def _env_key() -> str | None:
    raw = os.environ.get(ENV_KEY)
    if raw is None:
        return None
    key = _normalize_key(raw)
    if key is None:
        raise EncryptionError(f"{ENV_KEY} must be 64 hex characters (a raw 256-bit key); got an invalid value.")
    return key


def resolve_key(poppy_dir: Path) -> str | None:
    """The raw key for a store: the env override if set, else the keychain.

    Both sources are normalized, so an uppercase or newline-terminated key from
    a hand-restored keychain entry still resolves. Returns ``None`` when no
    usable key is found.
    """
    env = _env_key()
    if env:
        return env
    return _normalize_key(keychain.get_secret(_key_account(poppy_dir)))


# --- detection / state ------------------------------------------------------


def database_is_encrypted(db_path: Path) -> bool:
    """Whether an existing DB file is SQLCipher-encrypted (by header bytes).

    Only true for a readable non-magic header. An unreadable file (permission/IO)
    is NOT reported as plaintext here; callers use ``store_state`` for the full
    classification including the ``unreadable`` state (an unreadable encrypted
    file must never be mistaken for plaintext, or repair would delete its key).
    """
    from poppy.db import header_state  # noqa: PLC0415

    return header_state(db_path) == "nonmagic"


def _sentinel(poppy_dir: Path) -> Path:
    return poppy_dir / ENCRYPTION_SENTINEL


def is_enabled(poppy_dir: Path) -> bool:
    """Whether encryption is enabled for a store (on-disk file or sentinel)."""
    return database_is_encrypted(_db_path(poppy_dir)) or _sentinel(poppy_dir).exists()


def store_state(poppy_dir: Path) -> str:
    """Classify the store, reconciling the on-disk header with the sentinel.

    Returns one of:
      * ``absent``                  no DB file and no sentinel
      * ``pending-encrypted``       no DB file yet but the sentinel says encrypt
      * ``plaintext``               plaintext header, no sentinel
      * ``encrypted``               encrypted header and sentinel agree
      * ``encrypted-no-sentinel``   opens under the key but the marker is missing
      * ``plaintext-stale-sentinel``plaintext on disk but a leftover marker exists
      * ``corrupt``                 non-SQLite header that will not open under any resolved key
      * ``unknown``                 non-SQLite header that cannot be verified (no key/extra)
      * ``unreadable``              a permission / read-only / I/O error, not corruption

    A non-magic header is never classified as encrypted on the header alone: a
    sentinel we wrote is trusted (only ever created after a verified encryption),
    otherwise the file is opened read-only (``immutable``) under the resolved key
    to tell a real encrypted store from a corrupt or foreign file. The probe
    never replays a sidecar and never mutates the store. ``encrypted-no-sentinel``
    and ``plaintext-stale-sentinel`` are inconsistent states the lifecycle
    commands surface with a clear message plus a ``poppy encrypt repair`` recovery.
    """
    from poppy.db import header_state  # noqa: PLC0415

    db = _db_path(poppy_dir)
    sentinel = _sentinel(poppy_dir).exists()
    state = header_state(db)
    if state == "unreadable":
        # An unreadable file must never be called plaintext (that would let repair
        # delete the key of an intact encrypted store).
        return "unreadable"
    if state == "absent":
        return "pending-encrypted" if sentinel else "absent"
    if state == "plaintext":
        return "plaintext-stale-sentinel" if sentinel else "plaintext"
    # nonmagic header
    if sentinel:
        return "encrypted"
    verdict = _probe_open(poppy_dir)
    if verdict == "encrypted":
        return "encrypted-no-sentinel"
    return verdict  # "corrupt" | "unknown" | "unreadable"


def _candidate_keys(poppy_dir: Path) -> list[str]:
    """Every distinct key worth trying: the env override AND the keychain entry.

    Trying both means a stale ``POPPY_DB_KEY`` shadowing a good keychain key does
    not make a healthy store look corrupt.
    """
    keys: list[str] = []
    try:
        env = _env_key()
    except EncryptionError:
        env = None
    if env:
        keys.append(env)
    try:
        kc = _normalize_key(keychain.get_secret(_key_account(poppy_dir)))
    except EncryptionError:
        kc = None
    if kc and kc not in keys:
        keys.append(kc)
    return keys


def _probe_open(poppy_dir: Path) -> str:
    """Classify a non-magic-header store by a read-only (``immutable``) open.

    Returns ``encrypted`` (opened under some resolvable key), ``unreadable`` (a
    permission/IO error), ``corrupt`` (keys were available but none opened it), or
    ``unknown`` (cannot verify: the extra is missing or no key resolves). Never
    replays a sidecar and never mutates the store.
    """
    if not dependencies_installed():
        return "unknown"
    keys = _candidate_keys(poppy_dir)
    if not keys:
        return "unknown"
    db = _db_path(poppy_dir)
    for key in keys:
        try:
            conn = _open_encrypted(db, key, wal=False, immutable=True)
        except StorePermissionError:
            return "unreadable"
        except EncryptionError:
            continue
        conn.close()
        return "encrypted"
    return "corrupt"


_INCONSISTENT = ("plaintext-stale-sentinel", "encrypted-no-sentinel")


def _inconsistent_message(state: str, poppy_dir: Path) -> str:
    if state == "plaintext-stale-sentinel":
        detail = "the store on disk is plaintext, but a leftover encryption marker is present"
    else:
        detail = "the store on disk is encrypted, but its encryption marker is missing"
    return (
        f"the encryption state of {poppy_dir} is inconsistent: {detail}. "
        "Run `poppy encrypt repair` to reconcile it, then retry."
    )


# --- connection -------------------------------------------------------------


def _key_pragma(key: str) -> str:
    # ``key`` must be a normalized 64-hex-char value (callers validate before
    # reaching here), so this literal is injection-safe. The x'...' blob form
    # makes SQLCipher use the raw bytes directly, skipping the KDF.
    return f"PRAGMA key = \"x'{key}'\""


_PERMISSION_MARKERS = ("readonly", "read-only", "unable to open", "disk i/o", "permission", "not authorized")


def _is_permission_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _PERMISSION_MARKERS)


_GATED_SQLCIPHER_CLS = None


def _gated_sqlcipher_cls():
    """A SQLCipher connection subclass that releases its gate handle on close."""
    global _GATED_SQLCIPHER_CLS
    if _GATED_SQLCIPHER_CLS is None:
        dbapi2 = _import_sqlcipher()

        class _GatedSqlcipherConnection(dbapi2.Connection):
            def close(self):
                handle = getattr(self, "_poppy_gate", None)
                try:
                    super().close()
                finally:
                    if handle is not None:
                        handle.release()

        _GATED_SQLCIPHER_CLS = _GatedSqlcipherConnection
    return _GATED_SQLCIPHER_CLS


def _immutable_uri(db_path: Path) -> str:
    # Percent-encode the path: a '%' (or '?', '#') in POPPY_DIR would otherwise be
    # decoded by SQLite's URI parser into a different, nonexistent path.
    from urllib.parse import quote  # noqa: PLC0415

    return "file:" + quote(str(db_path)) + "?immutable=1"


def _open_encrypted(
    db_path: Path, key: str, *, wal: bool, check_same_thread: bool = False, factory=None, immutable=False
):
    """Open + key a SQLCipher connection, verifying the key with a page read.

    A wrong key (or a non-SQLCipher file) surfaces as a clear EncryptionError; a
    permission / read-only / I/O failure surfaces as StorePermissionError, so the
    two are never conflated. ``immutable=True`` opens read-only via the SQLite
    ``immutable=1`` URI, which does NOT replay the ``-wal`` sidecar and cannot
    mutate the file -- the safe way to probe a store whose cipher may be in doubt.
    """
    dbapi2 = _import_sqlcipher()
    kwargs = {"check_same_thread": check_same_thread, "timeout": BUSY_TIMEOUT_MS / 1000}
    if factory is not None:
        kwargs["factory"] = factory
    conn = None
    try:
        # Opening the file itself can already fail with a permission / "unable to
        # open" error, so it is inside the guard too.
        if immutable:
            conn = dbapi2.connect(_immutable_uri(db_path), uri=True, **kwargs)
        else:
            conn = dbapi2.connect(str(db_path), **kwargs)
        # PRAGMA key must run before anything else touches the database.
        conn.execute(_key_pragma(key))
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        if wal and not immutable:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        # Forces SQLCipher to derive and verify the key now.
        conn.execute("SELECT count(*) FROM sqlite_master")
    except Exception as exc:
        if conn is not None:
            conn.close()
        if _is_permission_error(exc):
            raise StorePermissionError(
                f"could not open the store at {db_path}: {exc}. This looks like a permission or read-only "
                "filesystem problem, not corruption; fix the permissions and retry."
            ) from exc
        raise EncryptionError(
            f"could not open the encrypted store at {db_path}: the key is wrong or the file is not a "
            f"SQLCipher database ({exc})."
        ) from exc
    return conn


def connect_encrypted(db_path: Path | str, *, check_same_thread: bool = False) -> sqlite3.Connection:
    """Open an encrypted store through SQLCipher (gated factory), applying Poppy's pragmas.

    Called only from ``poppy.db.connect``, which attaches the gate handle to the
    returned connection. Tries every resolvable key (env AND keychain), so a stale
    ``POPPY_DB_KEY`` shadowing a good keychain entry does not break recall; the
    wrong-key error names that possibility. This path is only taken for a store
    with an intact sentinel (trusted, non-ambiguous), so its ``-wal`` is the same
    cipher and safe to replay.
    """
    path = Path(db_path)
    factory = _gated_sqlcipher_cls()
    keys = _candidate_keys(path.parent)
    if not keys:
        raise EncryptionError(_no_key_message(path))
    last: Exception | None = None
    for key in keys:
        try:
            return _open_encrypted(path, key, wal=True, check_same_thread=check_same_thread, factory=factory)
        except StorePermissionError:
            raise
        except EncryptionError as exc:
            last = exc
    # None of the resolvable keys opened it.
    raise EncryptionError(_wrong_key_message(path, last))


def _no_key_message(path: Path) -> str:
    extra = ""
    if os.environ.get(ENV_KEY):
        extra = f" ({ENV_KEY} is set but did not resolve to a valid key)"
    return (
        f"the store at {path} looks encrypted but no usable key was found{extra}. Expected it in the OS keychain "
        f"(or the {ENV_KEY} environment variable). If this store was encrypted on another machine or user "
        "account, its key is not available here."
    )


def _wrong_key_message(path: Path, cause: Exception | None) -> str:
    # ``cause`` is the EncryptionError _open_encrypted already formatted with this
    # same sentence and path; unwrap to the driver error underneath so the store
    # path and the "key is wrong" sentence each appear exactly once.
    if isinstance(cause, EncryptionError) and cause.__cause__ is not None:
        cause = cause.__cause__
    hint = ""
    if os.environ.get(ENV_KEY):
        hint = (
            f" The resolved key may be wrong or stale: {ENV_KEY} is set and shadows the OS keychain, so if this "
            "store belongs to a different machine/account, unset it and retry."
        )
    return (
        f"could not open the encrypted store at {path}: the key is wrong or the file is not a SQLCipher "
        f"database ({cause}).{hint}"
    )


# --- migration primitives ---------------------------------------------------


def _remove_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        try:
            sidecar.unlink()
        except OSError:
            pass


def _remove_shm(db_path: Path) -> None:
    """Remove only the ``-shm`` (rebuildable wal-index), keeping the ``-wal`` data.

    A failed replay attempt can leave a ``-shm`` created with the main file's
    (read-only) mode, which then blocks a retry after permissions are fixed.
    SQLite rebuilds the ``-shm`` from the ``-wal`` on the next open, so clearing
    it is safe and loses nothing.
    """
    shm = db_path.with_name(db_path.name + "-shm")
    try:
        shm.unlink()
    except OSError:
        pass


def _memory_count(conn: sqlite3.Connection) -> int:
    """Rows in ``memories``, or 0 if the table doesn't exist yet (empty store)."""
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
    if not row:
        return 0
    return conn.execute("SELECT count(*) FROM memories").fetchone()[0]


def _table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row count of every user table (incl. FTS shadow + embedding tables).

    Used to verify a migration copied *everything*, not just ``memories``. Table
    names come from the trusted schema catalog.
    """
    names = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    ]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = conn.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
    return counts


def _checkpoint_or_abort(db_path: Path, key: str | None) -> None:
    """Fold the ``-wal`` back into the main DB, refusing if another process is active.

    ``PRAGMA wal_checkpoint(TRUNCATE)`` returns ``(busy, ...)``; ``busy == 1``
    means another connection holds a read/write lock, i.e. a live Poppy process
    is using the store. Migrating under that condition risks losing writes, so
    we abort with an actionable message instead. ``key`` is None for a plaintext
    source, or the raw key for an encrypted one.
    """
    if key is None:
        conn = sqlite3.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1000)
    else:
        conn = _open_encrypted(db_path, key, wal=False)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.close()
    if row is not None and row[0] == 1:
        raise EncryptionError(
            "the store has active connections (another Poppy process is reading or writing it). "
            "Stop `poppy serve` (MCP), `poppy ui`, and any capture or sync workers, then retry."
        )


def _export_plaintext_to_encrypted(plain_db: Path, enc_db: Path, key: str) -> tuple[int, dict[str, int]]:
    """Copy a plaintext DB into a new SQLCipher DB via ``sqlcipher_export``.

    Returns ``(memory_count, table_counts)`` of the source for verification.
    """
    dbapi2 = _import_sqlcipher()
    conn = dbapi2.connect(str(plain_db), timeout=BUSY_TIMEOUT_MS / 1000)
    try:
        memory_count = _memory_count(conn)
        table_counts = _table_counts(conn)
        conn.execute(f"ATTACH DATABASE ? AS encrypted KEY \"x'{key}'\"", (str(enc_db),))
        conn.execute("SELECT sqlcipher_export('encrypted')")
        conn.execute("DETACH DATABASE encrypted")
        conn.commit()
    finally:
        conn.close()
    return memory_count, table_counts


def _export_encrypted_to_plaintext(enc_db: Path, plain_db: Path, key: str) -> int:
    """Decrypt an encrypted DB to a plaintext copy. Returns the memory count.

    Opens via ``_open_encrypted`` so a wrong key raises the same clear
    EncryptionError as ``connect_encrypted`` rather than a raw driver error.
    """
    conn = _open_encrypted(enc_db, key, wal=False)
    try:
        rows = _memory_count(conn)
        conn.execute("ATTACH DATABASE ? AS plaintext KEY ''", (str(plain_db),))
        conn.execute("SELECT sqlcipher_export('plaintext')")
        conn.execute("DETACH DATABASE plaintext")
        conn.commit()
    finally:
        conn.close()
    return rows


def _create_empty_encrypted(db_path: Path, key: str) -> None:
    dbapi2 = _import_sqlcipher()
    conn = dbapi2.connect(str(db_path), timeout=BUSY_TIMEOUT_MS / 1000)
    try:
        conn.execute(_key_pragma(key))
        # A real page write forces the encrypted header to be laid down.
        conn.execute("CREATE TABLE IF NOT EXISTS _poppy_init (x)")
        conn.execute("DROP TABLE _poppy_init")
        conn.commit()
    finally:
        conn.close()


# --- residue + locking ------------------------------------------------------


def find_residue(poppy_dir: Path) -> list[Path]:
    """Leftover migration temp files (a killed migration can strand these).

    ``.plain-tmp*`` is a decrypted copy of the store, so leaving it around
    defeats encryption at rest; callers surface and remove it. The ``-journal``
    tail is included because a SIGKILL mid-export can strand a rollback journal
    beside the temp DB.
    """
    db = _db_path(poppy_dir)
    found: list[Path] = []
    for suffix in _TMP_SUFFIXES:
        for tail in ("", "-wal", "-shm", "-journal"):
            p = db.with_name(db.name + suffix + tail)
            if p.exists():
                found.append(p)
    return found


def cleanup_residue(poppy_dir: Path) -> list[Path]:
    """Remove leftover migration temp files. Returns what was removed."""
    removed: list[Path] = []
    for p in find_residue(poppy_dir):
        try:
            p.unlink()
            removed.append(p)
        except OSError:
            pass
    return removed


# The SH/EX gate in poppy.db is the single correctness mechanism. Migration
# holds ``db.exclusive_gate`` (LOCK_EX), which cannot be acquired while any
# connection holds LOCK_SH -- so an in-flight writer (CLI, MCP, UI, sync, hooks)
# blocks migration by construction, with no process enumeration or heuristics.
# The poppy.writers registry survives only to NAME live surfaces in the refusal
# message (see poppy.db._migration_refusal), never as the gate.


# --- enable / disable / repair ----------------------------------------------


def _unverified_key_message(detail: str) -> str:
    """Refusal copy for a keychain this session cannot read the key back from."""
    return (
        f"the encryption key could not be verified in the OS keychain: {detail}. Nothing was encrypted: a "
        "store written under a key this session cannot read back may be impossible to open again. "
        "Background sessions (launchd agents, cron jobs, background shells) are often refused keychain "
        "reads, so run `poppy encrypt enable` from an interactive login session, or set the "
        f"{ENV_KEY} environment variable to a 64-hex-character key instead, and keep it safe: "
        "without it the encrypted store cannot be opened."
    )


def _verify_key_readback(poppy_dir: Path, key: str) -> None:
    """Prove this session reads the just-written key back, byte-for-byte."""
    try:
        stored = keychain.get_secret(_key_account(poppy_dir))
    except EncryptionError as exc:
        raise EncryptionError(
            _unverified_key_message(f"the write succeeded but reading it back failed ({exc})")
        ) from exc
    if stored is None:
        raise EncryptionError(_unverified_key_message("the write succeeded but the key read back as empty"))
    if _normalize_key(stored) != key:
        raise EncryptionError(_unverified_key_message("the write succeeded but the key read back as a different value"))


def _prepare_key_for_enable(poppy_dir: Path) -> str:
    """Resolve the key to encrypt with, validating any pre-existing keychain value."""
    env_key = _env_key()
    if env_key:
        return env_key
    if not keychain.available():
        raise EncryptionError(
            "no OS keychain backend is available to store the key. On a headless host, set the "
            f"{ENV_KEY} environment variable to a 64-hex-character key instead, and keep it safe: "
            "without it the encrypted store cannot be opened."
        )
    # A backend that loads is not a backend this session may use: macOS hands an
    # ssh / launchd / background session the real Keychain and then refuses every
    # call with -25308. Probe before touching the real key, so the refusal is this
    # sentence and not a raw Security error code from mid-migration.
    if not keychain.writable():
        raise EncryptionError(
            _unverified_key_message(
                "a probe entry could not be written to it and read back, so the login keychain is locked "
                "or this session has no interactive access to it"
            )
        )
    try:
        existing = keychain.get_secret(_key_account(poppy_dir))
    except EncryptionError as exc:
        raise EncryptionError(_unverified_key_message(f"reading it failed ({exc})")) from exc
    if existing is not None:
        key = _normalize_key(existing)
        if key is None:
            raise EncryptionError(
                "the encryption key stored in the OS keychain for this store is malformed (not 64 hex "
                "characters). Remove or fix that keychain entry before enabling encryption, so the store "
                "is not encrypted under a key Poppy cannot resolve."
            )
    else:
        key = generate_key()
    # Persist the canonical form so every later open matches byte-for-byte.
    try:
        keychain.set_secret(_key_account(poppy_dir), key)
    except EncryptionError as exc:
        raise EncryptionError(_unverified_key_message(f"writing it failed ({exc})")) from exc
    # Read-back gate: only treat the store as keychain-backed once this process
    # reads the exact value again. A backend can accept the write and still deny
    # the read (a macOS Background session gets -25308), which would otherwise
    # leave a store encrypted under a key nobody here can resolve.
    try:
        _verify_key_readback(poppy_dir, key)
    except EncryptionError:
        # A key generated in this run protects nothing once the store stays
        # plaintext, so drop the entry instead of leaving an orphan behind. A
        # reused entry stays: it may be the only copy of an existing key.
        if existing is None:
            try:
                keychain.delete_secret(_key_account(poppy_dir))
            except Exception:  # noqa: BLE001 - cleanup is best effort by design
                pass
        raise
    return key


def enable(poppy_dir: Path) -> EnableResult:
    """Encrypt the store at ``poppy_dir``, migrating any existing plaintext data.

    Refuses (with a clear message, regardless of any caller ``--yes``) when the
    store is already encrypted, in an inconsistent state, or has live writers.
    The whole operation is under an exclusive migration lock, and migrated data
    is verified table-by-table before and after the atomic file swap.
    """
    _require_deps("encryption")
    ensure_poppy_dir(poppy_dir)
    db = _db_path(poppy_dir)
    # The exclusive gate proves no Poppy connection is open anywhere; acquiring it
    # is the live-writer refusal (it cannot be skipped by --yes).
    with exclusive_gate(poppy_dir):
        cleanup_residue(poppy_dir)
        state = store_state(poppy_dir)
        if state in ("encrypted", "encrypted-no-sentinel"):
            # store_state verified it opens under the key; self-heal any missing
            # marker, then report it as already encrypted.
            _sentinel(poppy_dir).touch()
            _set_config_flag(poppy_dir, True)
            raise EncryptionError("this store is already encrypted.")
        if state == "plaintext-stale-sentinel":
            raise EncryptionError(_inconsistent_message(state, poppy_dir))
        if state == "corrupt":
            raise EncryptionError(_corrupt_message(poppy_dir))
        if state == "unreadable":
            raise StorePermissionError(_unreadable_message(poppy_dir))
        if state == "unknown":
            raise EncryptionError(_unverifiable_message(poppy_dir))

        key = _prepare_key_for_enable(poppy_dir)

        # Fresh or empty store: nothing to migrate, just create it encrypted.
        if state in ("absent", "pending-encrypted"):
            _remove_sidecars(db)
            _create_empty_encrypted(db, key)
            _sentinel(poppy_dir).touch()
            _set_config_flag(poppy_dir, True)
            return EnableResult(migrated_rows=0, created_empty=True)

        _checkpoint_or_abort(db, key=None)  # folds the WAL in; defense-in-depth busy check
        tmp = db.with_name(db.name + ".enc-tmp")
        _remove_sidecars(tmp)
        tmp.unlink(missing_ok=True)
        try:
            source_memories, source_counts = _export_plaintext_to_encrypted(db, tmp, key)
            verify = _open_encrypted(tmp, key, wal=False, immutable=True)
            try:
                copied_counts = _table_counts(verify)
            finally:
                verify.close()
            if copied_counts != source_counts:
                raise EncryptionError(
                    "migration verification failed: the encrypted copy does not match the plaintext store "
                    f"(source {source_counts}, copy {copied_counts}). The original store is untouched."
                )
            # Remove the plaintext source's sidecars BEFORE the swap, so the
            # encrypted file that lands can never coexist with an opposite-cipher
            # -wal/-shm that a later open would replay and destroy.
            _remove_sidecars(db)
            os.replace(tmp, db)
        except BaseException:
            _remove_sidecars(tmp)
            tmp.unlink(missing_ok=True)
            raise

        # db is now encrypted with no sidecars beside it. The only crash window is
        # "encrypted-no-sentinel" (self-healed on open, reconciled by repair);
        # mark it immediately to shrink that window.
        _sentinel(poppy_dir).touch()
        _set_config_flag(poppy_dir, True)
        # Read-only (immutable) re-verify: never mutates the just-swapped store.
        check = _open_encrypted(db, key, wal=False, immutable=True)
        try:
            final_counts = _table_counts(check)
        finally:
            check.close()
        if final_counts != source_counts:
            raise EncryptionError(
                "post-migration verification failed: the encrypted store does not match what was copied "
                f"(expected {source_counts}, got {final_counts}). Check the store with `poppy encrypt status`."
            )
        return EnableResult(migrated_rows=source_memories, created_empty=False)


def disable(poppy_dir: Path) -> int:
    """Decrypt the store back to plaintext and remove the key. Returns row count.

    Same guards as ``enable``: migration lock, inconsistency refusal, and a
    live-writer refusal that ``--yes`` cannot skip.
    """
    _require_deps("decryption")
    db = _db_path(poppy_dir)
    with exclusive_gate(poppy_dir):
        cleanup_residue(poppy_dir)
        state = store_state(poppy_dir)
        if state in ("plaintext", "absent"):
            raise EncryptionError("this store is not encrypted.")
        if state == "plaintext-stale-sentinel":
            raise EncryptionError(_inconsistent_message(state, poppy_dir))
        if state == "corrupt":
            raise EncryptionError(_corrupt_message(poppy_dir))
        if state == "unreadable":
            raise StorePermissionError(_unreadable_message(poppy_dir))
        if state == "unknown":
            raise EncryptionError(_unverifiable_message(poppy_dir))

        # A missing or unreachable backend raises KeychainUnavailable (an
        # EncryptionError) out of resolve_key; that is still "no key here", so it
        # opens with the same line recall and remember print instead of the raw
        # keyring install text. What the keychain actually said is appended, so a
        # merely locked keychain is not misread as a store encrypted elsewhere.
        keychain_error: KeychainUnavailable | None = None
        try:
            key = resolve_key(poppy_dir)
        except KeychainUnavailable as exc:
            key = None
            keychain_error = exc
        if key is None:
            message = _no_key_message(db)
            # Only where a backend really is present: with none at all, the error
            # is the keyring library's "install keyrings.alt" advice, which is the
            # text we want to suppress, and _no_key_message already says
            # everything true about that case.
            if keychain_error is not None and keychain.available():
                message += (
                    f" The OS keychain itself could not be reached in this session ({keychain_error}); if it is "
                    "only locked, the key may still be in it and readable from an interactive login session."
                )
            raise EncryptionError(message)

        rows = 0
        if state in ("encrypted", "encrypted-no-sentinel"):
            _checkpoint_or_abort(db, key=key)  # folds the WAL in; defense-in-depth busy check
            tmp = db.with_name(db.name + ".plain-tmp")
            _remove_sidecars(tmp)
            tmp.unlink(missing_ok=True)
            try:
                rows = _export_encrypted_to_plaintext(db, tmp, key)
                # Remove the encrypted source's sidecars BEFORE the swap, so the
                # plaintext file that lands never coexists with an encrypted -wal.
                _remove_sidecars(db)
                os.replace(tmp, db)
            except BaseException:
                _remove_sidecars(tmp)
                tmp.unlink(missing_ok=True)
                raise
        else:  # pending-encrypted: sentinel only, no data file
            _remove_sidecars(db)
            if db.exists():
                db.unlink()

        # Crash-state machine: after the swap the store is plaintext. Drop the
        # key BEFORE the sentinel, so any crash window is "plaintext-stale-sentinel"
        # (repair reconciles it and also deletes the key). The key is dropped
        # unconditionally, even with POPPY_DB_KEY set, so a disable/enable cycle
        # rotates it (the CLI promises this).
        try:
            keychain.delete_secret(_key_account(poppy_dir))
        except KeychainUnavailable:
            pass
        _sentinel(poppy_dir).unlink(missing_ok=True)
        _set_config_flag(poppy_dir, False)
        return rows


def _opens_as_plaintext(db_path: Path) -> bool:
    """Whether the store actually opens as a plaintext SQLite DB (read-only probe).

    Uses ``immutable=1`` so it never replays a sidecar or mutates the file. Repair
    requires this proof before deleting a key: a merely magic-looking header is not
    enough, so an unreadable/corrupt file can never trigger a key-deleting lockout.
    """
    try:
        conn = sqlite3.connect(_immutable_uri(db_path), uri=True, timeout=BUSY_TIMEOUT_MS / 1000)
    except Exception:
        return False
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def _working_key(poppy_dir: Path) -> str | None:
    """The first resolvable key that opens the store read-only, or None."""
    for key in _candidate_keys(poppy_dir):
        try:
            conn = _open_encrypted(_db_path(poppy_dir), key, wal=False, immutable=True)
        except EncryptionError:
            continue
        conn.close()
        return key
    return None


def _test_open_checkpoint(copy_db: Path, mode: str, key: str | None) -> str:
    """Open + checkpoint + read a writable COPY. Returns ``ok`` | ``cipher`` | ``io``.

    ``ok``: the copy replayed its ``-wal`` cleanly (same cipher). ``cipher``: a
    clean cipher/format failure (wrong-cipher or corrupt under this mode). ``io``:
    a permission / read-only / resource error -- crucially NEVER conflated with
    ``cipher``, so a transient failure can never brand a legitimate wal as wrong.
    """
    try:
        if mode == "plaintext":
            c = sqlite3.connect(str(copy_db), timeout=BUSY_TIMEOUT_MS / 1000)
        else:
            c = _open_encrypted(copy_db, key, wal=False)  # StorePermissionError on IO, EncryptionError on cipher
    except StorePermissionError:
        return "io"
    except EncryptionError:
        return "cipher"
    try:
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except Exception as exc:
        return "io" if _is_permission_error(exc) else "cipher"
    finally:
        c.close()
    return "ok"


def _copy_store_for_test(db: Path, dest_dir: Path) -> Path:
    """Content-copy main+wal+shm into ``dest_dir`` with mode 0o600 (writable).

    Copies CONTENT, not permission bits: a read-only (0o444) main must not make
    the test copy's checkpoint fail and get mislabelled a cipher mismatch.
    """
    cp = dest_dir / db.name
    for suf in ("", "-wal", "-shm"):
        src = db.with_name(db.name + suf)
        if src.exists():
            dst = cp.with_name(cp.name + suf)
            dst.write_bytes(src.read_bytes())
            os.chmod(dst, 0o600)
    return cp


def _main_opens(db: Path, mode: str, key: str | None) -> bool:
    """Whether the MAIN file alone (ignoring the wal, read-only) opens cleanly."""
    if mode == "plaintext":
        return _opens_as_plaintext(db)
    try:
        c = _open_encrypted(db, key, wal=False, immutable=True)
    except EncryptionError:
        return False
    c.close()
    return True


def _reconcile_wal(poppy_dir: Path, mode: str, key: str | None) -> tuple[str, str | None]:
    """Under the EX gate, reconcile a pending ``-wal`` without ever destroying unrecoverable data.

    Returns ``(verdict, note)`` where verdict is ``none`` | ``reconciled`` |
    ``undetermined``. Invariant: repair never destroys data it cannot PROVE is
    disposable.

    * same cipher (proven on a WRITABLE copy) -> replay into the real store; if the
      real store is itself read-only, PRESERVE (leave the wal) rather than lose it.
    * not recoverable under the key AND the main file alone opens cleanly (its
      committed data is safe) -> set the wal aside (kept, never deleted).
    * any IO/permission/resource failure, or a main that also will not open ->
      ``undetermined``: mutate nothing, leave the wal in place.
    """
    import tempfile  # noqa: PLC0415

    from poppy.db import has_nonempty_wal  # noqa: PLC0415

    db = _db_path(poppy_dir)
    if not has_nonempty_wal(db):
        return "none", None
    try:
        with tempfile.TemporaryDirectory() as td:
            cp = _copy_store_for_test(db, Path(td))
            verdict = _test_open_checkpoint(cp, mode, key)
    except OSError as exc:
        return "undetermined", (
            f"could not verify the pending -wal ({exc}); it was left untouched. Free disk space or fix "
            "permissions and re-run. Your committed data is preserved in the -wal."
        )

    if verdict == "io":
        return "undetermined", (
            "could not verify the pending -wal (the store directory may be read-only or out of space); it was "
            "left untouched. Fix that and re-run. Your committed data is preserved in the -wal."
        )
    if verdict == "ok":  # the copy replayed cleanly -> the wal is the store's cipher
        # Clear any stale -shm (e.g. one created read-only by a prior failed
        # attempt) so it cannot block this replay; SQLite rebuilds it from the -wal.
        _remove_shm(db)
        try:
            if mode == "plaintext":
                c = sqlite3.connect(str(db), timeout=BUSY_TIMEOUT_MS / 1000)
            else:
                c = _open_encrypted(db, key, wal=False)
            try:
                c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                c.close()
        except Exception as exc:
            _remove_shm(db)  # do not leave this attempt's -shm to block a retry
            return "undetermined", (
                f"the pending -wal is valid but could not be replayed ({exc}); it was left untouched and your "
                "committed data is preserved in the -wal. Fix permissions/space and re-run."
            )
        _remove_sidecars(db)
        return "reconciled", "replayed the pending -wal into the store (committed data preserved)"
    # verdict == "cipher": the wal does not open under the key. Set it aside ONLY if
    # the main alone is provably intact, so nothing recoverable is discarded.
    if _main_opens(db, mode, key):
        moved = quarantine_sidecars(db)
        if moved:
            return "reconciled", (
                f"set aside an unrecoverable -wal that does not match the store's cipher (kept as "
                f"{', '.join(p.name for p in moved)}, not deleted); the store's committed data is intact in the "
                "main file."
            )
        return "reconciled", None
    return "undetermined", (
        "the pending -wal does not match the store and the store could not be independently verified; nothing "
        "was changed. Inspect the files manually before retrying."
    )


def _safe_mutation(actions: list[str], fn, label: str) -> bool:
    """Run a repair mutation, converting an OSError/ValueError into a reported note."""
    try:
        fn()
        return True
    except (OSError, ValueError) as exc:  # PermissionError (RO dir), JSONDecodeError (torn config)
        actions.append(f"could not {label}: {exc}. Fix permissions / config.json and re-run.")
        return False


def repair(poppy_dir: Path) -> list[str]:
    """Reconcile an inconsistent encryption state under the exclusive gate.

    The verified on-disk store is the source of truth, and reconciliation happens
    only here (never on the SH connect path), so quarantining/replaying a sidecar
    cannot race a live connection:

    * ``plaintext-stale-sentinel``: replay a same-cipher pending wal (or set aside
      a wrong-cipher one), remove the marker, delete the orphaned keychain key.
    * ``encrypted-no-sentinel``: replay a same-cipher pending wal, recreate the
      marker (key kept).
    * ``corrupt`` / ``unknown`` / ``unreadable``: never manufacture a marker;
      report honestly.

    Mutations (marker, config flag) are guarded so a read-only dir or torn
    config.json yields a clear note, not a raw traceback. Returns what changed.
    """
    actions: list[str] = []
    with exclusive_gate(poppy_dir):
        state = store_state(poppy_dir)
        if state == "plaintext-stale-sentinel":
            verdict, note = _reconcile_wal(poppy_dir, "plaintext", None)
            if note:
                actions.append(note)
            if verdict == "undetermined":
                actions.append("left the marker and key in place until the pending -wal is resolved.")
            else:
                _safe_mutation(
                    actions, lambda: _sentinel(poppy_dir).unlink(missing_ok=True), "remove the encryption marker"
                )
                # Only delete the key once the store is PROVEN to open as plaintext,
                # so an unreadable/corrupt file can never trigger a key-deleting lockout.
                if _opens_as_plaintext(_db_path(poppy_dir)):
                    try:
                        keychain.delete_secret(_key_account(poppy_dir))
                    except KeychainUnavailable:
                        pass
                    actions.append(
                        "removed a stale encryption marker and the orphaned keychain key; the store is plaintext"
                    )
                else:
                    actions.append(
                        "removed a stale encryption marker but KEPT the keychain key: the store did not open as "
                        "plaintext, so the key was not deleted (avoiding a possible lockout). Check the store."
                    )
                _safe_mutation(actions, lambda: _set_config_flag(poppy_dir, False), "update config.json")
        elif state == "encrypted-no-sentinel":
            verdict, note = _reconcile_wal(poppy_dir, "encrypted", _working_key(poppy_dir))
            if note:
                actions.append(note)
            if verdict == "undetermined":
                actions.append("left the marker in place until the pending -wal is resolved.")
            else:
                healed = _safe_mutation(actions, lambda: _sentinel(poppy_dir).touch(), "recreate the encryption marker")
                _safe_mutation(actions, lambda: _set_config_flag(poppy_dir, True), "update config.json")
                if healed:
                    actions.append("recreated the missing encryption marker; the store opens under the stored key")
        elif state == "corrupt":
            actions.append(
                "the store file is not a valid encrypted Poppy store (it does not open under any resolved "
                f"key). If a stale {ENV_KEY} is shadowing the real key, unset it and re-run; otherwise it may "
                "be corrupt. No marker created."
            )
        elif state == "unreadable":
            actions.append(
                "the store could not be read (permission or read-only filesystem), so its state cannot be "
                "verified. Fix the permissions and re-run. No marker created."
            )
        elif state == "unknown":
            actions.append(
                "cannot verify the store without the encryption extra and key; install the extra and "
                f"provide the key (keychain or {ENV_KEY}), then re-run. No marker created."
            )
        for p in cleanup_residue(poppy_dir):
            sensitive = ".plain-tmp" in p.name
            actions.append(f"removed leftover {'decrypted ' if sensitive else ''}temp file {p.name}")
        # Set-aside strays STAY on disk (that is what "set aside" means): repair
        # never unlinks a sidecar, so a misjudged wal can never be silently
        # destroyed. Report them so the user can inspect and delete manually.
        for p in stray_sidecars(_db_path(poppy_dir)):
            actions.append(f"a set-aside sidecar is kept for inspection: {p.name} (delete it if you do not need it)")
    if not actions:
        actions.append("nothing to repair; the encryption state is consistent")
    return actions


def _corrupt_message(poppy_dir: Path) -> str:
    stale = ""
    if os.environ.get(ENV_KEY):
        stale = (
            f" The resolved key may be wrong or stale: {ENV_KEY} is set and shadows the OS keychain, so if "
            "this store belongs to a different machine/account, unset it and retry before assuming corruption."
        )
    return (
        f"the store file at {_db_path(poppy_dir)} is not a plaintext SQLite database and does not open under "
        f"any resolved encryption key.{stale} It may otherwise be corrupt or a foreign file. Poppy will not "
        "encrypt over or decrypt it."
    )


def _unverifiable_message(poppy_dir: Path) -> str:
    return (
        f"the store file at {_db_path(poppy_dir)} is not a plaintext SQLite database, and its state cannot be "
        f"verified here: install the encryption extra and provide the key (keychain or {ENV_KEY}). If it is an "
        "encrypted Poppy store, do that and run `poppy encrypt repair`; otherwise it may be corrupt."
    )


def _unreadable_message(poppy_dir: Path) -> str:
    return (
        f"the store at {_db_path(poppy_dir)} could not be read: this looks like a permission or read-only "
        "filesystem problem, not corruption. Fix the permissions on the store and its directory, then retry."
    )


def open_without_sentinel(db_path: Path | str, *, check_same_thread: bool = False):
    """Open an encrypted store that lost its sentinel, for ``poppy.db.connect``.

    A read-only ``immutable`` probe (never replays a sidecar, never mutates)
    tells a healthy encrypted store that merely lost its marker from a corrupt /
    foreign / unreadable file. If a pending ``-wal`` is present the state is
    ambiguous (its cipher cannot be proven here without risking a destructive
    replay), so this REFUSES with a ``poppy encrypt repair`` hint rather than
    silently quarantining a possibly-legitimate wal. With no pending wal the real
    read-write open is safe; the marker is self-healed (best-effort, guarded).
    ``poppy.db.connect`` attaches the gate handle to the returned connection.
    """
    path = Path(db_path)
    poppy_dir = path.parent

    if not dependencies_installed():
        raise EncryptionError(_unverifiable_message(poppy_dir))
    keys = _candidate_keys(poppy_dir)
    if not keys:
        raise EncryptionError(_unverifiable_message(poppy_dir))

    # Classify read-only first (immutable never replays a sidecar, never mutates).
    working_key = None
    for key in keys:
        try:
            probe = _open_encrypted(path, key, wal=False, immutable=True)
        except StorePermissionError as exc:
            raise StorePermissionError(_unreadable_message(poppy_dir)) from exc
        except EncryptionError:
            continue
        probe.close()
        working_key = key
        break
    if working_key is None:
        raise EncryptionError(_corrupt_message(poppy_dir))

    from poppy.db import has_nonempty_wal  # noqa: PLC0415

    if has_nonempty_wal(path):
        raise EncryptionError(
            f"the store at {path} is encrypted but its marker is missing and a pending -wal is present. "
            "Run `poppy encrypt repair` to reconcile it safely (it decides under an exclusive lock whether "
            "to replay or set aside the -wal); recall/remember are refused until then to avoid data loss."
        )

    # Healthy encrypted store, marker missing, no pending wal: safe to open rw.
    factory = _gated_sqlcipher_cls()
    conn = _open_encrypted(path, working_key, wal=True, check_same_thread=check_same_thread, factory=factory)
    try:
        _sentinel(poppy_dir).touch()
        _set_config_flag(poppy_dir, True)
    except Exception:
        # Self-heal is best-effort: a torn config.json (JSONDecodeError) or an
        # unwritable dir must not fail an otherwise healthy open or leak the conn.
        pass
    return conn


# --- config flag (user-facing intent marker; connect() does not depend on it) --


def _set_config_flag(poppy_dir: Path, value: bool) -> None:
    from poppy.config import load_config, save_config  # noqa: PLC0415

    cfg = load_config(poppy_dir)
    cfg.encrypted = value
    save_config(cfg)


def _keychain_state(env_key_active: bool, deps: bool) -> str:
    """Observe the keychain for status, without paying for it when it is unused.

    The probe is a write + read + delete against the OS credential store. On a
    locked Secret Service that is an unlock prompt, and a second one for the
    delete, so it is only worth it where the answer is: when the keychain is
    where this session's key comes from. With POPPY_DB_KEY set the key is
    already in hand, the keychain is never consulted for it, and the CLI does
    not print this field, so status stays the zero-IPC call it has always been.
    """
    if not deps or env_key_active:
        return "not-probed"
    if not keychain.available():
        return "unavailable"
    return "usable" if keychain.writable() else "refused"


def status(poppy_dir: Path, *, deep: bool = False) -> EncryptionStatus:
    db = _db_path(poppy_dir)
    deps = dependencies_installed()
    kc = _keychain_state(os.environ.get(ENV_KEY) is not None, deps)
    state = store_state(poppy_dir)
    # store_state trusts an intact sentinel for cheapness. ``deep`` actually
    # opens the store (read-only) so bit-rot behind a healthy-looking sentinel is
    # not reported as fine.
    if deep and state == "encrypted" and deps:
        verdict = _probe_open(poppy_dir)
        if verdict != "encrypted":
            state = verdict
    env_raw = os.environ.get(ENV_KEY)
    env_key_active = env_raw is not None
    env_key_malformed = env_key_active and _normalize_key(env_raw) is None
    key_present = False
    if deps:
        try:
            key_present = resolve_key(poppy_dir) is not None
        except EncryptionError:
            key_present = False  # e.g. a malformed POPPY_DB_KEY, flagged separately
    # Report verified truth, not the header alone: a corrupt/unknown file is not
    # "encrypted", so it must not show as enabled/encrypted-on-disk.
    encrypted_states = ("encrypted", "encrypted-no-sentinel")
    return EncryptionStatus(
        enabled=state in encrypted_states + ("pending-encrypted",),
        encrypted_on_disk=state in encrypted_states,
        deps_installed=deps,
        keychain_state=kc,
        key_present=key_present,
        db_path=db,
        env_key_active=env_key_active,
        state=state,
        inconsistent=state in _INCONSISTENT,
        env_key_malformed=env_key_malformed,
        residue=find_residue(poppy_dir),
        stray=stray_sidecars(db),
    )
