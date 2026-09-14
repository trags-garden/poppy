"""Shared SQLite connection setup for Poppy's local store.

Poppy opens the same ``~/.poppy`` database from several places that can run at
the same time — the long-lived MCP server, interactive CLI commands, and the
background auto-sync that fires after each write. With SQLite's default
rollback-journal mode a writer takes an exclusive lock that blocks every other
connection, and a contended connection that needs to upgrade its lock gets an
immediate ``sqlite3.OperationalError: database is locked`` (the busy handler is
deliberately skipped in those deadlock-prone cases). The result was spurious
lock errors during normal MCP + CLI + sync overlap.

WAL mode lets a writer and any number of readers proceed concurrently, and an
explicit busy timeout makes a genuinely contended writer wait instead of
failing instantly. ``synchronous = NORMAL`` is the recommended durability level
under WAL (safe across application crashes; only an OS/power loss can lose the
last commit, which is acceptable for a local cache that also syncs to Trags).

Encryption gate. Every connection obtained here first takes a shared (``LOCK_SH``)
advisory lock on a permanent ``db.gate`` file and holds it for the connection's
lifetime (released on ``close``). ``poppy encrypt`` migration takes an exclusive
(``LOCK_EX``) lock on the same file. The kernel does the exclusion: an in-flight
connection (held ``LOCK_SH``) blocks a migration, and a migration (held
``LOCK_EX``) blocks new connections. This is constructive — no process
enumeration, no liveness heuristics — and covers every writer (CLI, MCP, UI,
sync, hooks) automatically. The plaintext path never imports the encryption
stack.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import weakref
from contextlib import contextmanager, suppress
from pathlib import Path

from poppy.errors import EncryptionError, StorePermissionError
from poppy.paths import ensure_poppy_dir

# Comfortably longer than any single Poppy write/commit, while still surfacing a
# real deadlock rather than hanging forever.
BUSY_TIMEOUT_MS = 5000

# The 16-byte header every plaintext SQLite file starts with. A SQLCipher file
# has an encrypted header (random salt) and therefore never matches, which lets
# ``connect`` tell an encrypted store from a plaintext one without any config.
SQLITE_MAGIC = b"SQLite format 3\x00"

# Marker file written next to the DB when encryption is enabled. It only decides
# routing for a store whose DB file does not exist yet (a fresh or empty store);
# once the file exists, its on-disk header is the source of truth.
ENCRYPTION_SENTINEL = ".poppy-encrypted"

# The SH/EX advisory gate. Permanent (created on first connect), so the shared
# probe on the hot path is a bare open + flock, with no encryption import.
GATE_FILENAME = "db.gate"

# A connection waits up to ~10s for an in-flight migration to release the gate
# before giving up; a migration waits ~5s for connections to drain.
_GATE_SHARED_ATTEMPTS = 100
_GATE_SHARED_SLEEP_S = 0.1
_GATE_EXCLUSIVE_ATTEMPTS = 50
_GATE_EXCLUSIVE_SLEEP_S = 0.1

# The write gate (below) is a different lock from the encryption gate: it
# serializes multi-statement local write sequences, not connections.
WRITE_GATE_FILENAME = "write.gate"
_WRITE_GATE_ATTEMPTS = 100
_WRITE_GATE_SLEEP_S = 0.1


def _fcntl():
    try:
        import fcntl  # noqa: PLC0415

        return fcntl
    except ImportError:  # pragma: no cover - Windows only
        return None


class _GateHandle:
    """Owns a gate fd and a store refcount, released exactly once (idempotent).

    Attached to a connection and released on ``close``; a weakref finalizer
    releases it too, so a connection that is GC'd or leaked on an error path
    (never explicitly closed) does not hold the gate open until process exit.
    """

    __slots__ = ("fd", "key")

    def __init__(self, fd: int | None, key: tuple[int, int] | None = None):
        self.fd = fd
        self.key = key

    def release(self) -> None:
        fd = self.fd
        if fd is not None:
            self.fd = None  # null first, so a double release never closes a reused fd
            try:
                os.close(fd)
            except OSError:
                pass
        key = self.key
        if key is not None:
            self.key = None  # null first, so a double release never double-decrements
            _release_store(key)


def _attach_gate(conn, gate_fd: int | None, key: tuple[int, int] | None):
    """Give ``conn`` ownership of the gate fd and the store refcount.

    ``close`` is the primary release. The weakref finalizer is a safety net for a
    caller that leaks a connection without closing it; it fires reliably for a
    plaintext connection but a leaked SQLCipher connection can lag a gc cycle
    before collection, so its gate fd may stay held slightly longer (fail-safe:
    it only over-blocks a migration transiently, never loses data, and the kernel
    releases at process exit). Internal callers always use ``close``/context
    managers, so this only affects a caller that forgets to close."""
    if gate_fd is None and key is None:
        return conn
    handle = _GateHandle(gate_fd, key)
    conn._poppy_gate = handle
    try:
        weakref.finalize(conn, handle.release)
    except TypeError:  # pragma: no cover - connection type without weakref support
        pass
    return conn


class _GatedConnection(sqlite3.Connection):
    """A plaintext connection that releases its gate lock when closed."""

    def close(self) -> None:
        handle = getattr(self, "_poppy_gate", None)
        try:
            super().close()
        finally:
            if handle is not None:
                handle.release()


def apply_row_factory(conn: sqlite3.Connection) -> None:
    """Set the driver-appropriate ``Row`` factory on ``conn``.

    Engines want dict-and-index rows (``row["id"]`` and ``row[0]`` both). The
    stdlib ``sqlite3.Row`` C type rejects a SQLCipher cursor and vice versa, so
    an encrypted connection needs ``sqlcipher3``'s own ``Row``. A plaintext
    connection (incl. the gated subclass) is a ``sqlite3.Connection``; a
    SQLCipher one is not, which is the reliable discriminator.
    """
    if isinstance(conn, sqlite3.Connection):
        conn.row_factory = sqlite3.Row
    else:
        from sqlcipher3 import dbapi2  # noqa: PLC0415

        conn.row_factory = dbapi2.Row


def rollback_and_close(conn: sqlite3.Connection) -> None:
    """Best-effort cleanup for a connection whose owner failed to initialize.

    Cleanup failures must never replace the construction error already in
    flight. Rollback is explicit even though close also rolls back an open
    transaction, so the lock-release contract does not depend on driver
    shutdown behavior.
    """
    with suppress(Exception):
        conn.rollback()
    with suppress(Exception):
        conn.close()


@contextmanager
def write_txn(conn: sqlite3.Connection):
    """Run a multi-statement write sequence as one transaction, or not at all.

    Two guarantees, and both are load-bearing for a store several processes
    share:

    * ``BEGIN IMMEDIATE`` claims SQLite's write lock UP FRONT, so a whole
      read-decide-write sequence is atomic against another process. SQLite
      would only take the lock at the first actual write, leaving a window in
      which another writer can change what the decision was made on — a schema
      probe, an enumeration of rows about to be deleted, an expiry check.
    * ANY exception rolls the sequence back. A failure part-way through a write
      sequence used to leave the transaction OPEN on a long-lived connection
      (the MCP server holds one for its whole lifetime): every other process
      then got ``database is locked``, and the next successful write on that
      connection committed whatever the failed one had already written.
      Rollback on ``BaseException``, not ``Exception``, because a
      ``KeyboardInterrupt`` mid-sequence must not commit a half-write either.

    No-ops when a transaction is already open, leaving commit/rollback to
    whoever began it — so these can nest, and a helper that claims the lock for
    itself when called alone (``delete_marked_closets``) cooperates when called
    from inside one.

    That branch is for SAME-THREAD nesting only, and every caller must hold its
    engine's lock for the length of the block. Two threads sharing one
    connection would otherwise mistake each other for nested callers: the second
    thread's write joins the first's transaction, reports success, and is rolled
    back when the FIRST one fails. Both engines take their own lock around this
    (``seed.py``, ``_closet_engine.py``); a new caller must do the same.
    """
    own_txn = not conn.in_transaction
    if own_txn:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except Exception as exc:
            # A connection shared by threads can open a transaction between the
            # check above and this statement. That transaction protects the
            # sequence just as well, so THAT refusal is not an error; anything
            # else — "database is locked" after the busy timeout above all —
            # must abort rather than run the sequence unguarded. Matched on the
            # message rather than the exception class because the SQLCipher
            # driver raises its own ``OperationalError``, which is not a
            # subclass of the stdlib one.
            if "within a transaction" not in str(exc):
                raise
            own_txn = False
    try:
        yield
    except BaseException:
        if own_txn:
            # Suppressed: a rollback that itself fails must not REPLACE the error
            # that is already in flight, which is the one the caller can act on.
            # Closing the connection rolls back too, and the commit-failure path
            # below treats its own rollback the same way.
            with suppress(Exception):
                conn.rollback()
        raise
    if own_txn:
        try:
            conn.commit()
        except BaseException:
            # A commit that fails (a contended write lock, a full disk) leaves
            # the transaction open just as surely as a failing statement does.
            with suppress(Exception):
                conn.rollback()
            raise


def _apply_pragmas(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


# --- the header sniff, and why it may not touch a live store ----------------
#
# Reading the header opens and closes a descriptor on the database file. POSIX
# advisory locks are per (process, inode) and are dropped the moment the process
# closes ANY descriptor on that inode -- so a sniff taken while this process
# already holds a Poppy connection silently releases SQLite's shared lock on the
# store, even though that other connection is still very much open. The next
# process to close its last connection then finds the database unlocked, decides
# it is the last user, checkpoints, and unlinks -wal/-shm out from under our live
# readers. They keep the now-unlinked wal-index mapped; as soon as any writer
# creates a fresh -wal/-shm pair, that mapping no longer describes the file and
# every read fails with "disk I/O error" / "database disk image is malformed" for
# the rest of the process's life (e.g. one `poppy list` next to `poppy ui`).
#
# Two things keep the sniff off a live store, and BOTH are needed.
#
# 1. The result is remembered per store inode for as long as this process holds a
#    connection to that store: exactly the window in which sniffing is unsafe, and
#    in which the answer provably cannot change (the header class only changes
#    under an encryption migration, which takes the exclusive gate and therefore
#    runs with zero connections open). With no connection open the header is read
#    fresh, so a store rewritten or replaced in place is still classified from its
#    real bytes.
#
# 2. Classification, opening and registration run under ``_store_lock`` as one
#    step. The cache alone is not enough: from a cold cache two threads would both
#    sniff, and the slower one's close would land after the faster one's connection
#    was already open and holding locks -- unlocking a live store just as surely as
#    the single-threaded case. Serialising removes the window instead of narrowing
#    it, and the invariant becomes checkable: a descriptor is only ever opened on
#    the store while no connection exists and no other thread can be establishing
#    one. ``connect`` is called at engine construction, not per query, so
#    serialising it costs nothing worth measuring.
#
# The lock is reentrant because ``connect`` holds it across ``header_state``, and
# because a connection's finalizer (``_release_store``) can fire on any thread at
# any allocation, including one already inside ``connect``.
_open_stores: dict[tuple[int, int], int] = {}
_header_cache: dict[tuple[int, int], str] = {}
_store_lock = threading.RLock()


def _store_key(path: Path) -> tuple[int, int] | None:
    """Filesystem identity of the store, or ``None`` if it cannot be stat'd."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _register_store(path: Path, state: str) -> tuple[int, int] | None:
    """Count a freshly opened connection against the store, seeding the header cache.

    Called once the connection exists, so the file is there to stat even for a
    store that was ``absent`` at sniff time. Seeding the cache from the state
    ``connect`` actually resolved is what lets the SECOND connect in a process
    skip the sniff entirely rather than unlock the store.
    """
    key = _store_key(path)
    if key is None:  # pragma: no cover - the file exists, we just opened it
        return None
    with _store_lock:
        _open_stores[key] = _open_stores.get(key, 0) + 1
        _header_cache[key] = state
    return key


def _release_store(key: tuple[int, int]) -> None:
    """Drop one connection's count, forgetting the cached header at zero."""
    with _store_lock:
        remaining = _open_stores.get(key, 0) - 1
        if remaining > 0:
            _open_stores[key] = remaining
        else:
            _open_stores.pop(key, None)
            _header_cache.pop(key, None)


def header_state(path: Path) -> str:
    """Classify the DB file by its header.

    Returns ``absent`` (missing/empty), ``plaintext`` (SQLite magic), ``nonmagic``
    (a header that is not the SQLite magic, i.e. encrypted or corrupt), or
    ``unreadable`` (the header could not be read because of a permission / IO
    error). ``unreadable`` is distinct from ``plaintext`` so an unreadable
    ENCRYPTED file is never misclassified as plaintext, which would let repair
    delete its key (the lockout chain).

    While this process holds a connection to the store the answer comes from the
    cache rather than the file, because opening the file would unlock the store
    for every other connection -- see the note above ``_open_stores``. The cache
    lookup and the fallback read happen under ``_store_lock`` as one step, so a
    caller outside ``connect`` (``poppy.encryption`` probing a store) can never
    slip a sniff in beside a connection another thread is still establishing.
    """
    try:
        st = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return "absent"
    except OSError:
        return "unreadable"
    if st.st_size == 0:
        return "absent"
    key = (st.st_dev, st.st_ino)
    with _store_lock:
        if _open_stores.get(key):
            cached = _header_cache.get(key)
            if cached is not None:
                return cached
        try:
            with open(path, "rb") as f:
                return "plaintext" if f.read(16) == SQLITE_MAGIC else "nonmagic"
        except OSError:
            return "unreadable"


def has_nonempty_wal(db_path: Path) -> bool:
    """Whether a non-empty ``-wal`` sits beside the store (data awaiting replay)."""
    wal = db_path.with_name(db_path.name + "-wal")
    try:
        return wal.exists() and wal.stat().st_size > 0
    except OSError:
        return False


# --- the SH/EX gate ---------------------------------------------------------


def _open_gate_fd(poppy_dir: Path) -> int:
    """Open the gate file for flock, tolerating a read-only dir.

    Shared by the SH (connect) and EX (migration) paths so their permission
    handling can never diverge. Falls back to O_RDONLY (flock works on a
    read-only fd) and raises ``StorePermissionError`` -- never a raw
    ``PermissionError`` -- when even that fails.
    """
    try:
        ensure_poppy_dir(poppy_dir)
        gate = poppy_dir / GATE_FILENAME
        try:
            return os.open(str(gate), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return os.open(str(gate), os.O_RDONLY)
    except OSError as exc:
        raise StorePermissionError(
            f"could not access the store directory {poppy_dir}: {exc}. This looks like a permission or "
            "read-only filesystem problem, not corruption; fix the permissions and retry."
        ) from exc


def acquire_shared_gate(poppy_dir: Path) -> int | None:
    """Take a shared lock on the gate, held until the returned fd is closed.

    Returns the fd, or ``None`` where ``fcntl`` is unavailable (the gate cannot
    be enforced there; migration fails closed instead). Raises ``EncryptionError``
    if a migration holds the exclusive lock and does not release within the wait.
    """
    fcntl = _fcntl()
    if fcntl is None:  # pragma: no cover - Windows only
        return None
    fd = _open_gate_fd(poppy_dir)
    for _ in range(_GATE_SHARED_ATTEMPTS):
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return fd
        except OSError:
            time.sleep(_GATE_SHARED_SLEEP_S)
    os.close(fd)
    raise EncryptionError("a `poppy encrypt` migration is in progress on this store; retry in a moment.")


@contextmanager
def write_gate(poppy_dir: Path):
    """Serialize one multi-step local write sequence against other processes.

    SQLite makes single statements atomic, never a read-decide-write sequence.
    The UI restoring a memory reads its tombstone, checks for a live row and
    then ingests, while the autosync worker's pull can apply a server edit in
    between — the restore would overwrite that edit with the older snapshot.
    Holding this advisory lock for the length of such a sequence keeps them from
    interleaving. It is separate from the encryption gate and always taken
    first, so the two can never deadlock.

    Never wedges a write: if the lock is not free within roughly ten seconds (a
    stuck holder, or a filesystem without flock) the sequence runs anyway, and
    the callers' own compare-and-swap guards still stand.
    """
    fcntl = _fcntl()
    if fcntl is None:  # pragma: no cover - Windows only
        yield
        return
    try:
        ensure_poppy_dir(poppy_dir)
        fd = os.open(str(poppy_dir / WRITE_GATE_FILENAME), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:  # pragma: no cover - read-only store dir
        yield
        return
    held = False
    try:
        for _ in range(_WRITE_GATE_ATTEMPTS):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except OSError:
                time.sleep(_WRITE_GATE_SLEEP_S)
        yield
    finally:
        if held:
            with suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
        with suppress(OSError):
            os.close(fd)


def _release_gate(fd: int | None) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def exclusive_gate(poppy_dir: Path):
    """Context manager holding the exclusive gate for a migration.

    Acquiring it proves no Poppy connection is open anywhere (every connection
    holds ``LOCK_SH``). Fails closed with a clear message when ``fcntl`` is
    unavailable, and refuses (naming live surfaces if it can) when connections
    are open. Migration internals must open the DB directly, not via ``connect``,
    or they would deadlock against their own exclusive lock.
    """
    from contextlib import contextmanager  # noqa: PLC0415

    @contextmanager
    def _cm():
        fcntl = _fcntl()
        if fcntl is None:  # pragma: no cover - Windows only
            raise EncryptionError(
                "encryption migration needs file locking, which is unavailable on this platform; "
                "refusing to proceed rather than migrate without protection."
            )
        fd = _open_gate_fd(poppy_dir)
        acquired = False
        try:
            for _ in range(_GATE_EXCLUSIVE_ATTEMPTS):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    time.sleep(_GATE_EXCLUSIVE_SLEEP_S)
            if not acquired:
                raise EncryptionError(_migration_refusal(poppy_dir))
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)

    return _cm()


def _migration_refusal(poppy_dir: Path) -> str:
    """Best-effort refusal message that names live writers if the registry knows them."""
    names: list[str] = []
    try:
        from poppy import writers  # noqa: PLC0415

        names = writers.live_writers(poppy_dir)
    except Exception:
        names = []
    base = "refusing to migrate: other Poppy processes have the store open"
    if names:
        listed = "\n".join(f"  - {n}" for n in names)
        message = f"{base}:\n{listed}\nStop them and retry."
        if any("poppy daemon" in name for name in names):
            message += "\nFor the daemon: run `poppy daemon stop` first, then `poppy daemon start` after the migration."
        return message
    return (
        f"{base} (e.g. `poppy serve`, `poppy ui`, a sync/capture worker, or another CLI command). Stop them and retry."
    )


# --- sidecar quarantine (never replay a sidecar of the wrong cipher) --------

STRAY_SUFFIX = ".stray-"


def _unique_stray_dest(sidecar: Path) -> Path:
    """A non-colliding ``<name>.stray-<ts>-<n>`` destination (never overwrites)."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    for n in range(10000):
        dest = sidecar.with_name(f"{sidecar.name}{STRAY_SUFFIX}{ts}-{n}")
        if not dest.exists():
            return dest
    return sidecar.with_name(f"{sidecar.name}{STRAY_SUFFIX}{ts}-{os.getpid()}")


def quarantine_sidecars(db_path: Path) -> list[Path]:
    """Move any ``-wal`` / ``-shm`` / ``-journal`` aside so SQLite cannot replay them.

    ONLY safe to call under the exclusive gate (``poppy encrypt repair``), where no
    other connection holds the sidecars: renaming a sidecar out from under a live
    connection would discard its committed writes. Renaming (not deleting)
    preserves the data for inspection; the main DB file is never touched. The
    destination name is uniquified so a second quarantine never overwrites an
    earlier one. Returns the quarantine paths created.
    """
    moved: list[Path] = []
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = db_path.with_name(db_path.name + suffix)
        try:
            if sidecar.exists() and sidecar.stat().st_size > 0:
                dest = _unique_stray_dest(sidecar)
                os.replace(sidecar, dest)
                moved.append(dest)
            elif sidecar.exists():
                sidecar.unlink()  # empty sidecar: nothing to replay, just clear it
        except OSError:
            pass
    return moved


def stray_sidecars(db_path: Path) -> list[Path]:
    """List quarantined ``.stray-*`` sidecars beside the store, if any."""
    parent = db_path.parent
    prefix = db_path.name
    found: list[Path] = []
    try:
        for p in sorted(parent.glob(f"{prefix}*{STRAY_SUFFIX}*")):
            found.append(p)
    except OSError:
        pass
    return found


# --- connect ----------------------------------------------------------------


def connect(db_path: Path | str, *, check_same_thread: bool = False) -> sqlite3.Connection:
    """Open a Poppy SQLite connection with WAL + a busy timeout applied.

    Drop-in replacement for ``sqlite3.connect(str(db_path), check_same_thread=...)``.
    Callers still set their own ``row_factory`` and run their schema/migrations.

    Takes the shared gate first (so a migration cannot run under this connection),
    then routes:

    * plaintext SQLite header -> plaintext driver (refusing, if a stale sentinel
      plus a pending -wal makes the state ambiguous, rather than risk replaying a
      wrong-cipher wal; the fix is ``poppy encrypt repair`` under the exclusive gate);
    * non-magic header + sentinel -> SQLCipher with the keychain/env key;
    * non-magic header + no sentinel -> verified open that tells a healthy
      encrypted store from a corrupt file and refuses if a pending -wal is present;
    * absent/empty + sentinel -> SQLCipher (a brand-new or emptied encrypted store);
    * unreadable header -> a clear permission error, never misclassified as plaintext.

    connect NEVER renames or replays a sidecar: that only happens in
    ``poppy encrypt repair`` under the exclusive gate, where no other connection
    can lose committed data.
    """
    path = Path(db_path)
    gate_fd = acquire_shared_gate(path.parent)
    try:
        # Classify, open and register as one step: a sniff must never overlap
        # another thread's half-built connection. See the note above
        # ``_open_stores``. The gate is taken outside, so a connection waiting on
        # a migration never holds this lock while it waits.
        with _store_lock:
            state = header_state(path)
            sentinel = (path.parent / ENCRYPTION_SENTINEL).exists()

            if state == "unreadable":
                raise StorePermissionError(
                    f"could not read the store at {path}: a permission or read-only filesystem problem, not "
                    "corruption. Fix the permissions and retry."
                )

            if state == "nonmagic" and not sentinel:
                from poppy import encryption  # noqa: PLC0415

                conn = encryption.open_without_sentinel(path, check_same_thread=check_same_thread)
                resolved = "nonmagic"
            elif state == "nonmagic" or (state == "absent" and sentinel):
                from poppy import encryption  # noqa: PLC0415

                conn = encryption.connect_encrypted(path, check_same_thread=check_same_thread)
                resolved = "nonmagic"
            elif state == "plaintext" and sentinel and has_nonempty_wal(path):
                # Ambiguous: plaintext data with a leftover encryption marker AND a
                # pending -wal that might be the wrong cipher. Refuse rather than risk
                # a destructive replay; repair reconciles it under the exclusive gate.
                raise EncryptionError(
                    f"the store at {path} is in an inconsistent state (a plaintext database with a leftover "
                    "encryption marker and pending -wal data). Run `poppy encrypt repair` to reconcile it safely."
                )
            else:
                conn = _apply_pragmas(
                    sqlite3.connect(
                        str(path),
                        check_same_thread=check_same_thread,
                        timeout=BUSY_TIMEOUT_MS / 1000,
                        factory=_GatedConnection,
                    )
                )
                resolved = "plaintext"

            _attach_gate(conn, gate_fd, _register_store(path, resolved))
            gate_fd = None  # ownership transferred to the connection
            return conn
    except BaseException:
        _release_gate(gate_fd)
        raise
