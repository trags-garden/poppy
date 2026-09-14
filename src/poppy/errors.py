"""Lightweight, dependency-free exception types shared across Poppy.

Kept import-cheap (no ML deps) so the CLI entrypoint can catch them without
importing the engine module, which would itself raise ImportError on an install
whose retrieval dependencies are broken.
"""


class ModelUnavailableError(RuntimeError):
    """Retrieval models couldn't be loaded (e.g. offline with a cold model cache).

    Carries an actionable message; the CLI entrypoint renders it cleanly instead
    of dumping a traceback.
    """


class EncryptionError(RuntimeError):
    """Local-store encryption could not be set up, opened, or migrated.

    Defined here (not in ``poppy.encryption``) so ``poppy.keychain`` can make its
    own error a subclass without a circular import, and so the CLI entrypoint can
    render any of them cleanly. ``poppy.encryption`` re-exports these names.
    """


class DependencyMissing(EncryptionError):
    """The optional encryption dependencies (sqlcipher3, keyring) are not installed."""


class KeychainUnavailable(EncryptionError):
    """No usable OS keychain backend, or a keychain operation failed.

    A subclass of ``EncryptionError`` so a keyring failure surfaced from deep in
    ``poppy.db.connect`` is caught by the same handlers as every other
    encryption error instead of escaping as a raw traceback.
    """


class StorePermissionError(EncryptionError):
    """The store could not be opened because of a permission / read-only / I/O error.

    Distinct from a wrong-key/corrupt failure so diagnostics never tell a user
    with a read-only directory to "restore a backup". Lives here (not in
    ``poppy.encryption``) so ``poppy.db`` can raise it for a gate-file permission
    failure without importing the encryption stack.
    """
