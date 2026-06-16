"""Lightweight, dependency-free exception types shared across Poppy.

Kept import-cheap (no ML deps) so the CLI entrypoint can catch them without
importing the engine module — importing the engine probes for sentence-transformers
and would itself raise ImportError on a deps-missing install.
"""


class ModelUnavailableError(RuntimeError):
    """Retrieval models couldn't be loaded (e.g. offline with a cold model cache).

    Carries an actionable message; the CLI entrypoint renders it cleanly instead
    of dumping a traceback.
    """
