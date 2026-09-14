"""Resolve a project name from a working directory.

Both the recall hooks and the auto-capture consolidator tag memories with a
project so recall can scope to the current repo. They MUST agree on how a cwd
maps to a project name, or a memory captured under one rule never resurfaces
under the other. This is the single shared resolver; do not reintroduce a naive
``Path(cwd).name`` variant — it mis-tags cases like ``~/code/personal``,
``~/scratch``, and bare home directories, burying those captures under a bogus
project tag they can never be recalled by.
"""

from __future__ import annotations

import os
from pathlib import Path

# Markers that indicate a project root, in priority order. CLAUDE.md / AGENTS.md
# beat package files because the user authored them deliberately; package files
# beat .git because a monorepo .git can sit far above the actual project.
# Scanning stops at the first match walking up from cwd.
PROJECT_MARKERS: tuple[str, ...] = (
    "CLAUDE.md",
    "AGENTS.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    ".git",
)


def project_from_cwd(cwd: str | None, max_depth: int = 6) -> str | None:
    """Resolve a project name from cwd by walking up looking for project markers.

    Returns the basename of the directory containing the first marker found.
    Returns None if cwd is unset or no marker is found within ``max_depth``
    parents. Falling back to ``Path(cwd).name`` would mis-tag cases like
    ``~/code/personal`` where the basename ("personal") is not actually a
    project — better to return None and let the caller search across all
    projects than to bury the memory under a tag it can never be recalled by.

    The path is canonicalized (symlinks resolved) before the name is derived, so
    a repo reached through a differently-named symlink alias yields the same
    project name as its physical path. Every consumer keys on this name — recall
    tagging, capture tagging, and the per-project capture deny-list — so
    they must agree; a symlink-dependent name would let the exact-string deny-list
    be silently bypassed. The CLI side already sees physical paths via
    ``os.getcwd()``, so this makes the hook side (whose payload cwd can be an
    unresolved alias) agree with it.
    """
    if not cwd:
        return None
    start = Path(cwd)
    if not start.is_absolute():
        return start.name or None
    # realpath() never raises (unlike strict resolve) and canonicalizes symlinks
    # even when the tail does not exist — safe on any hook payload cwd.
    current = Path(os.path.realpath(start))
    for _ in range(max_depth):
        for marker in PROJECT_MARKERS:
            if (current / marker).exists():
                return current.name or None
        if current.parent == current:
            break
        current = current.parent
    return None
