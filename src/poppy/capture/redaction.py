"""Secret redaction for captured memory content.

Automatic capture can embed a credential verbatim into a memory when the
extraction LLM copies one out of the transcript ("lesson: staging DB is
postgres://admin:hunter2@db"). Sync encryption is local-at-rest only (SQLCipher,
deliberately not end-to-end), so such a memory would reach Trags Cloud in
server-readable form — a launch-ending screenshot for a privacy-first product.

This is a **minimal deny-list pass**, not the full scrubbing the PRD deferred: a
handful of high-signal, low-false-positive token shapes are masked in place, and
the memory is always kept (masking a token loses far less than dropping a whole
lesson). It is applied once, early, at candidate construction
(`capture.orchestrator.build_capture_memories`) — the single earliest seam — so a raw secret
never becomes a Memory and every downstream consumer (the reconciler/store, sync,
and the plaintext local capture journal preview) sees only the masked content.
That construction seam is shared by all three capture paths (mid-session,
SessionEnd, PostCompact). Patterns are anchored and precompiled to keep the
per-capture cost negligible.

Patterns: AWS access-key ids, OpenAI ``sk-`` keys, GitHub tokens,
PEM private-key blocks, credentials in connection-string URLs, and bearer tokens.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poppy.config import PoppyConfig

# The placeholder a masked secret is replaced with. A visible, grep-able marker
# so a reader understands Poppy redacted something (vs. silently dropping it).
MASK = "[REDACTED]"

# Custom redaction is literal-only in v1. Four characters is long enough to
# avoid masking common prose fragments while still covering short credentials.
MIN_CUSTOM_SECRET_LENGTH = 4
_ENV_VAR_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

PatternReplacement = tuple[re.Pattern[str], str]

# Each entry is (compiled_pattern, replacement). Replacements that keep a capture
# group ("\1...") preserve surrounding structure (the URL, the "Bearer " prefix)
# and mask only the secret; the rest replace the whole matched token.
#
# All patterns are anchored (word boundaries / required prefixes) and require a
# minimum secret length, so ordinary prose and code do not trip them. Order:
# multi-line PEM block first, then structured (URL / bearer), then bare tokens.
_PATTERNS: tuple[PatternReplacement, ...] = (
    # PEM private-key block: mask the whole block (markers + body) — the body is
    # the secret and the markers alone carry no memory value.
    (
        re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        MASK,
    ),
    # Credentials in a connection-string URL: keep scheme://user, mask password.
    # e.g. postgres://admin:hunter2@db -> postgres://admin:[REDACTED]@db
    (
        re.compile(r"(://[^:/@\s]+:)([^@/\s]+)(@)"),
        r"\1" + MASK + r"\3",
    ),
    # Bearer token: keep the "Bearer " prefix, mask the token.
    (
        re.compile(r"(\bBearer\s+)([A-Za-z0-9._~+/=-]{16,})", re.IGNORECASE),
        r"\1" + MASK,
    ),
    # AWS access key id (long-term AKIA, temporary ASIA): prefix + 16 uppercase.
    (
        re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        MASK,
    ),
    # OpenAI-style secret key: sk- (optionally sk-proj-) + a long token body.
    (
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
        MASK,
    ),
    # GitHub tokens: PAT (ghp_), OAuth (gho_), user/server/refresh (ghu_/ghs_/ghr_).
    (
        re.compile(r"\bgh[posur]_[A-Za-z0-9]{20,}"),
        MASK,
    ),
)


def valid_env_var_name(name: str) -> bool:
    """Return whether ``name`` is a portable shell-style environment name."""
    return bool(_ENV_VAR_NAME.fullmatch(name.strip()))


def load_custom_redaction(config: PoppyConfig) -> tuple[tuple[PatternReplacement, ...], list[str]]:
    """Compile valid custom redaction entries and describe invalid ones.

    Literal entries are escaped before compilation, so user input can never
    become regex syntax. Environment entries resolve their current process
    value each time this loader runs, immediately before capture redaction.
    Missing and short environment values are normal states and stay silent;
    malformed configured entries are reported for ``poppy doctor``.
    """
    patterns: list[PatternReplacement] = []
    issues: list[str] = []

    for position, configured in enumerate(config.redaction_literals, start=1):
        literal = configured.strip()
        if len(literal) < MIN_CUSTOM_SECRET_LENGTH:
            # The literal is itself a secret: identify it by position and length,
            # never by value, so `poppy doctor` output stays safe to share.
            issues.append(
                f"Skipped custom literal #{position} ({len(literal)} characters): "
                f"must be at least {MIN_CUSTOM_SECRET_LENGTH} characters."
            )
            continue
        patterns.append((re.compile(re.escape(literal)), MASK))

    for configured in config.redaction_env_vars:
        name = configured.strip()
        if not valid_env_var_name(name):
            issues.append(f"Skipped environment variable {name!r}: name must be a valid identifier.")
            continue
        value = os.environ.get(name)
        if value is None or len(value) < MIN_CUSTOM_SECRET_LENGTH:
            continue
        patterns.append((re.compile(re.escape(value)), MASK))

    return tuple(patterns), issues


def redact_secrets(text: str, extra_patterns: Sequence[PatternReplacement] = ()) -> str:
    """Mask deny-listed secret token shapes in ``text``, keeping everything else.

    Never raises and never drops content: an input with no secrets is returned
    unchanged (same object). Applied to captured memory content before it is
    written.
    """
    if not text:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    # Only caller-supplied substitutions are isolated, so one bad pattern
    # cannot block the rest; the built-ins above stay exactly as infallible
    # as before custom redaction existed.
    for pattern, replacement in extra_patterns:
        try:
            text = pattern.sub(replacement, text)
        except Exception:
            continue
    return text
