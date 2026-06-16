"""Source-app provenance vocabulary and normalization.

Single home for resolving *which client app* wrote a memory, so the manual MCP
write path (`mcp_server/server.py`) and the capture path (`consolidation.py`)
share one vocabulary instead of sprinkling literals.

`Source.type` records the source app: ``claude-code``, ``cursor``, ``windsurf``,
``vscode``, ``claude-desktop``, ``manual``, ``ui``, ... — never the transport
string ``mcp``.
"""

import re

# The transport string. Never a valid source app; used as the "unset" sentinel
# by `poppy serve --source` (its default) and ignored wherever it appears.
TRANSPORT_SENTINEL = "mcp"

# Stamped when a client connects but does not identify itself in the handshake.
GENERIC_AGENT = "agent"

# Canonical source app -> raw clientInfo.name spellings (lowercased) we map to it.
# MCP `clientInfo.name` is NOT standardized across clients, so this map is
# best-effort and grows as we observe real handshakes. Anything not listed is
# slugified verbatim (see `normalize_client`) rather than guessed-at or dropped,
# so an unmapped-but-identifying client still records its real name.
_ALIASES: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude-code", "claudecode", "claude-code-cli"),
    "cursor": ("cursor",),
    "windsurf": ("windsurf",),
    "vscode": ("vscode", "visual studio code", "vs code"),
    "claude-desktop": ("claude-desktop",),
}
_RAW_TO_SOURCE: dict[str, str] = {raw: canon for canon, raws in _ALIASES.items() for raw in raws}


def _slugify(value: str) -> str:
    """Lowercase, collapse non-alphanumerics to single hyphens, strip ends."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def normalize_client(raw: str | None) -> str | None:
    """Normalize a reported client name to a source-vocabulary slug.

    Returns ``None`` when the client is effectively unidentified — empty/missing,
    or the transport string ``mcp`` itself (which some wrappers may report and
    which is never a real source app). Known names map via the alias table;
    anything else is slugified (e.g. ``"Claude Desktop"`` -> ``"claude-desktop"``,
    ``"Some Client"`` -> ``"some-client"``) so real provenance is preserved.
    """
    if not raw:
        return None
    key = raw.strip().lower()
    if not key or key == TRANSPORT_SENTINEL:
        return None
    if key in _RAW_TO_SOURCE:
        return _RAW_TO_SOURCE[key]
    return _slugify(key) or None


def resolve_source(*, client_name: str | None, configured: str | None) -> str:
    """Resolve the source app to stamp on an MCP-written memory.

    Precedence:
      1. An explicit configured source (``poppy serve --source <client>``, written
         by ``poppy setup``) — a deliberate operator choice, so it wins. The
         transport sentinel ``mcp`` counts as unset.
      2. The connecting client's normalized ``clientInfo.name`` — fixes the
         generic ``poppy serve`` path that has no ``--source``.
      3. ``agent`` — a client connected but did not identify itself. Never ``mcp``.
    """
    cfg = (configured or "").strip().lower()
    if cfg and cfg != TRANSPORT_SENTINEL:
        return cfg
    return normalize_client(client_name) or GENERIC_AGENT


def client_name_from_context(ctx: object) -> str | None:
    """Best-effort read of the connecting client's ``clientInfo.name``.

    The MCP handshake fields are all optional and absent before initialization,
    so every hop is guarded: a malformed/partial context yields ``None``.
    """
    try:
        return ctx.session.client_params.clientInfo.name  # type: ignore[attr-defined]
    except AttributeError:
        return None
