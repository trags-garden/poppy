"""MCPB entry shim — boots the Poppy MCP server when bundled as `.mcpb`.

Claude Desktop's MCPB UV runtime invokes this file directly (server.type=uv,
entry_point=server/main.py). It must keep stdout clean for JSON-RPC; banner
lines go to stderr.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_poppy_dir() -> Path:
    """Resolve POPPY_DIR with defensive expansion.

    Claude Desktop's MCPB user_config rendering can leak the literal template
    `${HOME}/.poppy` (or `~`) into the env var if the user hasn't picked an
    explicit path via Browse. If we passed that to Path() unchanged we'd
    create a directory called `${HOME}` in cwd. Expand both forms.
    """
    raw = os.environ.get("POPPY_DIR")
    if not raw or raw.strip() == "":
        return Path.home() / ".poppy"
    expanded = os.path.expandvars(raw)
    if expanded.startswith("~"):
        expanded = os.path.expanduser(expanded)
    if "${" in expanded or expanded.startswith("$"):
        return Path.home() / ".poppy"
    return Path(expanded)


def main() -> None:
    from poppy.mcp_server.server import create_mcp_server

    poppy_dir = _resolve_poppy_dir()
    poppy_dir.mkdir(parents=True, exist_ok=True)

    print(f"poppy mcpb: storage at {poppy_dir}", file=sys.stderr)
    mcp = create_mcp_server(poppy_dir=poppy_dir)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
