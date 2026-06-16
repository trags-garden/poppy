"""Install Poppy into Goose (Block's open-source agent).

Goose uses YAML config at ``~/.config/goose/config.yaml`` (overridable via
``XDG_CONFIG_HOME``) and an ``extensions:`` block where each entry registers
an MCP server. Global agent hints live in ``~/.config/goose/.goosehints``.

We hand-roll YAML manipulation (mirroring ``hermes.py``) to avoid a PyYAML
dependency. The handling is scoped — we only touch the ``extensions.poppy``
sub-block and never reformat surrounding YAML.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from poppy.setup.claude_code import (
    get_poppy_executable,
    install_primer_md_block,
)

GOOSE_EXTENSION_NAME = "poppy"


def get_goose_config_dir() -> Path:
    """Resolve Goose's config directory.

    Honors ``XDG_CONFIG_HOME``. Defaults to ``~/.config/goose`` on macOS/Linux.
    On Windows, Goose follows the standard ``%APPDATA%\\Block\\goose\\config``
    layout, which we mirror.
    """
    override = os.environ.get("POPPY_GOOSE_CONFIG_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Block" / "goose" / "config"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "goose"


def _poppy_extension_block(indent: str = "  ") -> str:
    """Render the YAML body for the poppy extension entry.

    Two-space indent is the Goose convention. The block is anchored at
    ``  poppy:`` and contains keys at four-space indent.
    """
    cmd = get_poppy_executable()
    body = (
        f"{indent}{GOOSE_EXTENSION_NAME}:\n"
        f"{indent}  type: stdio\n"
        f"{indent}  name: {GOOSE_EXTENSION_NAME}\n"
        f"{indent}  cmd: {cmd}\n"
        f"{indent}  args:\n"
        f"{indent}    - serve\n"
        f"{indent}  enabled: true\n"
        f"{indent}  envs: {{}}\n"
        f"{indent}  timeout: 300\n"
    )
    return body


def _strip_existing_poppy(extensions_body: str) -> str:
    """Remove an existing ``poppy:`` sub-block from an extensions block body.

    ``extensions_body`` is the text under ``extensions:`` — every line is
    indented (or blank). We detect the poppy sub-block as: a line matching
    ``^(\\s+)poppy:\\s*$`` followed by lines indented deeper than that anchor.
    """
    lines = extensions_body.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s+)poppy:\s*$", line.rstrip("\n"))
        if not m:
            out.append(line)
            i += 1
            continue
        # Skip this anchor + every following deeper-indented (or blank) line.
        anchor_indent = len(m.group(1))
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if nxt.strip() == "":
                i += 1
                continue
            leading = len(nxt) - len(nxt.lstrip(" "))
            if leading > anchor_indent:
                i += 1
                continue
            break
    return "".join(out)


def _extensions_block_span(config_text: str) -> tuple[int, int] | None:
    """Locate the byte range of the ``extensions:`` block body in the file.

    Returns ``(body_start, body_end)`` covering everything after the
    ``extensions:`` header line up to (but not including) the next top-level
    key or EOF. Returns ``None`` if no ``extensions:`` block exists.
    """
    header = re.search(r"(?m)^extensions:[ \t]*\n", config_text)
    if not header:
        return None
    body_start = header.end()
    # Next top-level key: a line starting at column 0 with non-whitespace,
    # followed by ``:``. Comments at column 0 don't terminate the block.
    tail = re.search(r"(?m)^[A-Za-z_][\w-]*:", config_text[body_start:])
    body_end = body_start + tail.start() if tail else len(config_text)
    return body_start, body_end


def _set_poppy_extension(config_text: str) -> str:
    """Insert/replace the ``poppy:`` entry under ``extensions:`` in YAML text.

    Three cases:
      1. ``extensions:`` block exists with a ``poppy:`` child — replace it.
      2. ``extensions:`` block exists without a ``poppy:`` child — append the
         poppy entry at the end of the block.
      3. No ``extensions:`` block — append a fresh one to the file.
    """
    span = _extensions_block_span(config_text)
    new_entry = _poppy_extension_block()

    if span is None:
        sep = "" if config_text == "" or config_text.endswith("\n") else "\n"
        return config_text + sep + "extensions:\n" + new_entry

    body_start, body_end = span
    body = config_text[body_start:body_end]
    stripped = _strip_existing_poppy(body)

    # Ensure the inserted block ends with a single newline boundary so the
    # next top-level key (or EOF) is cleanly separated.
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"

    new_body = stripped + new_entry
    return config_text[:body_start] + new_body + config_text[body_end:]


def _install_extension(config_path: Path) -> Path:
    existing = config_path.read_text() if config_path.exists() else ""
    new_text = _set_poppy_extension(existing)
    if new_text != existing:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(new_text)
    return config_path


def install_for_goose(goose_config_dir: Path | None = None) -> dict[str, Path]:
    """Install Poppy into Goose. Returns a {label: path} map.

    Writes:
      - ``$GOOSE_CONFIG/config.yaml`` with an ``extensions.poppy`` entry
        registering ``poppy serve`` as a stdio MCP server.
      - ``$GOOSE_CONFIG/.goosehints`` with the managed POPPY:BEGIN/END
        primer block describing the Poppy tool surface.
    """
    config_dir = goose_config_dir or get_goose_config_dir()
    config_path = _install_extension(config_dir / "config.yaml")
    primer_path = install_primer_md_block(config_dir / ".goosehints")
    return {
        "config.yaml": config_path,
        "Primer (.goosehints)": primer_path,
    }


def is_goose_installed(goose_config_dir: Path | None = None) -> bool:
    """Whether Poppy is registered as a Goose extension."""
    config_dir = goose_config_dir or get_goose_config_dir()
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        return False
    text = config_path.read_text()
    span = _extensions_block_span(text)
    if span is None:
        return False
    body = text[span[0] : span[1]]
    return bool(re.search(r"(?m)^\s+poppy:\s*$", body))
