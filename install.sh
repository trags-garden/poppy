#!/usr/bin/env bash
# Poppy — remember what matters.
#   curl -fsSL https://raw.githubusercontent.com/trags-garden/poppy/main/install.sh | bash
set -euo pipefail

PY="${PY:-python3.13}"

if ! command -v "$PY" >/dev/null 2>&1; then
    for candidate in python3.13 python3.12 python3.11; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    done
fi

if ! command -v "$PY" >/dev/null 2>&1; then
    echo "error: need Python 3.11+ on PATH (tried python3.13/3.12/3.11)." >&2
    exit 1
fi

if ! command -v pipx >/dev/null 2>&1; then
    echo "Installing pipx ..."
    "$PY" -m pip install --user pipx
    "$PY" -m pipx ensurepath
fi

pipx install --python "$PY" poppy-memory

cat <<'EOF'

Poppy installed.

Next step:
    poppy setup claude-code     # wire up MCP + lifecycle hooks for Claude Code
    poppy setup goose           # Goose (Block)
    poppy setup hermes-agent    # Hermes Agent (Nous Research)

Then in a new shell:
    poppy remember "we always use uv for python deps"
    poppy recall "uv"
EOF
