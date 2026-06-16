"""Per-session capture state file (ADR-0001).

A single JSON file under the Poppy data directory holds the per-session capture
state — the watermark and, later, the turn cadence counters.
It is keyed by session id so concurrent sessions never corrupt each other's
progress.

Writes go through ``save`` which does an atomic ``os.replace`` so a crash mid-write
can never leave a torn file. Cross-process races within one session are prevented
by the mid-session single-flight lock; SessionStart / SessionEnd do not
run concurrently with a mid-session worker in practice.

This file is a per-device cache of capture progress — it is NOT synced (the sync
boundary keeps the capture journal / state local).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STATE_FILENAME = "capture_state.json"


def state_path(poppy_dir: Path) -> Path:
    return poppy_dir / STATE_FILENAME


def load(poppy_dir: Path) -> dict:
    path = state_path(poppy_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save(poppy_dir: Path, data: dict) -> None:
    poppy_dir.mkdir(parents=True, exist_ok=True)
    path = state_path(poppy_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)
