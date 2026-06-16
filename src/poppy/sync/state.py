"""Watermark persistence for `poppy sync`.

State lives at ``$POPPY_DIR/sync_state.json``. Schema:

```json
{
  "<trags-url>": {
    "last_pulled_at": "<iso8601>",     // server-side updated_at of latest pulled row
    "last_pushed_at": "<iso8601>",     // local updated_at of latest pushed row
    "last_synced_at": "<iso8601>",     // when sync last completed
    "pushed_count": 0,
    "pulled_count": 0
  }
}
```

Keyed by URL so a user can sync to multiple Trags instances if desired.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

STATE_FILENAME = "sync_state.json"


@dataclass
class RemoteState:
    last_pulled_at: str | None = None
    last_pushed_at: str | None = None
    last_synced_at: str | None = None
    pushed_count: int = 0
    pulled_count: int = 0


@dataclass
class SyncState:
    remotes: dict[str, RemoteState] = field(default_factory=dict)


def _state_path(poppy_dir: Path) -> Path:
    return poppy_dir / STATE_FILENAME


def load(poppy_dir: Path) -> SyncState:
    path = _state_path(poppy_dir)
    if not path.exists():
        return SyncState()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return SyncState()
    remotes = {url: RemoteState(**values) for url, values in (data.get("remotes") or {}).items()}
    return SyncState(remotes=remotes)


def save(poppy_dir: Path, state: SyncState) -> None:
    path = _state_path(poppy_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"remotes": {url: asdict(rs) for url, rs in state.remotes.items()}}
    path.write_text(json.dumps(payload, indent=2))


def get_remote(state: SyncState, url: str) -> RemoteState:
    if url not in state.remotes:
        state.remotes[url] = RemoteState()
    return state.remotes[url]


def latest_iso(*values: str | datetime | None) -> str | None:
    """Return the largest ISO8601 string from a mix of strings / datetimes / Nones."""
    isos = [v.isoformat() if isinstance(v, datetime) else v for v in values if v is not None]
    return max(isos) if isos else None
