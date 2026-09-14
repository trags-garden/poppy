"""Regression tests: forget must tombstone before deleting.

Both forget entry points (the CLI `forget` command and the MCP `forget` tool)
used to call `engine.delete` with no tombstone. Sync's `push()` propagates
deletions *only* as tombstones, so a bare delete never reached the cloud: the
still-live cloud row survived and `pull()` re-ingested it on the next sync — the
"forgotten" memory silently came back.

These tests pin the fix: forget writes a tombstone, and that tombstone is what
`push()` sends so the deletion propagates (and cannot resurrect at the source).
"""

from __future__ import annotations

import json
import re

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.engine.seed import SeedEngine
from poppy.mcp_server.server import PoppyMcpServer
from poppy.sync import push
from poppy.ui.tombstones import TombstoneStore


def _first_id_from_list_json(output: str) -> str:
    """Extract the first memory id from `poppy list --json` output.

    The default engine loads an ONNX model on first use, which prints
    download/progress noise to the stream CliRunner mixes into `output`;
    a naive json.loads chokes on it (this is why CI failed while a warm local
    run passed). Pull the JSON array out with a regex, as test_cli.py does.
    """
    match = re.search(r"(\[\s*\{.*\}\s*\])", output, re.DOTALL)
    assert match, f"No JSON array found in output: {output[:200]}"
    return json.loads(match.group(1))[0]["id"]


class _RecordingClient:
    """Minimal TragsClient stand-in that records every upserted wire row."""

    base_url = "https://trags.test"

    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert(self, row: dict) -> None:
        self.upserts.append(row)

    def ping(self) -> None:
        """The probe a push with nothing to send makes."""
        return None


@pytest.mark.asyncio
async def test_mcp_forget_writes_tombstone(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "memories.db")
    server = PoppyMcpServer(poppy_dir=tmp_path, engine=engine)

    r = await server.handle_remember(content="secret to be forgotten", memory_type="fact")
    mem_id = r["id"]

    result = await server.handle_forget(id=mem_id)
    assert result["deleted"] is True

    # The live row is gone locally, and a tombstone was recorded for it.
    assert engine.get(mem_id) is None
    tombstones = TombstoneStore(tmp_path / "memories.db")
    assert tombstones.get(mem_id) is not None


def test_cli_forget_writes_tombstone(tmp_path):
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}
    runner.invoke(cli, ["remember", "secret to be forgotten"], env=env)

    listing = runner.invoke(cli, ["list", "--json"], env=env)
    mem_id = _first_id_from_list_json(listing.output)

    result = runner.invoke(cli, ["forget", mem_id, "--yes"], env=env)
    assert result.exit_code == 0

    tombstones = TombstoneStore(tmp_path / "memories.db")
    assert tombstones.get(mem_id) is not None


def test_cli_forget_cancelled_writes_no_tombstone(tmp_path):
    # A cancelled forget (confirmation declined) must record nothing — the
    # tombstone write lives after the confirmation gate.
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}
    runner.invoke(cli, ["remember", "keep me"], env=env)

    listing = runner.invoke(cli, ["list", "--json"], env=env)
    mem_id = _first_id_from_list_json(listing.output)

    result = runner.invoke(cli, ["forget", mem_id], input="n\n", env=env)
    assert result.exit_code == 0
    assert "Cancelled." in result.output

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    assert engine.get(mem_id) is not None  # still live
    tombstones = TombstoneStore(tmp_path / "memories.db")
    assert tombstones.get(mem_id) is None  # nothing recorded


@pytest.mark.asyncio
async def test_forget_tombstone_is_pushed_so_deletion_propagates(tmp_path, synced_state):
    """The core resurrection regression: after forget, push() must emit a
    tombstone wire row (deleted_at set) for the forgotten id. That is what tells
    the cloud to delete the row, so it can never be re-ingested on the next
    pull."""
    engine = SeedEngine(db_path=tmp_path / "memories.db")
    server = PoppyMcpServer(poppy_dir=tmp_path, engine=engine)

    r = await server.handle_remember(content="propagate my deletion", memory_type="fact")
    mem_id = r["id"]
    # This particular memory reached the remote before it was forgotten.
    client = _RecordingClient()
    tombstones = TombstoneStore(tmp_path / "memories.db")
    from poppy.sync.state import SyncState

    push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    await server.handle_forget(id=mem_id)

    client = _RecordingClient()
    tombstones = TombstoneStore(tmp_path / "memories.db")

    res = push(engine=engine, tombstones=tombstones, client=client, state=synced_state(), poppy_dir=tmp_path)

    # push sends the deletion as a tombstone, and nothing live for the id.
    assert res.sent_tombstones >= 1
    tombstone_rows = [row for row in client.upserts if row["id"] == mem_id and row["deleted_at"] is not None]
    assert len(tombstone_rows) == 1
    assert not any(row["id"] == mem_id and row["deleted_at"] is None for row in client.upserts)
