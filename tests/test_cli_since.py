"""--since filtering on `poppy recall` and `poppy list`.

These run against the SeedEngine (FTS5 only, no model downloads) by writing
{"engine": "seed"} into the test POPPY_DIR. The --since wiring is
engine-agnostic: the CLI parses the value into Filters.since and every
engine applies the same created_at >= since rule.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.models import Memory, Source


def _use_seed_engine(poppy_dir: Path) -> None:
    poppy_dir.mkdir(parents=True, exist_ok=True)
    (poppy_dir / "config.json").write_text('{"engine": "seed"}')


def _seed_memory(poppy_dir: Path, content: str, created_at: datetime) -> str:
    """Ingest a memory with a controlled created_at, bypassing the CLI."""
    from poppy.runtime import get_engine

    engine = get_engine(poppy_dir)
    mem = Memory(
        id=f"mem_{uuid.uuid4().hex[:12]}",
        content=content,
        memory_type="fact",
        source=Source(type="manual", session_id=None, timestamp=created_at),
        project=None,
        related_to=[],
        created_at=created_at,
        updated_at=created_at,
    )
    engine.ingest(mem)
    return mem.id


def _setup_old_and_new(poppy_dir: Path) -> None:
    _use_seed_engine(poppy_dir)
    _seed_memory(poppy_dir, "alpha old entry", datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc))
    _seed_memory(poppy_dir, "alpha new entry", datetime(2026, 6, 5, 10, 0, tzinfo=timezone.utc))


# --- poppy list --since ---


def test_list_since_iso_date(tmp_path):
    _setup_old_and_new(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "2026-06-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha new entry" in result.output
    assert "alpha old entry" not in result.output


def test_list_since_relative_duration(tmp_path):
    _use_seed_engine(tmp_path)
    now = datetime.now(timezone.utc)
    _seed_memory(tmp_path, "alpha stale entry", now - timedelta(days=10))
    _seed_memory(tmp_path, "alpha fresh entry", now - timedelta(hours=1))
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "7d"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha fresh entry" in result.output
    assert "alpha stale entry" not in result.output


def test_list_since_boundary_inclusive(tmp_path):
    # A memory created exactly at the threshold is included (created_at >= since).
    _use_seed_engine(tmp_path)
    _seed_memory(tmp_path, "alpha boundary entry", datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc))
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "2026-06-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha boundary entry" in result.output


def test_list_since_invalid_input_is_usage_error(tmp_path):
    _use_seed_engine(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "not-a-date"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 2  # click usage error, not a crash
    assert "--since" in result.output
    assert "not-a-date" in result.output
    assert "Traceback" not in result.output


# --- poppy recall --since ---


def test_recall_since_iso_date(tmp_path):
    _setup_old_and_new(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "alpha", "--since", "2026-06-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha new entry" in result.output
    assert "alpha old entry" not in result.output


def test_recall_since_relative_duration(tmp_path):
    _use_seed_engine(tmp_path)
    now = datetime.now(timezone.utc)
    _seed_memory(tmp_path, "alpha stale entry", now - timedelta(days=10))
    _seed_memory(tmp_path, "alpha fresh entry", now - timedelta(hours=1))
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "alpha", "--since", "7d"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha fresh entry" in result.output
    assert "alpha stale entry" not in result.output


def test_recall_since_boundary_inclusive(tmp_path):
    _use_seed_engine(tmp_path)
    _seed_memory(tmp_path, "alpha boundary entry", datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc))
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "alpha", "--since", "2026-06-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha boundary entry" in result.output


def test_recall_since_invalid_input_is_usage_error(tmp_path):
    _use_seed_engine(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "alpha", "--since", "nonsense!!"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 2  # click usage error, not a crash
    assert "--since" in result.output
    assert "Traceback" not in result.output


def test_list_since_timezone_offset_datetime(tmp_path):
    # Regression: a +02:00 --since must be compared in UTC, not as a raw
    # string. 2026-06-01T12:30:00+02:00 is 10:30Z, so a memory created at
    # 11:00Z is INSIDE the window and one at 10:00Z is outside.
    _use_seed_engine(tmp_path)
    _seed_memory(tmp_path, "alpha inside window", datetime(2026, 6, 1, 11, 0, tzinfo=timezone.utc))
    _seed_memory(tmp_path, "alpha outside window", datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc))
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "2026-06-01T12:30:00+02:00"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha inside window" in result.output
    assert "alpha outside window" not in result.output


def test_recall_since_timezone_offset_datetime(tmp_path):
    # Same window as the list test above; recall and list must agree.
    _use_seed_engine(tmp_path)
    _seed_memory(tmp_path, "alpha inside window", datetime(2026, 6, 1, 11, 0, tzinfo=timezone.utc))
    _seed_memory(tmp_path, "alpha outside window", datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc))
    runner = CliRunner()
    result = runner.invoke(
        cli, ["recall", "alpha", "--since", "2026-06-01T12:30:00+02:00"], env={"POPPY_DIR": str(tmp_path)}
    )
    assert result.exit_code == 0
    assert "alpha inside window" in result.output
    assert "alpha outside window" not in result.output


def test_list_since_timezone_offset_boundary_inclusive(tmp_path):
    # 2026-06-01T12:00:00+02:00 == 10:00Z exactly; created_at >= since keeps it.
    _use_seed_engine(tmp_path)
    _seed_memory(tmp_path, "alpha boundary entry", datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc))
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "2026-06-01T12:00:00+02:00"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha boundary entry" in result.output


def test_recall_since_excludes_nothing_when_old_enough(tmp_path):
    # A --since far in the past filters nothing out.
    _setup_old_and_new(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "alpha", "--since", "2020-01-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "alpha new entry" in result.output
    assert "alpha old entry" in result.output


# --- empty-state messaging (PR #3 review follow-up) ---


def test_list_filtered_empty_says_filters_not_empty_store(tmp_path):
    """Memories exist but the filter excludes them all: the message must not
    claim the store is empty."""
    _setup_old_and_new(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--since", "2099-01-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "No memories match the given filters" in result.output
    assert "No memories stored yet" not in result.output


def test_list_empty_store_still_says_stored_yet(tmp_path):
    _use_seed_engine(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["list"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "No memories stored yet" in result.output


def test_recall_filtered_empty_says_filters(tmp_path):
    _setup_old_and_new(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "alpha", "--since", "2099-01-01"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "No memories match the given filters" in result.output


def test_recall_unfiltered_miss_keeps_plain_message(tmp_path):
    _setup_old_and_new(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "zzz-nonexistent"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "No memories found." in result.output
