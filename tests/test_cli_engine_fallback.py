"""Human visibility and agent-surface silence for engine fallback."""

import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from mcp.server.fastmcp import FastMCP

import poppy.engine.registry as registry
from poppy.cli.main import _ENGINE_FALLBACK_NOTICE_FILE, cli
from poppy.engine.interface import EngineStats


class _SeedFallback:
    _engine_name = "seed"
    model_id = None

    def stats(self) -> EngineStats:
        return EngineStats(memory_count=0, storage_bytes=0, engine_name="seed", engine_version="test")


def _make_bloom_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback = _SeedFallback()

    def resolve(name, _db_path, *, delegate_models_to_daemon=False):
        if name == "bloom":
            raise ImportError("the onnx runtime failed to load\nsecondary import detail")
        if name == "seed":
            return fallback
        raise AssertionError(f"unexpected fallback candidate: {name}")

    monkeypatch.setattr(registry, "resolve_engine", resolve)


def test_human_cli_reports_fallback_once_per_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_bloom_unavailable(monkeypatch)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}
    notice = (
        "Poppy's ML dependencies cannot be imported (the onnx runtime failed to load); running keyword-only search."
    )

    first = runner.invoke(cli, ["stats"], env=env)
    second = runner.invoke(cli, ["stats"], env=env)

    assert first.exit_code == 0, first.output
    assert first.output.count(notice) == 1
    assert second.exit_code == 0, second.output
    assert notice not in second.output
    assert (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).is_file()


def test_human_cli_reports_lazy_model_fallback_once_per_day(tmp_path, monkeypatch):
    from poppy.engine import _fastembed_loader as fe

    def unavailable(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *args: False)
    monkeypatch.setattr("fastembed.TextEmbedding", unavailable)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}

    saved = runner.invoke(cli, ["remember", "offline keyword memory"], env=env)
    recalled = runner.invoke(cli, ["recall", "keyword"], env=env)

    assert saved.exit_code == recalled.exit_code == 0, saved.output + recalled.output
    assert "saving memories and running keyword-only search" in saved.stderr
    assert "once models are available" in saved.stderr
    assert "Reinstall" not in saved.stderr
    assert "keyword-only search" not in recalled.stderr
    assert "offline keyword memory" in recalled.stdout
    assert (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).is_file()


def test_changed_fallback_pair_renotifies_same_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The throttle keys on what substituted for what, so a same-day change in
    the fallback pair warns again instead of hiding a new degradation."""
    _make_bloom_unavailable(monkeypatch)
    today = datetime.date.today().isoformat()
    (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).write_text(f"{today} seed->bloom earlier reason\n")

    result = CliRunner().invoke(cli, ["stats"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "running keyword-only search" in result.output
    assert (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).read_text().strip() == (
        f"{today} bloom->seed the onnx runtime failed to load"
    )


def test_changed_fallback_reason_renotifies_same_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The throttle keys on the reason too, so a same-day change to a different
    actionable failure for the same pair warns again instead of staying silent."""
    _make_bloom_unavailable(monkeypatch)
    today = datetime.date.today().isoformat()
    (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).write_text(f"{today} bloom->seed some earlier reason\n")

    result = CliRunner().invoke(cli, ["stats"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "running keyword-only search" in result.output
    assert (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).read_text().strip() == (
        f"{today} bloom->seed the onnx runtime failed to load"
    )


def test_hook_and_serve_never_report_engine_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_bloom_unavailable(monkeypatch)
    monkeypatch.setattr(FastMCP, "run", lambda *_args, **_kwargs: None)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}

    hook_result = runner.invoke(cli, ["hook", "stop"], input="{}", env=env)
    serve_result = runner.invoke(cli, ["serve", "--no-daemon"], env=env)

    assert hook_result.exit_code == 0, hook_result.output
    assert serve_result.exit_code == 0, serve_result.output
    assert "running keyword-only search" not in hook_result.output
    assert "running keyword-only search" not in serve_result.output
    assert not (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).exists()


def test_serve_constructs_pipeline_without_model_access(tmp_path, monkeypatch):
    from poppy.engine._fastembed_loader import FastembedModels
    from poppy.engine.bloom import BloomEngine

    def unexpected(*args, **kwargs):
        pytest.fail("starting the server must leave model loading to first use")

    def run(server, **kwargs):
        assert isinstance(server._poppy_engine, BloomEngine)
        assert server._poppy_engine._models._bi is None
        assert server._poppy_engine._models._cross is None

    monkeypatch.setattr(BloomEngine, "check_models", unexpected)
    monkeypatch.setattr(FastembedModels, "_load_model", unexpected)
    monkeypatch.setattr(FastMCP, "run", run)

    result = CliRunner().invoke(cli, ["serve", "--no-daemon"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("args", [["list"], ["stats"], ["forget", "stored", "--yes"], ["expire", "--yes"]])
def test_storage_commands_do_not_check_or_load_models(tmp_path, monkeypatch, args):
    from test_engine_migration import _make_memory

    from poppy.engine._fastembed_loader import FastembedModels
    from poppy.engine.bloom import BloomEngine
    from poppy.engine.seed import SeedEngine

    SeedEngine(tmp_path / "memories.db").ingest(_make_memory("stored", "stored memory"))

    def unexpected(*args, **kwargs):
        pytest.fail("storage-only commands must not access models")

    monkeypatch.setattr(BloomEngine, "check_models", unexpected)
    monkeypatch.setattr(FastembedModels, "_load_model", unexpected)

    result = CliRunner().invoke(cli, args, env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output


def test_corrupt_unwritable_throttle_never_breaks_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_bloom_unavailable(monkeypatch)
    notice_path = tmp_path / _ENGINE_FALLBACK_NOTICE_FILE
    notice_path.write_bytes(b"\xff")
    real_write_text = Path.write_text

    def fail_notice_write(path: Path, *args, **kwargs):
        if path == notice_path:
            raise OSError("read-only filesystem")
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_notice_write)

    result = CliRunner().invoke(cli, ["stats"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "running keyword-only search" in result.output


@pytest.mark.parametrize(
    "args",
    [["engines", "use", "seed"], ["engines", "use", "unknown"], ["config", "set", "engine", "seed"]],
)
def test_retired_engine_commands_leave_config_and_pipeline_unchanged(tmp_path, args):
    from poppy.engine.bloom import BloomEngine
    from poppy.runtime import get_engine

    path = tmp_path / "config.json"
    original = '{"engine": "baseline", "auto_sync": "off"}'
    path.write_text(original)

    result = CliRunner().invoke(cli, args, env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "Poppy now uses one retrieval pipeline" in result.output
    assert path.read_text() == original
    assert isinstance(get_engine(tmp_path), BloomEngine)


def test_retired_use_migrate_points_to_manual_command(tmp_path):
    result = CliRunner().invoke(cli, ["engines", "use", "seed", "--migrate"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert "one retrieval pipeline" in result.output
    assert "poppy migrate-engine" in result.output
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("deps_ok", [True, False])
def test_engines_shows_pipeline_dependencies_and_fallback(tmp_path, monkeypatch, deps_ok):
    from poppy.engine.registry import EngineInfo

    monkeypatch.setattr(
        registry,
        "list_engines",
        lambda: [
            EngineInfo("bloom", "Retrieval pipeline", True, deps_ok, None if deps_ok else "missing onnxruntime"),
            EngineInfo("seed", "Built-in keyword-only fallback", True, True, None),
        ],
    )
    result = CliRunner().invoke(cli, ["engines"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0, result.output
    active = next(line for line in result.output.splitlines() if "★" in line)
    assert ("bloom" if deps_ok else "seed") in active
    if not deps_ok:
        assert "missing onnxruntime" in result.output
        assert "Keyword-only fallback is in use" in result.output
    assert "keyword-only fallback" in result.output
    assert "Switch with" not in result.output
