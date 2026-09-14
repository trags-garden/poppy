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
    notice = "configured engine 'bloom' unavailable (the onnx runtime failed to load), using 'seed'"

    first = runner.invoke(cli, ["stats"], env=env)
    second = runner.invoke(cli, ["stats"], env=env)

    assert first.exit_code == 0, first.output
    assert first.output.count(notice) == 1
    assert second.exit_code == 0, second.output
    assert notice not in second.output
    assert (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).is_file()


def test_changed_fallback_pair_renotifies_same_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The throttle keys on what substituted for what, so a same-day change in
    the fallback pair warns again instead of hiding a new degradation."""
    _make_bloom_unavailable(monkeypatch)
    today = datetime.date.today().isoformat()
    (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).write_text(f"{today} seed->bloom earlier reason\n")

    result = CliRunner().invoke(cli, ["stats"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "configured engine 'bloom' unavailable" in result.output
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
    assert "configured engine 'bloom' unavailable" in result.output
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
    assert "configured engine 'bloom' unavailable" not in hook_result.output
    assert "configured engine 'bloom' unavailable" not in serve_result.output
    assert not (tmp_path / _ENGINE_FALLBACK_NOTICE_FILE).exists()


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
    assert "configured engine 'bloom' unavailable" in result.output
