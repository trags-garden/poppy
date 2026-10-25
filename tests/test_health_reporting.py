"""Health reporting regressions that also run without the encryption extra."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy import encryption
from poppy.cli import doctor, main
from poppy.engine import _fastembed_loader as fe
from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SeedEngine
from poppy.errors import ModelUnavailableError


@pytest.mark.parametrize("failed_kind", ["bi", "cross"])
def test_doctor_reports_model_load_failure(tmp_path, monkeypatch, failed_kind):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("POPPY_FASTEMBED_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setattr(doctor, "check_version", lambda ctx: None)
    engine = BloomEngine(tmp_path / "memories.db")
    monkeypatch.setattr(main, "_get_engine", lambda: engine)

    def fail_loader(kind, model_name, cache_dir, *, local_files_only=False):
        assert local_files_only
        if kind == failed_kind:
            raise fe._model_unavailable(model_name, OSError("fake offline loader failure"))
        return object()

    monkeypatch.setattr(fe, "_load_fastembed", fail_loader)
    with pytest.raises(ModelUnavailableError) as failed:
        engine.check_models()

    result = CliRunner().invoke(main.cli, ["doctor"])

    assert result.exit_code == 1, result.output
    assert f"engine: FAIL, {failed.value}" in result.output
    assert "fake offline loader failure" in result.output
    assert "engine: OK" not in result.output


@pytest.mark.parametrize("engine_class", [BloomEngine, SeedEngine])
def test_doctor_reports_available_engine(tmp_path, monkeypatch, capsys, engine_class):
    from poppy.config import load_config, save_config

    engine = engine_class(tmp_path / "memories.db")
    cfg = load_config(tmp_path)
    cfg.engine = engine._engine_name
    save_config(cfg)
    calls = []

    def load(kind, model_name, cache_dir, *, local_files_only=False):
        assert local_files_only
        calls.append(kind)
        return object()

    monkeypatch.setattr(fe, "_load_fastembed", load)
    monkeypatch.setattr(main, "_get_engine", lambda: engine)
    ctx = doctor.DoctorContext()
    ctx.poppy_dir = tmp_path
    doctor.check_engine(ctx)
    doctor.check_engine(ctx)

    assert ctx.ok
    assert "engine: OK" in capsys.readouterr().out
    assert calls == (["bi", "cross"] if engine_class is BloomEngine else [])


def _marked_store(tmp_path: Path) -> None:
    (tmp_path / "memories.db").write_bytes(b"random non-SQLite bytes" * 100)
    (tmp_path / encryption.ENCRYPTION_SENTINEL).touch()


@pytest.mark.parametrize("verdict", ["corrupt", "wrong-key", "unknown", "unreadable"])
def test_deep_status_preserves_encryption_state(tmp_path, monkeypatch, verdict):
    _marked_store(tmp_path)
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    monkeypatch.setattr(encryption, "dependencies_installed", lambda: True)
    calls = []

    def probe(path, *, check_active_key=False):
        assert path == tmp_path
        assert check_active_key
        calls.append(path)
        return verdict

    monkeypatch.setattr(encryption, "_probe_open", probe)
    shallow = encryption.status(tmp_path)
    assert shallow.state == "encrypted"
    assert not calls

    status = encryption.status(tmp_path, deep=True)
    assert status.enabled
    assert status.encrypted_on_disk
    assert status.state == verdict
    result = CliRunner().invoke(main.cli, ["encrypt", "status", "--deep"])
    assert result.exit_code == 0, result.output
    assert "Encryption: on" in result.output
    assert "encrypted on disk: yes" in result.output
    assert "warning:" in result.output


def test_doctor_probes_marked_store(tmp_path, monkeypatch, capsys):
    _marked_store(tmp_path)
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    monkeypatch.setattr(encryption, "dependencies_installed", lambda: True)
    monkeypatch.setattr(encryption, "_probe_open", lambda path, **kwargs: "corrupt")
    ctx = doctor.DoctorContext()
    ctx.poppy_dir = tmp_path

    doctor.check_encryption(ctx)

    assert not ctx.ok
    assert "encryption at rest: FAIL" in capsys.readouterr().out


def test_repair_probes_marked_store(tmp_path, monkeypatch):
    _marked_store(tmp_path)
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr(encryption, "_probe_open", lambda path: "corrupt")
    original = (tmp_path / "memories.db").read_bytes()

    result = CliRunner().invoke(main.cli, ["encrypt", "repair"])

    assert result.exit_code == 0, result.output
    assert "corrupt" in result.output
    assert "nothing to repair" not in result.output
    assert (tmp_path / encryption.ENCRYPTION_SENTINEL).exists()
    assert (tmp_path / "memories.db").read_bytes() == original
