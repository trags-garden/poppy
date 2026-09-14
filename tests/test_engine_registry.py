"""Tests for the engine registry, config plumbing, and runtime dispatch.

The public registry is a hardcoded two-engine catalog (bloom / seed). We assert
behavior against those names directly.
"""

import builtins

import pytest

import poppy.engine.registry as registry
from poppy.config import PoppyConfig, load_config, save_config
from poppy.engine.registry import EngineInfo, canonical_name, known_names, list_engines, resolve_engine
from poppy.engine.seed import SeedEngine
from poppy.runtime import get_engine


@pytest.fixture(autouse=True)
def _reset_retired_notices():
    """Notices are once-per-process; tests that assert on them need a clean slate."""
    registry._retired_notices_emitted.clear()
    yield
    registry._retired_notices_emitted.clear()


def test_list_engines_contains_two_builtins():
    rows = list_engines()
    names = [r.name for r in rows]
    assert names == ["bloom", "seed"]
    assert all(r.builtin for r in rows)
    # seed is FTS5 only — no optional deps, must always be ok.
    seed_row = next(r for r in rows if r.name == "seed")
    assert seed_row.deps_ok
    # bloom needs fastembed + numpy, both base dependencies, so the default
    # engine must be available on a plain install with no extras.
    bloom_row = next(r for r in rows if r.name == "bloom")
    assert bloom_row.deps_ok


def test_engine_info_has_description():
    info = next(e for e in list_engines() if e.name == "seed")
    assert info.description
    assert "\n" not in info.description


def test_bloom_probe_reports_underlying_import_error(monkeypatch):
    """A broken fastembed install is a broken environment, not a missing extra —
    the underlying error is what the user needs to see.
    """
    real_import = builtins.__import__

    def fail_fastembed(name, *args, **kwargs):
        if name == "fastembed":
            raise ImportError("onnxruntime wheel is missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_fastembed)

    info = registry._probe("bloom")

    assert not info.deps_ok
    assert "onnxruntime wheel is missing" in info.deps_error


def test_resolve_seed_returns_seed_engine(tmp_path):
    eng = resolve_engine("seed", tmp_path / "t.db")
    assert isinstance(eng, SeedEngine)


def test_resolve_unknown_engine_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match="Unknown engine"):
        resolve_engine("no_such_engine_xyz", tmp_path / "t.db")


def test_config_set_engine_validates_against_registry(tmp_path):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("engine", "seed")
    assert cfg.engine == "seed"
    with pytest.raises(ValueError, match="engine must be one of"):
        cfg.set("engine", "no_such_engine_xyz")


def test_config_save_load_roundtrip_engine(tmp_path):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("engine", "seed")
    save_config(cfg)
    loaded = load_config(tmp_path)
    assert loaded.engine == "seed"


def test_config_save_omits_engine_when_default(tmp_path):
    # Default = "bloom"; should not be persisted so the
    # file stays minimal for fresh installs.
    cfg = PoppyConfig(poppy_dir=tmp_path)
    assert cfg.engine == "bloom"
    save_config(cfg)
    raw = (tmp_path / "config.json").read_text()
    assert '"engine"' not in raw


def test_runtime_get_engine_honors_config(tmp_path):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("engine", "seed")
    save_config(cfg)
    eng = get_engine(tmp_path)
    assert isinstance(eng, SeedEngine)


def test_runtime_falls_back_when_chosen_engine_fails(tmp_path, monkeypatch, capsys):
    """A stored engine name that no longer resolves must not crash the runtime.

    We simulate this by writing a config entry directly (bypassing the set()
    validator that would otherwise reject the unknown name) and then asserting
    that get_engine() falls back without raising.
    """
    (tmp_path / "config.json").write_text('{"engine": "no_such_engine_xyz"}')
    eng = get_engine(tmp_path)
    # Should be BloomEngine, or the SeedEngine floor — either way, not a crash.
    # Runtime surfaces stay silent unless they explicitly opt in via error_sink.
    assert eng is not None
    assert capsys.readouterr().err == ""


def test_runtime_reports_configured_engine_failure_when_fallback_succeeds(tmp_path, monkeypatch):
    fallback = object()
    errors = []

    def resolve(name, _db_path, *, delegate_models_to_daemon=False):
        if name == "bloom":
            raise ImportError("engine construction failed")
        assert name == "seed"
        return fallback

    monkeypatch.setattr(registry, "resolve_engine", resolve)

    engine = get_engine(tmp_path, error_sink=errors.append)

    assert engine is fallback
    assert errors == ["bloom: ImportError: engine construction failed"]


# ---------- legacy engine-name fallback mapping ----------


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("petal", "bloom"),
        ("speaker_closet", "bloom"),
        ("best", "bloom"),
        ("baseline", "seed"),
    ],
)
def test_canonical_name_maps_legacy_names(legacy, expected):
    assert canonical_name(legacy) == expected


def test_petal_maps_to_bloom_silently(capsys):
    """petal WAS this engine — same architecture, same models, same model_id —
    so the rename must not nag anyone whose config still says petal.
    """
    assert canonical_name("petal") == "bloom"
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_sprout_maps_to_bloom_with_a_notice(capsys):
    """sprout's vectors came from a different bi-encoder, so silence would hide
    that those rows are keyword-only until migrate-engine re-embeds them.
    """
    assert canonical_name("sprout") == "bloom"
    err = capsys.readouterr().err
    assert "sprout" in err
    assert "migrate-engine" in err
    assert len(err.strip().splitlines()) == 1

    # Once per process, not once per lookup: this sits on the config hot path.
    assert canonical_name("sprout") == "bloom"
    assert capsys.readouterr().err == ""


def test_canonical_name_passes_through_builtins_and_unknowns():
    for name in ("bloom", "seed", "no_such_engine_xyz"):
        assert canonical_name(name) == name


def test_resolve_engine_accepts_legacy_baseline(tmp_path):
    # baseline was the dev-era FTS-only engine; it must resolve to seed.
    eng = resolve_engine("baseline", tmp_path / "t.db")
    assert isinstance(eng, SeedEngine)


def test_load_config_maps_legacy_engine_name_silently(tmp_path, capsys):
    # A config.json written by an older install must keep working without
    # any warning on load.
    (tmp_path / "config.json").write_text('{"engine": "speaker_closet"}')
    cfg = load_config(tmp_path)
    assert cfg.engine == "bloom"
    assert capsys.readouterr().err == ""


def test_load_config_maps_petal_to_bloom(tmp_path, capsys):
    (tmp_path / "config.json").write_text('{"engine": "petal"}')
    assert load_config(tmp_path).engine == "bloom"
    assert capsys.readouterr().err == ""


def test_load_config_maps_legacy_baseline_to_seed(tmp_path):
    (tmp_path / "config.json").write_text('{"engine": "baseline"}')
    assert load_config(tmp_path).engine == "seed"


def test_config_set_accepts_legacy_engine_name(tmp_path):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("engine", "baseline")
    assert cfg.engine == "seed"


def test_config_set_accepts_retired_engine_name(tmp_path):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("engine", "sprout")
    assert cfg.engine == "bloom"


def test_runtime_get_engine_with_legacy_baseline_config(tmp_path, capsys):
    # End to end: a stored legacy name resolves silently, no fallback warning.
    (tmp_path / "config.json").write_text('{"engine": "baseline"}')
    eng = get_engine(tmp_path)
    assert isinstance(eng, SeedEngine)
    assert capsys.readouterr().err == ""


def test_known_names_matches_list_engines():
    assert known_names() == [e.name for e in list_engines()]


def test_engine_info_is_a_dataclass_with_expected_fields():
    # Cheap shape check so future refactors don't break the CLI formatter silently.
    fields = {f for f in EngineInfo.__dataclass_fields__}
    assert fields == {"name", "description", "builtin", "deps_ok", "deps_error"}
