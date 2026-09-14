import json
import os
import stat
import sys
from pathlib import Path

import pytest

from poppy.config import (
    _REGISTRY,
    _SETTABLE,
    PoppyConfig,
    _registry_field_names,
    load_config,
    resolve_trags_api_key,
    save_config,
)

_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")


def test_default_config():
    config = PoppyConfig()
    assert config.poppy_dir == Path.home() / ".poppy"
    assert config.obsidian_vault is None
    assert config.trags_api_key is None
    assert config.trags_api_url == "https://trags.ai"


def test_save_and_load_config(tmp_path):
    config = PoppyConfig(poppy_dir=tmp_path)
    config.obsidian_vault = Path("/Users/test/cortex")
    save_config(config)

    loaded = load_config(poppy_dir=tmp_path)
    assert loaded.obsidian_vault == Path("/Users/test/cortex")


def test_disabled_projects_round_trip(tmp_path):
    """The per-project capture deny-list persists and reloads."""
    config = PoppyConfig(poppy_dir=tmp_path)
    assert config.disabled_projects == []  # default empty
    assert config.disable_project("client-secret") is True
    assert config.disable_project("client-secret") is False  # idempotent add
    save_config(config)

    loaded = load_config(poppy_dir=tmp_path)
    assert loaded.disabled_projects == ["client-secret"]
    assert loaded.is_project_disabled("client-secret") is True
    assert loaded.is_project_disabled("other") is False
    assert loaded.is_project_disabled(None) is False

    assert loaded.enable_project("client-secret") is True
    assert loaded.enable_project("client-secret") is False  # idempotent remove
    save_config(loaded)
    # Empty list is not persisted, so a reload sees the default.
    assert (tmp_path / "config.json").read_text().find("disabled_projects") == -1
    assert load_config(poppy_dir=tmp_path).disabled_projects == []


def test_disabled_projects_ignores_junk_entries(tmp_path):
    """A hand-edited config with non-strings / blanks / whitespace / dupes loads
    cleanly: entries are stripped, blank/whitespace-only dropped, dupes removed."""
    (tmp_path / "config.json").write_text('{"disabled_projects": ["a", "a", "", "   ", 5, null, "  b  ", "b"]}')
    loaded = load_config(poppy_dir=tmp_path)
    assert loaded.disabled_projects == ["a", "b"]


def test_redaction_config_round_trip_and_junk_cleanup(tmp_path):
    config = PoppyConfig(
        poppy_dir=tmp_path,
        redaction_literals=["staging-password", "api-secret"],
        redaction_env_vars=["DATABASE_URL", "SERVICE_TOKEN"],
    )
    save_config(config)

    persisted = json.loads((tmp_path / "config.json").read_text())
    assert persisted["redaction"] == {
        "literals": ["staging-password", "api-secret"],
        "env_vars": ["DATABASE_URL", "SERVICE_TOKEN"],
    }
    loaded = load_config(tmp_path)
    assert loaded.redaction_literals == ["staging-password", "api-secret"]
    assert loaded.redaction_env_vars == ["DATABASE_URL", "SERVICE_TOKEN"]

    (tmp_path / "config.json").write_text(
        '{"redaction": {'
        '"literals": [" keep-me ", "keep-me", "", 7, null], '
        '"env_vars": [" TOKEN ", "TOKEN", "   ", false]}}'
    )
    loaded = load_config(tmp_path)
    assert loaded.redaction_literals == ["keep-me"]
    assert loaded.redaction_env_vars == ["TOKEN"]


@pytest.mark.parametrize(
    "redaction",
    [None, [], "bad", {}, {"literals": []}, {"literals": [], "env_vars": "bad"}],
)
def test_malformed_redaction_section_is_ignored(tmp_path, redaction):
    (tmp_path / "config.json").write_text(json.dumps({"redaction": redaction}))
    loaded = load_config(tmp_path)
    assert loaded.redaction_literals == []
    assert loaded.redaction_env_vars == []


def test_partial_redaction_section_loads_the_valid_half(tmp_path):
    """A hand-edited config with one missing or malformed list must still mask
    with the other: dropping valid entries would silently unprotect secrets."""
    (tmp_path / "config.json").write_text(json.dumps({"redaction": {"literals": ["keep-this-secret"]}}))
    loaded = load_config(tmp_path)
    assert loaded.redaction_literals == ["keep-this-secret"]
    assert loaded.redaction_env_vars == []

    (tmp_path / "config.json").write_text(json.dumps({"redaction": {"literals": "wrong", "env_vars": ["KEEP_TOKEN"]}}))
    loaded = load_config(tmp_path)
    assert loaded.redaction_literals == []
    assert loaded.redaction_env_vars == ["KEEP_TOKEN"]


def test_load_config_missing_file(tmp_path):
    loaded = load_config(poppy_dir=tmp_path)
    assert loaded.obsidian_vault is None
    assert loaded.trags_api_key is None


def test_config_creates_directory(tmp_path):
    poppy_dir = tmp_path / "poppy"
    config = PoppyConfig(poppy_dir=poppy_dir)
    save_config(config)
    assert poppy_dir.exists()
    assert (poppy_dir / "config.json").exists()


def test_config_set_get():
    config = PoppyConfig()
    config.set("obsidian-vault", "/Users/test/cortex")
    assert config.obsidian_vault == Path("/Users/test/cortex")

    config.set("trags-api-key", "sk-test123")
    assert config.trags_api_key == "sk-test123"


def test_config_set_unknown_key():
    config = PoppyConfig()
    try:
        config.set("unknown-key", "value")
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown-key" in str(e)


# Config.json holds the plaintext API key, so it (and ~/.poppy) must be
# owner-only. Default umask would otherwise leave the file world-readable 0644.
@_POSIX_ONLY
def test_save_config_file_is_owner_only(tmp_path):
    config = PoppyConfig(poppy_dir=tmp_path)
    config.set("trags-api-key", "usr_secret")
    save_config(config)
    mode = stat.S_IMODE((tmp_path / "config.json").stat().st_mode)
    assert mode == 0o600, f"config.json mode {oct(mode)} is not 0600"


@_POSIX_ONLY
def test_save_config_dir_is_owner_only(tmp_path):
    poppy_dir = tmp_path / "poppy"
    save_config(PoppyConfig(poppy_dir=poppy_dir))
    mode = stat.S_IMODE(poppy_dir.stat().st_mode)
    assert mode == 0o700, f"~/.poppy mode {oct(mode)} is not 0700"


@_POSIX_ONLY
def test_save_config_tightens_preexisting_loose_perms(tmp_path):
    # A dir/file that already exist with world-readable perms must be tightened,
    # not left as-is (O_CREAT's mode is ignored for existing files).
    poppy_dir = tmp_path / "poppy"
    poppy_dir.mkdir()
    os.chmod(poppy_dir, 0o755)
    (poppy_dir / "config.json").write_text("{}")
    os.chmod(poppy_dir / "config.json", 0o644)

    config = PoppyConfig(poppy_dir=poppy_dir)
    config.set("trags-api-key", "usr_secret")
    save_config(config)

    assert stat.S_IMODE((poppy_dir / "config.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(poppy_dir.stat().st_mode) == 0o700
    # And the key still resolves after the secure rewrite, whether it landed in
    # the keychain (backend available) or fell back to config.json.
    assert resolve_trags_api_key(load_config(poppy_dir)) == "usr_secret"


# ---------- consent tri-state + legacy migration ----------


def test_consent_defaults_to_pending_and_is_not_persisted(tmp_path):
    config = PoppyConfig(poppy_dir=tmp_path)
    assert config.consent == "pending"
    save_config(config)
    assert '"consent"' not in (tmp_path / "config.json").read_text()


def test_consent_save_load_roundtrip(tmp_path):
    for value in ("granted", "denied"):
        config = PoppyConfig(poppy_dir=tmp_path)
        config.consent = value
        save_config(config)
        assert load_config(tmp_path).consent == value


def test_legacy_consolidate_enabled_true_migrates_to_granted(tmp_path):
    (tmp_path / "config.json").write_text('{"consolidate_enabled": true}')
    assert load_config(tmp_path).consent == "granted"


def test_legacy_consolidate_enabled_false_migrates_to_denied(tmp_path):
    (tmp_path / "config.json").write_text('{"consolidate_enabled": false}')
    assert load_config(tmp_path).consent == "denied"


def test_explicit_consent_wins_over_legacy_bool(tmp_path):
    (tmp_path / "config.json").write_text('{"consent": "denied", "consolidate_enabled": true}')
    assert load_config(tmp_path).consent == "denied"


def test_invalid_consent_value_stays_pending(tmp_path):
    (tmp_path / "config.json").write_text('{"consent": "maybe"}')
    assert load_config(tmp_path).consent == "pending"


# ---------- Recall relevance floor config ----------


def test_relevance_floor_defaults_off():
    config = PoppyConfig()
    assert config.recall_min_score == 0.0
    assert config.session_start_min_score == 0.0


def test_set_and_roundtrip_relevance_floors(tmp_path):
    config = PoppyConfig(poppy_dir=tmp_path)
    config.set("recall-min-score", "0.3")
    config.set("session-start-min-score", "0.5")
    assert config.recall_min_score == 0.3
    assert config.session_start_min_score == 0.5
    save_config(config)

    loaded = load_config(poppy_dir=tmp_path)
    assert loaded.recall_min_score == 0.3
    assert loaded.session_start_min_score == 0.5


def test_relevance_floor_rejects_out_of_range(tmp_path):
    config = PoppyConfig(poppy_dir=tmp_path)
    with pytest.raises(ValueError):
        config.set("recall-min-score", "1.5")
    with pytest.raises(ValueError):
        config.set("recall-min-score", "-0.1")
    with pytest.raises(ValueError):
        config.set("recall-min-score", "notanumber")


def test_relevance_floor_load_clamps_bad_stored_value(tmp_path):
    import json

    from poppy.config import CONFIG_FILENAME

    (tmp_path / CONFIG_FILENAME).write_text(json.dumps({"recall_min_score": 9.0, "session_start_min_score": "oops"}))
    loaded = load_config(poppy_dir=tmp_path)
    # Out-of-range clamps to 1.0; unparseable degrades to off (0.0).
    assert loaded.recall_min_score == 1.0
    assert loaded.session_start_min_score == 0.0


# ---------- Config key registry ----------

# One round-trip sample per settable key: (input string, expected loaded value).
# trags-api-key is covered by the keychain tests above — its persistence is
# backend-dependent, so it does not fit this generic set/save/load loop.
_SETTABLE_SAMPLES = {
    "obsidian-vault": ("/tmp/poppy-vault-xyz", Path("/tmp/poppy-vault-xyz")),
    "trags-api-url": ("https://example.test", "https://example.test"),
    "consolidate-enabled": ("true", True),
    "consolidate-model": ("gpt-x", "gpt-x"),
    "consolidate-base-url": ("https://llm.test/v1", "https://llm.test/v1"),
    "consolidate-api-key": ("sk-xyz", "sk-xyz"),
    "auto-supersede": ("auto", "auto"),
    "auto-sync": ("off", "off"),
    "engine": ("seed", "seed"),
    "recall-min-score": ("0.4", 0.4),
    "session-start-min-score": ("0.6", 0.6),
    "update_check": ("off", False),
    "telemetry": ("off", False),
}


def test_registry_covers_every_config_field():
    """Every persisted dataclass field is owned by exactly one registry entry, so
    a future field added without an entry fails loudly here."""
    covered: list[str] = []
    for entry in _REGISTRY:
        covered.extend(entry.attrs)
    assert len(covered) == len(set(covered)), f"a field is claimed by >1 entry: {covered}"
    assert set(covered) == _registry_field_names()


def test_every_settable_key_has_a_round_trip_sample():
    """A newly added `config set` key must also get a round-trip sample below."""
    assert set(_SETTABLE_SAMPLES) == set(_SETTABLE) - {"trags-api-key"}


@pytest.mark.parametrize("name", sorted(_SETTABLE_SAMPLES))
def test_settable_key_set_save_load_round_trip(tmp_path, name, monkeypatch):
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)
    input_str, expected = _SETTABLE_SAMPLES[name]
    attr = _SETTABLE[name].attr

    config = PoppyConfig(poppy_dir=tmp_path)
    config.set(name, input_str)
    assert getattr(config, attr) == expected
    save_config(config)

    loaded = load_config(poppy_dir=tmp_path)
    assert getattr(loaded, attr) == expected


@pytest.mark.parametrize("key", ["consent", "encrypted", "redaction", "disabled_projects"])
def test_non_settable_keys_reject_config_set(key):
    """Keys owned by their own command (autocapture / encrypt / redaction) are not
    reachable through `config set`."""
    with pytest.raises(ValueError):
        PoppyConfig().set(key, "x")
