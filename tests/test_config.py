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
    assert config.trags_api_url == "https://api.trags.ai"


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


# --- Default API host and the pin for installs that synced with the old one --

_OLD_API = "https://trags.ai"
_NEW_API = "https://api.trags.ai"


def _write_sync_state(poppy_dir: Path, *urls: str) -> None:
    remotes = {url: {"last_pushed_at": "2026-10-01T00:00:00+00:00", "pushed_count": 3} for url in urls}
    (poppy_dir / "sync_state.json").write_text(json.dumps({"remotes": remotes}))


def _saved(poppy_dir: Path) -> dict:
    return json.loads((poppy_dir / "config.json").read_text())


def test_fresh_install_defaults_to_api_host(tmp_path):
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    # Nothing to pin, so loading writes nothing.
    assert not (tmp_path / "config.json").exists()


def test_missing_poppy_dir_loads_default(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert load_config(poppy_dir=missing).trags_api_url == _NEW_API
    assert not missing.exists()


def test_existing_config_without_old_host_state_gets_new_default(tmp_path):
    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    _write_sync_state(tmp_path, "https://self-hosted.example")
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert _saved(tmp_path) == {"engine": "seed"}


def test_install_that_synced_with_new_host_is_not_pinned(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    _write_sync_state(tmp_path, _NEW_API)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert _saved(tmp_path) == {}


@pytest.mark.parametrize("state_key", [_OLD_API, _OLD_API + "/"])
def test_install_that_synced_with_old_host_stays_there(tmp_path, state_key):
    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    _write_sync_state(tmp_path, state_key)

    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"engine": "seed", "trags_api_url": _OLD_API}

    # Runs once: the pin is now an explicit value, so a second load reads it
    # back without rewriting the file.
    before = (tmp_path / "config.json").read_text()
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert (tmp_path / "config.json").read_text() == before


def test_old_host_state_without_config_file_is_pinned(tmp_path):
    """A key from the environment needs no config.json, yet can have synced."""
    _write_sync_state(tmp_path, _OLD_API)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}


def test_error_record_alone_pins_old_host(tmp_path):
    (tmp_path / "sync_state.json").write_text(json.dumps({"remotes": {_OLD_API: {"errors": {"auth": "401"}}}}))
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API


@pytest.mark.parametrize("content", ["not json", "[]", '{"remotes": []}'])
def test_unreadable_sync_state_uses_old_host_without_pinning(tmp_path, content):
    """Unknown state means the old host for this process only; the next load,
    once the file reads again, makes the real decision."""
    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    (tmp_path / "sync_state.json").write_text(content)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"engine": "seed"}


@_POSIX_ONLY
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
def test_permission_denied_sync_state_is_not_pinned(tmp_path):
    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    _write_sync_state(tmp_path, _NEW_API)
    state = tmp_path / "sync_state.json"
    state.chmod(0)
    try:
        assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    finally:
        state.chmod(0o600)
    assert _saved(tmp_path) == {"engine": "seed"}
    # Readable again, the state names the new host: no pin, then or later.
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert _saved(tmp_path) == {"engine": "seed"}


def _note_db_remote(poppy_dir: Path, url: str) -> None:
    from poppy.tombstones import TombstoneStore

    store = TombstoneStore(poppy_dir / "memories.db")
    store.note_remote_memories({"m1"}, url)
    store._conn.close()


def test_old_host_known_only_to_memory_db_is_pinned(tmp_path):
    """Sync records what a server holds before it writes sync_state.json, so a
    first sync whose state write failed leaves only the database record."""
    _note_db_remote(tmp_path, _OLD_API + "/")
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}


def test_memory_db_without_old_host_is_not_pinned(tmp_path):
    _note_db_remote(tmp_path, "https://self-hosted.example")
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert not (tmp_path / "config.json").exists()


def test_state_naming_new_host_skips_memory_db(tmp_path):
    _note_db_remote(tmp_path, _OLD_API)
    _write_sync_state(tmp_path, _NEW_API)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API


def test_empty_memory_db_is_a_fresh_install(tmp_path):
    (tmp_path / "memories.db").write_bytes(b"")
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert not (tmp_path / "config.json").exists()


def test_unreadable_memory_db_uses_old_host_without_pinning(tmp_path):
    """An encrypted or corrupt store cannot show what it holds, so the install
    stays on the old host for this process and nothing is written."""
    (tmp_path / "memories.db").write_bytes(b"not a database, or an encrypted one" * 200)
    cfg = load_config(poppy_dir=tmp_path)
    assert cfg.trags_api_url == _OLD_API
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "sync_state.json").exists()


def test_failed_db_probe_retries_on_next_load(tmp_path, monkeypatch):
    import poppy.db

    _note_db_remote(tmp_path, _OLD_API)

    def broken_probe(_path):
        raise OSError("probe failed")

    monkeypatch.setattr(poppy.db, "read_only_probe", broken_probe)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "sync_state.json").exists()

    monkeypatch.undo()
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}


def test_failed_db_probe_on_install_that_never_synced_still_defers(tmp_path, monkeypatch):
    import poppy.db

    _note_db_remote(tmp_path, "https://self-hosted.example")
    monkeypatch.setattr(poppy.db, "read_only_probe", lambda _path: (_ for _ in ()).throw(OSError("locked")))
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API

    monkeypatch.undo()
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert not (tmp_path / "config.json").exists()


def test_provisional_old_host_never_reaches_disk(tmp_path):
    """A later save in the same process (key migration, setup, any config set)
    must not turn this load's guess into a pin."""
    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    (tmp_path / "sync_state.json").write_text("not json")
    cfg = load_config(poppy_dir=tmp_path)
    assert cfg.trags_api_url == _OLD_API
    cfg.set("auto-sync", "off")
    save_config(cfg)
    assert _saved(tmp_path) == {"engine": "seed", "auto_sync": "off"}


def test_cli_save_does_not_write_provisional_old_host(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    (tmp_path / "sync_state.json").write_text("not json")
    result = CliRunner().invoke(cli, ["config", "set", "telemetry", "off"])
    assert result.exit_code == 0, result.output
    assert "trags_api_url" not in _saved(tmp_path)


def test_explicit_url_set_while_provisional_is_written(tmp_path):
    (tmp_path / "sync_state.json").write_text("not json")
    cfg = load_config(poppy_dir=tmp_path)
    cfg.set("trags-api-url", _OLD_API)
    save_config(cfg)
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}


def test_non_string_url_on_disk_is_treated_as_absent(tmp_path):
    (tmp_path / "config.json").write_text('{"trags_api_url": 5}')
    _write_sync_state(tmp_path, _OLD_API)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}

    (tmp_path / "config.json").write_text('{"trags_api_url": null}')
    _write_sync_state(tmp_path, _NEW_API)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API


def test_pin_reread_ignores_non_string_url(tmp_path, monkeypatch):
    import poppy.config as config_module

    _write_sync_state(tmp_path, _OLD_API)
    real = config_module._synced_with_legacy_api_url

    def racing(poppy_dir):
        answer = real(poppy_dir)
        (tmp_path / "config.json").write_text('{"trags_api_url": ["x"], "engine": "seed"}')
        return answer

    monkeypatch.setattr(config_module, "_synced_with_legacy_api_url", racing)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API, "engine": "seed"}


@_POSIX_ONLY
def test_save_waits_for_a_pin_in_progress(tmp_path, monkeypatch):
    """A save cannot land between the pin's re-read and its replace: it waits
    for the lock, so neither write is lost."""
    import threading

    import poppy.config as config_module

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    _write_sync_state(tmp_path, _OLD_API)
    in_pin = threading.Event()
    release = threading.Event()
    real_write = config_module._write_config_payload

    def paused_write(poppy_dir, payload):
        if threading.current_thread().name == "pin":
            in_pin.set()
            assert release.wait(5)
        real_write(poppy_dir, payload)

    monkeypatch.setattr(config_module, "_write_config_payload", paused_write)
    pin = threading.Thread(target=config_module._write_api_url_pin, args=(tmp_path,), name="pin")
    pin.start()
    assert in_pin.wait(5)

    other = PoppyConfig(poppy_dir=tmp_path, engine="seed", auto_sync="off", trags_api_url=_OLD_API)
    save = threading.Thread(target=save_config, args=(other,), name="save")
    save.start()
    save.join(0.5)
    assert save.is_alive(), "save must wait while the pin holds the lock"

    release.set()
    pin.join(5)
    save.join(5)
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API, "engine": "seed", "auto_sync": "off"}


def test_pin_keeps_settings_saved_by_another_process(tmp_path, monkeypatch):
    """Another process can save between this load's read and its pin write. The
    pin re-reads the file and adds only the URL, and a URL saved meanwhile wins."""
    import poppy.config as config_module

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    _write_sync_state(tmp_path, _OLD_API)
    concurrent = {"engine": "seed", "auto_sync": "off", "trags_api_url": _NEW_API}
    real = config_module._synced_with_legacy_api_url

    def racing(poppy_dir):
        answer = real(poppy_dir)
        (tmp_path / "config.json").write_text(json.dumps(concurrent))
        return answer

    monkeypatch.setattr(config_module, "_synced_with_legacy_api_url", racing)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API
    assert _saved(tmp_path) == concurrent


def test_pin_adds_only_the_url_to_the_file_on_disk(tmp_path, monkeypatch):
    import poppy.config as config_module

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    _write_sync_state(tmp_path, _OLD_API)
    real = config_module._synced_with_legacy_api_url

    def racing(poppy_dir):
        answer = real(poppy_dir)
        (tmp_path / "config.json").write_text('{"engine": "seed", "auto_sync": "off"}')
        return answer

    monkeypatch.setattr(config_module, "_synced_with_legacy_api_url", racing)
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"engine": "seed", "auto_sync": "off", "trags_api_url": _OLD_API}


def test_pin_and_key_migration_store_the_key_once(tmp_path, monkeypatch):
    import poppy.config as config_module

    stores = []
    monkeypatch.setattr(config_module, "_store_trags_key_in_keychain", lambda cfg: stores.append(1) or True)
    (tmp_path / "config.json").write_text('{"trags_api_key": "usr_plain"}')
    _write_sync_state(tmp_path, _OLD_API)

    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert stores == [1]
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}


@pytest.mark.parametrize("explicit", ["https://self-hosted.example", _NEW_API, _OLD_API + "/"])
def test_explicit_url_is_untouched_by_old_host_state(tmp_path, explicit):
    (tmp_path / "config.json").write_text(json.dumps({"trags_api_url": explicit}))
    _write_sync_state(tmp_path, _OLD_API)
    before = (tmp_path / "config.json").read_text()
    assert load_config(poppy_dir=tmp_path).trags_api_url == explicit
    assert (tmp_path / "config.json").read_text() == before


def test_switching_a_pinned_install_to_the_default_sticks(tmp_path):
    """Setting the default on a pinned install is written down, or the next
    load would see no URL plus old-host state and pin it straight back."""
    _write_sync_state(tmp_path, _OLD_API)
    cfg = load_config(poppy_dir=tmp_path)
    assert cfg.trags_api_url == _OLD_API

    cfg.set("trags-api-url", _NEW_API)
    save_config(cfg)

    assert _saved(tmp_path)["trags_api_url"] == _NEW_API
    assert load_config(poppy_dir=tmp_path).trags_api_url == _NEW_API


def test_default_url_is_not_written_without_old_host_state(tmp_path):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("trags-api-url", _NEW_API)
    save_config(cfg)
    assert "trags_api_url" not in _saved(tmp_path)


@_POSIX_ONLY
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes")
def test_read_only_poppy_dir_still_uses_old_host(tmp_path):
    _write_sync_state(tmp_path, _OLD_API)
    tmp_path.chmod(0o500)
    try:
        assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    finally:
        tmp_path.chmod(0o700)
    assert not (tmp_path / "config.json").exists()
    # Writable again: the next load records the pin.
    assert load_config(poppy_dir=tmp_path).trags_api_url == _OLD_API
    assert _saved(tmp_path) == {"trags_api_url": _OLD_API}


def test_sync_status_reports_the_host_in_use(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")
    runner = CliRunner()

    fresh = runner.invoke(cli, ["sync", "status"])
    assert fresh.exit_code == 0, fresh.output
    assert f"url:             {_NEW_API}" in fresh.output

    _write_sync_state(tmp_path, _OLD_API)
    pinned = runner.invoke(cli, ["sync", "status"])
    assert pinned.exit_code == 0, pinned.output
    assert f"url:             {_OLD_API}" in pinned.output
    assert "pushed (total):  3" in pinned.output
