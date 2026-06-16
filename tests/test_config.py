import os
import stat
import sys
from pathlib import Path

import pytest

from poppy.config import PoppyConfig, load_config, save_config

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
    # And the key still round-trips after the secure rewrite.
    assert load_config(poppy_dir).trags_api_key == "usr_secret"


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
