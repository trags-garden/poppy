import json
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILENAME = "config.json"

# Config.json holds the plaintext trags_api_key (and any consolidate
# API key), and ~/.poppy holds the local memory DB. Both are owner-only secrets,
# so the file is 0600 and the directory 0700. Default umask leaves new files
# world-readable (0644), which would leak the key to other local accounts.
_CONFIG_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600
_POPPY_DIR_MODE = stat.S_IRWXU  # 0o700

SETTINGS_MAP = {
    "obsidian-vault": "obsidian_vault",
    "trags-api-key": "trags_api_key",
    "trags-api-url": "trags_api_url",
    # End-of-session LLM consolidation (Stop hook)
    "consolidate-enabled": "consolidate_enabled",
    "consolidate-model": "consolidate_model",
    "consolidate-base-url": "consolidate_base_url",
    "consolidate-api-key": "consolidate_api_key",
    # LLM conflict detection on remember writes.
    # off (default) | suggest (warn after write) | auto (auto-supersede)
    "auto-supersede": "auto_supersede",
    # Auto-sync to Trags after every local write. on (default) | off.
    "auto-sync": "auto_sync",
    # Retrieval engine selection (see `poppy engines`). Default = "bloom".
    "engine": "engine",
}

# Tri-state consent for auto-capture (ADR-0002). Owned by the
# ConsolidationPolicy; set via `poppy consent` / the setup prompt, not config set.
_CONSENT_VALUES = {"pending", "granted", "denied"}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off", ""}
_AUTO_SUPERSEDE_VALUES = {"off", "suggest", "auto"}
_AUTO_SYNC_VALUES = {"on", "off"}


@dataclass
class PoppyConfig:
    poppy_dir: Path = field(default_factory=lambda: Path.home() / ".poppy")
    obsidian_vault: Path | None = None
    trags_api_key: str | None = None
    trags_api_url: str = "https://trags.ai"
    # Stop-hook consolidation. When enabled, the Stop hook tries the host CLI
    # (claude -p / codex exec / gemini -p) first, then falls back to an
    # OpenAI-compatible endpoint configured here.
    consolidate_enabled: bool = False
    consolidate_model: str | None = None
    consolidate_base_url: str | None = None
    consolidate_api_key: str | None = None
    # Tri-state consent for auto-capture (ADR-0002): pending | granted |
    # denied. Migrated from the legacy consolidate_enabled bool on load; managed
    # by `poppy consent` and the setup prompt. Both granted and denied persist.
    consent: str = "pending"
    # Conflict detection on writes.
    auto_supersede: str = "off"
    # Auto-sync to Trags after every local write. "on" (default) | "off".
    # Only active when trags_api_key is set.
    auto_sync: str = "on"
    # Retrieval engine for recall/remember on the runtime surface. Validated
    # against the registry in PoppyConfig.set(). See `poppy engines`. The
    # default "bloom" is a floating tier name pointing at the current local
    # champion — upgrading poppy can change what bloom resolves to.
    engine: str = "bloom"
    # Anonymous usage telemetry. Tri-state: None = unset (defaults to
    # on), True/False = explicit user choice via `poppy telemetry on|off`.
    # POPPY_TELEMETRY_OFF=1 in the environment overrides this at read time;
    # see poppy.telemetry.is_enabled() for the full precedence.
    telemetry_enabled: bool | None = None

    def set(self, key: str, value: str) -> None:
        attr = SETTINGS_MAP.get(key)
        if attr is None:
            raise ValueError(f"Unknown config key: {key}. Valid keys: {', '.join(SETTINGS_MAP.keys())}")
        if attr == "obsidian_vault":
            setattr(self, attr, Path(value))
        elif attr == "consolidate_enabled":
            v = value.strip().lower()
            if v in _BOOL_TRUE:
                self.consolidate_enabled = True
            elif v in _BOOL_FALSE:
                self.consolidate_enabled = False
            else:
                raise ValueError(f"consolidate-enabled must be true/false, got {value!r}")
        elif attr == "auto_supersede":
            v = value.strip().lower()
            if v not in _AUTO_SUPERSEDE_VALUES:
                raise ValueError(f"auto-supersede must be one of {sorted(_AUTO_SUPERSEDE_VALUES)}, got {value!r}")
            self.auto_supersede = v
        elif attr == "auto_sync":
            v = value.strip().lower()
            if v not in _AUTO_SYNC_VALUES:
                raise ValueError(f"auto-sync must be one of {sorted(_AUTO_SYNC_VALUES)}, got {value!r}")
            self.auto_sync = v
        elif attr == "engine":
            # Lazy import — keeps the registry off the hot path for callers
            # that never touch the engine setting.
            from poppy.engine.registry import canonical_name, known_names  # noqa: PLC0415

            # Legacy names (speaker_closet, best, baseline) map silently to
            # their current builtin equivalents.
            v = canonical_name(value.strip())
            valid = known_names()
            if v not in valid:
                raise ValueError(
                    f"engine must be one of {len(valid)} known engines, got {value!r}. "
                    f"Run `poppy engines` for the full list with descriptions."
                )
            self.engine = v
        else:
            setattr(self, attr, value)


def save_config(config: PoppyConfig) -> None:
    config.poppy_dir.mkdir(parents=True, exist_ok=True)
    # Tighten the dir even if it pre-existed with looser perms (mkdir's mode is
    # ignored for existing dirs and masked by umask for new ones).
    try:
        config.poppy_dir.chmod(_POPPY_DIR_MODE)
    except OSError:
        pass  # Best-effort; never block a config write on a chmod failure.
    data: dict = {}
    if config.obsidian_vault is not None:
        data["obsidian_vault"] = str(config.obsidian_vault)
    if config.trags_api_key is not None:
        data["trags_api_key"] = config.trags_api_key
    if config.trags_api_url != "https://trags.ai":
        data["trags_api_url"] = config.trags_api_url
    if config.consolidate_enabled:
        data["consolidate_enabled"] = True
    if config.consolidate_model is not None:
        data["consolidate_model"] = config.consolidate_model
    if config.consolidate_base_url is not None:
        data["consolidate_base_url"] = config.consolidate_base_url
    if config.consolidate_api_key is not None:
        data["consolidate_api_key"] = config.consolidate_api_key
    if config.consent != "pending":
        data["consent"] = config.consent
    if config.auto_supersede != "off":
        data["auto_supersede"] = config.auto_supersede
    if config.auto_sync != "on":
        data["auto_sync"] = config.auto_sync
    if config.engine != "bloom":
        data["engine"] = config.engine
    if config.telemetry_enabled is not None:
        data["telemetry_enabled"] = config.telemetry_enabled
    config_path = config.poppy_dir / CONFIG_FILENAME
    payload = json.dumps(data, indent=2)
    # Atomic, never-world-readable write: tempfile.mkstemp creates a fresh file
    # with 0600 honored (it is guaranteed-new, unlike os.open(O_CREAT, 0600)
    # whose mode is ignored for an existing 0644 file), so the plaintext key
    # never lands in a world-readable inode. os.replace then renames it over the
    # target atomically, so a crash can't leave a half-written or loose-perm
    # config, and an existing 0644 config is replaced by the 0600 inode.
    fd, tmp_name = tempfile.mkstemp(dir=str(config.poppy_dir), prefix=".config-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp_name, _CONFIG_FILE_MODE)  # belt-and-suspenders over mkstemp's 0600
        os.replace(tmp_name, config_path)
    except BaseException:
        # Never leave a stale temp file behind on any failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_config(poppy_dir: Path | None = None) -> PoppyConfig:
    poppy_dir = poppy_dir or Path.home() / ".poppy"
    config = PoppyConfig(poppy_dir=poppy_dir)
    config_path = poppy_dir / CONFIG_FILENAME
    if config_path.exists():
        data = json.loads(config_path.read_text())
        if "obsidian_vault" in data:
            config.obsidian_vault = Path(data["obsidian_vault"])
        if "trags_api_key" in data:
            config.trags_api_key = data["trags_api_key"]
        if "trags_api_url" in data:
            config.trags_api_url = data["trags_api_url"]
        if "consolidate_enabled" in data:
            config.consolidate_enabled = bool(data["consolidate_enabled"])
        if "consolidate_model" in data:
            config.consolidate_model = data["consolidate_model"]
        if "consolidate_base_url" in data:
            config.consolidate_base_url = data["consolidate_base_url"]
        if "consolidate_api_key" in data:
            config.consolidate_api_key = data["consolidate_api_key"]
        # Consent. An explicit `consent` is authoritative; otherwise
        # migrate the legacy consolidate_enabled bool: true -> granted
        # (grandfather), false -> denied (opt-out persists), unset -> pending.
        if "consent" in data:
            v = str(data["consent"]).strip().lower()
            if v in _CONSENT_VALUES:
                config.consent = v
        elif "consolidate_enabled" in data:
            config.consent = "granted" if bool(data["consolidate_enabled"]) else "denied"
        if "auto_supersede" in data:
            v = str(data["auto_supersede"]).strip().lower()
            if v in _AUTO_SUPERSEDE_VALUES:
                config.auto_supersede = v
        if "auto_sync" in data:
            v = str(data["auto_sync"]).strip().lower()
            if v in _AUTO_SYNC_VALUES:
                config.auto_sync = v
        if "engine" in data:
            # No registry validation here — load_config sits on the hot path
            # for every CLI invocation. The runtime will fall back gracefully
            # if the stored name is unknown. Legacy names from configs written
            # by older installs are mapped silently so they keep working
            # (speaker_closet/best -> bloom, baseline -> seed).
            v = str(data["engine"]).strip()
            if v:
                from poppy.engine.registry import canonical_name  # noqa: PLC0415

                config.engine = canonical_name(v)
        if "telemetry_enabled" in data:
            config.telemetry_enabled = bool(data["telemetry_enabled"])
    return config
