import json
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path

from poppy.paths import ensure_poppy_dir

CONFIG_FILENAME = "config.json"

# The Trags sync API key is kept in the OS keychain (the same credential store
# the encryption key uses), not in config.json, whenever a keychain backend is
# available. config.json still holds any consolidate API key and, on hosts with
# no keychain, the trags key as a documented fallback. ~/.poppy holds the local
# memory DB. These are owner-only secrets, so the file is 0600 and the directory
# 0700 (the latter via ``ensure_poppy_dir``). Default umask leaves new files
# world-readable (0644), which would leak them to other local accounts.
_CONFIG_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR  # 0o600

# Environment override for the Trags API key. A deliberate escape hatch for
# headless / CI hosts with no OS keychain, and the seam the test suite uses.
# When set it wins over both the keychain and config.json. The key is then
# visible to anything that can read the process environment, so it is the lesser
# option, documented as such.
TRAGS_API_KEY_ENV = "POPPY_TRAGS_API_KEY"

# Tri-state consent for auto-capture (ADR-0002). Owned by the
# ConsolidationPolicy; set via `poppy autocapture` / the setup prompt, not config set.
_CONSENT_VALUES = {"pending", "granted", "denied"}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off", ""}
_AUTO_SUPERSEDE_VALUES = {"off", "suggest", "auto"}
_AUTO_SYNC_VALUES = {"on", "off"}
# `poppy config set telemetry` accepts the same aliases the CLI accepted before
# the key moved into the registry. It deliberately does NOT treat "" as off (the
# empty string is off for the generic bool keys, via _BOOL_FALSE).
_TELEMETRY_ON = {"on", "true", "1", "yes"}
_TELEMETRY_OFF = {"off", "false", "0", "no"}


@dataclass
class PoppyConfig:
    poppy_dir: Path = field(default_factory=lambda: Path.home() / ".poppy")
    obsidian_vault: Path | None = None
    trags_api_key: str | None = None
    trags_api_url: str = "https://trags.ai"
    # Stop-hook consolidation. When enabled, the Stop hook tries the host CLI
    # (claude -p / cursor-agent -p / codex exec / gemini -p) first, then falls back to an
    # OpenAI-compatible endpoint configured here.
    consolidate_enabled: bool = False
    consolidate_model: str | None = None
    consolidate_base_url: str | None = None
    consolidate_api_key: str | None = None
    # Tri-state consent for auto-capture (ADR-0002): pending | granted |
    # denied. Migrated from the legacy consolidate_enabled bool on load; managed
    # by `poppy autocapture` and the setup prompt. Both granted and denied persist.
    # Consent evidence record (GDPR Art 7(1)); never rename with CLI surface changes.
    consent: str = "pending"
    # Per-project capture off switch. Project names for which
    # auto-capture is disabled — a scoping control layered on top of the global
    # consent, for a sensitive repo (employer/client code) the user wants to keep
    # out of memory without turning capture off everywhere. Managed by
    # `poppy autocapture off/on`, checked by the ConsolidationPolicy. Deny-list
    # only: an allowlist mode was explicitly rejected (it re-introduces the
    # per-project friction zero-touch exists to remove).
    disabled_projects: list[str] = field(default_factory=list)
    # User-defined capture redaction, layered on top of the built-in secret
    # shapes. These are deliberately not settable via `config set`: the dedicated
    # `poppy redaction` commands validate and mutate them as a coherent unit.
    redaction_literals: list[str] = field(default_factory=list)
    redaction_env_vars: list[str] = field(default_factory=list)
    # Conflict detection on writes.
    auto_supersede: str = "off"
    # Auto-sync to Trags after every local write. "on" (default) | "off".
    # Only active when trags_api_key is set.
    auto_sync: str = "on"
    # Retrieval engine for recall/remember on the runtime surface. Validated
    # against the registry in the engine key's parser. See `poppy engines`.
    engine: str = "bloom"
    # Anonymous usage telemetry. None = unanswered (off); True/False = an
    # explicit choice from the first-run prompt or `poppy telemetry on|off`.
    # POPPY_TELEMETRY_OFF=1 in the environment overrides this at read time;
    # see poppy.telemetry.is_enabled() for the full precedence.
    telemetry_enabled: bool | None = None
    # Anonymous daily PyPI version check. The telemetry environment opt-out
    # and the persistent telemetry opt-out (`poppy telemetry off`) are both
    # applied by poppy.update_check.status().
    update_check: bool = True
    # Whether the local store is encrypted at rest (see `poppy encrypt`). A
    # user-facing intent marker only; the actual open path in poppy.db.connect
    # decides from the DB file header + the .poppy-encrypted sentinel, never
    # from this flag, so a stale value here can never open the wrong driver.
    encrypted: bool = False
    # Recall relevance floor. A cross-encoder relevance probability in
    # [0, 1]; a retrieved memory is kept only when sigmoid(its CE score) >= this.
    # 0.0 = off, an exact no-op == today's top-k behaviour. See poppy.relevance
    # and RECOMMENDED_MIN_SCORE for the recommended value.
    recall_min_score: float = 0.0
    # SessionStart floor. The SessionStart hook injects the most recent memories
    # by pure recency, with no query to score against, so there is no relevance
    # signal to floor. Any value > 0 means "don't inject an unscored dump":
    # SessionStart shows only its status banner. 0.0 = off (inject as before).
    session_start_min_score: float = 0.0

    def set(self, key: str, value: str) -> object:
        """Parse and apply a `config set` value; returns the parsed value."""
        entry = _settable_entry(key)
        parsed = entry.parse(value, key)
        entry.apply(self, parsed)
        return parsed

    # --- Per-project capture off switch -------------------------

    def is_project_disabled(self, project: str | None) -> bool:
        """Whether auto-capture is disabled for ``project`` (None never matches)."""
        return bool(project) and project in self.disabled_projects

    def disable_project(self, project: str) -> bool:
        """Add ``project`` to the capture deny-list. Returns True if newly added."""
        if project in self.disabled_projects:
            return False
        self.disabled_projects.append(project)
        return True

    def enable_project(self, project: str) -> bool:
        """Remove ``project`` from the capture deny-list. Returns True if it was set."""
        if project not in self.disabled_projects:
            return False
        self.disabled_projects.remove(project)
        return True


# --- Config key registry --------------------------------------------------
#
# One declarative entry per config key is the single source of truth for the
# whole surface: `config set` parsing/validation, save_config's field
# enumeration, and load_config's per-field normalization. Adding a key is one
# dataclass field plus one registry entry -- a meta-test asserts every field
# (except poppy_dir) is covered by exactly one entry, so a field without one
# fails loudly.
#
# Each entry carries:
#   attr         -- the dataclass attribute(s) it owns (a tuple only for the
#                   redaction key, which maps two fields onto one JSON key).
#   settings_name -- the `config set` name, or None for keys set through their
#                   own command (consent via autocapture, redaction via
#                   `poppy redaction`, encrypted via `poppy encrypt`).
#   parse        -- (value, key) -> typed value for set(); None if not settable.
#   apply        -- how set() stores the parsed value (default: plain setattr).
#   save         -- writes the key into the on-disk dict, only when it differs
#                   from its default (byte-compatible with the old save_config).
#   load         -- reads + normalizes the key from the on-disk dict (the same
#                   tolerant parsing the old load_config did).


@dataclass(frozen=True)
class ConfigKey:
    attr: str | tuple[str, ...]
    settings_name: str | None
    parse: Callable[[str, str], object] | None
    apply: Callable[["PoppyConfig", object], None]
    save: Callable[["PoppyConfig", dict], None]
    load: Callable[["PoppyConfig", dict], None]
    # How `config set` echoes the value back. None -> echo the raw CLI input
    # (as every scalar key does); a hook maps the parsed value to its canonical
    # form (telemetry True/False -> "on"/"off"), so aliases echo normalized.
    display: Callable[[object], str] | None = None
    # True when ``apply`` already persists the change itself (telemetry's hook
    # goes through telemetry.set_enabled, which writes config.json + analytics.json
    # and latches the first-run notice); config_set then skips the outer save.
    self_persist: bool = False

    @property
    def attrs(self) -> tuple[str, ...]:
        return self.attr if isinstance(self.attr, tuple) else (self.attr,)


def _identity(value):
    return value


def _setattr_apply(attr: str) -> Callable[["PoppyConfig", object], None]:
    def apply(config: "PoppyConfig", value: object) -> None:
        setattr(config, attr, value)

    return apply


def _default_save(attr: str, default, to_json) -> Callable[["PoppyConfig", dict], None]:
    """Persist ``attr`` under its own name only when it differs from ``default``."""

    def save(config: "PoppyConfig", data: dict) -> None:
        value = getattr(config, attr)
        if value != default:
            data[attr] = to_json(value)

    return save


def _default_load(attr: str, coerce) -> Callable[["PoppyConfig", dict], None]:
    def load(config: "PoppyConfig", data: dict) -> None:
        if attr in data:
            setattr(config, attr, coerce(data[attr]))

    return load


def _key(
    attr,
    *,
    settings_name=None,
    default=None,
    to_json=_identity,
    coerce=_identity,
    parse=None,
    apply=None,
    save=None,
    load=None,
    display=None,
    self_persist=False,
) -> ConfigKey:
    if apply is None:
        apply = _setattr_apply(attr)
    if save is None:
        save = _default_save(attr, default, to_json)
    if load is None:
        load = _default_load(attr, coerce)
    return ConfigKey(
        attr=attr,
        settings_name=settings_name,
        parse=parse,
        apply=apply,
        save=save,
        load=load,
        display=display,
        self_persist=self_persist,
    )


# --- Parsers (set): same messages the hand-written branches raised -----------


def _bool_parser(label: str, form: str) -> Callable[[str, str], bool]:
    def parse(value: str, key: str) -> bool:
        v = value.strip().lower()
        if v in _BOOL_TRUE:
            return True
        if v in _BOOL_FALSE:
            return False
        raise ValueError(f"{label} must be {form}, got {value!r}")

    return parse


def _choice_parser(name: str, values: set[str]) -> Callable[[str, str], str]:
    def parse(value: str, key: str) -> str:
        v = value.strip().lower()
        if v not in values:
            raise ValueError(f"{name} must be one of {sorted(values)}, got {value!r}")
        return v

    return parse


def _parse_obsidian(value: str, key: str) -> Path:
    return Path(value)


def _parse_str(value: str, key: str) -> str:
    return value


def _parse_engine(value: str, key: str) -> str:
    # Lazy import — keeps the registry off the hot path for callers that never
    # touch the engine setting. Legacy names (petal, speaker_closet, best,
    # baseline) map silently to their current builtin equivalents.
    from poppy.engine.registry import canonical_name, known_names  # noqa: PLC0415

    v = canonical_name(value.strip())
    valid = known_names()
    if v not in valid:
        raise ValueError(
            f"engine must be one of {len(valid)} known engines, got {value!r}. "
            f"Run `poppy engines` for the full list with descriptions."
        )
    return v


def _parse_score(value: str, key: str) -> float:
    try:
        score = float(value)
    except ValueError:
        raise ValueError(f"{key} must be a number in [0, 1], got {value!r}") from None
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {score}")
    return score


def _parse_telemetry(value: str, key: str) -> bool:
    v = value.strip().lower()
    if v in _TELEMETRY_ON:
        return True
    if v in _TELEMETRY_OFF:
        return False
    raise ValueError("telemetry must be on or off")


def _load_telemetry_enabled(config: "PoppyConfig", data: dict) -> None:
    """Read the stored answer, accepting only a real JSON boolean.

    The generic loader coerces with ``bool()``, where every non-empty string is
    true, so a hand-edited ``"false"`` or ``"off"`` would have read as an
    opt-in and turned telemetry on. Anything that is not a boolean, ``null``
    included, means the question is still unanswered: telemetry stays off and
    the user is asked once rather than being held to a choice they never made.
    """
    value = data.get("telemetry_enabled")
    config.telemetry_enabled = value if isinstance(value, bool) else None


def _apply_telemetry(config: "PoppyConfig", enabled: object) -> None:
    # telemetry.set_enabled writes telemetry_enabled through config AND mirrors
    # analytics.json (+ latches the first-run notice), so the switch keeps its
    # side effects. Mirror the flag onto this object too so a following
    # save_config on the same instance stays consistent with what was written.
    config.telemetry_enabled = bool(enabled)
    from poppy import telemetry  # noqa: PLC0415

    telemetry.set_enabled(config.poppy_dir, bool(enabled))


# --- Loaders (load): the tolerant parsing load_config used to inline ---------


def _clean_str_list(entries: list) -> list[str]:
    """Strip, drop blanks/non-strings, dedupe, preserve order.

    A hand-edited config can carry junk (non-strings, blank or whitespace-only
    entries, dupes); the policy / redaction checks must never choke on it.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        entry = entry.strip()
        if entry and entry not in seen:
            seen.add(entry)
            cleaned.append(entry)
    return cleaned


def _choice_loader(attr: str, values: set[str]) -> Callable[["PoppyConfig", dict], None]:
    def load(config: "PoppyConfig", data: dict) -> None:
        if attr in data:
            v = str(data[attr]).strip().lower()
            if v in values:
                setattr(config, attr, v)

    return load


def _load_consent(config: "PoppyConfig", data: dict) -> None:
    # An explicit `consent` is authoritative; otherwise migrate the legacy
    # consolidate_enabled bool: true -> granted (grandfather), false -> denied
    # (opt-out persists), unset -> pending.
    if "consent" in data:
        v = str(data["consent"]).strip().lower()
        if v in _CONSENT_VALUES:
            config.consent = v
    elif "consolidate_enabled" in data:
        config.consent = "granted" if bool(data["consolidate_enabled"]) else "denied"


def _load_disabled_projects(config: "PoppyConfig", data: dict) -> None:
    if isinstance(data.get("disabled_projects"), list):
        config.disabled_projects = _clean_str_list(data["disabled_projects"])


def _save_redaction(config: "PoppyConfig", data: dict) -> None:
    if config.redaction_literals or config.redaction_env_vars:
        data["redaction"] = {
            "literals": config.redaction_literals,
            "env_vars": config.redaction_env_vars,
        }


def _load_redaction(config: "PoppyConfig", data: dict) -> None:
    redaction = data.get("redaction")
    if not isinstance(redaction, dict):
        return
    # The redaction engine, rather than config parsing, owns semantic validation
    # such as minimum lengths and environment-variable name syntax. Keeping
    # invalid strings here lets `poppy doctor` explain exactly which hand-edited
    # entries it skipped. Each list loads independently: a hand-edited config with
    # one malformed (or missing) half must not silently drop the valid half's
    # masking.
    for json_key, attr in (("literals", "redaction_literals"), ("env_vars", "redaction_env_vars")):
        entries = redaction.get(json_key)
        if isinstance(entries, list):
            setattr(config, attr, _clean_str_list(entries))


def _load_engine(config: "PoppyConfig", data: dict) -> None:
    # No registry validation here — load_config sits on the hot path for every
    # CLI invocation. The runtime falls back gracefully if the stored name is
    # unknown. Legacy names from older installs map silently
    # (petal/speaker_closet/best -> bloom, baseline -> seed).
    if "engine" in data:
        v = str(data["engine"]).strip()
        if v:
            from poppy.engine.registry import canonical_name  # noqa: PLC0415

            config.engine = canonical_name(v)


def _load_update_check(config: "PoppyConfig", data: dict) -> None:
    if "update_check" not in data:
        return
    value = data["update_check"]
    if isinstance(value, bool):
        config.update_check = value
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BOOL_TRUE | _BOOL_FALSE:
            config.update_check = normalized in _BOOL_TRUE


def _score_loader(attr: str) -> Callable[["PoppyConfig", dict], None]:
    # Clamp to [0, 1] on read; an out-of-range or unparseable stored value
    # degrades to off rather than erroring the hot path.
    def load(config: "PoppyConfig", data: dict) -> None:
        if attr in data:
            try:
                v = float(data[attr])
            except (TypeError, ValueError):
                v = 0.0
            setattr(config, attr, min(1.0, max(0.0, v)))

    return load


def _save_trags_api_key(config: "PoppyConfig", data: dict) -> None:
    # Prefer the OS keychain over plaintext in config.json.
    #   ""    -> an explicit user clear: drop any keychain entry, write nothing.
    #   value -> store in the keychain; only fall back to the file (documented,
    #            for headless / CI) when the keychain write cannot be read back.
    #   None  -> not materialized in this object (e.g. already in the keychain);
    #            leave the keychain untouched and write nothing.
    if config.trags_api_key == "":
        _delete_trags_key_from_keychain(config)
    elif config.trags_api_key is not None:
        if not _store_trags_key_in_keychain(config):
            data["trags_api_key"] = config.trags_api_key


def _load_trags_api_key(config: "PoppyConfig", data: dict) -> None:
    if "trags_api_key" in data:
        config.trags_api_key = data["trags_api_key"]


# The registry order is the on-disk write order, kept byte-identical to the old
# save_config. SETTINGS_MAP and _SETTABLE are derived views of the settable
# entries (settings_name is not None).
_REGISTRY: tuple[ConfigKey, ...] = (
    _key(
        "obsidian_vault", settings_name="obsidian-vault", default=None, to_json=str, coerce=Path, parse=_parse_obsidian
    ),
    _key(
        "trags_api_key",
        settings_name="trags-api-key",
        parse=_parse_str,
        save=_save_trags_api_key,
        load=_load_trags_api_key,
    ),
    _key("trags_api_url", settings_name="trags-api-url", default="https://trags.ai", parse=_parse_str),
    _key(
        "consolidate_enabled",
        settings_name="consolidate-enabled",
        default=False,
        coerce=bool,
        parse=_bool_parser("consolidate-enabled", "true/false"),
    ),
    _key("consolidate_model", settings_name="consolidate-model", default=None, parse=_parse_str),
    _key("consolidate_base_url", settings_name="consolidate-base-url", default=None, parse=_parse_str),
    _key("consolidate_api_key", settings_name="consolidate-api-key", default=None, parse=_parse_str),
    _key("consent", default="pending", save=_default_save("consent", "pending", _identity), load=_load_consent),
    _key(
        "disabled_projects",
        default=[],
        save=_default_save("disabled_projects", [], _identity),
        load=_load_disabled_projects,
    ),
    _key(
        ("redaction_literals", "redaction_env_vars"),
        save=_save_redaction,
        load=_load_redaction,
    ),
    _key(
        "auto_supersede",
        settings_name="auto-supersede",
        default="off",
        parse=_choice_parser("auto-supersede", _AUTO_SUPERSEDE_VALUES),
        load=_choice_loader("auto_supersede", _AUTO_SUPERSEDE_VALUES),
    ),
    _key(
        "auto_sync",
        settings_name="auto-sync",
        default="on",
        parse=_choice_parser("auto-sync", _AUTO_SYNC_VALUES),
        load=_choice_loader("auto_sync", _AUTO_SYNC_VALUES),
    ),
    _key("engine", settings_name="engine", default="bloom", parse=_parse_engine, load=_load_engine),
    _key(
        "telemetry_enabled",
        settings_name="telemetry",
        default=None,
        parse=_parse_telemetry,
        apply=_apply_telemetry,
        load=_load_telemetry_enabled,
        display=lambda parsed: "on" if parsed else "off",
        self_persist=True,
    ),
    _key(
        "update_check",
        settings_name="update_check",
        default=True,
        parse=_bool_parser("update_check", "on/off"),
        load=_load_update_check,
    ),
    _key("encrypted", default=False, coerce=bool),
    _key(
        "recall_min_score",
        settings_name="recall-min-score",
        default=0.0,
        parse=_parse_score,
        load=_score_loader("recall_min_score"),
    ),
    _key(
        "session_start_min_score",
        settings_name="session-start-min-score",
        default=0.0,
        parse=_parse_score,
        load=_score_loader("session_start_min_score"),
    ),
)

_SETTABLE: dict[str, ConfigKey] = {e.settings_name: e for e in _REGISTRY if e.settings_name is not None}

# Backwards-compatible derived view (CLI settings name -> dataclass attr). Kept
# because it was importable module state; all settable entries own a single attr.
SETTINGS_MAP: dict[str, str] = {name: entry.attr for name, entry in _SETTABLE.items()}


def _settable_entry(key: str) -> ConfigKey:
    """The registry entry for a `config set` key, or ValueError if not settable."""
    entry = _SETTABLE.get(key)
    if entry is None:
        raise ValueError(f"Unknown config key: {key}. Valid keys: {', '.join(SETTINGS_MAP.keys())}")
    return entry


def config_key(key: str) -> ConfigKey:
    """Public lookup of a settable key's registry entry (raises ValueError if unknown)."""
    return _settable_entry(key)


def parse_setting(key: str, value: str) -> object:
    """Parse + validate a `config set` value with NO I/O, so a caller can reject a
    bad value before touching config.json. Raises ValueError with the same
    messages PoppyConfig.set() raises."""
    entry = _settable_entry(key)
    return entry.parse(value, key)


# --- Effective consolidate settings (env over config) ---------------------


@dataclass(frozen=True)
class ConsolidateSettings:
    """End-of-session LLM consolidation settings, env overriding config.

    ``POPPY_CONSOLIDATE_*`` env vars win over config.json; the OpenAI-compatible
    api_key also falls back to ``OPENAI_API_KEY``. ``max_items`` is read lazily
    (only when accessed) so building this for the model/base_url/api_key trio
    never parses ``POPPY_CONSOLIDATE_MAX_ITEMS`` — matching the old split where a
    bad max-items value only surfaced where max-items was actually used.
    """

    model: str | None
    base_url: str | None
    api_key: str | None

    @property
    def max_items(self) -> int:
        return int(os.environ.get("POPPY_CONSOLIDATE_MAX_ITEMS", "5"))


def resolved_consolidate_settings(cfg: PoppyConfig) -> ConsolidateSettings:
    """The single seam for consolidate model/base_url/api_key/max_items precedence."""
    model = os.environ.get("POPPY_CONSOLIDATE_MODEL") or cfg.consolidate_model
    # OPENAI_BASE_URL is the last resort so that pointing the shell at a local
    # or proxied OpenAI-compatible server keeps working without poppy-specific
    # config, mirroring the OPENAI_API_KEY fallback below.
    base_url = (
        os.environ.get("POPPY_CONSOLIDATE_BASE_URL") or cfg.consolidate_base_url or os.environ.get("OPENAI_BASE_URL")
    )
    api_key = os.environ.get("POPPY_CONSOLIDATE_API_KEY") or cfg.consolidate_api_key or os.environ.get("OPENAI_API_KEY")
    return ConsolidateSettings(model=model, base_url=base_url, api_key=api_key)


# --- Trags API key: OS keychain storage + resolution ----------------------
#
# The key lives in the OS keychain (via poppy.keychain, the module the
# encryption feature introduced), not plaintext config.json, whenever a keyring
# backend is available. resolve_trags_api_key() is the single read seam; every
# consumer goes through it so the storage location stays an implementation
# detail. This is local secret hygiene, not zero-knowledge: anything running as
# your user that can read the keychain can read the key.


def _trags_key_account(poppy_dir: Path) -> str:
    """Keychain account for a store's Trags API key, unique per store directory.

    Distinct from the ``db-key:`` accounts poppy.encryption uses, and keyed on
    the resolved directory so several stores (the real store, per-project dirs,
    test dirs) never clobber each other's key in one keychain.
    """
    return f"trags-api-key:{poppy_dir.resolve()}"


# Per-process cache of the keychain read, keyed by account. config.json is read
# on every CLI invocation and on the auto-sync trigger's hot path, so the
# keychain must not be hit on every read. The cache entry is invalidated by a
# stat signature of config.json (see _config_signature): every key change goes
# through save_config, which rewrites config.json atomically (mkstemp + rename)
# and so gives it a NEW inode, changing the signature. Keying on that signature
# rather than trusting same-process writes fixes cross-process staleness -- a
# long-lived process (the MCP server) that cached "no key" before onboarding
# will re-read the keychain once `poppy setup trags` in another process rewrites
# the file. Each account maps to (signature, value).
_TRAGS_KEY_CACHE: dict[str, tuple[object, str | None]] = {}


# What this process's last keychain attempt for a store's Trags key actually
# did, keyed by account: one of the ``keychain`` outcome constants. The write
# and delete helpers below record it so a caller reporting a fallback names the
# failure that happened, instead of probing a different entry afterwards and
# classifying a different attempt at a different time. Absent means
# this process has not tried.
_TRAGS_KEY_OUTCOME: dict[str, str] = {}


def trags_key_keychain_outcome(poppy_dir: Path) -> str | None:
    """What this process's last Trags-key keychain write or delete did.

    One of the ``poppy.keychain`` outcome constants, or ``None`` when this
    process never touched the keychain for this store (a fresh ``poppy doctor``),
    in which case the caller has to probe the session instead.
    """
    return _TRAGS_KEY_OUTCOME.get(_trags_key_account(poppy_dir))


def _reset_trags_key_cache() -> None:
    """Clear the keychain read cache. For tests and after an out-of-band change."""
    _TRAGS_KEY_CACHE.clear()
    _TRAGS_KEY_OUTCOME.clear()


def _config_signature(poppy_dir: Path) -> object:
    """A cheap stat fingerprint of config.json that changes on every rewrite.

    (inode, mtime_ns, size): save_config renames a fresh temp file over
    config.json, so the inode changes on every write regardless of the
    filesystem's mtime granularity. ``None`` when the file is absent (no key has
    ever been set for this store). One stat syscall, cheaper than the full config
    read the hot path already performs.
    """
    try:
        st = os.stat(poppy_dir / CONFIG_FILENAME)
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _keychain_get_trags_key(account: str) -> str | None:
    """Read the key from the OS keychain, or None if unavailable / absent.

    Any failure (no backend, ``keyring`` not installed, backend error) is treated
    as absent so resolution falls back to config.json rather than erroring.
    """
    try:
        from poppy import keychain  # noqa: PLC0415

        return keychain.get_secret(account)
    except Exception:
        return None


def resolve_trags_api_key(config: PoppyConfig) -> str | None:
    """The effective Trags API key: env override, else config file, else keychain.

    ``POPPY_TRAGS_API_KEY`` always wins. A plaintext key in config.json is
    only kept when the keychain could not be verified for it in the writing
    session, so the file is always the newest write; the keychain may hold the
    same or an older value. Without a file key, the OS keychain is
    consulted (cached per process). Returns None when no key can be read from
    any source in this session.
    """
    env = os.environ.get(TRAGS_API_KEY_ENV)
    if env:
        return env
    if config.trags_api_key:
        return config.trags_api_key
    account = _trags_key_account(config.poppy_dir)
    signature = _config_signature(config.poppy_dir)
    cached = _TRAGS_KEY_CACHE.get(account)
    if cached is None or cached[0] != signature:
        value = _keychain_get_trags_key(account)
        _TRAGS_KEY_CACHE[account] = (signature, value)
    else:
        value = cached[1]
    if value:
        return value
    # Nothing anywhere: the file key was already returned above, so whatever is
    # left on the config object here is empty.
    return None


def _store_trags_key_in_keychain(config: PoppyConfig) -> bool:
    """Best effort: put ``config.trags_api_key`` in the OS keychain.

    Probe before writing: an unreadable session must not attempt a write, and
    an exact existing value needs no rewrite. Otherwise, write and read back
    the exact value in this process before returning True. False (no backend,
    error, or mismatch) tells save_config to fall back to plaintext config.json,
    and records which of the three steps failed (see
    ``trags_key_keychain_outcome``) so a caller explaining the fallback names
    the real failure instead of guessing at one later.

    Gates on ``keychain.available`` and not ``keychain.writable``: this writes
    and reads back the real key anyway, which answers the same question for
    free, so a throwaway probe would only be a second round-trip (and, on a
    locked keychain, a second prompt) for a result already in hand.
    """
    from poppy import keychain  # noqa: PLC0415

    account = _trags_key_account(config.poppy_dir)

    def record(outcome: str) -> bool:
        _TRAGS_KEY_OUTCOME[account] = outcome
        return outcome == keychain.PROBE_OK

    try:
        if not keychain.available():
            return record(keychain.PROBE_NO_BACKEND)
        try:
            stored = keychain.get_secret(account)
        except Exception:
            # Reads are refused, so a write could not have been verified either.
            return record(keychain.PROBE_READBACK_FAILED)
        if stored != config.trags_api_key:
            try:
                keychain.set_secret(account, config.trags_api_key)
            except Exception:
                return record(keychain.PROBE_WRITE_REJECTED)
            try:
                written = keychain.get_secret(account)
            except Exception:
                return record(keychain.PROBE_READBACK_FAILED)
            if written != config.trags_api_key:
                return record(keychain.PROBE_READBACK_FAILED)
    except Exception:
        # A failure outside those three steps: nothing honest to name, so record
        # nothing and let the caller fall back to probing the session.
        _TRAGS_KEY_OUTCOME.pop(account, None)
        return False
    # Do not retain a cached miss even if the subsequent config rewrite fails.
    _TRAGS_KEY_CACHE.pop(account, None)
    return record(keychain.PROBE_OK)


def _delete_trags_key_from_keychain(config: PoppyConfig) -> None:
    """Remove the key from the keychain (best effort).

    A refused delete (``KeychainUnavailable``) is swallowed here so the
    file copy is still cleared, and recorded (see ``trags_key_keychain_outcome``)
    so the caller can report the entry that may have survived. The caller also
    re-reads ``trags_api_key_location`` as a backstop, for a refusal this layer
    could not see.

    No explicit cache invalidation: the surrounding save_config rewrites
    config.json, which changes the stat signature resolve_trags_api_key keys on,
    so the next resolve re-reads.
    """
    from poppy import keychain  # noqa: PLC0415

    account = _trags_key_account(config.poppy_dir)
    try:
        keychain.delete_secret(account)
    except Exception:
        # Only a real backend refusing the delete means a secret may have
        # survived it. Where no backend is usable at all (a headless host, where
        # the key was in config.json for exactly that reason) the delete fails
        # because there was nowhere to delete from, which is not a survival.
        _TRAGS_KEY_OUTCOME[account] = keychain.DELETE_REFUSED if keychain.available() else keychain.PROBE_NO_BACKEND
        return
    _TRAGS_KEY_OUTCOME[account] = keychain.PROBE_OK


def trags_api_key_location(poppy_dir: Path) -> str:
    """Where the key is actually persisted for this store: ``keychain``,
    ``config-file``, ``keychain-unreadable``, or ``none``.

    Reads real state rather than trusting ``keychain.available()``: a backend can
    report available yet reject writes or read-back, in which case save_config
    falls back to config.json. Without a file key, a read error is reported as
    ``keychain-unreadable``: this session cannot inspect the backend, so it
    cannot tell whether a key is stored there. Independent of the env override.

    Reports what the read found, so it needs no ``keychain.writable`` probe: a
    session that can read the key has it, whether or not it could write one.
    """
    cfg_path = poppy_dir / CONFIG_FILENAME
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
        except ValueError:
            data = {}
        if data.get("trags_api_key"):
            return "config-file"
    try:
        from poppy import keychain  # noqa: PLC0415

        if not keychain.available():
            return "none"
        if keychain.get_secret(_trags_key_account(poppy_dir)):
            return "keychain"
    except Exception:
        return "keychain-unreadable"
    return "none"


def _migrate_trags_key_to_keychain(config: PoppyConfig) -> None:
    """Move a plaintext key from config.json into the OS keychain, once.

    Transparent upgrade path: the first load after an upgrade on a
    keychain-capable host relocates any plaintext key and scrubs it from the
    file only after reading back the exact value in this process. A no-op when
    there is no plaintext key, or when the keychain write cannot be read back
    (the key stays in config.json as the documented headless fallback).
    Never raises: a migration problem must not break a config load.
    """
    if not config.trags_api_key:
        return
    if not _store_trags_key_in_keychain(config):
        return
    # Scrub from the in-memory copy and, via save_config, from the file on disk.
    # The save rewrites config.json, changing the signature resolve keys on, so
    # the next resolve re-reads the now-populated keychain.
    config.trags_api_key = None
    try:
        save_config(config)
    except Exception:
        # The key is safely in the keychain now; if the scrub write fails the
        # file still holds its (already-plaintext) copy and a later save retries.
        pass


def save_config(config: PoppyConfig) -> None:
    ensure_poppy_dir(config.poppy_dir)
    data: dict = {}
    for entry in _REGISTRY:
        entry.save(config, data)
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
        for entry in _REGISTRY:
            entry.load(config, data)
    # Transparent one-time upgrade: relocate any plaintext key into the keychain
    # and scrub it from the file. No-op once migrated, or on headless hosts.
    _migrate_trags_key_to_keychain(config)
    return config


# Fields that never persist to config.json (runtime-only state). The registry
# meta-test asserts every OTHER dataclass field is owned by exactly one entry.
_NON_PERSISTED_FIELDS = {"poppy_dir"}


def _registry_field_names() -> set[str]:
    """All dataclass fields the registry is expected to cover."""
    return {f.name for f in fields(PoppyConfig)} - _NON_PERSISTED_FIELDS
