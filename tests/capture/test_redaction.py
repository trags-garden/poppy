"""Secret redaction at capture ingest.

Covers each deny-listed token family, the mask-not-drop semantics (surrounding
memory text survives), a no-false-positive sanity pass on ordinary prose/code,
and that redaction actually runs at the reconciler choke point so every capture
path is covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poppy.capture import journal
from poppy.capture.orchestrator import build_capture_memories as _build_memories
from poppy.capture.reconciler import reconcile_and_ingest
from poppy.capture.redaction import MASK, load_custom_redaction, redact_secrets
from poppy.config import PoppyConfig, load_config, save_config
from poppy.engine.seed import SeedEngine
from poppy.models import Filters

# --- per-pattern family ----------------------------------------------------


def test_masks_aws_access_key_id() -> None:
    out = redact_secrets("deploy key AKIAIOSFODNN7EXAMPLE is in the CI env")
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert MASK in out
    assert out.startswith("deploy key ") and out.endswith(" is in the CI env")


def test_masks_openai_secret_key() -> None:
    out = redact_secrets("use sk-abcDEF0123456789abcDEF0123 for the call")
    assert "sk-abcDEF0123456789abcDEF0123" not in out
    assert MASK in out
    # sk-proj- prefixed keys are masked too.
    assert "sk-proj-" not in redact_secrets("sk-proj-ABCdef0123456789ABCdef0123")


def test_masks_github_tokens() -> None:
    for tok in (
        "ghp_" + "A1b2C3d4E5f6G7h8I9j0" + "K1L2M3",
        "gho_" + "A1b2C3d4E5f6G7h8I9j0" + "K1L2M3",
        "ghs_" + "A1b2C3d4E5f6G7h8I9j0" + "K1L2M3",
    ):
        out = redact_secrets(f"token {tok} committed by mistake")
        assert tok not in out
        assert MASK in out


def test_masks_pem_private_key_block() -> None:
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA1234567890abcdEFGHijkl\n"
        "moreBase64LinesHere+/=\n"
        "-----END RSA PRIVATE KEY-----"
    )
    out = redact_secrets(f"lesson: the leaked key was\n{pem}\nrotate it")
    assert "BEGIN RSA PRIVATE KEY" not in out
    assert "MIIEpAIBAAKCAQEA" not in out
    assert MASK in out
    assert out.startswith("lesson: the leaked key was") and out.endswith("rotate it")


def test_masks_connection_string_password_only() -> None:
    out = redact_secrets("staging DB is postgres://admin:hunter2@db.internal:5432/app")
    assert "hunter2" not in out
    # Structure preserved: scheme, user, host all survive — only the password goes.
    assert "postgres://admin:" in out
    assert "@db.internal:5432/app" in out
    assert MASK in out


def test_masks_bearer_token_keeps_prefix() -> None:
    out = redact_secrets("call with header Authorization: Bearer abcDEF012345_678-ghiJKL")
    assert "abcDEF012345_678-ghiJKL" not in out
    assert "Bearer " in out  # the prefix stays; only the token is masked
    assert MASK in out


# --- mask-not-drop + no-false-positive ------------------------------------


def test_masks_multiple_secrets_but_keeps_the_memory() -> None:
    text = "prefer AKIAIOSFODNN7EXAMPLE and sk-abcDEF0123456789abcDEF0123 over hardcoding"
    out = redact_secrets(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "sk-abcDEF0123456789abcDEF0123" not in out
    # The lesson itself — the part worth remembering — is preserved.
    assert out.startswith("prefer ") and out.endswith(" over hardcoding")
    assert out.count(MASK) == 2


@pytest.mark.parametrize(
    "text",
    [
        "The team uses ruff for linting and formatting.",
        "Refactored auth.py to use asyncpg; connection pool lives in db.py.",
        "See https://example.com/docs and the task-runner in scripts/build.sh.",
        "The sprint goal is to ship the skeleton UI by Friday.",
        "def make_token(): return uuid4().hex  # placeholder id, not a secret",
        "Prefer Bearer auth over basic, but rotate short-lived tokens.",  # word 'Bearer' w/o a token
    ],
)
def test_ordinary_prose_and_code_is_untouched(text: str) -> None:
    """No false positives on normal memory content: returned unchanged (same object)."""
    assert redact_secrets(text) is text


def test_empty_input_is_returned_unchanged() -> None:
    assert redact_secrets("") == ""


# --- custom literal and environment-value layers --------------------------


def test_custom_literal_is_masked_case_sensitively() -> None:
    config = PoppyConfig(redaction_literals=["ExactSecret"])
    patterns, issues = load_custom_redaction(config)

    assert issues == []
    assert redact_secrets("ExactSecret exactsecret", patterns) == f"{MASK} exactsecret"


def test_current_environment_variable_value_is_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_TEST_CAPTURE_TOKEN", "rotated-secret-value")
    patterns, issues = load_custom_redaction(PoppyConfig(redaction_env_vars=["POPPY_TEST_CAPTURE_TOKEN"]))

    assert issues == []
    assert redact_secrets("token=rotated-secret-value", patterns) == f"token={MASK}"


def test_unset_and_short_environment_values_are_silently_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPPY_TEST_UNSET_TOKEN", raising=False)
    monkeypatch.setenv("POPPY_TEST_SHORT_TOKEN", "abc")
    config = PoppyConfig(redaction_env_vars=["POPPY_TEST_UNSET_TOKEN", "POPPY_TEST_SHORT_TOKEN"])

    patterns, issues = load_custom_redaction(config)
    assert patterns == ()
    assert issues == []
    assert redact_secrets("abc remains", patterns) == "abc remains"


def test_custom_extras_run_after_built_in_patterns() -> None:
    config = PoppyConfig(redaction_literals=["deployment-secret"])
    patterns, _issues = load_custom_redaction(config)
    output = redact_secrets("AKIAIOSFODNN7EXAMPLE deployment-secret", patterns)

    assert output == f"{MASK} {MASK}"


def test_empty_custom_config_preserves_built_in_behavior() -> None:
    text = "deploy AKIAIOSFODNN7EXAMPLE after review"
    patterns, issues = load_custom_redaction(PoppyConfig())
    assert patterns == ()
    assert issues == []
    assert redact_secrets(text, patterns) == redact_secrets(text)


def test_malformed_redaction_config_is_skipped_without_error(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"redaction": {"literals": "wrong", "env_vars": []}}')
    config = load_config(tmp_path)
    patterns, issues = load_custom_redaction(config)
    assert patterns == ()
    assert issues == []


# --- construction seam: candidates, store, and journal all masked ----------

# A raw item as the extraction LLM would emit it, carrying a live-looking secret.
_SECRET_ITEM = {"content": "staging DB is postgres://admin:hunter2@db.internal/app", "type": "lesson"}


def test_build_memories_masks_candidate_content() -> None:
    """Redaction happens at construction, so a candidate Memory never holds the raw
    secret — this is what every downstream consumer (store, sync, journal) reads."""
    [mem] = _build_memories([_SECRET_ITEM], source_type="claude-code", session_id="sess", project="poppy")
    assert "hunter2" not in mem.content
    assert MASK in mem.content


def test_build_memories_loads_custom_config_from_poppy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    save_config(PoppyConfig(poppy_dir=tmp_path, redaction_literals=["bespoke-secret"]))

    [mem] = _build_memories(
        [{"content": "contains bespoke-secret", "type": "lesson"}],
        source_type="claude-code",
        session_id="sess",
        project="poppy",
    )
    assert mem.content == f"contains {MASK}"


def test_build_memories_uses_threaded_config_without_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The orchestrator passes its already-loaded config, so no directory
    resolution happens at all — a break in that wire would fall back to the
    (empty) default store here and leave the literal unmasked."""
    monkeypatch.delenv("POPPY_DIR", raising=False)
    config = PoppyConfig(poppy_dir=tmp_path, redaction_literals=["bespoke-secret"])

    [mem] = _build_memories(
        [{"content": "contains bespoke-secret", "type": "lesson"}],
        source_type="claude-code",
        session_id="sess",
        project="poppy",
        config=config,
    )
    assert mem.content == f"contains {MASK}"


def test_build_memories_redaction_config_failure_falls_back_to_built_ins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("poppy.config.load_config", lambda _dir: (_ for _ in ()).throw(OSError("unreadable")))

    [mem] = _build_memories(
        [{"content": "AKIAIOSFODNN7EXAMPLE", "type": "lesson"}],
        source_type="claude-code",
        session_id="sess",
        project="poppy",
    )
    assert mem.content == MASK


def test_reconciler_stores_the_masked_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The store only ever receives masked candidates (built via _build_memories)."""
    monkeypatch.setattr("poppy.consolidation.call_llm", lambda *a, **k: pytest.fail("no LLM expected"))
    engine = SeedEngine(db_path=tmp_path / "memories.db")

    candidates = _build_memories([_SECRET_ITEM], source_type="claude-code", session_id="sess", project="poppy")
    summary = reconcile_and_ingest(candidates, engine=engine, cfg=PoppyConfig(), poppy_dir=tmp_path)
    assert summary.added == 1

    stored = engine.list_all(filters=Filters(project="poppy"), limit=10)
    assert len(stored) == 1
    assert "hunter2" not in stored[0].content
    assert MASK in stored[0].content


def test_capture_journal_preview_is_masked(tmp_path: Path) -> None:
    """Regression: the journal records an 80-char preview of each candidate in
    plaintext (not SQLCipher-encrypted) BEFORE the reconciler runs. Because the
    candidate is built already-masked, that preview carries no raw secret."""
    candidates = _build_memories([_SECRET_ITEM], source_type="claude-code", session_id="sess", project="poppy")
    journal.record(tmp_path, session_id="sess", project="poppy", count=len(candidates), items=candidates)

    raw_journal = (tmp_path / journal.JOURNAL_FILENAME).read_text()
    assert "hunter2" not in raw_journal  # the leak this fix closes
    preview = journal.read_all(tmp_path)[0]["items"][0]["preview"]
    assert "hunter2" not in preview
    assert MASK in preview
