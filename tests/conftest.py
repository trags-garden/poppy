"""Shared test fixtures.

Every test runs with POPPY_DIR pointed at a per-test temp directory and with
POPPY_TELEMETRY_OFF=1, so the suite can never read or write the real
``~/.poppy`` and never makes telemetry network calls. A few CLI tests used to
fall through to ``Path.home() / ".poppy"`` when they only set client-specific
env vars; this fixture closes that hole for good.

Tests that exercise telemetry-on behavior opt back in explicitly, either via
``monkeypatch.delenv("POPPY_TELEMETRY_OFF")`` or CliRunner's
``env={"POPPY_TELEMETRY_OFF": None}``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_poppy_env(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    poppy_dir = tmp_path_factory.mktemp("poppy-dir")
    monkeypatch.setenv("POPPY_DIR", str(poppy_dir))
    monkeypatch.setenv("POPPY_TELEMETRY_OFF", "1")
