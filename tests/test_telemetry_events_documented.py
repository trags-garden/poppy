"""Keep telemetry emit sites, the canonical event set, and README in sync."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from poppy.telemetry import DOCUMENTED_EVENTS

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src" / "poppy"
README_PATH = REPO_ROOT / "README.md"
EVENT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CONTRACT_MESSAGE = "Update DOCUMENTED_EVENTS in src/poppy/telemetry.py and the README telemetry table together."


class _TelemetryEventVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.events: set[str] = set()
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        callee_name = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            callee_name = node.func.attr

        if callee_name not in {"capture", "capture_once"}:
            self.generic_visit(node)
            return

        candidates = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and EVENT_NAME_RE.fullmatch(arg.value)
        ]

        if not candidates and self._is_telemetry_wrapper_forward(node):
            self.generic_visit(node)
            return

        location = f"{self.path.relative_to(REPO_ROOT)}:{node.lineno}"
        assert candidates, (
            f"Dynamic telemetry event name at {location}; use a string-literal event name.\n{CONTRACT_MESSAGE}"
        )
        assert len(candidates) == 1, (
            f"Ambiguous telemetry event arguments at {location}: {candidates!r}.\n{CONTRACT_MESSAGE}"
        )
        self.events.add(candidates[0])
        self.generic_visit(node)

    def _is_telemetry_wrapper_forward(self, node: ast.Call) -> bool:
        """Ignore the two wrapper-internal forwards of the caller's event value."""
        return (
            self.path == SOURCE_ROOT / "telemetry.py"
            and bool(self._function_stack)
            and self._function_stack[-1] in {"capture", "capture_once"}
            and any(isinstance(arg, ast.Name) and arg.id == "event" for arg in node.args)
        )


def _emitted_event_names() -> set[str]:
    events: set[str] = set()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(SOURCE_ROOT).parts:
            continue
        visitor = _TelemetryEventVisitor(path)
        visitor.visit(ast.parse(path.read_text(), filename=str(path)))
        events.update(visitor.events)
    return events


def _readme_event_names(readme: str) -> set[str]:
    telemetry_heading = re.search(r"^## Telemetry\s*$", readme, re.MULTILINE)
    assert telemetry_heading, f"README telemetry section not found.\n{CONTRACT_MESSAGE}"

    section = readme[telemetry_heading.end() :]
    next_heading = re.search(r"^#{1,2}\s+", section, re.MULTILINE)
    if next_heading:
        section = section[: next_heading.start()]

    lines = section.splitlines()
    table_start = next(
        (
            index + 2
            for index in range(len(lines) - 1)
            if re.fullmatch(r"\|\s*Event\s*\|\s*When\s*\|\s*Properties\s*\|", lines[index])
            and re.fullmatch(
                r"\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|\s*:?-{3,}:?\s*\|",
                lines[index + 1],
            )
        ),
        None,
    )
    assert table_start is not None, f"README telemetry event table not found.\n{CONTRACT_MESSAGE}"

    events: set[str] = set()
    for line in lines[table_start:]:
        if not line.startswith("|"):
            break
        first_cell = line.strip().strip("|").split("|", 1)[0].strip()
        event_match = re.fullmatch(r"`([a-z][a-z0-9_]*)`", first_cell)
        if event_match:
            events.add(event_match.group(1))
    return events


def _difference(actual: set[str] | frozenset[str], expected: frozenset[str]) -> str:
    return f"unexpected={sorted(actual - expected)!r}, missing={sorted(expected - actual)!r}"


def test_emitted_events_match_documented_events() -> None:
    emitted_events = _emitted_event_names()

    assert emitted_events == DOCUMENTED_EVENTS, (
        f"Telemetry emit sites do not match the canonical event set: "
        f"{_difference(emitted_events, DOCUMENTED_EVENTS)}.\n{CONTRACT_MESSAGE}"
    )


def test_readme_events_match_documented_events() -> None:
    readme_events = _readme_event_names(README_PATH.read_text())

    assert readme_events == DOCUMENTED_EVENTS, (
        f"README telemetry events do not match the canonical event set: "
        f"{_difference(readme_events, DOCUMENTED_EVENTS)}.\n{CONTRACT_MESSAGE}"
    )
