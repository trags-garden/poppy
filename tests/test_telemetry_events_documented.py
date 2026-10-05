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


# The README promises its memory_write row lists every path that sends it.
# memory_write is sent only from write_flow.remember, so the sending paths are
# the modules that call it, directly or through the MCP handlers that wrap it.
# Each maps to the README phrases naming it. Surfaces that store memories
# without it are named as exclusions, so a module that starts calling any of
# these functions fails here until the README says so. The behaviour of each
# documented path is pinned next to its own tests (test_mcp_server.py,
# test_ui_lifecycle.py, test_write_flow.py).
MEMORY_WRITE_CALLERS = {
    "cli/main.py": ["`poppy remember`"],
    "mcp_server/server.py": ["MCP `remember`", "`consolidate`"],
}
# capture/, integrations/ and ui/ store memories without calling any sending function.
MEMORY_WRITE_EXCLUDED = ("automatic capture", "imports", "edited or superseded in `poppy ui`")
SENDING_FUNCTIONS = frozenset({"remember", "_emit_memory_write", "handle_remember", "handle_consolidate"})


def _memory_write_name_files() -> set[str]:
    """Files that spell the event name anywhere: a call argument, keyword, or constant."""
    return {
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Constant) and node.value == "memory_write"
    }


def _sending_callers() -> set[str]:
    """Files (relative to src/poppy) that call a sending function.

    Matches on the called name, so module aliases (``wf.remember``), dotted
    paths (``poppy.write_flow.remember``) and ``from ... import remember as r``
    are all caught, and an import that is never called is not.
    """
    callers: set[str] = set()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        rel = path.relative_to(SOURCE_ROOT).as_posix()
        if rel == "write_flow.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        names = set(SENDING_FUNCTIONS)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.update(alias.asname for alias in node.names if alias.name in SENDING_FUNCTIONS and alias.asname)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if called in names:
                callers.add(rel)
    return callers


def _readme_memory_write_cells(readme: str) -> tuple[str, str]:
    """(When, Properties) cells of the README memory_write row."""
    row = next((line for line in readme.splitlines() if line.startswith("| `memory_write` |")), None)
    assert row is not None, "README telemetry table has no memory_write row."
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return cells[1], cells[2]


def _readme_memory_write_when(readme: str) -> str:
    return _readme_memory_write_cells(readme)[0]


def test_readme_memory_write_sources_cover_known_sources() -> None:
    """Known clients are covered by the "known client" wording; every other
    value telemetry can report must be spelled out in the row."""
    from poppy.models import MEMORY_TYPES
    from poppy.sources import _ALIASES, _SETUP_ONLY_SOURCES, IMPORT_SOURCE_TYPES, KNOWN_SOURCES

    properties = _readme_memory_write_cells(README_PATH.read_text())[1]
    assert "a known client" in properties
    non_client_sources = (KNOWN_SOURCES - set(_ALIASES) - _SETUP_ONLY_SOURCES) | {"other"}
    for value in sorted(non_client_sources | MEMORY_TYPES):
        assert f"`{value}`" in properties, f"memory_write row does not list {value!r}: {properties!r}"

    # The UI and importers never send the event themselves, so their names can
    # only reach it as a hand-set `--source`; the row must say so.
    configured_only = next((clause for clause in properties.split(";") if "`--source`" in clause), "")
    for value in sorted(IMPORT_SOURCE_TYPES | {"ui"}):
        assert f"`{value}`" in configured_only, (
            f"memory_write row must say {value!r} only appears through a configured `--source`: {properties!r}"
        )


def test_memory_write_is_sent_only_by_write_flow_remember() -> None:
    # The event name may appear only in its emit helper and the canonical set.
    assert _memory_write_name_files() == {"write_flow.py", "telemetry.py"}
    tree = ast.parse((SOURCE_ROOT / "write_flow.py").read_text())
    emitting_functions = {
        func.name
        for func in ast.walk(tree)
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_emit_memory_write"
            for node in ast.walk(func)
        )
    }
    assert emitting_functions == {"remember"}


def test_readme_memory_write_row_names_exactly_the_sending_paths() -> None:
    callers = _sending_callers()
    assert callers == set(MEMORY_WRITE_CALLERS), (
        f"memory_write callers changed: unexpected={sorted(callers - set(MEMORY_WRITE_CALLERS))!r}, "
        f"missing={sorted(set(MEMORY_WRITE_CALLERS) - callers)!r}. "
        "Update the README memory_write row and MEMORY_WRITE_CALLERS together."
    )

    when = _readme_memory_write_when(README_PATH.read_text())
    sends, _, excluded = when.partition(";")
    assert "do not send it" in excluded, f"memory_write row must name the surfaces that do not send it: {when!r}"
    for phrases in MEMORY_WRITE_CALLERS.values():
        for phrase in phrases:
            assert phrase in sends, f"memory_write row does not name {phrase!r} as a sending path: {when!r}"
    for phrase in MEMORY_WRITE_EXCLUDED:
        assert phrase in excluded, f"memory_write row does not exclude {phrase!r}: {when!r}"
        assert phrase not in sends, f"memory_write row names {phrase!r} as a sending path: {when!r}"
