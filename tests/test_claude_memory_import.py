from pathlib import Path

from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.integrations.claude_memory_import import (
    build_content,
    import_claude_memories,
    map_memory_type,
    parse_memory_file,
    project_name_from_slug,
    stable_id,
)
from poppy.runtime import get_engine


def _write_memory(projects_dir: Path, slug: str, filename: str, body: str) -> Path:
    mem_dir = projects_dir / slug / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    path = mem_dir / filename
    path.write_text(body)
    return path


def test_parse_memory_file(tmp_path):
    path = _write_memory(
        tmp_path,
        "-Users-haris-code-personal-trags-poppy",
        "feedback_commit_push.md",
        "---\nname: Always commit and push\ndescription: Push immediately after commit\ntype: feedback\n---\n\nBody text here.\n",
    )
    parsed = parse_memory_file(path)
    assert parsed is not None
    assert parsed.name == "Always commit and push"
    assert parsed.description == "Push immediately after commit"
    assert parsed.type == "feedback"
    assert "Body text here" in parsed.body


def test_parse_memory_file_no_frontmatter(tmp_path):
    path = tmp_path / "raw.md"
    path.write_text("just a body, no frontmatter")
    assert parse_memory_file(path) is None


def test_project_name_from_slug():
    assert project_name_from_slug("-Users-haris-code-personal-trags-poppy") == "poppy"
    assert project_name_from_slug("-Users-haris--config") == "config"


def test_map_memory_type():
    assert map_memory_type("feedback") == "preference"
    assert map_memory_type("user") == "fact"
    assert map_memory_type("project") == "fact"
    assert map_memory_type("reference") == "fact"


def test_stable_id_is_deterministic():
    a = stable_id("slug", "name")
    b = stable_id("slug", "name")
    c = stable_id("slug", "other")
    assert a == b
    assert a != c
    assert a.startswith("mem_clmem_")


def test_build_content_includes_header_and_body(tmp_path):
    path = _write_memory(
        tmp_path,
        "slug",
        "f.md",
        "---\nname: N\ndescription: D\ntype: user\n---\n\nBody.\n",
    )
    parsed = parse_memory_file(path)
    out = build_content(parsed)
    assert "N — D" in out
    assert "Body." in out


def test_import_claude_memories_idempotent(tmp_path):
    projects = tmp_path / "projects"
    _write_memory(
        projects,
        "-Users-haris-code-personal-trags-poppy",
        "user_role.md",
        "---\nname: Role\ndescription: senior dev\ntype: user\n---\n\nUser is a senior dev.\n",
    )
    _write_memory(
        projects,
        "-Users-haris-code-personal-trags-poppy",
        "feedback_x.md",
        "---\nname: X\ndescription: do X\ntype: feedback\n---\n\nDo X.\n",
    )

    poppy_dir = tmp_path / "poppy"
    engine = get_engine(poppy_dir)

    r1 = import_claude_memories(engine, projects_dir=projects)
    assert r1.imported == 2
    assert r1.skipped == 0

    r2 = import_claude_memories(engine, projects_dir=projects)
    assert r2.imported == 0
    assert r2.skipped == 2


def test_import_claude_memories_skips_index(tmp_path):
    projects = tmp_path / "projects"
    mem_dir = projects / "slug" / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text("- [foo](foo.md) — index entry\n")
    _write_memory(
        projects,
        "slug",
        "foo.md",
        "---\nname: Foo\ndescription: f\ntype: user\n---\n\nbody\n",
    )

    poppy_dir = tmp_path / "poppy"
    engine = get_engine(poppy_dir)

    result = import_claude_memories(engine, projects_dir=projects)
    assert result.imported == 1


def test_import_claude_memories_dry_run(tmp_path):
    projects = tmp_path / "projects"
    _write_memory(
        projects,
        "slug",
        "a.md",
        "---\nname: A\ndescription: d\ntype: user\n---\n\nbody\n",
    )

    poppy_dir = tmp_path / "poppy"
    engine = get_engine(poppy_dir)

    result = import_claude_memories(engine, projects_dir=projects, dry_run=True)
    assert result.imported == 1
    assert engine.stats().memory_count == 0


def test_cli_import_claude_memories(tmp_path):
    projects = tmp_path / "projects"
    _write_memory(
        projects,
        "-Users-haris-code-personal-trags-poppy",
        "p.md",
        "---\nname: P\ndescription: d\ntype: project\n---\n\nbody\n",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["import", "claude-memories", "--projects-dir", str(projects)],
        env={"POPPY_DIR": str(tmp_path / "poppy")},
    )
    assert result.exit_code == 0, result.output
    assert "Imported: 1" in result.output


def test_cli_import_claude_memories_dry_run(tmp_path):
    projects = tmp_path / "projects"
    _write_memory(
        projects,
        "slug",
        "p.md",
        "---\nname: P\ndescription: d\ntype: project\n---\n\nbody\n",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["import", "claude-memories", "--projects-dir", str(projects), "--dry-run"],
        env={"POPPY_DIR": str(tmp_path / "poppy")},
    )
    assert result.exit_code == 0, result.output
    assert "Would import: 1" in result.output
