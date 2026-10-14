"""Keep standalone plugin and browser type lists aligned with the Python model."""

import ast
import json
import re
from importlib import resources

from poppy.models import WRITABLE_MEMORY_TYPES


def test_standalone_memory_type_lists_match_writable_types():
    plugin_source = resources.files("poppy.setup").joinpath("hermes_plugin.py.txt").read_text(encoding="utf-8")
    plugin_tree = ast.parse(plugin_source)
    schema = next(
        ast.literal_eval(node.value)
        for node in plugin_tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "REMEMBER_SCHEMA" for target in node.targets)
    )
    assert tuple(schema["parameters"]["properties"]["memory_type"]["enum"]) == WRITABLE_MEMORY_TYPES

    remember = next(
        node for node in ast.walk(plugin_tree) if isinstance(node, ast.FunctionDef) and node.name == "_tool_remember"
    )
    guards = [
        ast.literal_eval(node.comparators[0])
        for node in ast.walk(remember)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "memory_type"
        and isinstance(node.ops[0], ast.NotIn)
    ]
    assert guards == [WRITABLE_MEMORY_TYPES]

    app_source = resources.files("poppy.ui").joinpath("static", "app.js").read_text(encoding="utf-8")
    patterns = (r"const TYPES = (\[[^\]]+\]);", r"const pinClass = (\[[^\]]+\])\.includes\(m\.memory_type\)")
    for pattern in patterns:
        (literal,) = re.findall(pattern, app_source)
        # The UI includes the context pin in its filters, forms, and feed styling.
        assert tuple(json.loads(literal)) == (*WRITABLE_MEMORY_TYPES, "context")
