"""Tools for the Messages API agent: JSON-schema definitions plus local handlers.

With the raw Messages API there are no built-in tools, so the read-only file
tools that the Agent SDK provides (Read / Glob / Grep) are implemented here,
confined to the project root.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from caludedemos.tools import current_time, evaluate

ROOT = Path.cwd().resolve()
SKIP_DIRS = {".git", ".venv", "__pycache__"}
MAX_CHARS = 50_000


class ToolError(Exception):
    """Raised by a handler; reported back to Claude as an is_error tool_result."""


def _safe_path(path: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ToolError(f"path is outside the project: {path}")
    return resolved


def _project_files(pattern: str) -> list[Path]:
    return sorted(
        p for p in ROOT.glob(pattern)
        if p.is_file() and not SKIP_DIRS & set(p.relative_to(ROOT).parts)
    )


def _calculate(expression: str) -> str:
    try:
        return evaluate(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        raise ToolError(str(e)) from e


def _read_file(path: str) -> str:
    file = _safe_path(path)
    if not file.is_file():
        raise ToolError(f"not a file: {path}")
    text = file.read_text(errors="replace")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n[truncated]"


def _glob(pattern: str) -> str:
    matches = [str(p.relative_to(ROOT)) for p in _project_files(pattern)]
    return "\n".join(matches) or "No files matched."


def _grep(pattern: str, glob: str = "**/*") -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"invalid regex: {e}") from e
    hits = []
    for file in _project_files(glob):
        try:
            lines = file.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        rel = file.relative_to(ROOT)
        hits += [f"{rel}:{n}: {line}" for n, line in enumerate(lines, 1) if regex.search(line)]
    return "\n".join(hits[:200]) or "No matches."


TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_current_time",
        "description": "Get the current local date and time (ISO 8601 with UTC offset).",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression using + - * / ( ) and numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '(3 + 4) * 12'"}
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "read_file",
        "description": "Read a text file in the project. Path is relative to the project root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "e.g. 'pyproject.toml'"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "glob",
        "description": "List project files matching a glob pattern, e.g. '**/*.py'.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "grep",
        "description": "Search project files for a regex; returns file:line: text matches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "glob": {"type": "string", "description": "Files to search (default '**/*')"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
]

HANDLERS: dict[str, Callable[..., str]] = {
    "get_current_time": lambda: current_time(),
    "calculate": _calculate,
    "read_file": _read_file,
    "glob": _glob,
    "grep": _grep,
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool; returns (result_text, is_error)."""
    handler = HANDLERS.get(name)
    if handler is None:
        return f"Error: unknown tool {name}", True
    try:
        return handler(**tool_input), False
    except ToolError as e:
        return f"Error: {e}", True
    except TypeError as e:  # wrong/missing arguments
        return f"Error: bad input for {name}: {e}", True
