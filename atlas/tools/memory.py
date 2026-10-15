"""Long-term memory via Anthropic's memory tool (`memory_20250818`).

CONCEPT: the context window is working memory - it vanishes when the run
ends. Long-term memory needs storage outside the model. With the memory tool,
Claude is *trained* to check a `/memories` directory before starting work and
to record durable learnings (user preferences, project facts, what failed)
as it goes. Anthropic defines the tool's name and command schema; we implement
the storage backend - here, plain files, one directory per user/namespace.

Commands: view, create, str_replace, insert, delete, rename.
Security: every path is confined to the memory root, and secrets must never be
written to memory (see guardrails.redact_secrets).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from atlas.core.guardrails import redact_secrets
from atlas.llm.base import Capabilities
from atlas.tools.base import BaseTool, ToolContext, ToolOutput, safe_path

MEMORY_PREFIX = "/memories"

# For providers without Anthropic's built-in memory tool: the same command set
# as an ordinary function schema, so the file backend below works unchanged.
PORTABLE_SPEC: dict[str, Any] = {
    "name": "memory",
    "description": (
        "Long-term memory stored as files under /memories that persists across conversations. "
        "Check it (command=view, path=/memories) at the start of a task; record durable facts about "
        "the user and ongoing work. Never store secrets. Commands: view(path, view_range?), "
        "create(path, file_text), str_replace(path, old_str, new_str), insert(path, insert_line, "
        "insert_text), delete(path), rename(old_path, new_path)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert", "delete", "rename"]},
            "path": {"type": "string", "description": "Path starting with /memories"},
            "view_range": {"type": "array", "items": {"type": "integer"}, "description": "[start, end] lines"},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"},
            "insert_text": {"type": "string"},
            "old_path": {"type": "string"},
            "new_path": {"type": "string"},
        },
        "required": ["command"],
    },
}


class MemoryTool(BaseTool):
    name = "memory"
    parallel_safe = False  # writes must not race each other

    def __init__(self, root: Path):
        self.root = root

    def spec(self) -> dict[str, Any]:
        # Anthropic-defined tool: no input_schema - Claude already knows it.
        return {"type": "memory_20250818", "name": "memory"}

    def spec_for(self, caps: Capabilities | None) -> dict[str, Any]:
        if caps is None or caps.native_memory_tool:
            return self.spec()
        return PORTABLE_SPEC  # same commands, described explicitly for other providers

    def _resolve(self, path: str) -> Path:
        if not path.startswith(MEMORY_PREFIX):
            raise PermissionError(f"Memory paths must start with {MEMORY_PREFIX}")
        self.root.mkdir(parents=True, exist_ok=True)
        return safe_path(self.root, path[len(MEMORY_PREFIX):] or ".")

    async def execute(self, raw_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        command = raw_input.get("command")
        try:
            handler = getattr(self, f"_cmd_{command}", None)
            if handler is None:
                return ToolOutput.error(f"Unknown memory command: {command}")
            return ToolOutput(content=handler(raw_input))
        except (PermissionError, FileNotFoundError, FileExistsError, ValueError, KeyError) as exc:
            return ToolOutput.error(f"{type(exc).__name__}: {exc}")

    # -- commands -------------------------------------------------------------
    def _cmd_view(self, inp: dict[str, Any]) -> str:
        path = self._resolve(inp["path"])
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file())
            if not files:
                return f"Directory {inp['path']} is empty."
            root = self.root.resolve()
            lines = [f"{MEMORY_PREFIX}/{p.relative_to(root).as_posix()} ({p.stat().st_size} B)" for p in files]
            return "Files:\n" + "\n".join(lines)
        if not path.is_file():
            raise FileNotFoundError(inp["path"])
        lines = path.read_text(encoding="utf-8").splitlines()
        start, end = 1, len(lines)
        if inp.get("view_range"):
            start, end = inp["view_range"][0], inp["view_range"][1]
            end = len(lines) if end == -1 else end
        return "\n".join(f"{i:>4}\t{lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1))

    def _cmd_create(self, inp: dict[str, Any]) -> str:
        path = self._resolve(inp["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact_secrets(inp["file_text"]), encoding="utf-8")
        return f"Created {inp['path']}"

    def _cmd_str_replace(self, inp: dict[str, Any]) -> str:
        path = self._resolve(inp["path"])
        text = path.read_text(encoding="utf-8")
        count = text.count(inp["old_str"])
        if count != 1:
            raise ValueError(f"old_str must appear exactly once (found {count})")
        path.write_text(text.replace(inp["old_str"], redact_secrets(inp["new_str"])), encoding="utf-8")
        return f"Updated {inp['path']}"

    def _cmd_insert(self, inp: dict[str, Any]) -> str:
        path = self._resolve(inp["path"])
        lines = path.read_text(encoding="utf-8").splitlines()
        line = int(inp["insert_line"])
        if not 0 <= line <= len(lines):
            raise ValueError(f"insert_line {line} out of range 0..{len(lines)}")
        lines[line:line] = redact_secrets(inp["insert_text"]).splitlines()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Inserted into {inp['path']} at line {line}"

    def _cmd_delete(self, inp: dict[str, Any]) -> str:
        path = self._resolve(inp["path"])
        if path == self.root.resolve():
            raise PermissionError("Refusing to delete the memory root")
        if path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                child.rmdir() if child.is_dir() else child.unlink()
            path.rmdir()
        else:
            path.unlink()
        return f"Deleted {inp['path']}"

    def _cmd_rename(self, inp: dict[str, Any]) -> str:
        src, dst = self._resolve(inp["old_path"]), self._resolve(inp["new_path"])
        if dst.exists():
            raise FileExistsError(inp["new_path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        return f"Renamed {inp['old_path']} -> {inp['new_path']}"
