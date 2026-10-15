"""General-purpose client tools.

Each tool shows a different harness concern:
* `calculator`        - never `eval()` model output; parse an AST whitelist instead.
* `current_time`      - volatile data lives in a tool, NOT the system prompt,
                        so the cached prompt prefix stays byte-identical.
* `list_files` / `read_file` - read-only + parallel-safe, sandboxed to a workspace.
* `write_file`        - side-effecting: requires human approval, runs serially.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from atlas.tools.base import Tool, ToolContext, ToolOutput, safe_path

# --------------------------------------------------------------------------- calculator
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil", "fabs")
} | {"abs": abs, "round": round, "min": min, "max": max}
_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval(expression: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            right = ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](ev(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"Unsupported expression element: {ast.dump(node)[:60]}")

    return ev(ast.parse(expression, mode="eval"))


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(1.07 ** 10) * 2500' or 'sqrt(2)'.")


class Calculator(Tool[CalculatorInput]):
    name = "calculator"
    description = (
        "Evaluate an arithmetic expression exactly. Use this for any calculation instead of "
        "doing math in your head. Supports + - * / // % **, parentheses, pi, e, and "
        "sqrt/log/log10/exp/sin/cos/tan/floor/ceil/abs/round/min/max."
    )
    Input = CalculatorInput

    async def run(self, args: CalculatorInput, ctx: ToolContext) -> ToolOutput:
        try:
            return ToolOutput(content=repr(safe_eval(args.expression)))
        except (ValueError, ZeroDivisionError, OverflowError, SyntaxError) as exc:
            return ToolOutput.error(f"Could not evaluate: {exc}")


# --------------------------------------------------------------------------- time
class NoInput(BaseModel):
    pass


class CurrentTime(Tool[NoInput]):
    name = "current_time"
    description = "Get the current date and time (UTC, ISO-8601). Use when recency matters."
    Input = NoInput

    async def run(self, args: NoInput, ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=datetime.now(timezone.utc).isoformat(timespec="seconds"))


# --------------------------------------------------------------------------- workspace files
class ListFilesInput(BaseModel):
    directory: str = Field(default=".", description="Directory relative to the workspace root.")


class ListFiles(Tool[ListFilesInput]):
    name = "list_files"
    description = "List files in the agent workspace (a sandboxed directory)."
    Input = ListFilesInput

    async def run(self, args: ListFilesInput, ctx: ToolContext) -> ToolOutput:
        root = ctx.settings.workspace_dir
        root.mkdir(parents=True, exist_ok=True)
        try:
            target = safe_path(root, args.directory)
        except PermissionError as exc:
            return ToolOutput.error(str(exc))
        if not target.is_dir():
            return ToolOutput.error(f"Not a directory: {args.directory}")
        entries = sorted(
            f"{p.relative_to(root.resolve()).as_posix()}{'/' if p.is_dir() else ''} ({p.stat().st_size} B)"
            for p in target.iterdir()
        )
        return ToolOutput(content="\n".join(entries) or "(empty)")


class ReadFileInput(BaseModel):
    path: str = Field(description="File path relative to the workspace root.")


class ReadFile(Tool[ReadFileInput]):
    name = "read_file"
    description = "Read a UTF-8 text file from the agent workspace."
    Input = ReadFileInput
    untrusted_output = True  # file contents may contain injected instructions

    async def run(self, args: ReadFileInput, ctx: ToolContext) -> ToolOutput:
        try:
            path = safe_path(ctx.settings.workspace_dir, args.path)
        except PermissionError as exc:
            return ToolOutput.error(str(exc))
        if not path.is_file():
            return ToolOutput.error(f"File not found: {args.path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        limit = 50_000
        if len(text) > limit:
            text = text[:limit] + f"\n\n[truncated: file is {len(text)} chars; showing first {limit}]"
        return ToolOutput(content=text)


class WriteFileInput(BaseModel):
    path: str = Field(description="File path relative to the workspace root, e.g. 'reports/ai.md'.")
    content: str = Field(description="Full file contents to write.")


class WriteFile(Tool[WriteFileInput]):
    name = "write_file"
    description = (
        "Create or overwrite a text file in the agent workspace, e.g. to save a final report. "
        "A human must approve each write."
    )
    Input = WriteFileInput
    parallel_safe = False
    requires_approval = True

    async def run(self, args: WriteFileInput, ctx: ToolContext) -> ToolOutput:
        try:
            path = safe_path(ctx.settings.workspace_dir, args.path)
        except PermissionError as exc:
            return ToolOutput.error(str(exc))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return ToolOutput(content=f"Wrote {len(args.content)} chars to {args.path}")


def builtin_tools() -> list[Tool]:
    return [Calculator(), CurrentTime(), ListFiles(), ReadFile(), WriteFile()]
