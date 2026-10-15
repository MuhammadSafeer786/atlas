"""Tool abstraction.

CONCEPT: tools are how an LLM acts on the world. Claude never executes
anything itself - it emits a `tool_use` block (name + JSON input) and *your
harness* decides whether and how to run it. That makes the tool layer the
natural place for typing, validation, permissions and auditing.

Three kinds of tools appear in Atlas:

1. User-defined client tools (`Tool`): you write the schema and the code.
   We derive a *strict* JSON schema from a Pydantic model, so the input Claude
   produces always matches it, and we re-validate on our side anyway because
   eager input streaming moves validation to the client.
2. Anthropic-defined client tools (e.g. the memory tool): Anthropic defines
   the name and schema, Claude is trained on them, *you* run them.
3. Server tools (web search, web fetch, code execution): declared in the
   request, executed entirely on Anthropic's infrastructure.

Metadata on each tool drives the harness:
* `parallel_safe`      - read-only tools run concurrently with asyncio.gather.
* `requires_approval`  - side-effecting tools pause for human approval.
* `untrusted_output`   - output came from outside (web, documents, MCP) and is
                         screened for prompt injection before Claude sees it.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from atlas.config import Settings
from atlas.llm.base import Capabilities

InputT = TypeVar("InputT", bound=BaseModel)


@dataclass
class ToolContext:
    settings: Settings
    agent: str = "agent"
    run_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    """`content` is a string or a list of content blocks (text / search_result / image)."""

    content: str | list[dict[str, Any]]
    is_error: bool = False

    @classmethod
    def error(cls, message: str) -> ToolOutput:
        return cls(content=message, is_error=True)

    def preview(self, limit: int = 400) -> str:
        if isinstance(self.content, str):
            text = self.content
        else:
            parts = []
            for block in self.content:
                if block.get("type") == "text":
                    parts.append(block["text"])
                elif block.get("type") == "search_result":
                    parts.append(f"[{block.get('title')}] " + " ".join(
                        c.get("text", "") for c in block.get("content", [])
                    ))
            text = "\n".join(parts)
        return text if len(text) <= limit else text[:limit] + "…"


class BaseTool(ABC):
    name: ClassVar[str]
    parallel_safe: ClassVar[bool] = True
    requires_approval: ClassVar[bool] = False
    untrusted_output: ClassVar[bool] = False

    @abstractmethod
    def spec(self) -> dict[str, Any]:
        """The tool definition sent in the API request's `tools` array."""

    def spec_for(self, caps: Capabilities | None) -> dict[str, Any]:
        """Provider-specific variant of `spec()` (override when a provider needs another shape)."""
        return self.spec()

    @abstractmethod
    async def execute(self, raw_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        """Validate the raw model-produced input, then run."""


class Tool(BaseTool, Generic[InputT]):
    """A user-defined tool whose input is a Pydantic model."""

    description: ClassVar[str]
    Input: ClassVar[type[BaseModel]]

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_schema(self.Input),
            # strict: the API guarantees tool_use.input validates against the schema.
            "strict": True,
            # Stream large inputs as generated (we validate client-side below).
            "eager_input_streaming": True,
        }

    async def execute(self, raw_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            args = self.Input.model_validate(raw_input)
        except ValidationError as exc:
            # Never run a tool on unvalidated input. Tell Claude what was wrong so
            # it can correct itself on the next turn.
            return ToolOutput.error(f"Invalid input for {self.name}: {exc.errors(include_url=False)}")
        return await self.run(args, ctx)  # type: ignore[arg-type]

    @abstractmethod
    async def run(self, args: InputT, ctx: ToolContext) -> ToolOutput: ...


_FOLDED_CONSTRAINTS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                       "minLength", "maxLength", "multipleOf", "minItems", "maxItems")


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema -> the subset strict tool use accepts.

    Strict mode needs `additionalProperties: false` on every object. We also drop
    `title` noise and fold defaults and numeric/length bounds into the
    description so the schema stays within the supported keyword set. Pydantic
    still enforces every constraint when we validate the input.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def clean(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            notes = []
            if "default" in node:
                notes.append(f"default: {node.pop('default')!r}")
            for key in _FOLDED_CONSTRAINTS:
                if key in node and not isinstance(node[key], (dict, list)):
                    notes.append(f"{key}: {node.pop(key)}")
            if notes:
                desc = node.get("description", "")
                node["description"] = f"{desc} ({', '.join(notes)})".strip()
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                clean(value)
        elif isinstance(node, list):
            for item in node:
                clean(item)

    clean(schema)
    return schema


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a model-supplied path and refuse anything that escapes `root`.

    Model output is untrusted input: '../../etc/passwd' must not work.
    """
    root = root.resolve()
    candidate = (root / relative.lstrip("/\\")).resolve()
    if candidate != root and root not in candidate.parents:
        raise PermissionError(f"Path escapes sandbox: {relative}")
    return candidate
