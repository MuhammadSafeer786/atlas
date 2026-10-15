"""Provider-agnostic LLM interface.

CONCEPT: the ports-and-adapters (hexagonal) pattern. The agent loop, tools
and orchestrator depend only on the small `LLM` protocol defined here. Each
provider - Claude, OpenAI-compatible servers, Gemini - is an *adapter* that
translates between Atlas's canonical message format and its own wire format.

Canonical format
----------------
Atlas stores conversations as Anthropic-style content blocks because that is
the richest of the three formats (it can express thinking, tool use, tool
results, citations and search results without loss):

    {"role": "user" | "assistant" | "system", "content": str | [blocks]}
    blocks: text · thinking · tool_use{id,name,input} · tool_result{tool_use_id,content,is_error}
            search_result · server_tool_use · ...

Adapters may attach private, provider-specific data to a block under keys that
start with "_" (e.g. Gemini's thought signatures). Other adapters drop them.

Capabilities
------------
Providers differ. Rather than scattering `if provider == ...` through the
code, each adapter declares `Capabilities`, and the runtime picks tools and
features accordingly (e.g. hosted web search on Claude, a client-side
`fetch_url` tool elsewhere).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from atlas.llm.usage import Usage

T = TypeVar("T", bound=BaseModel)
DeltaCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class Capabilities:
    provider: str
    server_tools: bool = False  # Anthropic-hosted web_search / web_fetch / code_execution
    native_memory_tool: bool = False  # Anthropic-defined `memory_20250818` schema
    native_citations: bool = False  # `search_result` blocks produce structured citations


class RefusalError(RuntimeError):
    """The model (and every fallback) declined the request."""

    def __init__(self, details: dict[str, Any] | None):
        super().__init__(f"Model declined the request: {details}")
        self.details = details


@dataclass
class TurnRequest:
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    effort: str = "medium"
    max_tokens: int = 64_000


@dataclass
class TurnResult:
    content: list[dict[str, Any]]  # assistant content blocks (canonical format)
    stop_reason: str | None  # end_turn | tool_use | max_tokens | refusal | pause_turn
    usage: Usage
    model: str  # the model that actually served the turn
    stop_details: dict[str, Any] | None = None


class LLM(Protocol):
    capabilities: Capabilities

    async def turn(self, req: TurnRequest, on_delta: DeltaCallback) -> TurnResult: ...

    async def parse(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        output_format: type[T],
        effort: str = "medium",
    ) -> tuple[T, Usage]: ...


# --------------------------------------------------------------------------- helpers
def block_text(content: str | list[dict[str, Any]]) -> str:
    """Flatten canonical content (incl. search_result blocks) to plain text."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block["text"])
        elif kind == "search_result":
            body = "\n".join(c.get("text", "") for c in block.get("content", []))
            parts.append(f"[source: {block.get('source')}] {block.get('title')}\n{body}")
        elif kind == "tool_result":
            parts.append(block_text(block.get("content", "")))
    return "\n\n".join(p for p in parts if p)


def tool_names_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    """tool_use id -> tool name (Gemini's function responses need the name)."""
    names: dict[str, str] = {}
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
    return names


def portable_tool_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only plain function tools (name/description/input_schema)."""
    return [
        {"name": t["name"], "description": t.get("description", ""),
         "input_schema": inline_refs(t.get("input_schema") or {"type": "object", "properties": {}})}
        for t in tools if "type" not in t  # typed specs are provider-specific (server tools etc.)
    ]


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local `$ref`s into inline schemas. Not every provider accepts $defs."""
    defs = schema.get("$defs", {}) or schema.get("definitions", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
                resolved = walk(defs.get(ref.split("/")[-1], {}))
                extra = {k: walk(v) for k, v in node.items() if k != "$ref"}
                return {**resolved, **extra}
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull a JSON object out of a reply that may include fences or chatter."""
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


JsonCaller = Callable[[str, list[dict[str, Any]], dict[str, Any]], Awaitable[tuple[str, Usage]]]


async def parse_with_repair(call: JsonCaller, *, system: str, messages: list[dict[str, Any]],
                            output_format: type[T], max_repairs: int = 2) -> tuple[T, Usage]:
    """Structured output for providers without guaranteed schema adherence.

    CONCEPT: validate-and-repair. Ask for JSON matching the schema (natively
    where the provider supports it), validate with Pydantic, and on failure
    feed the exact validation error back so the model can fix its own output.
    """
    schema = inline_refs(output_format.model_json_schema())
    system = (f"{system}\n\nRespond with a single JSON object that conforms to this JSON Schema. "
              f"Output JSON only, no prose.\n{json.dumps(schema)}")
    convo = list(messages)
    total = Usage()
    for attempt in range(max_repairs + 1):
        text, usage = await call(system, convo, schema)
        total = total + usage
        try:
            return output_format.model_validate_json(extract_json(text)), total
        except ValidationError as exc:
            if attempt == max_repairs:
                raise ValueError(f"Model did not return valid {output_format.__name__}: {exc}") from exc
            convo = [*convo, {"role": "assistant", "content": text or "(empty)"},
                     {"role": "user", "content": f"That JSON failed validation:\n{exc}\nReturn only corrected JSON."}]
    raise AssertionError("unreachable")
