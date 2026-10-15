"""Test doubles.

CONCEPT: agents must be testable without the network. Because every model
call goes through the small `LLM` interface, tests swap in a scripted fake
that returns predetermined turns - so the loop, tools, approvals and
orchestration logic are verified deterministically and for free.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from atlas.config import Settings
from atlas.llm.base import Capabilities, TurnRequest, TurnResult
from atlas.llm.usage import Usage

ROOT = Path(__file__).resolve().parents[1]


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def tool_use(id_: str, name: str, input_: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": id_, "name": name, "input": input_ or {}}


def turn(*blocks: dict[str, Any], stop: str | None = None, model: str = "claude-opus-5-5",
         details: dict[str, Any] | None = None) -> TurnResult:
    if stop is None:
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    return TurnResult(list(blocks), stop, Usage(1000, 200, 0, 800), model, details)


class FakeLLM:
    """Either a script (list of turns, consumed in order) or a handler(req) -> TurnResult."""

    def __init__(self, turns: list[Any] | None = None, parses: list[Any] | None = None,
                 handler: Callable[[TurnRequest], TurnResult] | None = None,
                 capabilities: Capabilities | None = None):
        # Defaults to a Claude-like provider; pass Capabilities("x") to simulate others.
        self.capabilities = capabilities or Capabilities(
            "fake", server_tools=True, native_memory_tool=True, native_citations=True)
        self.turns = list(turns or [])
        self.parses = list(parses or [])
        self.handler = handler
        self.requests: list[TurnRequest] = []
        self.parse_calls: list[dict[str, Any]] = []

    async def turn(self, req: TurnRequest, on_delta) -> TurnResult:
        self.requests.append(TurnRequest(req.model, req.system, copy.deepcopy(req.messages),
                                         req.tools, req.effort, req.max_tokens))
        if self.handler is not None:
            result = self.handler(req)
        else:
            item = self.turns.pop(0)
            result = item(req) if callable(item) else item
        for block in result.content:
            if block["type"] == "text":
                await on_delta("text", {"text": block["text"]})
        return result

    async def parse(self, *, model, system, messages, output_format, effort="medium"):
        self.parse_calls.append({"system": system, "messages": messages, "format": output_format})
        item = self.parses.pop(0)
        value = item(messages) if callable(item) else item
        return value, Usage(500, 100)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(provider="anthropic", data_dir=tmp_path / ".atlas", enable_compaction=False, max_turns=8,
                    max_cost_usd=5.0, tool_timeout_s=5.0, _env_file=None)


@pytest.fixture
def knowledge_dir() -> Path:
    return ROOT / "examples" / "knowledge"
