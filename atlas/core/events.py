"""Typed event stream emitted by every agent.

CONCEPT: observability & streaming UX. An agent run is long and opaque; the
event stream turns it into something a CLI, a web UI (via Server-Sent Events)
or a test can watch in real time. Every event names the agent that produced
it, so a multi-agent run renders as parallel lanes.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    type: str  # see EventType below
    agent: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "agent": self.agent, "data": self.data, "ts": self.ts}


class EventType:
    RUN_START = "run_start"
    RUN_END = "run_end"
    THINKING = "thinking_delta"
    TEXT = "text_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SERVER_TOOL = "server_tool"  # web_search / web_fetch / code_execution (run by Anthropic)
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESULT = "approval_result"
    USAGE = "usage"
    GUARDRAIL = "guardrail"
    REFUSAL = "refusal"
    WARNING = "warning"
    ERROR = "error"
    # multi-agent lifecycle
    PLAN = "plan"
    WORKER_START = "worker_start"
    WORKER_END = "worker_end"
    CRITIQUE = "critique"


EventSink = Callable[[AgentEvent], Awaitable[None]]


async def null_sink(_: AgentEvent) -> None:  # default: drop events
    return None
