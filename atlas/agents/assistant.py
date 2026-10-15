"""The conversational assistant + session persistence.

CONCEPTS
--------
* Agents as tools (hierarchical delegation). The assistant is a single agent
  for everyday questions, but it can hand a big question to the whole research
  team by calling `delegate_research`. The sub-team's work happens in its own
  contexts; only the final report flows back - which keeps the assistant's
  context window small.
* Short-term vs. long-term memory. The conversation transcript is short-term
  memory, persisted per session as JSON so a chat survives restarts. The memory
  tool (/memories) is long-term memory that spans all sessions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from atlas.agents import prompts
from atlas.core.agent import Agent, RunResult
from atlas.runtime import Runtime
from atlas.tools.base import Tool, ToolContext, ToolOutput


class DelegateInput(BaseModel):
    objective: str = Field(description="The full research question, with any constraints from the user.")


class DelegateResearch(Tool[DelegateInput]):
    name = "delegate_research"
    description = (
        "Hand a broad or multi-part research question to a team of specialist agents (planner, "
        "researchers, analyst, librarian, writer, critic). Slower and costlier than answering "
        "directly - use it only for questions that need several sources or real analysis. "
        "Returns a reviewed Markdown report."
    )
    Input = DelegateInput
    parallel_safe = False

    def __init__(self, rt: Runtime):
        self.rt = rt

    async def run(self, args: DelegateInput, ctx: ToolContext) -> ToolOutput:
        from atlas.agents.orchestrator import Orchestrator  # avoid import cycle

        outcome = await Orchestrator(self.rt).run(args.objective)
        score = outcome.critiques[-1].score if outcome.critiques else "n/a"
        return ToolOutput(content=f"[Team report - reviewer score {score}/10]\n\n{outcome.report}")


def build_assistant(rt: Runtime) -> Agent:
    tools = rt.tools_for("assistant")
    tools.add(DelegateResearch(rt))
    return rt.agent("assistant", "assistant", system_prompt=prompts.ASSISTANT, tools=tools,
                    effort=rt.settings.orchestrator_effort, guard_input=True)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    messages: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, directory: Path, session_id: str) -> Session:
        path = directory / f"{session_id}.json"
        if path.exists():
            return cls(session_id, json.loads(path.read_text(encoding="utf-8")))
        return cls(session_id)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{self.id}.json").write_text(json.dumps(self.messages, default=str), encoding="utf-8")

    async def send(self, agent: Agent, text: str) -> RunResult:
        result = await agent.run(text, history=self.messages)
        if result.stop != "blocked":
            self.messages = result.messages
        return result
