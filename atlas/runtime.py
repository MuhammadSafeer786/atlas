"""Runtime: the dependency container shared by every agent in a run.

CONCEPT: agents are cheap, infrastructure is shared. One run has one LLM
gateway, one cost tracker (so the budget covers sub-agents too), one tracer
(so sub-agent spans nest under the parent), one event sink and one approval
policy. Agents are assembled from a role -> (prompt, tools, effort) recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas.agents import prompts
from atlas.config import Settings
from atlas.core.agent import Agent, AgentConfig
from atlas.core.approvals import ApprovalPolicy, AutoDeny
from atlas.core.events import EventSink, null_sink
from atlas.core.tracing import Tracer
from atlas.llm.base import LLM, Capabilities
from atlas.llm.usage import CostTracker
from atlas.rag.store import KnowledgeBase
from atlas.tools import server as server_tools
from atlas.tools.base import BaseTool
from atlas.tools.builtin import Calculator, CurrentTime, builtin_tools
from atlas.tools.knowledge import SearchKnowledgeBase
from atlas.tools.memory import MemoryTool
from atlas.tools.registry import ToolRegistry
from atlas.tools.web import FetchUrl, TavilySearch

ROLE_PROMPTS = {
    "researcher": prompts.RESEARCHER,
    "analyst": prompts.ANALYST,
    "librarian": prompts.LIBRARIAN,
    "writer": prompts.WRITER,
}


@dataclass
class Runtime:
    settings: Settings
    llm: LLM
    kb: KnowledgeBase
    tracer: Tracer = field(default_factory=Tracer)
    costs: CostTracker = field(default_factory=CostTracker)
    sink: EventSink = null_sink
    approvals: ApprovalPolicy = field(default_factory=AutoDeny)
    extra_tools: list[BaseTool] = field(default_factory=list)  # e.g. MCP tools

    # -- tool recipes ---------------------------------------------------------
    @property
    def capabilities(self) -> Capabilities:
        return getattr(self.llm, "capabilities", Capabilities("unknown"))

    @property
    def has_web_search(self) -> bool:
        """Can agents actually *search* the web (not just fetch a known URL)?"""
        s = self.settings
        return s.enable_web_tools and (self.capabilities.server_tools or bool(s.tavily_api_key))

    NO_SEARCH_NOTE = (
        "NOTE: No web search tool is available in this run (only fetch_url for pages whose exact URL "
        "is already known). Do not guess URLs. Rely on the knowledge base, state plainly which facts "
        "could not be checked online, and never label anything as a web finding unless a tool returned it."
    )

    def _web_tools(self) -> tuple[list[BaseTool], list[dict]]:
        """Hosted web tools on Claude; portable client-side ones on other providers."""
        if not self.settings.enable_web_tools:
            return [], []
        if self.capabilities.server_tools:
            return [], [server_tools.WEB_SEARCH, server_tools.WEB_FETCH]
        client: list[BaseTool] = [FetchUrl()]
        if self.settings.tavily_api_key:
            client.append(TavilySearch(self.settings.tavily_api_key))
        return client, []

    def tools_for(self, role: str) -> ToolRegistry:
        s = self.settings
        kb_tool = SearchKnowledgeBase(self.kb)
        web_client, web_server = self._web_tools()
        if role == "researcher":
            return ToolRegistry([kb_tool, CurrentTime(), *web_client, *self.extra_tools], web_server)
        if role == "analyst":
            hosted = s.enable_code_execution and self.capabilities.server_tools
            return ToolRegistry([kb_tool, Calculator()], [server_tools.CODE_EXECUTION] if hosted else [])
        if role == "librarian":
            return ToolRegistry([kb_tool])
        if role == "writer":
            return ToolRegistry()
        if role == "assistant":
            return ToolRegistry(
                [*builtin_tools(), kb_tool, MemoryTool(s.memory_dir), *web_client, *self.extra_tools], web_server
            )
        raise ValueError(f"Unknown role: {role}")

    def agent(self, name: str, role: str, *, system_prompt: str | None = None,
              tools: ToolRegistry | None = None, effort: str | None = None,
              model: str | None = None, guard_input: bool = False) -> Agent:
        worker = role in ROLE_PROMPTS
        config = AgentConfig(
            name=name,
            system_prompt=system_prompt or ROLE_PROMPTS[role],
            model=model or (self.settings.worker_model if worker else self.settings.model),
            effort=effort or (self.settings.worker_effort if worker else self.settings.orchestrator_effort),
            max_turns=self.settings.max_turns,
            max_tokens=self.settings.max_tokens,
            guard_input=guard_input,
        )
        return Agent(config, self.llm, tools if tools is not None else self.tools_for(role), self.settings,
                     approvals=self.approvals, tracer=self.tracer, costs=self.costs, sink=self.sink)
