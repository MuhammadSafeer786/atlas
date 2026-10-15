"""Tool registry: one place that knows every tool an agent may use.

CONCEPT: prompt-cache-friendly tool lists. Tool definitions are the very
first thing in the request prefix, so their order must be deterministic -
re-ordering or adding a tool mid-session invalidates the whole cache. The
registry always renders tools sorted by name.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from atlas.llm.base import Capabilities
from atlas.tools.base import BaseTool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] = (), server_tools: Iterable[dict[str, Any]] = ()):
        self._tools: dict[str, BaseTool] = {}
        self._server: dict[str, dict[str, Any]] = {}
        for tool in tools:
            self.add(tool)
        for spec in server_tools:
            self.add_server_tool(spec)

    def add(self, tool: BaseTool) -> None:
        if tool.name in self._tools or tool.name in self._server:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def add_server_tool(self, spec: dict[str, Any]) -> None:
        self._server[spec["name"]] = spec

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted([*self._tools, *self._server])

    def specs(self, caps: Capabilities | None = None) -> list[dict[str, Any]]:
        client = [self._tools[n].spec_for(caps) for n in sorted(self._tools)]
        server = [self._server[n] for n in sorted(self._server)]
        return client + server

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        wanted = set(names)
        return ToolRegistry(
            [t for n, t in self._tools.items() if n in wanted],
            [s for n, s in self._server.items() if n in wanted],
        )

    def __len__(self) -> int:
        return len(self._tools) + len(self._server)
