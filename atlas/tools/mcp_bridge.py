"""MCP client bridge: use tools from ANY MCP server inside an Atlas agent.

CONCEPT: tool interoperability. Atlas connects to MCP servers declared in
`atlas.mcp.json` (same shape as Claude Desktop's config), discovers their tools
at runtime with `list_tools`, and wraps each one as an Atlas tool. From the
agent loop's point of view an MCP tool is indistinguishable from a local one -
it gets the same approval, tracing and untrusted-output handling.

    {
      "servers": {
        "atlas":      {"command": "atlas", "args": ["mcp-server"]},
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "./docs"]},
        "remote":     {"url": "https://example.com/mcp"}
      }
    }

Server tools are namespaced `<server>__<tool>` to avoid collisions.
"""

from __future__ import annotations

import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from atlas.tools.base import BaseTool, ToolContext, ToolOutput

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


class MCPTool(BaseTool):
    untrusted_output = True  # a third-party server is outside our trust boundary

    def __init__(self, client: Client, server_name: str, tool: Any, requires_approval: bool = False):
        self.client = client
        self.remote_name = tool.name
        self.name = _NAME_RE.sub("_", f"{server_name}__{tool.name}")[:64]  # type: ignore[misc]
        self.description = (tool.description or f"MCP tool {tool.name}")[:1024]
        self.input_schema = tool.input_schema or {"type": "object", "properties": {}}
        annotations = getattr(tool, "annotations", None)
        read_only = bool(getattr(annotations, "read_only_hint", False)) if annotations else False
        self.parallel_safe = read_only  # type: ignore[misc]
        # Honour the server's own hints: destructive tools need a human.
        destructive = bool(getattr(annotations, "destructive_hint", False)) if annotations else False
        self.requires_approval = requires_approval or destructive  # type: ignore[misc]

    def spec(self) -> dict[str, Any]:
        # MCP schemas are arbitrary JSON Schema, so we don't mark them strict.
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    async def execute(self, raw_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        result = await self.client.call_tool(self.remote_name, raw_input)
        blocks: list[dict[str, Any]] = []
        for item in getattr(result, "content", None) or []:
            if getattr(item, "type", None) == "text":
                blocks.append({"type": "text", "text": item.text})
            else:
                blocks.append({"type": "text", "text": f"[{getattr(item, 'type', 'unknown')} content omitted]"})
        if not blocks and getattr(result, "structured_content", None) is not None:
            blocks.append({"type": "text", "text": json.dumps(result.structured_content)})
        return ToolOutput(content=blocks or "(no output)", is_error=bool(getattr(result, "is_error", False)))


class MCPToolset:
    """Async context manager owning live MCP connections for the lifetime of a run."""

    def __init__(self, servers: dict[str, Any]):
        """`servers` maps a name to an MCPServer instance (in-process), a URL, or a stdio config."""
        self.servers = servers
        self.tools: list[MCPTool] = []
        self._stack = AsyncExitStack()

    @classmethod
    def from_config(cls, path: Path) -> MCPToolset:
        if not path.exists():
            return cls({})
        config = json.loads(path.read_text(encoding="utf-8")).get("servers", {})
        return cls(config)

    @staticmethod
    def _target(spec: Any) -> Any:
        if isinstance(spec, dict):
            if "url" in spec:
                return spec["url"]
            return StdioServerParameters(command=spec["command"], args=spec.get("args", []),
                                         env=spec.get("env"))
        return spec  # MCPServer instance or URL string

    async def __aenter__(self) -> MCPToolset:
        for name, spec in self.servers.items():
            approval = bool(spec.get("require_approval")) if isinstance(spec, dict) else False
            client = await self._stack.enter_async_context(Client(self._target(spec)))
            listing = await client.list_tools()
            self.tools += [MCPTool(client, name, t, requires_approval=approval) for t in listing.tools]
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()
