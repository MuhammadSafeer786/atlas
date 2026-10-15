"""Atlas as an MCP server.

CONCEPT: the Model Context Protocol (MCP) is the open standard ("USB-C for
AI tools") for exposing tools, resources and prompts to any AI application.
Write a capability once as an MCP server and Claude Desktop, Claude Code,
IDEs and other agents - including Atlas itself, via tools/mcp_bridge.py - can
all use it.

This server publishes the Atlas knowledge base:
* tools     - search_knowledge_base, add_note, list_sources
* resources - atlas://kb/stats, atlas://kb/sources
* prompts   - research_brief (a reusable prompt template)

Run it:
    atlas mcp-server                       # stdio (for Claude Desktop / Claude Code)
    atlas mcp-server --transport http      # Streamable HTTP on :8765/mcp

Claude Code:  claude mcp add atlas -- atlas mcp-server
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer

from atlas.config import get_settings
from atlas.rag.store import KnowledgeBase


def build_server(kb: KnowledgeBase | None = None) -> MCPServer:
    kb = kb or KnowledgeBase(get_settings().kb_path)
    server = MCPServer(
        "atlas-knowledge",
        instructions="Search and extend the Atlas private knowledge base. Cite sources you use.",
        version="1.0.0",
    )

    @server.tool()
    def search_knowledge_base(query: str, k: int = 5) -> str:
        """Hybrid (BM25 + vector) search over ingested documents. Returns passages with sources."""
        hits = kb.search(query, k=max(1, min(k, 10)))
        if not hits:
            return "No matching passages."
        return "\n\n".join(
            f"[{i}] {h.title}{' > ' + h.heading if h.heading else ''}  (source: {h.source})\n{h.text}"
            for i, h in enumerate(hits, 1)
        )

    @server.tool()
    def add_note(title: str, content: str) -> str:
        """Save a note into the knowledge base so it becomes searchable."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        n = kb.add_text(content, source=f"note://{stamp}/{title}", title=title)
        return f"Saved note '{title}' as {n} chunk(s)."

    @server.tool()
    def list_sources() -> list[str]:
        """List every document source in the knowledge base."""
        return kb.sources()

    @server.resource("atlas://kb/stats")
    def kb_stats() -> str:
        """Knowledge base statistics."""
        return json.dumps(kb.stats())

    @server.resource("atlas://kb/sources")
    def kb_sources() -> str:
        """All document sources, one per line."""
        return "\n".join(kb.sources())

    @server.prompt()
    def research_brief(topic: str) -> str:
        """Template: a sourced research brief on a topic."""
        return (
            f"Write a concise research brief on: {topic}\n"
            "First search the knowledge base, then fill gaps from other sources. "
            "Structure: Summary, Key findings (with citations), Open questions."
        )

    return server


def main(transport: str = "stdio", port: int = 8765) -> None:
    server = build_server()
    if transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", port=port)


if __name__ == "__main__":
    main()
