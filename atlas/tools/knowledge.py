"""RAG exposed as a tool - "agentic RAG".

CONCEPT: classic RAG retrieves once, up front. Agentic RAG lets the model
decide *when* to search, *what* to search for, and whether to search again
with a refined query after reading the first results.

The tool returns `search_result` content blocks rather than plain text. That
switches on Claude's native citations: answers come back with
`search_result_location` citations pointing at the exact source passage -
grounding you can verify, not just trust.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from atlas.rag.store import KnowledgeBase
from atlas.tools.base import Tool, ToolContext, ToolOutput


class SearchKBInput(BaseModel):
    query: str = Field(description="Natural-language or keyword query. Rephrase and retry if results miss.")
    k: int = Field(default=5, ge=1, le=10, description="Number of passages to return.")
    mode: Literal["hybrid", "keyword", "vector"] = Field(
        default="hybrid", description="hybrid (default), keyword (exact terms/IDs), or vector (meaning)."
    )


class SearchKnowledgeBase(Tool[SearchKBInput]):
    name = "search_knowledge_base"
    description = (
        "Search the private knowledge base of ingested documents. Use it for any question about "
        "the user's own documents before relying on memory or the web. Cite what you use."
    )
    Input = SearchKBInput
    untrusted_output = True  # document text is third-party data

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    async def run(self, args: SearchKBInput, ctx: ToolContext) -> ToolOutput:
        hits = self.kb.search(args.query, k=args.k, mode=args.mode)
        if not hits:
            return ToolOutput(content="No matching passages. Try different keywords or mode='keyword'.")
        return ToolOutput(content=[
            {
                "type": "search_result",
                "source": hit.source,
                "title": f"{hit.title} - {hit.heading}" if hit.heading else hit.title,
                "content": [{"type": "text", "text": hit.text}],
                "citations": {"enabled": True},
            }
            for hit in hits
        ])
