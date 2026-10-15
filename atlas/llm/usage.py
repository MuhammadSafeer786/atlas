"""Token accounting and cost tracking.

CONCEPT: agents are loops, and loops spend money. Every response carries a
`usage` object; we normalise it, price it, and aggregate it per run so the
agent can enforce a hard dollar budget and the UI can show live spend.

Prompt caching shows up here: `cache_read_input_tokens` are billed at ~10% of
the normal input price, `cache_creation_input_tokens` at ~125%. Watching the
cache-read ratio is the quickest way to spot a broken cache prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# USD per 1M tokens: (input, output, cache_write_5m, cache_read).
# Only Claude prices are built in. For other providers set ATLAS_PRICE_INPUT_PER_MTOK /
# ATLAS_PRICE_OUTPUT_PER_MTOK (see register_pricing); unknown models are priced at $0.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-fable-5-1": (10.00, 50.00, 12.50, 0.25),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}
WEB_SEARCH_USD_PER_REQUEST = 10.00 / 1000


def register_pricing(model: str, input_per_mtok: float, output_per_mtok: float) -> None:
    """Price a model Atlas doesn't know, so cost budgets work on any provider."""
    PRICING[model] = (input_per_mtok, output_per_mtok, input_per_mtok, input_per_mtok)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    web_search_requests: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> Usage:
        """Build from an SDK usage object or a plain dict (tests)."""
        get = (lambda k: usage.get(k)) if isinstance(usage, dict) else (
            lambda k: getattr(usage, k, None)
        )
        server = get("server_tool_use")
        searches = 0
        if server is not None:
            searches = (
                server.get("web_search_requests", 0)
                if isinstance(server, dict)
                else getattr(server, "web_search_requests", 0)
            ) or 0
        return cls(
            input_tokens=get("input_tokens") or 0,
            output_tokens=get("output_tokens") or 0,
            cache_creation_input_tokens=get("cache_creation_input_tokens") or 0,
            cache_read_input_tokens=get("cache_read_input_tokens") or 0,
            web_search_requests=searches,
        )

    def cost_usd(self, model: str) -> float:
        inp, out, cw, cr = PRICING.get(model, (0.0, 0.0, 0.0, 0.0))
        return (
            self.input_tokens * inp
            + self.output_tokens * out
            + self.cache_creation_input_tokens * cw
            + self.cache_read_input_tokens * cr
        ) / 1_000_000 + self.web_search_requests * WEB_SEARCH_USD_PER_REQUEST

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            self.web_search_requests + other.web_search_requests,
        )


@dataclass
class CostTracker:
    """Aggregates spend across every model call in a run (including sub-agents)."""

    total_usd: float = 0.0
    calls: int = 0
    usage: Usage = field(default_factory=Usage)

    def record(self, model: str, usage: Usage) -> float:
        cost = usage.cost_usd(model)
        self.total_usd += cost
        self.calls += 1
        self.usage = self.usage + usage
        return cost

    @property
    def cache_hit_ratio(self) -> float:
        cached = self.usage.cache_read_input_tokens
        total = cached + self.usage.input_tokens + self.usage.cache_creation_input_tokens
        return cached / total if total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 4),
            "calls": self.calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_input_tokens,
            "cache_write_tokens": self.usage.cache_creation_input_tokens,
            "cache_hit_ratio": round(self.cache_hit_ratio, 3),
            "web_searches": self.usage.web_search_requests,
        }
