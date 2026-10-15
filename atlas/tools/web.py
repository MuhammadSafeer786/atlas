"""Portable web tools (used when the provider has no hosted web tools).

On Claude, Atlas uses Anthropic's server-side `web_search` / `web_fetch`. Other
providers get these client-side equivalents:

* `fetch_url`  - always available. Downloads a page and returns readable text.
* `web_search` - available when TAVILY_API_KEY is set (Tavily is a search API
                 built for agents; swap in any provider behind the same tool).

CONCEPT: Server-Side Request Forgery (SSRF). A tool that fetches URLs chosen by
a model can be steered - by a prompt injection, say - into reading internal
services (http://169.254.169.254 cloud metadata, http://localhost:6379 ...).
`fetch_url` resolves the host first and refuses private, loopback, link-local
and reserved addresses, and re-checks after every redirect.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from atlas.rag.store import _strip_html
from atlas.tools.base import Tool, ToolContext, ToolOutput

MAX_BYTES = 2_000_000
MAX_CHARS = 40_000


async def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise PermissionError("Only http(s) URLs with a hostname are allowed")
    infos = await asyncio.to_thread(socket.getaddrinfo, parsed.hostname, parsed.port or 443)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise PermissionError(f"Refusing to fetch non-public address {ip} ({parsed.hostname})")


class FetchUrlInput(BaseModel):
    url: str = Field(description="Absolute http(s) URL to fetch.")


class FetchUrl(Tool[FetchUrlInput]):
    name = "fetch_url"
    description = "Fetch a web page or text file by URL and return its readable text content."
    Input = FetchUrlInput
    untrusted_output = True

    async def run(self, args: FetchUrlInput, ctx: ToolContext) -> ToolOutput:
        url = args.url
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=False,
                                         headers={"User-Agent": "AtlasAgent/1.0"}) as client:
                for _ in range(5):  # follow redirects manually so each hop is checked
                    await assert_public_url(url)
                    resp = await client.get(url)
                    if resp.is_redirect and "location" in resp.headers:
                        url = urljoin(url, resp.headers["location"])
                        continue
                    break
        except PermissionError as exc:
            return ToolOutput.error(str(exc))
        except httpx.HTTPError as exc:
            return ToolOutput.error(f"Fetch failed: {type(exc).__name__}: {exc}")
        if resp.status_code >= 400:
            return ToolOutput.error(f"HTTP {resp.status_code} for {url}")
        raw = resp.content[:MAX_BYTES].decode(resp.encoding or "utf-8", errors="replace")
        text = _strip_html(raw) if "html" in resp.headers.get("content-type", "") else raw
        text = " ".join(text.split())
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f" [truncated at {MAX_CHARS} chars]"
        return ToolOutput(content=f"URL: {url}\n\n{text}")


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query.")
    max_results: int = Field(default=5, ge=1, le=10)


class TavilySearch(Tool[WebSearchInput]):
    name = "web_search"
    description = ("Search the web for current information. Returns titles, URLs and snippets; "
                   "use fetch_url to read a page.")
    Input = WebSearchInput
    untrusted_output = True

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def run(self, args: WebSearchInput, ctx: ToolContext) -> ToolOutput:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"query": args.query, "max_results": args.max_results},
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolOutput.error(f"Search failed: {exc}")
        results = resp.json().get("results", [])
        if not results:
            return ToolOutput(content="No results.")
        return ToolOutput(content=[
            {"type": "search_result", "source": r.get("url", ""), "title": r.get("title", ""),
             "content": [{"type": "text", "text": r.get("content", "")}], "citations": {"enabled": True}}
            for r in results
        ])
