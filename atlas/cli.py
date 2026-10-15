"""Command-line interface.

    atlas ingest ./docs                 # build the knowledge base
    atlas search "rate limits"          # inspect retrieval directly
    atlas chat                          # interactive assistant (memory, tools, delegation)
    atlas research "objective..."       # full multi-agent team -> Markdown report
    atlas serve                         # web UI + API on http://localhost:8000
    atlas mcp-server                    # expose the knowledge base over MCP
    atlas eval                          # run the evaluation suite
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from atlas.config import get_settings
from atlas.core.approvals import policy_from_mode
from atlas.core.events import AgentEvent, EventType
from atlas.core.tracing import Tracer

for _stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    if hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(add_completion=False, help="Atlas - multi-agent research platform built on Claude.")
console = Console()


class ConsoleRenderer:
    """Turns the agent event stream into a readable live terminal transcript."""

    def __init__(self, show_thinking: bool = False, stream_agents: tuple[str, ...] = ("assistant",)):
        self.show_thinking = show_thinking
        self.stream_agents = stream_agents
        self._streaming: str | None = None
        self._thinking: str | None = None

    def _break(self) -> None:
        if self._streaming or self._thinking:
            console.print()
            self._streaming = self._thinking = None

    async def __call__(self, ev: AgentEvent) -> None:
        d, a = ev.data, ev.agent
        if ev.type == EventType.TEXT and a in self.stream_agents:
            if self._streaming != a:
                self._break()
                self._streaming = a
            console.print(d["text"], end="", markup=False, highlight=False)
        elif ev.type == EventType.THINKING and self.show_thinking:
            if self._thinking != a:
                self._break()
                self._thinking = a
                console.print(f"[dim]({a} thinking) [/]", end="")
            console.print(f"[dim italic]{d['text']}[/]", end="", markup=True, highlight=False)
        elif ev.type == EventType.TOOL_CALL:
            self._break()
            args = json.dumps(d["input"])
            console.print(f"[cyan]{a}[/] → [bold]{d['name']}[/] [dim]{args[:140]}[/]")
        elif ev.type == EventType.SERVER_TOOL:
            self._break()
            console.print(f"[cyan]{a}[/] ☁ [bold]{d['name']}[/] [dim]{json.dumps(d['input'])[:140]}[/]")
        elif ev.type == EventType.TOOL_RESULT and d.get("is_error"):
            console.print(f"[red]  ✗ {d['name']}: {d['preview'][:200]}[/]")
        elif ev.type == EventType.PLAN:
            self._break()
            table = Table(title="Research plan", show_lines=False)
            for col in ("id", "specialist", "title", "depends on"):
                table.add_column(col)
            for t in d["plan"]["subtasks"]:
                table.add_row(t["id"], t["specialist"], t["title"], ", ".join(t["depends_on"]))
            console.print(table)
        elif ev.type == EventType.WORKER_START:
            console.print(f"[green]▶ {d['worker']}[/] {d['title']}")
        elif ev.type == EventType.WORKER_END:
            console.print(f"[green]■ {d['worker']}[/] [dim]{d['status']}[/]")
        elif ev.type == EventType.CRITIQUE:
            self._break()
            colour = "green" if d["passes"] else "yellow"
            console.print(f"[{colour}]Critique round {d['round']}: {d['score']}/10[/] "
                          f"{'passes' if d['passes'] else '- revising'}")
        elif ev.type in (EventType.GUARDRAIL, EventType.WARNING):
            self._break()
            console.print(f"[yellow]⚠ {a}: {d.get('message') or d.get('findings')}[/]")
        elif ev.type in (EventType.REFUSAL, EventType.ERROR):
            self._break()
            console.print(f"[red]✗ {a}: {d}[/]")
        elif ev.type == EventType.RUN_END and a in self.stream_agents:
            self._break()


def _runtime(sink, approval_mode: str | None = None):
    from atlas.llm.factory import create_llm
    from atlas.rag.store import KnowledgeBase
    from atlas.runtime import Runtime

    s = get_settings()
    return Runtime(s, create_llm(s), KnowledgeBase(s.kb_path), tracer=Tracer(s.trace_dir),
                   sink=sink, approvals=policy_from_mode(approval_mode or s.approval_mode))


async def _with_mcp(fn):
    from atlas.tools.mcp_bridge import MCPToolset

    async with MCPToolset.from_config(Path("atlas.mcp.json")) as mcp:
        return await fn(mcp.tools)


def _footer(rt) -> None:
    c = rt.costs.summary()
    console.print(f"\n[dim]cost ${c['total_usd']:.4f} · {c['calls']} model calls · "
                  f"cache hit {c['cache_hit_ratio']:.0%} · trace {rt.tracer.trace_id}[/]")


@app.command()
def ingest(path: Path = typer.Argument(..., exists=True, help="File or directory to ingest.")) -> None:
    """Chunk, embed and index documents into the knowledge base."""
    from atlas.rag.store import KnowledgeBase

    kb = KnowledgeBase(get_settings().kb_path)
    added = kb.add_path(path)
    for source, n in added.items():
        console.print(f"[green]+[/] {source} [dim]({n} chunks)[/]")
    console.print(f"Knowledge base: {kb.stats()}")


@app.command()
def search(query: str,
           k: int = typer.Option(5, "-k", "--k", help="Number of passages."),
           mode: str = typer.Option("hybrid", help="hybrid | keyword | vector")) -> None:
    """Query the knowledge base directly (no LLM) - useful for debugging retrieval."""
    from atlas.rag.store import KnowledgeBase

    for hit in KnowledgeBase(get_settings().kb_path).search(query, k=k, mode=mode):
        console.rule(f"{hit.title} · {hit.heading or '-'} · rrf {hit.score}")
        console.print(hit.text[:600])


@app.command()
def chat(session: str | None = typer.Option(None, help="Resume a saved session id."),
         thinking: bool = typer.Option(False, "--thinking", help="Show summarized reasoning."),
         approval: str | None = typer.Option(None, help="ask | auto_approve | auto_deny")) -> None:
    """Interactive assistant with tools, memory and delegation to the research team."""
    from atlas.agents.assistant import Session, build_assistant

    async def main(mcp_tools) -> None:
        renderer = ConsoleRenderer(show_thinking=thinking)
        s = get_settings()
        sess = Session.load(s.session_dir, session) if session else Session()
        console.print(f"[bold]Atlas[/] · {s.provider} · {s.model} · session [cyan]{sess.id}[/] · type 'exit' to quit")
        if not _runtime(renderer, approval).has_web_search:
            console.print("[yellow]Web search unavailable on this provider - "
                          "set TAVILY_API_KEY (free) to enable it.[/]")
        console.print()
        while True:
            text = await asyncio.to_thread(console.input, "[bold orange3]you › [/]")
            if text.strip().lower() in {"exit", "quit", ":q"}:
                break
            if not text.strip():
                continue
            rt = _runtime(renderer, approval)
            rt.extra_tools = mcp_tools
            console.print("[bold]atlas ›[/] ", end="")
            await sess.send(build_assistant(rt), text)
            sess.save(s.session_dir)
            _footer(rt)

    asyncio.run(_with_mcp(main))


@app.command()
def research(objective: str,
             out: Path = typer.Option(Path("report.md"), help="Where to save the report."),
             thinking: bool = typer.Option(False, "--thinking")) -> None:
    """Run the full multi-agent pipeline: plan → parallel specialists → write → critique → revise."""
    from atlas.agents.orchestrator import Orchestrator

    async def main(mcp_tools) -> None:
        rt = _runtime(ConsoleRenderer(show_thinking=thinking, stream_agents=()), "auto_deny")
        rt.extra_tools = mcp_tools
        outcome = await Orchestrator(rt).run(objective)
        console.rule("Report")
        console.print(Markdown(outcome.report))
        await asyncio.to_thread(out.write_text, outcome.report, encoding="utf-8")
        console.print(f"\n[green]Saved to {out}[/]")
        console.print(f"[dim]{rt.tracer.tree()}[/]")
        _footer(rt)

    asyncio.run(_with_mcp(main))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the web UI and streaming API."""
    import uvicorn

    uvicorn.run("atlas.api.app:app_factory", factory=True, host=host, port=port)


@app.command("mcp-server")
def mcp_server(transport: str = typer.Option("stdio", help="stdio | http"), port: int = 8765) -> None:
    """Expose the knowledge base as an MCP server (for Claude Desktop, Claude Code, IDEs)."""
    from atlas.mcp_server.server import main

    main(transport, port)


@app.command("eval")
def run_eval(cases: Path = typer.Option(Path("evals/cases.json")),
             judge: bool = typer.Option(True, help="Use LLM-as-judge for rubric checks.")) -> None:
    """Run the evaluation suite and print a scorecard."""
    from atlas.evals.runner import main as eval_main

    raise typer.Exit(asyncio.run(eval_main(cases, use_judge=judge)))


def friendly_error(exc: BaseException) -> str | None:
    """One-line explanations for the failures new users actually hit."""
    import anthropic
    import openai
    from google.genai import errors as genai_errors
    from pydantic import ValidationError

    from atlas.config import ConfigurationError

    if isinstance(exc, ValidationError):
        return "Configuration: " + "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
    if isinstance(exc, ConfigurationError):
        return str(exc)
    try:
        s = get_settings()
        where = f"{s.provider} ({s.base_url or 'default endpoint'}, model {s.model})"
    except Exception:  # noqa: BLE001 - settings themselves may be the problem
        where = "the LLM provider"
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return f"Could not reach {where}. Is the server running and the URL correct?"
    if isinstance(exc, (anthropic.AuthenticationError, openai.AuthenticationError)):
        return f"Authentication failed for {where}. Check your API key."
    if isinstance(exc, (anthropic.NotFoundError, openai.NotFoundError)):
        return f"Not found at {where}. Check ATLAS_MODEL (and that the model is pulled/loaded)."
    if isinstance(exc, (anthropic.APIStatusError, openai.APIStatusError)):
        return f"{where} returned HTTP {exc.status_code}: {exc.message}"
    if isinstance(exc, genai_errors.APIError):
        if exc.code in (429, 503):
            return (f"Gemini is rate-limiting or overloaded ({exc.code}: {exc.message}). Retry shortly, or use a "
                    "less busy model, e.g. ATLAS_MODEL=gemini-flash-lite-latest.")
        return f"Gemini API error {exc.code}: {exc.message}"
    if type(exc).__name__ in ("ReadError", "RemoteProtocolError", "ConnectError", "ReadTimeout"):
        return (f"The connection to {where} dropped ({type(exc).__name__}). This is usually a temporary network "
                "or provider overload issue - try again, or switch to a less busy model.")
    return None


def main() -> None:
    try:
        app()
    except Exception as exc:  # noqa: BLE001
        message = friendly_error(exc)
        if message is None:
            raise
        console.print(f"[red]Error:[/] {message}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
