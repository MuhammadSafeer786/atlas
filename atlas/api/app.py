"""HTTP API: agents as a streaming web service.

CONCEPTS
--------
* Server-Sent Events (SSE). An agent run takes seconds to minutes. Instead of
  a blocking request, each run streams its event feed (thinking, tool calls,
  text deltas, costs) to the browser as it happens.
* Async human-in-the-loop. When an agent needs approval it emits an
  `approval_request` event and *awaits*; the UI answers through
  `POST /api/approvals/{id}`, which resolves the waiting future.
* Cancellation. If the client disconnects, the run is cancelled so we stop
  paying for tokens nobody will read.
* App factory + lifespan. `create_app()` takes its dependencies as arguments,
  so tests inject a fake LLM; the lifespan opens and closes MCP connections.

Endpoints
    GET  /                      web UI
    GET  /health
    POST /api/chat              {message, session_id?}   -> SSE
    POST /api/research          {objective}              -> SSE
    GET  /api/approvals         pending approvals
    POST /api/approvals/{id}    {approved, reason?}
    GET  /api/kb/stats | POST /api/kb/ingest | POST /api/kb/search
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from atlas import __version__
from atlas.agents.assistant import Session, build_assistant
from atlas.agents.orchestrator import Orchestrator
from atlas.config import Settings, get_settings
from atlas.core.approvals import ApprovalBroker
from atlas.core.events import AgentEvent
from atlas.core.tracing import Tracer
from atlas.llm.base import LLM
from atlas.llm.factory import create_llm
from atlas.rag.store import KnowledgeBase
from atlas.runtime import Runtime
from atlas.tools.mcp_bridge import MCPToolset

STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=100_000)
    session_id: str | None = None


class ResearchRequest(BaseModel):
    objective: str = Field(min_length=5, max_length=20_000)


class ApprovalBody(BaseModel):
    approved: bool
    reason: str = ""


class IngestBody(BaseModel):
    text: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=500)
    title: str | None = None


class SearchBody(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=20)
    mode: str = "hybrid"


def create_app(settings: Settings | None = None, llm: LLM | None = None,
               kb: KnowledgeBase | None = None, mcp_config: Path | None = None) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state["llm"] = llm or create_llm(settings)
        state["kb"] = kb or KnowledgeBase(settings.kb_path)
        state["broker"] = ApprovalBroker()
        async with MCPToolset.from_config(mcp_config or Path("atlas.mcp.json")) as mcp:
            state["mcp_tools"] = mcp.tools
            yield

    app = FastAPI(title="Atlas", version=__version__, lifespan=lifespan)

    def runtime(sink: Callable[[AgentEvent], Awaitable[None]]) -> Runtime:
        return Runtime(settings, state["llm"], state["kb"], tracer=Tracer(settings.trace_dir),
                       sink=sink, approvals=state["broker"], extra_tools=state["mcp_tools"])

    def stream(job: Callable[[Runtime], Awaitable[dict[str, Any]]], request: Request) -> StreamingResponse:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def sink(event: AgentEvent) -> None:
            await queue.put(event.to_dict())

        async def worker() -> None:
            rt = runtime(sink)
            try:
                result = await job(rt)
                await queue.put({"type": "done", "agent": "system",
                                 "data": result | {"trace_id": rt.tracer.trace_id}})
            except Exception as exc:  # noqa: BLE001 - report to client, keep server alive
                await queue.put({"type": "error", "agent": "system",
                                 "data": {"message": f"{type(exc).__name__}: {exc}"}})
            finally:
                await queue.put(None)

        async def events() -> AsyncIterator[str]:
            task = asyncio.create_task(worker())
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"  # stops proxies closing idle streams
                        continue
                    if item is None:
                        break
                    yield f"event: {item['type']}\ndata: {json.dumps(item, default=str)}\n\n"
                    if await request.is_disconnected():
                        break
            finally:
                if not task.done():
                    task.cancel()  # client went away: stop spending tokens

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- routes
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "provider": settings.provider, "model": settings.model,
                "mcp_tools": [t.name for t in state.get("mcp_tools", [])]}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        session = Session.load(settings.session_dir, body.session_id) if body.session_id else Session()

        async def job(rt: Runtime) -> dict[str, Any]:
            result = await session.send(build_assistant(rt), body.message)
            session.save(settings.session_dir)
            return {"session_id": session.id, "text": result.text, "stop": result.stop,
                    "citations": result.citations, "cost": rt.costs.summary()}

        return stream(job, request)

    @app.post("/api/research")
    async def research(body: ResearchRequest, request: Request) -> StreamingResponse:
        async def job(rt: Runtime) -> dict[str, Any]:
            outcome = await Orchestrator(rt).run(body.objective)
            return {"report": outcome.report, "plan": outcome.plan.model_dump(),
                    "critiques": [c.model_dump() for c in outcome.critiques], "cost": outcome.cost}

        return stream(job, request)

    @app.get("/api/approvals")
    async def approvals() -> list[dict[str, Any]]:
        return [{"id": r.id, "tool": r.tool, "agent": r.agent, "input": r.input}
                for r in state["broker"].pending()]

    @app.post("/api/approvals/{request_id}")
    async def decide(request_id: str, body: ApprovalBody) -> dict[str, bool]:
        if not state["broker"].resolve(request_id, body.approved, body.reason):
            raise HTTPException(404, "No pending approval with that id")
        return {"ok": True}

    @app.get("/api/kb/stats")
    async def kb_stats() -> dict[str, Any]:
        return state["kb"].stats()

    @app.post("/api/kb/ingest")
    async def kb_ingest(body: IngestBody) -> dict[str, Any]:
        n = await asyncio.to_thread(state["kb"].add_text, body.text, body.source, body.title)
        return {"chunks": n, **state["kb"].stats()}

    @app.post("/api/kb/search")
    async def kb_search(body: SearchBody) -> list[dict[str, Any]]:
        hits = await asyncio.to_thread(state["kb"].search, body.query, body.k, body.mode)
        return [h.__dict__ for h in hits]

    return app


def app_factory() -> FastAPI:  # for `uvicorn --factory atlas.api.app:app_factory`
    return create_app()
