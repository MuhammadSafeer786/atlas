import json

from fastapi.testclient import TestClient

from atlas.api.app import create_app
from atlas.config import Settings
from atlas.llm.anthropic_llm import AnthropicLLM
from atlas.llm.usage import Usage
from atlas.mcp_server.server import build_server
from atlas.rag.embeddings import HashingEmbedder
from atlas.rag.store import KnowledgeBase
from atlas.tools.base import ToolContext
from atlas.tools.mcp_bridge import MCPToolset
from tests.conftest import FakeLLM, text, tool_use, turn


def _kb(knowledge_dir):
    kb = KnowledgeBase(":memory:", embedder=HashingEmbedder())
    kb.add_path(knowledge_dir)
    return kb


# ------------------------------------------------------------------ MCP
async def test_mcp_server_tools_via_bridge(knowledge_dir, settings):
    server = build_server(_kb(knowledge_dir))
    async with MCPToolset({"atlas": server}) as toolset:
        names = sorted(t.name for t in toolset.tools)
        assert names == ["atlas__add_note", "atlas__list_sources", "atlas__search_knowledge_base"]
        search = next(t for t in toolset.tools if t.name == "atlas__search_knowledge_base")
        assert search.spec()["input_schema"]["properties"]["query"]["type"] == "string"
        out = await search.execute({"query": "Growth plan monthly price"}, ToolContext(settings))
        assert not out.is_error and "$249" in out.preview(2000)

        add = next(t for t in toolset.tools if t.name == "atlas__add_note")
        await add.execute({"title": "Decision", "content": "We picked the Growth plan."}, ToolContext(settings))
        out = await search.execute({"query": "which plan did we pick"}, ToolContext(settings))
        assert "Growth plan" in out.preview(4000)


async def test_mcp_resources_and_prompts(knowledge_dir):
    from mcp import Client

    async with Client(build_server(_kb(knowledge_dir))) as client:
        resources = await client.list_resources()
        assert {str(r.uri) for r in resources.resources} >= {"atlas://kb/stats", "atlas://kb/sources"}
        prompts = await client.list_prompts()
        assert "research_brief" in [p.name for p in prompts.prompts]


# ------------------------------------------------------------------ API
def _sse_events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]


def test_api_chat_streams_and_persists_session(settings, knowledge_dir, tmp_path):
    llm = FakeLLM([
        turn(tool_use("s", "search_knowledge_base", {"query": "Growth SLA"})),
        turn(text("The Growth plan has a 99.9% SLA.")),
        turn(text("Second answer.")),
    ])
    app = create_app(settings, llm=llm, kb=_kb(knowledge_dir), mcp_config=tmp_path / "none.json")
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert "Atlas" in client.get("/").text

        res = client.post("/api/chat", json={"message": "What is the Growth SLA?"})
        events = _sse_events(res.text)
        types = [e["type"] for e in events]
        assert "tool_call" in types and "text_delta" in types and types[-1] == "done"
        done = events[-1]["data"]
        assert done["text"] == "The Growth plan has a 99.9% SLA." and done["trace_id"]

        res = client.post("/api/chat", json={"message": "and?", "session_id": done["session_id"]})
        assert _sse_events(res.text)[-1]["data"]["text"] == "Second answer."
        assert len(llm.requests[-1].messages) == 5  # previous 4 messages + new question


def test_api_kb_and_approvals(settings, tmp_path):
    app = create_app(settings, llm=FakeLLM(), kb=KnowledgeBase(":memory:", embedder=HashingEmbedder()),
                     mcp_config=tmp_path / "none.json")
    with TestClient(app) as client:
        r = client.post("/api/kb/ingest", json={"text": "Atlas supports MCP servers.", "source": "note1"})
        assert r.json()["chunks"] == 1
        hits = client.post("/api/kb/search", json={"query": "MCP"}).json()
        assert hits[0]["source"] == "note1"
        assert client.get("/api/approvals").json() == []
        assert client.post("/api/approvals/nope", json={"approved": True}).status_code == 404


# ------------------------------------------------------------------ LLM gateway
def test_request_construction_enables_caching_thinking_and_fallbacks():
    llm = AnthropicLLM(Settings(_env_file=None), client=object())  # client unused here
    params, betas = llm._base_params("claude-opus-5-5", "SYSTEM", "high")
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["thinking"]["type"] == "adaptive" and params["output_config"] == {"effort": "high"}
    assert params["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in betas
    params, betas = llm._base_params("claude-haiku-5-5", "SYSTEM", "low")
    assert "fallbacks" not in params and not betas  # no server-side fallback for Haiku


def test_usage_cost_accounting():
    u = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=2_000_000)
    # 1M * $4 + 0.1M * $20 + 2M * $0.20
    assert round(u.cost_usd("claude-opus-5-5"), 4) == 6.4
    assert Usage.from_api({"input_tokens": 5, "server_tool_use": {"web_search_requests": 2}}).web_search_requests == 2
