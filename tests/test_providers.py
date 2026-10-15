"""Provider adapters, exercised through each vendor's real SDK against mocked HTTP.

No network or API keys: every request goes through the actual `openai` /
`google-genai` client code, and a mock transport plays the server.
"""

import base64
import json

import httpx
import httpx2
import openai
import pytest

from atlas.agents.schemas import Critique
from atlas.config import ConfigurationError, Settings
from atlas.core.agent import Agent, AgentConfig
from atlas.core.approvals import AutoDeny
from atlas.llm.anthropic_llm import AnthropicLLM
from atlas.llm.base import Capabilities, TurnRequest, inline_refs
from atlas.llm.factory import create_llm
from atlas.llm.gemini_llm import GeminiLLM, to_gemini_contents
from atlas.llm.openai_llm import OpenAICompatLLM, to_openai_messages
from atlas.rag.store import KnowledgeBase
from atlas.runtime import Runtime
from atlas.tools.builtin import builtin_tools
from atlas.tools.registry import ToolRegistry
from atlas.tools.web import FetchUrl, assert_public_url
from tests.conftest import FakeLLM, text, tool_use, turn

KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
        "ATLAS_BASE_URL", "ATLAS_PROVIDER", "ATLAS_MODEL", "ATLAS_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def mk(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# ----------------------------------------------------------------------- configuration
def test_provider_auto_detection(clean_env):
    assert mk().provider == "anthropic"
    clean_env.setenv("GEMINI_API_KEY", "g")
    s = mk()
    assert (s.provider, s.model, s.api_key) == ("gemini", "gemini-flash-latest", "g")
    clean_env.setenv("OPENAI_API_KEY", "o")
    assert mk().provider == "openai"  # OpenAI wins over Gemini in detection order
    clean_env.setenv("ATLAS_BASE_URL", "http://localhost:9999/v1")
    clean_env.setenv("ATLAS_MODEL", "my-model")
    s = mk()
    assert (s.provider, s.base_url, s.model) == ("openai_compatible", "http://localhost:9999/v1", "my-model")


def test_local_presets_and_validation(clean_env):
    s = mk(provider="ollama")
    assert s.base_url == "http://localhost:11434/v1" and s.model == "qwen3:8b" and s.is_local and s.has_credentials()
    assert mk(provider="lmstudio", model="qwen2.5-7b-instruct").base_url == "http://localhost:1234/v1"
    with pytest.raises(ValueError, match="ATLAS_MODEL is required"):
        mk(provider="vllm")
    assert not mk(provider="openai").has_credentials()


def test_factory_picks_adapter(clean_env):
    assert isinstance(create_llm(mk(provider="anthropic", api_key="a")), AnthropicLLM)
    assert isinstance(create_llm(mk(provider="openai", api_key="o")), OpenAICompatLLM)
    assert isinstance(create_llm(mk(provider="ollama")), OpenAICompatLLM)
    assert isinstance(create_llm(mk(provider="gemini", api_key="g")), GeminiLLM)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        create_llm(mk(provider="openai"))


def test_custom_pricing_enables_budgets(clean_env):
    from atlas.llm.usage import Usage

    create_llm(mk(provider="ollama", model="priced-model", price_input_per_mtok=1.0, price_output_per_mtok=2.0))
    assert Usage(1_000_000, 1_000_000).cost_usd("priced-model") == 3.0
    assert Usage(1_000_000, 1_000_000).cost_usd("never-heard-of-it") == 0.0


# ----------------------------------------------------------------------- canonical -> OpenAI
HISTORY = [
    {"role": "user", "content": "What is 6*7? Also check the docs."},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "plan", "signature": "s"},
        {"type": "text", "text": "Let me check."},
        tool_use("call_1", "calculator", {"expression": "6*7"}),
        tool_use("call_2", "search_knowledge_base", {"query": "docs"}),
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "42"},
        {"type": "tool_result", "tool_use_id": "call_2", "is_error": False, "content": [
            {"type": "search_result", "source": "a.md", "title": "A",
             "content": [{"type": "text", "text": "fact"}]}]},
    ]},
    {"role": "system", "content": "Wrap up now."},
]


def test_openai_message_translation():
    msgs = to_openai_messages("SYS", HISTORY)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assistant = msgs[2]
    assert assistant["content"] == "Let me check."  # thinking dropped
    assert [c["function"]["name"] for c in assistant["tool_calls"]] == ["calculator", "search_knowledge_base"]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"expression": "6*7"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
    assert msgs[4]["role"] == "tool" and "[source: a.md]" in msgs[4]["content"]
    assert msgs[5]["role"] == "user" and "<system-reminder>" in msgs[5]["content"]


# ----------------------------------------------------------------------- OpenAI over HTTP
def _openai_sse(*chunks: dict) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


def _chunk(delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> dict:
    choices = [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5.5",
            "choices": choices, "usage": usage}


TOOL_STREAM = _openai_sse(
    _chunk({"role": "assistant", "reasoning_content": "thinking about math"}),
    _chunk({"content": "Computing. "}),
    _chunk({"tool_calls": [{"index": 0, "id": "call_a", "type": "function",
                            "function": {"name": "calculator", "arguments": "{\"expr"}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "ession\": \"6*7\"}"}}]}),
    _chunk({}, finish="tool_calls"),
    _chunk(None, usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                        "prompt_tokens_details": {"cached_tokens": 60}}),
)
ANSWER_STREAM = _openai_sse(_chunk({"content": "6*7 = 42."}), _chunk({}, finish="stop"))


def _openai_llm(handler, **settings_kw) -> OpenAICompatLLM:
    settings = mk(**({"provider": "openai", "api_key": "k", "model": "gpt-5.5"} | settings_kw))
    client = openai.AsyncOpenAI(api_key="k", base_url=settings.base_url or None, max_retries=0,
                                http_client=openai.DefaultAsyncHttpxClient(
                                    transport=httpx2.MockTransport(handler)))
    return OpenAICompatLLM(settings, client=client)


async def test_openai_streaming_turn_reassembles_tool_calls(clean_env):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, content=TOOL_STREAM, headers={"content-type": "text/event-stream"})

    llm = _openai_llm(handler)
    deltas = []

    async def on_delta(kind, payload):
        deltas.append(kind)

    tools = [t.spec() for t in builtin_tools()]
    result = await llm.turn(TurnRequest("gpt-5.5", "SYS", [{"role": "user", "content": "6*7?"}], tools, "high"),
                            on_delta)
    body = bodies[0]
    assert body["reasoning_effort"] == "high" and body["max_completion_tokens"] == 64000
    assert body["stream_options"] == {"include_usage": True}
    assert body["tools"][0]["type"] == "function" and "strict" not in body["tools"][0]["function"]
    assert result.stop_reason == "tool_use"
    assert result.content == [
        {"type": "text", "text": "Computing. "},
        {"type": "tool_use", "id": "call_a", "name": "calculator", "input": {"expression": "6*7"}},
    ]
    assert (result.usage.input_tokens, result.usage.cache_read_input_tokens, result.usage.output_tokens) == (40, 60, 20)
    assert "thinking" in deltas and "text" in deltas


async def test_openai_local_server_downgrades_unsupported_options(clean_env):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "stream_options" in body:
            return httpx2.Response(400, json={"error": {"message": "unknown field: stream_options"}})
        return httpx2.Response(200, content=ANSWER_STREAM, headers={"content-type": "text/event-stream"})

    llm = _openai_llm(handler, provider="ollama", api_key="")

    async def on_delta(kind, payload):
        pass

    result = await llm.turn(TurnRequest("qwen3:8b", "SYS", [{"role": "user", "content": "hi"}]), on_delta)
    assert result.content[0]["text"] == "6*7 = 42."
    assert "max_completion_tokens" not in seen[0] and "reasoning_effort" not in seen[0]  # local defaults
    assert "stream_options" not in seen[1] and llm._stream_usage is False


async def test_openai_structured_output_with_repair(clean_env):
    replies = iter(["Sure! {\"score\": \"high\"}",  # invalid for the schema -> repair round
                    '```json\n{"score": 9, "passes": true, "strengths": [], "issues": [], '
                    '"revision_instructions": ""}\n```'])
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx2.Response(400, json={"error": {"message": "response_format json_schema unsupported"}})
        return httpx2.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": next(replies)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    llm = _openai_llm(handler, provider="lmstudio", model="local", api_key="")
    critique, usage = await llm.parse(model="local", system="judge", messages=[{"role": "user", "content": "x"}],
                                      output_format=Critique)
    assert critique.score == 9 and llm._json_mode == "json_object"
    assert "failed validation" in bodies[-1]["messages"][-1]["content"]  # repair feedback was sent
    assert usage.input_tokens == 20


async def test_agent_loop_end_to_end_on_openai_adapter(clean_env, settings):
    streams = iter([TOOL_STREAM, ANSWER_STREAM])
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, content=next(streams), headers={"content-type": "text/event-stream"})

    llm = _openai_llm(handler)
    agent = Agent(AgentConfig("t", "SYS", "gpt-5.5"), llm, ToolRegistry(builtin_tools()), settings,
                  approvals=AutoDeny())
    result = await agent.run("What is 6*7?")
    assert result.text == "6*7 = 42." and result.tool_calls == ["calculator"]
    tool_msg = bodies[1]["messages"][-1]
    assert tool_msg == {"role": "tool", "tool_call_id": "call_a", "content": "42"}


# ----------------------------------------------------------------------- Gemini
SIG = base64.b64encode(b"opaque-signature").decode()


def test_gemini_content_translation():
    history = [
        {"role": "user", "content": "6*7?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {**tool_use("atlas_1", "calculator", {"expression": "6*7"}), "_gemini_sig": SIG},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "atlas_1", "content": "42"}]},
        {"role": "system", "content": "Wrap up."},
    ]
    contents = to_gemini_contents(history)
    assert [c.role for c in contents] == ["user", "model", "user"]  # system reminder merged into user turn
    call_part = contents[1].parts[1]
    assert call_part.function_call.name == "calculator" and call_part.function_call.id is None
    assert call_part.thought_signature == b"opaque-signature"
    resp = contents[2].parts[0].function_response
    assert resp.name == "calculator" and resp.response == {"result": "42"}
    assert "<system-reminder>" in contents[2].parts[1].text


def _gemini_sse(*payloads: dict) -> bytes:
    return "".join(f"data: {json.dumps(p)}\r\n\r\n" for p in payloads).encode()


GEMINI_TOOL = _gemini_sse(
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "Considering math", "thought": True}]}}]},
    {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "calculator", "args": {"expression": "6*7"}}, "thoughtSignature": SIG}]},
        "finishReason": "STOP"}],
     "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10, "thoughtsTokenCount": 5},
     "modelVersion": "gemini-flash-latest"},
)
GEMINI_ANSWER = _gemini_sse(
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "It is 42."}]}, "finishReason": "STOP"}]})


def _gemini_llm(handler) -> GeminiLLM:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key="g", http_options=types.HttpOptions(
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    return GeminiLLM(mk(provider="gemini", api_key="g"), client=client)


async def test_agent_loop_end_to_end_on_gemini_preserves_thought_signatures(clean_env, settings):
    streams = iter([GEMINI_TOOL, GEMINI_ANSWER])
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=next(streams), headers={"content-type": "text/event-stream"})

    llm = _gemini_llm(handler)
    agent = Agent(AgentConfig("t", "SYS", "gemini-flash-latest"), llm, ToolRegistry(builtin_tools()), settings,
                  approvals=AutoDeny())
    result = await agent.run("What is 6*7?")
    assert result.text == "It is 42." and result.tool_calls == ["calculator"]

    first = bodies[0]
    assert first["systemInstruction"]["parts"][0]["text"] == "SYS"
    decls = first["tools"][0]["functionDeclarations"]
    assert {d["name"] for d in decls} >= {"calculator", "write_file"}
    thinking = first["generationConfig"]["thinkingConfig"]
    assert thinking.get("includeThoughts", thinking.get("include_thoughts")) is True

    replay = bodies[1]["contents"]
    model_turn = next(c for c in replay if c["role"] == "model")
    call = next(p for p in model_turn["parts"] if "functionCall" in p)
    assert call["thoughtSignature"] == SIG  # signature round-tripped unchanged
    response = replay[-1]["parts"][0]["functionResponse"]
    assert response["name"] == "calculator" and response["response"] == {"result": "42"}


async def test_gemini_safety_block_maps_to_refusal(clean_env):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_gemini_sse(
            {"candidates": [{"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "SAFETY"}]}))

    async def on_delta(kind, payload):
        pass

    result = await _gemini_llm(handler).turn(TurnRequest("m", "S", [{"role": "user", "content": "x"}]), on_delta)
    assert result.stop_reason == "refusal" and result.stop_details == {"category": "SAFETY"}


# ----------------------------------------------------------------------- capability-driven tools
def test_runtime_tools_follow_capabilities(settings):
    kb = KnowledgeBase(":memory:")
    claude_rt = Runtime(settings, FakeLLM(), kb)
    claude_specs = claude_rt.tools_for("assistant").specs(claude_rt.capabilities)
    assert {"type": "memory_20250818", "name": "memory"} in claude_specs
    assert any(s.get("type") == "web_search_20260209" for s in claude_specs)

    local_rt = Runtime(settings, FakeLLM(capabilities=Capabilities("ollama")), kb)
    specs = local_rt.tools_for("assistant").specs(local_rt.capabilities)
    names = [s["name"] for s in specs]
    assert "fetch_url" in names and "web_search" not in names  # no Tavily key
    assert not any("type" in s for s in specs)  # only portable function tools
    memory = next(s for s in specs if s["name"] == "memory")
    assert "command" in memory["input_schema"]["properties"]
    assert [s["name"] for s in local_rt.tools_for("analyst").specs()] == ["calculator", "search_knowledge_base"]

    settings_with_search = settings.model_copy(update={"tavily_api_key": "tvly"})
    rt = Runtime(settings_with_search, FakeLLM(capabilities=Capabilities("gemini")), kb)
    assert "web_search" in rt.tools_for("researcher").names()


async def test_fetch_url_blocks_ssrf(settings):
    for url in ("http://127.0.0.1:8000/admin", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd"):
        with pytest.raises(PermissionError):
            await assert_public_url(url)
    from atlas.tools.base import ToolContext

    out = await FetchUrl().execute({"url": "http://localhost:6379"}, ToolContext(settings))
    assert out.is_error and "non-public" in out.content


async def test_invalid_json_arguments_become_error_results(settings):
    llm = FakeLLM([turn(tool_use("x", "calculator", {"_invalid_json": "{expression: 6*7"})), turn(text("fixed"))])
    agent = Agent(AgentConfig("t", "S", "m"), llm, ToolRegistry(builtin_tools()), settings)
    await agent.run("go")
    block = llm.requests[1].messages[-1]["content"][0]
    assert block["is_error"] and "not valid JSON" in block["content"]


def test_inline_refs_resolves_nested_models():
    from atlas.agents.schemas import ResearchPlan

    schema = inline_refs(ResearchPlan.model_json_schema())
    assert "$defs" not in json.dumps(schema) and "$ref" not in json.dumps(schema)
    assert schema["properties"]["subtasks"]["items"]["properties"]["specialist"]["enum"]


def test_vendor_keys_in_dotenv_file_are_detected(clean_env, tmp_path):
    """Regression: GEMINI_API_KEY etc. written in .env (not the shell) must select the provider."""
    import os

    from atlas.config import get_settings

    (tmp_path / ".env").write_text(" GEMINI_API_KEY=from-dotenv\n ATLAS_MODEL=gemini-flash-latest\n")
    clean_env.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        s = get_settings()
        assert (s.provider, s.api_key, s.model) == ("gemini", "from-dotenv", "gemini-flash-latest")
    finally:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("ATLAS_MODEL", None)
        get_settings.cache_clear()


def test_missing_credentials_give_clear_error(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    with pytest.raises(ConfigurationError, match="No LLM credentials found"):
        create_llm(mk())
