"""Drive the real Anthropic SDK against a mocked HTTP transport.

This exercises the production code path in `AnthropicLLM` end to end - request
body construction, beta headers, SSE stream parsing, thinking/text/tool_use
accumulation and `to_dict()` serialisation - without a network or API key.
"""

import json

import anthropic
import httpx2

from atlas.agents.schemas import Critique
from atlas.config import Settings
from atlas.llm.anthropic_llm import AnthropicLLM
from atlas.llm.base import TurnRequest


def _sse(*events: dict) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


STREAM = _sse(
    {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 1, "cache_read_input_tokens": 900,
                  "cache_creation_input_tokens": 0}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Need math."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Computing."}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {
        "type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {}}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "input_json_delta", "partial_json": "{\"expression\": "}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "input_json_delta", "partial_json": "\"2+2\"}"}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
     "usage": {"output_tokens": 42}},
    {"type": "message_stop"},
)


def _client(handler) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="test-key", max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


async def test_streaming_turn_through_real_sdk():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        captured["beta"] = request.headers.get("anthropic-beta", "")
        return httpx2.Response(200, content=STREAM, headers={"content-type": "text/event-stream"})

    llm = AnthropicLLM(Settings(_env_file=None), client=_client(handler))
    deltas = []

    async def on_delta(kind, payload):
        deltas.append((kind, payload))

    tools = [{"name": "calculator", "description": "math", "strict": True, "eager_input_streaming": True,
              "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                               "required": ["expression"], "additionalProperties": False}}]
    result = await llm.turn(TurnRequest("claude-opus-5-5", "SYS", [{"role": "user", "content": "2+2?"}],
                                        tools, "high", 64000), on_delta)

    # --- what we sent
    body = captured["body"]
    assert body["model"] == "claude-opus-5-5" and body["stream"] is True
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["fallbacks"] == "default"
    assert body["context_management"] == {"edits": [{"type": "compact_20260112"}]}
    assert body["tools"][0]["eager_input_streaming"] is True
    assert "server-side-fallback-2026-07-01" in captured["beta"] and "compact-2026-01-12" in captured["beta"]

    # --- what we parsed
    assert result.stop_reason == "tool_use" and result.model == "claude-opus-5-5"
    assert [b["type"] for b in result.content] == ["thinking", "text", "tool_use"]
    assert result.content[0]["signature"] == "sig-abc"  # must round-trip unchanged
    assert result.content[2]["input"] == {"expression": "2+2"}
    assert result.usage.cache_read_input_tokens == 900 and result.usage.output_tokens == 42
    assert ("thinking", {"text": "Need math."}) in deltas and ("text", {"text": "Computing."}) in deltas
    json.dumps(result.content)  # history must be JSON-serialisable for session persistence


async def test_structured_parse_through_real_sdk():
    critique = {"score": 9, "passes": True, "strengths": ["clear"], "issues": [], "revision_instructions": ""}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert body["output_config"]["effort"] == "medium"
        return httpx2.Response(200, json={
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(critique)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 50, "output_tokens": 20},
        })

    llm = AnthropicLLM(Settings(_env_file=None), client=_client(handler))
    parsed, usage = await llm.parse(model="claude-opus-5-5", system="judge",
                                    messages=[{"role": "user", "content": "rate"}],
                                    output_format=Critique, effort="medium")
    assert isinstance(parsed, Critique) and parsed.score == 9 and usage.input_tokens == 50
