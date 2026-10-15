"""OpenAI-compatible adapter (Chat Completions API).

One adapter, many backends: OpenAI itself, plus every server that speaks the
same protocol - Ollama, LM Studio, vLLM, llama.cpp, LocalAI, OpenRouter, Groq,
Together and more. Only `base_url`, `api_key` and `model` change.

CONCEPTS
--------
* Message translation. Canonical blocks map onto OpenAI's shape:
    assistant tool_use blocks   -> assistant.tool_calls[] (arguments as a JSON *string*)
    user tool_result blocks     -> one {"role": "tool", "tool_call_id"} message each
    thinking blocks             -> dropped (OpenAI-style APIs don't take them back)
    mid-conversation system msg -> a <system-reminder> user message (portable across
                                   local chat templates that only allow one system msg)
* Streaming tool calls arrive as fragments keyed by `index`; we reassemble the
  id, name and argument string, then parse the JSON once at the end.
* Capability negotiation. Servers differ in what optional parameters they
  accept (`stream_options`, `reasoning_effort`, `response_format`). On a 400
  that names one, we drop it and retry, and remember for next time.
* Reasoning streams. Local reasoning models (DeepSeek-R1, Qwen3, ...) often
  stream `reasoning_content` / `reasoning` deltas; we surface them as thinking.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, TypeVar

import openai
from pydantic import BaseModel

from atlas.config import Settings
from atlas.llm.base import (
    Capabilities,
    DeltaCallback,
    TurnRequest,
    TurnResult,
    block_text,
    parse_with_repair,
    portable_tool_specs,
)
from atlas.llm.usage import Usage

T = TypeVar("T", bound=BaseModel)
REASONING_FAMILIES = ("gpt-5", "o1", "o3", "o4")


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            out.append({"role": "user", "content": f"<system-reminder>\n{block_text(content)}\n</system-reminder>"})
            continue
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if role == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                for b in content if b.get("type") == "tool_use"
            ]
            if text or calls:
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            continue
        user_parts: list[str] = []
        for b in content:  # tool messages must directly follow the assistant tool_calls
            if b.get("type") == "tool_result":
                text = block_text(b.get("content", "")) or "(empty)"
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                            "content": f"ERROR: {text}" if b.get("is_error") else text})
            elif text := block_text([b]):
                user_parts.append(text)
        if user_parts:
            out.append({"role": "user", "content": "\n\n".join(user_parts)})
    return out


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}}
            for t in portable_tool_specs(tools)]


def _parse_args(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"_invalid_json": raw}  # the agent loop turns this into an error result
    return value if isinstance(value, dict) else {"_invalid_json": raw}


class OpenAICompatLLM:
    def __init__(self, settings: Settings, client: openai.AsyncOpenAI | None = None):
        self.settings = settings
        self.capabilities = Capabilities(settings.provider)
        self.client = client or openai.AsyncOpenAI(
            api_key=settings.api_key or None, base_url=settings.base_url or None, max_retries=4,
        )
        # Optional features, switched off permanently if the server rejects them.
        self._stream_usage = True
        self._json_mode = "json_schema"  # -> json_object -> prompt-only
        self._effort_ok = True

    def _wants_effort(self, model: str) -> bool:
        mode = self.settings.reasoning_effort
        if not self._effort_ok:
            return False
        if mode != "auto":
            return mode == "on"
        return self.settings.provider == "openai" and model.startswith(REASONING_FAMILIES) and "chat" not in model

    async def _create(self, params: dict[str, Any]) -> Any:
        """chat.completions.create with graceful downgrade of optional parameters."""
        while True:
            try:
                return await self.client.chat.completions.create(**params)
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if "stream_options" in params and "stream_options" in msg:
                    params.pop("stream_options")
                    self._stream_usage = False
                elif "reasoning_effort" in params and "reasoning" in msg:
                    params.pop("reasoning_effort")
                    self._effort_ok = False
                else:
                    raise

    # -- agent turn (streaming) -------------------------------------------------
    async def turn(self, req: TurnRequest, on_delta: DeltaCallback) -> TurnResult:
        params: dict[str, Any] = {"model": req.model, "messages": to_openai_messages(req.system, req.messages),
                                  "stream": True}
        if req.tools:
            params["tools"] = to_openai_tools(req.tools)
        if self._stream_usage:
            params["stream_options"] = {"include_usage": True}
        if not self.settings.is_local:  # local servers size outputs to their own context window
            params["max_completion_tokens"] = req.max_tokens
        if self._wants_effort(req.model):
            params["reasoning_effort"] = req.effort

        stream = await self._create(params)
        text: list[str] = []
        refusal: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish, served, usage = None, req.model, Usage()
        async for chunk in stream:
            served = chunk.model or served
            if chunk.usage:
                details = chunk.usage.prompt_tokens_details
                cached = (details.cached_tokens if details else 0) or 0
                usage = Usage(input_tokens=(chunk.usage.prompt_tokens or 0) - cached,
                              output_tokens=chunk.usage.completion_tokens or 0, cache_read_input_tokens=cached)
            for choice in chunk.choices:
                delta = choice.delta
                extra = delta.model_extra or {}
                reasoning = extra.get("reasoning_content") or extra.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    await on_delta("thinking", {"text": reasoning})
                if delta.content:
                    text.append(delta.content)
                    await on_delta("text", {"text": delta.content})
                if delta.refusal:
                    refusal.append(delta.refusal)
                for tc in delta.tool_calls or []:
                    slot = calls.setdefault(tc.index if tc.index is not None else len(calls),
                                            {"id": None, "name": "", "args": ""})
                    slot["id"] = tc.id or slot["id"]
                    if tc.function:
                        slot["name"] = slot["name"] or (tc.function.name or "")
                        slot["args"] += tc.function.arguments or ""
                if choice.finish_reason:
                    finish = choice.finish_reason

        content: list[dict[str, Any]] = []
        if text or refusal:
            content.append({"type": "text", "text": "".join(text) or "".join(refusal)})
        for _, slot in sorted(calls.items()):
            content.append({"type": "tool_use", "id": slot["id"] or f"call_{uuid.uuid4().hex[:20]}",
                            "name": slot["name"], "input": _parse_args(slot["args"])})

        if refusal or finish == "content_filter":
            stop = "refusal"
        elif finish == "length":
            stop = "max_tokens"
        elif calls:  # some local servers report "stop" even when they called tools
            stop = "tool_use"
        else:
            stop = "end_turn"
        return TurnResult(content, stop, usage, served)

    # -- structured output -------------------------------------------------------
    async def parse(self, *, model: str, system: str, messages: list[dict[str, Any]],
                    output_format: type[T], effort: str = "medium") -> tuple[T, Usage]:
        async def call(sys_prompt: str, convo: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[str, Usage]:
            while True:
                params: dict[str, Any] = {"model": model, "messages": to_openai_messages(sys_prompt, convo)}
                if self._json_mode == "json_schema":
                    params["response_format"] = {"type": "json_schema", "json_schema": {
                        "name": output_format.__name__, "schema": schema, "strict": False}}
                elif self._json_mode == "json_object":
                    params["response_format"] = {"type": "json_object"}
                if self._wants_effort(model):
                    params["reasoning_effort"] = effort
                try:
                    resp = await self._create(params)
                    break
                except openai.BadRequestError:
                    if self._json_mode == "prompt":
                        raise
                    # Server lacks this JSON mode: degrade and rely on prompt + validation.
                    self._json_mode = "json_object" if self._json_mode == "json_schema" else "prompt"
            u = resp.usage
            usage = Usage(input_tokens=u.prompt_tokens or 0, output_tokens=u.completion_tokens or 0) if u else Usage()
            return resp.choices[0].message.content or "", usage

        return await parse_with_repair(call, system=system, messages=messages, output_format=output_format)
