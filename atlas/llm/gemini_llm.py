"""Google Gemini adapter (native `google-genai` SDK).

CONCEPTS
--------
* Message translation. Gemini uses `Content(role="user"|"model", parts=[...])`:
    text blocks        -> Part(text)
    tool_use blocks    -> Part(function_call=FunctionCall(name, args))
    tool_result blocks -> Part(function_response=FunctionResponse(name, response))
  Function responses are matched by *name*, so we look the name up from the
  tool_use id. Consecutive same-role contents are merged.
* Thought signatures. Gemini's thinking models attach an opaque
  `thought_signature` to parts (notably function calls). It must be sent back
  unchanged on the next request or tool calling fails. We keep it on the
  canonical block under a private `_gemini_sig` key (base64); other adapters
  ignore keys that start with "_".
* Thought summaries (`include_thoughts`) stream to the UI as thinking deltas.
"""

from __future__ import annotations

import base64
import uuid
from typing import Any, TypeVar

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
    tool_names_by_id,
)
from atlas.llm.usage import Usage

T = TypeVar("T", bound=BaseModel)
LOCAL_ID_PREFIX = "atlas_"  # ids we generated (Gemini returned none)
REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def _sig_bytes(block: dict[str, Any]) -> bytes | None:
    sig = block.get("_gemini_sig")
    return base64.b64decode(sig) if sig else None


def _sig_str(raw: bytes | None) -> str | None:
    return base64.b64encode(raw).decode() if raw else None


def to_gemini_contents(messages: list[dict[str, Any]]) -> list[Any]:
    from google.genai import types

    names = tool_names_by_id(messages)
    contents: list[types.Content] = []

    def add(role: str, parts: list[types.Part]) -> None:
        if not parts:
            return
        if contents and contents[-1].role == role:
            contents[-1].parts = [*(contents[-1].parts or []), *parts]
        else:
            contents.append(types.Content(role=role, parts=parts))

    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            add("user", [types.Part(text=f"<system-reminder>\n{block_text(content)}\n</system-reminder>")])
        elif isinstance(content, str):
            add("model" if role == "assistant" else "user", [types.Part(text=content)])
        elif role == "assistant":
            parts = []
            for b in content:
                sig, kind = _sig_bytes(b), b.get("type")
                if kind == "text" and (b.get("text") or sig):
                    parts.append(types.Part(text=b.get("text", ""), thought_signature=sig))
                elif kind == "thinking" and sig:
                    parts.append(types.Part(text=b.get("thinking", ""), thought=True, thought_signature=sig))
                elif kind == "tool_use":
                    call_id = None if b["id"].startswith(LOCAL_ID_PREFIX) else b["id"]
                    parts.append(types.Part(
                        function_call=types.FunctionCall(id=call_id, name=b["name"], args=b.get("input") or {}),
                        thought_signature=sig))
            add("model", parts)
        else:
            parts = []
            for b in content:
                if b.get("type") == "tool_result":
                    tid = b["tool_use_id"]
                    text = block_text(b.get("content", ""))
                    parts.append(types.Part(function_response=types.FunctionResponse(
                        id=None if tid.startswith(LOCAL_ID_PREFIX) else tid,
                        name=names.get(tid, "tool"),
                        response={"error": text} if b.get("is_error") else {"result": text})))
                elif text := block_text([b]):
                    parts.append(types.Part(text=text))
            add("user", parts)
    return contents


class GeminiLLM:
    capabilities = Capabilities("gemini")

    def __init__(self, settings: Settings, client: Any = None):
        from google import genai
        from google.genai import types

        self.settings = settings
        self.types = types
        self.client = client or genai.Client(
            api_key=settings.api_key or None,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=4)),
        )
        self._thoughts = settings.thinking_display == "summarized"
        self._json_schema_ok = True

    def _config(self, req: TurnRequest) -> Any:
        t = self.types
        tools = portable_tool_specs(req.tools)
        return t.GenerateContentConfig(
            system_instruction=req.system,
            max_output_tokens=req.max_tokens,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=s["name"], description=s["description"],
                                      parameters_json_schema=s["input_schema"]) for s in tools
            ])] if tools else None,
            thinking_config=t.ThinkingConfig(include_thoughts=True) if self._thoughts else None,
        )

    async def turn(self, req: TurnRequest, on_delta: DeltaCallback) -> TurnResult:
        from google.genai import errors

        contents = to_gemini_contents(req.messages)
        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=req.model, contents=contents, config=self._config(req))
        except errors.ClientError as exc:
            if not (self._thoughts and "think" in str(exc).lower()):
                raise
            self._thoughts = False  # model has no thinking support: retry without it
            stream = await self.client.aio.models.generate_content_stream(
                model=req.model, contents=contents, config=self._config(req))

        blocks: list[dict[str, Any]] = []
        finish, served, usage, blocked = None, req.model, Usage(), None
        async for chunk in stream:
            served = chunk.model_version or served
            meta = chunk.usage_metadata
            if meta:
                cached = meta.cached_content_token_count or 0
                usage = Usage(input_tokens=(meta.prompt_token_count or 0) - cached,
                              output_tokens=(meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0),
                              cache_read_input_tokens=cached)
            feedback = getattr(chunk, "prompt_feedback", None)
            if feedback is not None and feedback.block_reason:
                blocked = str(getattr(feedback.block_reason, "name", feedback.block_reason))
            for cand in (chunk.candidates or [])[:1]:
                if cand.finish_reason:
                    finish = getattr(cand.finish_reason, "name", str(cand.finish_reason))
                for part in (cand.content.parts if cand.content and cand.content.parts else []):
                    await self._absorb(part, blocks, on_delta)

        has_calls = any(b["type"] == "tool_use" for b in blocks)
        if blocked or finish in REFUSAL_REASONS:
            stop, details = "refusal", {"category": blocked or finish}
        elif finish == "MAX_TOKENS":
            stop, details = "max_tokens", None
        else:
            stop, details = ("tool_use" if has_calls else "end_turn"), None
            if finish == "MALFORMED_FUNCTION_CALL":
                await on_delta("warning", {"message": "Gemini produced a malformed function call"})
        return TurnResult(blocks, stop, usage, served, details)

    async def _absorb(self, part: Any, blocks: list[dict[str, Any]], on_delta: DeltaCallback) -> None:
        sig = _sig_str(part.thought_signature)
        last = blocks[-1] if blocks else None
        if part.function_call:
            fc = part.function_call
            block = {"type": "tool_use", "id": fc.id or f"{LOCAL_ID_PREFIX}{uuid.uuid4().hex[:16]}",
                     "name": fc.name, "input": dict(fc.args or {})}
            if sig:
                block["_gemini_sig"] = sig
            blocks.append(block)
        elif part.thought and part.text is not None:
            await on_delta("thinking", {"text": part.text})
            if last and last["type"] == "thinking" and "_gemini_sig" not in last:
                last["thinking"] += part.text
            else:
                blocks.append({"type": "thinking", "thinking": part.text})
                last = blocks[-1]
            if sig:
                last["_gemini_sig"] = sig
        elif part.text is not None or sig:
            text = part.text or ""
            if text:
                await on_delta("text", {"text": text})
            if last and last["type"] == "text" and "_gemini_sig" not in last:
                last["text"] += text
            else:
                blocks.append({"type": "text", "text": text})
                last = blocks[-1]
            if sig:
                last["_gemini_sig"] = sig

    async def parse(self, *, model: str, system: str, messages: list[dict[str, Any]],
                    output_format: type[T], effort: str = "medium") -> tuple[T, Usage]:
        from google.genai import errors

        async def call(sys_prompt: str, convo: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[str, Usage]:
            t = self.types
            while True:
                config = t.GenerateContentConfig(
                    system_instruction=sys_prompt, response_mime_type="application/json",
                    response_json_schema=schema if self._json_schema_ok else None)
                try:
                    resp = await self.client.aio.models.generate_content(
                        model=model, contents=to_gemini_contents(convo), config=config)
                    break
                except errors.ClientError:
                    if not self._json_schema_ok:
                        raise
                    self._json_schema_ok = False  # fall back to JSON mode + validation
            meta = resp.usage_metadata
            usage = Usage(input_tokens=meta.prompt_token_count or 0,
                          output_tokens=(meta.candidates_token_count or 0) + (meta.thoughts_token_count or 0)) \
                if meta else Usage()
            return resp.text or "", usage

        return await parse_with_repair(call, system=system, messages=messages, output_format=output_format)
