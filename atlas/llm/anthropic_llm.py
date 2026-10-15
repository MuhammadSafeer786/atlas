"""Claude adapter: the only module that talks to the Anthropic API.

CONCEPTS demonstrated here
--------------------------
* Streaming           - every agent turn streams, so long outputs never hit HTTP
                        timeouts and the UI shows tokens as they arrive.
* Adaptive thinking   - Claude decides how much to reason per request; we surface
                        a summarized view of that reasoning to the UI.
* Effort              - the per-call dial between thoroughness and token spend.
* Prompt caching      - the static prefix (tools + system prompt) is cached with an
                        explicit breakpoint; top-level auto-caching covers the growing
                        conversation tail. Cached reads cost ~10% of normal input.
* Compaction (beta)   - when a conversation approaches the context limit the API
                        summarises older turns server-side; we just round-trip the
                        returned `compaction` blocks.
* Refusal fallbacks   - `fallbacks="default"` re-runs a safety-declined request on
                        Anthropic's recommended fallback model inside the same call.
* Structured outputs  - `parse()` constrains the response to a Pydantic schema.
* Eager input streaming - large tool inputs stream as generated; we guard the
                        parse failures that this moves to the client.

Keeping all of this behind one small interface (`LLM`) means the agent loop is
provider-agnostic and fully unit-testable with a scripted fake (see tests/).
"""

from __future__ import annotations

import copy
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from atlas.config import Settings
from atlas.llm.base import Capabilities, DeltaCallback, RefusalError, TurnRequest, TurnResult
from atlas.llm.usage import Usage

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
COMPACTION_BETA = "compact-2026-01-12"
# Server-side fallbacks exist for the Opus / Fable / Sonnet 5.5 lines, not Haiku.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop other adapters' private "_" keys and empty text blocks before sending."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list):
            content = [
                {k: v for k, v in b.items() if not k.startswith("_")}
                for b in content
                if not (b.get("type") == "text" and not b.get("text"))
            ]
        out.append({**m, "content": content})
    return out


class AnthropicLLM:
    capabilities = Capabilities("anthropic", server_tools=True, native_memory_tool=True, native_citations=True)

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None):
        self.settings = settings
        # The SDK already retries 408/409/429/5xx with exponential backoff.
        # With no explicit key it resolves ANTHROPIC_API_KEY / auth token / `ant` profile.
        self.client = client or anthropic.AsyncAnthropic(api_key=settings.api_key or None, max_retries=4)

    # -- request construction --------------------------------------------------
    def _base_params(self, model: str, system: str, effort: str) -> tuple[dict[str, Any], list[str]]:
        params: dict[str, Any] = {
            "model": model,
            # Stable system prompt with an explicit cache breakpoint. Tools render
            # before system, so this breakpoint caches tools + system together.
            # Never put timestamps or request IDs here - they silently bust the cache.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "thinking": {"type": "adaptive", "display": self.settings.thinking_display},
            "output_config": {"effort": effort},
        }
        betas: list[str] = []
        if self.settings.enable_fallbacks and model in FALLBACK_MODELS:
            params["fallbacks"] = "default"
            betas.append(FALLBACK_BETA)
        return params, betas

    # -- agent turn (streaming) -------------------------------------------------
    async def turn(self, req: TurnRequest, on_delta: DeltaCallback) -> TurnResult:
        params, betas = self._base_params(req.model, req.system, req.effort)
        params.update(
            max_tokens=req.max_tokens,
            messages=_sanitize(copy.deepcopy(req.messages)),
            # Auto-cache the last cacheable block -> the conversation tail is reused
            # on the next turn of the agent loop.
            cache_control={"type": "ephemeral"},
        )
        if req.tools:
            params["tools"] = req.tools
        if self.settings.enable_compaction:
            params["context_management"] = {"edits": [{"type": "compact_20260112"}]}
            betas.append(COMPACTION_BETA)
        if betas:
            params["betas"] = betas

        json_retries = 0
        while True:
            try:
                async with self.client.beta.messages.stream(**params) as stream:
                    async for event in stream:
                        await self._forward(event, on_delta)
                    final = await stream.get_final_message()
                break
            except ValueError:
                # Eager input streaming: the SDK could not parse a tool input at all.
                # The tool_use block never completed, so re-issue the turn (bounded).
                # Typed API errors are not ValueError and propagate untouched.
                json_retries += 1
                if json_retries > 2:
                    raise
                await on_delta("warning", {"message": "malformed tool input; retrying turn"})

        stop_details = final.stop_details.to_dict() if getattr(final, "stop_details", None) else None
        return TurnResult(
            content=[block.to_dict() for block in final.content],
            stop_reason=final.stop_reason,
            usage=Usage.from_api(final.usage),
            model=final.model,
            stop_details=stop_details,
        )

    @staticmethod
    async def _forward(event: Any, on_delta: DeltaCallback) -> None:
        """Translate raw SDK stream events into our small delta vocabulary."""
        if event.type == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                await on_delta("text", {"text": delta.text})
            elif delta.type == "thinking_delta" and delta.thinking:
                await on_delta("thinking", {"text": delta.thinking})
        elif event.type == "content_block_stop":
            block = getattr(event, "content_block", None)
            if block is not None and block.type == "server_tool_use":
                await on_delta("server_tool", {"name": block.name, "input": block.input})
            elif block is not None and block.type == "compaction":
                await on_delta("warning", {"message": "context compacted server-side"})

    # -- structured output (non-streaming, short) ------------------------------
    async def parse(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        output_format: type[T],
        effort: str = "medium",
    ) -> tuple[T, Usage]:
        params, betas = self._base_params(model, system, effort)
        if betas:
            params["betas"] = betas
        response = await self.client.beta.messages.parse(
            max_tokens=16_000, messages=_sanitize(messages), output_format=output_format, **params
        )
        if response.stop_reason == "refusal":
            details = response.stop_details.to_dict() if response.stop_details else None
            raise RefusalError(details)
        if response.parsed_output is None:
            raise ValueError(f"No structured output (stop_reason={response.stop_reason})")
        return response.parsed_output, Usage.from_api(response.usage)
