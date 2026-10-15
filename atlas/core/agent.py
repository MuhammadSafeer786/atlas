"""The agent loop - the single most important file in the project.

CONCEPT: an "agent" is an LLM using tools in a loop, deciding for itself what
to do next, until the task is done:

    messages = [user task]
    loop:
        response = model(system, tools, messages)        # think + act
        append response to messages                      # append-only history
        if response asks for tools:
            results = run tools (parallel where safe)     # observe
            append ONE user message holding every result
        else:
            return the answer

This is the ReAct pattern (reason -> act -> observe) as Claude implements it
natively via `tool_use` / `tool_result` blocks. Everything else in this file
is what makes the loop production-grade:

* Stop-reason handling  - end_turn, tool_use, pause_turn (server tools want to
                          continue), max_tokens (a tool input may be truncated -
                          never run it), refusal (stop cleanly).
* Parallel tool calls   - Claude can request several tools at once; read-only
                          tools run concurrently, side-effecting tools serially.
                          All results go back in a single user message.
* Human approval        - tools flagged `requires_approval` pause for a human.
* Untrusted output      - third-party content is fenced before Claude reads it.
* Budgets               - max turns and a hard USD ceiling; a mid-conversation
                          `system` message warns the model before the limit so it
                          wraps up gracefully instead of being cut off.
* Errors as data        - a failing tool returns `is_error: true` so the model can
                          recover; it never crashes the loop.
* Append-only history   - we never edit earlier turns. Claude's thinking blocks are
                          bound to the exact conversation that produced them, and
                          an unchanged prefix is what keeps the prompt cache warm.
* Tracing + events      - every turn and tool call is a span and a streamed event.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from atlas.config import Settings
from atlas.core.approvals import ApprovalPolicy, ApprovalRequest, AutoDeny
from atlas.core.events import AgentEvent, EventSink, EventType, null_sink
from atlas.core.guardrails import check_user_input, fence_untrusted
from atlas.core.tracing import Tracer
from atlas.llm.base import LLM, TurnRequest
from atlas.llm.usage import CostTracker
from atlas.tools.base import ToolContext, ToolOutput
from atlas.tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    model: str
    effort: str = "medium"
    max_turns: int = 25
    max_tokens: int = 64_000
    guard_input: bool = True


@dataclass
class RunResult:
    text: str
    stop: str  # completed | max_turns | budget | refusal | max_tokens | blocked
    turns: int
    messages: list[dict[str, Any]]
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLM,
        tools: ToolRegistry,
        settings: Settings,
        *,
        approvals: ApprovalPolicy | None = None,
        tracer: Tracer | None = None,
        costs: CostTracker | None = None,
        sink: EventSink = null_sink,
    ):
        self.config = config
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.approvals = approvals or AutoDeny()  # fail closed by default
        self.tracer = tracer or Tracer()
        self.costs = costs or CostTracker()
        self.sink = sink
        self._serial = asyncio.Lock()

    async def emit(self, type_: str, **data: Any) -> None:
        await self.sink(AgentEvent(type_, self.config.name, data))

    # ------------------------------------------------------------------ run
    async def run(self, task: str | list[dict[str, Any]], history: list[dict[str, Any]] | None = None) -> RunResult:
        cfg = self.config
        if isinstance(task, str) and cfg.guard_input:
            guard = check_user_input(task)
            if guard.findings:
                await self.emit(EventType.GUARDRAIL, stage="input", findings=guard.findings)
            if not guard.allowed:
                await self.emit(EventType.ERROR, message=f"Input blocked: {guard.reason}")
                return RunResult(f"Input blocked: {guard.reason}", "blocked", 0, list(history or []))
            task = guard.text

        messages: list[dict[str, Any]] = [*(history or []), {"role": "user", "content": task}]
        tool_calls: list[str] = []
        warned = nudged = False
        await self.emit(EventType.RUN_START, model=cfg.model, effort=cfg.effort, tools=self.tools.names())

        async with self.tracer.span(f"agent:{cfg.name}", model=cfg.model, effort=cfg.effort) as run_span:
            for turn in range(1, cfg.max_turns + 1):
                if self.costs.total_usd >= self.settings.max_cost_usd:
                    return await self._finish(messages, "budget", turn - 1, tool_calls,
                                              note=f"Stopped: cost budget ${self.settings.max_cost_usd:.2f} reached.")

                # Graceful wind-down: an operator-authority system message appended to
                # the transcript (not an edit of the system prompt -> cache stays valid).
                if not warned and turn >= cfg.max_turns - 1 and messages[-1]["role"] == "user":
                    messages.append({"role": "system", "content": (
                        "You are about to reach your step limit. Stop calling tools and give "
                        "your best final answer now, noting anything left unverified.")})
                    warned = True

                async with self.tracer.span("llm.turn", turn=turn) as span:
                    result = await self.llm.turn(
                        TurnRequest(cfg.model, cfg.system_prompt, messages,
                                    self.tools.specs(getattr(self.llm, "capabilities", None)),
                                    cfg.effort, cfg.max_tokens),
                        self._on_delta,
                    )
                    cost = self.costs.record(result.model, result.usage)
                    span.set(stop_reason=result.stop_reason, served_by=result.model,
                             input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens,
                             cache_read=result.usage.cache_read_input_tokens, cost_usd=round(cost, 5))
                await self.emit(EventType.USAGE, turn=turn, served_by=result.model,
                                turn_cost_usd=round(cost, 5), **self.costs.summary())

                # Empty-response recovery: some models (often small local ones) reason
                # and then end the turn with neither text nor a tool call. Record nothing
                # for that turn and nudge once instead of returning an empty answer.
                if (result.stop_reason == "end_turn" and not has_output(result.content) and not nudged
                        and messages[-1]["role"] == "user"):  # system messages must follow a user turn
                    nudged = True
                    await self.emit(EventType.WARNING, message="empty response from model; retrying once")
                    messages.append({"role": "system", "content": (
                        "Your previous reply was empty. Respond now: call a tool if one is needed, "
                        "otherwise answer the user directly.")})
                    continue

                messages.append({"role": "assistant", "content": result.content})

                if result.stop_reason == "refusal":
                    await self.emit(EventType.REFUSAL, details=result.stop_details)
                    # A refusal can cut a tool_use off mid-input: never run it, but answer
                    # it so the session history stays valid for the next user message.
                    dangling = [b for b in result.content if b.get("type") == "tool_use"]
                    if dangling:
                        messages.append({"role": "user", "content": [self._tool_result(
                            b["id"], ToolOutput.error("Not executed (response was declined).")
                        ) for b in dangling]})
                    return await self._finish(messages, "refusal", turn, tool_calls,
                                              note="The model declined this request.")
                if result.stop_reason == "pause_turn":
                    continue  # a server tool is mid-flight; re-send to let it continue

                tool_uses = [b for b in result.content if b.get("type") == "tool_use"]
                if not tool_uses:
                    stop = "max_tokens" if result.stop_reason == "max_tokens" else "completed"
                    run_span.set(turns=turn, stop=stop)
                    return await self._finish(messages, stop, turn, tool_calls)

                if result.stop_reason == "max_tokens":
                    # The last tool input was cut off mid-generation. Every tool_use must
                    # still get a tool_result, but we must not execute truncated input.
                    results = [self._tool_result(b["id"], ToolOutput.error(
                        "Not executed: your response hit max_tokens and this input was truncated. "
                        "Retry with smaller pieces.")) for b in tool_uses]
                else:
                    tool_calls += [b["name"] for b in tool_uses]
                    results = await self._run_tools(tool_uses)
                # All results in ONE user message - splitting them teaches the model
                # to stop making parallel calls.
                messages.append({"role": "user", "content": results})

            run_span.set(turns=cfg.max_turns, stop="max_turns")
            return await self._finish(messages, "max_turns", cfg.max_turns, tool_calls,
                                      note="Stopped: maximum number of steps reached.")

    # ------------------------------------------------------------------ tools
    async def _run_tools(self, tool_uses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(self._run_one(b) for b in tool_uses)))

    async def _run_one(self, block: dict[str, Any]) -> dict[str, Any]:
        name, tool_input, tool_id = block["name"], block.get("input") or {}, block["id"]
        await self.emit(EventType.TOOL_CALL, id=tool_id, name=name, input=tool_input)
        tool = self.tools.get(name)
        if tool is None:
            return self._tool_result(tool_id, ToolOutput.error(f"Unknown tool: {name}"))
        if "_invalid_json" in tool_input:  # adapters flag unparseable arguments this way
            return self._tool_result(tool_id, ToolOutput.error(
                f"Your arguments for {name} were not valid JSON: {tool_input['_invalid_json'][:500]}"))

        if tool.parallel_safe and not tool.requires_approval:
            output, elapsed = await self._execute(tool, tool_input, tool_id)
        else:
            async with self._serial:  # side effects + approval prompts happen one at a time
                if tool.requires_approval:
                    request = ApprovalRequest(tool=name, input=tool_input, agent=self.config.name)
                    await self.emit(EventType.APPROVAL_REQUEST, id=request.id, tool=name, input=tool_input)
                    decision = await self.approvals.decide(request)
                    await self.emit(EventType.APPROVAL_RESULT, id=request.id, tool=name,
                                    approved=decision.approved, reason=decision.reason)
                    if not decision.approved:
                        out = ToolOutput.error(f"The human reviewer declined this action ({decision.reason}). "
                                               "Do not retry it; continue without it or ask the user.")
                        await self.emit(EventType.TOOL_RESULT, id=tool_id, name=name, is_error=True,
                                        preview=out.preview(), duration_ms=0)
                        return self._tool_result(tool_id, out)
                output, elapsed = await self._execute(tool, tool_input, tool_id)

        await self.emit(EventType.TOOL_RESULT, id=tool_id, name=name, is_error=output.is_error,
                        preview=output.preview(), duration_ms=elapsed)
        return self._tool_result(tool_id, output)

    async def _execute(self, tool: Any, tool_input: dict[str, Any], tool_id: str) -> tuple[ToolOutput, float]:
        ctx = ToolContext(self.settings, agent=self.config.name, run_id=self.tracer.trace_id)
        start = time.perf_counter()
        async with self.tracer.span(f"tool:{tool.name}") as span:
            try:
                output = await asyncio.wait_for(tool.execute(tool_input, ctx), self.settings.tool_timeout_s)
            except asyncio.TimeoutError:
                output = ToolOutput.error(f"Tool timed out after {self.settings.tool_timeout_s:.0f}s")
            except Exception as exc:  # noqa: BLE001 - tool failures are data for the model
                output = ToolOutput.error(f"Tool raised {type(exc).__name__}: {exc}")
            if tool.untrusted_output and not output.is_error:
                content, hits = fence_untrusted(output.content, source=tool.name)
                output = ToolOutput(content, output.is_error)
                if hits:
                    span.set(injection_flags=hits)
                    await self.emit(EventType.GUARDRAIL, stage="tool_output", tool=tool.name, findings=hits)
            span.set(is_error=output.is_error)
        return output, round((time.perf_counter() - start) * 1000, 1)

    @staticmethod
    def _tool_result(tool_use_id: str, output: ToolOutput) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": output.content}
        if output.is_error:
            block["is_error"] = True
        return block

    # ------------------------------------------------------------------ helpers
    async def _on_delta(self, kind: str, payload: dict[str, Any]) -> None:
        mapping = {"text": EventType.TEXT, "thinking": EventType.THINKING,
                   "server_tool": EventType.SERVER_TOOL, "warning": EventType.WARNING}
        await self.emit(mapping.get(kind, kind), **payload)

    async def _finish(self, messages: list[dict[str, Any]], stop: str, turns: int,
                      tool_calls: list[str], note: str = "") -> RunResult:
        text, citations = final_text(messages)
        if not text and not note:
            note = ("The model returned an empty response. Small local models often struggle with "
                    "tool use - try a larger model.")
        if note:
            text = f"{text}\n\n_{note}_".strip()
        await self.emit(EventType.RUN_END, stop=stop, turns=turns, text=text, **self.costs.summary())
        return RunResult(text, stop, turns, messages, citations, tool_calls)


def has_output(content: list[dict[str, Any]]) -> bool:
    """True if a turn produced visible text or any non-reasoning block (e.g. a tool call)."""
    for block in content:
        kind = block.get("type")
        if kind == "text":
            if block.get("text", "").strip():
                return True
        elif kind not in ("thinking", "redacted_thinking"):
            return True
    return False


def final_text(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Text + citations from the last assistant message."""
    for message in reversed(messages):
        if message["role"] != "assistant":
            continue
        parts, citations = [], []
        for block in message["content"]:
            if block.get("type") == "text":
                parts.append(block["text"])
                citations += block.get("citations") or []
        return "".join(parts).strip(), citations
    return "", []
