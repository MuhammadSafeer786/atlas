"""Human-in-the-loop (HITL) approval.

CONCEPT: autonomy is a dial, not a switch. Read-only tools run freely;
anything hard to reverse (writing files, sending messages, spending money)
pauses the agent until a human approves. The agent loop doesn't know *how*
approval happens - it calls an `ApprovalPolicy`:

* `AutoApprove` / `AutoDeny` - for tests, evals and trusted batch jobs.
* `ConsolePolicy`            - asks in the terminal (CLI).
* `ApprovalBroker`           - asynchronous web approval: the agent emits an
                               `approval_request` event over SSE and awaits a
                               future that the `POST /approvals/{id}` endpoint
                               resolves. Times out to "deny" (fail closed).

A denial is not an error: Claude receives a tool_result saying the human
declined, and adapts its plan.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ApprovalRequest:
    tool: str
    input: dict[str, Any]
    agent: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class ApprovalDecision:
    approved: bool
    reason: str = ""


class ApprovalPolicy(Protocol):
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision: ...


class AutoApprove:
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(True, "auto-approved")


class AutoDeny:
    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(False, "auto-denied by policy")


class ConsolePolicy:
    """Blocking terminal prompt, run in a worker thread so the event loop keeps going."""

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()
        preview = json.dumps(request.input, indent=2)
        if len(preview) > 1500:
            preview = preview[:1500] + "\n…"
        console.print(f"\n[bold yellow]Approval needed[/] - [cyan]{request.agent}[/] wants to run "
                      f"[bold]{request.tool}[/]:\n{preview}")
        ok = await asyncio.to_thread(Confirm.ask, "Allow?", default=False)
        return ApprovalDecision(ok, "approved by user" if ok else "declined by user")


class ApprovalBroker:
    """Bridges agent runs and an HTTP API. One broker per server process."""

    def __init__(self, timeout_s: float = 300.0):
        self.timeout_s = timeout_s
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._requests: dict[str, ApprovalRequest] = {}

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self._pending[request.id] = future
        self._requests[request.id] = request
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        except asyncio.TimeoutError:
            return ApprovalDecision(False, f"no response within {self.timeout_s:.0f}s (fail closed)")
        finally:
            self._pending.pop(request.id, None)
            self._requests.pop(request.id, None)

    def resolve(self, request_id: str, approved: bool, reason: str = "") -> bool:
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(ApprovalDecision(approved, reason or ("approved" if approved else "declined")))
        return True

    def pending(self) -> list[ApprovalRequest]:
        return list(self._requests.values())


def policy_from_mode(mode: str) -> ApprovalPolicy:
    return {"auto_approve": AutoApprove(), "auto_deny": AutoDeny()}.get(mode) or ConsolePolicy()
