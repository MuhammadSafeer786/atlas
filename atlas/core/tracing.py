"""Lightweight distributed-style tracing.

CONCEPT: you cannot debug what you cannot see. Each agent run is a *trace*;
every model call, tool call and sub-agent is a *span* with a parent, timing,
attributes and status. Spans nest automatically through `contextvars`, so
parallel sub-agents launched with asyncio.gather get correct parents.

Spans are written as JSON lines to `.atlas/traces/<trace_id>.jsonl`. The
format mirrors OpenTelemetry (trace_id / span_id / parent_id / attributes), so
swapping in an OTel exporter is a one-class change.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar("atlas_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 1)


class Tracer:
    def __init__(self, trace_dir: Path | None = None, trace_id: str | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.spans: list[Span] = []
        self.path = None
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = trace_dir / f"{self.trace_id}.jsonl"

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[Span]:
        parent = _current_span.get()
        span = Span(name, self.trace_id, parent_id=parent.span_id if parent else None, attributes=attributes)
        token = _current_span.set(span)
        try:
            yield span
        except BaseException as exc:
            span.status = f"error: {type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            span.end = time.time()
            _current_span.reset(token)
            self.spans.append(span)
            self._export(span)

    def _export(self, span: Span) -> None:
        if self.path is None:
            return
        record = asdict(span) | {"duration_ms": span.duration_ms}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def tree(self) -> str:
        """Render the trace as an indented tree (for the CLI)."""
        children: dict[str | None, list[Span]] = {}
        for s in sorted(self.spans, key=lambda s: s.start):
            children.setdefault(s.parent_id, []).append(s)
        lines: list[str] = []

        def walk(parent: str | None, depth: int) -> None:
            for s in children.get(parent, []):
                mark = "" if s.status == "ok" else f"  [{s.status}]"
                lines.append(f"{'  ' * depth}- {s.name} ({s.duration_ms} ms){mark}")
                walk(s.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)
