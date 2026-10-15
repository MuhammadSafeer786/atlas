"""Multi-agent orchestration: plan -> parallel specialists -> write -> critique -> revise.

CONCEPTS
--------
* Workflow vs. agent. The *outer* control flow here is ordinary code (a
  workflow): predictable, testable, cheap to reason about. Model-driven
  autonomy lives *inside* each step, where open-ended exploration pays off.
* Planning (plan-and-execute). A planner produces a typed `ResearchPlan` via
  structured outputs: subtasks, the specialist for each, dependencies and
  success criteria.
* Orchestrator-workers. Subtasks form a DAG; we execute it level by level
  (Kahn's topological sort). Independent subtasks run concurrently, bounded by
  a semaphore. Each worker receives only what it needs: its instructions plus
  its dependencies' outputs - small focused contexts beat one giant one.
* Specialisation. Each role gets its own prompt and the minimum tool set
  (least privilege): the librarian can't browse the web, the writer has no
  tools at all.
* Evaluator-optimizer (reflection). A critic scores the draft against the
  plan's success criteria with a structured `Critique`; if it fails, the
  writer revises *in the same conversation* (append-only), up to N rounds.
* Graceful degradation. A failed worker becomes a noted gap, not a crash.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from atlas.agents import prompts
from atlas.agents.schemas import Critique, ResearchPlan, Subtask
from atlas.core.events import AgentEvent, EventType
from atlas.runtime import Runtime


@dataclass
class ResearchOutcome:
    report: str
    plan: ResearchPlan
    findings: dict[str, str]
    critiques: list[Critique] = field(default_factory=list)
    cost: dict = field(default_factory=dict)


class PlanError(ValueError):
    pass


def execution_levels(plan: ResearchPlan) -> list[list[Subtask]]:
    """Group subtasks into dependency levels (Kahn's algorithm). Raises on cycles."""
    by_id = {t.id: t for t in plan.subtasks}
    if len(by_id) != len(plan.subtasks):
        raise PlanError("duplicate subtask ids")
    deps = {t.id: {d for d in t.depends_on if d in by_id and d != t.id} for t in plan.subtasks}
    levels: list[list[Subtask]] = []
    done: set[str] = set()
    while len(done) < len(by_id):
        ready = [by_id[i] for i, d in deps.items() if i not in done and d <= done]
        if not ready:
            raise PlanError(f"dependency cycle among: {sorted(set(by_id) - done)}")
        levels.append(ready)
        done |= {t.id for t in ready}
    return levels


class Orchestrator:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.settings = rt.settings

    async def emit(self, type_: str, **data) -> None:
        await self.rt.sink(AgentEvent(type_, "orchestrator", data))

    # ------------------------------------------------------------ public API
    async def run(self, objective: str) -> ResearchOutcome:
        if not self.rt.has_web_search:
            await self.emit(EventType.WARNING, message=(
                "Web search is unavailable with this provider/configuration - the team can only use the "
                "knowledge base and known URLs. Set TAVILY_API_KEY (free tier) to enable search."))
        async with self.rt.tracer.span("orchestrator", objective=objective[:200]):
            plan = await self.plan(objective)
            findings = await self.execute(plan)
            report, critiques = await self.write_and_review(plan, findings)
        return ResearchOutcome(report, plan, findings, critiques, self.rt.costs.summary())

    # ------------------------------------------------------------ 1. plan
    async def plan(self, objective: str) -> ResearchPlan:
        async with self.rt.tracer.span("plan") as span:
            plan, usage = await self.rt.llm.parse(
                model=self.settings.model,
                system=prompts.PLANNER,
                messages=[{"role": "user", "content": objective if self.rt.has_web_search
                           else f"{objective}\n\n{self.rt.NO_SEARCH_NOTE}"}],
                output_format=ResearchPlan,
                effort=self.settings.orchestrator_effort,
            )
            self.rt.costs.record(self.settings.model, usage)
            if not plan.subtasks:
                raise PlanError("planner returned no subtasks")
            levels = execution_levels(plan)  # validate early
            span.set(subtasks=len(plan.subtasks), levels=len(levels))
        await self.emit(EventType.PLAN, plan=plan.model_dump())
        return plan

    # ------------------------------------------------------------ 2. execute
    async def execute(self, plan: ResearchPlan) -> dict[str, str]:
        outputs: dict[str, str] = {}
        limit = asyncio.Semaphore(self.settings.max_parallel_workers)

        async def run_one(task: Subtask) -> None:
            async with limit:
                outputs[task.id] = await self._run_worker(task, outputs)

        for level in execution_levels(plan):
            await asyncio.gather(*(run_one(t) for t in level))
        return outputs

    async def _run_worker(self, task: Subtask, outputs: dict[str, str]) -> str:
        name = f"{task.specialist}:{task.id}"
        context = "\n\n".join(
            f"<dependency id=\"{d}\">\n{outputs[d]}\n</dependency>" for d in task.depends_on if d in outputs
        )
        brief = f"## Subtask: {task.title}\n\n{task.instructions}"
        if context:
            brief += f"\n\n## Results from earlier subtasks\n{context}"
        if not self.rt.has_web_search and task.specialist in ("researcher", "analyst"):
            brief += f"\n\n{self.rt.NO_SEARCH_NOTE}"
        await self.emit(EventType.WORKER_START, worker=name, title=task.title)
        try:
            result = await self.rt.agent(name, task.specialist).run(brief)
            text = result.text or "(no findings)"
            status = result.stop
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the team
            text, status = f"FAILED: {type(exc).__name__}: {exc}", "error"
        await self.emit(EventType.WORKER_END, worker=name, status=status, chars=len(text))
        return text

    # ------------------------------------------------------------ 3+4. write & review
    async def write_and_review(self, plan: ResearchPlan, findings: dict[str, str]) -> tuple[str, list[Critique]]:
        titles = {t.id: f"{t.title} ({t.specialist})" for t in plan.subtasks}
        findings_md = "\n\n".join(f"### {titles.get(i, i)}\n{text}" for i, text in findings.items())
        criteria = "\n".join(f"- {c}" for c in plan.success_criteria)
        writer = self.rt.agent("writer", "writer", effort=self.settings.orchestrator_effort)

        result = await writer.run(
            f"# Objective\n{plan.objective}\n\n# Success criteria\n{criteria}\n\n"
            f"# Team findings\n{findings_md}\n\nWrite the final report."
        )
        report, critiques = result.text, []

        for round_ in range(1, self.settings.max_critique_rounds + 1):
            critique = await self._critique(plan, findings_md, report)
            critiques.append(critique)
            await self.emit(EventType.CRITIQUE, round=round_, **critique.model_dump())
            if critique.passes:
                break
            feedback = (
                f"Reviewer score: {critique.score}/10.\nIssues:\n"
                + "\n".join(f"- {i}" for i in critique.issues)
                + f"\n\nRevision instructions: {critique.revision_instructions}\n\n"
                "Output the complete revised report."
            )
            # Same conversation, appended - the writer sees its own draft and the feedback.
            result = await writer.run(feedback, history=result.messages)
            report = result.text
        return report, critiques

    async def _critique(self, plan: ResearchPlan, findings_md: str, report: str) -> Critique:
        async with self.rt.tracer.span("critique") as span:
            critique, usage = await self.rt.llm.parse(
                model=self.settings.model,
                system=prompts.CRITIC,
                messages=[{"role": "user", "content": (
                    f"<objective>{plan.objective}</objective>\n"
                    f"<success_criteria>\n" + "\n".join(plan.success_criteria) + "\n</success_criteria>\n"
                    f"<findings>\n{findings_md}\n</findings>\n<report>\n{report}\n</report>"
                )}],
                output_format=Critique,
                effort=self.settings.critic_effort,
            )
            self.rt.costs.record(self.settings.model, usage)
            span.set(score=critique.score, passes=critique.passes)
        return critique
