"""Evaluation harness.

CONCEPT: you can't improve what you don't measure. Prompt tweaks, model
upgrades and new tools all change behaviour in ways spot-checking misses, so
every change should run against a fixed suite. Three layers of checks:

1. Retrieval evals (deterministic, free, no API key): does the right document
   appear in the top-k? Reported as recall@k and MRR (mean reciprocal rank).
2. Trajectory checks (deterministic): did the agent call the tools it should
   have (e.g. search_knowledge_base for a private-doc question) and avoid the
   ones it shouldn't? Did the answer contain required facts / omit forbidden
   ones? Did it stay under its cost budget?
3. LLM-as-judge (model-graded): a rubric scored by Claude with a structured
   `JudgeVerdict`. Great for qualities code can't check (faithfulness, tone),
   but always pair it with deterministic checks - judges have biases.

Each run writes `.atlas/evals/<timestamp>.json` so results can be diffed over time.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from atlas.agents.schemas import JudgeVerdict
from atlas.config import Settings, get_settings
from atlas.llm.base import LLM
from atlas.rag.store import KnowledgeBase

console = Console()

JUDGE_PROMPT = """You are an impartial evaluator. Grade the RESPONSE to the QUESTION against the RUBRIC.
Score 1-5 (5 = fully satisfies the rubric). passed is true only for scores of 4 or 5.
Judge only what the rubric asks; do not reward length."""


@dataclass
class CaseResult:
    id: str
    kind: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_cases(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_fixture_kb(suite: dict[str, Any], base: Path) -> KnowledgeBase:
    kb = KnowledgeBase(":memory:")
    for rel in suite.get("fixtures", []):
        kb.add_path((base / rel).resolve())
    return kb


# ----------------------------------------------------------------- retrieval
def eval_retrieval(case: dict[str, Any], kb: KnowledgeBase) -> CaseResult:
    k = case.get("k", 5)
    hits = kb.search(case["query"], k=k, mode=case.get("mode", "hybrid"))
    expected = case["expected_source"]
    rank = next((i + 1 for i, h in enumerate(hits) if expected in h.source), None)
    must = case.get("must_contain_text")
    text_ok = must is None or any(must.lower() in h.text.lower() for h in hits)
    checks = {f"source_in_top_{k}": rank is not None, "passage_contains_fact": text_ok}
    return CaseResult(case["id"], "retrieval", all(checks.values()), checks,
                      {"rank": rank, "reciprocal_rank": (1 / rank) if rank else 0.0})


# ----------------------------------------------------------------- agent
async def eval_agent(case: dict[str, Any], kb: KnowledgeBase, settings: Settings,
                     llm: LLM, use_judge: bool) -> CaseResult:
    from atlas.agents.assistant import build_assistant
    from atlas.core.approvals import AutoDeny
    from atlas.runtime import Runtime

    rt = Runtime(settings, llm, kb, approvals=AutoDeny())
    start = time.perf_counter()
    result = await build_assistant(rt).run(case["input"])
    elapsed = time.perf_counter() - start
    text = result.text.lower()

    checks: dict[str, bool] = {"completed": result.stop == "completed"}
    for tool in case.get("expect_tools", []):
        checks[f"called:{tool}"] = tool in result.tool_calls
    for tool in case.get("forbid_tools", []):
        checks[f"avoided:{tool}"] = tool not in result.tool_calls
    for needle in case.get("must_contain", []):
        checks[f"contains:{needle}"] = needle.lower() in text
    for needle in case.get("must_not_contain", []):
        checks[f"omits:{needle}"] = needle.lower() not in text
    if "max_cost_usd" in case:
        checks["under_budget"] = rt.costs.total_usd <= case["max_cost_usd"]
    if case.get("expect_citations"):
        if llm.capabilities.native_citations:
            checks["has_citations"] = bool(result.citations)
        else:  # no structured citations on this provider: the answer must name a source
            checks["has_citations"] = any(Path(s).stem.lower() in text for s in kb.sources())

    notes: list[str] = []
    if use_judge and case.get("rubric"):
        verdict, usage = await llm.parse(
            model=settings.model, system=JUDGE_PROMPT, output_format=JudgeVerdict,
            effort=settings.critic_effort,
            messages=[{"role": "user", "content": (
                f"<question>{case['input']}</question>\n<rubric>{case['rubric']}</rubric>\n"
                f"<response>{result.text}</response>")}],
        )
        rt.costs.record(settings.model, usage)
        checks["judge"] = verdict.passed
        notes.append(f"judge {verdict.score}/5: {verdict.reasoning[:200]}")

    return CaseResult(case["id"], "agent", all(checks.values()), checks, {
        "turns": result.turns, "tools": result.tool_calls, "seconds": round(elapsed, 1),
        "cost_usd": round(rt.costs.total_usd, 4), "cache_hit_ratio": round(rt.costs.cache_hit_ratio, 3),
    }, notes)


# ----------------------------------------------------------------- driver
async def run_suite(cases_path: Path, *, llm: LLM | None = None, settings: Settings | None = None,
                    use_judge: bool = True, include_agent: bool = True) -> list[CaseResult]:
    settings = settings or get_settings()
    suite = load_cases(cases_path)
    kb = build_fixture_kb(suite, cases_path.parent)
    results = [eval_retrieval(c, kb) for c in suite.get("retrieval", [])]
    if include_agent and suite.get("agent"):
        if llm is None:
            from atlas.llm.factory import create_llm

            llm = create_llm(settings)
        for case in suite["agent"]:
            try:
                results.append(await eval_agent(case, kb, settings, llm, use_judge))
            except Exception as exc:  # noqa: BLE001 - one broken case must not hide the rest
                results.append(CaseResult(case["id"], "agent", False, notes=[f"{type(exc).__name__}: {exc}"]))
    return results


def report(results: list[CaseResult], out_dir: Path) -> int:
    table = Table(title="Atlas eval scorecard")
    for col in ("case", "kind", "result", "failed checks", "metrics"):
        table.add_column(col)
    for r in results:
        failed = [k for k, ok in r.checks.items() if not ok]
        table.add_row(r.id, r.kind, "[green]PASS[/]" if r.passed else "[red]FAIL[/]",
                      ", ".join(failed) or "-", json.dumps(r.metrics)[:80])
    console.print(table)

    retrieval = [r for r in results if r.kind == "retrieval"]
    if retrieval:
        recall = sum(1 for r in retrieval if r.metrics["rank"]) / len(retrieval)
        mrr = sum(r.metrics["reciprocal_rank"] for r in retrieval) / len(retrieval)
        console.print(f"Retrieval: recall@k = {recall:.2f}   MRR = {mrr:.3f}")
    passed = sum(r.passed for r in results)
    console.print(f"[bold]{passed}/{len(results)} cases passed[/]")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]Saved {path}[/]")
    return 0 if passed == len(results) else 1


async def main(cases_path: Path, use_judge: bool = True) -> int:
    settings = get_settings()
    has_credentials = settings.has_credentials()
    if has_credentials:
        console.print(f"Provider: [bold]{settings.provider}[/] · model [bold]{settings.model}[/]")
    else:
        console.print(f"[yellow]No credentials for provider '{settings.provider}' - running retrieval evals only.[/]")
    results = await run_suite(cases_path, settings=settings, use_judge=use_judge,
                              include_agent=has_credentials)
    return report(results, settings.data_dir / "evals")
