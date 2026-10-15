import pytest

from atlas.agents import prompts
from atlas.agents.assistant import Session, build_assistant
from atlas.agents.orchestrator import Orchestrator, PlanError, execution_levels
from atlas.agents.schemas import Critique, ResearchPlan, Subtask
from atlas.core.events import AgentEvent
from atlas.rag.store import KnowledgeBase
from atlas.runtime import Runtime
from tests.conftest import FakeLLM, text, tool_use, turn

PLAN = ResearchPlan(
    objective="Compare Nimbus plans",
    subtasks=[
        Subtask(id="t1", title="Collect pricing", specialist="librarian", instructions="Find prices"),
        Subtask(id="t2", title="Collect SLAs", specialist="researcher", instructions="Find SLAs"),
        Subtask(id="t3", title="Cost model", specialist="analyst", instructions="Compute", depends_on=["t1", "t2"]),
    ],
    success_criteria=["Covers all plans", "Cites sources"],
)


def test_execution_levels_parallelism_and_cycles():
    levels = execution_levels(PLAN)
    assert [[t.id for t in lvl] for lvl in levels] == [["t1", "t2"], ["t3"]]
    cyclic = PLAN.model_copy(deep=True)
    cyclic.subtasks[0].depends_on = ["t3"]
    with pytest.raises(PlanError):
        execution_levels(cyclic)


async def test_full_pipeline_with_revision(settings):
    role_by_prompt = {prompts.RESEARCHER: "researcher", prompts.ANALYST: "analyst",
                      prompts.LIBRARIAN: "librarian", prompts.WRITER: "writer"}

    def handler(req):
        role = role_by_prompt[req.system]
        if role == "writer":
            drafts = sum(1 for m in req.messages if m["role"] == "user")
            return turn(text(f"REPORT v{drafts}"))
        return turn(text(f"findings from {role}"))

    llm = FakeLLM(handler=handler, parses=[
        PLAN,
        Critique(score=5, passes=False, strengths=[], issues=["missing SLA table"], revision_instructions="add it"),
        Critique(score=9, passes=True, strengths=["complete"], issues=[], revision_instructions=""),
    ])
    events: list[AgentEvent] = []

    async def sink(ev):
        events.append(ev)

    rt = Runtime(settings, llm, KnowledgeBase(":memory:"), sink=sink)
    outcome = await Orchestrator(rt).run("Compare Nimbus plans")

    assert outcome.report == "REPORT v2"  # revised once after failing critique
    assert [c.passes for c in outcome.critiques] == [False, True]
    assert outcome.findings == {"t1": "findings from librarian", "t2": "findings from researcher",
                                "t3": "findings from analyst"}
    # dependency outputs are passed to the dependent worker
    analyst_req = next(r for r in llm.requests if r.system == prompts.ANALYST)
    brief = analyst_req.messages[0]["content"]
    assert "findings from librarian" in brief and "findings from researcher" in brief
    # least privilege: the librarian only gets the knowledge base
    librarian_req = next(r for r in llm.requests if r.system == prompts.LIBRARIAN)
    assert [t["name"] for t in librarian_req.tools] == ["search_knowledge_base"]
    # revision happened in the writer's existing conversation (append-only)
    writer_reqs = [r for r in llm.requests if r.system == prompts.WRITER]
    assert writer_reqs[1].messages[: len(writer_reqs[0].messages)] == writer_reqs[0].messages
    types = [e.type for e in events]
    assert types.count("worker_start") == 3 and "plan" in types and types.count("critique") == 2
    assert outcome.cost["calls"] == 3 + 2 + 3  # parses + writer turns + workers


async def test_worker_failure_degrades_gracefully(settings):
    def handler(req):
        if req.system == prompts.RESEARCHER:
            raise RuntimeError("search backend down")
        return turn(text("ok"))

    llm = FakeLLM(handler=handler, parses=[
        PLAN, Critique(score=8, passes=True, strengths=[], issues=[], revision_instructions=""),
    ])
    outcome = await Orchestrator(Runtime(settings, llm, KnowledgeBase(":memory:"))).run("x")
    assert outcome.findings["t2"].startswith("FAILED: RuntimeError")
    assert outcome.report == "ok"


async def test_assistant_session_persists_and_delegates(settings):
    llm = FakeLLM(
        turns=[
            turn(tool_use("d1", "delegate_research", {"objective": "Compare Nimbus plans"})),
            *[turn(text("findings")) for _ in range(3)],  # three workers
            turn(text("TEAM REPORT")),  # writer
            turn(text("Here is the team's report.")),  # assistant wraps up
            turn(text("You asked about Nimbus plans.")),  # second chat message
        ],
        parses=[PLAN, Critique(score=9, passes=True, strengths=[], issues=[], revision_instructions="")],
    )
    rt = Runtime(settings, llm, KnowledgeBase(":memory:"))
    session = Session()
    first = await session.send(build_assistant(rt), "Research Nimbus plans in depth")
    assert first.tool_calls == ["delegate_research"]
    tool_result = next(m for m in first.messages if m["role"] == "user" and isinstance(m["content"], list))
    assert "TEAM REPORT" in tool_result["content"][0]["content"]

    session.save(settings.session_dir)
    restored = Session.load(settings.session_dir, session.id)
    second = await restored.send(build_assistant(rt), "What did I ask?")
    assert second.text == "You asked about Nimbus plans."
    assert llm.requests[-1].messages[0]["content"] == "Research Nimbus plans in depth"


async def test_team_is_told_when_web_search_is_unavailable(settings):
    from atlas.llm.base import Capabilities

    llm = FakeLLM(handler=lambda req: turn(text("ok")), capabilities=Capabilities("gemini"), parses=[
        PLAN, Critique(score=9, passes=True, strengths=[], issues=[], revision_instructions="")])
    events: list[AgentEvent] = []

    async def sink(ev):
        events.append(ev)

    rt = Runtime(settings, llm, KnowledgeBase(":memory:"), sink=sink)
    assert not rt.has_web_search
    await Orchestrator(rt).run("Research something")
    assert "No web search tool is available" in llm.parse_calls[0]["messages"][0]["content"]
    researcher = next(r for r in llm.requests if r.system == prompts.RESEARCHER)
    assert "Do not guess URLs" in researcher.messages[0]["content"]
    assert any(e.type == "warning" and "TAVILY_API_KEY" in e.data["message"] for e in events)

    with_key = Runtime(settings.model_copy(update={"tavily_api_key": "tvly"}), llm, KnowledgeBase(":memory:"))
    assert with_key.has_web_search
