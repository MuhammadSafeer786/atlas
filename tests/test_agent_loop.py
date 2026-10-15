import asyncio

from atlas.core.agent import Agent, AgentConfig
from atlas.core.approvals import AutoApprove, AutoDeny
from atlas.core.events import AgentEvent
from atlas.llm.usage import CostTracker
from atlas.rag.store import KnowledgeBase
from atlas.tools.builtin import builtin_tools
from atlas.tools.knowledge import SearchKnowledgeBase
from atlas.tools.registry import ToolRegistry
from tests.conftest import FakeLLM, text, tool_use, turn


def make_agent(llm, settings, tools=None, approvals=None, events=None, max_turns=8, costs=None):
    async def sink(ev: AgentEvent):
        if events is not None:
            events.append(ev)
    cfg = AgentConfig("tester", "You are a test agent.", "claude-opus-5-5", max_turns=max_turns)
    return Agent(cfg, llm, tools or ToolRegistry(builtin_tools()), settings,
                 approvals=approvals or AutoDeny(), sink=sink, costs=costs)


async def test_simple_answer_without_tools(settings):
    llm = FakeLLM([turn(text("Paris."))])
    result = await make_agent(llm, settings).run("Capital of France?")
    assert result.text == "Paris."
    assert result.stop == "completed" and result.turns == 1


async def test_tool_round_trip_and_history_shape(settings):
    llm = FakeLLM([
        turn(text("Let me compute."), tool_use("t1", "calculator", {"expression": "2**10"})),
        turn(text("It is 1024.")),
    ])
    result = await make_agent(llm, settings).run("What is 2^10?")
    assert result.text == "It is 1024."
    assert result.tool_calls == ["calculator"]
    second_request = llm.requests[1].messages
    # user -> assistant(tool_use) -> user(tool_result)
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
    tool_result = second_request[2]["content"][0]
    assert tool_result == {"type": "tool_result", "tool_use_id": "t1", "content": "1024"}


async def test_parallel_tool_results_in_single_message(settings):
    llm = FakeLLM([
        turn(tool_use("a", "calculator", {"expression": "1+1"}),
             tool_use("b", "current_time"),
             tool_use("c", "calculator", {"expression": "3*3"})),
        turn(text("done")),
    ])
    await make_agent(llm, settings).run("go")
    results = llm.requests[1].messages[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b", "c"]  # order preserved, one message


async def test_invalid_and_unknown_tools_are_errors_not_crashes(settings):
    llm = FakeLLM([
        turn(tool_use("x", "calculator", {"wrong_field": 1}), tool_use("y", "does_not_exist")),
        turn(text("recovered")),
    ])
    result = await make_agent(llm, settings).run("go")
    results = llm.requests[1].messages[-1]["content"]
    assert all(r["is_error"] for r in results)
    assert "Invalid input" in results[0]["content"]
    assert "Unknown tool" in results[1]["content"]
    assert result.text == "recovered"


async def test_approval_denied_blocks_side_effect(settings):
    llm = FakeLLM([
        turn(tool_use("w", "write_file", {"path": "a.txt", "content": "hi"})),
        turn(text("The write was declined.")),
    ])
    events: list[AgentEvent] = []
    await make_agent(llm, settings, approvals=AutoDeny(), events=events).run("save")
    assert not (settings.workspace_dir / "a.txt").exists()
    result_block = llm.requests[1].messages[-1]["content"][0]
    assert result_block["is_error"] and "declined" in result_block["content"]
    assert any(e.type == "approval_request" for e in events)


async def test_approval_granted_runs_side_effect(settings):
    llm = FakeLLM([
        turn(tool_use("w", "write_file", {"path": "out/a.txt", "content": "hi"})),
        turn(text("Saved.")),
    ])
    await make_agent(llm, settings, approvals=AutoApprove()).run("save")
    assert (settings.workspace_dir / "out" / "a.txt").read_text() == "hi"


async def test_sandbox_escape_rejected(settings):
    llm = FakeLLM([
        turn(tool_use("r", "read_file", {"path": "../../../etc/passwd"})),
        turn(text("blocked")),
    ])
    await make_agent(llm, settings).run("read")
    block = llm.requests[1].messages[-1]["content"][0]
    assert block["is_error"] and "escapes sandbox" in block["content"]


async def test_refusal_stops_and_answers_dangling_tool_use(settings):
    llm = FakeLLM([turn(tool_use("t", "calculator", {"expression": "1"}), stop="refusal",
                        details={"type": "refusal", "category": "cyber"})])
    events: list[AgentEvent] = []
    result = await make_agent(llm, settings, events=events).run("bad request")
    assert result.stop == "refusal"
    assert result.messages[-1]["role"] == "user"  # history stays valid for the next turn
    assert result.messages[-1]["content"][0]["tool_use_id"] == "t"
    assert any(e.type == "refusal" for e in events)


async def test_pause_turn_is_resumed(settings):
    llm = FakeLLM([
        turn({"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "x"}},
             stop="pause_turn"),
        turn(text("final")),
    ])
    result = await make_agent(llm, settings).run("search")
    assert result.text == "final" and result.turns == 2
    assert llm.requests[1].messages[-1]["role"] == "assistant"  # paused turn re-sent as-is


async def test_max_tokens_truncated_tool_input_not_executed(settings):
    llm = FakeLLM([
        turn(tool_use("w", "write_file", {"path": "a.txt", "content": "trunc"}), stop="max_tokens"),
        turn(text("ok")),
    ])
    await make_agent(llm, settings, approvals=AutoApprove()).run("write")
    assert not (settings.workspace_dir / "a.txt").exists()
    assert "truncated" in llm.requests[1].messages[-1]["content"][0]["content"]


async def test_max_turns_with_wind_down_system_message(settings):
    def looping(req):
        return turn(tool_use(f"t{len(req.messages)}", "current_time"))

    llm = FakeLLM(handler=looping)
    result = await make_agent(llm, settings, max_turns=3).run("loop forever")
    assert result.stop == "max_turns" and result.turns == 3
    system_msgs = [m for m in llm.requests[-1].messages if m["role"] == "system"]
    assert len(system_msgs) == 1 and "step limit" in system_msgs[0]["content"]


async def test_cost_budget_stops_run(settings):
    settings.max_cost_usd = 0.0001
    llm = FakeLLM(handler=lambda req: turn(tool_use(f"t{len(req.messages)}", "current_time")))
    result = await make_agent(llm, settings, costs=CostTracker()).run("go")
    assert result.stop == "budget" and result.turns == 1


async def test_history_is_append_only_across_turns(settings):
    llm = FakeLLM([
        turn(tool_use("a", "calculator", {"expression": "1+1"})),
        turn(tool_use("b", "calculator", {"expression": "2+2"})),
        turn(text("done")),
    ])
    await make_agent(llm, settings).run("go")
    first, second, third = (r.messages for r in llm.requests)
    assert second[: len(first)] == first and third[: len(second)] == second


async def test_untrusted_document_content_is_fenced(settings, knowledge_dir):
    kb = KnowledgeBase(":memory:")
    kb.add_path(knowledge_dir / "vendor-note.md")
    events: list[AgentEvent] = []
    llm = FakeLLM([
        turn(tool_use("s", "search_knowledge_base", {"query": "DataPipe connector"})),
        turn(text("It syncs every 15 minutes.")),
    ])
    agent = make_agent(llm, settings, tools=ToolRegistry([SearchKnowledgeBase(kb)]), events=events)
    await agent.run("Summarize the vendor note")
    content = llm.requests[1].messages[-1]["content"][0]["content"]
    assert content[0]["type"] == "text" and "WARNING" in content[0]["text"]
    assert content[1]["type"] == "search_result" and content[1]["citations"] == {"enabled": True}
    assert any(e.type == "guardrail" and e.data["stage"] == "tool_output" for e in events)


async def test_input_guard_redacts_secrets_and_pii(settings):
    llm = FakeLLM([turn(text("ok"))])
    await make_agent(llm, settings).run("my key is sk-ant-abcdefghijklmnopqrstuvwxyz123 and mail a@b.com")
    sent = llm.requests[0].messages[0]["content"]
    assert "sk-ant-" not in sent and "a@b.com" not in sent
    assert "[REDACTED:anthropic_key]" in sent and "[EMAIL]" in sent


async def test_tool_timeout_is_reported(settings):
    from pydantic import BaseModel

    from atlas.tools.base import Tool, ToolOutput

    class SlowInput(BaseModel):
        pass

    class Slow(Tool[SlowInput]):
        name = "slow"
        description = "sleeps"
        Input = SlowInput

        async def run(self, args, ctx):
            await asyncio.sleep(10)
            return ToolOutput("never")

    settings.tool_timeout_s = 0.05
    llm = FakeLLM([turn(tool_use("s", "slow")), turn(text("timed out"))])
    await make_agent(llm, settings, tools=ToolRegistry([Slow()])).run("go")
    assert "timed out" in llm.requests[1].messages[-1]["content"][0]["content"]


async def test_empty_response_is_retried_once(settings):
    llm = FakeLLM([
        turn({"type": "thinking", "thinking": "I should call the calculator..."}),  # reasons, then says nothing
        turn(tool_use("c", "calculator", {"expression": "1234*5678"})),
        turn(text("7006652")),
    ])
    events: list[AgentEvent] = []
    result = await make_agent(llm, settings, events=events).run("1234*5678?")
    assert result.text == "7006652" and result.tool_calls == ["calculator"]
    second = llm.requests[1].messages
    assert [m["role"] for m in second] == ["user", "system"]  # empty turn not recorded, nudge appended
    assert any(e.type == "warning" and "empty" in e.data["message"] for e in events)
