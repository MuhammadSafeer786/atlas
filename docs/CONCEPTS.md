# Study guide: agentic AI through the Atlas codebase

This guide walks through every concept in Atlas in the order that builds understanding fastest. Each module gives you:

- **Idea:** the concept in plain words.
- **Read:** the exact files, in order.
- **Notice:** the details that separate a demo from a production system.
- **Try:** experiments that make it stick. Most need no API key.
- **Check yourself:** questions you should be able to answer, in an interview or to a client.

Plan for about two focused weeks. Run `pytest -q` after every experiment so you know you haven't broken anything.

---

## Module 0: Setup and orientation

```bash
python -m venv .venv && .venv\Scripts\activate     # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # 67 tests should pass with no network
atlas ingest examples/knowledge
atlas search "service credit" -k 3
```

Read [README.md](../README.md), then [ARCHITECTURE.md](ARCHITECTURE.md). Then skim the *docstrings at the top of every module*. Each one names the concept that file teaches.

---

## Module 1: What an "agent" actually is

**Idea.** An LLM call is a function: text in, text out. An *agent* is an LLM in a **loop with tools**. The model decides which tool to call, your code runs it, the result goes back in, and the loop repeats until the model answers. This is the ReAct pattern (reason → act → observe). Claude implements it natively with `tool_use` and `tool_result` content blocks.

**Read.** [atlas/core/agent.py](../atlas/core/agent.py): start with the module docstring, then `Agent.run`.

**Notice.**
- The model never executes anything. It *requests* a tool, and your harness decides. That's where safety lives.
- Every `tool_use` must get a matching `tool_result` (same `tool_use_id`) in the **next user message**, or the API rejects the conversation.
- All results from one step go back in **one** user message. Splitting them teaches the model to stop making parallel calls.
- The loop has several exits: `end_turn`, `max_tokens`, `refusal`, max turns, budget. Each needs deliberate handling.

**Try.**
1. Read `test_tool_round_trip_and_history_shape` in [tests/test_agent_loop.py](../tests/test_agent_loop.py) and print `llm.requests[1].messages` to see the exact wire format.
2. Write a test where the model calls `calculator` twice in a row (two turns), then answers.

**Check yourself.**
- Why does the history alternate user → assistant → user, and why are tool results sent with the *user* role?
- What's the difference between a *workflow* and an *agent*? (Anthropic's "Building effective agents" essay is the canonical reference.)
- When would you *not* build an agent? (Hint: when one call or a fixed pipeline is enough.)

---

## Module 2: Talking to Claude properly (the Claude adapter)

**Idea.** One module owns every API call, so model features are configured in one place and everything else stays testable.

**Read.** [atlas/llm/anthropic_llm.py](../atlas/llm/anthropic_llm.py), [atlas/llm/usage.py](../atlas/llm/usage.py), [tests/test_llm_gateway_http.py](../tests/test_llm_gateway_http.py).

**Notice.**
- **Streaming** every turn: no HTTP timeouts on long outputs, and the UI can show tokens live.
- **Adaptive thinking** (`thinking: {"type": "adaptive"}`): Claude decides how much to reason. `display: "summarized"` returns a readable summary of that reasoning. On Claude Opus 5.5, thinking can't be turned off; **effort** (`low`…`max`) is the dial, and its default is `medium`, so Atlas always sets it explicitly.
- **Prompt caching:** the request prefix renders as `tools → system → messages`. A `cache_control` breakpoint on the system prompt caches tools and system together. Top-level `cache_control` auto-caches the growing conversation. Cache reads cost about 10% of normal input. Any byte change in the prefix (a timestamp in the system prompt, a reordered tool list) silently breaks the cache. That's why `current_time` is a *tool*, not part of the prompt.
- **Refusal fallbacks:** `fallbacks="default"` (beta) re-runs a safety-declined request on a recommended fallback model inside the same call. You still handle `stop_reason == "refusal"`.
- **Compaction** (beta): when context gets large, the API summarises older turns server-side. You round-trip the returned blocks unchanged.
- **Eager input streaming:** tool inputs stream as they are generated, so validation moves to the client. See `Tool.execute` in [tools/base.py](../atlas/tools/base.py).

**Try.**
1. In `test_streaming_turn_through_real_sdk`, print `captured["body"]` to see the exact JSON Atlas sends.
2. Compute by hand what `Usage(1_000_000, 100_000, 0, 2_000_000).cost_usd("claude-opus-5-5")` should be, then check it.
3. With an API key, run `atlas chat` twice with the same question and watch the cache-hit ratio in the footer rise.

**Check yourself.**
- Why does putting `datetime.now()` in a system prompt hurt cost?
- What do adaptive thinking and effort each control?
- Why should a refusal be checked *before* reading `content`?

---

## Module 2b: Being provider-agnostic (Claude, OpenAI, Gemini, local models)

**Idea.** Separate *what the agent does* from *which vendor runs the model*. The core depends on one small interface (`LLM.turn`, `LLM.parse`), and each provider is an **adapter** that translates between Atlas's canonical message format and the vendor's wire format. This is the ports-and-adapters pattern, also called hexagonal architecture.

**Read.** [llm/base.py](../atlas/llm/base.py) → [llm/factory.py](../atlas/llm/factory.py) → [llm/openai_llm.py](../atlas/llm/openai_llm.py) → [llm/gemini_llm.py](../atlas/llm/gemini_llm.py) → `_resolve_provider` in [config.py](../atlas/config.py) → `tools_for` in [runtime.py](../atlas/runtime.py) → [tests/test_providers.py](../tests/test_providers.py).

**Notice.**
- **One canonical format.** History is stored as Claude-style content blocks, the most expressive of the three formats, so nothing is lost. The same `tool_use` → `tool_result` pair becomes:
  - OpenAI: `assistant.tool_calls[]` (arguments as a JSON *string*) and a `{"role": "tool", "tool_call_id"}` message per result.
  - Gemini: `Part(function_call)` and `Part(function_response)`, matched by *name*, so the adapter looks the name up from the id.
- **Opaque provider state.** Gemini returns `thought_signature`s that must be echoed back or tool calling breaks. The adapter stores them on the canonical block under a private `_gemini_sig` key, and other adapters strip `_` keys. Reasoning blocks from one provider are never sent to another.
- **Capabilities, not `if provider == ...`.** Adapters declare `Capabilities`. The runtime gives Claude the hosted web tools and code sandbox, and gives everyone else the portable `fetch_url` (with an SSRF guard) plus optional Tavily search. The memory tool renders Anthropic's built-in spec on Claude and an equivalent explicit JSON schema elsewhere.
- **Capability negotiation.** Local servers differ in what they accept. The OpenAI-compatible adapter drops `stream_options`, `reasoning_effort` or `response_format` when a server rejects them, retries, and remembers.
- **Structured output without guarantees.** Only some providers enforce schemas, so `parse_with_repair` asks for JSON, validates with Pydantic, and sends the exact validation error back for the model to fix: a *validate-and-repair* loop.
- **Robust tool-call parsing.** Streaming tool calls arrive as fragments keyed by index. Invalid JSON arguments become an `is_error` result the model can correct, not a crash.

**Try.**
1. `ATLAS_PROVIDER=ollama`, run `ollama pull qwen3:8b`, then `atlas chat`. Ask something that needs `calculator` and `search_knowledge_base`.
2. Run the same eval suite on two providers (`atlas eval`) and compare the scorecards. That's how you'd choose a model for a client.
3. Write an adapter for another API (for example a provider with its own SDK), using `GeminiLLM` as the template, and add a mocked-HTTP test like those in `test_providers.py`.

**Check yourself.**
- Why store history in the *richest* format rather than the lowest common denominator?
- What breaks if you replay a Gemini function call without its thought signature?
- Why are capability flags better than checking the provider name throughout the code?

---

## Module 3: Tools done right

**Idea.** Tools are your API to the model. Good tools are typed, validated, narrowly scoped and labelled with metadata the harness can act on.

**Read.** [tools/base.py](../atlas/tools/base.py) → [tools/builtin.py](../atlas/tools/builtin.py) → [tools/registry.py](../atlas/tools/registry.py) → [tools/server.py](../atlas/tools/server.py).

**Notice.**
- A Pydantic model becomes a **strict** JSON schema (`strict: true`, `additionalProperties: false`), so the API guarantees schema-valid input. Atlas re-validates anyway because of eager streaming.
- Errors are **data**: an invalid input returns `is_error: true` with a precise message, and the model corrects itself.
- Metadata drives the harness: `parallel_safe`, `requires_approval`, `untrusted_output`.
- The `calculator` parses an AST whitelist. Never `eval()` model output.
- `safe_path()` treats a model-supplied path as hostile input (`../../etc/passwd`).
- Three kinds of tools: user-defined (you write schema and code), Anthropic-defined client tools (memory: Anthropic defines the schema, you run it), and server tools (web search, web fetch, code execution: Anthropic runs them).
- Web search/fetch and code execution go to *different* roles, because the new web tools already use code execution internally.

**Try.**
1. Add a `unit_converter` tool with an `Enum` field, register it in `builtin_tools()`, and write a test.
2. Print `strict_schema(SearchKBInput)` and compare it with `SearchKBInput.model_json_schema()`.

**Check yourself.**
- Why return an error *to the model* instead of raising?
- When should an action be a dedicated tool rather than a generic `bash` tool? (Gating, auditing, rendering, parallelism.)
- Why must tool order be deterministic?

---

## Module 4: Parallelism, budgets and robustness in the loop

**Read.** `_run_tools`, `_run_one`, `_execute` and the stop-reason branches in [core/agent.py](../atlas/core/agent.py).

**Notice.**
- Read-only tools run concurrently with `asyncio.gather`. Side-effecting and approval-gated tools share a lock and run one at a time.
- Per-tool timeout (`asyncio.wait_for`), and every exception is caught and turned into an error result.
- `pause_turn`: a server tool is mid-flight, so the loop re-sends to let it continue.
- `max_tokens` with a `tool_use` present: the input may be truncated, so never run it, but still answer it.
- `refusal` with a dangling `tool_use`: answer it so the *session* stays valid.
- Near the turn limit, a **mid-conversation system message** tells the model to wrap up. It's appended, never edited into the system prompt, so the cache survives.
- A dollar budget is checked before every model call and covers sub-agents too (one shared `CostTracker`).

**Try.** Each behaviour has a test. Break one deliberately (for example, remove the `max_tokens` branch) and watch the right test fail.

**Check yourself.** What could go wrong if you executed a tool input from a `max_tokens` response?

---

## Module 5: Retrieval-augmented generation (RAG)

**Idea.** Ground answers in your own documents by retrieving relevant passages at question time.

**Read.** [rag/chunking.py](../atlas/rag/chunking.py) → [rag/bm25.py](../atlas/rag/bm25.py) → [rag/embeddings.py](../atlas/rag/embeddings.py) → [rag/store.py](../atlas/rag/store.py) → [tools/knowledge.py](../atlas/tools/knowledge.py).

**Notice.**
- **Chunking** is heading-aware and overlapping, and each chunk carries its heading path ("contextual chunking").
- **BM25** (sparse) wins on exact terms like error codes and IDs. **Embeddings** (dense) win on meaning. **Hybrid** combines them with **Reciprocal Rank Fusion**, which needs ranks, not comparable scores.
- **Agentic RAG:** search is a *tool*, so the model decides when and what to search and can refine its query.
- The tool returns `search_result` blocks, which enable **native citations** that point at the exact passage.
- Changing the embedding model triggers re-embedding, because vectors from different models aren't comparable.

**Try.**
1. `atlas search "NIMBUS_DIM_MISMATCH" --mode vector` vs `--mode keyword` vs `--mode hybrid`. Explain the differences.
2. `pip install -e ".[embeddings]"`, re-ingest, and compare vector-mode results with the hashing embedder.
3. Add three retrieval cases to [evals/cases.json](../evals/cases.json) and run `atlas eval --no-judge`. Find a query that fails and fix it (chunk size? mode?).
4. Stretch: add an LLM reranker stage (retrieve 20, have Claude pick the best 5 with a structured output).

**Check yourself.** Why does hybrid search beat either retriever alone? What do recall@k and MRR measure?

---

## Module 6: Memory

**Idea.** The context window is working memory and vanishes when the run ends. Agents need **short-term memory** (this conversation) and **long-term memory** (facts that outlive conversations).

**Read.** [tools/memory.py](../atlas/tools/memory.py); `Session` in [agents/assistant.py](../atlas/agents/assistant.py).

**Notice.**
- Short-term: the transcript is persisted as JSON per session and resumed with `atlas chat --session <id>`.
- Long-term: the `memory_20250818` tool. Claude is trained to check `/memories` first and record durable learnings. You implement the six commands; paths are confined; secrets are redacted before writing.
- Inside a session, context growth is handled by compaction, not by deleting history.

**Try.** With a key: tell `atlas chat` a preference, exit, start a fresh session, ask something, and see whether the preference is used. Inspect `.atlas/memories/`.

**Check yourself.** Why not just stuff every past conversation into the prompt? What belongs in memory and what doesn't?

---

## Module 7: MCP, the Model Context Protocol

**Idea.** An open standard for exposing **tools, resources and prompts** to any AI application. Write a capability once, and every MCP client can use it.

**Read.** [mcp_server/server.py](../atlas/mcp_server/server.py) (Atlas as a server) and [tools/mcp_bridge.py](../atlas/tools/mcp_bridge.py) (Atlas as a client), then [tests/test_mcp_and_api.py](../tests/test_mcp_and_api.py).

**Notice.**
- Three primitives: tools (actions), resources (readable data, by URI), prompts (templates).
- Transports: stdio (local subprocess) and Streamable HTTP (remote).
- The bridge discovers tools at runtime and wraps them as normal Atlas tools, with approval, tracing and untrusted-output fencing included. It honours the server's `destructive_hint` and `read_only_hint` annotations.
- Tests connect client and server **in-process**, with no subprocess.

**Try.**
1. `claude mcp add atlas -- atlas mcp-server`, then ask Claude Code to search your knowledge base.
2. Add an MCP tool `summarize_source(source)` to the server and a test for it.
3. Create `atlas.mcp.json` with the filesystem server from `atlas.mcp.example.json` and watch its tools appear in `/health`.

**Check yourself.** How is MCP different from plain function calling? Why treat MCP output as untrusted?

---

## Module 8: Structured outputs and planning

**Idea.** Constrain the model's output to a schema so code can branch on it. Then use that to make the model *plan*.

**Read.** [agents/schemas.py](../atlas/agents/schemas.py), `Orchestrator.plan`, `execution_levels` in [agents/orchestrator.py](../atlas/agents/orchestrator.py).

**Notice.**
- `parse(output_format=ResearchPlan)` returns a validated Pydantic object. No regex, no "please reply in JSON".
- The plan is a **DAG**. Kahn's algorithm groups it into levels that can run in parallel and rejects cycles.

**Try.** Add `estimated_minutes: int` to `Subtask` and show it in the CLI plan table.

**Check yourself.** Plan-and-execute vs ReAct: when is each better?

---

## Module 9: Multi-agent orchestration

**Idea.** Split a big task across specialists with focused contexts and least-privilege tools, then combine their results.

**Read.** [agents/orchestrator.py](../atlas/agents/orchestrator.py), [runtime.py](../atlas/runtime.py), [agents/prompts.py](../atlas/agents/prompts.py), [tests/test_orchestrator.py](../tests/test_orchestrator.py).

**Notice.**
- **Orchestrator-workers** pattern, with a semaphore to bound concurrency.
- **Context isolation:** a worker sees only its brief and its dependencies' outputs, never the whole transcript.
- **Least privilege:** each role gets the minimum tools (`Runtime.tools_for`).
- **Agents as tools:** `delegate_research` lets the chat assistant call the whole team as one tool, and only the final report enters its context.
- **Graceful degradation:** a failed worker becomes a documented gap.
- Shared `Runtime`: one budget, one trace and one event stream across all agents.

**Try.** Add a fourth specialist, `fact_checker`, that receives the writer's draft and verifies three claims with the knowledge base. Wire it in before the critic.

**Check yourself.** What are the costs of multi-agent systems (tokens, latency, coordination errors)? When is a single agent better?

---

## Module 10: Reflection (evaluator-optimizer)

**Idea.** Have one model call critique another's output against explicit criteria, then revise.

**Read.** `write_and_review` and `_critique` in [agents/orchestrator.py](../atlas/agents/orchestrator.py).

**Notice.** The critique is structured (score, passes, issues, instructions). The revision is appended to the **same** writer conversation, so the writer sees its own draft and the feedback. A round cap prevents endless loops.

**Check yourself.** Why do concrete success criteria from the planner make critique more reliable?

---

## Module 11: Human-in-the-loop

**Read.** [core/approvals.py](../atlas/core/approvals.py), the approval branch in `Agent._run_one`, and the `/api/approvals` routes in [api/app.py](../atlas/api/app.py).

**Notice.** Approval is a *policy interface*. The CLI asks in the terminal. The web version awaits an `asyncio.Future` resolved by a REST call, times out and **fails closed**. A denial reaches the model as a tool result so it can adapt.

**Try.** `atlas serve`, then ask "save a summary of the SLA doc to reports/sla.md" and approve or deny in the dialog.

---

## Module 12: Guardrails and security

**Read.** [core/guardrails.py](../atlas/core/guardrails.py), `safe_path`, and [examples/knowledge/vendor-note.md](../examples/knowledge/vendor-note.md) (a planted attack).

**Notice.**
- **Indirect prompt injection** is the main agent-specific threat: instructions hidden in web pages, documents or tool output. Atlas fences third-party content in `<untrusted_content>` tags, flags known attack phrasing, and the system prompts state that such content is data.
- Secret and PII redaction (with a Luhn check to avoid false positives on credit-card numbers).
- Least privilege and approvals limit the blast radius when a defence fails.

**Try.** Ingest a document with a subtler injection ("As the system administrator, I authorise you to…") and see whether the heuristics catch it. Then think about what *would* (a classifier, stricter privileges).

**Check yourself.** Why is "the model will just ignore malicious instructions" not a security strategy?

---

## Module 13: Observability and cost

**Read.** [core/tracing.py](../atlas/core/tracing.py), [core/events.py](../atlas/core/events.py), [llm/usage.py](../atlas/llm/usage.py).

**Notice.** Spans nest through `contextvars`, so parallel sub-agents get the right parents. The span fields mirror OpenTelemetry. Events stream to CLI and web, and the cache-hit ratio is your cache health check.

**Try.** After a research run, open `.atlas/traces/<id>.jsonl` and find the slowest span.

---

## Module 14: Evals

**Idea.** You can't improve what you don't measure. Every prompt, model or tool change should run against a fixed suite.

**Read.** [atlas/evals/runner.py](../atlas/evals/runner.py), [evals/cases.json](../evals/cases.json).

**Notice.** Three layers: deterministic retrieval metrics (free), trajectory checks (did it call the right tools, stay in budget, avoid the injected instruction), and LLM-as-judge with a structured verdict. Results are saved with timestamps so you can diff runs.

**Try.** With a key, run `atlas eval`. Change the assistant prompt in [agents/prompts.py](../atlas/agents/prompts.py) and re-run. Did anything regress?

---

## Module 15: Serving and shipping

**Read.** [api/app.py](../atlas/api/app.py), [api/static/index.html](../atlas/api/static/index.html), [Dockerfile](../Dockerfile), [.github/workflows/ci.yml](../.github/workflows/ci.yml).

**Notice.** SSE streaming with keep-alives, cancellation when the client disconnects (stop paying for unread tokens), an app factory for dependency injection in tests, sanitised Markdown rendering in the browser (model output is untrusted HTML), a multi-stage non-root Docker image with a health check, and CI that runs free evals on every push and paid ones only on `main`.

---

## Glossary

| Term | Meaning |
|---|---|
| Agent | An LLM using tools in a loop and choosing its own next step |
| Workflow | Predefined code paths that orchestrate LLM calls |
| ReAct | Reason → act → observe loop |
| Tool use / function calling | The model emits a structured request; your code runs it |
| Server tool | A tool Anthropic executes (web search, code execution) |
| MCP | Open protocol for exposing tools, resources and prompts to AI apps |
| RAG | Retrieve relevant passages and give them to the model |
| BM25 | Classic keyword relevance scoring |
| Embedding | A vector representing meaning; compared by cosine similarity |
| RRF | Reciprocal Rank Fusion, which merges ranked lists |
| Structured output | Response constrained to a JSON schema |
| Prompt caching | Reusing a processed request prefix at reduced cost |
| Compaction | Server-side summarisation of old context |
| Adaptive thinking / effort | Model-controlled reasoning depth / the thoroughness-cost dial |
| Evaluator-optimizer | Generate, critique against criteria, revise |
| Prompt injection | Malicious instructions smuggled in via content the model reads |
| HITL | Human-in-the-loop approval of agent actions |
| LLM-as-judge | Using a model to grade outputs against a rubric |

## Further reading

- Anthropic, "Building effective agents" (workflows vs agents, the core patterns used here)
- Claude API docs: tool use, prompt caching, extended/adaptive thinking, citations, the memory tool
- modelcontextprotocol.io: the MCP specification and SDKs
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
- Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods"
