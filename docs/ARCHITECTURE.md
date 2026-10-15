# Atlas architecture

## Layers

```
┌───────────────────────────────────────────────────────────────────┐
│ Interfaces     CLI (Typer + Rich) · FastAPI + SSE · Web UI · MCP  │
├───────────────────────────────────────────────────────────────────┤
│ Agents         Assistant · Orchestrator · Planner · Specialists · │
│                Writer · Critic          (atlas/agents)            │
├───────────────────────────────────────────────────────────────────┤
│ Core           Agent loop · Events · Approvals · Guardrails ·     │
│                Tracing                  (atlas/core)              │
├───────────────────────────────────────────────────────────────────┤
│ Capabilities   Tools · Memory · RAG · MCP bridge · Server tools   │
├───────────────────────────────────────────────────────────────────┤
│ Gateway        LLM interface → adapters: Claude · OpenAI-compat · │
│                Gemini                   (atlas/llm)               │
└───────────────────────────────────────────────────────────────────┘
```

Dependencies point downward only. Only the adapters in `atlas/llm/` import a vendor SDK (`anthropic`, `openai`, `google-genai`). Everything above them depends on the `LLM` protocol, which is why the provider is a config change and the test suite can replace the model with a script.

## Lifecycle of a chat message

```mermaid
sequenceDiagram
    participant UI as Web UI
    participant API as FastAPI
    participant AG as Assistant (Agent)
    participant GW as LLM gateway
    participant C as Claude API
    participant T as Tools

    UI->>API: POST /api/chat {message, session_id}
    API->>AG: session.send(agent, message)  (background task)
    AG->>AG: input guardrail (redact secrets/PII)
    loop until no tool_use / budget / max_turns
        AG->>GW: turn(system, tools, messages, effort)
        GW->>C: messages.stream (thinking, caching, fallbacks, compaction)
        C-->>GW: SSE deltas (thinking, text, tool_use)
        GW-->>API: events → SSE → UI (live)
        GW-->>AG: TurnResult(content, stop_reason, usage)
        AG->>AG: append assistant turn (append-only)
        alt stop_reason == tool_use
            par read-only tools
                AG->>T: execute (validated, timed, traced)
            and side-effecting tools
                AG-->>UI: approval_request (SSE)
                UI->>API: POST /api/approvals/{id}
                AG->>T: execute if approved
            end
            AG->>AG: fence untrusted output; append ONE user msg with all tool_results
        end
    end
    AG-->>API: RunResult(text, citations, stop)
    API-->>UI: done {text, citations, cost, trace_id}
```

## Lifecycle of a research request

1. **Plan.** `llm.parse(ResearchPlan)` returns validated subtasks with a specialist, dependencies and success criteria for each. `execution_levels()` topologically sorts them (Kahn's algorithm) and rejects cycles.
2. **Execute.** Each dependency level runs concurrently under a semaphore (`max_parallel_workers`). A worker is a normal `Agent` with a role prompt and a least-privilege tool registry. Its brief contains only its own instructions plus its dependencies' outputs.
3. **Write.** The writer agent turns all findings into a Markdown report.
4. **Review.** `llm.parse(Critique)` scores the report against the plan's success criteria.
5. **Revise.** If the report fails review, the critique is appended to the *same* writer conversation, and the cycle repeats up to `max_critique_rounds` times.

A failing worker produces a `FAILED: ...` finding instead of an exception, so the writer can report the gap.

## Key decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Canonical Claude-style message format + per-provider adapters | Lossless history, one agent loop for every vendor, capability flags instead of provider checks | Each adapter must translate tool calls, results and reasoning state carefully (covered by mocked-HTTP tests) |
| Hand-written agent loop | Full control of approvals, budgets, parallelism and stop reasons, and a good way to learn | More code than the SDK Tool Runner |
| Workflow-shaped orchestrator | Predictable, testable, cheaper than a fully autonomous manager agent | Less flexible for wildly open-ended tasks |
| Same model for workers, lower effort | Lower effort on a top model usually beats a weaker model, and one cache namespace | Higher per-token price than a small model; measure for your workload |
| Append-only transcripts + server-side compaction | Thinking blocks stay valid and the prompt cache stays warm | Compaction is a beta feature |
| Strict tool schemas + Pydantic re-validation | Schema-valid inputs from the API, safe handling of eagerly streamed inputs | Constraints like `minimum` are moved into descriptions |
| `search_result` blocks for RAG | Native, verifiable citations | Claude-specific content type |
| SQLite + in-memory BM25/vector search | Zero infrastructure, easy to read | Linear scan; switch to pgvector or Qdrant for large corpora |
| Hashing embedder fallback | Runs everywhere with no downloads | Not semantic; install `[embeddings]` for real vectors |
| Heuristic injection detection | Fast, explainable first line of defence | Bypassable; pair with model-side judgement and least privilege |
| SSE instead of WebSockets | One-way stream, plain HTTP, proxy-friendly | Approvals need a separate POST |

## Data on disk (`ATLAS_DATA_DIR`, default `.atlas/`)

| Path | Contents |
|---|---|
| `knowledge.db` | SQLite: documents, chunks, embeddings, embedder name |
| `memories/` | Long-term memory files written by the memory tool |
| `sessions/<id>.json` | Chat transcripts (short-term memory) |
| `workspace/` | Sandbox for `read_file` / `write_file` |
| `traces/<trace_id>.jsonl` | One span per line |
| `evals/<timestamp>.json` | Eval results for regression tracking |

## Scaling path

- **Retrieval:** swap `KnowledgeBase` for pgvector or Qdrant behind the same `search()` signature, and add a reranker stage.
- **Execution:** move agent runs to a task queue (Celery, Arq or Temporal) and stream events through Redis pub/sub to the SSE endpoint.
- **State:** move sessions and approvals from files and memory to Postgres or Redis so multiple API replicas work.
- **Tracing:** replace `Tracer._export` with an OpenTelemetry exporter (the span fields already match).
- **Hosted agents:** Claude Managed Agents can host the loop and a per-session sandbox if you don't want to run them yourself.
