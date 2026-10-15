# Presenting Atlas

Ready-to-adapt text for a resume, LinkedIn and Upwork. Every claim below maps to code in this repository. Keep it that way: only add numbers (latency, cost per report, eval scores) that you have measured with your own API key.

## Resume entry

**Atlas: Provider-Agnostic Multi-Agent Research Platform** · Python, Claude / OpenAI / Gemini APIs, Ollama, MCP, FastAPI, SQLite, Docker
- Built a production-style agentic AI system in which a planner decomposes research questions into a dependency graph executed by parallel specialist agents (web researcher, knowledge-base librarian, sandboxed data analyst), followed by a writer and an LLM critic that drives revision loops.
- Designed a provider-agnostic LLM layer (ports and adapters) so the same agents run on Claude, OpenAI, Gemini or local models (Ollama, LM Studio, vLLM): canonical message translation, capability-based tool selection, Gemini thought-signature round-tripping, and validate-and-repair structured outputs.
- Implemented the agent loop from scratch: parallel tool execution, all API stop reasons (`pause_turn`, `max_tokens`, `refusal`), turn and dollar budgets, and human approval gates for side-effecting tools over Server-Sent Events.
- Designed hybrid RAG (BM25 written from scratch, dense embeddings, Reciprocal Rank Fusion) that returns citation-enabled search results; reached recall@k of 1.0 and MRR of 1.0 on the bundled retrieval eval suite.
- Built both sides of the Model Context Protocol: an MCP server that exposes the knowledge base to Claude Desktop and Claude Code, and a client bridge that loads tools from any MCP server at runtime.
- Added defence-in-depth guardrails (indirect prompt-injection fencing, secret and PII redaction, path sandboxing), OpenTelemetry-style tracing, per-call cost and cache-hit accounting, and an eval harness (retrieval metrics, tool-trajectory checks, LLM-as-judge).
- Used each platform's strengths where available (on Claude: adaptive thinking, prompt caching, server-side compaction, refusal fallbacks). Shipped with 67 offline tests (each vendor SDK exercised against mocked HTTP), CI, and a non-root Docker image.

Shorter, 3-bullet version:
- Built a provider-agnostic multi-agent research platform (Claude, OpenAI, Gemini, local LLMs): a planner, parallel specialist agents, and a critic-driven revision loop with human approval gates.
- Implemented hybrid RAG with native citations, long-term memory, and MCP as both server and client.
- Production practices throughout: guardrails, tracing, cost budgets, evals, 67 tests, Docker and CI.

## Upwork profile: portfolio item

**Title:** Atlas: Multi-Agent AI Research Platform (Claude, OpenAI, Gemini or local LLMs + MCP + RAG)

**Description:**
Atlas is an agentic AI system I designed and built end to end. Ask it a question and it searches your private documents and the web and answers with citations. Give it a big research task and it assembles a team of AI agents: one plans the work, several research in parallel, one analyses data in a secure sandbox, one writes the report, and a reviewer agent sends it back for revision until it meets the quality bar. Sensitive actions wait for human approval, every step is traced, and spending is capped.

What this shows I can build for you:
- AI agents that run on the model you choose (Claude, OpenAI, Gemini) or privately on your own hardware (Ollama, vLLM), and can switch between them
- AI assistants and agents grounded in **your** documents (RAG with citations)
- Multi-agent workflows that automate research, analysis and reporting
- MCP servers that connect Claude Desktop and Claude Code to your internal tools and data
- Production essentials: guardrails, human approval steps, cost control, monitoring, automated evaluation, Docker deployment

**Skills/tags:** AI Agent Development · LLM · Claude API · OpenAI API · Gemini API · Ollama / Local LLMs · Model Context Protocol (MCP) · RAG · Multi-Agent Systems · Python · FastAPI · Prompt Engineering · Vector Search · Docker

## Talking points for interviews and client calls

1. **"Walk me through the agent loop."** Model call, then append, then run tools (parallel if read-only, gated if side-effecting), then one tool-result message, then repeat. Mention how each stop reason is handled and why history is append-only.
2. **"How do you stop prompt injection?"** No single fix. Fence untrusted content, detect known patterns, give each role least privilege, and require human approval for irreversible actions. Then show the planted attack in `examples/knowledge/vendor-note.md` and its eval.
3. **"How do you control cost?"** Prompt caching with a stable prefix, effort per role, a hard dollar budget shared across sub-agents, context isolation for workers, and cache-hit monitoring.
4. **"How do you know it works?"** Three eval layers: deterministic retrieval metrics, trajectory checks, and LLM-as-judge. Free evals run in CI on every push.
5. **"Why not LangChain / CrewAI?"** Frameworks are fine. This project owns the loop so every behaviour is explicit, testable and debuggable, and the same patterns map directly onto any framework or onto Claude Managed Agents.

## Before you publish

- Put your name in `LICENSE`.
- Run it with your key and record a 60–90 second demo (web UI: chat with citations, then a research run showing parallel workers and a critique round). Put a GIF at the top of the README.
- Add measured numbers from your own runs (cost per research report, cache-hit ratio, agent eval pass rate).
- Push to a public GitHub repo; CI runs automatically.
