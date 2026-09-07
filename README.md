# Agentic Research Pipeline

A agentic research pipeline built with **LangGraph** that takes a research question and
produces a structured, fact-checked report through a pipeline of specialized
agents that plan, research, analyze, write, and critique each other's work.

This is a portfolio/learning project demonstrating agent orchestration,
tool-calling, structured outputs, a reflection loop with an objective
fallback check, concurrent tool execution, and an async job-based API
suited to long-running LLM pipelines. It is **not** a hardened production
system — see [Known limitations](#known-limitations) below for an honest
account of what's missing for that.

---

## Architecture

```
                 ┌─────────────┐
   User task ──▶ │   Planner   │  breaks the task into research subtasks
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ Researcher  │  runs subtasks CONCURRENTLY (thread pool)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Analyst   │  synthesizes findings, flags gaps/conflicts
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │   Writer    │  drafts the structured report
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
        ┌───────▶│   Critic    │  LLM score + objective structural checks
        │        └──────┬──────┘
        │ revise        │ approve, or limit reached
        │  (loop)       ▼
        └────────  Final report (flagged if shipped past the revision limit)
```

State is passed between agents as a typed `GraphState` object (see
`src/schemas.py`), and every agent's output is validated against a Pydantic
schema before being handed to the next node, so a malformed LLM response
fails fast and visibly instead of silently corrupting downstream steps.

## Stack

- **LangGraph** — explicit state-machine orchestration (not just a linear
  chain — includes a conditional loop)
- **Anthropic Claude API** — the LLM backing every agent
- **duckduckgo-search** — free, no-API-key web search tool for the Researcher
- **Pydantic** — structured output validation for every agent
- **FastAPI** — exposes the graph as a `POST /research` endpoint
- **Docker / docker-compose** — containerized, one-command run
- **pytest** — unit tests for the graph logic and schemas

## Project layout

```
src/
  config.py            # env/config loading
  schemas.py           # Pydantic models for state + each agent's output
  timing.py            # per-node timing/logging decorator
  tools/
    web_search.py       # DuckDuckGo search tool used by the Researcher
  agents/
    planner.py
    researcher.py        # runs subtasks concurrently
    analyst.py
    writer.py
    critic.py             # LLM judge + objective structural checks
  graph.py              # LangGraph wiring: nodes, edges, conditional loop
  main.py               # FastAPI app: async job submission + polling
tests/
  test_graph.py
```

## Setup

```bash
git clone <your-repo-url>
cd multi-agent-research-crew
cp .env.example .env        # add your ANTHROPIC_API_KEY
pip install -r requirements.txt
```

Get a free API key at [console.groq.com](https://console.groq.com) (no card
required) and put it in `.env` as `GROQ_API_KEY`. This is the default
provider (`LLM_PROVIDER=groq`) so no other setup is needed. If you'd rather
use Anthropic's Claude, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
instead — `src/llm.py` handles the switch.

## Run locally

```bash
uvicorn src.main:app --reload
```

Submit a job:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"task": "What are the tradeoffs between LoRA and full fine-tuning for small LLMs?"}'
# -> {"job_id": "...", "status": "running"}
```

Poll for the result:

```bash
curl http://localhost:8000/research/<job_id>
```

Or use the interactive docs at `http://localhost:8000/docs`.

## Run with Docker

```bash
docker compose up --build
```

## Run tests

```bash
pytest
```

## Design notes / why these choices

- **LangGraph over a plain agent framework**: the critic → writer revision
  loop needs a real cyclic graph with conditional edges, which LangGraph
  models explicitly and makes inspectable — you can log/visualize exactly
  which node ran when and why.
- **DuckDuckGo search over a paid API**: keeps the project runnable by
  anyone cloning the repo with zero cost beyond the LLM key, while still
  demonstrating real tool-calling with live external data.
- **Pydantic validation between every agent**: mirrors how this would be
  built in production — an LLM's raw text output is never trusted directly
  by the next stage.

## Possible extensions

- Swap the web-search tool for a real API (Tavily, Serper) behind the
  same interface for more reliable, higher-quality results.
- Add LangSmith or OpenTelemetry tracing for full run visibility.
- Replace the in-memory job store with Redis/Postgres + a real task
  queue for multi-worker deployments.
- Add a held-out eval set to measure Critic judge quality directly,
  rather than trusting it by construction.
- Add a vector-store long-term memory so the crew remembers prior
  research sessions.