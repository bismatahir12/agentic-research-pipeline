# Agentic Research Pipeline

A agentic research pipeline built with **LangGraph** that takes a research question and
produces a structured, fact-checked report through a pipeline of specialized
agents that plan, research, analyze, write, and critique each other's work.

This project demonstrates: agent orchestration, tool-calling, structured
outputs, a reflection/critic loop, state management, and a production-style
FastAPI wrapper with Docker deployment.

---

## Architecture

```
                 ┌─────────────┐
   User task ──▶ │   Planner   │  breaks the task into research subtasks
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │ Researcher  │  runs web-search tool calls per subtask
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
        ┌───────▶│   Critic    │  scores the draft against a rubric
        │        └──────┬──────┘
        │ revise        │ approve
        │  (loop)       ▼
        └────────  Final report
```

The Critic loop is the key differentiator: if the draft doesn't meet the
quality bar (coverage, accuracy, structure), it's sent back to the Writer
with specific feedback, up to a configurable number of retries. This is a
real reflection loop, not a single linear chain.

State is passed between agents as a typed `GraphState` object (see
`src/schemas.py`), and every agent's output is validated against a Pydantic
schema before being handed to the next node — so a malformed LLM response
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
  config.py          # env/config loading
  schemas.py          # Pydantic models for state + each agent's output
  tools/
    web_search.py     # DuckDuckGo search tool used by the Researcher
  agents/
    planner.py
    researcher.py
    analyst.py
    writer.py
    critic.py
  graph.py             # LangGraph wiring: nodes, edges, conditional loop
  main.py              # FastAPI app exposing the graph
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

## Run locally

```bash
uvicorn src.main:app --reload
```

Then:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"task": "What are the tradeoffs between LoRA and full fine-tuning for small LLMs?"}'
```

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

- Swap the web-search tool for a real API (Tavily, Serper) behind the same
  interface for higher-quality results.
- Add a vector-store long-term memory so the crew remembers prior research
  sessions.
- Add an eval harness (LLM-as-judge) to score report quality over many runs
  instead of relying on the in-loop Critic alone.
- Stream intermediate agent steps to the client over WebSockets so a UI can
  show live progress.
