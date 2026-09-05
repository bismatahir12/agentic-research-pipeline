# Agentic Research Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Tested-Pytest-0A9EDC?logo=pytest&logoColor=white)

A multi-agent system built with **LangGraph** that takes a research question and
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

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent orchestration | **LangGraph** | Explicit state-machine graph wiring 5 agents together, including a real conditional revision loop (Critic → Writer) |
| LLM inference | **Groq API** (`openai/gpt-oss-120b`) | Free-tier LLM backing every agent; swappable to Anthropic's Claude via one env var |
| Agent framework core | **LangChain Core** | Prompt handling and structured-output binding that LangGraph builds on |
| Data validation | **Pydantic** | Every agent's output is validated against a schema before being passed to the next agent |
| Search tool | **duckduckgo-search** | Free, no-API-key web search tool used by the Researcher agent |
| Backend API | **FastAPI** | Exposes the pipeline as a REST API (`POST /research`, `GET /health`) |
| Server | **Uvicorn** | ASGI server running the FastAPI app |
| Config | **python-dotenv** | Loads secrets/config from `.env` |
| Containerization | **Docker / docker-compose** | One-command, reproducible deployment |
| Testing | **pytest** | Unit tests for graph routing logic and schema validation |
| Language | **Python 3.11+** | |

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
cp .env.example .env
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

## Sample run

Request:
```json
{"task": "What are the pros and cons of remote work for software teams?"}
```

Response (truncated — full report is longer):
```json
{
  "report": "# Pros and Cons of Remote Work for Software Teams – Report\n\n**Executive Summary**\n\nRemote work offers software teams potential advantages such as increased flexibility, access to a broader talent pool, and possible cost savings, while also presenting challenges related to communication, collaboration, team cohesion, and security...\n\n## Potential Advantages (Pros)\n1. Geographic Talent Expansion\n2. Flexibility and Work-Life Balance\n3. Cost Savings\n4. Increased Autonomy\n5. Business Continuity\n\n## Potential Disadvantages (Cons)\n1. Communication Overhead\n2. Collaboration Challenges\n3. Team Cohesion & Culture\n4. Employee Well-Being Risks\n5. Security & Compliance Concerns\n6. Project Management Complexity\n...",
  "revisions": 0,
  "final_score": 8
}
```

The Critic approved this draft on the first pass (`revisions: 0`, `final_score: 8/10`) — no rewrite needed. When the Critic rejects a draft, the same request instead loops back through the Writer (up to `MAX_REVISIONS` times) before returning.

## Design notes / why these choices

- **LangGraph over a plain agent framework**: the critic → writer revision
  loop needs a real cyclic graph with conditional edges, which LangGraph
  models explicitly and makes inspectable — you can log/visualize exactly
  which node ran when and why.
- **Groq as the default LLM provider**: keeps the whole project runnable
  by anyone cloning it for free, while the provider is abstracted behind
  `src/llm.py` so swapping to Anthropic/OpenAI is a one-line env change,
  not a rewrite.
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
