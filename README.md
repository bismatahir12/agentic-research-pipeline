# Agentic Research Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-1C3C3C)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20Job%20API-009688?logo=fastapi&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/Tested-Pytest-0A9EDC?logo=pytest&logoColor=white)

A multi-agent system built with **LangGraph** that takes a research question
and produces a structured report through a pipeline of specialized agents
that plan, research (in parallel), analyze, write, and critique their own
output — including a real conditional revision loop, not just a linear
chain.

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

### The revision loop, honestly

If the Critic rejects a draft, it goes back to the Writer with specific
feedback, up to `MAX_REVISIONS` times (default 2). **What happens when the
limit is hit and the Critic still rejects**: the pipeline does not error
or loop forever — `_finalize_node` in `src/graph.py` ships the last draft
anyway, with a visible note appended stating it hit the revision limit and
the last critic score. This is a deliberate "always return something"
choice, not a silent failure — see `src/graph.py::_finalize_node` and the
tests in `tests/test_graph.py` that pin down this exact behavior.

This path is demonstrated with real output, not just unit tests — see
"Sample runs" below, where one run shows `revisions: 1` (Critic rejected
the first draft, Writer revised, Critic approved on the second pass).
`MAX_REVISIONS=0` in `.env` is a quick way to force the "hit the limit
immediately" path for testing the finalize-with-warning behavior.

### Critic validity — the judge is not neutral

The Critic is an LLM-as-judge running on the same model tier as the
Writer it's grading, which is a known source of noisy, self-biased
scoring in agentic pipelines — the judge can be swayed by fluent
phrasing rather than genuine accuracy. Two things are done about this,
neither of which fully solves it:

1. `src/agents/critic.py::_objective_checks` runs deterministic,
   non-LLM checks (empty sections, missing executive summary, a report
   under a minimum word count) that can force a rejection regardless of
   what the LLM judge says — a hard floor the judge can't talk past.
2. The LLM critique is still the primary signal for things objective
   checks can't catch (factual grounding, honesty about gaps).

A more rigorous fix — a held-out eval set, or a stronger/different model
as judge — is a natural next step, not something this project claims to
have solved.

## API design: async jobs, not a blocking call

A 5-agent sequential pipeline with an LLM call per step routinely takes
30-90+ seconds. A synchronous `POST` that blocks for that long hits
client-side timeouts and can't be load-tested realistically, so this
repo uses a submit/poll pattern instead:

```
POST /research          -> 202 {"job_id": "...", "status": "running"}
GET  /research/{job_id} -> {"status": "running" | "done" | "failed", "result": {...} | "error": "..."}
```

Jobs run in a background thread and are tracked in an in-memory dict
(`src/main.py::_jobs`). This is fine for a single-process demo but is
**not** durable across restarts and **not** shared across multiple
worker processes — swap the job store for Redis/Postgres and the
background thread for a real task queue (Celery/RQ/arq) if this needs
to run behind more than one worker. There is also no auth and no rate
limiting on these endpoints; both would be required before exposing
this publicly.

## Known limitations

Being upfront about what this project does *not* solve:

- **Search tool is demo-grade, not production-grade.** `duckduckgo-search`
  scrapes DuckDuckGo's HTML — there's no official API, it can be
  rate-limited or blocked under load (this happened during testing; see
  the sample run below), and it has no SLA. It was chosen so anyone can
  clone this repo and run it for free with zero signup beyond an LLM key.
  For real use, swap `src/tools/web_search.py` for a paid API (Tavily,
  Serper, Bing) behind the same `run_search(query) -> list[dict]`
  interface — nothing else needs to change.
- **No distributed tracing.** Logging is structured per-agent (see
  `src/timing.py` — every node logs start/finish/duration/exceptions),
  but there's no LangSmith or OpenTelemetry integration, so you can't
  visualize a full trace across a run. Set `LANGCHAIN_TRACING_V2=true`
  plus a LangSmith API key to get that for free from LangChain's side.
- **In-memory job store.** Jobs vanish on restart and aren't shared
  across processes — acceptable for a single-instance demo, not for
  anything running multiple replicas.
- **Critic self-bias.** Covered above — objective checks are a partial
  mitigation, not a solved problem.
- **No auth/rate limiting** on the API.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent orchestration | **LangGraph** | Explicit state-machine graph wiring 5 agents together, including a real conditional revision loop (Critic → Writer) |
| LLM inference | **Groq API** (`openai/gpt-oss-120b`) | Free-tier LLM backing every agent; swappable to Anthropic's Claude via one env var |
| Agent framework core | **LangChain Core** | Prompt handling and structured-output binding that LangGraph builds on |
| Data validation | **Pydantic** | Every agent's output is validated against a schema before being passed to the next agent |
| Concurrency | **concurrent.futures.ThreadPoolExecutor** | Runs the Researcher's independent subtasks in parallel instead of sequentially |
| Search tool | **duckduckgo-search** | Free, no-API-key web search tool used by the Researcher agent (see Known limitations) |
| Backend API | **FastAPI** | Async job-submission API (`POST /research`, `GET /research/{id}`, `GET /health`) |
| Server | **Uvicorn** | ASGI server running the FastAPI app |
| Config | **python-dotenv** | Loads secrets/config from `.env` |
| Containerization | **Docker / docker-compose** | One-command, reproducible deployment |
| Testing | **pytest** | Unit tests for graph routing logic, objective checks, and schema validation |
| Language | **Python 3.11+** | |

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
cd agentic-research-pipeline
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

## Sample runs

**Run 1 — revision loop actually triggering** (the previously-undemonstrated
path): the Critic rejected the first draft, sent it back to the Writer with
feedback, and the revised draft was approved.

Request:
```json
{"task": "What are the tradeoffs between LoRA and full fine-tuning?"}
```

Result (truncated — full report is longer):
```json
{
  "status": "done",
  "result": {
    "report": "# Trade-off Analysis of LoRA vs. Full Fine-Tuning\n\n**Executive Summary**\n\nLow-Rank Adaptation (LoRA) and full fine-tuning represent contrasting strategies for adapting large language models. Empirical evidence shows LoRA can achieve near-state-of-the-art performance while using 2-4x less GPU memory and cutting training time by up to 50% for 7B and 13B models...\n\n## Computational Resource Requirements\n\n| Model | Method | GPU Memory | Training Time | Cost |\n|-------|--------|-----------|----------------|------|\n| LLaMA-7B | Full FT | 23 GB | 12h | $1,200 |\n| LLaMA-7B | LoRA (r=8) | 7 GB | 6h | $600 |\n...",
    "revisions": 1,
    "final_score": 9,
    "hit_revision_limit": false
  }
}
```

`revisions: 1` confirms the reflection loop actually ran end-to-end, not
just in unit tests: Critic → reject → Writer revises → Critic → approve.

**Run 2 — first-pass approval, no revision needed:**

Request:
```json
{"task": "What are the pros and cons of remote work for software teams?"}
```

```json
{
  "status": "done",
  "result": {
    "report": "# Pros and Cons of Remote Work for Software Teams – Report\n\n**Executive Summary**\n\nRemote work offers software teams potential advantages such as increased flexibility, access to a broader talent pool, and possible cost savings...",
    "revisions": 0,
    "final_score": 8,
    "hit_revision_limit": false
  }
}
```

Note: this second run happened to hit a period where DuckDuckGo
rate-limited several search calls (logged, not silently swallowed), so the
report leans on the LLM's general knowledge more than fresh search results
for that particular example — see [Known limitations](#known-limitations).
A separate, real bug was also found and fixed during testing: the Writer's
structured-output call occasionally produced malformed JSON on long reports
via Groq's tool-calling parser (`tool_use_failed`) — this surfaced as a
clean `"status": "failed"` with the real error captured (the async job
design working as intended, not crashing), and was fixed by adding retry
logic (`src/llm.py::get_structured_llm`) plus asking the Writer for more
concise sections (`src/agents/writer.py`).

## Design notes / why these choices

- **LangGraph over a plain agent framework**: the critic → writer revision
  loop needs a real cyclic graph with conditional edges, which LangGraph
  models explicitly and makes inspectable.
- **Groq as the default LLM provider**: keeps the whole project runnable
  by anyone cloning it for free, while the provider is abstracted behind
  `src/llm.py` so swapping to Anthropic/OpenAI is a one-line env change.
- **Async job API over a blocking endpoint**: long-running LLM pipelines
  need a submit/poll (or webhook/streaming) pattern in practice; a
  synchronous endpoint that blocks for a minute is a demo shortcut, not
  something you'd ship.
- **Parallel research, sequential everything else**: the Researcher's
  subtasks are independent of each other, so they run concurrently.
  Planner → Researcher → Analyst → Writer → Critic stay sequential
  because each genuinely depends on the previous stage's output.

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