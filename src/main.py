"""FastAPI wrapper around the research crew graph.

Design note: a 5-agent sequential pipeline with an LLM call per step
routinely takes 30-90+ seconds. A synchronous POST that blocks for that
long will hit client-side timeouts and can't be load-tested realistically,
so this exposes a job-submission pattern instead:

  POST /research      -> returns {job_id} immediately (202 Accepted)
  GET  /research/{id} -> poll for status: "running" | "done" | "failed"

This is the standard pattern for long-running LLM pipelines in production
(the same shape as e.g. OpenAI's batch API or any async task queue), and
it's what actually lets a frontend show progress instead of hanging on a
single request. Jobs are stored in-memory here for simplicity - swap the
`_jobs` dict for Redis/Postgres if you need persistence across restarts
or multiple worker processes.
"""

import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph import research_crew_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-crew")

app = FastAPI(
    title="Agentic Research Pipeline",
    description="Planner -> Researcher -> Analyst -> Writer -> Critic "
    "agent pipeline built with LangGraph, with an async job API.",
    version="1.1.0",
)


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ResearchRequest(BaseModel):
    task: str


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus


class ResearchResult(BaseModel):
    report: str
    revisions: int
    final_score: Optional[int] = None
    hit_revision_limit: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    finished_at: Optional[str] = None
    result: Optional[ResearchResult] = None
    error: Optional[str] = None


# In-memory job store. Fine for a single-process demo/portfolio deployment;
# not durable across restarts and not shared across multiple workers.
_jobs: dict[str, JobStatusResponse] = {}
_jobs_lock = threading.Lock()


def _run_job(job_id: str, task: str) -> None:
    logger.info("[job %s] starting research run for task=%r", job_id, task)
    try:
        result = research_crew_graph.invoke({"task": task})
        critique = result.get("critique")
        revisions = result.get("revision_count", 0)
        hit_limit = bool(critique and not critique.approved)

        research_result = ResearchResult(
            report=result["final_report"],
            revisions=revisions,
            final_score=critique.score if critique else None,
            hit_revision_limit=hit_limit,
        )
        with _jobs_lock:
            _jobs[job_id].status = JobStatus.DONE
            _jobs[job_id].result = research_result
            _jobs[job_id].finished_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "[job %s] finished: revisions=%d hit_revision_limit=%s score=%s",
            job_id,
            revisions,
            hit_limit,
            research_result.final_score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[job %s] failed", job_id)
        with _jobs_lock:
            _jobs[job_id].status = JobStatus.FAILED
            _jobs[job_id].error = str(exc)
            _jobs[job_id].finished_at = datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=JobSubmitResponse, status_code=202)
def submit_research(request: ResearchRequest):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="task must not be empty")

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = JobStatusResponse(
            job_id=job_id,
            status=JobStatus.RUNNING,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    # Run in a background thread so the request returns immediately instead
    # of blocking for the full multi-agent run. For real production scale
    # (multiple workers, restart-safety) swap this for a task queue like
    # Celery/RQ/arq backed by Redis - the job-store interface stays the same.
    thread = threading.Thread(target=_run_job, args=(job_id, request.task), daemon=True)
    thread.start()

    return JobSubmitResponse(job_id=job_id, status=JobStatus.RUNNING)


@app.get("/research/{job_id}", response_model=JobStatusResponse)
def get_research_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job