import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph import research_crew_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("research-crew")

app = FastAPI(
    title="Multi-Agent Research & Report Crew",
    description="Planner -> Researcher -> Analyst -> Writer -> Critic "
    "agent pipeline built with LangGraph.",
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    task: str


class ResearchResponse(BaseModel):
    report: str
    revisions: int
    final_score: int | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest):
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="task must not be empty")

    logger.info("Starting research crew run for task=%r", request.task)

    try:
        result = research_crew_graph.invoke({"task": request.task})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Research crew run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    critique = result.get("critique")
    return ResearchResponse(
        report=result["final_report"],
        revisions=result.get("revision_count", 0),
        final_score=critique.score if critique else None,
    )
