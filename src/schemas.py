from typing import Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured outputs — every agent must return one of these. Validating LLM
# output against a schema (instead of trusting raw text) is what keeps a
# multi-agent pipeline from silently corrupting downstream steps.
# ---------------------------------------------------------------------------


class PlanOutput(BaseModel):
    subtasks: list[str] = Field(
        description="3-6 concrete, independently-searchable research questions "
        "that together cover the user's task."
    )


class ResearchFinding(BaseModel):
    subtask: str
    summary: str = Field(description="2-4 sentence summary of what was found.")
    sources: list[str] = Field(default_factory=list)


class ResearchOutput(BaseModel):
    findings: list[ResearchFinding]


class AnalysisOutput(BaseModel):
    key_points: list[str]
    gaps_or_conflicts: list[str] = Field(
        default_factory=list,
        description="Any contradictions between sources, or subtasks with "
        "thin/no evidence. Empty list if none found.",
    )


class ReportOutput(BaseModel):
    title: str
    executive_summary: str
    sections: list[str] = Field(
        description="Each item is a full section of the report, "
        "'## Heading\\n body text'."
    )


class CritiqueOutput(BaseModel):
    approved: bool
    score: int = Field(ge=1, le=10)
    feedback: str = Field(
        description="If not approved, specific, actionable feedback for the "
        "Writer to address in the next revision."
    )


# ---------------------------------------------------------------------------
# Graph state — the object threaded through every node in the LangGraph.
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    task: str
    plan: Optional[PlanOutput]
    research: Optional[ResearchOutput]
    analysis: Optional[AnalysisOutput]
    report: Optional[ReportOutput]
    critique: Optional[CritiqueOutput]
    revision_count: int
    final_report: Optional[str]
