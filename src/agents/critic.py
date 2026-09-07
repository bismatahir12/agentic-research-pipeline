"""Critic agent.

Note on validity: this Critic is an LLM-as-judge, and it's the same model
tier as the Writer it's judging. That's a known weak spot in agentic
pipelines - LLM self-judgment is noisy and can be biased toward its own
phrasing/style rather than genuine quality. Two mitigations here:

1. A small set of objective, non-LLM checks (`_objective_checks`) that
   catch structural failures (empty sections, no executive summary, a
   report that's suspiciously short) regardless of what the LLM judge
   says - these can force a rejection even if the LLM score is high.
2. The LLM critique is still the primary signal for substantive quality
   (factual grounding, honesty about gaps) that objective checks can't
   catch. This isn't a complete fix for judge validity, just a partial
   guardrail - a production system would want a held-out eval set and/or
   a stronger/different judge model to reduce self-bias further.
"""

from src.llm import get_structured_llm
from src.schemas import CritiqueOutput, GraphState, ReportOutput

SYSTEM_PROMPT = """You are a strict editorial critic reviewing a research \
report. Score it 1-10 on: coverage of the original task, factual grounding \
in the provided findings, clarity of structure, and whether flagged \
gaps/conflicts were handled honestly rather than hidden. Approve (score >= 7 \
AND no major issues) or reject with specific, actionable feedback."""

MIN_WORD_COUNT = 150


def _objective_checks(report: ReportOutput) -> list[str]:
    """Cheap, deterministic checks that don't rely on LLM judgment.
    Returns a list of failure reasons (empty list = all checks passed)."""
    failures = []

    if not report.executive_summary.strip():
        failures.append("executive_summary is empty")

    if len(report.sections) == 0:
        failures.append("report has zero sections")

    if any(not s.strip() for s in report.sections):
        failures.append("one or more sections are empty")

    total_words = len(report.executive_summary.split()) + sum(
        len(s.split()) for s in report.sections
    )
    if total_words < MIN_WORD_COUNT:
        failures.append(
            f"report is only {total_words} words (min {MIN_WORD_COUNT})"
        )

    return failures


def critic_node(state: GraphState) -> GraphState:
    report = state["report"]

    objective_failures = _objective_checks(report)

    llm = get_structured_llm(CritiqueOutput)
    report_text = (
        f"{report.title}\n\n{report.executive_summary}\n\n"
        + "\n\n".join(report.sections)
    )

    critique: CritiqueOutput = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                f"Original task: {state['task']}\n\nDraft report:\n{report_text}",
            ),
        ]
    )

    # Objective checks can force a rejection regardless of the LLM's
    # opinion - this is a hard floor the judge can't talk its way past.
    if objective_failures:
        critique = CritiqueOutput(
            approved=False,
            score=min(critique.score, 4),
            feedback=(
                "Objective structural checks failed: "
                + "; ".join(objective_failures)
                + f". LLM feedback: {critique.feedback}"
            ),
        )

    revision_count = state.get("revision_count", 0)
    if not critique.approved:
        revision_count += 1

    return {**state, "critique": critique, "revision_count": revision_count}