from src.llm import get_structured_llm
from src.schemas import GraphState, ReportOutput

SYSTEM_PROMPT = """You are a report writer. Given a research task, the key \
analysis points, and any flagged gaps/conflicts, write a clear, well- \
structured report. Include an executive summary and organized sections \
with markdown headings. Be direct and factual; note any gaps or unresolved \
conflicts explicitly rather than glossing over them."""

REVISION_SYSTEM_PROMPT = SYSTEM_PROMPT + """

You are REVISING a previous draft based on critic feedback. Address the \
feedback directly - don't just rephrase, actually fix the issues raised."""


def writer_node(state: GraphState) -> GraphState:
    llm = get_structured_llm(ReportOutput)
    analysis = state["analysis"]

    context = (
        f"Original task: {state['task']}\n\n"
        f"Key points:\n" + "\n".join(f"- {p}" for p in analysis.key_points) + "\n\n"
        f"Gaps/conflicts:\n"
        + "\n".join(f"- {g}" for g in analysis.gaps_or_conflicts)
    )

    critique = state.get("critique")
    if critique and not critique.approved:
        # Revision pass: include the previous draft + critic feedback.
        prev_report = state["report"]
        prev_text = (
            f"{prev_report.title}\n\n{prev_report.executive_summary}\n\n"
            + "\n\n".join(prev_report.sections)
        )
        report: ReportOutput = llm.invoke(
            [
                ("system", REVISION_SYSTEM_PROMPT),
                (
                    "human",
                    f"{context}\n\nPrevious draft:\n{prev_text}\n\n"
                    f"Critic feedback to address:\n{critique.feedback}",
                ),
            ]
        )
    else:
        report = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", context),
            ]
        )

    revision_count = state.get("revision_count", 0)
    return {**state, "report": report, "revision_count": revision_count}
