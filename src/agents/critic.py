from src.llm import get_structured_llm
from src.schemas import CritiqueOutput, GraphState

SYSTEM_PROMPT = """You are a strict editorial critic reviewing a research \
report. Score it 1-10 on: coverage of the original task, factual grounding \
in the provided findings, clarity of structure, and whether flagged \
gaps/conflicts were handled honestly rather than hidden. Approve (score >= 7 \
AND no major issues) or reject with specific, actionable feedback."""


def critic_node(state: GraphState) -> GraphState:
    llm = get_structured_llm(CritiqueOutput)
    report = state["report"]
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

    revision_count = state.get("revision_count", 0)
    if not critique.approved:
        revision_count += 1

    return {**state, "critique": critique, "revision_count": revision_count}
