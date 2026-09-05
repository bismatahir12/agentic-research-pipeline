from src.llm import get_structured_llm
from src.schemas import AnalysisOutput, GraphState

SYSTEM_PROMPT = """You are a research analyst. Given a set of research \
findings (one per subtask), extract the key points that matter for the \
overall research task, and separately flag any gaps (subtasks with thin or \
no evidence) or conflicts (findings that contradict each other)."""


def _format_findings(research) -> str:
    return "\n\n".join(
        f"Subtask: {f.subtask}\nSummary: {f.summary}" for f in research.findings
    )


def analyst_node(state: GraphState) -> GraphState:
    llm = get_structured_llm(AnalysisOutput)
    formatted = _format_findings(state["research"])

    analysis: AnalysisOutput = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                f"Original task: {state['task']}\n\nFindings:\n{formatted}",
            ),
        ]
    )
    return {**state, "analysis": analysis}
