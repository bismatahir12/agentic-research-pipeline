from langgraph.graph import END, StateGraph

from src.agents.analyst import analyst_node
from src.agents.critic import critic_node
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.writer import writer_node
from src.config import MAX_REVISIONS
from src.schemas import GraphState


def _route_after_critic(state: GraphState) -> str:
    """Conditional edge: approve and finish, retry, or give up after
    MAX_REVISIONS and ship the best draft we have anyway."""
    critique = state["critique"]
    if critique.approved:
        return "finalize"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        return "finalize"
    return "revise"


def _finalize_node(state: GraphState) -> GraphState:
    report = state["report"]
    text = (
        f"# {report.title}\n\n"
        f"**Executive Summary**\n\n{report.executive_summary}\n\n"
        + "\n\n".join(report.sections)
    )
    critique = state.get("critique")
    if critique and not critique.approved:
        text += (
            f"\n\n---\n*Note: shipped after reaching the max revision limit "
            f"({MAX_REVISIONS}). Last critic score: {critique.score}/10.*"
        )
    return {**state, "final_report": text}


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", _finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "critic")

    # This conditional edge is the reflection loop: critic can send the
    # graph back to the writer, or forward to finalize.
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"revise": "writer", "finalize": "finalize"},
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled once at import time; reused across requests.
research_crew_graph = build_graph()
