from src.llm import get_structured_llm
from src.schemas import GraphState, PlanOutput

SYSTEM_PROMPT = """You are a research planner. Given a user's research task, \
break it into 3-6 concrete, independently-searchable subtasks that together \
would let someone fully answer the original task. Avoid overlap between \
subtasks. Each subtask should be phrased as a specific question."""


def planner_node(state: GraphState) -> GraphState:
    llm = get_structured_llm(PlanOutput)
    plan: PlanOutput = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", state["task"]),
        ]
    )
    return {**state, "plan": plan}
