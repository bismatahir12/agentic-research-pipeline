"""Researcher agent.

Runs one search + summarize step per subtask from the Planner. These
subtasks are independent of each other (no subtask depends on another's
result), so they're dispatched concurrently via a thread pool instead of
a sequential for-loop - this cuts wall-clock latency roughly by the number
of subtasks (5-6x on a typical run) since the bottleneck is network I/O
(the search call and the LLM call), not CPU.
"""

from concurrent.futures import ThreadPoolExecutor

from src.llm import get_llm
from src.schemas import GraphState, ResearchFinding, ResearchOutput
from src.tools.web_search import run_search

SUMMARY_SYSTEM_PROMPT = """You are a research assistant. You will be given a \
question and a set of raw web search results (title, snippet, url). Write a \
2-4 sentence factual summary that answers the question using ONLY the given \
results. If the results don't answer the question, say so plainly instead \
of guessing."""


def _format_results(results: list[dict]) -> str:
    if not results:
        return "(no search results found)"
    return "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}"
        for r in results
    )


def _research_one_subtask(subtask: str) -> ResearchFinding:
    llm = get_llm(temperature=0.2)
    results = run_search(subtask)
    formatted = _format_results(results)

    response = llm.invoke(
        [
            ("system", SUMMARY_SYSTEM_PROMPT),
            ("human", f"Question: {subtask}\n\nSearch results:\n{formatted}"),
        ]
    )

    return ResearchFinding(
        subtask=subtask,
        summary=response.content,
        sources=[r["url"] for r in results if r.get("url")],
    )


def researcher_node(state: GraphState) -> GraphState:
    plan = state["plan"]

    # Concurrent, not sequential: each subtask's search+summarize is
    # independent, so run them in parallel rather than one at a time.
    with ThreadPoolExecutor(max_workers=min(len(plan.subtasks), 6)) as executor:
        findings = list(executor.map(_research_one_subtask, plan.subtasks))

    return {**state, "research": ResearchOutput(findings=findings)}