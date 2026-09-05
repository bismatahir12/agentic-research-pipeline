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


def researcher_node(state: GraphState) -> GraphState:
    plan = state["plan"]
    llm = get_llm(temperature=0.2)

    findings: list[ResearchFinding] = []
    for subtask in plan.subtasks:
        results = run_search(subtask)
        formatted = _format_results(results)

        response = llm.invoke(
            [
                ("system", SUMMARY_SYSTEM_PROMPT),
                ("human", f"Question: {subtask}\n\nSearch results:\n{formatted}"),
            ]
        )

        findings.append(
            ResearchFinding(
                subtask=subtask,
                summary=response.content,
                sources=[r["url"] for r in results if r.get("url")],
            )
        )

    return {**state, "research": ResearchOutput(findings=findings)}
