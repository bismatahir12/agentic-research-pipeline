"""Web search tool for the Researcher agent.

Uses DuckDuckGo so the whole project is runnable by anyone who clones it
without needing a paid search API key. Swap `run_search` for a call to
Tavily/Serper/Bing if you want higher-quality results — the interface
(a list of {title, snippet, url} dicts) is the contract the rest of the
pipeline depends on, so nothing else needs to change.
"""

from duckduckgo_search import DDGS


def run_search(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search and return a normalized list of results.

    Returns an empty list (rather than raising) on failure, so a flaky
    search provider degrades the Researcher's output quality instead of
    crashing the whole graph run.
    """
    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a
        # best-effort external call and callers only need a safe fallback.
        print(f"[web_search] search failed for query={query!r}: {exc}")
        return []

    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in raw_results
    ]
