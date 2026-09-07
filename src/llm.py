"""Shared LLM client factory.

Every agent gets its own structured-output-bound client via
`get_structured_llm`, so each agent's LLM call is guaranteed to return an
instance of that agent's Pydantic schema (or raise) rather than raw text
that has to be parsed by hand downstream.

Provider is swappable via LLM_PROVIDER in .env:
- "groq" (default) - free tier, no card required, sign up at console.groq.com
- "anthropic" - if you have a Claude API key

This keeps the project runnable by anyone cloning the repo with zero cost.

Known failure mode: tool-calling based structured output on open-weight
models (Groq's included) can occasionally emit malformed JSON for long,
deeply-nested responses (observed on the Writer's ReportOutput schema in
testing - Groq's function-call parser choked on a large multi-section
report with escaped markdown). This is a real, observed reliability gap,
not hypothetical. The mitigation here is `.with_retry(...)`: a transient
parse failure gets retried with fresh sampling rather than failing the
whole job outright. This does not eliminate the failure mode - it reduces
its practical impact. A more robust fix for production would be to keep
each structured object smaller (e.g. one LLM call per report section
instead of the whole report in one shot) or use a provider with more
reliable structured-output support for large payloads.
"""

from pydantic import BaseModel

from src.config import LLM_PROVIDER, CREW_MODEL, GROQ_API_KEY, ANTHROPIC_API_KEY

STRUCTURED_OUTPUT_MAX_ATTEMPTS = 3


def get_llm(temperature: float = 0.3):
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=CREW_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=CREW_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Use 'groq' or 'anthropic'."
        )


def get_structured_llm(schema: type[BaseModel], temperature: float = 0.3):
    """Return an LLM runnable that outputs a validated instance of `schema`,
    retrying on transient failures (including malformed tool-call JSON from
    the model itself - see module docstring)."""
    return (
        get_llm(temperature=temperature)
        .with_structured_output(schema)
        .with_retry(stop_after_attempt=STRUCTURED_OUTPUT_MAX_ATTEMPTS)
    )