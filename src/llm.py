"""Shared LLM client factory."""

from pydantic import BaseModel

from src.config import LLM_PROVIDER, CREW_MODEL, GROQ_API_KEY, ANTHROPIC_API_KEY


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
    return get_llm(temperature=temperature).with_structured_output(schema)