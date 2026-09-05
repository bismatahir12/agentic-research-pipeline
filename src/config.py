import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "anthropic"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

CREW_MODEL = os.getenv("CREW_MODEL", "llama-3.3-70b-versatile")

MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

_active_key = GROQ_API_KEY if LLM_PROVIDER == "groq" else ANTHROPIC_API_KEY
if not _active_key:
    print(
        f"[config] WARNING: no API key set for LLM_PROVIDER={LLM_PROVIDER!r}. "
        "Set it in .env before running the API for real."
    )