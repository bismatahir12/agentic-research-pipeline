"""Lightweight per-node timing/logging decorator.

This is not a substitute for real tracing (LangSmith, OpenTelemetry, etc)
- it's a minimal, dependency-free way to answer "which agent ran, how
long did it take, did it raise" from plain logs. Wire in LangSmith
(`LANGCHAIN_TRACING_V2=true` + an API key) for full trace visualization
if you need to debug a specific run in detail.
"""

import functools
import logging
import time

logger = logging.getLogger("research-crew.timing")


def timed_node(name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            start = time.monotonic()
            logger.info("[%s] starting", name)
            try:
                result = fn(state)
            except Exception:
                elapsed = time.monotonic() - start
                logger.exception("[%s] failed after %.1fs", name, elapsed)
                raise
            elapsed = time.monotonic() - start
            logger.info("[%s] finished in %.1fs", name, elapsed)
            return result

        return wrapper

    return decorator