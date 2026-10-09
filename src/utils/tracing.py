"""
LangSmith tracing integration module

Provides observability for the DeepResearch Agent without depending on LangChain/LangGraph.
Core capabilities:
  1. Automatic tracing of LLM calls (wrapping the VLLMPolicy client via wrap_openai)
  2. Manual instrumentation of the agent flow (decorating key methods with @traceable)
  3. Environment-variable switch (LANGSMITH_TRACING=true/false)

Usage:
  1. Configure the LangSmith environment variables in .env
  2. The system detects them and enables tracing automatically
  3. Log in to https://smith.langchain.com to view the trace tree

Design principles:
  - Zero intrusion: business code is unaware of tracing
  - Switchable: enable/disable with one environment variable
  - Low cost: when disabled, no LangSmith objects are created
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, TypeVar


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def is_tracing_enabled() -> bool:
    """Check whether LangSmith tracing is enabled."""
    from .env_config import get_env
    # get_env turns an empty string into None, so guard against it; otherwise an AttributeError occurs when LangSmith is not configured
    return (get_env("LANGSMITH_TRACING") or "").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# OpenAI client wrapper (automatically traces all LLM calls)
# ---------------------------------------------------------------------------


def maybe_wrap_openai_client(client: Any) -> Any:
    """Wrap an OpenAI client to enable LangSmith automatic LLM tracing.

    Args:
        client: the original openai.OpenAI instance.

    Returns:
        The wrapped client (when tracing is on) or the original client (when off).
    """
    if not is_tracing_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai
        return wrap_openai(client, chat_name="ChatOpenAI")
    except Exception as e:
        logger.warning(f"wrap_openai failed, falling back to the original client: {e}")
        return client


def maybe_wrap_anthropic_client(client: Any) -> Any:
    """Wrap an Anthropic client to enable LangSmith automatic LLM tracing (returned unchanged when disabled / not installed)."""
    if not is_tracing_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_anthropic
        return wrap_anthropic(client)
    except Exception as e:
        logger.warning(f"wrap_anthropic failed, falling back to the original client: {e}")
        return client


# ---------------------------------------------------------------------------
# Compatibility decorators (support sync / async / class methods)
# ---------------------------------------------------------------------------


def traceable(
    run_type: str = "chain",
    name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable:
    """Compatibility decorator: uses @traceable if LangSmith is on, otherwise a no-op decorator.

    Supports sync functions, async functions and class methods.

    Args:
        run_type: LangSmith run type. Common values:
                  "chain"   — generic flow block
                  "llm"     — LLM call (wrap_openai already handles it, usually no need to set manually)
                  "tool"    — tool call
                  "agent"   — agent execution
                  "retriever" — retrieval operation
        name: name shown in the LangSmith UI. When None, the function name is used.
        tags: list of tags for filtering and grouping.
        metadata: dict of extra metadata.

    Usage example:
        @traceable(run_type="agent", tags=["m5", "red"])
        async def attack(self, report): ...
    """
    def decorator(func: Callable) -> Callable:
        # if tracing is not enabled, return the original function (zero overhead)
        if not is_tracing_enabled():
            return func

        try:
            from langsmith import traceable as _ls_traceable
            # use the native LangSmith decorator
            return _ls_traceable(
                run_type=run_type,
                name=name or func.__name__,
                tags=tags or [],
                metadata=metadata or {},
            )(func)
        except Exception as e:
            logger.warning(f"[LangSmith] failed to apply the traceable decorator ({func.__name__}): {e}")
            return func

    return decorator


# ---------------------------------------------------------------------------
# Manual trace context (for cases where decorators are inconvenient)
# ---------------------------------------------------------------------------


def trace_block(
    name: str,
    run_type: str = "chain",
    inputs: dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """Context manager: manually wrap a block of code.

    Usage example:
        with trace_block("adversarial_loop", run_type="chain", inputs={"query": q}) as run:
            report = await loop.run(report)
            run.add_output({"score": report.final_score})
    """
    if not is_tracing_enabled():
        # return a dummy context manager
        class _DummyRun:
            def add_output(self, outputs: dict) -> None:
                pass
        from contextlib import contextmanager
        @contextmanager
        def _dummy():
            yield _DummyRun()
        return _dummy()

    try:
        from langsmith.run_helpers import trace
        return trace(name=name, run_type=run_type, inputs=inputs or {}, tags=tags or [])
    except Exception as e:
        logger.warning(f"[LangSmith] failed to create trace_block: {e}")
        from contextlib import contextmanager
        @contextmanager
        def _dummy():
            class _DummyRun:
                def add_output(self, outputs: dict) -> None:
                    pass
            yield _DummyRun()
        return _dummy()


# ---------------------------------------------------------------------------
# Shortcut decorators (preconfigured per scenario)
# ---------------------------------------------------------------------------


def trace_agent(name: str | None = None, tags: list[str] | None = None):
    """Agent execution tracing (run_type="chain"; LangSmith does not support "agent")."""
    return traceable(run_type="chain", name=name, tags=tags)


def trace_tool(name: str | None = None, tags: list[str] | None = None):
    """Tool-call tracing (run_type="tool")."""
    return traceable(run_type="tool", name=name, tags=tags)


def trace_chain(name: str | None = None, tags: list[str] | None = None):
    """Generic flow tracing (run_type="chain")."""
    return traceable(run_type="chain", name=name, tags=tags)


def trace_retriever(name: str | None = None, tags: list[str] | None = None):
    """Retrieval tracing (run_type="retriever")."""
    return traceable(run_type="retriever", name=name, tags=tags)
