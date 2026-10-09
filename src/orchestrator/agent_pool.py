"""
Agent lifecycle management (AgentPool)

Handles creation, reuse, timeout and degradation of worker agents.
Uses the object-pool pattern to cut repeated creation overhead, and routes by TaskType to different Agent implementations.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.base_agent import BaseAgent
    from .schemas import TaskType


__all__ = ["AgentPool"]


class AgentPool:
    """Agent object pool.

    Design points:
      - lazy creation: an Agent type is instantiated only on its first request
      - reuse: an Agent of the same type returns to the pool after release
      - degradation: when an Agent times out or raises, it is marked "needs rebuild"
      - thread safety: no lock needed in the single-threaded asyncio model, but state changes must be atomic

    Attributes:
        policy_factory: zero-argument factory returning a policy instance.
        tools_factory: zero-argument factory returning the tools list.
        max_idle: maximum idle Agents per type, to prevent memory bloat.
    """

    def __init__(
        self,
        policy_factory,
        tools_factory=None,
        max_idle: int = 3,
        researcher_cls=None,
    ) -> None:
        self.policy_factory = policy_factory
        self.tools_factory = tools_factory
        self.researcher_cls = researcher_cls  # None -> ResearcherAgent; the finance scenario injects FinanceResearcherAgent
        self.max_idle = max(max_idle, 1)

        # type -> list of idle Agents
        self._idle: dict[str, list[BaseAgent]] = {}
        # type -> number of active Agents (to limit concurrency, not exact object tracking)
        self._active_count: dict[str, int] = {}
        # type -> total number created (for monitoring)
        self._created_count: dict[str, int] = {}
        # type -> number of times marked failed / needs rebuild
        self._degraded_count: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    async def get_agent(self, task_type: "TaskType") -> BaseAgent:
        """Get an available Agent instance for the task type.

        Prefers reuse from the pool; creates a new one when none is idle.
        """
        type_key = task_type.value

        # initialize the counters for this type
        if type_key not in self._idle:
            self._idle[type_key] = []
            self._active_count[type_key] = 0
            self._created_count[type_key] = 0
            self._degraded_count[type_key] = 0

        # try to reuse an idle Agent
        while self._idle[type_key]:
            agent = self._idle[type_key].pop()
            # simple health check: if the Agent's policy is marked truncated / contaminated, discard it
            if hasattr(agent, "policy") and getattr(agent.policy, "was_truncated", False):
                self._degraded_count[type_key] += 1
                continue  # discard, try the next one
            self._active_count[type_key] += 1
            return agent

        # create a new Agent
        agent = self._create_agent(type_key)
        self._created_count[type_key] += 1
        self._active_count[type_key] += 1
        return agent

    async def release_agent(self, agent: "BaseAgent") -> None:
        """Release an Agent back to the object pool.

        If the Agent state is abnormal (e.g. policy was_truncated), it is discarded, not recycled.
        """
        if agent is None:
            return

        # infer the type (from the agent name or class name)
        type_key = self._infer_type_key(agent)

        self._active_count[type_key] = max(0, self._active_count.get(type_key, 0) - 1)

        # health check
        if hasattr(agent, "policy") and getattr(agent.policy, "was_truncated", False):
            self._degraded_count[type_key] = self._degraded_count.get(type_key, 0) + 1
            return  # do not recycle

        # recycle
        idle_list = self._idle.setdefault(type_key, [])
        if len(idle_list) < self.max_idle:
            idle_list.append(agent)

    def get_stats(self) -> dict[str, dict]:
        """Return object pool statistics."""
        stats = {}
        for key in set(list(self._idle.keys()) + list(self._active_count.keys())):
            stats[key] = {
                "idle": len(self._idle.get(key, [])),
                "active": self._active_count.get(key, 0),
                "created": self._created_count.get(key, 0),
                "degraded": self._degraded_count.get(key, 0),
            }
        return stats

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _create_agent(self, type_key: str) -> "BaseAgent":
        """Create the Agent instance for a type key."""
        policy = self.policy_factory()
        tools = self.tools_factory() if self.tools_factory else []

        # lazy import to avoid a circular dependency
        from ..agents.researcher import ResearcherAgent
        from ..agents.summarizer import SummarizerAgent
        from .schemas import TaskType

        Researcher = self.researcher_cls or ResearcherAgent
        if type_key == TaskType.SEARCH.value:
            return Researcher(name=f"researcher_{type_key}", policy=policy, tools=tools)
        elif type_key == TaskType.ANALYZE.value:
            return Researcher(name=f"analyzer_{type_key}", policy=policy, tools=tools)
        elif type_key == TaskType.VERIFY.value:
            return Researcher(name=f"verifier_{type_key}", policy=policy, tools=tools)
        elif type_key == "synthesize":
            return SummarizerAgent(name="summarizer", policy=policy, tools=tools)
        else:
            # default fallback to Researcher
            return Researcher(name=f"researcher_default", policy=policy, tools=tools)

    def _infer_type_key(self, agent: "BaseAgent") -> str:
        """Infer the type key from an Agent instance."""
        # simple heuristic: infer from the class name
        cls_name = agent.__class__.__name__
        if "Summarizer" in cls_name:
            return "synthesize"
        # ResearcherAgent is used for search/analyze/verify, all grouped under search
        from .schemas import TaskType
        return TaskType.SEARCH.value
