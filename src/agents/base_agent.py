"""
Agent abstract base class

Every Agent that can execute a SubTask must inherit BaseAgent.
Uses the Strategy Pattern: the policy object is injected via dependency injection, making it easy to mock in unit tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.schemas import SubTask, AgentResult


__all__ = ["BaseAgent"]


class BaseAgent(ABC):
    """Agent abstract base class.

    Attributes:
        name: Agent instance name, used for logging and monitoring.
        policy: VLLMPolicy instance providing LLM call capability.
        tools: list of tools available to this Agent.
    """

    def __init__(self, name: str, policy, tools: list | None = None):
        """Initialize the Agent.

        Args:
            name: Agent name.
            policy: VLLMPolicy instance (or any object implementing the __call__(messages) interface).
            tools: optional list of tools; each element needs name / description / execute.
        """
        self.name = name
        self.policy = policy
        self.tools = tools or []

    @abstractmethod
    async def run(self, task: "SubTask", context: dict) -> "AgentResult":
        """Execute the given SubTask.

        Args:
            task: the atomic task to execute.
            context: global shared context (a snapshot of Memory), read-only.

        Returns:
            AgentResult: containing status, output, trajectory, etc.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} tools={len(self.tools)}>"
