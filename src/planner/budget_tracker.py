"""
Token budget tracker

In a long Deep Research chain the context length can balloon quickly.
BudgetTracker provides explicit token usage monitoring so the orchestrator can decide whether to trigger compression or truncation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


__all__ = ["BudgetTracker", "BudgetSnapshot"]


@dataclass
class BudgetSnapshot:
    """Budget snapshot at a point in time."""
    total_tokens: int = 0
    budget_limit: int = 0
    usage_ratio: float = 0.0
    is_over_budget: bool = False


class BudgetTracker:
    """Track cumulative token consumption, with dynamic budget thresholds.

    Design points:
      - thread safety is the caller's responsibility (the orchestrator calls from a single asyncio event loop)
      - thresholds can be adjusted at runtime, supporting progressive compression strategies
      - usage history is recorded to ease later analysis of the token growth curve
    """

    def __init__(self, budget_limit: int = 100_000) -> None:
        """Initialize the budget tracker.

        Args:
            budget_limit: token budget limit, default 100K (a safe zone for a ~64K-context model).
        """
        self._budget_limit = max(budget_limit, 1)
        self._total_tokens: int = 0
        self._history: list[int] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def track(self, tokens: int) -> None:
        """Record the tokens consumed this time."""
        if tokens < 0:
            raise ValueError(f"Token consumption cannot be negative: {tokens}")
        self._total_tokens += tokens
        self._history.append(tokens)

    def get_usage(self) -> int:
        """Return the current cumulative token consumption."""
        return self._total_tokens

    def get_usage_ratio(self) -> float:
        """Return the share of the budget consumed so far [0.0, 1.0+]."""
        return self._total_tokens / self._budget_limit

    def is_over_budget(self) -> bool:
        """Whether the budget limit has been exceeded."""
        return self._total_tokens >= self._budget_limit

    def is_near_budget(self, threshold: float = 0.8) -> bool:
        """Whether the budget limit is close (default 80%).

        Used to trigger compression early and avoid information loss from hard truncation.
        """
        return self.get_usage_ratio() >= threshold

    def set_budget_limit(self, new_limit: int) -> None:
        """Dynamically adjust the budget limit."""
        self._budget_limit = max(new_limit, 1)

    def reset(self) -> None:
        """Reset the cumulative counter (usually called after a replan)."""
        self._total_tokens = 0
        self._history.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> BudgetSnapshot:
        """Get the current budget snapshot."""
        return BudgetSnapshot(
            total_tokens=self._total_tokens,
            budget_limit=self._budget_limit,
            usage_ratio=self.get_usage_ratio(),
            is_over_budget=self.is_over_budget(),
        )

    def get_history(self) -> list[int]:
        """Return the history of each track() call."""
        return list(self._history)

    def __repr__(self) -> str:
        return (
            f"<BudgetTracker used={self._total_tokens}/{self._budget_limit} "
            f"ratio={self.get_usage_ratio():.2%}>"
        )
