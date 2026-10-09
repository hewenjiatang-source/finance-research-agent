"""
Deep Research Agent — core data structure definitions (shared M1/M2 schema)

All data structures passed between modules are defined here, for type consistency and maintainability.
Uses the | union type syntax of Python 3.10+.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


__all__ = [
    "OrchestratorState",
    "TaskType",
    "AgentStatus",
    "SubTask",
    "AgentResult",
    "ResearchReport",
    "RunConfig",
]


# ============================================================================
# Enum definitions
# ============================================================================

class OrchestratorState(Enum):
    """M1 orchestration layer 9-state state machine.

    Normal flow: IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
    Exception flow:
      - partial failure → REPLANNING (incremental re-planning) → DISPATCHING
      - global failure / max re-plan count exceeded → FAILED
    """
    IDLE = "idle"
    PLANNING = "planning"
    DISPATCHING = "dispatching"
    COLLECTING = "collecting"
    SYNTHESIZING = "synthesizing"
    ADVERSARIAL = "adversarial"
    REPLANNING = "replanning"
    DONE = "done"
    FAILED = "failed"


class TaskType(Enum):
    """Task type of a sub-task, deciding which kind of Agent runs it."""
    SEARCH = "search"
    ANALYZE = "analyze"
    VERIFY = "verify"


class AgentStatus(Enum):
    """Execution result status of a single sub-task."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# Dataclass definitions
# ============================================================================

@dataclass
class SubTask:
    """Atomic task unit produced by the planner.

    Attributes:
        task_id: globally unique id, used for DAG dependency references.
        task_type: task type, deciding which Agent runs it.
        description: natural-language description, the instruction passed to the Agent.
        dependencies: list of task_ids this task depends on; they must finish before this task runs.
        context_keys: names of context keys to read from the shared Memory.
        timeout_seconds: single-task timeout threshold (seconds).
        priority: priority, a smaller number means higher priority.
        expected_type: expected result type, helps the Agent adjust its output format.
        search_hints: extra keyword hints for search-type tasks.
    """
    task_id: str
    task_type: TaskType
    description: str
    dependencies: list[str] = field(default_factory=list)
    context_keys: list[str] = field(default_factory=list)
    timeout_seconds: int = 300
    priority: int = 1
    expected_type: str = "factual"  # factual | analytical | comparative | temporal
    search_hints: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result after an Agent executes a SubTask.

    Attributes:
        task_id: task_id of the corresponding SubTask.
        status: execution status (success / failed / timeout).
        output: the actual output; its type depends on the task (str | dict | list).
        trajectory: multi-turn interaction trajectory, for logging and later analysis.
        token_usage: tokens consumed by this task.
        confidence: result confidence [0.0, 1.0].
    """
    task_id: str
    status: AgentStatus
    output: Any = None
    trajectory: list[dict] = field(default_factory=list)
    token_usage: int = 0
    confidence: float = 0.0


@dataclass
class ResearchReport:
    """The research report finally delivered to the user.

    Attributes:
        query: the original research question.
        content: report body (Markdown).
        sources: list of cited sources, each with url/title/snippet.
        confidence: overall confidence.
        num_searches: number of search / analysis rounds actually run.
        num_replan: number of re-plans.
        adversarial_rounds: number of adversarial verification rounds.
        final_score: final composite score (written by the external evaluation module).
        evidence: snapshot of the evidence ledger (finance scenario). Each item has id/kind/url/title/text;
                  the [n] citations in the report map to evidence[n-1], so citation verification / data-accuracy evaluation can replay offline.
    """
    query: str
    content: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    num_searches: int = 0
    num_replan: int = 0
    adversarial_rounds: int = 0
    final_score: float = 0.0
    evidence: list[dict] = field(default_factory=list)


@dataclass
class RunConfig:
    """Global configuration of a single run.

    Attributes:
        max_concurrent: maximum number of concurrent sub-agents.
        global_timeout_seconds: global hard timeout (seconds).
        max_replan_rounds: maximum number of re-planning rounds.
        max_sub_questions: maximum number of sub-questions per planning pass.
        enable_adversarial: whether to enable adversarial verification.
        enable_evolution: whether to enable self-evolution (reserved M6 interface).
    """
    max_concurrent: int = 5
    global_timeout_seconds: int = 600
    max_replan_rounds: int = 3
    max_sub_questions: int = 8
    enable_adversarial: bool = True
    enable_evolution: bool = False
