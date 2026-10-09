"""Orchestrator subpackage: core of the M1 orchestration layer."""
from __future__ import annotations

# Note: to avoid circular imports, orchestrator.Orchestrator is not imported in __init__
# Import it directly from the submodule:
#   from orchestrator.schemas import SubTask, RunConfig
#   from orchestrator.orchestrator import Orchestrator
#   from orchestrator.agent_pool import AgentPool

__all__ = []
