"""
M6 self-evolution engine — Trajectory collector

TrajectoryCollector gathers the full execution trajectory of the DeepResearch Agent and converts it into
the format veRL training needs. It is the adapter layer between the Solver (DeepResearch Agent) and the training framework.

Design decisions:
1. Collected content includes: query, report, multi-turn interaction trajectory, search count, replan count, etc.
2. The to_verl_format method reuses project one's parquet-building logic and outputs the standard format.
3. Batch collection is supported to ease building training datasets later.
"""
from __future__ import annotations

import json
from typing import Any

from src.orchestrator.schemas import ResearchReport


__all__ = ["TrajectoryCollector"]


class TrajectoryCollector:
    """Trajectory collector and format converter.

    Attributes:
        system_prompt: optional system-level prompt used in the veRL data format.
    """

    def __init__(self, system_prompt: str = ""):
        self.system_prompt = system_prompt

    def collect(
        self,
        query: str,
        report: ResearchReport,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Collect the full trajectory of one DeepResearch run.

        Args:
            query: the original research question.
            report: the final generated research report.
            trajectories: list of multi-turn interaction trajectories, each turn containing role/content/tool_calls, etc.

        Returns:
            Unified trajectory dict containing every field veRL needs.
        """
        return {
            "query": query,
            "report_content": report.content,
            "sources": report.sources,
            "confidence": report.confidence,
            "num_searches": report.num_searches,
            "num_replan": report.num_replan,
            "adversarial_rounds": report.adversarial_rounds,
            "final_score": report.final_score,
            "trajectories": trajectories,
            # Meta information
            "trajectory_length": len(trajectories),
            "content_length": len(report.content),
            "source_count": len(report.sources),
        }

    def to_verl_format(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert a collected trajectory into the parquet row format veRL training needs.

        Fields veRL expects (aligned with project one's scripts/11_build_grpo_parquet.py):
        - prompt: list[dict] — multi-turn conversation format, containing system + the initial user query
        - response: str — the model's full output (report content)
        - trajectories: list[dict] — multi-turn interaction trajectory (observation, action pairs)
        - metadata: dict — extra meta information

        Args:
            data: output of collect().

        Returns:
            Dict in veRL format, ready to write to parquet.
        """
        query = data.get("query", "")
        trajectories = data.get("trajectories", [])
        report_content = data.get("report_content", "")

        # Build the prompt field: system + user query
        prompt_messages: list[dict[str, str]] = []
        if self.system_prompt:
            prompt_messages.append({"role": "system", "content": self.system_prompt})
        prompt_messages.append({"role": "user", "content": query})

        # metadata contains all original fields (large fields removed to avoid parquet bloat)
        metadata = {
            "query": query,
            "num_searches": data.get("num_searches", 0),
            "num_replan": data.get("num_replan", 0),
            "adversarial_rounds": data.get("adversarial_rounds", 0),
            "final_score": data.get("final_score", 0.0),
            "trajectory_length": data.get("trajectory_length", 0),
            "source_count": data.get("source_count", 0),
            "content_length": data.get("content_length", 0),
        }

        return {
            "prompt": prompt_messages,
            "response": report_content,
            "trajectories": trajectories,
            "metadata": metadata,
        }

    def batch_to_verl(
        self, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert a batch to veRL format.

        Args:
            batch: list of collect() outputs.

        Returns:
            List of veRL-format dicts.
        """
        return [self.to_verl_format(item) for item in batch]

    def serialize(self, data: dict[str, Any]) -> str:
        """Serialize a trajectory to a JSON string (for logging or persistence)."""
        return json.dumps(data, ensure_ascii=False, indent=2)
