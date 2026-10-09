#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/report.py
================================================================================
Evaluation report generator: aggregates multi-dimension evaluation results into a structured report.
================================================================================
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class EvaluationReport:
    """Unified evaluation report container."""

    def __init__(self, name: str, num_questions: int = 0) -> None:
        self.name = name
        self.num_questions = num_questions
        self.timestamp = datetime.now().isoformat()
        self.details: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

    def add_detail(self, detail: dict[str, Any]) -> None:
        """Add one evaluation detail record."""
        self.details.append(detail)

    def set_summary(self, summary: dict[str, Any]) -> None:
        """Set the summary statistics."""
        self.summary = summary

    def to_dict(self) -> dict[str, Any]:
        """Export as a dict."""
        return {
            "evaluation_name": self.name,
            "timestamp": self.timestamp,
            "num_questions": self.num_questions,
            "summary": self.summary,
            "details": self.details,
        }

    def save(self, output_dir: str, filename: str | None = None) -> str:
        """Save as a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return filepath

    def to_markdown(self) -> str:
        """Generate a Markdown summary."""
        lines = [
            f"# {self.name}",
            "",
            f"- **Evaluation time**: {self.timestamp}",
            f"- **Number of questions**: {self.num_questions}",
            "",
            "## Summary",
            "",
        ]
        for key, value in self.summary.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        lines.append("## Details")
        lines.append("")
        for d in self.details:
            lines.append(f"### {d.get('question_id', 'unknown')}")
            for k, v in d.items():
                if k != "question_id":
                    lines.append(f"- {k}: {v}")
            lines.append("")
        return "\n".join(lines)
