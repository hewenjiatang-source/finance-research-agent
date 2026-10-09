#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/judge.py
================================================================================
MiMo 2.5 Pro LLM-as-Judge unified interface.

Public interface:
    - LLMJudge.score_single(report, query, ground_truth=None) -> dict
    - LLMJudge.compare_two(report_a, report_b, query) -> dict
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("judge")


class LLMJudge:
    """LLM-as-Judge reviewer (originally built on MiMo 2.5 Pro; any backend works)."""

    def __init__(self, backend: str = "claude") -> None:
        """
        Args:
            backend: judge backend name, matching a backend registered in ModelRouter.
        """
        self.backend = backend
        self._policy = None

    def _get_policy(self):
        """Lazily initialize the policy, to avoid triggering network requests at import time."""
        if self._policy is None:
            from src.models.model_router import ModelRouter
            self._policy = ModelRouter.create_backend(self.backend)
        return self._policy

    # -----------------------------------------------------------------------
    # Deep scoring of a single report
    # -----------------------------------------------------------------------
    def score_single(
        self,
        report: str,
        query: str,
        ground_truth: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Score a single report on 5 dimensions.

        Return structure:
            {
              "overall": {"score": 7.5, "reason": "..."},
              "dimensions": {
                "factual_accuracy": {"score": 8, "reason": "..."},
                "logical_consistency": {"score": 7, "reason": "..."},
                "citation_quality": {"score": 8, "reason": "..."},
                "comprehensiveness": {"score": 7, "reason": "..."}
              },
              "average": 7.5,
              "judge_backend": "claude"
            }
        """
        gt_section = ""
        if ground_truth:
            gt_lines = "\n".join(f"- {k}: {v}" for k, v in ground_truth.items())
            gt_section = f"Key facts expected to be included:\n{gt_lines}\n"

        prompt = f"""You are a rigorous research-report reviewer. Please score the following research report.

Research question: {query}

{gt_section}
--- Research report ---
{report[:4000]}

Score on the following dimensions (0-10 each, 10 is the highest):
1. factual_accuracy: factual accuracy (are numbers, dates, person and organization names correct)
2. logical_consistency: logical consistency (is the argument self-consistent, any contradictions)
3. citation_quality: citation quality (are the sources reliable, are the citations sufficient)
4. comprehensiveness: coverage (does it answer every sub-dimension of the research question)
5. overall: overall quality

Output strict JSON:
{{
  "factual_accuracy": {{"score": number, "reason": "short reason"}},
  "logical_consistency": {{"score": number, "reason": "short reason"}},
  "citation_quality": {{"score": number, "reason": "short reason"}},
  "comprehensiveness": {{"score": number, "reason": "short reason"}},
  "overall": {{"score": number, "reason": "short reason"}}
}}"""

        try:
            policy = self._get_policy()
            messages = [
                {"role": "system", "content": "You are a research-report review expert. You must output valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ]
            resp = policy(messages)
            content = resp.get("content", "")

            result = self._extract_json(content)
            if result:
                scores = [
                    v["score"]
                    for v in result.values()
                    if isinstance(v, dict) and "score" in v
                ]
                avg = sum(scores) / len(scores) if scores else 0.0
                dimensions = {k: v for k, v in result.items() if k != "overall"}
                overall = result.get("overall", {"score": avg, "reason": ""})
                return {
                    "overall": overall,
                    "dimensions": dimensions,
                    "average": avg,
                    "judge_backend": self.backend,
                }
        except Exception as e:
            logger.warning(f"Judge single-report scoring failed: {e}")
            return {"error": str(e), "judge_backend": self.backend}

        return {"error": "Unable to parse the Judge output", "judge_backend": self.backend}

    # -----------------------------------------------------------------------
    # Head-to-head comparison of two reports
    # -----------------------------------------------------------------------
    def compare_two(
        self,
        report_a: str,
        report_b: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Run a head-to-head comparison score on two reports.

        Return structure:
            {
              "comprehensiveness": {"A": 4, "B": 5, "reason": "..."},
              "accuracy": {"A": 3, "B": 4, "reason": "..."},
              "structure": {"A": 4, "B": 4, "reason": "..."},
              "sources": {"A": 3, "B": 5, "reason": "..."},
              "judge_backend": "claude"
            }
        """
        prompt = f"""You are a rigorous research-report reviewer. Please compare the following two research reports and score them on 4 dimensions (1-5 each).

Research question: {query}

--- Report A ---
{report_a[:3000]}

--- Report B ---
{report_b[:3000]}

Scoring criteria:
- comprehensiveness: does the report answer every sub-dimension of the research question
- accuracy: are the facts and data in the report correct, any obvious hallucinations
- structure: is the report well organized and the logic smooth
- sources: does the report cite reliable sources, are the citations sufficient

Output strict JSON:
{{
  "comprehensiveness": {{"A": number, "B": number, "reason": "short reason"}},
  "accuracy": {{"A": number, "B": number, "reason": "short reason"}},
  "structure": {{"A": number, "B": number, "reason": "short reason"}},
  "sources": {{"A": number, "B": number, "reason": "short reason"}}
}}"""

        try:
            policy = self._get_policy()
            messages = [
                {"role": "system", "content": "You are a research-report review expert. You must output valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ]
            resp = policy(messages)
            content = resp.get("content", "")

            result = self._extract_json(content)
            if result:
                result["judge_backend"] = self.backend
                return result
        except Exception as e:
            logger.warning(f"Judge head-to-head scoring failed: {e}")
            return {"error": str(e), "judge_backend": self.backend}

        return {"error": "Unable to parse the Judge output", "judge_backend": self.backend}

    # -----------------------------------------------------------------------
    # Internal helper: JSON extraction
    # -----------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from text, with several fallback strategies."""
        # Strategy 1: find the outermost {} directly
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Strategy 2: find a ```json ... ``` code block
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: repair common JSON errors, then parse
        cleaned = text.strip()
        # strip possible Markdown markers
        cleaned = re.sub(r"^```.*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        # fix single quotes
        cleaned = cleaned.replace("'", '"')
        # fix trailing commas
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return None
