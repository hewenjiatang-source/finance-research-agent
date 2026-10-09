#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/judge_based.py
================================================================================
Deep evaluation metrics based on LLM-as-Judge.

Suited to scenarios that need subjective expert judgment (spot checks, comparative analysis, final quality validation).
Calls src.core.judge.LLMJudge underneath, keeping a one-way dependency on the core layer.
================================================================================
"""

from __future__ import annotations

from typing import Any


class JudgeBasedMetrics:
    """Collection of research-report quality evaluation metrics (LLM Judge edition)."""

    @staticmethod
    def judge_score(
        report: str,
        query: str,
        ground_truth: dict[str, Any] | None = None,
        backend: str = "claude",
    ) -> dict[str, Any]:
        """
        Use MiMo 2.5 Pro as the Judge to score the report across multiple dimensions.

        When rule-based metrics are insufficient for precise evaluation, call this method to obtain the LLM's qualitative judgment.
        The returned structure includes factual accuracy, logical consistency, citation quality, and overall confidence.

        Args:
            report: Generated research report text.
            query: Original research question.
            ground_truth: Key facts expected to be included (optional).
            backend: Judge backend name.

        Returns:
            Dict containing per-dimension scores and rationales.
        """
        from src.core.judge import LLMJudge

        judge = LLMJudge(backend=backend)
        return judge.score_single(report, query, ground_truth)
