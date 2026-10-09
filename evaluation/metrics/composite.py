#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/composite.py
================================================================================
Composite score computation utility.

Aggregates rule-based metrics and Judge metrics into a single composite score.
================================================================================
"""

from __future__ import annotations

from typing import Any


def compute_composite_score(
    rule_metrics: dict[str, float] | None = None,
    judge_result: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Composite scoring: combines rule-based metrics and Judge metrics.

    Args:
        rule_metrics: Dict of rule-based metrics (e.g., factual_accuracy, citation_coverage).
        judge_result: Judge scoring result (e.g., the return value of LLMJudge.score_single).
        weights: Custom weights. By default rule-based metrics 60%, Judge metrics 40%.

    Returns:
        Dict containing the composite score and per-dimension details.
    """
    default_weights = {
        "rule": 0.6,
        "judge": 0.4,
    }
    w = weights if weights is not None else default_weights

    rule_score = 0.0
    if rule_metrics:
        # Rule-based metrics are already 0-1 scores; take a weighted average directly
        rule_vals = [v for v in rule_metrics.values() if isinstance(v, (int, float))]
        rule_score = sum(rule_vals) / len(rule_vals) if rule_vals else 0.0

    judge_score = 0.0
    judge_dims = {}
    if judge_result:
        # Judge results are on a 0-10 scale; normalize to 0-1
        dims = judge_result.get("dimensions", {})
        judge_dims = {
            k: v["score"] / 10.0
            for k, v in dims.items()
            if isinstance(v, dict) and "score" in v
        }
        judge_score = sum(judge_dims.values()) / len(judge_dims) if judge_dims else 0.0

    composite = w.get("rule", 0.6) * rule_score + w.get("judge", 0.4) * judge_score

    return {
        "composite_score": round(composite, 4),
        "rule_score": round(rule_score, 4),
        "judge_score": round(judge_score, 4),
        "rule_metrics": rule_metrics or {},
        "judge_dimensions": judge_dims,
    }
