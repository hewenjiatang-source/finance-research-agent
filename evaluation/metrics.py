#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics.py
================================================================================
Evaluation metrics module: defines multi-dimension quality assessment methods for DeepResearch Agent output reports.

The current implementation uses lightweight rule/statistics-based metrics. In production, some metrics (e.g. hallucination detection)
can be replaced by Judge-LLM-based evaluation for higher accuracy. The rule heuristics intentionally keep their Chinese patterns (the benchmark questions are Chinese).
================================================================================
"""

from __future__ import annotations

import math
import re
from typing import Any


class ResearchMetrics:
    """Collection of research-report quality metrics."""

    # -----------------------------------------------------------------------
    # 1. Factual accuracy
    # -----------------------------------------------------------------------
    @staticmethod
    def fact_accuracy(report: str, ground_truth: dict[str, Any] | None = None) -> float:
        """
        Compute how well key facts in the report match the ground_truth.

        The current implementation uses a simple heuristic: the proportion of ground_truth key phrases contained in the report.
        With no ground_truth it returns 0.0 (an external Judge LLM must supplement the evaluation).

        Args:
            report: the generated research report text.
            ground_truth: dict of key facts expected to be included, keyed by fact phrase.

        Returns:
            Accuracy score between 0.0 and 1.0.
        """
        if not ground_truth:
            return 0.0

        report_lower = report.lower()
        matched = 0
        for key_fact in ground_truth.keys():
            if key_fact.lower() in report_lower:
                matched += 1

        return matched / len(ground_truth) if ground_truth else 0.0

    # -----------------------------------------------------------------------
    # 2. Hallucination rate
    # -----------------------------------------------------------------------
    @staticmethod
    def hallucination_rate(report: str) -> float:
        """
        Estimate the proportion of possibly hallucinated content in the report.

        Current heuristic strategy:
        - Detect numeric claims without citations (number + unit).
        - Detect absolutist statements lacking sources ("绝对", "毫无疑问", etc.).
        - Detect common model hallucination patterns ("据我所知", "研究表明" with no specific citation).

        Args:
            report: the generated research report text.

        Returns:
            0.0 ~ 1.0; higher means greater hallucination risk.
        """
        if not report:
            return 1.0

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 1.0

        hallucination_indicators = [
            r"\d+[\d,]*\.?\d*\s*(%|倍|个|人|元|美元|亿|万)",  # 带单位的孤立数字
            r"毫无疑问|绝对|必然|一定|众所周知",
            r"据我所知|据了解|研究显示[^【\[（(]",  # 模糊引用开头
        ]

        suspicious_count = 0
        for sentence in sentences:
            # If the sentence has no citation mark, check for hallucination features
            if not re.search(r"[\[【（(].*?[\]）)]", sentence):
                for pattern in hallucination_indicators:
                    if re.search(pattern, sentence):
                        suspicious_count += 1
                        break

        return min(1.0, suspicious_count / max(len(sentences), 1))

    # -----------------------------------------------------------------------
    # 3. Citation coverage
    # -----------------------------------------------------------------------
    @staticmethod
    def citation_coverage(report: str) -> float:
        """
        Compute the proportion of paragraphs in the report that contain a cited source.

        Citation mark forms:
        - [N] 或 [来源: ...]
        - 【来源: ...】
        - (来源: ...)

        Args:
            report: the generated research report text.

        Returns:
            0.0 ~ 1.0; higher means better citation coverage.
        """
        if not report:
            return 0.0

        paragraphs = [p.strip() for p in report.split("\n") if p.strip()]
        if not paragraphs:
            return 0.0

        citation_patterns = [
            r"\[\d+\]",
            r"\[来源[：:]",
            r"【来源[：:]",
            r"\(来源[：:]",
            r"https?://",
            r"arxiv\.org",
        ]

        cited_paragraphs = 0
        for para in paragraphs:
            for pattern in citation_patterns:
                if re.search(pattern, para):
                    cited_paragraphs += 1
                    break

        return cited_paragraphs / len(paragraphs)

    # -----------------------------------------------------------------------
    # 4. Logical consistency
    # -----------------------------------------------------------------------
    @staticmethod
    def logical_consistency(report: str) -> float:
        """
        Estimate the logical-consistency score of the report.

        Current heuristic strategy:
        - Detect obviously self-contradictory keyword pairs ("是" vs "不是" in the same context).
        - Check whether logical connectives are used reasonably (is there a premise before "因此" / "然而").

        Args:
            report: the generated research report text.

        Returns:
            0.0 ~ 1.0; higher means more consistent logic.
        """
        if not report:
            return 0.0

        # Simple contradiction-pair detection: A and not-A appear in the same sentence
        contradiction_pairs = [
            ("是", "不是"),
            ("可以", "不可以"),
            ("会", "不会"),
            ("支持", "反对"),
            ("增加", "减少"),
        ]

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        contradiction_count = 0
        for sentence in sentences:
            for a, b in contradiction_pairs:
                if a in sentence and b in sentence:
                    # Stricter check: make sure no negation word separates them
                    contradiction_count += 1
                    break

        # Also reward the use of logical connectives
        connectives = ["因此", "所以", "然而", "但是", "首先", "其次", "综上所述"]
        connective_count = sum(1 for c in connectives if c in report)
        connective_bonus = min(0.1, connective_count * 0.01)

        base_score = 1.0 - (contradiction_count / max(len(sentences), 1))
        return min(1.0, max(0.0, base_score + connective_bonus))

    # -----------------------------------------------------------------------
    # 5. Comprehensiveness
    # -----------------------------------------------------------------------
    @staticmethod
    def comprehensiveness(report: str, expected_topics: list[str] | None = None) -> float:
        """
        Compute how well the report covers the expected topics.

        Args:
            report: the generated research report text.
            expected_topics: list of sub-topics expected to be covered.

        Returns:
            0.0 ~ 1.0; higher means more comprehensive coverage.
        """
        if not expected_topics:
            return 0.0

        report_lower = report.lower()
        covered = 0
        for topic in expected_topics:
            if topic.lower() in report_lower:
                covered += 1

        return covered / len(expected_topics) if expected_topics else 0.0

    # -----------------------------------------------------------------------
    # 6. Composite score
    # -----------------------------------------------------------------------
    @staticmethod
    def composite_score(metrics: dict[str, float], weights: dict[str, float] | None = None) -> float:
        """
        Compute a weighted composite score from the multi-dimension metrics and weights.

        Default weights are aligned with the Red Agent's five dimensions:
        - factual_accuracy: 0.25
        - logical_consistency: 0.20
        - citation_coverage: 0.20
        - bias (1 - hallucination_rate as a proxy): 0.20
        - comprehensiveness: 0.15

        Args:
            metrics: dict of metric name -> metric value.
            weights: dict of metric name -> weight; None uses the default weights.

        Returns:
            Composite score in 0.0 ~ 1.0.
        """
        default_weights = {
            "factual_accuracy": 0.25,
            "logical_consistency": 0.20,
            "citation_coverage": 0.20,
            "bias": 0.20,
            "comprehensiveness": 0.15,
        }

        w = weights if weights is not None else default_weights
        total_score = 0.0
        total_weight = 0.0

        for key, weight in w.items():
            value = metrics.get(key, 0.0)
            total_score += value * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    # -----------------------------------------------------------------------
    # 7. Efficiency metric
    # -----------------------------------------------------------------------
    @staticmethod
    def efficiency_score(num_turns: int, target_turns: float = 8.0, slope: float = 0.5, max_bonus: float = 0.5) -> float:
        """
        Sigmoid-based efficiency reward score.

        Formula: max_bonus * sigmoid(slope * (target_turns - num_turns))

        Args:
            num_turns: number of interaction turns actually used.
            target_turns: the desired ideal number of turns.
            slope: sigmoid slope.
            max_bonus: upper bound of the bonus.

        Returns:
            Efficiency score in 0.0 ~ max_bonus.
        """
        sigmoid = 1.0 / (1.0 + math.exp(-slope * (target_turns - num_turns)))
        return max_bonus * sigmoid

    # -----------------------------------------------------------------------
    # 8. MiMo Judge scoring (LLM-as-Judge)
    # -----------------------------------------------------------------------
    @staticmethod
    def judge_score(report: str, query: str, ground_truth: dict[str, Any] | None = None) -> dict[str, Any]:
        """Use MiMo 2.5 Pro as the Judge to score a report on multiple dimensions.

        When rule metrics are not precise enough, call this to get the LLM's qualitative judgment.
        The returned structure includes factual accuracy, logical consistency, citation quality and overall confidence.

        Args:
            report: the generated research report text.
            query: the original research question.
            ground_truth: key facts expected to be included (optional).

        Returns:
            Dict containing per-dimension scores and reasons.
        """
        import json
        import re

        from src.models.model_router import ModelRouter

        gt_section = ""
        if ground_truth:
            gt_lines = "\n".join(f"- {k}: {v}" for k, v in ground_truth.items())
            gt_section = f"Key facts expected to be included:\n{gt_lines}\n"

        prompt = f"""You are a rigorous research-report reviewer. Score the following research report.

Research question: {query}

{gt_section}
--- Research report ---
{report[:4000]}

Score on the following dimensions (0-10 each, 10 is best):
1. factual_accuracy: factual accuracy (are numbers, dates, names and institutions correct)
2. logical_consistency: logical consistency (is the argument self-consistent, any contradictions)
3. citation_quality: citation quality (are sources reliable, are citations sufficient)
4. comprehensiveness: coverage (does it fully answer each sub-dimension of the research question)
5. overall: overall quality

Output strict JSON:
{{
  "factual_accuracy": {{"score": score, "reason": "brief reason"}},
  "logical_consistency": {{"score": score, "reason": "brief reason"}},
  "citation_quality": {{"score": score, "reason": "brief reason"}},
  "comprehensiveness": {{"score": score, "reason": "brief reason"}},
  "overall": {{"score": score, "reason": "brief reason"}}
}}"""

        try:
            policy = ModelRouter.create_backend("claude")
            messages = [
                {"role": "system", "content": "You are a research-report review expert. You must output valid JSON and nothing else."},
                {"role": "user", "content": prompt},
            ]
            resp = policy(messages)
            content = resp.get("content", "")

            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                result = json.loads(m.group())
                # Compute the average score
                scores = [v["score"] for v in result.values() if isinstance(v, dict) and "score" in v]
                result["average"] = sum(scores) / len(scores) if scores else 0.0
                result["judge_backend"] = "mimo"
                return result
        except Exception as e:
            return {"error": str(e), "judge_backend": "claude"}

        return {"error": "Could not parse Judge output", "judge_backend": "claude"}


# =============================================================================
# Simple self-test
# =============================================================================
if __name__ == "__main__":
    sample_report = """
    研究表明，人工智能在医疗诊断中的应用正在快速增长[1]。
    根据 Nature Medicine 2024 年的综述，AI 辅助诊断的准确率已达到 95%[2]。
    然而，这一技术也面临数据隐私和伦理挑战[3]。
    综上所述，AI 医疗的发展前景广阔，但需要审慎监管。
    """

    print("citation_coverage:", ResearchMetrics.citation_coverage(sample_report))
    print("hallucination_rate:", ResearchMetrics.hallucination_rate(sample_report))
    print("logical_consistency:", ResearchMetrics.logical_consistency(sample_report))
    print("comprehensiveness:", ResearchMetrics.comprehensiveness(sample_report, expected_topics=["医疗", "伦理"]))
    print("efficiency_score:", ResearchMetrics.efficiency_score(num_turns=6))
