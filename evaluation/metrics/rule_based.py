#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/rule_based.py
================================================================================
Lightweight evaluation metrics based on rules/statistics.

Suited to batch runs, CI/CD, ablation experiments, and other scenarios needing fast, free, reproducible scoring.
================================================================================
"""

from __future__ import annotations

import math
import re
from typing import Any


class RuleBasedMetrics:
    """Collection of research-report quality evaluation metrics (rule-based edition)."""

    # -----------------------------------------------------------------------
    # 1. Factual Accuracy -- string-matching version (fast but crude)
    # -----------------------------------------------------------------------
    @staticmethod
    def fact_accuracy(report: str, ground_truth: dict[str, Any] | None = None) -> float:
        """
        Compute how well the key facts in the report match the ground_truth.

        The current implementation uses a simple heuristic: the proportion of ground_truth key phrases contained in the report.
        Returns 0.0 if there is no ground_truth (an external Judge LLM must supplement the evaluation).
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
    # 1b. Semantic Factual Accuracy -- enhanced version
    # -----------------------------------------------------------------------
    @staticmethod
    def semantic_fact_accuracy(
        report: str,
        ground_truth: dict[str, Any] | None = None,
        threshold: float = 0.65,
    ) -> float:
        """
        Factual accuracy verification based on embedding semantic similarity.

        Improvements (compared with string matching):
        1. Encode each ground_truth key + description as a semantic vector
        2. Split the report into sentence chunks and encode each
        3. Compute the cosine similarity between each ground_truth entry and the most similar chunk in the report
        4. A fact is judged "covered" only if the similarity exceeds the threshold (default 0.65)

        This avoids false positives such as "GPT-4o was released by Google", where keywords match but the semantics are wrong.

        Args:
            report: Full text of the research report
            ground_truth: Dict of expected facts {key: description}
            threshold: Semantic similarity threshold, 0-1

        Returns:
            Coverage rate from 0.0 to 1.0
        """
        if not ground_truth:
            return 0.0

        import numpy as np
        from src.memory.embedder import Embedder

        embedder = Embedder()

        # Split the report into sentence chunks (so a long report does not drown out short facts)
        chunks = [s.strip() for s in re.split(r"[。！？\n]", report) if len(s.strip()) > 10]
        if not chunks:
            return 0.0

        # Batch-encode chunks (SentenceTransformer supports batching)
        try:
            chunk_embs = np.array(embedder._load_model().encode(chunks, normalize_embeddings=True))
        except Exception:
            # fallback: encode one by one
            chunk_embs = np.array([embedder.encode(c) for c in chunks])

        matched = 0
        for key_fact, expected_desc in ground_truth.items():
            # Combine key + description as the semantic query
            fact_text = f"{key_fact}：{expected_desc}"
            fact_emb = np.array(embedder.encode(fact_text))

            # Compute cosine similarity against all chunks
            sims = chunk_embs.dot(fact_emb)
            max_sim = float(np.max(sims)) if sims.size > 0 else 0.0

            if max_sim > threshold:
                matched += 1

        return matched / len(ground_truth)

    # -----------------------------------------------------------------------
    # 2. Hallucination Rate
    # -----------------------------------------------------------------------
    @staticmethod
    def hallucination_rate(report: str) -> float:
        """
        Estimate the proportion of potentially hallucinated content in the report.

        Current heuristic strategy:
        - Detect uncited numeric claims (number + unit).
        - Detect absolute statements lacking sources ("absolutely", "without a doubt", etc.).
        - Detect common model hallucination patterns ("as far as I know", "research shows" without a specific citation).

        Returns:
            0.0 to 1.0; higher means greater hallucination risk.
        """
        if not report:
            return 1.0

        sentences = re.split(r"[。！？\n]", report)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 1.0

        hallucination_indicators = [
            r"\d+[\d,]*\.?\d*\s*(%|倍|个|人|元|美元|亿|万)",  # isolated number with a unit
            r"毫无疑问|绝对|必然|一定|众所周知",
            r"据我所知|据了解|研究显示[^【\[（(]",  # vague citation opener
        ]

        suspicious_count = 0
        for sentence in sentences:
            # If the sentence has no citation marker, check whether it contains hallucination features
            if not re.search(r"[\[【（(].*?[\]）)]", sentence):
                for pattern in hallucination_indicators:
                    if re.search(pattern, sentence):
                        suspicious_count += 1
                        break

        return min(1.0, suspicious_count / max(len(sentences), 1))

    # -----------------------------------------------------------------------
    # 3. Citation Coverage
    # -----------------------------------------------------------------------
    @staticmethod
    def citation_coverage(report: str) -> float:
        """
        Compute the proportion of paragraphs in the report that contain a cited source.

        Citation marker forms:
        - [N] or [来源: ...] ("来源" = "source")
        - 【来源: ...】
        - (来源: ...)
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
    # 4. Logical Consistency
    # -----------------------------------------------------------------------
    @staticmethod
    def logical_consistency(report: str) -> float:
        """
        Estimate the report's logical consistency score.

        Current heuristic strategy:
        - Detect obvious self-contradicting keyword pairs ("is" vs "is not" in the same context).
        - Check whether logical connectives are used reasonably (is there a premise before "therefore" / "however").
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
                    # Stricter check: ensure no negation word separates them
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
    # 6. Composite Score
    # -----------------------------------------------------------------------
    @staticmethod
    def composite_score(
        metrics: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> float:
        """
        Compute a weighted composite score from multi-dimensional metrics and weights.

        Default weights are aligned with the Red Agent's five dimensions:
        - factual_accuracy: 0.25
        - logical_consistency: 0.20
        - citation_coverage: 0.20
        - bias (1 - hallucination_rate as a proxy): 0.20
        - comprehensiveness: 0.15
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
    # 7. Efficiency
    # -----------------------------------------------------------------------
    @staticmethod
    def efficiency_score(
        num_turns: int,
        target_turns: float = 8.0,
        slope: float = 0.5,
        max_bonus: float = 0.5,
    ) -> float:
        """
        Sigmoid-based efficiency reward score.

        Formula: max_bonus * sigmoid(slope * (target_turns - num_turns))
        """
        sigmoid = 1.0 / (1.0 + math.exp(-slope * (target_turns - num_turns)))
        return max_bonus * sigmoid
