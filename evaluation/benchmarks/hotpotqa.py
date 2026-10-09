#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/benchmarks/hotpotqa.py
================================================================================
HotpotQA multi-hop QA evaluation adapter.

HotpotQA is a classic multi-hop QA dataset that requires models to connect multiple pieces of information
through multi-step reasoning/retrieval in order to answer a question. This adapter converts it into the DeepResearch Agent input format,
and computes evaluation metrics such as pass@k, exact match, and F1.
================================================================================
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from typing import Any


class HotpotQABenchmark:
    """HotpotQA evaluation set loader and evaluator."""

    # Built-in small-scale test data (fallback when HuggingFace is unavailable)
    _MOCK_DATA: list[dict[str, Any]] = [
        {
            "question": "《红楼梦》的作者曹雪芹生活在哪个朝代？",
            "answer": "清朝",
            "type": "bridge",
            "level": "easy",
            "context": [["红楼梦", ["《红楼梦》是中国古典小说巅峰之作。"]], ["曹雪芹", ["曹雪芹，清代小说家。"]]],
        },
        {
            "question": "2024 年诺贝尔物理学奖授予了哪位科学家，他以什么研究闻名？",
            "answer": "John Hopfield 和 Geoffrey Hinton，以神经网络和机器学习的基础性发现",
            "type": "bridge",
            "level": "medium",
            "context": [["诺贝尔物理学奖", ["2024 年诺贝尔物理学奖授予机器学习领域。"]], ["Hinton", ["Geoffrey Hinton 是深度学习先驱。"]]],
        },
        {
            "question": "OpenAI 的 GPT 系列模型和 Google 的 Gemini 模型分别由哪家公司开发？",
            "answer": "GPT 由 OpenAI 开发，Gemini 由 Google DeepMind 开发",
            "type": "comparison",
            "level": "easy",
            "context": [["OpenAI", ["OpenAI 是人工智能研究公司。"]], ["Google", ["Google DeepMind 开发了 Gemini 模型。"]]],
        },
        {
            "question": "NVIDIA 的 H100 芯片采用什么制程工艺，主要用于什么场景？",
            "answer": "4 纳米制程，主要用于 AI 训练和推理",
            "type": "bridge",
            "level": "medium",
            "context": [["NVIDIA", ["NVIDIA 是全球领先的 GPU 制造商。"]], ["H100", ["H100 采用台积电 4nm 工艺。"]]],
        },
        {
            "question": "Transformer 架构中的 Attention 机制是谁提出的，发表于哪一年？",
            "answer": "Vaswani 等人，2017 年",
            "type": "bridge",
            "level": "medium",
            "context": [["Transformer", ["Transformer 架构发表于 2017 年。"]], ["Attention", ["Attention Is All You Need 由 Google 团队发表。"]]],
        },
    ]

    def __init__(self, data_path: str | None = None, split: str = "validation", use_mock: bool = False) -> None:
        """
        Initialize the HotpotQA evaluation set.

        Args:
            data_path: Path to the HotpotQA data file (JSON format).
                       If None, try loading from HuggingFace datasets.
            split: Data split, usually "train" / "validation" / "test".
            use_mock: If True, use the built-in test data (for pipeline verification; no download needed).
        """
        self.split = split
        self.data: list[dict[str, Any]] = []

        if use_mock:
            self.data = self._MOCK_DATA
            print(f"[HotpotQA] Using built-in mock data: {len(self.data)} samples")
            return

        if data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.data = raw if isinstance(raw, list) else raw.get("data", [])
        else:
            # Try loading via the datasets library (with timeout control)
            try:
                import os as _os
                # Set the HuggingFace download timeout
                _os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
                from datasets import load_dataset

                ds = load_dataset("hotpot_qa", "distractor", split=split)
                self.data = list(ds)
                print(f"[HotpotQA] Loaded {len(self.data)} samples from HuggingFace")
            except Exception as e:
                print(f"[HotpotQA] Unable to load from HuggingFace: {e}")
                print(f"[HotpotQA] Suggestions: 1) check your network connection; 2) or download the data manually and pass it via data_path;")
                print(f"[HotpotQA] 3) or use the --use_mock flag to enable the built-in test data")
                self.data = []

    # -----------------------------------------------------------------------
    # Data format conversion
    # -----------------------------------------------------------------------
    def to_research_format(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a single HotpotQA sample into an input format compatible with ResearchReport.

        Args:
            sample: Raw HotpotQA sample.

        Returns:
            Dict containing query, context, and expected_answer.
        """
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # The context in HotpotQA is usually a list of (title, sentences)
        contexts = sample.get("context", [])
        context_text = ""
        if contexts and isinstance(contexts[0], (list, tuple)) and len(contexts[0]) == 2:
            # Standard format: [(title, [sent1, sent2, ...]), ...]
            for title, sentences in contexts:
                context_text += f"\n## {title}\n" + " ".join(sentences)
        elif isinstance(contexts, str):
            context_text = contexts

        return {
            "query": question,
            "context": context_text.strip(),
            "expected_answer": answer,
            "type": sample.get("type", "bridge"),  # bridge / comparison
            "level": sample.get("level", "medium"),  # easy / medium / hard
        }

    def get_samples(self, n: int | None = None, shuffle: bool = False) -> list[dict[str, Any]]:
        """
        Get the list of converted samples.

        Args:
            n: Return the first n samples; None means all.
            shuffle: Whether to shuffle the order randomly.

        Returns:
            List of converted samples.
        """
        samples = [self.to_research_format(s) for s in self.data]
        if shuffle:
            random.shuffle(samples)
        if n is not None:
            samples = samples[:n]
        return samples

    # -----------------------------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------------------------
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize an answer: lowercase, strip punctuation, remove articles."""
        text = text.lower().strip()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = " ".join(text.split())
        return text

    @staticmethod
    def exact_match(pred: str, gold: str) -> bool:
        """Compute exact match after normalization."""
        return HotpotQABenchmark.normalize_answer(pred) == HotpotQABenchmark.normalize_answer(gold)

    @staticmethod
    def f1_score(pred: str, gold: str) -> float:
        """Compute the token-level F1 score."""
        pred_tokens = HotpotQABenchmark.normalize_answer(pred).split()
        gold_tokens = HotpotQABenchmark.normalize_answer(gold).split()

        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def pass_at_k(preds: list[str], gold: str, k: int = 1) -> bool:
        """
        Determine whether any of the first k predictions is correct (exact match).

        Args:
            preds: List of k candidate answers generated by the model.
            gold: Gold answer.
            k: Number of candidates considered.

        Returns:
            Whether any candidate hits.
        """
        for pred in preds[:k]:
            if HotpotQABenchmark.exact_match(pred, gold):
                return True
        return False

    # -----------------------------------------------------------------------
    # Deep-research evaluation: treat HotpotQA as research queries and evaluate full-report quality
    # -----------------------------------------------------------------------
    @staticmethod
    def gold_entity_coverage(report: str, gold_answer: str) -> float:
        """
        Check whether the entities/keywords in the gold answer are covered in the research report.

        Strategy:
        1. Split gold_answer into tokens (removing stopwords)
        2. Check whether each token appears in the report
        3. Return the coverage rate
        """
        if not gold_answer or not report:
            return 0.0

        stopwords = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "of", "for", "with", "is", "was", "are", "were", "be", "been", "by"}
        gold_tokens = [t for t in HotpotQABenchmark.normalize_answer(gold_answer).split() if t not in stopwords and len(t) > 1]
        if not gold_tokens:
            return 0.0

        report_lower = report.lower()
        covered = sum(1 for t in gold_tokens if t in report_lower)
        return covered / len(gold_tokens)

    @staticmethod
    def semantic_gold_coverage(report: str, gold_answer: str, threshold: float = 0.60) -> float:
        """
        Use embedding semantic similarity to assess how well the gold answer is covered by the report.

        Encode gold_answer and the report separately and compute their similarity.
        If the gold answer is short, compare it directly with the whole report;
        if it is long, compare segment by segment and average.
        """
        if not gold_answer or not report:
            return 0.0

        from src.memory.embedder import Embedder
        import numpy as np

        embedder = Embedder()
        gold_emb = np.array(embedder.encode(gold_answer))

        # Split the report into chunks and compare each with gold_answer
        chunks = [s.strip() for s in re.split(r"[。！？\n]", report) if len(s.strip()) > 10]
        if not chunks:
            return 0.0

        try:
            chunk_embs = np.array(embedder._load_model().encode(chunks, normalize_embeddings=True))
        except Exception:
            chunk_embs = np.array([embedder.encode(c) for c in chunks])

        sims = chunk_embs.dot(gold_emb)
        # Return the proportion of chunks above the threshold (measures how many passages in the report are semantically related to the answer)
        above_threshold = np.sum(sims > threshold)
        return float(above_threshold / len(chunks)) if len(chunks) > 0 else 0.0

    def evaluate_report(
        self,
        report: str,
        gold_answer: str,
    ) -> dict[str, float]:
        """
        Perform a deep evaluation of a single research report (based on the HotpotQA gold answer).

        Returns:
            - gold_entity_coverage: gold answer entity coverage
            - semantic_gold_coverage: semantic coverage
            - report_length: report length in characters (efficiency reference)
        """
        return {
            "gold_entity_coverage": self.gold_entity_coverage(report, gold_answer),
            "semantic_gold_coverage": self.semantic_gold_coverage(report, gold_answer),
            "report_length": len(report),
        }

    # -----------------------------------------------------------------------
    # Batch evaluation
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        predictions: list[dict[str, Any]],
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Batch-evaluate prediction results.

        Args:
            predictions: List of items each containing {"query_id": ..., "prediction": ..., "gold": ...}.
                         If a "report" field is included, deep-research metrics are computed as well.
            metrics: List of metrics to compute; default ["em", "f1", "pass@1"].

        Returns:
            Dict mapping metric name -> average value.
        """
        if metrics is None:
            metrics = ["em", "f1", "pass@1"]

        total = len(predictions)
        if total == 0:
            return {m: 0.0 for m in metrics}

        em_sum = 0.0
        f1_sum = 0.0
        pass1_sum = 0.0
        entity_cov_sum = 0.0
        sem_cov_sum = 0.0

        for item in predictions:
            pred = item.get("prediction", "")
            gold = item.get("gold", "")

            if "em" in metrics and HotpotQABenchmark.exact_match(pred, gold):
                em_sum += 1.0
            if "f1" in metrics:
                f1_sum += HotpotQABenchmark.f1_score(pred, gold)
            if "pass@1" in metrics:
                pass1_sum += 1.0 if HotpotQABenchmark.exact_match(pred, gold) else 0.0

            # Deep-research metrics (if a full report is provided)
            report = item.get("report", "")
            if report:
                depth_metrics = self.evaluate_report(report, gold)
                entity_cov_sum += depth_metrics["gold_entity_coverage"]
                sem_cov_sum += depth_metrics["semantic_gold_coverage"]

        results: dict[str, float] = {}
        if "em" in metrics:
            results["exact_match"] = em_sum / total
        if "f1" in metrics:
            results["f1"] = f1_sum / total
        if "pass@1" in metrics:
            results["pass@1"] = pass1_sum / total
        if entity_cov_sum > 0:
            results["gold_entity_coverage"] = entity_cov_sum / total
        if sem_cov_sum > 0:
            results["semantic_gold_coverage"] = sem_cov_sum / total

        return results


# =============================================================================
# Simple self-test
# =============================================================================
if __name__ == "__main__":
    bench = HotpotQABenchmark()
    print(f"Samples loaded: {len(bench.data)}")

    # Mock predictions
    preds = [
        {"query_id": 0, "prediction": "Shanghai", "gold": "Shanghai"},
        {"query_id": 1, "prediction": "Beijing", "gold": "Shanghai"},
    ]
    print("Evaluation results:", bench.evaluate(preds))
