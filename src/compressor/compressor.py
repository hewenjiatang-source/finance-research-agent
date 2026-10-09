"""
Context Compressor module: unified compression entry point

Design decisions:
1. Three-level progressive compression: L1 relevance filtering → L2 key-sentence extraction → L3 hierarchical summary
2. Automatic triggering: the compression level is chosen from the ratio of current token usage to the budget
   - >60% budget: L1
   - >80% budget: L1+L2
   - >95% budget: L1+L2+L3
3. Budget management: available = budget - system_prompt_tokens - output_reserve
4. Quantitative evaluation interface: compression_ratio + information_retention (entity/keyword retention rate)
5. Works with SlidingWindowCompressor: try semantic compression first, fall back to sliding-window truncation last
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.memory.embedder import Embedder
from src.models.vllm_policy import VLLMPolicy
from src.compressor.extractive import ExtractiveCompressor
from src.compressor.sliding_window import SlidingWindowCompressor
from src.compressor.summarizer import LLMSummarizer
from src.utils.tracing import trace_chain

logger = logging.getLogger(__name__)

# compression level trigger thresholds
_L1_THRESHOLD = 0.60
_L2_THRESHOLD = 0.80
_L3_THRESHOLD = 0.95

# token estimation parameters
_CHARS_PER_TOKEN = 3.5
_OUTPUT_RESERVE = 2048  # tokens reserved for LLM output


class ContextCompressor:
    """
    Unified context compressor.

    Exposes a compress() interface that picks the compression level automatically and runs progressive compression.
    Also provides a quantitative evaluation interface for ablation experiments.
    """

    def __init__(
        self,
        llm_policy: VLLMPolicy,
        embedder: Optional[Embedder] = None,
        budget: int = 16000,
        output_reserve: int = _OUTPUT_RESERVE,
    ) -> None:
        """
        Initialize the context compressor.

        Args:
            llm_policy: VLLMPolicy instance (needed by L3 summaries and llm_judge)
            embedder: the vectorizer
            budget: total context token budget
            output_reserve: tokens reserved for model output
        """
        self.llm_policy = llm_policy
        self.embedder = embedder or Embedder()
        self.budget = budget
        self.output_reserve = output_reserve
        self.available_budget = budget - output_reserve

        # sub-compressors
        self.sliding = SlidingWindowCompressor(max_tokens=self.available_budget)
        self.extractive = ExtractiveCompressor(embedder=self.embedder)
        self.summarizer = LLMSummarizer(llm_policy=llm_policy)

        # accumulated statistics
        self._stats_history: list[dict[str, Any]] = []

    def calculate_tokens(self, texts: list[str]) -> int:
        """
        Estimate the total token count of a list of texts.

        Args:
            texts: list of texts

        Returns:
            estimated token count
        """
        total_chars = sum(len(t) for t in texts)
        return int(total_chars / _CHARS_PER_TOKEN)

    @trace_chain(name="compressor.compress", tags=["m3", "compressor"])
    def compress(
        self,
        texts: list[str],
        query: str = "",
        level: Optional[int] = None,
        system_prompt_tokens: int = 0,
    ) -> list[str]:
        """
        Compress a list of texts.

        Args:
            texts: original texts (one element per document / message)
            query: the current query (for L1/L2 relevance weighting)
            level: force a compression level (1/2/3); None decides automatically
            system_prompt_tokens: tokens taken by the system prompt

        Returns:
            the compressed list of texts
        """
        if not texts:
            return []

        # compute the actually available budget
        actual_budget = self.available_budget - system_prompt_tokens
        if actual_budget <= 0:
            logger.warning("Actual budget <= 0 after system prompt, forcing max compression.")
            actual_budget = self.available_budget // 2

        current_tokens = self.calculate_tokens(texts)
        usage_ratio = current_tokens / max(actual_budget, 1)

        # decide the compression level
        if level is None:
            if usage_ratio > _L3_THRESHOLD:
                level = 3
            elif usage_ratio > _L2_THRESHOLD:
                level = 2
            elif usage_ratio > _L1_THRESHOLD:
                level = 1
            else:
                # no compression needed
                self._record_stats(texts, texts, 0, current_tokens)
                return texts

        logger.info(
            f"[ContextCompressor] Triggered L{level}: "
            f"{current_tokens} tokens / {actual_budget} budget ({usage_ratio:.1%})"
        )

        compressed = list(texts)

        # L1: relevance filtering
        if level >= 1:
            compressed = self._l1_filter(compressed, query, actual_budget)

        # L2: key-sentence extraction
        if level >= 2 and compressed:
            compressed = self._l2_extract(compressed, query, actual_budget)

        # L3: hierarchical summary
        if level >= 3 and compressed:
            compressed = self._l3_summarize(compressed, query, actual_budget)

        # if still over the limit after compression, fall back to sliding-window truncation
        final_tokens = self.calculate_tokens(compressed)
        if final_tokens > actual_budget:
            logger.warning(
                f"[ContextCompressor] Still over budget after L{level}: "
                f"{final_tokens} > {actual_budget}. Falling back to sliding window."
            )
            # merge the text list into message format for truncation
            messages = [{"role": "user", "content": t} for t in compressed]
            truncated_msgs = self.sliding.compress(messages)
            compressed = [m["content"] for m in truncated_msgs]

        self._record_stats(texts, compressed, level, self.calculate_tokens(compressed))
        return compressed

    def _l1_filter(
        self,
        texts: list[str],
        query: str,
        budget: int,
    ) -> list[str]:
        """
        L1 relevance filtering: embedding cosine-similarity scoring with an adaptive threshold.

        Strategy:
        - compute the similarity of each text to the query
        - start at threshold 0.25; if still over budget after filtering, lower it step by step to 0.15
        - keep texts with similarity >= threshold
        """
        if not query or not query.strip():
            # no filtering without a query
            return texts

        query_emb = self.embedder.encode(query)
        query_vec = self._to_norm_vec(query_emb)

        scored: list[tuple[str, float]] = []
        for text in texts:
            text_emb = self.embedder.encode(text[:1000])  # use only the first 1000 chars for speed
            text_vec = self._to_norm_vec(text_emb)
            sim = float(query_vec.dot(text_vec)) if text_vec is not None else 0.0
            scored.append((text, sim))

        # adaptive threshold: start at 0.25 and decrease if not strict enough
        best_result: list[str] = []
        for threshold in [0.25, 0.20, 0.15]:
            filtered = [t for t, s in scored if s >= threshold]
            tokens = self.calculate_tokens(filtered)
            if tokens <= budget * 0.8:
                best_result = filtered
                break
            if threshold == 0.15:
                best_result = filtered

        # safety net: if filtering leaves nothing but the original had content, keep at least the single most similar one
        if not best_result and texts:
            best_text = max(scored, key=lambda x: x[1])[0]
            best_result = [best_text]
            logger.info(f"[L1] Fallback: kept top-1 similar doc.")

        logger.info(
            f"[L1] Filtered {len(texts)} -> {len(best_result)} docs, tokens={self.calculate_tokens(best_result)}"
        )
        return best_result

    def _l2_extract(
        self,
        texts: list[str],
        query: str,
        budget: int,
    ) -> list[str]:
        """
        L2 key-sentence extraction: TextRank + query-biased, dynamic keep ratio.

        Strategy:
        - compute each document's target keep ratio from the remaining budget
        - the tighter the budget, the lower the top_ratio (minimum 0.15)
        """
        current_tokens = self.calculate_tokens(texts)
        # target ratio: linear mapping; keep 30% at 80% budget use, 15% at 100%
        target_ratio = max(0.15, min(0.40, 0.50 - (current_tokens / max(budget, 1)) * 0.35))

        compressed = []
        for text in texts:
            comp = self.extractive.compress(text, query, target_ratio=target_ratio)
            compressed.append(comp)

        after_tokens = self.calculate_tokens(compressed)
        logger.info(
            f"[L2] Extractive compression: {current_tokens} -> {after_tokens} tokens, "
            f"ratio={target_ratio:.2f}"
        )
        return compressed

    def _l3_summarize(
        self,
        texts: list[str],
        query: str,
        budget: int,
    ) -> list[str]:
        """
        L3 hierarchical summary: per-document summaries → aggregate summary.

        Strategy:
        - first summarize each document on its own (length controlled)
        - then aggregate all summaries into one overview
        - finally return a single-element list (the aggregated result)
        """
        current_tokens = self.calculate_tokens(texts)
        # target length of a single-document summary
        per_doc_max = max(200, budget // max(len(texts), 1))
        summaries = []
        for text in texts:
            summary = self.summarizer.summarize_document(
                text, query, max_length=per_doc_max
            )
            summaries.append(summary)

        # aggregate summary
        aggregate_max = max(400, budget // 2)
        aggregate = self.summarizer.summarize_documents(
            summaries, query, max_length=aggregate_max
        )

        after_tokens = self.calculate_tokens([aggregate])
        logger.info(
            f"[L3] LLM summarization: {current_tokens} -> {after_tokens} tokens"
        )
        return [aggregate]

    @staticmethod
    def _to_norm_vec(embedding: list[float]) -> Optional[Any]:
        """Convert an embedding into a normalized numpy vector."""
        import numpy as np
        vec = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        return vec / norm

    def _record_stats(
        self,
        original: list[str],
        compressed: list[str],
        level: int,
        after_tokens: int,
    ) -> None:
        """Record the statistics of this compression."""
        orig_tokens = self.calculate_tokens(original)
        ratio = after_tokens / max(orig_tokens, 1)
        retention = self._estimate_retention(original, compressed)
        stats = {
            "level": level,
            "original_tokens": orig_tokens,
            "compressed_tokens": after_tokens,
            "compression_ratio": round(ratio, 3),
            "information_retention": round(retention, 3),
        }
        self._stats_history.append(stats)

    def _estimate_retention(
        self,
        original: list[str],
        compressed: list[str],
    ) -> float:
        """
        Estimate the information retention rate.

        Simple heuristic:
        - extract numeric entities and English proper nouns from original
        - check how many appear in compressed
        - retention = entities that appear / total entities
        """
        orig_text = " ".join(original)
        comp_text = " ".join(compressed)

        # numeric entities (including percentages and dates)
        numbers = set(re.findall(r"\d+[\d,]*\.?\d*\s*%?|\d{4}-\d{2}-\d{2}", orig_text))
        # English proper nouns (sequences of capitalized words)
        names = set(re.findall(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}", orig_text))
        entities = numbers | names
        if not entities:
            return 1.0

        preserved = sum(1 for e in entities if e in comp_text)
        return preserved / len(entities)

    def get_stats(self) -> dict[str, Any]:
        """
        Return the accumulated compression statistics.

        Returns:
            {
                "total_compresses": number of compressions,
                "avg_compression_ratio": average compression ratio,
                "avg_retention": average information retention rate,
                "level_distribution": usage count of each level,
                "history": detailed record of each compression,
            }
        """
        if not self._stats_history:
            return {
                "total_compresses": 0,
                "avg_compression_ratio": 1.0,
                "avg_retention": 1.0,
                "level_distribution": {0: 0, 1: 0, 2: 0, 3: 0},
                "history": [],
            }

        total = len(self._stats_history)
        avg_ratio = sum(s["compression_ratio"] for s in self._stats_history) / total
        avg_retention = sum(s["information_retention"] for s in self._stats_history) / total
        level_dist: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for s in self._stats_history:
            level_dist[s["level"]] = level_dist.get(s["level"], 0) + 1

        return {
            "total_compresses": total,
            "avg_compression_ratio": round(avg_ratio, 3),
            "avg_retention": round(avg_retention, 3),
            "level_distribution": level_dist,
            "history": list(self._stats_history),
        }
