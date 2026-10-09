"""
Extractive Compressor module: TextRank + query-biased key-sentence extraction

Design decisions:
1. A simple TextRank implementation: build a fully connected graph from sentence cosine similarities,
   compute PageRank scores by power iteration, with no external dependency (e.g. networkx)
2. Query-biased scoring: blend the TextRank score with query relevance,
   so the extracted sentences are highly relevant to the current query
3. Keep numbers, source citations and key conclusions: pre-mark sentences with numbers / URLs / citations by regex and give them a bonus
4. Dynamic keep ratio: computed from the remaining budget; the tighter the budget, the lower the ratio
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np

from src.memory.embedder import Embedder

logger = logging.getLogger(__name__)

# sentence delimiter pattern: supports Chinese and English periods, question marks, exclamation marks, newlines
_SENTENCE_PATTERN = re.compile(r'[^.!?。！？\n]+[.!?。！？\n]*')

# regex for numbers / URLs / citations (these sentences are information-dense and get a bonus)
_HIGH_VALUE_PATTERN = re.compile(
    r"(\d+[\d,]*\.?\d*\s*%?|\d{4}-\d{2}-\d{2}|https?://|www\.|\[[\d\w]+\]|"
    r"according to|cited|reported|found that|结论|结果表明|数据显示)",
    re.IGNORECASE,
)


def _tokenize_sentences(text: str) -> list[str]:
    """Split text into sentences and filter out very short ones."""
    raw = _SENTENCE_PATTERN.findall(text)
    sentences = [s.strip() for s in raw if len(s.strip()) > 8]
    return sentences


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute the cosine-similarity matrix of a vector matrix.

    Args:
        vectors: normalized vector matrix of shape (n_sentences, dim)

    Returns:
        similarity matrix of shape (n_sentences, n_sentences)
    """
    # defensive normalization
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normalized = vectors / norms
    return normalized.dot(normalized.T)


def _textrank_scores(sim_matrix: np.ndarray, damping: float = 0.85, max_iter: int = 30, tol: float = 1e-4) -> np.ndarray:
    """
    Simple PageRank iteration to compute TextRank scores.

    Args:
        sim_matrix: sentence similarity matrix (weights already normalized)
        damping: damping factor
        max_iter: maximum number of iterations
        tol: convergence threshold

    Returns:
        the TextRank score of each sentence
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return np.array([])

    # convert the similarity matrix into a transition probability matrix (row-normalized)
    # keep only edges with similarity > 0.1 (sparsification, less noise)
    adj = sim_matrix.copy()
    adj[adj < 0.1] = 0.0
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0
    transition = adj / row_sums

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T.dot(scores)
        if np.linalg.norm(new_scores - scores) < tol:
            break
        scores = new_scores
    return scores


class ExtractiveCompressor:
    """
    Extractive compressor: TextRank + query-biased key-sentence extraction.
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        """
        Initialize the extractive compressor.

        Args:
            embedder: the vectorizer; created automatically when None
        """
        self.embedder = embedder or Embedder()

    def compress(
        self,
        text: str,
        query: str,
        target_ratio: float = 0.3,
    ) -> str:
        """
        Run extractive compression on a single text.

        Args:
            text: the original text
            query: the current query (for relevance weighting)
            target_ratio: share of sentences to keep (0-1)

        Returns:
            the compressed text (kept sentences joined in their original order)
        """
        sentences = _tokenize_sentences(text)
        if not sentences:
            return text
        if len(sentences) <= 3:
            # too short to compress
            return text

        top_sents = self.textrank_sentences(sentences, query, target_ratio)
        return " ".join(top_sents)

    def textrank_sentences(
        self,
        sentences: list[str],
        query: str,
        top_ratio: float = 0.3,
    ) -> list[str]:
        """
        TextRank + query-biased key-sentence extraction, returning the sentence list in original order.

        Args:
            sentences: the sentence-split result
            query: the query text
            top_ratio: keep ratio

        Returns:
            the kept sentences (in original order)
        """
        n = len(sentences)
        if n == 0:
            return []

        # 1. compute sentence embeddings
        embeddings = self.embedder.encode_batch(sentences)
        emb_matrix = np.array(embeddings, dtype=np.float32)

        # 2. compute TextRank scores
        sim_matrix = _cosine_similarity_matrix(emb_matrix)
        textrank = _textrank_scores(sim_matrix)

        # 3. compute query-biased scores
        query_emb = np.array(self.embedder.encode(query), dtype=np.float32)
        q_norm = float(np.linalg.norm(query_emb))
        if q_norm > 1e-9:
            query_emb = query_emb / q_norm
        else:
            query_emb = query_emb

        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        normalized_emb = emb_matrix / norms
        query_sims = normalized_emb.dot(query_emb)

        # 4. bonus for high-value sentences
        value_bonus = np.array([
            1.2 if _HIGH_VALUE_PATTERN.search(s) else 1.0
            for s in sentences
        ], dtype=np.float32)

        # 5. blended score = TextRank * query_sim * value_bonus
        combined = textrank * query_sims * value_bonus

        # 6. pick the top_k sentences but keep the original order
        k = max(1, int(n * top_ratio))
        top_indices = set(np.argsort(combined)[::-1][:k].tolist())
        result = [s for i, s in enumerate(sentences) if i in top_indices]
        return result

    def query_biased_score(
        self,
        sentence: str,
        query: str,
        embedding: Optional[list[float]] = None,
    ) -> float:
        """
        Compute the query-biased score of a single sentence.

        Args:
            sentence: the sentence text
            query: the query text
            embedding: precomputed sentence embedding; computed on the fly when None

        Returns:
            the query relevance score (between 0 and 1)
        """
        if embedding is None:
            embedding = self.embedder.encode(sentence)
        sent_vec = np.array(embedding, dtype=np.float32)
        q_vec = np.array(self.embedder.encode(query), dtype=np.float32)
        s_norm = float(np.linalg.norm(sent_vec))
        q_norm = float(np.linalg.norm(q_vec))
        if s_norm < 1e-9 or q_norm < 1e-9:
            return 0.0
        return float(np.dot(sent_vec, q_vec) / (s_norm * q_norm))

    def compress_documents(
        self,
        documents: list[str],
        query: str,
        top_ratio: float = 0.3,
    ) -> list[str]:
        """
        Run extractive compression on several documents separately.

        Args:
            documents: list of documents
            query: the query text
            top_ratio: keep ratio per document

        Returns:
            the list of compressed documents (one-to-one with the input)
        """
        return [self.compress(doc, query, top_ratio) for doc in documents]

    def get_stats(self, original: str, compressed: str) -> dict[str, Any]:
        """
        Return compression statistics.

        Args:
            original: the original text
            compressed: the compressed text

        Returns:
            {"compression_ratio": ..., "original_chars": ..., "compressed_chars": ...}
        """
        orig_len = len(original)
        comp_len = len(compressed)
        ratio = comp_len / max(orig_len, 1)
        return {
            "compression_ratio": round(ratio, 3),
            "original_chars": orig_len,
            "compressed_chars": comp_len,
        }
