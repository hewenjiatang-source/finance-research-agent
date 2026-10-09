"""
Embedder module: text vectorization wrapper

Design decisions:
1. The main model is all-MiniLM-L6-v2 (lightweight, 384 dimensions, good enough)
2. Provides a graceful fallback: when sentence-transformers is not installed,
   return a deterministic random embedding (based on the text hash), so tests are reproducible
3. Single-instance model loading + lazy init, avoiding repeated initialization cost
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from typing import Optional

import numpy as np

# in mainland-China environments use the HuggingFace mirror (hf-mirror.com) automatically
if os.environ.get("HF_ENDPOINT", "").strip() == "":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logger = logging.getLogger(__name__)

# try to import sentence-transformers, flag it if not installed
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not installed. Embedder will use deterministic random fallback."
    )


class Embedder:
    """Text vectorizer: wraps sentence-transformers and provides a fallback."""

    # class-level cache: avoids reloading the model
    _model_instance: Optional[object] = None
    _model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _embedding_dim: int = 384  # output dimension of all-MiniLM-L6-v2

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initialize the Embedder.

        Args:
            model_name: sentence-transformers model name; None uses the default all-MiniLM-L6-v2
        """
        self.model_name = model_name or self._model_name
        self._model: Optional[object] = None
        self._available = _SENTENCE_TRANSFORMERS_AVAILABLE

    def _load_model(self) -> object:
        """Lazily load the model, returning a SentenceTransformer instance or None (fallback mode)."""
        if not self._available:
            return None
        if self._model is not None:
            return self._model
        # try to load from the class cache
        if Embedder._model_instance is not None:
            self._model = Embedder._model_instance
            return self._model
        try:
            Embedder._model_instance = SentenceTransformer(self.model_name)
            self._model = Embedder._model_instance
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._available = False
            self._model = None
        return self._model

    def encode(self, text: str) -> list[float]:
        """
        Convert text into an embedding vector.

        Args:
            text: the input text

        Returns:
            list of floats, of length 384 (main model) or the fallback dimension
        """
        if not text or not text.strip():
            # empty text returns a zero vector
            return [0.0] * self._embedding_dim

        model = self._load_model()
        if model is not None:
            try:
                embedding = model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Model encode failed, fallback to random: {e}")

        # Fallback: deterministic random embedding (based on the text hash)
        return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list[float]:
        """
        Deterministic random embedding fallback.

        Uses the text's MD5 hash as the random seed, so identical text always yields the identical vector,
        which helps tests and the consistency checks of the dedup logic.
        """
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31)
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self._embedding_dim)]
        # L2 normalization
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        return vec

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batch encoding, more efficient than many single encode calls.

        Args:
            texts: list of texts

        Returns:
            list of embeddings
        """
        if not texts:
            return []
        model = self._load_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, normalize_embeddings=True)
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.warning(f"Batch encode failed, fallback to loop: {e}")
        return [self.encode(t) for t in texts]

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    @property
    def is_available(self) -> bool:
        """Return whether the real model is used (False means fallback mode)."""
        _ = self._load_model()
        return self._available and self._model is not None
