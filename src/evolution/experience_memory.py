"""
M6 self-evolution engine — experience memory module

ExperienceMemory stores key patterns of successful/failed trajectories, supports sentence-transformer
embedding-based similarity retrieval, and evicts old experiences by composite score.

Design decisions:
1. SQLite persistence: lightweight, no extra service, suited to research experiments.
2. Embedding cache: stored in the database after first encoding to avoid recomputation.
3. Eviction policy: composite score = 0.4×quality + 0.3×freshness + 0.3×usefulness; low-scoring experiences are cleaned periodically.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

from src.orchestrator.schemas import ResearchReport


__all__ = ["ExperienceMemory"]


class ExperienceMemory:
    """Experience memory: stores, retrieves and evicts key patterns of research trajectories.

    Attributes:
        db_path: SQLite database file path.
        embedding_dim: embedding vector dimension (default 384, matching all-MiniLM-L6-v2).
    """

    def __init__(self, db_path: str = "experience.db", embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._embedder: Any | None = None
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # ------------------------------------------------------------------
    # Database initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the SQLite table schema (if absent)."""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_summary TEXT NOT NULL,
                trajectory_json TEXT NOT NULL,
                success INTEGER NOT NULL,
                score REAL NOT NULL,
                embedding TEXT,
                created_round INTEGER NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_access_round INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_round ON experiences(created_round)"
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ------------------------------------------------------------------
    # Lazy-loaded embedder
    # ------------------------------------------------------------------

    def _get_embedder(self) -> Any | None:
        """Lazily load the embedder, preferring the project's existing Embedder."""
        if self._embedder is not None:
            return self._embedder
        try:
            from memory.embedder import Embedder
            self._embedder = Embedder()
        except Exception:
            self._embedder = None
        return self._embedder

    def _encode(self, text: str) -> list[float]:
        """Encode text into an embedding vector; return a zero vector on failure."""
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                return embedder.encode(text)
            except Exception:
                pass
        return [0.0] * self.embedding_dim

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute the cosine similarity of two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(
        self,
        trajectory: list[dict[str, Any]],
        success: bool,
        score: float,
        strategy_summary: str,
        current_round: int = 0,
    ) -> int:
        """Add an experience to the memory store.

        Args:
            trajectory: interaction trajectory list.
            success: whether the task was completed successfully.
            score: final composite score.
            strategy_summary: strategy summary (for embedding and human reading).
            current_round: current evolution round, used for freshness computation.

        Returns:
            The id of the newly inserted record.
        """
        embedding = self._encode(strategy_summary)
        embedding_str = json.dumps(embedding)
        traj_str = json.dumps(trajectory, ensure_ascii=False)

        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO experiences
            (strategy_summary, trajectory_json, success, score, embedding, created_round, access_count, last_access_round)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                strategy_summary,
                traj_str,
                1 if success else 0,
                score,
                embedding_str,
                current_round,
                current_round,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        current_round: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant experiences by semantic similarity.

        Args:
            query: query text (e.g. the current research question or strategy summary).
            top_k: return the k most relevant experiences.
            current_round: current evolution round, used to update access statistics.

        Returns:
            List of experience dicts sorted by similarity, descending.
        """
        query_emb = self._encode(query)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, strategy_summary, trajectory_json, success, score, embedding, "
            "created_round, access_count, last_access_round FROM experiences"
        ).fetchall()

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            emb_str = row["embedding"] or ""
            if not emb_str:
                continue
            try:
                emb = json.loads(emb_str)
                sim = self._cosine_similarity(query_emb, emb)
                scored.append((sim, row))
            except (json.JSONDecodeError, ValueError):
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for sim, row in scored[:top_k]:
            results.append(
                {
                    "id": row["id"],
                    "strategy_summary": row["strategy_summary"],
                    "trajectory": json.loads(row["trajectory_json"]),
                    "success": bool(row["success"]),
                    "score": row["score"],
                    "similarity": round(sim, 4),
                    "created_round": row["created_round"],
                }
            )
            # Update access statistics
            conn.execute(
                "UPDATE experiences SET access_count = access_count + 1, last_access_round = ? WHERE id = ?",
                (current_round, row["id"]),
            )
        conn.commit()
        return results

    def evict_old_experiences(
        self,
        max_age_rounds: int = 5,
        current_round: int = 0,
        retain_min: int = 100,
    ) -> int:
        """Evict old experiences.

        Eviction rules:
        1. Experiences not accessed for more than max_age_rounds (large last_access_round gap).
        2. Experiences with a low composite score: composite = 0.4×quality + 0.3×freshness + 0.3×usefulness.
           - quality: score / 10
           - freshness: 1 - min(age, max_age) / max_age
           - usefulness: min(access_count / 5, 1.0)
        3. Keep at least retain_min entries.

        Args:
            max_age_rounds: maximum allowed age (in rounds).
            current_round: current evolution round.
            retain_min: minimum number of entries to keep.

        Returns:
            Number of deleted records.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, score, created_round, access_count, last_access_round FROM experiences"
        ).fetchall()

        if len(rows) <= retain_min:
            return 0

        # Compute the composite score of each record
        scored: list[tuple[float, int]] = []  # (composite_score, id)
        for row in rows:
            age = current_round - row["last_access_round"]
            quality = row["score"] / 10.0
            freshness = 1.0 - min(age, max_age_rounds) / max_age_rounds
            utility = min(row["access_count"] / 5.0, 1.0)
            composite = 0.4 * quality + 0.3 * freshness + 0.3 * utility
            scored.append((composite, row["id"]))

        # Sort by composite score ascending; low scores are evicted first
        scored.sort(key=lambda x: x[0])
        to_evict_count = max(0, len(scored) - retain_min)
        to_evict_ids = [sid for _, sid in scored[:to_evict_count]]

        if to_evict_ids:
            placeholders = ",".join("?" * len(to_evict_ids))
            conn.execute(
                f"DELETE FROM experiences WHERE id IN ({placeholders})",
                to_evict_ids,
            )
            conn.commit()
        return len(to_evict_ids)

    def get_stats(self) -> dict[str, Any]:
        """Return memory-store statistics."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt, AVG(score) as avg_score, "
            "SUM(success) as total_success FROM experiences"
        ).fetchone()
        return {
            "total_experiences": row["cnt"] or 0,
            "avg_score": row["avg_score"] or 0.0,
            "success_rate": (row["total_success"] or 0) / max(row["cnt"], 1),
        }
