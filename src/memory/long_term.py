"""
Long-term Memory module: SQLite persistence + structured queries

Design decisions:
1. SQLite needs no ops, persists at file level, has ACID transactions, and suits knowledge bases of <100K entries
2. Embeddings are stored as JSON text and deserialized into numpy arrays on load
3. All write operations are guarded by a threading.Lock for thread safety
4. metadata uses a JSON field, keeping the schema flexible
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Memory entry data model."""

    entry_id: str
    claim: str              # core information (one sentence)
    source: str             # source URL / title
    confidence: float       # 0-1
    agent_id: str           # the agent that wrote it
    timestamp: float
    evidence_type: str      # "primary" | "secondary" | "inference"
    embedding: list[float]  # semantic vector
    topic: str
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""    # owning session ID; an empty string means global / uncategorized

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict; embedding and metadata become JSON strings."""
        return {
            "entry_id": self.entry_id,
            "claim": self.claim,
            "source": self.source,
            "confidence": self.confidence,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "evidence_type": self.evidence_type,
            "embedding_json": json.dumps(self.embedding, ensure_ascii=False),
            "topic": self.topic,
            "metadata_json": json.dumps(self.metadata, ensure_ascii=False),
            "session_id": self.session_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> MemoryEntry:
        """Deserialize from a SQLite Row."""
        return cls(
            entry_id=row["entry_id"],
            claim=row["claim"],
            source=row["source"],
            confidence=row["confidence"],
            agent_id=row["agent_id"],
            timestamp=row["timestamp"],
            evidence_type=row["evidence_type"],
            embedding=json.loads(row["embedding_json"]),
            topic=row["topic"],
            metadata=json.loads(row["metadata_json"]),
            session_id=row["session_id"] if "session_id" in row.keys() else "",
        )


@dataclass
class ConflictRecord:
    """Contradiction record data model."""

    conflict_id: str
    entry_id_1: str
    entry_id_2: str
    claim_1: str
    claim_2: str
    similarity: float       # cosine similarity of the two entries
    status: str             # "open" | "resolved" | "dismissed"
    resolution: Optional[str] = None  # resolution result entry_id
    detected_at: float = field(default_factory=time.time)


class LongTermMemory:
    """
    SQLite wrapper for long-term memory.

    Provides CRUD for the entries and conflicts tables, plus basic filtered queries.
    The upper-level SharedMemoryStore handles vector-similarity computation and high-level semantic operations.
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        """
        Initialize and create the tables.

        Args:
            db_path: SQLite database file path
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        # create the parent directory automatically (SQLite does not create a missing directory)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()
        self._migrate_add_session_id()

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection (SQLite connections are not thread-safe, so create one per operation)."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        """Initialize the table schema (if it does not exist)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entries (
                        entry_id TEXT PRIMARY KEY,
                        claim TEXT NOT NULL,
                        source TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        agent_id TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        evidence_type TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        session_id TEXT NOT NULL DEFAULT ''
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entries_topic ON entries(topic)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entries_agent ON entries(agent_id)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_entries_timestamp ON entries(timestamp)
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conflicts (
                        conflict_id TEXT PRIMARY KEY,
                        entry_id_1 TEXT NOT NULL,
                        entry_id_2 TEXT NOT NULL,
                        claim_1 TEXT NOT NULL,
                        claim_2 TEXT NOT NULL,
                        similarity REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open',
                        resolution TEXT,
                        detected_at REAL NOT NULL,
                        FOREIGN KEY (entry_id_1) REFERENCES entries(entry_id),
                        FOREIGN KEY (entry_id_2) REFERENCES entries(entry_id)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def _migrate_add_session_id(self) -> None:
        """Add the session_id column to an existing old table (backward compatible)."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("PRAGMA table_info(entries)")
                columns = [row["name"] for row in cur.fetchall()]
                if "session_id" not in columns:
                    conn.execute("ALTER TABLE entries ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_session ON entries(session_id)")
                    conn.commit()
                    logger.info("[LongTermMemory] Migrated: added the session_id column to the entries table")
            except Exception as e:
                logger.warning(f"[LongTermMemory] session_id migration failed (it may already exist): {e}")
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Entries operations
    # ------------------------------------------------------------------

    def insert_entry(self, entry: MemoryEntry) -> None:
        """Insert or replace an entry (REPLACE semantics)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO entries
                    (entry_id, claim, source, confidence, agent_id, timestamp,
                     evidence_type, embedding_json, topic, metadata_json, session_id)
                    VALUES
                    (:entry_id, :claim, :source, :confidence, :agent_id, :timestamp,
                     :evidence_type, :embedding_json, :topic, :metadata_json, :session_id)
                    """,
                    entry.to_dict(),
                )
                conn.commit()
            finally:
                conn.close()

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Query a single entry by ID."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM entries WHERE entry_id = ?", (entry_id,)
                )
                row = cur.fetchone()
                return MemoryEntry.from_row(row) if row else None
            finally:
                conn.close()

    def get_all_entries(self, session_id: Optional[str] = None) -> list[MemoryEntry]:
        """Load entries. If session_id is given, load only that session's data."""
        with self._lock:
            conn = self._connect()
            try:
                if session_id is not None:
                    cur = conn.execute("SELECT * FROM entries WHERE session_id = ?", (session_id,))
                else:
                    cur = conn.execute("SELECT * FROM entries")
                return [MemoryEntry.from_row(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def query_by_topic(self, topic: str) -> list[MemoryEntry]:
        """Exact query by topic."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM entries WHERE topic = ? ORDER BY timestamp DESC",
                    (topic,),
                )
                return [MemoryEntry.from_row(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def query_by_agent(self, agent_id: str) -> list[MemoryEntry]:
        """Query by agent_id."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM entries WHERE agent_id = ? ORDER BY timestamp DESC",
                    (agent_id,),
                )
                return [MemoryEntry.from_row(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry and its associated conflicts. Returns whether the deletion succeeded."""
        with self._lock:
            conn = self._connect()
            try:
                # delete the associated conflicts first
                conn.execute(
                    "DELETE FROM conflicts WHERE entry_id_1 = ? OR entry_id_2 = ?",
                    (entry_id, entry_id),
                )
                cur = conn.execute(
                    "DELETE FROM entries WHERE entry_id = ?", (entry_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_sessions(self) -> list[dict[str, Any]]:
        """List all session_ids with their entry counts and last-update times."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT session_id, COUNT(*) as count, MAX(timestamp) as last_update
                    FROM entries
                    WHERE session_id != ''
                    GROUP BY session_id
                    ORDER BY last_update DESC
                    """
                )
                return [
                    {
                        "session_id": r["session_id"],
                        "count": r["count"],
                        "last_update": r["last_update"],
                    }
                    for r in cur.fetchall()
                ]
            finally:
                conn.close()

    def get_entries_by_session(self, session_id: str) -> list[MemoryEntry]:
        """Query all entries by session_id."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT * FROM entries WHERE session_id = ? ORDER BY timestamp DESC",
                    (session_id,),
                )
                return [MemoryEntry.from_row(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def count_entries(self, session_id: Optional[str] = None) -> int:
        """Return the number of entries. Can be filtered by session_id."""
        with self._lock:
            conn = self._connect()
            try:
                if session_id is not None:
                    cur = conn.execute("SELECT COUNT(*) FROM entries WHERE session_id = ?", (session_id,))
                else:
                    cur = conn.execute("SELECT COUNT(*) FROM entries")
                return cur.fetchone()[0]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Conflicts operations
    # ------------------------------------------------------------------

    def insert_conflict(self, record: ConflictRecord) -> None:
        """Insert a contradiction record (duplicates ignored)."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO conflicts
                    (conflict_id, entry_id_1, entry_id_2, claim_1, claim_2,
                     similarity, status, resolution, detected_at)
                    VALUES
                    (:conflict_id, :entry_id_1, :entry_id_2, :claim_1, :claim_2,
                     :similarity, :status, :resolution, :detected_at)
                    """,
                    {
                        "conflict_id": record.conflict_id,
                        "entry_id_1": record.entry_id_1,
                        "entry_id_2": record.entry_id_2,
                        "claim_1": record.claim_1,
                        "claim_2": record.claim_2,
                        "similarity": record.similarity,
                        "status": record.status,
                        "resolution": record.resolution,
                        "detected_at": record.detected_at,
                    },
                )
                conn.commit()
            finally:
                conn.close()

    def get_conflicts(self, status: Optional[str] = None) -> list[ConflictRecord]:
        """Query contradiction records, optionally filtered by status."""
        with self._lock:
            conn = self._connect()
            try:
                if status:
                    cur = conn.execute(
                        "SELECT * FROM conflicts WHERE status = ? ORDER BY detected_at DESC",
                        (status,),
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM conflicts ORDER BY detected_at DESC"
                    )
                rows = cur.fetchall()
                return [
                    ConflictRecord(
                        conflict_id=r["conflict_id"],
                        entry_id_1=r["entry_id_1"],
                        entry_id_2=r["entry_id_2"],
                        claim_1=r["claim_1"],
                        claim_2=r["claim_2"],
                        similarity=r["similarity"],
                        status=r["status"],
                        resolution=r["resolution"],
                        detected_at=r["detected_at"],
                    )
                    for r in rows
                ]
            finally:
                conn.close()

    def update_conflict_resolution(
        self,
        conflict_id: str,
        status: str,
        resolution: Optional[str] = None,
    ) -> bool:
        """Update the status and resolution result of a contradiction record."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE conflicts
                    SET status = ?, resolution = ?
                    WHERE conflict_id = ?
                    """,
                    (status, resolution, conflict_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def delete_conflict(self, conflict_id: str) -> bool:
        """Delete a contradiction record."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM conflicts WHERE conflict_id = ?", (conflict_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_lowest_score_entries(self, limit: int, session_id: Optional[str] = None) -> list[tuple[str, float]]:
        """
        Return the lowest-scoring entries in ascending composite-score order, for eviction.

        Composite score = confidence × evidence_weight × recency × conflict_bonus
        Here the evidence_type weights are: primary=1.0, secondary=0.8, inference=0.6
        recency uses time decay: exp(-days/30)
        conflict_bonus: entries with a contradiction are kept (an extremely low score is returned so they sort last)
        """
        evidence_weights = {"primary": 1.0, "secondary": 0.8, "inference": 0.6}
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                if session_id is not None:
                    cur = conn.execute(
                        """
                        SELECT e.entry_id, e.confidence, e.evidence_type, e.timestamp,
                               (SELECT COUNT(*) FROM conflicts c
                                WHERE (c.entry_id_1 = e.entry_id OR c.entry_id_2 = e.entry_id)
                                AND c.status = 'open') AS open_conflicts
                        FROM entries e
                        WHERE e.session_id = ?
                        ORDER BY e.timestamp ASC
                        """,
                        (session_id,),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT e.entry_id, e.confidence, e.evidence_type, e.timestamp,
                               (SELECT COUNT(*) FROM conflicts c
                                WHERE (c.entry_id_1 = e.entry_id OR c.entry_id_2 = e.entry_id)
                                AND c.status = 'open') AS open_conflicts
                        FROM entries e
                        ORDER BY e.timestamp ASC
                        """
                    )
                rows = cur.fetchall()
                scores: list[tuple[str, float]] = []
                for r in rows:
                    entry_id = r["entry_id"]
                    confidence = r["confidence"]
                    ew = evidence_weights.get(r["evidence_type"], 0.5)
                    days_old = max((now - r["timestamp"]) / 86400.0, 0.0)
                    recency = np.exp(-days_old / 30.0)
                    conflict_bonus = 2.0 if r["open_conflicts"] > 0 else 1.0
                    score = confidence * ew * recency * conflict_bonus
                    scores.append((entry_id, score))
                scores.sort(key=lambda x: x[1])
                return scores[:limit]
            finally:
                conn.close()
