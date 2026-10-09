"""
Shared Memory Store module: unified read/write interface across Agents

Design decisions:
1. The persistence layer is LongTermMemory (SQLite); the in-memory layer keeps a numpy vector index, balancing reads and writes
2. Automatic dedup before write (cosine > 0.92) and contradiction detection (0.65 < cosine < 0.92 + semantic opposition)
3. Contradiction resolution supports three strategies: majority_vote / source_weight / llm_judge
4. Eviction score = confidence × evidence_weight × recency × conflict_bonus
5. get_context_for_query assembles related memories into text, keeping the total tokens within budget
6. All in-memory index operations are protected by threading.Lock, for thread safety
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from typing import Any, Optional

import numpy as np

from src.memory.embedder import Embedder
from src.memory.long_term import ConflictRecord, LongTermMemory, MemoryEntry
from src.utils.tracing import trace_retriever

logger = logging.getLogger(__name__)

# similarity thresholds for dedup and contradiction detection
_DEDUP_THRESHOLD = 0.92
_CONFLICT_LOW = 0.65
_CONFLICT_HIGH = 0.92

# evidence_type weights (for eviction scoring and contradiction resolution)
_EVIDENCE_WEIGHTS = {"primary": 1.0, "secondary": 0.8, "inference": 0.6}

# lists of common negation words and antonyms (for simple heuristic contradiction detection; the Chinese words are intentional)
_NEGATION_WORDS = {"不", "没", "无", "非", "未", "否", "not", "no", "never", "without", "not"}
_ANTONYM_PAIRS: list[tuple[set[str], set[str]]] = [
    ({"增加", "上升", "增长", "提高", "扩大", "increase", "rise", "grow"},
     {"减少", "下降", "降低", "缩减", "收缩", "decrease", "fall", "drop"}),
    ({"好", "优", "强", "positive", "good", "strong"},
     {"坏", "劣", "弱", "negative", "bad", "weak"}),
    ({"支持", "赞成", "agree", "support"},
     {"反对", "reject", "oppose", "disagree"}),
    ({"成功", "success"}, {"失败", "failure"}),
    ({"高", "high"}, {"低", "low"}),
    ({"大", "big", "large"}, {"小", "small", "tiny"}),
]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine similarity of two vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _is_semantically_opposite(claim_a: str, claim_b: str) -> bool:
    """
    Simple heuristic to decide whether two claims are semantically opposed.

    Strategy:
    1. check whether one side contains a negation word and the other does not
    2. check whether they contain an antonym pair
    """
    ca = claim_a.lower()
    cb = claim_b.lower()

    # negation check: one side negated, the other not, with similar structure
    a_has_neg = any(w in ca for w in _NEGATION_WORDS)
    b_has_neg = any(w in cb for w in _NEGATION_WORDS)
    if a_has_neg != b_has_neg:
        # further check: whether they are highly similar once negation words are removed
        # simple test: Jaccard similarity after removing negation words
        def _strip_neg(text: str) -> set[str]:
            words = set(text.split())
            for w in _NEGATION_WORDS:
                words.discard(w)
            return words
        sim_words = len(_strip_neg(ca) & _strip_neg(cb))
        union_words = len(_strip_neg(ca) | _strip_neg(cb))
        if union_words > 0 and sim_words / union_words > 0.5:
            return True

    # antonym pair check
    for pos_set, neg_set in _ANTONYM_PAIRS:
        a_has_pos = any(w in ca for w in pos_set)
        a_has_neg_word = any(w in ca for w in neg_set)
        b_has_pos = any(w in cb for w in pos_set)
        b_has_neg_word = any(w in cb for w in neg_set)
        if (a_has_pos and b_has_neg_word) or (a_has_neg_word and b_has_pos):
            return True

    return False


class SharedMemoryStore:
    """
    Cross-Agent shared memory store.

    Provides high-level semantic interfaces such as put/query/conflict_resolution/evict;
    underneath, LongTermMemory does SQLite persistence and a numpy vector index in memory speeds up similarity queries.
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        embedder: Optional[Embedder] = None,
        session_id: str = "",
    ) -> None:
        """
        Initialize the shared memory store.

        Args:
            db_path: SQLite database path
            embedder: the vectorizer; created automatically when None
            session_id: session ID; an empty string loads all history (no isolation)
        """
        self.lt = LongTermMemory(db_path=db_path)
        self.embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self.session_id = session_id

        # in-memory vector index
        self._entry_ids: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._entries_cache: dict[str, MemoryEntry] = {}

        # load existing data into the in-memory index (filtered by session)
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Internal index management
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild the in-memory vector index from SQLite. If session_id is given, load only that session's data."""
        entries = self.lt.get_all_entries(session_id=self.session_id or None)
        with self._lock:
            self._entry_ids = [e.entry_id for e in entries]
            self._entries_cache = {e.entry_id: e for e in entries}
            if entries:
                mat = np.array([e.embedding for e in entries], dtype=np.float32)
                # normalize (defensive)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                self._embeddings = mat / norms
            else:
                self._embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
        scope = f"session={self.session_id}" if self.session_id else "all sessions"
        logger.info(f"Memory index rebuilt: {len(entries)} entries loaded ({scope}).")

    def _add_to_index(self, entry: MemoryEntry) -> None:
        """Append a single entry to the in-memory index."""
        vec = np.array(entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-9:
            vec = vec / norm
        with self._lock:
            self._entry_ids.append(entry.entry_id)
            self._entries_cache[entry.entry_id] = entry
            if self._embeddings.shape[0] == 0:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])

    def _remove_from_index(self, entry_id: str) -> None:
        """Remove an entry from the in-memory index."""
        with self._lock:
            if entry_id not in self._entry_ids:
                return
            idx = self._entry_ids.index(entry_id)
            self._entry_ids.pop(idx)
            self._entries_cache.pop(entry_id, None)
            if self._embeddings.shape[0] > 0:
                self._embeddings = np.delete(self._embeddings, idx, axis=0)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    # junk-content detection patterns (regex / keywords, filtered before writing; the Chinese greetings are intentional)
    _JUNK_PATTERNS = [
        re.compile(r"i'm ready to help", re.I),
        re.compile(r"您好|你好|hello|hi there", re.I),
        re.compile(r"error\s*:\s*error code:\s*\d+", re.I),
        re.compile(r"^\s*error\s*:\s*", re.I),
        re.compile(r"could you please|请问您想要|您想让我", re.I),
    ]

    def _is_junk(self, entry: MemoryEntry) -> bool:
        """Heuristically decide whether an entry is low-quality / junk and should be rejected."""
        claim = entry.claim or ""
        # 1. too short
        if len(claim.strip()) < 30:
            return True
        # 2. confidence too low
        if entry.confidence < 0.3:
            return True
        # 3. matches a junk pattern
        for pat in self._JUNK_PATTERNS:
            if pat.search(claim):
                return True
        return False

    @trace_retriever(name="memory.put", tags=["m4", "memory"])
    def put(self, entry: MemoryEntry) -> str:
        """
        Write a memory entry.

        Flow:
        1. quality filter: low-quality / junk content is not stored
        2. if the embedding is empty, generate it automatically
        3. dedup check: cosine > 0.92 → merge (keep the one with higher confidence)
        4. contradiction check: 0.65 < cosine < 0.92 and semantically opposed → mark a ConflictRecord
        5. persist to SQLite and update the in-memory index

        Args:
            entry: the memory entry

        Returns:
            entry_id (on merge, the id of the existing entry merged into)
        """
        # 1. quality filter
        if self._is_junk(entry):
            logger.info(f"[M4] Junk entry rejected (conf={entry.confidence:.2f}, len={len(entry.claim)}): {entry.claim[:60]}...")
            return entry.entry_id

        # 2. make sure there is an embedding
        if not entry.embedding:
            entry.embedding = self.embedder.encode(entry.claim)

        # dedup check
        duplicate_id = self._find_duplicate(entry)
        if duplicate_id:
            existing = self.lt.get_entry(duplicate_id)
            if existing and entry.confidence > existing.confidence:
                # update with the new entry but keep the old ID
                entry.entry_id = duplicate_id
                entry.timestamp = max(entry.timestamp, existing.timestamp)
                self.lt.insert_entry(entry)
                self._remove_from_index(duplicate_id)
                self._add_to_index(entry)
                logger.info(f"Merged entry {duplicate_id} with higher confidence.")
            else:
                logger.info(f"Duplicate detected, kept existing {duplicate_id}.")
            return duplicate_id

        # write the session_id and persist
        entry.session_id = self.session_id
        self.lt.insert_entry(entry)
        self._add_to_index(entry)

        # contradiction check (compare with the new entry)
        self._detect_conflicts(entry)

        return entry.entry_id

    def _find_duplicate(self, entry: MemoryEntry) -> Optional[str]:
        """
        Search the in-memory index for existing entries with cosine similarity > 0.92 to the entry.

        Returns:
            the most similar existing entry_id, or None
        """
        if self._embeddings.shape[0] == 0:
            return None
        vec = np.array(entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return None
        vec = vec / norm
        with self._lock:
            sims = self._embeddings.dot(vec)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim > _DEDUP_THRESHOLD:
            return self._entry_ids[best_idx]
        return None

    def _detect_conflicts(self, new_entry: MemoryEntry) -> None:
        """
        Detect potential contradictions between the new entry and existing entries.

        Conditions:
        - cosine similarity in the interval (0.65, 0.92) (related topic but not identical)
        - semantically opposed claims (heuristic judgment)
        """
        if self._embeddings.shape[0] == 0:
            return
        vec = np.array(new_entry.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return
        vec = vec / norm
        with self._lock:
            sims = self._embeddings.dot(vec)
        # exclude itself (the last one is the new entry, which was just written into the index)
        # in fact the put flow writes SQLite first and then the index, so the new entry is already at the end of _embeddings
        # we only check the entries before it
        for idx, sim in enumerate(sims[:-1]):
            if _CONFLICT_LOW < float(sim) < _CONFLICT_HIGH:
                existing_id = self._entry_ids[idx]
                existing = self._entries_cache.get(existing_id)
                if existing is None:
                    continue
                if _is_semantically_opposite(new_entry.claim, existing.claim):
                    conflict = ConflictRecord(
                        conflict_id=str(uuid.uuid4()),
                        entry_id_1=existing.entry_id,
                        entry_id_2=new_entry.entry_id,
                        claim_1=existing.claim,
                        claim_2=new_entry.claim,
                        similarity=float(sim),
                        status="open",
                    )
                    self.lt.insert_conflict(conflict)
                    logger.info(
                        f"Conflict detected between {existing.entry_id} and {new_entry.entry_id}: "
                        f"sim={sim:.3f}"
                    )

    @trace_retriever(name="memory.query", tags=["m4", "memory"])
    def query_by_similarity(
        self, query: str, top_k: int = 5, min_sim: float = 0.50
    ) -> list[tuple[MemoryEntry, float]]:
        """
        Search memories by query semantic similarity, filtered by a relevance threshold.

        Args:
            query: the query text
            top_k: maximum number of results
            min_sim: minimum similarity threshold; memories below it are treated as irrelevant

        Returns:
            list of (MemoryEntry, similarity) (descending similarity)
        """
        if self._embeddings.shape[0] == 0:
            return []
        q_vec = np.array(self.embedder.encode(query), dtype=np.float32)
        norm = float(np.linalg.norm(q_vec))
        if norm < 1e-9:
            return []
        q_vec = q_vec / norm
        with self._lock:
            sims = self._embeddings.dot(q_vec)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            sim = float(sims[int(idx)])
            if sim < min_sim:
                continue
            entry_id = self._entry_ids[int(idx)]
            entry = self._entries_cache.get(entry_id)
            if entry:
                results.append((entry, sim))
        return results

    def query_by_topic(self, topic: str) -> list[MemoryEntry]:
        """
        Exact query by topic.

        Args:
            topic: the topic name

        Returns:
            all entries under that topic
        """
        return self.lt.query_by_topic(topic)

    def get_conflicts(self, status: Optional[str] = None) -> list[ConflictRecord]:
        """
        Get contradiction records.

        Args:
            status: filter by status ("open" / "resolved" / "dismissed"); None means all

        Returns:
            list of ConflictRecord
        """
        return self.lt.get_conflicts(status=status)

    def resolve_conflict(
        self,
        conflict_id: str,
        strategy: str,
        llm_policy: Optional[Any] = None,
    ) -> Optional[MemoryEntry]:
        """
        Resolve the given contradiction.

        Supported strategies:
        - "majority_vote": the claim supported by the most agents under the same topic wins
        - "source_weight": weighted by evidence_type weight × confidence
        - "llm_judge": call the VLLMPolicy for an LLM judgment (llm_policy must be passed)

        Args:
            conflict_id: contradiction record ID
            strategy: resolution strategy
            llm_policy: VLLMPolicy instance (required by the llm_judge strategy)

        Returns:
            the winning MemoryEntry, or None (if it cannot be resolved)
        """
        conflicts = self.lt.get_conflicts()
        target: Optional[ConflictRecord] = None
        for c in conflicts:
            if c.conflict_id == conflict_id:
                target = c
                break
        if target is None:
            logger.warning(f"Conflict {conflict_id} not found.")
            return None

        entry_1 = self.lt.get_entry(target.entry_id_1)
        entry_2 = self.lt.get_entry(target.entry_id_2)
        if entry_1 is None or entry_2 is None:
            logger.warning("One or both entries missing, marking dismissed.")
            self.lt.update_conflict_resolution(conflict_id, "dismissed")
            return None

        winner: Optional[MemoryEntry] = None

        if strategy == "majority_vote":
            winner = self._resolve_by_majority(entry_1, entry_2)
        elif strategy == "source_weight":
            winner = self._resolve_by_source_weight(entry_1, entry_2)
        elif strategy == "llm_judge":
            if llm_policy is None:
                raise ValueError("llm_judge strategy requires llm_policy")
            winner = self._resolve_by_llm(entry_1, entry_2, llm_policy)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if winner is not None:
            self.lt.update_conflict_resolution(
                conflict_id, "resolved", resolution=winner.entry_id
            )
            logger.info(f"Conflict {conflict_id} resolved by {strategy}: {winner.entry_id}")
        return winner

    def _resolve_by_majority(
        self, e1: MemoryEntry, e2: MemoryEntry
    ) -> Optional[MemoryEntry]:
        """Count the supporting agents of each claim under the same topic; the majority wins."""
        topic_entries = self.lt.query_by_topic(e1.topic)
        count_1 = sum(1 for e in topic_entries if e.claim == e1.claim)
        count_2 = sum(1 for e in topic_entries if e.claim == e2.claim)
        if count_1 >= count_2:
            return e1
        return e2

    def _resolve_by_source_weight(
        self, e1: MemoryEntry, e2: MemoryEntry
    ) -> Optional[MemoryEntry]:
        """Score by evidence_type weight × confidence."""
        w1 = _EVIDENCE_WEIGHTS.get(e1.evidence_type, 0.5) * e1.confidence
        w2 = _EVIDENCE_WEIGHTS.get(e2.evidence_type, 0.5) * e2.confidence
        if w1 >= w2:
            return e1
        return e2

    def _resolve_by_llm(
        self,
        e1: MemoryEntry,
        e2: MemoryEntry,
        llm_policy: Any,
    ) -> Optional[MemoryEntry]:
        """Ask the LLM which claim is more credible."""
        prompt = f"""Judge which of the following two statements is more credible, and answer only "A" or "B".

Statement A (source: {e1.source}, confidence: {e1.confidence}, evidence type: {e1.evidence_type}):
{e1.claim}

Statement B (source: {e2.source}, confidence: {e2.confidence}, evidence type: {e2.evidence_type}):
{e2.claim}

Output only A or B:"""
        try:
            resp = llm_policy([{"role": "user", "content": prompt}])
            content = str(resp.content or "").strip().upper()
            if content.startswith("A"):
                return e1
            elif content.startswith("B"):
                return e2
        except Exception as ex:
            logger.warning(f"LLM judge failed: {ex}")
        # fall back to source_weight
        return self._resolve_by_source_weight(e1, e2)

    def evict(self, max_entries: int = 10000) -> int:
        """
        Evict low-scoring entries so the total does not exceed max_entries.

        Eviction logic:
        - first compute a composite score for every entry
        - delete the lowest-scoring entries until the total fits
        - entries with an open conflict are protected (conflict_bonus puts them last)

        Args:
            max_entries: maximum number of entries to keep

        Returns:
            number of entries actually deleted
        """
        current_count = self.lt.count_entries(session_id=self.session_id or None)
        if current_count <= max_entries:
            return 0

        to_remove = current_count - max_entries
        low_score_entries = self.lt.get_lowest_score_entries(limit=to_remove, session_id=self.session_id or None)
        removed = 0
        for entry_id, score in low_score_entries:
            # double-check: skip if there is still an open conflict
            conflicts = self.lt.get_conflicts(status="open")
            has_conflict = any(
                c.entry_id_1 == entry_id or c.entry_id_2 == entry_id
                for c in conflicts
            )
            if has_conflict:
                logger.info(f"Protected entry {entry_id} from eviction (open conflict).")
                continue
            if self.lt.delete_entry(entry_id):
                self._remove_from_index(entry_id)
                removed += 1
                logger.info(f"Evicted entry {entry_id} (score={score:.4f}).")
        return removed

    def get_context_for_query(self, query: str, max_tokens: int = 4000) -> str:
        """
        Assemble memory context text relevant to the query for an Agent.

        Strategy:
        1. semantic-similarity search for the top 10 (threshold min_sim=0.55, to keep unrelated memories out)
        2. add time decay: lower the weight of memories that are too old
        3. sort by composite score and join one by one until close to max_tokens
        4. return a formatted text block

        Args:
            query: the current query
            max_tokens: token budget limit

        Returns:
            the assembled context text (an empty string means no related memory)
        """
        import time

        entries_with_sim = self.query_by_similarity(query, top_k=10, min_sim=0.55)
        if not entries_with_sim:
            return ""

        now = time.time()

        def _score(entry: MemoryEntry, sim: float) -> float:
            """Composite score = similarity × confidence × time decay."""
            days_old = max((now - entry.timestamp) / 86400.0, 0.0)
            recency = np.exp(-days_old / 30.0)  # 30-day half-life
            return sim * entry.confidence * recency

        # sort by composite score, descending
        entries_with_sim.sort(key=lambda x: _score(x[0], x[1]), reverse=True)

        max_chars = int(max_tokens * 3.5)  # empirical token → character conversion
        parts: list[str] = []
        current_chars = 0

        header = "## Related background knowledge\n"
        current_chars += len(header)
        parts.append(header)

        for entry, sim in entries_with_sim:
            block = (
                f"- [{entry.topic}] {entry.claim}\n"
                f"  source: {entry.source} | confidence: {entry.confidence:.2f} | "
                f"evidence type: {entry.evidence_type} | relevance: {sim:.2f}\n"
            )
            if current_chars + len(block) > max_chars:
                break
            parts.append(block)
            current_chars += len(block)

        return "".join(parts)

    def __len__(self) -> int:
        return self.lt.count_entries(session_id=self.session_id or None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions in the database with their statistics."""
        return self.lt.get_sessions()
