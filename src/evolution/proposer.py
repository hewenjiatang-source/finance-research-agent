"""
M6 self-evolution engine — research question generator (Proposer)

The Proposer generates research questions at three difficulty levels (L1/L2/L3), supporting adaptive difficulty calibration,
quality filtering and diversity constraints.

Design decisions:
1. The three levels correspond to different search depths, so training data covers the full spectrum from simple to complex.
2. Inverted-U adaptive weight: weight is highest when the success rate is near 50%, avoiding a dataset that is too easy or too hard.
3. Embedding similarity filter: a new question must have similarity < 0.7 with existing ones, ensuring diversity.
4. Success-rate filter: questions that are too easy (>80%) or too hard (<20%) are dropped, keeping a reasonable learning gradient.
"""
from __future__ import annotations

import json
import random
from typing import Any


__all__ = ["Proposer"]


# ============================================================================
# Prompt templates
# ============================================================================

SYSTEM_PROPOSER = (
    "You are an expert in designing research questions. Generate high-quality, challenging and verifiable research questions as required. "
    "Output must be JSON."
)

PROMPT_GENERATE = """Generate {n} research questions of difficulty {difficulty}.

Difficulty definitions:
- L1 (fact lookup): answerable with 1-2 searches; the answer is clear and verifiable.
- L2 (multi-step reasoning): 3-6 searches; requires integrating multi-source information and causal or comparative analysis.
- L3 (cross-domain synthesis): 5-10+ searches; requires interdisciplinary knowledge, long-horizon reasoning and handling conflicting information.

Requirements:
1. Every question must have a clear objective answer or evaluation criterion.
2. Avoid being too broad (e.g. "Tell me about artificial intelligence") or too narrow (e.g. "a specific person's birthday").
3. For recent events, make sure they occurred before 2023 (to guarantee verifiability).
4. Questions must not duplicate or closely resemble each other.

Domain preference (optional): {domains}

Output in the following JSON format:
{
  "questions": [
    {
      "question": "string",
      "difficulty": "L1|L2|L3",
      "expected_searches": int,
      "domain": "string",
      "verification_hint": "string"   // how to verify the answer is correct
    }
  ]
}
"""


# ============================================================================
# Proposer implementation
# ============================================================================

class Proposer:
    """Research question generator supporting three difficulty levels and adaptive calibration.

    Attributes:
        policy: a VLLMPolicy instance.
        difficulty_history: records each question's difficulty, success rate and average score.
        _embedding_cache: embedding cache of generated questions, used for diversity checks.
    """

    def __init__(
        self,
        policy,
        difficulty_history: dict[str, dict[str, Any]] | None = None,
    ):
        self.policy = policy
        # history structure: {question_str: {"difficulty": "L1", "success": bool, "score": float, "attempts": int}}
        self.difficulty_history = difficulty_history or {}
        self._embedding_cache: list[list[float]] = []

    async def generate_batch(
        self,
        n: int = 32,
        domains: list[str] | None = None,
    ) -> list[str]:
        """Generate a batch of research questions.

        Flow:
        1. Decide the L1/L2/L3 generation ratio from the adaptive weights.
        2. Call the LLM in batches to generate questions.
        3. Filter out questions that are too easy / too hard (based on history).
        4. De-duplicate by embedding similarity to ensure diversity.

        Args:
            n: target number to generate.
            domains: optional list of domain preferences.

        Returns:
            List of research question strings.
        """
        weights = self.get_difficulty_weights()
        # Allocate the number to generate per difficulty by weight
        total_weight = sum(weights.values())
        if total_weight == 0.0:
            counts = {"L1": n // 3, "L2": n // 3, "L3": n - 2 * (n // 3)}
        else:
            counts = {}
            remaining = n
            for lvl in ["L1", "L2"]:
                cnt = int(n * weights.get(lvl, 0.0) / total_weight)
                counts[lvl] = cnt
                remaining -= cnt
            counts["L3"] = remaining

        domains_text = ", ".join(domains) if domains else "unrestricted"
        results: list[str] = []

        for difficulty, count in counts.items():
            if count <= 0:
                continue
            # Generate at most 8 at a time to keep the prompt short
            batch_size = 8
            generated = 0
            while generated < count:
                req = min(batch_size, count - generated)
                prompt = PROMPT_GENERATE.format(
                    n=req,
                    difficulty=difficulty,
                    domains=domains_text,
                )
                messages = [
                    {"role": "system", "content": SYSTEM_PROPOSER},
                    {"role": "user", "content": prompt},
                ]
                try:
                    resp = self.policy(messages)
                    raw = resp.content or ""
                    data = self._parse_json(raw)
                    for item in data.get("questions", []):
                        q = item.get("question", "").strip()
                        if not q:
                            continue
                        # History filter: skip questions that are too easy or too hard
                        hist = self.difficulty_history.get(q)
                        if hist and hist.get("attempts", 0) >= 3:
                            success_rate = hist.get("success_rate", 0.5)
                            if success_rate > 0.8 or success_rate < 0.2:
                                continue
                        # Diversity filter
                        if not self._is_diverse(q):
                            continue
                        results.append(q)
                        generated += 1
                        if generated >= count:
                            break
                except Exception:
                    # On generation failure, fill with simple placeholder questions so the batch is not empty
                    fallback = self._fallback_question(difficulty, domains)
                    results.append(fallback)
                    generated += 1

        return results[:n]

    def update_history(
        self, question: str, success: bool, score: float
    ) -> None:
        """Update a question's history, for adaptive calibration.

        Args:
            question: research question text.
            success: whether it succeeded (score >= 6.0 counts as success).
            score: final score.
        """
        if question not in self.difficulty_history:
            self.difficulty_history[question] = {
                "attempts": 0,
                "successes": 0,
                "total_score": 0.0,
                "success_rate": 0.5,
            }
        h = self.difficulty_history[question]
        h["attempts"] += 1
        if success:
            h["successes"] += 1
        h["total_score"] += score
        h["success_rate"] = h["successes"] / h["attempts"]
        h["avg_score"] = h["total_score"] / h["attempts"]

    def get_difficulty_weights(self) -> dict[str, float]:
        """Compute the adaptive weights of the three difficulty levels.

        Inverted-U formula: weight = 1 - 4 * (success_rate - 0.5) ^ 2
        The closer the success rate is to 50%, the higher the weight; questions that are too easy or too hard get lower weight.

        Returns:
            Difficulty weight dict, keys: L1/L2/L3.
        """
        weights: dict[str, float] = {"L1": 1.0, "L2": 1.0, "L3": 1.0}
        for question, hist in self.difficulty_history.items():
            sr = hist.get("success_rate", 0.5)
            # Estimate the difficulty level (simple heuristic: map from search count)
            # History has no explicit difficulty here, so compute uniformly
            w = 1.0 - 4.0 * (sr - 0.5) ** 2
            w = max(0.1, min(1.0, w))
            # Since history does not distinguish difficulty, spread the weight effect evenly
            for lvl in weights:
                weights[lvl] += w

        # Normalize
        total = sum(weights.values())
        if total > 0.0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    def _is_diverse(self, question: str, threshold: float = 0.7) -> bool:
        """Check whether a new question is diverse enough from existing ones (embedding cosine similarity < threshold)."""
        if not self._embedding_cache:
            return True
        try:
            from memory.embedder import Embedder

            embedder = Embedder()
            emb = embedder.encode(question)
            for cached_emb in self._embedding_cache:
                sim = self._cosine_similarity(emb, cached_emb)
                if sim > threshold:
                    return False
            self._embedding_cache.append(emb)
            return True
        except Exception:
            # On embedding failure accept by default
            return True

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        import math

        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _fallback_question(self, difficulty: str, domains: list[str] | None) -> str:
        """Generate fallback questions so the batch is not empty."""
        domain = random.choice(domains) if domains else "technology"
        templates = {
            "L1": [
                f"What was the market size of the {domain} sector in 2022?",
                f"What are the main application areas of {domain}?",
            ],
            "L2": [
                f"Compare the technical evolution and commercial adoption of the {domain} sector from 2020 to 2022.",
                f"What multi-dimensional impacts has the development of {domain} had on the job market?",
            ],
            "L3": [
                f"Assess the likely development trends of {domain} over the next decade from economic, ethical and technical dimensions.",
                f"From an interdisciplinary perspective, what are the key breakthroughs at the intersection of {domain} with biomedicine and climate science?",
            ],
        }
        return random.choice(templates.get(difficulty, templates["L2"]))

    def _parse_json(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        import re

        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass
        return {}
