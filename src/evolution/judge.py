"""
M6 self-evolution engine — multi-dimension scorer (Judge)

The Judge scores research reports continuously along five dimensions, supporting an Ensemble (mean of 3 different prompts),
a rule-based efficiency formula, reward shaping and related mechanisms.

Design decisions:
1. Ensemble: 3 prompts with different viewpoints score independently and are averaged, lowering single-prompt bias.
2. Efficiency score: a pure rule formula (sigmoid decay) to prevent reward hacking (a model could farm score via pointless searches).
3. Reward shaping: maps the [0,10] composite score to [-1,1], fitting GRPO's clip range.
4. Held-out calibration interface: periodically compare with human labels to expose drift.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from src.orchestrator.schemas import ResearchReport


__all__ = ["Judge"]


# ============================================================================
# Prompt templates — 3 Ensemble viewpoints
# ============================================================================

SYSTEM_JUDGE = (
    "You are a professional research-report review expert. Score the report on the following five dimensions, "
    "each 0-10. Output must be strict JSON."
)

# Viewpoint 1: academic rigor
PROMPT_JUDGE_1 = """Review the following research report from the **academic rigor** viewpoint.

Scoring dimensions:
1. factual_accuracy (0-10): factual accuracy — is every claim backed by a reliable source.
2. coverage (0-10): coverage — does it fully answer all sub-topics of the query.
3. logical_coherence (0-10): logic — is the argument tight and free of self-contradiction.
4. citation_quality (0-10): citation quality — are sources authoritative and citations well formed.
5. efficiency (0-10): efficiency score (computed by system rules; you only score the first 4).

Output in the following JSON format (no extra text):
{
  "factual_accuracy": float,
  "coverage": float,
  "logical_coherence": float,
  "citation_quality": float,
  "rationale": "string"   // brief rationale for the scores
}

--- Original question ---
{query}

--- Research report ---
{content}

--- Source list ---
{sources}
"""

# Viewpoint 2: practicality-oriented
PROMPT_JUDGE_2 = """Review the following research report from the **practicality** viewpoint.

Focus on:
- Does the report directly answer the user's question?
- Is the information actually helpful for decisions?
- Is the structure clear and easy to read?

Scoring dimensions:
1. factual_accuracy (0-10)
2. coverage (0-10)
3. logical_coherence (0-10)
4. citation_quality (0-10)
5. efficiency (0-10)

Output in the following JSON format:
{
  "factual_accuracy": float,
  "coverage": float,
  "logical_coherence": float,
  "citation_quality": float,
  "rationale": "string"
}

--- Original question ---
{query}

--- Research report ---
{content}

--- Source list ---
{sources}
"""

# Viewpoint 3: critical fault-finding
PROMPT_JUDGE_3 = """Strictly review the following research report from the **critical fault-finding** viewpoint.

Your task is to find as many defects in the report as possible:
- Any specific number without source support
- Any over-generalized conclusion
- Any logical leap
- Any missing key perspective

Scoring dimensions (be deliberately strict):
1. factual_accuracy (0-10)
2. coverage (0-10)
3. logical_coherence (0-10)
4. citation_quality (0-10)
5. efficiency (0-10)

Output in the following JSON format:
{
  "factual_accuracy": float,
  "coverage": float,
  "logical_coherence": float,
  "citation_quality": float,
  "rationale": "string"
}

--- Original question ---
{query}

--- Research report ---
{content}

--- Source list ---
{sources}
"""

JUDGE_PROMPTS = [PROMPT_JUDGE_1, PROMPT_JUDGE_2, PROMPT_JUDGE_3]

# Five dimension weights (consistent with the project plan)
DIMENSION_WEIGHTS = {
    "factual_accuracy": 0.30,
    "coverage": 0.25,
    "logical_coherence": 0.20,
    "citation_quality": 0.15,
    "efficiency": 0.10,
}


# ============================================================================
# Judge implementation
# ============================================================================

@dataclass
class CalibrationSample:
    """Held-out calibration sample."""
    query: str
    report_content: str
    human_scores: dict[str, float]
    model_scores: dict[str, float] | None = None


class Judge:
    """Multi-dimension scorer supporting Ensemble and reward shaping.

    Attributes:
        policy: a VLLMPolicy instance.
        ensemble_size: number of Judges in the Ensemble (default 3).
        efficiency_optimal: optimal number of searches for the efficiency score.
        efficiency_scale: sigmoid decay scale of the efficiency score.
    """

    def __init__(
        self,
        policy,
        ensemble_size: int = 3,
        efficiency_optimal: int = 5,
        efficiency_scale: float = 3.0,
    ):
        self.policy = policy
        self.ensemble_size = min(max(ensemble_size, 1), len(JUDGE_PROMPTS))
        self.efficiency_optimal = efficiency_optimal
        self.efficiency_scale = efficiency_scale
        # held-out calibration sample pool
        self._calibration_pool: list[CalibrationSample] = []

    async def evaluate(
        self, report: ResearchReport, query: str | None = None
    ) -> dict[str, float]:
        """Score a research report on five dimensions.

        Flow:
        1. Call ensemble_size Judges with different prompts.
        2. Average each dimension.
        3. Compute the efficiency score with the rule formula.
        4. Return the five-dimension score dict.

        Args:
            report: the research report to score.
            query: optional original question (defaults to report.query).

        Returns:
            Five-dimension score dict, keys: factual_accuracy / coverage / logical_coherence /
            citation_quality / efficiency, values in [0.0, 10.0].
        """
        q = query or report.query
        sources_text = self._format_sources(report.sources)

        # Collect each Judge's scores in the ensemble
        dim_lists: dict[str, list[float]] = {
            "factual_accuracy": [],
            "coverage": [],
            "logical_coherence": [],
            "citation_quality": [],
        }
        rationales: list[str] = []

        for i in range(self.ensemble_size):
            prompt_template = JUDGE_PROMPTS[i]
            prompt = prompt_template.format(
                query=q,
                content=report.content,
                sources=sources_text,
            )
            messages = [
                {"role": "system", "content": SYSTEM_JUDGE},
                {"role": "user", "content": prompt},
            ]
            try:
                resp = self.policy(messages)
                raw = resp.content or ""
                scores, rationale = self._parse_judge_output(raw)
                for dim in dim_lists:
                    dim_lists[dim].append(scores.get(dim, 5.0))
                rationales.append(rationale)
            except Exception as e:
                # A single Judge failing does not interrupt; fill with a conservative score
                for dim in dim_lists:
                    dim_lists[dim].append(5.0)
                rationales.append(f"judge_{i}_error: {e}")

        # Take the ensemble mean
        final_scores: dict[str, float] = {}
        for dim, vals in dim_lists.items():
            final_scores[dim] = sum(vals) / len(vals) if vals else 5.0

        # Efficiency score: rule formula, to prevent reward hacking
        final_scores["efficiency"] = self._compute_efficiency_score(report.num_searches)

        # Record the rationale in an internal field (for debugging)
        self._last_rationales = rationales
        return final_scores

    def shape_reward(self, scores: dict[str, float]) -> float:
        """Convert the five-dimension scores into a single-value reward usable by GRPO.

        Formula: R_grpo = clip(composite * 2 - 1, -1, 1)
        where composite is the weighted mean of the five dimensions, in [0, 10].

        Args:
            scores: the five-dimension score dict returned by evaluate().

        Returns:
            Single-value reward in [-1.0, 1.0].
        """
        composite = 0.0
        weight_sum = 0.0
        for dim, weight in DIMENSION_WEIGHTS.items():
            s = scores.get(dim, 0.0)
            composite += weight * max(0.0, min(10.0, s))
            weight_sum += weight
        if weight_sum == 0.0:
            composite = 0.0
        else:
            composite /= weight_sum

        # Map to [-1, 1]
        r = composite * 2.0 - 1.0
        return max(-1.0, min(1.0, r))

    def _compute_efficiency_score(self, num_searches: int) -> float:
        """Compute the efficiency score: sigmoid decay.

        Formula: score = 10.0 / (1.0 + exp((num_searches - optimal) / scale))
        The closer the search count is to optimal, the higher the score; excessive searching is penalized significantly.

        Args:
            num_searches: actual number of searches.

        Returns:
            Efficiency score in (0.0, 10.0].
        """
        exp_term = math.exp((num_searches - self.efficiency_optimal) / self.efficiency_scale)
        score = 10.0 / (1.0 + exp_term)
        return score

    def add_calibration_sample(
        self, query: str, report_content: str, human_scores: dict[str, float]
    ) -> None:
        """Add a held-out calibration sample.

        Args:
            query: research question.
            report_content: report body.
            human_scores: human-labelled five-dimension scores.
        """
        self._calibration_pool.append(
            CalibrationSample(
                query=query,
                report_content=report_content,
                human_scores=human_scores,
            )
        )

    def calibrate(self) -> dict[str, float]:
        """Run held-out calibration: compare model scores with human labels.

        Returns:
            Calibration metrics dict, containing per-dimension mean absolute error (MAE) and overall correlation.
        """
        if not self._calibration_pool:
            return {"status": "no_samples"}

        dim_mae: dict[str, list[float]] = {
            "factual_accuracy": [],
            "coverage": [],
            "logical_coherence": [],
            "citation_quality": [],
            "efficiency": [],
        }

        # calibrate is a synchronous method, so only the existing model_scores are compared here
        # If model_scores is None it has not been evaluated yet; the caller must call evaluate first to fill it
        for sample in self._calibration_pool:
            if sample.model_scores is None:
                continue
            for dim in dim_mae:
                human = sample.human_scores.get(dim, 0.0)
                model = sample.model_scores.get(dim, 0.0)
                dim_mae[dim].append(abs(human - model))

        result: dict[str, float] = {}
        for dim, errs in dim_mae.items():
            if errs:
                result[f"{dim}_mae"] = sum(errs) / len(errs)
            else:
                result[f"{dim}_mae"] = -1.0  # mark as not computed

        result["sample_count"] = float(len(self._calibration_pool))
        return result

    def update_model_scores_for_calibration(self, idx: int, model_scores: dict[str, float]) -> None:
        """Update the model scores for the calibration sample at the given index.

        Args:
            idx: calibration sample index.
            model_scores: five-dimension scores given by the model.
        """
        if 0 <= idx < len(self._calibration_pool):
            self._calibration_pool[idx].model_scores = model_scores

    def _format_sources(self, sources: list[dict]) -> str:
        if not sources:
            return "(no sources)"
        lines = []
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Unknown title")
            url = s.get("url", "")
            lines.append(f"[{i}] {title} ({url})")
        return "\n".join(lines)

    def _parse_judge_output(self, raw: str) -> tuple[dict[str, float], str]:
        """Parse the Judge's JSON output."""
        raw = raw.strip()
        if not raw:
            return {}, ""

        import re

        # Try to extract the JSON block
        candidates = [raw]
        code_match = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        candidates.extend(code_match.findall(raw))
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))

        for cand in candidates:
            try:
                data = json.loads(cand.strip())
                scores = {}
                for dim in ["factual_accuracy", "coverage", "logical_coherence", "citation_quality"]:
                    scores[dim] = float(data.get(dim, 5.0))
                rationale = data.get("rationale", "")
                return scores, rationale
            except (json.JSONDecodeError, ValueError):
                continue

        return {}, "parse_failed"
