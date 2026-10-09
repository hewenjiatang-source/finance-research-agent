"""
M5 Red-Blue adversarial denoising loop — verdict and data-structure layer

This module defines the core data structures of the adversarial loop (Issue / RedVerdict / FixOperation)
and the scoring engine VerdictEngine. All scores lie in [0.0, 10.0] so they align with human intuition.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


__all__ = [
    "Severity",
    "FixType",
    "Dimension",
    "Issue",
    "RedVerdict",
    "FixOperation",
    "VerdictEngine",
]


# ============================================================================
# Enum definitions
# ============================================================================

class Severity(Enum):
    """Issue severity level, used to compute fix priority."""
    CRITICAL = "critical"  # factual errors, core hallucinations
    MAJOR = "major"        # significant inconsistencies, important omissions
    MINOR = "minor"        # wording drift, minor source problems


class FixType(Enum):
    """Blue Agent fix strategy type."""
    IN_PLACE = "in_place"      # in-place correction: replace numbers/dates/names directly
    SUPPLEMENTARY = "search"   # supplementary search: unsourced claims -> trigger a new search
    REMOVAL = "removal"        # removal: delete high-confidence hallucinated paragraphs


class Dimension(Enum):
    """The five attack dimensions of the Red Agent."""
    FACTUAL = "fact_check"      # fact checking
    HALLUCINATION = "hallucination"  # hallucination detection
    LOGICAL = "logical"         # logical consistency
    SOURCE_CREDIBILITY = "source_credibility"  # source credibility
    COVERAGE = "coverage"       # coverage completeness


# ============================================================================
# Dataclass definitions
# ============================================================================

@dataclass
class Issue:
    """A single issue found by the Red Agent.

    Attributes:
        severity: severity level (critical / major / minor).
        dimension: the attack dimension it belongs to.
        description: natural-language description, guidance passed to the Blue Agent.
        location: marker for where the issue sits in the report, e.g. a paragraph index or citation mark.
        fix_type: suggested fix type.
        evidence: evidence snippet supporting the issue (e.g. original source text).
    """
    severity: Severity
    dimension: Dimension
    description: str
    location: str = ""
    fix_type: FixType = FixType.IN_PLACE
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "dimension": self.dimension.value,
            "description": self.description,
            "location": self.location,
            "fix_type": self.fix_type.value,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Issue":
        return cls(
            severity=Severity(data.get("severity", "minor")),
            dimension=Dimension(data.get("dimension", "fact_check")),
            description=data.get("description", ""),
            location=data.get("location", ""),
            fix_type=FixType(data.get("fix_type", "in_place")),
            evidence=data.get("evidence", ""),
        )

    def __hash__(self) -> int:
        """Used to de-duplicate the resolved_issues set: derives a deterministic hash from the core fields."""
        return hash((self.severity, self.dimension, self.description, self.location, self.fix_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Issue):
            return NotImplemented
        return (
            self.severity == other.severity
            and self.dimension == other.dimension
            and self.description == other.description
            and self.location == other.location
            and self.fix_type == other.fix_type
        )


@dataclass
class RedVerdict:
    """The full attack result of the Red Agent on one report.

    Attributes:
        dimension_scores: scores for the five dimensions; keys are Dimension, values are floats in [0, 10].
        overall_score: weighted overall score.
        issues: list of all issues found.
        raw_feedback: raw model output, kept for auditing and debugging.
    """
    dimension_scores: dict[Dimension, float] = field(default_factory=dict)
    overall_score: float = 0.0
    issues: list[Issue] = field(default_factory=list)
    raw_feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_scores": {k.value: v for k, v in self.dimension_scores.items()},
            "overall_score": self.overall_score,
            "issues": [i.to_dict() for i in self.issues],
            "raw_feedback": self.raw_feedback,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedVerdict":
        return cls(
            dimension_scores={
                Dimension(k): v
                for k, v in data.get("dimension_scores", {}).items()
            },
            overall_score=data.get("overall_score", 0.0),
            issues=[Issue.from_dict(i) for i in data.get("issues", [])],
            raw_feedback=data.get("raw_feedback", ""),
        )


@dataclass
class FixOperation:
    """Record of a single fix operation performed by the Blue Agent.

    Attributes:
        issue: the original issue that was fixed.
        action: description of the action actually taken.
        success: whether the fix passed self_verify.
        detail: detailed change content, e.g. before/after comparison.
    """
    issue: Issue
    action: str = ""
    success: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue.to_dict(),
            "action": self.action,
            "success": self.success,
            "detail": self.detail,
        }


# ============================================================================
# Scoring engine
# ============================================================================

class VerdictEngine:
    """Scoring engine of the Red-Blue adversarial loop.

    Design decisions:
    1. The five dimension weights are strictly aligned with the project plan and sum to 1.0.
    2. All input scores are assumed to be normalized to [0.0, 10.0].
    3. Round-trip serialization (dict / json) is provided for persistent auditing.
    """

    # Five dimension weights (consistent with the project plan)
    DIMENSION_WEIGHTS: dict[Dimension, float] = {
        Dimension.FACTUAL: 0.30,
        Dimension.HALLUCINATION: 0.25,
        Dimension.LOGICAL: 0.20,
        Dimension.SOURCE_CREDIBILITY: 0.15,
        Dimension.COVERAGE: 0.10,
    }

    # severity -> numeric mapping (used for priority computation)
    SEVERITY_WEIGHTS: dict[Severity, float] = {
        Severity.CRITICAL: 10.0,
        Severity.MAJOR: 5.0,
        Severity.MINOR: 1.0,
    }

    # fix_type -> difficulty coefficient (used for priority computation)
    FIX_DIFFICULTY: dict[FixType, float] = {
        FixType.IN_PLACE: 1.0,
        FixType.REMOVAL: 0.8,
        FixType.SUPPLEMENTARY: 0.6,
    }

    @classmethod
    def compute_overall(cls, dimension_scores: dict[Dimension, float]) -> float:
        """Compute the weighted overall score.

        Args:
            dimension_scores: dict of the five dimension scores, each should be in [0.0, 10.0].

        Returns:
            Weighted average score in [0.0, 10.0].
        """
        if not dimension_scores:
            return 0.0
        total = 0.0
        weight_sum = 0.0
        for dim, score in dimension_scores.items():
            w = cls.DIMENSION_WEIGHTS.get(dim, 0.0)
            total += w * max(0.0, min(10.0, score))
            weight_sum += w
        if weight_sum == 0.0:
            return 0.0
        return total / weight_sum

    @classmethod
    def compute_delta(
        cls,
        prev: dict[Dimension, float],
        curr: dict[Dimension, float],
    ) -> float:
        """Compute the score change Δ between two rounds (Euclidean distance).

        Design decision: Euclidean distance rather than a simple absolute difference captures multi-dimension fluctuation.
        If a dimension is missing from either dict, it is filled with 0.0.

        Args:
            prev: previous round's five dimension scores.
            curr: current round's five dimension scores.

        Returns:
            Non-negative float; smaller means a smoother change.
        """
        all_dims = set(prev.keys()) | set(curr.keys())
        if not all_dims:
            return 0.0
        sq_sum = 0.0
        for dim in all_dims:
            p = max(0.0, min(10.0, prev.get(dim, 0.0)))
            c = max(0.0, min(10.0, curr.get(dim, 0.0)))
            sq_sum += (c - p) ** 2
        return math.sqrt(sq_sum)

    @classmethod
    def compute_priority(cls, issue: Issue) -> float:
        """Compute the fix priority of an Issue.

        Formula: priority = severity_weight × dimension_weight × fix_difficulty
        The higher the priority, the sooner it should be handled.

        Args:
            issue: the issue whose priority is computed.

        Returns:
            Priority score (unbounded; larger means more urgent).
        """
        sw = cls.SEVERITY_WEIGHTS.get(issue.severity, 1.0)
        dw = cls.DIMENSION_WEIGHTS.get(issue.dimension, 0.1)
        fd = cls.FIX_DIFFICULTY.get(issue.fix_type, 1.0)
        return sw * dw * fd

    @staticmethod
    def to_json(verdict: RedVerdict, indent: int = 2) -> str:
        """Serialize a RedVerdict to a JSON string."""
        return json.dumps(verdict.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def from_json(raw: str) -> RedVerdict:
        """Deserialize a RedVerdict from a JSON string."""
        return RedVerdict.from_dict(json.loads(raw))
