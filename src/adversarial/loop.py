"""
M5 Red-Blue adversarial denoising loop — main controller

AdversarialLoop drives the full adversarial flow Red Agent -> Blue Agent -> scoring,
with robust mechanisms for dead-loop detection, oscillation detection and convergence checks.

Design decisions:
1. Maintain a resolved_issues set: an issue that was fixed but reappears in a later round is treated as oscillation.
2. Any one of three convergence conditions: round >= 3 / overall >= 8.0 / Δscore < 0.3.
3. The full score history is recorded each round for later analysis and auditing.
"""
from __future__ import annotations

import copy
import logging
from typing import Any

from src.adversarial.blue_agent import BlueAgent
from src.adversarial.red_agent import RedAgent
from src.adversarial.verdict import (
    Dimension,
    FixOperation,
    Issue,
    RedVerdict,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_chain


__all__ = ["AdversarialLoop"]

logger = logging.getLogger(__name__)


class AdversarialLoop:
    """Main controller of the Red-Blue adversarial denoising loop.

    Attributes:
        red_agent: Red Agent instance, responsible for attacking.
        blue_agent: Blue Agent instance, responsible for fixing.
        policy: policy object used for self_verify or auxiliary scoring (optional).
        max_rounds: hard cap on the number of rounds.
        score_threshold: overall-score threshold for success.
        delta_threshold: convergence threshold on the between-round change.
    """

    def __init__(
        self,
        red_agent: RedAgent,
        blue_agent: BlueAgent,
        policy: Any | None = None,
        max_rounds: int = 3,
        score_threshold: float = 8.0,
        delta_threshold: float = 0.3,
    ):
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.policy = policy
        self.max_rounds = max(max_rounds, 1)
        self.score_threshold = score_threshold
        self.delta_threshold = delta_threshold

    @trace_chain(name="adversarial_loop.run", tags=["m5", "loop", "adversarial"])
    async def run(
        self, report: ResearchReport
    ) -> tuple[ResearchReport, list[dict[str, Any]]]:
        """Run the full adversarial denoising loop.

        Flow:
        1. Each round the Red Agent attacks the current report.
        2. The Blue Agent fixes it according to the Verdict.
        3. Record this round's scores and fix operations.
        4. Check convergence conditions or oscillation / dead loops.
        5. Return the final report and the full history.

        Args:
            report: initial research report (not modified; deep-copied internally).

        Returns:
            (fixed report, list of per-round score records)
            Each record contains: round, dimension_scores, overall_score, delta, issues_count,
            fix_operations, resolved_count, oscillation_detected, stop_reason
        """
        current = copy.deepcopy(report)
        history: list[dict[str, Any]] = []
        prev_scores: dict[Dimension, float] | None = None
        resolved_issues: set[Issue] = set()  # set of fixed issues
        oscillation_detected = False
        stop_reason = ""

        for round_idx in range(1, self.max_rounds + 1):
            logger.info(f"[AdversarialLoop] Round {round_idx} starting...")

            # ---- Step 1: Red Attack ----
            verdict = await self.red_agent.attack(current)
            logger.info(
                f"[AdversarialLoop] Red attack done: overall={verdict.overall_score:.2f}, "
                f"issues={len(verdict.issues)}"
            )

            # ---- Step 2: oscillation detection ----
            # If a previously fixed issue reappears among the current issues, treat it as oscillation
            reappeared = resolved_issues.intersection(set(verdict.issues))
            if reappeared:
                oscillation_detected = True
                logger.warning(
                    f"[AdversarialLoop] Oscillation detected at round {round_idx}: "
                    f"{len(reappeared)} previously resolved issues reappeared."
                )
                stop_reason = f"oscillation_at_round_{round_idx}"
                history.append(self._build_round_record(
                    round_idx, verdict, [], len(reappeared), oscillation_detected, stop_reason
                ))
                break

            # ---- Step 3: Blue Defend ----
            fixed_report, operations = await self.blue_agent.defend(current, verdict)
            logger.info(
                f"[AdversarialLoop] Blue defend done: operations={len(operations)}"
            )

            # Add the issues fixed this round to the resolved set
            for op in operations:
                if op.success:
                    resolved_issues.add(op.issue)

            # ---- Step 4: compute delta ----
            delta = 0.0
            if prev_scores is not None:
                delta = VerdictEngine.compute_delta(prev_scores, verdict.dimension_scores)
            prev_scores = copy.deepcopy(verdict.dimension_scores)

            # ---- Step 5: record this round ----
            stop_reason = self._check_convergence(round_idx, verdict.overall_score, delta)
            record = self._build_round_record(
                round_idx=round_idx,
                verdict=verdict,
                operations=operations,
                resolved_count=len(resolved_issues),
                oscillation=oscillation_detected,
                stop_reason=stop_reason,
            )
            history.append(record)

            # Update the current report to the fixed version
            current = fixed_report
            current.adversarial_rounds = round_idx

            # ---- Step 6: decide whether to stop ----
            if stop_reason:
                logger.info(f"[AdversarialLoop] Stopping: {stop_reason}")
                break

        # Write the final scores after the loop ends
        if history:
            current.final_score = history[-1]["overall_score"]

        return current, history

    def _check_convergence(
        self, round_idx: int, overall_score: float, delta: float
    ) -> str:
        """Check whether any convergence condition is met.

        Returns:
            Empty string means continue; a non-empty string is the stop reason.
        """
        if round_idx >= self.max_rounds:
            return f"max_rounds_reached({self.max_rounds})"
        if overall_score >= self.score_threshold:
            return f"score_threshold_met({overall_score:.2f}>={self.score_threshold})"
        # The first round has no previous round to compare against, so delta is always 0; skip delta convergence check
        if round_idx > 1 and delta < self.delta_threshold:
            return f"delta_converged({delta:.3f}<{self.delta_threshold})"
        return ""

    def _build_round_record(
        self,
        round_idx: int,
        verdict: RedVerdict,
        operations: list[FixOperation],
        resolved_count: int,
        oscillation: bool,
        stop_reason: str,
    ) -> dict[str, Any]:
        """Build a single-round record dict."""
        return {
            "round": round_idx,
            "dimension_scores": {
                k.value: round(v, 3) for k, v in verdict.dimension_scores.items()
            },
            "overall_score": round(verdict.overall_score, 3),
            "issues_count": len(verdict.issues),
            "fix_operations": [op.to_dict() for op in operations],
            "resolved_count": resolved_count,
            "oscillation_detected": oscillation,
            "stop_reason": stop_reason,
            "raw_feedback": verdict.raw_feedback,
        }
