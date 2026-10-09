"""
M6 self-evolution engine — training-loop orchestration (Self-Evolution Engine)

SelfEvolutionEngine implements the full MAE (Maker-Advisor-Evaluator) triangle architecture:
- Proposer: generates research questions (L1/L2/L3)
- Solver: the DeepResearch Agent itself
- Judge: five-dimension continuous reward scoring
- GRPO Trainer: reuses project one's veRL framework

Design decisions:
1. Each round generates 32 questions, expanded to 32 × 8 group = 256 trajectories, keeping gradient variance under control.
2. After Judge scoring, shape_reward maps to [-1, 1], fed directly into the veRL GRPO trainer.
3. Symbolic Learning triggers every 3 rounds and Judge calibration every 5 rounds.
4. All intermediate data (parquet, checkpoint, log) is isolated in per-round directories for traceability.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from src.evolution.collector import TrajectoryCollector
from src.evolution.experience_memory import ExperienceMemory
from src.evolution.judge import Judge
from src.evolution.proposer import Proposer
from src.evolution.symbolic_learning import SymbolicLearner
from src.orchestrator.schemas import ResearchReport


__all__ = ["SelfEvolutionEngine"]

logger = logging.getLogger(__name__)


class SelfEvolutionEngine:
    """Main controller of the self-evolution engine.

    Attributes:
        proposer: question generator.
        solver: DeepResearch Agent (must implement the async run(query: str) -> ResearchReport interface).
        judge: multi-dimension scorer.
        trainer_config: veRL GRPO training config dict.
        collector: trajectory collector.
        experience_memory: experience memory store.
        symbolic_learner: prompt self-optimizer (optional).
        output_dir: root output directory for per-round data.
    """

    def __init__(
        self,
        proposer: Proposer,
        solver: Any,
        judge: Judge,
        trainer_config: dict[str, Any],
        collector: TrajectoryCollector | None = None,
        experience_memory: ExperienceMemory | None = None,
        symbolic_learner: SymbolicLearner | None = None,
        output_dir: str = "./evolution_output",
    ):
        self.proposer = proposer
        self.solver = solver
        self.judge = judge
        self.trainer_config = trainer_config
        self.collector = collector or TrajectoryCollector()
        self.experience_memory = experience_memory or ExperienceMemory()
        self.symbolic_learner = symbolic_learner
        self.output_dir = output_dir

        # Runtime state
        self.current_round: int = 0
        self._failed_trajectories: list[dict[str, Any]] = []
        self._current_prompts: dict[str, str] = {}

    async def run(self, num_rounds: int = 20) -> dict[str, Any]:
        """Run multiple rounds of self-evolution.

        Args:
            num_rounds: total number of evolution rounds.

        Returns:
            Summary statistics dict.
        """
        logger.info(f"[SelfEvolutionEngine] Starting {num_rounds} rounds of self-evolution")
        summary = {"rounds": [], "final_avg_score": 0.0}

        for r in range(1, num_rounds + 1):
            self.current_round = r
            logger.info(f"[SelfEvolutionEngine] === Round {r} ===")
            round_result = await self.run_round()
            summary["rounds"].append(round_result)

        # Compute the final average score
        scores = [r["avg_score"] for r in summary["rounds"] if "avg_score" in r]
        summary["final_avg_score"] = sum(scores) / len(scores) if scores else 0.0
        return summary

    async def run_round(self) -> dict[str, Any]:
        """Run one round of self-evolution.

        Full flow:
        1. Proposer.generate_batch() → 32 questions
        2. Solver runs in parallel -> 32 reports + trajectories
        3. Collector gathers -> veRL-format data
        4. Judge.evaluate() -> five-dimension continuous reward
        5. shape_reward_for_grpo() -> single-value reward [-1, 1]
        6. build_evolution_parquet() -> standard veRL parquet
        7. veRL GRPO trainer: 50 steps (reuses project one's veRL)
        8. Experience Memory update
        9. (every 3 rounds) Symbolic Learning -> optimize prompts
        10. (every 5 rounds) Judge calibration
        11. Proposer.update_history()

        Returns:
            This round's statistics dict.
        """
        round_stats: dict[str, Any] = {"round": self.current_round}

        # =====================================================================
        # Step 1: generate research questions
        # =====================================================================
        questions = await self.proposer.generate_batch(n=32)
        round_stats["num_questions"] = len(questions)
        logger.info(f"[Round {self.current_round}] Generated {len(questions)} questions")

        # =====================================================================
        # Step 2: Solver execution (parallel)
        # =====================================================================
        reports: list[ResearchReport] = []
        trajectories_list: list[list[dict[str, Any]]] = []

        # Run in parallel with asyncio.gather, but cap concurrency to avoid OOM
        semaphore = asyncio.Semaphore(self.trainer_config.get("max_concurrent_solve", 8))

        async def _solve_one(q: str) -> tuple[ResearchReport, list[dict[str, Any]]]:
            async with semaphore:
                # The Solver must implement async run(query) -> (report, trajectory)
                if hasattr(self.solver, "run"):
                    if asyncio.iscoroutinefunction(self.solver.run):
                        result = await self.solver.run(q)
                    else:
                        result = self.solver.run(q)
                else:
                    # fallback: return an empty report
                    result = ResearchReport(query=q, content="")

                # Unified return format
                if isinstance(result, tuple) and len(result) == 2:
                    report, traj = result
                elif isinstance(result, ResearchReport):
                    report = result
                    traj = []
                else:
                    report = ResearchReport(query=q, content=str(result))
                    traj = []
                return report, traj

        solve_tasks = [_solve_one(q) for q in questions]
        solve_results = await asyncio.gather(*solve_tasks, return_exceptions=True)

        for q, res in zip(questions, solve_results):
            if isinstance(res, Exception):
                logger.warning(f"[Round {self.current_round}] Solver failed for query={q}: {res}")
                reports.append(ResearchReport(query=q, content=""))
                trajectories_list.append([])
            else:
                reports.append(res[0])
                trajectories_list.append(res[1])

        # =====================================================================
        # Step 3: Collector gathers and converts the format
        # =====================================================================
        collected_batch: list[dict[str, Any]] = []
        for q, report, traj in zip(questions, reports, trajectories_list):
            collected = self.collector.collect(q, report, traj)
            collected_batch.append(collected)

        verl_data = self.collector.batch_to_verl(collected_batch)
        round_stats["num_trajectories"] = len(verl_data)

        # =====================================================================
        # Step 4 & 5: Judge scoring + reward shaping
        # =====================================================================
        rewards: list[float] = []
        scores_list: list[dict[str, float]] = []
        for report in reports:
            if not report.content:
                # Empty reports get the lowest score
                rewards.append(-1.0)
                scores_list.append({})
                continue
            try:
                scores = await self.judge.evaluate(report)
                r = self.judge.shape_reward(scores)
                rewards.append(r)
                scores_list.append(scores)
            except Exception as e:
                logger.warning(f"Judge failed: {e}")
                rewards.append(0.0)
                scores_list.append({})

        round_stats["avg_reward"] = sum(rewards) / len(rewards) if rewards else 0.0
        round_stats["avg_score"] = (
            sum(
                s.get("factual_accuracy", 0.0) * 0.3
                + s.get("coverage", 0.0) * 0.25
                + s.get("logical_coherence", 0.0) * 0.2
                + s.get("citation_quality", 0.0) * 0.15
                + s.get("efficiency", 0.0) * 0.1
                for s in scores_list
                if s
            )
            / max(len([s for s in scores_list if s]), 1)
        )

        # Write the reward back into verl_data
        for item, r in zip(verl_data, rewards):
            item["reward"] = r

        # =====================================================================
        # Step 6: build the evolution parquet
        # =====================================================================
        round_dir = os.path.join(self.output_dir, f"round_{self.current_round:03d}")
        os.makedirs(round_dir, exist_ok=True)
        parquet_path = os.path.join(round_dir, "evolution_data.parquet")

        try:
            import pandas as pd

            df = pd.DataFrame(verl_data)
            df.to_parquet(parquet_path, index=False)
            round_stats["parquet_path"] = parquet_path
            logger.info(f"[Round {self.current_round}] Parquet saved to {parquet_path}")
        except Exception as e:
            logger.warning(f"Failed to save parquet: {e}")
            round_stats["parquet_path"] = ""

        # =====================================================================
        # Step 7: veRL GRPO training (reuses project one's veRL)
        # =====================================================================
        # NOTE: training is done by calling the veRL GRPO trainer already implemented in project one.
        # The user must ensure trainer_config contains all necessary parameters (model_path, rollout_size, etc.).
        # The following is a pseudo-code skeleton showing how to hook into project one's veRL:
        #
        # from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        # from verl.utils.dataset.rl_dataset import RLHFDataset
        #
        # dataset = RLHFDataset(parquet_files=[parquet_path], ...)
        # trainer = RayPPOTrainer(config=self.trainer_config, dataset=dataset)
        # trainer.fit()
        #
        # After training, the updated model weights are saved to checkpoint_dir automatically.
        # =====================================================================
        checkpoint_dir = os.path.join(round_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        round_stats["checkpoint_dir"] = checkpoint_dir
        logger.info(
            f"[Round {self.current_round}] GRPO training placeholder: "
            f"load {parquet_path} -> veRL trainer -> save to {checkpoint_dir}"
        )

        # =====================================================================
        # Step 8: Experience Memory update
        # =====================================================================
        success_threshold = self.trainer_config.get("success_threshold", 6.0)
        for q, report, traj, scores, r in zip(
            questions, reports, trajectories_list, scores_list, rewards
        ):
            success = r >= (success_threshold / 5.0 - 1.0)  # map to [-1, 1]
            score = max(scores.values()) if scores else 0.0
            strategy_summary = self._summarize_strategy(traj)
            self.experience_memory.add(
                trajectory=traj,
                success=success,
                score=score,
                strategy_summary=strategy_summary,
                current_round=self.current_round,
            )

        # Evict old experiences
        evicted = self.experience_memory.evict_old_experiences(
            max_age_rounds=5,
            current_round=self.current_round,
        )
        round_stats["evicted_experiences"] = evicted

        # =====================================================================
        # Step 9: (every 3 rounds) Symbolic Learning
        # =====================================================================
        if self.symbolic_learner is not None and self.current_round % 3 == 0:
            logger.info(f"[Round {self.current_round}] Triggering Symbolic Learning...")
            # Collect this round's failed trajectories
            failed = [
                collected_batch[i]
                for i, r in enumerate(rewards)
                if r < 0.0
            ]
            self._failed_trajectories.extend(failed)
            # Keep at most the latest 100 failed trajectories
            self._failed_trajectories = self._failed_trajectories[-100:]

            new_prompts = await self.symbolic_learner.optimize_prompts(
                failed_trajectories=self._failed_trajectories,
                current_prompts=self._current_prompts,
            )
            # Rollback check
            performance = {"avg_score": round_stats.get("avg_score", 0.0)}
            final_prompts = self.symbolic_learner.rollback_if_needed(
                new_prompts=new_prompts,
                performance=performance,
            )
            self._current_prompts = final_prompts
            round_stats["symbolic_learning_triggered"] = True
            round_stats["prompt_changed"] = new_prompts != final_prompts
        else:
            round_stats["symbolic_learning_triggered"] = False

        # =====================================================================
        # Step 10: (every 5 rounds) Judge calibration
        # =====================================================================
        if self.current_round % 5 == 0:
            logger.info(f"[Round {self.current_round}] Triggering Judge Calibration...")
            calib_result = self.judge.calibrate()
            round_stats["judge_calibration"] = calib_result
        else:
            round_stats["judge_calibration"] = {"status": "skipped"}

        # =====================================================================
        # Step 11: Proposer history update
        # =====================================================================
        for q, scores in zip(questions, scores_list):
            if not scores:
                continue
            composite = (
                scores.get("factual_accuracy", 0.0) * 0.3
                + scores.get("coverage", 0.0) * 0.25
                + scores.get("logical_coherence", 0.0) * 0.2
                + scores.get("citation_quality", 0.0) * 0.15
                + scores.get("efficiency", 0.0) * 0.1
            )
            success = composite >= success_threshold
            self.proposer.update_history(q, success, composite)

        logger.info(
            f"[Round {self.current_round}] Done: avg_reward={round_stats['avg_reward']:.3f}, "
            f"avg_score={round_stats['avg_score']:.3f}"
        )
        return round_stats

    def _summarize_strategy(self, trajectory: list[dict[str, Any]]) -> str:
        """Extract a strategy summary from a trajectory, for the Experience Memory embedding."""
        if not trajectory:
            return "empty_trajectory"
        # Extract all tool-call names as the strategy fingerprint
        tool_names: list[str] = []
        for step in trajectory:
            tool_calls = step.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    name = tc.get("function", {}).get("name", "") if isinstance(tc, dict) else ""
                    if name:
                        tool_names.append(name)
        if tool_names:
            return "strategy: " + " -> ".join(tool_names)
        # fallback: use the first 100 characters of the content
        first_content = str(trajectory[0].get("content", ""))[:100]
        return first_content or "unknown_strategy"
