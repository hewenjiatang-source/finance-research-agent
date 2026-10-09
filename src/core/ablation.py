#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/ablation.py
================================================================================
General framework for ablation experiments.

Public interface:
    - AblationStudy.run_module_ablation(config, questions, systems) -> dict
    - AblationStudy.run_rounds_ablation(config, questions, max_rounds) -> dict
================================================================================
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from .runner import initialize_modules, run_research

logger = logging.getLogger("ablation")


class AblationStudy:
    """Ablation framework: supports module ablation and adversarial-round ablation."""

    # default configuration mapping for module ablation
    DEFAULT_MODULE_ABLATIONS: dict[str, tuple[str, dict]] = {
        "full": ("Full system", {}),
        "no_adversarial": ("Adversarial denoising off", {"adversarial": {"enabled": False}}),
        "no_compressor": ("Context compression off", {"compressor": {"enable_multilevel": False}}),
        "no_memory": ("Memory store off", {"memory": {"enabled": False}}),
        "no_evolution": ("Evolution learning off", {"evolution": {"enabled": False}}),
    }

    @staticmethod
    def override_config(config: dict, overrides: dict) -> dict:
        """Deep-merge configuration overrides (supports nested dicts)."""
        cfg = copy.deepcopy(config)

        def _deep_merge(base: dict, patch: dict) -> dict:
            for key, value in patch.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    base[key] = _deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        return _deep_merge(cfg, overrides)

    # -----------------------------------------------------------------------
    # Module ablation: full / no_XXX
    # -----------------------------------------------------------------------
    @classmethod
    def run_module_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        systems: dict[str, tuple[str, dict]] | None = None,
    ) -> dict[str, Any]:
        """
        Run the module ablation experiment.

        Args:
            config: base configuration.
            questions: list of evaluation questions (each with id, query).
            systems: ablation configuration mapping. Keys are system_name, values are (description, config override).
                     DEFAULT_MODULE_ABLATIONS is used by default.

        Returns:
            a dict containing the scores and details of each system.
        """
        if systems is None:
            systems = cls.DEFAULT_MODULE_ABLATIONS

        results: list[dict[str, Any]] = []

        for name, (desc, overrides) in systems.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"[Ablation] {name}: {desc}")
            logger.info(f"{'='*60}")

            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)

            scores: list[float] = []
            details: list[dict[str, Any]] = []

            for q in questions:
                qid = q.get("id", "unknown")
                query = q.get("query", "")
                logger.info(f"  [{qid}] {query[:60]}...")

                start = time.time()
                try:
                    report = asyncio.run(run_research(query, cfg, modules))
                    elapsed = time.time() - start

                    # scores are injected by the external caller (avoids evaluation/ depending back)
                    # only the raw report and metadata are recorded here
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "elapsed_seconds": elapsed,
                        "report_length": len(report),
                        "system": name,
                    })
                    scores.append(1.0)  # placeholder; the real score is filled in by the external evaluator
                    logger.info(f"    → success, time={elapsed:.1f}s, len={len(report)}")

                except Exception as e:
                    logger.warning(f"    → failed: {e}")
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "error": str(e),
                        "system": name,
                    })
                    scores.append(0.0)

            results.append({
                "system_name": name,
                "description": desc,
                "num_questions": len(questions),
                "average_composite_score": sum(scores) / len(scores) if scores else 0.0,
                "details": details,
            })

        return {
            "evaluation_name": "DeepResearch Agent module ablation experiment",
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "systems": results,
            "summary": {r["system_name"]: r["average_composite_score"] for r in results},
        }

    # -----------------------------------------------------------------------
    # Adversarial-round ablation: 0/1/2/3 rounds
    # -----------------------------------------------------------------------
    @classmethod
    def run_rounds_ablation(
        cls,
        config: dict,
        questions: list[dict[str, Any]],
        max_rounds: int = 3,
    ) -> dict[str, Any]:
        """
        Run the evaluation under different numbers of adversarial rounds.

        Args:
            config: base configuration.
            questions: list of evaluation questions.
            max_rounds: maximum number of adversarial rounds.

        Returns:
            result dicts keyed adv_0 / adv_1 / ... / adv_N.
        """
        summary: dict[str, float] = {}
        full_details: dict[str, Any] = {}

        for rounds in range(max_rounds + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Running with adversarial rounds = {rounds}")
            logger.info(f"{'='*50}")

            overrides = {
                "adversarial": {
                    "max_rounds": rounds,
                    "enabled": rounds > 0,
                }
            }
            cfg = cls.override_config(config, overrides)
            modules = initialize_modules(cfg)

            scores: list[float] = []
            details: list[dict[str, Any]] = []

            for idx, q in enumerate(questions, 1):
                qid = q.get("id", f"q{idx}")
                query = q.get("query", "")
                logger.info(f"  [{idx}/{len(questions)}] {qid}")

                try:
                    report = asyncio.run(run_research(query, cfg, modules))
                    scores.append(1.0)  # placeholder
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "rounds": rounds,
                        "report_length": len(report),
                    })
                except Exception as e:
                    logger.warning(f"    → failed: {e}")
                    scores.append(0.0)
                    details.append({
                        "question_id": qid,
                        "query": query,
                        "rounds": rounds,
                        "error": str(e),
                    })

            avg_score = sum(scores) / len(scores) if scores else 0.0
            key = f"adv_{rounds}"
            summary[key] = avg_score
            full_details[key] = details
            logger.info(f"Adversarial rounds {rounds} average score: {avg_score:.4f}")

        return {
            "evaluation_name": "DeepResearch Agent adversarial-round ablation experiment",
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "details": full_details,
            "config": config,
        }

    # -----------------------------------------------------------------------
    # Saving results
    # -----------------------------------------------------------------------
    @staticmethod
    def save_results(data: dict[str, Any], output_dir: str, prefix: str = "ablation") -> str:
        """Save the ablation results to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Ablation results saved: {filepath}")
        return filepath
