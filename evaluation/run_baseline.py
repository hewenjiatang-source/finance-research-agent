#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/run_baseline.py
================================================================================
Baseline evaluation script — real-run version.

Runs the full system vs several ablated systems (adversarial off / evolution off / compression off),
uses MiMo 2.5 Pro as the Judge backend to score reports,
and finally outputs a structured JSON evaluation report.

Ablation configurations:
  - full:           full system (all modules on)
  - no_adversarial: M5 adversarial denoising off
  - no_evolution:   M6 evolutionary learning off
  - no_compressor:  M3 context compression off
  - no_memory:      M4 memory store off
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmarks.research_bench import ResearchBench

# Import the real research flow (migrated from scripts/ to src/core/)
from src.core.runner import initialize_modules, run_research


def load_config(config_path: str | None = None) -> dict:
    """Load a YAML config file."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config: dict, system_name: str) -> dict:
    """Override the module switches in the config according to the ablation name."""
    cfg = copy.deepcopy(config)

    if system_name == "no_adversarial":
        cfg.setdefault("adversarial", {})["enabled"] = False
    elif system_name == "no_evolution":
        cfg.setdefault("evolution", {})["enabled"] = False
    elif system_name == "no_compressor":
        cfg.setdefault("compressor", {})["enable_multilevel"] = False
    elif system_name == "no_memory":
        cfg.setdefault("memory", {})["enabled"] = False
    # "full" makes no modifications

    return cfg


class BaselineEvaluator:
    """Baseline evaluator: runs several system configurations and generates a comparison report."""

    def __init__(self, output_dir: str = "outputs/evaluation") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_system(
        self,
        system_name: str,
        config: dict,
        questions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the system with the given configuration and collect evaluation metrics (real run)."""
        print(f"[BaselineEvaluator] Running system: {system_name}")

        # Override the config according to the ablation name
        cfg = override_config(config, system_name)

        # Initialize modules
        modules = initialize_modules(cfg)

        scores = []
        details = []

        for q in questions:
            qid = q["id"]
            query = q["query"]
            print(f"  [{qid}] {query[:60]}...")

            start = time.time()
            try:
                report = asyncio.run(run_research(query, cfg, modules))
                elapsed = time.time() - start

                # Real scoring
                bench = ResearchBench()
                eval_result = bench.evaluate_report(report, qid)
                composite = eval_result.get("composite_score", 0.0)

                details.append({
                    "question_id": qid,
                    "composite_score": composite,
                    "metrics": eval_result.get("metrics", {}),
                    "elapsed_seconds": elapsed,
                    "system": system_name,
                })
                scores.append(composite)
                print(f"    → composite={composite:.3f}, time={elapsed:.1f}s")

            except Exception as e:
                print(f"    → FAILED: {e}")
                details.append({
                    "question_id": qid,
                    "composite_score": 0.0,
                    "error": str(e),
                    "system": system_name,
                })
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "system_name": system_name,
            "num_questions": len(questions),
            "average_composite_score": avg_score,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

    def run_all_baselines(
        self,
        questions: list[dict[str, Any]],
        config: dict,
        systems: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run all baseline systems and generate a comparison report."""
        if systems is None:
            systems = {
                "full": "Full system",
                "no_adversarial": "Adversarial denoising off",
                "no_evolution": "Evolutionary learning off",
                "no_compressor": "Context compression off",
                "no_memory": "Memory store off",
            }

        results = []
        for name, desc in systems.items():
            print(f"\n{'='*60}")
            print(f"[Ablation] {name}: {desc}")
            print(f"{'='*60}")
            result = self.run_system(name, config, questions)
            result["description"] = desc
            results.append(result)

        report = {
            "evaluation_name": "DeepResearch Agent ablation baseline evaluation",
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(questions),
            "systems": results,
            "summary": {
                r["system_name"]: r["average_composite_score"] for r in results
            },
        }

        return report

    def save_report(self, report: dict[str, Any], filename: str | None = None) -> str:
        """Save the evaluation report as a JSON file."""
        if filename is None:
            filename = f"baseline_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[BaselineEvaluator] Report saved: {path}")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent baseline evaluation script (real run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/run_baseline.py --questions 5 --output_dir outputs/evaluation
  python evaluation/run_baseline.py --domain tech --questions 3
        """,
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=20,
        help="Number of evaluation questions (default 20)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["tech", "med", "fin"],
        help="Filter questions by domain",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (default configs/default.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Report output directory",
    )
    args = parser.parse_args()

    # Load the config
    config = load_config(args.config)

    bench = ResearchBench()
    questions = bench.get_questions(domain=args.domain, n=args.questions)
    print(f"[main] Loaded {len(questions)} evaluation questions")

    evaluator = BaselineEvaluator(output_dir=args.output_dir)
    report = evaluator.run_all_baselines(questions, config)
    evaluator.save_report(report)

    print("\n===== Evaluation summary =====")
    for name, score in report["summary"].items():
        print(f"  {name:20s}: {score:.4f}")


if __name__ == "__main__":
    main()
