#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_eval.py
================================================================================
Standard evaluation set entry script (merges the former run_evaluation.py).

Supports:
  --benchmark research_bench : self-built deep research evaluation set (rule-based metrics)
  --benchmark hotpotqa      : public multi-hop QA evaluation set (EM/F1)

Usage:
    python scripts/run_eval.py --benchmark research_bench --num_questions 20
    python scripts/run_eval.py --benchmark hotpotqa --num_questions 100
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config, run_research, setup_logging
from evaluation.benchmarks.research_bench import ResearchBench
from evaluation.benchmarks.hotpotqa import HotpotQABenchmark
from evaluation.report import EvaluationReport


def evaluate_research_bench(
    num_questions: int,
    domain: str | None,
    config: dict,
) -> EvaluationReport:
    """Run the evaluation on ResearchBench."""
    logger = logging.getLogger("run_eval")
    bench = ResearchBench()
    questions = bench.get_questions(domain=domain, n=num_questions)
    logger.info(f"ResearchBench loaded {len(questions)} questions")

    modules = initialize_modules(config)
    report = EvaluationReport(name="ResearchBench_Evaluation", num_questions=len(questions))

    for idx, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["query"]
        logger.info(f"[{idx}/{len(questions)}] Evaluating question: {qid}")

        start = time.time()
        try:
            report_text = asyncio.run(run_research(query, config, modules))
            elapsed = time.time() - start

            eval_result = bench.evaluate_report(report_text, qid)
            eval_result["elapsed_seconds"] = elapsed
            report.add_detail(eval_result)
            logger.info(f"  → composite={eval_result['composite_score']:.3f}, time={elapsed:.1f}s")
        except Exception as e:
            logger.warning(f"  → FAILED: {e}")
            report.add_detail({
                "question_id": qid,
                "error": str(e),
                "composite_score": 0.0,
            })

    # Summary
    valid_scores = [d["composite_score"] for d in report.details if "composite_score" in d]
    report.set_summary({
        "average_composite": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "num_success": len([d for d in report.details if "error" not in d]),
        "num_failed": len([d for d in report.details if "error" in d]),
    })

    return report


def evaluate_hotpotqa(
    num_questions: int,
    config: dict,
    use_mock: bool = False,
) -> EvaluationReport:
    """Run the evaluation on HotpotQA (deep research variant: assesses full report quality)."""
    logger = logging.getLogger("run_eval")
    bench = HotpotQABenchmark(use_mock=use_mock)
    questions = bench.get_samples(n=num_questions, shuffle=True)
    logger.info(f"HotpotQA loaded {len(questions)} questions")

    modules = initialize_modules(config)
    report = EvaluationReport(name="HotpotQA_DeepResearch_Evaluation", num_questions=len(questions))

    predictions = []
    for idx, q in enumerate(questions, 1):
        query = q["query"]
        gold = q["expected_answer"]
        logger.info(f"[{idx}/{len(questions)}] Evaluating: {query[:60]}...")

        try:
            report_text = asyncio.run(run_research(query, config, modules))
            pred_answer = report_text.strip().split("\n")[0] if report_text.strip() else ""
        except Exception as e:
            logger.warning(f"  → FAILED: {e}")
            pred_answer = ""
            report_text = ""

        predictions.append({
            "query_id": idx,
            "prediction": pred_answer,
            "gold": gold,
            "report": report_text,
        })

        depth = bench.evaluate_report(report_text, gold) if report_text else {}
        report.add_detail({
            "query_id": idx,
            "query": query,
            "prediction": pred_answer,
            "gold": gold,
            "depth_metrics": depth,
        })

    metrics = bench.evaluate(predictions, metrics=["em", "f1", "pass@1"])
    report.set_summary(metrics)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch Agent standard evaluation script")
    parser.add_argument("--benchmark", type=str, choices=["research_bench", "hotpotqa"],
                        required=True, help="Evaluation benchmark")
    parser.add_argument("--num_questions", type=int, default=20, help="Number of evaluation questions")
    parser.add_argument("--domain", type=str, default=None, help="Domain filter (ResearchBench only)")
    parser.add_argument("--use_mock", action="store_true", help="Use built-in mock data (HotpotQA only, for pipeline verification)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Output directory")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("main")

    config = load_config(args.config)
    logger.info(f"Config loaded: {args.config or 'configs/default.yaml'}")

    if args.benchmark == "research_bench":
        report = evaluate_research_bench(args.num_questions, args.domain, config)
    elif args.benchmark == "hotpotqa":
        report = evaluate_hotpotqa(args.num_questions, config, use_mock=args.use_mock)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    filepath = report.save(args.output_dir)
    logger.info(f"Evaluation report saved: {filepath}")

    print("\n" + "=" * 60)
    print("Evaluation summary")
    print("=" * 60)
    print(json.dumps(report.summary, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
