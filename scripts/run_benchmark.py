#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_benchmark.py
================================================================================
DeepResearch Agent quantitative evaluation script

Evaluation design:
    Compare research quality of "single-pass LLM direct answer" vs "full Agent pipeline".

Metrics:
    1. comprehensiveness (1-5): how many sub-topics the report covers
    2. accuracy (1-5): whether the information is accurate and free of hallucination
    3. source_count: number of cited sources
    4. report_length: report length (characters)
    5. confidence: confidence reported by the system

Usage:
    python scripts/run_benchmark.py --queries_file data/benchmark_queries.txt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Single-pass LLM baseline (ask DeepSeek directly, bypassing the Agent)
# ---------------------------------------------------------------------------
def run_baseline(query: str, config: dict) -> dict:
    """Call the LLM directly with the default backend and return the report text."""
    from src.models.model_router import ModelRouter

    policy = ModelRouter.create_backend("claude")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Answer the user's question "
                "with a comprehensive, well-structured report in Markdown. "
                "Cite sources if possible. End with: Overall Confidence: X.XX"
            ),
        },
        {"role": "user", "content": query},
    ]
    resp = policy(messages)
    content = resp.get("content", "")
    return {
        "query": query,
        "content": content,
        "length": len(content),
        "source_count": content.count("http"),
    }


# ---------------------------------------------------------------------------
# Full Agent pipeline
# ---------------------------------------------------------------------------
async def run_agent(query: str, config: dict) -> dict:
    """Run the full Agent pipeline and return the report."""
    from src.core.runner import initialize_modules, run_research

    modules = initialize_modules(config)
    report_md = await run_research(query, config, modules)

    # Simple parsing of metadata
    confidence = 0.0
    if "**置信度**:" in report_md:
        try:
            line = [l for l in report_md.splitlines() if "**置信度**:" in l][0]
            confidence = float(line.split(":")[-1].strip())
        except (IndexError, ValueError):
            pass

    return {
        "query": query,
        "content": report_md,
        "length": len(report_md),
        "source_count": report_md.count("http"),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# MiMo 2.5 Pro automatic scoring (LLM-as-Judge)
# ---------------------------------------------------------------------------
def auto_score(report_a: str, report_b: str, query: str) -> dict:
    """
    Call MiMo 2.5 Pro to score the two reports comparatively.
    MiMo serves as the Judge backend, scoring on four dimensions: coverage, accuracy, structure, and citations.
    """
    from src.core.judge import LLMJudge
    try:
        judge = LLMJudge(backend="claude")
        return judge.compare_two(report_a, report_b, query)
    except Exception as e:
        print(f"[AutoScore] MiMo Judge scoring failed: {e}")
    return {}


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch Agent Benchmark")
    parser.add_argument("--queries_file", type=str, default=None, help="File with one query per line")
    parser.add_argument("--queries", type=str, nargs="+", default=None, help="Pass queries directly on the command line")
    parser.add_argument("--output", type=str, default="outputs/benchmark_results.json", help="Result output path")
    parser.add_argument("--skip_baseline", action="store_true", help="Skip the baseline, run only the Agent")
    parser.add_argument("--skip_agent", action="store_true", help="Skip the Agent, run only the baseline")
    args = parser.parse_args()

    # Load queries
    if args.queries:
        queries = args.queries
    elif args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        # Default evaluation set
        queries = [
            "分析2026年中国互联网公司对于后训练岗位的需求性并建议我该怎么准备",
            "对比 GPT-4o、Claude 3.5 Sonnet、DeepSeek-V3 的推理能力差异",
            "2025年诺贝尔物理学奖得主的主要贡献是什么",
        ]

    print(f"[Benchmark] Number of evaluation queries: {len(queries)}")

    # Load config
    from src.core.runner import load_config
    config = load_config()

    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"[Benchmark] Query {i}/{len(queries)}: {query[:50]}...")
        print("=" * 60)

        record = {"query": query, "baseline": None, "agent": None, "scores": None}

        # Baseline
        if not args.skip_baseline:
            print("[Benchmark] Running baseline (single-turn LLM)...")
            t0 = time.time()
            baseline = run_baseline(query, config)
            baseline["elapsed"] = round(time.time() - t0, 2)
            record["baseline"] = baseline
            print(f"[Baseline] chars={baseline['length']}, sources={baseline['source_count']}, elapsed={baseline['elapsed']}s")

        # Agent
        if not args.skip_agent:
            print("[Benchmark] Running Agent (full pipeline)...")
            t0 = time.time()
            agent_result = await run_agent(query, config)
            agent_result["elapsed"] = round(time.time() - t0, 2)
            record["agent"] = agent_result
            print(f"[Agent] chars={agent_result['length']}, sources={agent_result['source_count']}, confidence={agent_result.get('confidence', 0):.2f}, elapsed={agent_result['elapsed']}s")

        # Auto score (if both available)
        if record["baseline"] and record["agent"]:
            print("[Benchmark] Auto-scoring...")
            scores = auto_score(record["baseline"]["content"], record["agent"]["content"], query)
            record["scores"] = scores
            if scores:
                print(f"[Score] {json.dumps(scores, ensure_ascii=False, indent=2)}")
            else:
                print("[Score] Automatic scoring failed; please compare the two reports manually")

        results.append(record)

    # ------------------------------------------------------------------
    # Summary + statistical significance
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[Benchmark] Evaluation complete, summary:")
    print("=" * 60)

    for r in results:
        print(f"\nQ: {r['query'][:40]}...")
        if r["baseline"]:
            b = r["baseline"]
            print(f"  Baseline: {b['length']} chars, {b['source_count']} sources, {b['elapsed']}s")
        if r["agent"]:
            a = r["agent"]
            print(f"  Agent:    {a['length']} chars, {a['source_count']} sources, conf={a.get('confidence', 0):.2f}, {a['elapsed']}s")

    # Statistical significance: collect paired scores per dimension for each question
    if not args.skip_baseline and not args.skip_agent:
        from evaluation.metrics.stats import bootstrap_ci_paired

        dimensions = ["comprehensiveness", "accuracy", "structure", "sources"]
        dim_scores: dict[str, dict[str, list[float]]] = {d: {"agent": [], "baseline": []} for d in dimensions}

        for r in results:
            scores = r.get("scores", {})
            for dim in dimensions:
                dim_data = scores.get(dim, {})
                if isinstance(dim_data, dict) and "A" in dim_data and "B" in dim_data:
                    # A=baseline, B=agent (per the LLMJudge.compare_two convention)
                    dim_scores[dim]["baseline"].append(float(dim_data["A"]))
                    dim_scores[dim]["agent"].append(float(dim_data["B"]))

        print(f"\n{'='*60}")
        print("[Benchmark] Statistical significance (Agent vs Baseline, paired bootstrap 95% CI)")
        print("=" * 60)
        stats_summary: dict[str, Any] = {}
        for dim in dimensions:
            a_scores = dim_scores[dim]["agent"]
            b_scores = dim_scores[dim]["baseline"]
            if len(a_scores) < 2:
                continue
            diffs = [a - b for a, b in zip(a_scores, b_scores)]
            stats = bootstrap_ci_paired(diffs)
            stats_summary[dim] = stats
            sig = "✓ Significant" if stats["significant"] else "✗ Not significant"
            print(f"  {dim:20s}: Agent={sum(a_scores)/len(a_scores):.2f} Baseline={sum(b_scores)/len(b_scores):.2f} "
                  f"Δ={stats['mean_diff']:+.2f} CI=[{stats['ci_lower']:+.2f}, {stats['ci_upper']:+.2f}] "
                  f"p={stats['p_value']:.4f} {sig}")

        # Save results
        final_output = {
            "results": results,
            "statistical_tests": stats_summary,
            "num_questions": len(queries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    else:
        final_output = {
            "results": results,
            "num_questions": len(queries),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print(f"\n[Benchmark] Results saved: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
