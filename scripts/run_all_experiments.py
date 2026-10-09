#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_all_experiments.py
================================================================================
DeepResearch Agent batch experiment script

Run all core experiments in one command and generate a structured summary report:
  1. Module ablation experiment (full / no_adversarial / no_compressor / no_memory / no_evolution）
  2. Adversarial-round ablation (0/1/2/3 rounds)
  3. Standard evaluation set (ResearchBench rule-based metrics)
  4. Multi-domain comparison (per-domain evaluation for tech / med / fin)
  5. Agent vs single-pass LLM (head-to-head benchmark)
  6. MiMo Judge in-depth scoring (expert review of a single report)
  7. Summary report generation (Markdown format)

Usage:
    python scripts/run_all_experiments.py \
        --report_file outputs/reports1/report_xxx.md \
        --report_query "your research question"

Each experiment runs in its own subprocess, independent of the others. Failed experiments are recorded but do not interrupt the overall flow.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Experiment configuration and runner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Batch experiment runner."""

    def __init__(
        self,
        config_path: str | None,
        output_dir: str,
        ablation_q: int,
        eval_q: int,
        domain_q: int,
        benchmark_queries: list[str],
        report_file: str | None,
        report_query: str | None,
    ) -> None:
        self.config_path = config_path
        self.output_dir = output_dir
        # 0 means use all available questions
        self.ablation_q = ablation_q if ablation_q > 0 else None
        self.eval_q = eval_q if eval_q > 0 else None
        self.domain_q = domain_q if domain_q > 0 else None
        self.benchmark_queries = benchmark_queries
        self.report_file = report_file
        self.report_query = report_query
        self.results: list[dict[str, Any]] = []
        self.start_time = time.time()

        os.makedirs(output_dir, exist_ok=True)

    def _run_subprocess(self, name: str, cmd: list[str], cwd: str = str(PROJECT_ROOT)) -> dict[str, Any]:
        """Run an experiment in a subprocess and return a result summary."""
        print(f"\n{'='*70}")
        print(f"[Batch] Started: {name}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=False,
                text=True,
                timeout=None,  # No per-experiment timeout; controlled by the user via Ctrl+C
            )
            elapsed = time.time() - t0
            status = "success" if proc.returncode == 0 else "failed"
            print(f"[Batch] {name} done | status={status} | elapsed={elapsed:.1f}s")
            return {
                "name": name,
                "status": status,
                "elapsed_seconds": elapsed,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"[Batch] {name} timed out (>2h)")
            return {
                "name": name,
                "status": "timeout",
                "elapsed_seconds": elapsed,
                "returncode": -1,
            }
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[Batch] {name} error: {e}")
            return {
                "name": name,
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": str(e),
            }

    def _build_config_args(self) -> list[str]:
        return ["--config", self.config_path] if self.config_path else []

    # ------------------------------------------------------------------
    # Experiment 1: Module ablation
    # ------------------------------------------------------------------
    def run_ablation_module(self) -> dict[str, Any]:
        out = os.path.join(self.output_dir, "ablation_module")
        cmd = [
            sys.executable, "scripts/run_ablation.py",
            "--mode", "module",
            "--output_dir", out,
        ] + self._build_config_args()
        if self.ablation_q is not None:
            cmd.extend(["--questions", str(self.ablation_q)])
        return self._run_subprocess("Module ablation experiment", cmd)

    # ------------------------------------------------------------------
    # Experiment 2: Adversarial-round ablation
    # ------------------------------------------------------------------
    def run_ablation_rounds(self) -> dict[str, Any]:
        out = os.path.join(self.output_dir, "ablation_rounds")
        cmd = [
            sys.executable, "scripts/run_ablation.py",
            "--mode", "rounds",
            "--max_rounds", "3",
            "--output_dir", out,
        ] + self._build_config_args()
        if self.ablation_q is not None:
            cmd.extend(["--questions", str(self.ablation_q)])
        return self._run_subprocess("Adversarial-round ablation", cmd)

    # ------------------------------------------------------------------
    # Experiment 3: Standard evaluation set
    # ------------------------------------------------------------------
    def run_eval_research_bench(self) -> dict[str, Any]:
        out = os.path.join(self.output_dir, "eval_research_bench")
        cmd = [
            sys.executable, "scripts/run_eval.py",
            "--benchmark", "research_bench",
            "--output_dir", out,
        ] + self._build_config_args()
        if self.eval_q is not None:
            cmd.extend(["--num_questions", str(self.eval_q)])
        return self._run_subprocess("Standard evaluation set (ResearchBench)", cmd)

    # ------------------------------------------------------------------
    # Experiment 4: Multi-domain comparison
    # ------------------------------------------------------------------
    def run_domain_comparison(self) -> dict[str, Any]:
        domains = ["tech", "med", "fin"]
        sub_results = []
        for domain in domains:
            out = os.path.join(self.output_dir, "domain_comparison", domain)
            cmd = [
                sys.executable, "scripts/run_eval.py",
                "--benchmark", "research_bench",
                "--domain", domain,
                "--output_dir", out,
            ] + self._build_config_args()
            if self.domain_q is not None:
                cmd.extend(["--num_questions", str(self.domain_q)])
            r = self._run_subprocess(f"Domain comparison ({domain})", cmd)
            sub_results.append(r)

        # Aggregate into a single result
        return {
            "name": "Multi-domain comparison",
            "status": "success" if all(s["status"] == "success" for s in sub_results) else "partial",
            "sub_results": sub_results,
        }

    # ------------------------------------------------------------------
    # Experiment 5: Agent vs single-pass LLM
    # ------------------------------------------------------------------
    def run_benchmark(self) -> dict[str, Any]:
        out = os.path.join(self.output_dir, "benchmark")
        os.makedirs(out, exist_ok=True)
        cmd = [
            sys.executable, "scripts/run_benchmark.py",
            "--output", os.path.join(out, "results.json"),
            "--queries",
        ] + self.benchmark_queries
        return self._run_subprocess("Agent vs single-pass LLM", cmd)

    # ------------------------------------------------------------------
    # Experiment 5b: HotpotQA deep research evaluation (optional)
    # ------------------------------------------------------------------
    def run_hotpotqa(self) -> dict[str, Any]:
        out = os.path.join(self.output_dir, "eval_hotpotqa")
        cmd = [
            sys.executable, "scripts/run_eval.py",
            "--benchmark", "hotpotqa",
            "--use_mock",
            "--output_dir", out,
        ] + self._build_config_args()
        if self.eval_q is not None:
            cmd.extend(["--num_questions", str(self.eval_q)])
        return self._run_subprocess("HotpotQA deep research evaluation (mock)", cmd)

    # ------------------------------------------------------------------
    # Experiment 6: Judge in-depth scoring
    # ------------------------------------------------------------------
    def run_judge(self) -> dict[str, Any]:
        if not self.report_file or not self.report_query:
            return {
                "name": "MiMo Judge in-depth scoring",
                "status": "skipped",
                "reason": "--report_file or --report_query not specified",
            }
        out = os.path.join(self.output_dir, "judge")
        os.makedirs(out, exist_ok=True)
        cmd = [
            sys.executable, "scripts/run_judge.py",
            "--report_file", self.report_file,
            "--query", self.report_query,
            "--output", os.path.join(out, "score.json"),
        ]
        return self._run_subprocess("MiMo Judge in-depth scoring", cmd)

    # ------------------------------------------------------------------
    # Summary report generation
    # ------------------------------------------------------------------
    def generate_summary(self) -> str:
        """Generate the Markdown summary report."""
        total_elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "# DeepResearch Agent Batch Experiment Summary Report",
            "",
            f"- **Experiment time**: {timestamp}",
            f"- **Total elapsed**: {total_elapsed/60:.1f} minutes",
            f"- **Number of experiments**: {len(self.results)}",
            f"- **Succeeded**: {sum(1 for r in self.results if r.get('status') == 'success')}",
            f"- **Failed/skipped**: {sum(1 for r in self.results if r.get('status') != 'success')}",
            "",
            "## Experiment Results Overview",
            "",
            "| Experiment | Status | Elapsed (s) | Notes |",
            "|---------|------|---------|------|",
        ]

        for r in self.results:
            name = r.get("name", "unknown")
            status = r.get("status", "unknown")
            elapsed = r.get("elapsed_seconds", 0.0)
            note = ""
            if status == "success":
                note = "✓ Completed"
            elif status == "skipped":
                note = f"Skipped: {r.get('reason', '')}"
            elif status == "partial":
                note = "Partially completed"
            else:
                note = f"✗ {r.get('error', '')[:40]}"
            lines.append(f"| {name} | {status} | {elapsed:.1f} | {note} |")

        lines.extend([
            "",
            "## Output Files",
            "",
        ])

        # List the output files of each experiment
        for subdir, desc in [
            ("ablation_module", "Module ablation results"),
            ("ablation_rounds", "Adversarial-round ablation results"),
            ("eval_research_bench", "Standard evaluation set results"),
            ("domain_comparison", "Multi-domain comparison results"),
            ("benchmark", "Agent vs LLM comparison results"),
            ("judge", "MiMo Judge in-depth scoring results"),
        ]:
            path = os.path.join(self.output_dir, subdir)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith(".json")]
                if files:
                    lines.append(f"- **{desc}**: `{path}`")
                    for f in sorted(files):
                        lines.append(f"  - `{f}`")

        lines.extend([
            "",
            "## Takeaways for Interviews",
            "",
            "### Ablation Experiments",
            "- Check the JSON under `ablation_module/` for each module's `mean_diff` and `significant`",
            "- If the CI of `no_adversarial` excludes 0 and p<0.05, the adversarial module makes an independent contribution",
            "",
            "### Standard Evaluation Set",
            "- See `average_composite` and per-domain statistics under `eval_research_bench/`",
            "- 35 questions × 5-dimension rule-based metrics = reproducible objective scores",
            "",
            "### Agent vs LLM",
            "- See `statistical_tests` in `benchmark/results.json`",
            "- If the CIs of all 4 dimensions lie to the right of 0, the Agent is significantly better than the single-pass LLM",
            "",
            "### Judge In-Depth Scoring",
            "- See the 5-dimension scores + rationale in `judge/score.json`",
            "- Useful for spot-check verification and final quality control",
            "",
        ])

        md = "\n".join(lines)
        md_path = os.path.join(self.output_dir, "SUMMARY.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

        # Also save the JSON summary
        json_path = os.path.join(self.output_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "total_elapsed_seconds": total_elapsed,
                "results": self.results,
            }, f, ensure_ascii=False, indent=2)

        return md_path

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------
    def run_all(self) -> None:
        """Run all experiments in order (skipping none)."""
        print("=" * 70)
        print("DeepResearch Agent batch experiments started -- full mode")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        abl_str = str(self.ablation_q) if self.ablation_q else "all available"
        eval_str = str(self.eval_q) if self.eval_q else "all available"
        domain_str = str(self.domain_q) if self.domain_q else "all available"
        print(f"Ablation questions: {abl_str}")
        print(f"Evaluation questions: {eval_str}")
        print(f"Domain comparison questions: {domain_str}")
        print(f"Benchmark queries: {len(self.benchmark_queries)}")
        print(f"HotpotQA: mock mode, all questions")
        print()

        self.results.append(self.run_ablation_module())
        self.results.append(self.run_ablation_rounds())
        self.results.append(self.run_eval_research_bench())
        self.results.append(self.run_domain_comparison())
        self.results.append(self.run_benchmark())
        self.results.append(self.run_hotpotqa())
        self.results.append(self.run_judge())

        # Generate summary
        md_path = self.generate_summary()
        print(f"\n{'='*70}")
        print("All batch experiments completed!")
        print(f"Summary report: {md_path}")
        print(f"Total elapsed: {(time.time() - self.start_time)/60:.1f} minutes")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Command-line entry
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent batch experiment script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended default configuration for interviews (about 12 hours, maximum sample size, no experiment skipped):
  Module ablation 5 configs × 12 questions = 60 runs
  Round ablation 4 configs × 12 questions = 48 runs
  Standard eval 35 questions (full) = 35 runs
  Domain comparison 3 domains × 5 questions = 15 runs
  Agent vs LLM 3 questions × 2 =  6 runs
  Judge in-depth scoring =  1 run
  ───────────────────────────────
  Total: about 165 research runs

Quick verification (about 2 hours):
  python scripts/run_all_experiments.py \
      --ablation_questions 3 --eval_questions 5 --domain_questions 2
        """,
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs/experiments", help="Experiment output root directory")
    parser.add_argument(
        "--ablation_questions", type=int, default=12,
        help="Number of ablation questions (default 12, 0=all available)"
    )
    parser.add_argument(
        "--eval_questions", type=int, default=35,
        help="Number of standard evaluation questions (default 35, 0=all available)"
    )
    parser.add_argument(
        "--domain_questions", type=int, default=5,
        help="Number of questions per domain for domain comparison (default 5, 0=all available)"
    )
    parser.add_argument("--report_file", type=str, default=None, help="Report file path for Judge scoring")
    parser.add_argument("--report_query", type=str, default=None, help="Original query corresponding to the Judge-scored report")
    args = parser.parse_args()

    # If no benchmark queries are specified, draw 3 in-depth questions from ResearchBench by default
    benchmark_queries = [
        "分析2026年中国互联网公司对于后训练岗位的需求性并建议我该怎么准备",
        "对比 GPT-4o、Claude 3.5 Sonnet、DeepSeek-V3 的推理能力差异",
        "2025年诺贝尔物理学奖得主的主要贡献是什么",
    ]

    # If report_file is given without report_query, try to infer it from the report filename
    report_query = args.report_query
    if args.report_file and not report_query:
        # Extract the query prefix from the filename (report_timestamp_first-20-chars-of-query.md)
        fname = Path(args.report_file).stem
        parts = fname.split("_")
        if len(parts) >= 4:
            # report_YYYYMMDD_HHMMSS_query-prefix
            report_query = "_".join(parts[3:])
        else:
            report_query = fname
        print(f"[Note] --report_query not specified; inferred from filename: {report_query}")

    runner = ExperimentRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        ablation_q=args.ablation_questions,
        eval_q=args.eval_questions,
        domain_q=args.domain_questions,
        benchmark_queries=benchmark_queries,
        report_file=args.report_file,
        report_query=report_query,
    )

    runner.run_all()


if __name__ == "__main__":
    main()
