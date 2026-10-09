#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_judge.py
================================================================================
MiMo Judge in-depth scoring entry script.

Score a single research report on 5 dimensions as an expert reviewer and output structured JSON.

Usage:
    python scripts/run_judge.py --report_file outputs/reports/report_xxx.md --query "original query"
    python scripts/run_judge.py --report_text "report content..." --query "original query"
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.judge import LLMJudge


def main() -> None:
    parser = argparse.ArgumentParser(description="MiMo Judge in-depth scoring")
    parser.add_argument("--report_file", type=str, default=None, help="Report file path (Markdown)")
    parser.add_argument("--report_text", type=str, default=None, help="Report text content")
    parser.add_argument("--query", type=str, required=True, help="Original research query")
    parser.add_argument("--ground_truth_file", type=str, default=None, help="ground_truth JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for scoring results")
    parser.add_argument("--backend", type=str, default="claude", help="Judge backend name")
    args = parser.parse_args()

    # Read the report
    if args.report_file:
        with open(args.report_file, "r", encoding="utf-8") as f:
            report_text = f.read()
    elif args.report_text:
        report_text = args.report_text
    else:
        print("Error: must specify --report_file or --report_text")
        sys.exit(1)

    # Read ground_truth
    ground_truth = None
    if args.ground_truth_file:
        with open(args.ground_truth_file, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

    print(f"[Judge] Scoring the report in depth with {args.backend}...")
    judge = LLMJudge(backend=args.backend)
    result = judge.score_single(report_text, args.query, ground_truth)

    if "error" in result:
        print(f"[Judge] Scoring failed: {result['error']}")
        sys.exit(1)

    print("\n===== MiMo Judge Scoring Results =====")
    print(f"Overall quality: {result['overall']['score']:.1f}/10 — {result['overall']['reason']}")
    print(f"Average score: {result['average']:.2f}")
    print("\nPer-dimension:")
    for dim, data in result.get("dimensions", {}).items():
        print(f"  {dim:25s}: {data['score']:5.1f} — {data['reason']}")
    print("=" * 40)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Scoring results saved: {args.output}")


if __name__ == "__main__":
    main()
