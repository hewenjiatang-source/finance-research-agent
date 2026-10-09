#!/usr/bin/env python3
"""Finance research entry point: python scripts/run_finance.py "<question>" [--config configs/finance.yaml]

Writes <report>.md and <report>.evidence.json (evidence sidecar for offline evaluation by scripts/run_finance_eval.py).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.runner import initialize_modules, load_config, run_research_full, save_report, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Claude-powered financial-filings research agent")
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/finance.yaml")
    ap.add_argument("--output-dir", default="outputs/finance")
    ap.add_argument("--no-adversarial", action="store_true", help="skip the Red/Blue adversarial loop (cheaper)")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.no_adversarial:
        config.setdefault("adversarial", {})["enabled"] = False
    setup_logging(config.get("system", {}).get("log_level", "INFO"))
    modules = initialize_modules(config, session_id=uuid.uuid4().hex[:8])
    text, report = asyncio.run(run_research_full(args.query, config, modules))
    path = save_report(text, args.query, args.output_dir, evidence=report.evidence)
    print(text)
    print(f"\n[saved] {path}  (+ .evidence.json, {len(report.evidence)} evidence items)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
