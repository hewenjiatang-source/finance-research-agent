#!/usr/bin/env python3
"""Build evaluation cases with XBRL gold from watchlist.yaml + SEC companyfacts. Needs network access to data.sec.gov and SEC_USER_AGENT.

    python scripts/build_finance_cases.py --watchlist evaluation/finance/cases/watchlist.yaml \
        --out evaluation/finance/cases/real_cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.finance.suite import build_case  # noqa: E402
from src.tools.sec_edgar import SecClient  # noqa: E402


def build(watch: list[dict], client: SecClient) -> list[dict]:
    cases = []
    for w in watch:
        info = client.resolve_company(w["ticker"])
        if not info:
            print(f"[skip] {w['id']}: ticker {w['ticker']} not found", file=sys.stderr)
            continue
        cf = client.get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{info['cik']:010d}.json")
        case = build_case(w["id"], w["query"], cf, w["fiscal_year"], w["ticker"], w.get("aliases"))
        if w["fiscal_year"] and f"FY{w['fiscal_year']}" not in case["gold"]:
            print(f"[skip] {w['id']}: no FY{w['fiscal_year']} 10-K data in companyfacts", file=sys.stderr)
            continue
        cases.append(case)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default="evaluation/finance/cases/watchlist.yaml")
    ap.add_argument("--out", default="evaluation/finance/cases/real_cases.json")
    args = ap.parse_args()
    watch = yaml.safe_load(Path(args.watchlist).read_text(encoding="utf-8"))
    cases = build(watch, SecClient())
    Path(args.out).write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
