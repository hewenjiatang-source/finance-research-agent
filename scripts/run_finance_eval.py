#!/usr/bin/env python3
"""Finance research evaluation entry point.

  # 1) Offline evaluation of existing reports (no API / network): directory holds <case_id>.md + <case_id>.evidence.json
  python scripts/run_finance_eval.py --cases evaluation/finance/cases/real_cases.json --reports outputs/finance_eval

  # 2) Run the agent on every case first, then evaluate (needs ANTHROPIC_API_KEY, SEC_USER_AGENT)
  python scripts/run_finance_eval.py --cases ... --run --reports outputs/finance_eval

  # 3) Evaluator self-check only (meta-evaluation, fully offline): inject known errors into clean synthetic reports
  python scripts/run_finance_eval.py --meta-only

  # Optional: --judge enables the LLM judge (non-numeric claims only; default claude-opus-5-5, ideally different from the model under test)
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.finance.perturb import meta_eval, synth_case  # noqa: E402
from evaluation.finance.suite import aggregate, evaluate_dir, load_cases, render_markdown  # noqa: E402

FIXTURE_CF = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "finance" / "acme_companyfacts.json"


def run_meta() -> dict:
    cf = json.loads(FIXTURE_CF.read_text(encoding="utf-8"))
    cases = []
    for k, name in [(1, "Acme Corp"), (0.37, "Beta Inc"), (2.9, "Gamma Ltd"), (11.3, "Delta Co")]:
        c = copy.deepcopy(cf)
        c["entityName"] = name
        for con in c["facts"]["us-gaap"].values():
            for unit, rows in con["units"].items():
                for r in rows:
                    if unit == "USD":
                        r["val"] = round(r["val"] * k / 1e6) * 1e6
                    elif unit == "USD/shares":
                        r["val"] = round(r["val"] * k, 2)
        for lang in ("en", "zh"):
            cases.append(synth_case(c, 2023, name[:4].upper(), lang))
    return meta_eval(cases)


def run_agent(cases: list[dict], out_dir: Path, config_path: str, adversarial: bool) -> None:
    from src.core.runner import initialize_modules, load_config, run_research_full, save_report

    config = load_config(config_path)
    config.setdefault("adversarial", {})["enabled"] = adversarial
    modules = initialize_modules(config, session_id=uuid.uuid4().hex[:8])
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in cases:
        if (out_dir / f"{c['id']}.md").exists():
            print(f"[skip] {c['id']} report already exists")
            continue
        print(f"[run] {c['id']}: {c['query'][:60]}")
        text, report = asyncio.run(run_research_full(c["query"], config, modules))
        p = save_report(text, c["query"], str(out_dir), evidence=report.evidence)
        # normalise to <case_id>.md / .evidence.json
        Path(p).rename(out_dir / f"{c['id']}.md")
        Path(p[:-3] + ".evidence.json").rename(out_dir / f"{c['id']}.evidence.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="evaluation/finance/cases/real_cases.json")
    ap.add_argument("--reports", default="outputs/finance_eval")
    ap.add_argument("--config", default="configs/finance.yaml")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--no-adversarial", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default="claude-opus-5-5")
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--out", default="outputs/finance_eval/eval_report")
    args = ap.parse_args()

    meta = None if args.no_meta else run_meta()
    if args.meta_only:
        print(json.dumps(meta, ensure_ascii=False, indent=1))
        return 0

    cases = load_cases(args.cases)
    if args.run:
        run_agent(cases, Path(args.reports), args.config, not args.no_adversarial)
    judge = None
    if args.judge:
        from evaluation.finance.judge import ClaimJudge
        from src.models.model_router import ModelRouter

        judge = ClaimJudge(ModelRouter.create_backend("claude", model_name=args.judge_model, temperature=0.0, max_tokens=512))
    results = evaluate_dir(args.reports, cases, judge)
    agg = aggregate(results)
    md = render_markdown(agg, results, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"aggregate": agg, "results": results, "meta": meta},
                                                    ensure_ascii=False, indent=1), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
