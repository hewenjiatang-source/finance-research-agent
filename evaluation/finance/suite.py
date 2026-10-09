"""Evaluation suite: load cases -> evaluate a batch of reports -> aggregate (pooled counts + Wilson CI) -> Markdown report."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .core import HEADLINE_METRICS, evaluate_report
from .gold import Gold, gold_from_companyfacts

__all__ = ["load_cases", "build_case", "evaluate_dir", "aggregate", "render_markdown", "wilson"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def build_case(case_id: str, query: str, companyfacts: dict, fiscal_year: int, ticker: str,
               aliases: list[str] | None = None, headline: list[str] | None = None) -> dict:
    """用 SEC companyfacts 生成带金标准的用例（金标准 = 申报中的 XBRL 数值，不是 LLM 输出）。"""
    gold = gold_from_companyfacts(companyfacts, fiscal_year, (aliases or []) + [ticker])
    return {
        "id": case_id, "query": query, "ticker": ticker, "company": gold.company, "fiscal_year": fiscal_year,
        "headline": headline or HEADLINE_METRICS, "aliases": gold.aliases,
        "gold": gold.periods, "period_ends": gold.period_ends,
    }


def load_cases(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"] if isinstance(data, dict) else data


def _gold_of(case: dict) -> Gold:
    return Gold(company=case.get("company", ""), aliases=case.get("aliases", []), periods=case["gold"],
                period_ends=case.get("period_ends", {}))


def evaluate_dir(reports_dir: str | Path, cases: list[dict], judge=None) -> list[dict]:
    """目录约定：<case_id>.md + <case_id>.evidence.json（run_finance_eval --run 会这样落盘）。"""
    out = []
    d = Path(reports_dir)
    for c in cases:
        md, ev = d / f"{c['id']}.md", d / f"{c['id']}.evidence.json"
        if not md.exists():
            out.append({"id": c["id"], "status": "missing_report"})
            continue
        if not ev.exists():
            out.append({"id": c["id"], "status": "missing_evidence"})
            continue
        evidence = json.loads(ev.read_text(encoding="utf-8"))["evidence"]
        res = evaluate_report(md.read_text(encoding="utf-8"), evidence, _gold_of(c), judge, c.get("headline"))
        res.update(id=c["id"], status="ok")
        out.append(res)
    return out


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    agg: dict = {"n_cases": len(results), "n_evaluated": len(ok),
                 "missing": [r["id"] for r in results if r.get("status") != "ok"]}
    if not ok:
        return agg
    cnt = {k: sum(r["citation"]["counts"].get(k, 0) for r in ok) for k in
           ("supported", "misattributed", "unsupported", "dangling", "derived_unverified",
            "uncited_grounded", "uncited_ungrounded")}
    cited = sum(cnt[k] for k in ("supported", "misattributed", "unsupported", "dangling", "derived_unverified"))
    uncited = cnt["uncited_grounded"] + cnt["uncited_ungrounded"]
    agg["citation"] = {
        "counts": cnt,
        "precision": {"value": round(cnt["supported"] / cited, 4) if cited else None, "n": cited,
                      "ci95": wilson(cnt["supported"], cited)},
        "coverage": {"value": round(cited / (cited + uncited), 4) if cited + uncited else None, "n": cited + uncited},
        "uncited_ungrounded_rate": {"value": round(cnt["uncited_ungrounded"] / uncited, 4) if uncited else None, "n": uncited},
        "reports_with_dangling_ids": sum(1 for r in ok if r["citation"]["dangling_ids"]),
    }
    acc = [r["accuracy"] for r in ok if "accuracy" in r]
    mapped = sum(a["n_mapped"] for a in acc)
    correct = sum(a["n_correct"] for a in acc)
    errs = {t: sum(a["errors"].get(t, 0) for a in acc) for t in acc[0]["errors"]} if acc else {}
    heads = [a["headline_recall"] for a in acc if a["headline_recall"] is not None]
    agg["accuracy"] = {
        "accuracy": {"value": round(correct / mapped, 4) if mapped else None, "n": mapped, "ci95": wilson(correct, mapped)},
        "mapping_rate": round(mapped / max(sum(a["n_numeric"] for a in acc), 1), 4),
        "errors": errs,
        "headline_recall_mean": round(sum(heads) / len(heads), 4) if heads else None,
    }
    agg["reports_with_hard_flags"] = sum(1 for r in ok if r["hard_flags"])
    return agg


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def render_markdown(agg: dict, results: list[dict], meta: dict | None = None) -> str:
    L = ["# Finance Research Agent — Evaluation Report", "", f"Cases evaluated: {agg['n_evaluated']}/{agg['n_cases']}"]
    if agg.get("missing"):
        L.append(f"(missing: {', '.join(agg['missing'])})")
    if "citation" in agg:
        c, a = agg["citation"], agg["accuracy"]
        ci = lambda d: f" (95% CI {_pct(d['ci95'][0])}–{_pct(d['ci95'][1])})" if d.get("ci95") else ""  # noqa: E731
        L += ["", "## Citation verification", "",
              f"- Citation precision (cited source really contains the number): **{_pct(c['precision']['value'])}**{ci(c['precision'])}, n={c['precision']['n']}",
              f"- Citation coverage (share of numbers that carry a citation): {_pct(c['coverage']['value'])}",
              f"- Uncited numbers with no supporting evidence (likely from memory): {_pct(c['uncited_ungrounded_rate']['value'])}",
              f"- Reports with dangling citation ids: {c['reports_with_dangling_ids']}",
              f"- Breakdown: {c['counts']}", "", "## Data accuracy (vs SEC XBRL gold)", "",
              f"- Accuracy: **{_pct(a['accuracy']['value'])}**{ci(a['accuracy'])}, mapped numbers n={a['accuracy']['n']}, mapping rate {_pct(a['mapping_rate'])}",
              f"- Headline recall (revenue / net income / EPS reported correctly): {_pct(a['headline_recall_mean'])}",
              f"- Error taxonomy: {a['errors']}", f"- Reports with hard flags: {agg['reports_with_hard_flags']}"]
    L += ["", "## Per case", "", "| case | citation precision | accuracy | dangling ids | hard flags |", "|---|---|---|---|---|"]
    for r in results:
        if r.get("status") != "ok":
            L.append(f"| {r['id']} | {r['status']} | | | |")
            continue
        L.append(f"| {r['id']} | {_pct(r['citation']['citation_precision'])} | {_pct(r.get('accuracy', {}).get('accuracy'))} "
                 f"| {r['citation']['dangling_ids'] or '-'} | {', '.join(r['hard_flags']) or '-'} |")
    if meta:
        cl = meta["clean"]
        L += ["", "## Evaluator self-check (meta-evaluation: known errors injected into clean synthetic reports)", "",
              f"- False-positive rate on clean reports: {_pct(cl['false_positive_rate'])} ({cl['reports_with_false_flags']}/{cl['n_reports']} reports), "
              f"number-mapping rate {_pct(cl['mapping_rate'])}", "",
              "| Injected error | n | Detection rate | Type accuracy |", "|---|---|---|---|"]
        for k, v in meta["perturbations"].items():
            L.append(f"| {k} | {v['n']} | {_pct(v['detection_rate'])} | {_pct(v['type_accuracy'])} |")
        L += ["", "> Injected errors are synthetic and the report phrasing is regular, so these detection rates are an **upper bound**; real model errors are messier."]
    L += ["", "## Limitations", "",
          "- Mapping a number to (metric, period) is heuristic (alias table + proximity). Unmappable numbers are counted as unmapped, never as correct.",
          "- Gold covers standard XBRL concepts only; MD&A segment data, non-GAAP measures, guidance and dates are not auto-checked.",
          "- Citations are judged at clause level: if a clause carries several citations, one of them containing the number is enough.",
          "- Gold extraction shares `select_period` with the agent's tool; extraction correctness itself is covered by the hand-checked truth.json regression test.",
          "- The LLM judge (if enabled) only grades non-numeric claims and has self-preference bias; it is not part of the hard metrics."]
    return "\n".join(L) + "\n"
