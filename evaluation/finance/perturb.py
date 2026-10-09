"""Meta-evaluation: is the evaluator itself reliable?

Method: build "clean" synthetic reports + evidence from the gold (fully independent of the LLM under test), then **inject errors of known types**
and measure the evaluator's (1) detection rate, (2) error-type classification accuracy, (3) false-positive rate on clean reports.
Without this step a number like "93% accuracy" cannot be interpreted: you cannot tell whether the model is good or the evaluator misses errors.

Limitations (see README): the injected errors are synthetic and real model errors are messier; the synthetic reports are phrased regularly,
which overstates the "mapping rate". So the detection rate here should be read as an upper bound on the evaluator, not real-world recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src.finance.evidence import _render_facts
from src.finance.xbrl import format_value, select_period

from .core import evaluate_report
from .gold import Gold, gold_from_companyfacts

__all__ = ["SynthCase", "synth_case", "PERTURBATIONS", "meta_eval"]


@dataclass
class SynthCase:
    report: str
    evidence: list[dict]
    gold: Gold
    name: str


def _facts(cf: dict, fy: int, metrics: list[str]) -> list[dict]:
    out = []
    for m in metrics:
        for fv in select_period(cf, m, fy, "FY"):
            d = fv.to_dict()
            d["display"] = format_value(fv.value, fv.unit)
            out.append(d)
    return out


def synth_case(cf: dict, fy: int, ticker: str, lang: str = "en") -> SynthCase:
    metrics = ["revenue", "gross_profit", "net_income", "eps_diluted", "operating_income"]
    facts = _facts(cf, fy, metrics)
    gold = gold_from_companyfacts(cf, fy, [ticker])
    name = cf.get("entityName", ticker)
    cur, pri = f"FY{fy}", f"FY{fy - 1}"
    g = lambda p, m: gold.value(p, m)  # noqa: E731
    growth = (g(cur, "revenue") / g(pri, "revenue") - 1) * 100
    gm = g(cur, "gross_profit") / g(cur, "revenue") * 100
    ev = [
        dict(id=1, kind="xbrl_facts", url="https://data.sec.gov/api/xbrl/companyfacts/", title=f"XBRL company facts — {name}",
             text=_render_facts({"facts": facts}), meta={}),
        dict(id=2, kind="computation", url="", title="calc growth",
             text=f"({g(cur, 'revenue'):.0f} / {g(pri, 'revenue'):.0f} - 1) * 100 = {growth}", meta={}),
        dict(id=3, kind="computation", url="", title="calc gross margin",
             text=f"{g(cur, 'gross_profit'):.0f} / {g(cur, 'revenue'):.0f} * 100 = {gm}", meta={}),
    ]
    fmt = (lambda v: f"{v / 1e6:,.1f} million USD") if lang == "en" else (lambda v: f"{v / 1e8:,.1f}亿美元")
    if lang == "en":
        lines = [
            "## Summary",
            f"{name} reported {cur} revenue of {fmt(g(cur, 'revenue'))} [1], up {growth:.1f}% from {pri} ({fmt(g(pri, 'revenue'))}) [1][2].",
            f"Net income in {cur} was {fmt(g(cur, 'net_income'))} [1]. Diluted EPS for {cur} was ${g(cur, 'eps_diluted'):.2f} [1].",
            f"Gross margin in {cur} was {gm:.1f}% [1][3].",
            "## Table", f"| Metric | {cur} | {pri} |", "|---|---|---|",
            f"| Revenue | {fmt(g(cur, 'revenue'))} [1] | {fmt(g(pri, 'revenue'))} [1] |",
            f"| Net income | {fmt(g(cur, 'net_income'))} [1] | {fmt(g(pri, 'net_income'))} [1] |",
        ]
    else:
        lines = [
            "## 摘要",
            f"{name} {cur} 营收为{fmt(g(cur, 'revenue'))} [1]，同比增长{growth:.1f}% [1][2]。",
            f"{cur} 净利润为{fmt(g(cur, 'net_income'))} [1]，稀释每股收益为{g(cur, 'eps_diluted'):.2f}美元 [1]。",
            f"{cur} 毛利率为{gm:.1f}% [1][3]。",
            "## 数据表", f"| 指标 | {cur} | {pri} |", "|---|---|---|",
            f"| 营收 | {fmt(g(cur, 'revenue'))} [1] | {fmt(g(pri, 'revenue'))} [1] |",
            f"| 净利润 | {fmt(g(cur, 'net_income'))} [1] | {fmt(g(pri, 'net_income'))} [1] |",
        ]
    return SynthCase("\n".join(lines) + "\n", ev, gold, f"{ticker}-{cur}-{lang}")


# --- Injectors: return (new report, new evidence, expected signal), or None if not applicable ------------------------------------
# expected signal: (module, category); for accuracy the error_type is checked, for citation the status

def _sub(report: str, pattern: str, repl: str, count: int = 1) -> str | None:
    new, n = re.subn(pattern, repl, report, count=count)
    return new if n else None


def _scale_up(c: SynthCase):
    r = _sub(c.report, r"(revenue of |营收为)([\d,\.]+)( million USD|亿美元)",
             lambda m: m.group(1) + m.group(2) + (" billion USD" if "million" in m.group(3) else "万亿美元"))
    return (r, c.evidence, ("accuracy", "scale_error")) if r else None


def _digit_typo(c: SynthCase):
    m = re.search(r"(Net income in FY\d{4} was |净利润为)([\d,]+)\.(\d)", c.report)
    if not m:
        return None
    digits = m.group(2).replace(",", "")
    swapped = digits[:-2] + digits[-1] + digits[-2] if digits[-1] != digits[-2] else None
    if not swapped:
        return None
    new_int = f"{int(swapped):,}"
    return c.report[: m.start(2)] + new_int + c.report[m.end(2):], c.evidence, ("accuracy", "any")


def _wrong_period(c: SynthCase):
    m = re.search(r"\| (Metric|指标) \| (FY\d{4}) \| (FY\d{4}) \|", c.report)
    if not m:
        return None
    new = c.report[: m.start(2)] + m.group(3) + " | " + m.group(2) + c.report[m.end(3):]
    return new, c.evidence, ("accuracy", "wrong_period")


def _dangling(c: SynthCase):
    r = _sub(c.report, r"(Net income in[^\n]*?)\[1\]", r"\1[99]") or _sub(c.report, r"(净利润为[^\n]*?)\[1\]", r"\1[99]")
    return (r, c.evidence, ("citation", "dangling")) if r else None


def _misattribute(c: SynthCase):
    r = _sub(c.report, r"(Net income in[^\n]*?)\[1\]", r"\1[3]") or _sub(c.report, r"(净利润为[^\n]*?)\[1\]", r"\1[3]")
    return (r, c.evidence, ("citation", "misattributed")) if r else None


def _fabricate_cited(c: SynthCase):
    extra = "\nOperating income was 12,345.0 million USD [1].\n" if "Summary" in c.report else "\n营业利润为123.4亿美元 [1]。\n"
    return c.report + extra, c.evidence, ("citation", "unsupported")


def _fabricate_uncited(c: SynthCase):
    extra = "\nR&D spending reached 9,876.5 million USD.\n" if "Summary" in c.report else "\n研发费用为98.8亿美元。\n"
    return c.report + extra, c.evidence, ("citation", "uncited_ungrounded")


def _sign_flip(c: SynthCase):
    r = _sub(c.report, r"up (\d)", r"down \1") or _sub(c.report, r"同比增长", "同比下降")
    return (r, c.evidence, ("accuracy", "sign_error")) if r else None


def _uncite(c: SynthCase):
    r = _sub(c.report, r"(Net income in[^\n]*?) \[1\]", r"\1") or _sub(c.report, r"(净利润为[^\n]*?) \[1\]", r"\1")
    return (r, c.evidence, ("citation", "uncited_grounded")) if r else None


PERTURBATIONS: dict[str, Callable] = {
    "scale_x1000": _scale_up,
    "digit_transposition": _digit_typo,
    "wrong_period": _wrong_period,
    "dangling_citation": _dangling,
    "misattributed_citation": _misattribute,
    "fabricated_cited_number": _fabricate_cited,
    "fabricated_uncited_number": _fabricate_uncited,
    "sign_flip": _sign_flip,
    "dropped_citation": _uncite,
}


def _count(res: dict, module: str, cat: str) -> int:
    if module == "accuracy":
        acc = res.get("accuracy", {})
        if cat == "any":
            return sum(1 for r in acc.get("records", []) if r["status"] == "incorrect")
        return acc.get("errors", {}).get(cat, 0)
    counts = res["citation"]["counts"]
    return counts.get(cat, 0)


def _any_flag(res: dict) -> int:
    c = res["citation"]["counts"]
    acc = res.get("accuracy", {})
    return (len(res["citation"]["dangling_ids"])
            + sum(c.get(k, 0) for k in ("dangling", "misattributed", "unsupported", "uncited_ungrounded"))
            + sum(1 for r in acc.get("records", []) if r["status"] == "incorrect"))


def meta_eval(cases: list[SynthCase]) -> dict:
    base = {c.name: evaluate_report(c.report, c.evidence, c.gold) for c in cases}
    clean_flags = {n: _any_flag(r) for n, r in base.items()}
    total_records = sum(len(r["records"]) for r in base.values())
    out: dict = {
        "clean": {
            "n_reports": len(cases),
            "reports_with_false_flags": sum(1 for v in clean_flags.values() if v),
            "false_positive_rate": round(sum(1 for v in clean_flags.values() if v) / len(cases), 4) if cases else None,
            "flagged_items": sum(clean_flags.values()),
            "total_numeric_items": total_records,
            "mapping_rate": round(sum(r["accuracy"]["n_mapped"] for r in base.values())
                                  / max(sum(r["accuracy"]["n_numeric"] for r in base.values()), 1), 4),
            "flagged_by_report": {n: v for n, v in clean_flags.items() if v},
        },
        "perturbations": {},
    }
    for pname, fn in PERTURBATIONS.items():
        n = det = typ = 0
        for c in cases:
            res = fn(c)
            if res is None:
                continue
            rep, ev, (mod, cat) = res
            n += 1
            r2 = evaluate_report(rep, ev, c.gold)
            before = base[c.name]
            if _any_flag(r2) > _any_flag(before) or _count(r2, "citation", "uncited_grounded") > _count(before, "citation", "uncited_grounded"):
                det += 1
            if _count(r2, mod, cat) > _count(before, mod, cat):
                typ += 1
        out["perturbations"][pname] = {
            "n": n, "detected": det, "detection_rate": round(det / n, 4) if n else None,
            "type_correct": typ, "type_accuracy": round(typ / n, 4) if n else None,
        }
    return out
