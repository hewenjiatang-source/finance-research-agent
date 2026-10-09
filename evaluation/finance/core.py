"""Entry point for evaluating one report: evaluate_report(report_md, evidence, gold, judge) -> dict."""
from __future__ import annotations

import re
from collections import Counter

from .accuracy import ERROR_TYPES, check_accuracy
from .citations import check_citations
from .claims import parse_report, strip_boilerplate
from .gold import Gold
from .numbers import CITE_RE

__all__ = ["evaluate_report", "HEADLINE_METRICS"]

HEADLINE_METRICS = ["revenue", "net_income", "eps_diluted"]


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def evaluate_report(report: str, evidence: list[dict], gold: Gold | None = None, judge=None,
                    headline: list[str] | None = None) -> dict:
    body = strip_boilerplate(report)
    units = parse_report(report)
    ids_in_text = [int(x) for x in CITE_RE.findall(body)]
    known = {e["id"] for e in evidence}
    dangling_ids = sorted({i for i in ids_in_text if i not in known})

    mentions = [m for u in units for m in u.mentions if m.kind != "plain"]
    crecs = check_citations(units, evidence)
    cc = Counter(r.status for r in crecs)
    cited = sum(cc[s] for s in ("supported", "misattributed", "unsupported", "derived_unverified", "dangling"))
    uncited = cc["uncited_grounded"] + cc["uncited_ungrounded"]

    citation = {
        "n_numeric_claims": len(crecs),
        "n_citations_in_text": len(ids_in_text),
        "dangling_ids": dangling_ids,
        "counts": dict(cc),
        # citation precision: of the cited numbers, the share whose cited source really contains the number
        "citation_precision": _rate(cc["supported"], cited),
        # citation coverage: share of numbers that carry a citation
        "citation_coverage": _rate(cited, len(crecs)),
        # among uncited numbers, share that no evidence supports = suspected fabrication / from memory
        "uncited_ungrounded_rate": _rate(cc["uncited_ungrounded"], uncited),
        "unsupported_rate": _rate(cc["unsupported"] + cc["dangling"] + cc["misattributed"], cited),
    }

    result: dict = {"citation": citation}

    if gold is not None:
        arecs = check_accuracy(mentions, gold)
        mapped = [r for r in arecs if r.status != "unmapped"]
        correct = [r for r in mapped if r.status == "correct"]
        errs = Counter(r.error_type for r in mapped if r.status == "incorrect")
        want = headline or HEADLINE_METRICS
        latest = gold.latest_period()
        got = {r.metric for r in correct if r.period == latest and r.check == "value"}
        result["accuracy"] = {
            "n_numeric": len(arecs),
            "n_mapped": len(mapped),
            "mapping_rate": _rate(len(mapped), len(arecs)),
            "n_correct": len(correct),
            "accuracy": _rate(len(correct), len(mapped)),
            "errors": {t: errs.get(t, 0) for t in ERROR_TYPES},
            "headline_recall": _rate(sum(1 for w in want if w in got), len(want)),
            "headline_missing": [w for w in want if w not in got],
            "records": [r.to_dict() for r in arecs],
        }
        bad_scale = errs.get("scale_error", 0)
    else:
        bad_scale = 0

    flags = []
    if dangling_ids:
        flags.append("dangling_citation")
    if bad_scale:
        flags.append("scale_error")
    if cc["unsupported"] or cc["misattributed"]:
        flags.append("unsupported_number")
    result["hard_flags"] = flags
    result["records"] = [r.to_dict() for r in crecs]

    if judge is not None:
        result["judge"] = judge.judge_report(units, evidence)
    return result
