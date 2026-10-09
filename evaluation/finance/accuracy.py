"""Data accuracy: map numbers in the report to (company, metric, period), compare with XBRL gold, and classify errors.

The comparison is rounding-aware: a report saying 48,250.0 million only needs to fall within [48,249.95, 48,250.05] million,
so writing 48,250,000,000 as 482.5亿 is not misjudged, and a x1000 scale error is not let through.
Chinese tokens in the regexes below are intentional: they let the same checks run on Chinese-language reports.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from src.finance.xbrl import METRICS

from .gold import Gold
from .numbers import Mention

__all__ = ["AccuracyRecord", "check_accuracy", "find_metric", "find_period", "ERROR_TYPES"]

ERROR_TYPES = ["scale_error", "wrong_period", "wrong_metric", "sign_error", "imprecise", "wrong_value"]

_PERIOD_RES = [
    re.compile(r"FY\s?'?(\d{4})", re.I),
    re.compile(r"fiscal\s+(?:year\s+)?(\d{4})", re.I),
    re.compile(r"(\d{4})\s?(?:财年|财政年度|年度)"),
    re.compile(r"(\d{4})\s?年"),
    re.compile(r"\b(20\d{2})\b"),
]
_GROWTH = re.compile(r"增长|同比|增速|增幅|下降|下滑|减少|降幅|growth|grew|increase|rose|decline|decrease|fell|yoy|year-over-year|change|变化|\bup\b|\bdown\b|上升|提升|提高", re.I)
_DECLINE = re.compile(r"下降|下滑|减少|降幅|decline|decrease|fell|drop|\bdown\b", re.I)
_MARGIN = re.compile(r"margin|利润率|毛利率|净利率|占比|占营收|as a percent", re.I)
_EXTRA_ALIASES = [
    ("gross margin", "gross_profit"), ("operating margin", "operating_income"), ("net margin", "net_income"),
    ("net profit margin", "net_income"), ("营业利润率", "operating_income"), ("经营利润率", "operating_income"),
    ("净利润率", "net_income"), ("net sales", "revenue"),
]
_ALIASES = sorted(
    [(a.lower(), k) for k, spec in METRICS.items() for a in spec["en"] + spec["zh"]] + _EXTRA_ALIASES,
    key=lambda t: -len(t[0]),
)


@dataclass
class AccuracyRecord:
    raw: str
    sentence: str
    metric: str | None
    period: str | None
    kind: str
    reported: float
    expected: float | None
    status: str            # correct | incorrect | unmapped
    error_type: str | None = None
    check: str = "value"   # value | growth | margin
    period_assumed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def find_metric(text: str, anchor: int | None = None) -> str | None:
    """Find a metric alias in text (longest first, non-overlapping); with several hits, take the one nearest to anchor."""
    low = text.lower()
    taken: list[tuple[int, int]] = []
    hits: list[tuple[int, int, str]] = []
    for alias, key in _ALIASES:
        start = 0
        while True:
            i = low.find(alias, start)
            if i < 0:
                break
            j = i + len(alias)
            ascii_alias = alias.isascii()
            ok = not ascii_alias or (
                (i == 0 or not low[i - 1].isalnum()) and (j >= len(low) or not low[j].isalnum() or low[j] == "s")
            )
            if ok and not any(a < j and i < b for a, b in taken):
                taken.append((i, j))
                hits.append((i, j, key))
            start = j
    if not hits:
        return None
    if anchor is None:
        return min(hits)[2]
    return min(hits, key=lambda h: min(abs(h[0] - anchor), abs(h[1] - anchor)))[2]


def find_period(text: str, anchor: int | None = None, latest: bool = False) -> str | None:
    hits: list[tuple[int, str]] = []
    for rx in _PERIOD_RES:
        for m in rx.finditer(text):
            hits.append((m.start(), f"FY{m.group(1)}"))
        if hits and rx is not _PERIOD_RES[-1]:
            break  # once a higher-priority format matches, do not fall back to bare years
    if not hits:
        return None
    if latest:  # growth-type mentions: take the latest period in the sentence ("FY2023 grew 8.2% (vs FY2022)")
        return max(h[1] for h in hits)
    if anchor is None:
        return hits[0][1]
    return min(hits, key=lambda h: abs(h[0] - anchor))[1]


def _local_anchor(m: Mention) -> int:
    """Position of the mention in its sentence (uses find, since the sentence contains the text)."""
    i = m.sentence.find(m.raw)
    return max(i, 0)


def _tol(m: Mention, g: float) -> float:
    return max(m.half_ulp, 1e-9 * abs(g)) + 1e-9


def _classify(m: Mention, v: float, metric: str, period: str, gold: Gold) -> str:
    g = gold.value(period, metric)
    if g is None:
        return "wrong_value"
    if g != 0 and v != 0 and math.isclose(abs(v), abs(g), rel_tol=1e-6) and (v < 0) != (g < 0):
        return "sign_error"
    if g != 0 and v != 0:
        ratio = abs(v / g)
        for k in (3, 6, 9, 12, 4, 8):
            for r in (10 ** k, 10 ** -k):
                if math.isclose(ratio, r, rel_tol=2e-3):
                    return "scale_error"
    for p, vals in gold.periods.items():
        if p != period and metric in vals and abs(v - vals[metric]) <= _tol(m, vals[metric]):
            return "wrong_period"
    for k2, gv in gold.periods.get(period, {}).items():
        if k2 != metric and abs(v - gv) <= _tol(m, gv) and m.kind != "per_share":
            return "wrong_metric"
    if g != 0 and abs(v - g) / abs(g) <= 0.01:
        return "imprecise"
    return "wrong_value"


def _expected_derived(metric: str, period: str, gold: Gold, margin: bool) -> float | None:
    cur = gold.value(period, metric)
    if cur is None:
        return None
    if margin:
        rev = gold.value(period, "revenue")
        return None if not rev or metric == "revenue" else cur / rev * 100
    prior_p = gold.prior_period(period)
    prior = gold.value(prior_p, metric) if prior_p else None
    return None if not prior else (cur / prior - 1) * 100


def check_accuracy(mentions: list[Mention], gold: Gold) -> list[AccuracyRecord]:
    recs: list[AccuracyRecord] = []
    default_period = gold.latest_period()
    for m in mentions:
        if m.kind == "plain":
            continue
        sent = m.sentence
        anchor = _local_anchor(m)
        # table: first column = metric, column header = period or "change"
        metric = find_metric(m.row) if m.row else find_metric(sent, anchor)
        col_period = find_period(m.col) if m.col else None
        is_pct = m.kind == "percent"
        period = col_period or find_period(sent, anchor, latest=is_pct and not _MARGIN.search(sent))
        assumed = period is None
        period = period or default_period
        if metric is None or period is None or period not in gold.periods:
            if metric is not None and period is not None and period not in gold.periods:
                recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, None, "unmapped"))
            else:
                recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, None, "unmapped"))
            continue

        if m.kind == "percent":
            ctx = f"{m.row} {m.col} {sent}"
            margin = bool(_MARGIN.search(ctx)) and not (m.col and _GROWTH.search(m.col))
            growth = bool(_GROWTH.search(ctx))
            if not (margin or growth):
                recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, None, "unmapped"))
                continue
            kind = "margin" if margin else "growth"
            exp = _expected_derived(metric, period, gold, margin)
            if exp is None:
                recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, None, "unmapped", kind))
                continue
            v = -abs(m.value) if (kind == "growth" and _DECLINE.search(ctx) and m.value > 0) else m.value
            tol = m.half_ulp + 0.02
            ok = abs(v - exp) <= tol
            err = None
            if not ok:
                err = "sign_error" if math.isclose(abs(v), abs(exp), abs_tol=tol) else (
                    "imprecise" if abs(v - exp) <= 0.5 else "wrong_value")
            recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, v, exp, "correct" if ok else "incorrect",
                                       err, kind, assumed))
            continue

        spec = METRICS[metric]
        if (m.kind == "per_share") != bool(spec.get("per_share")):
            recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, None, "unmapped"))
            continue
        g = gold.value(period, metric)
        if g is None:
            recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, None, "unmapped"))
            continue
        if not m.unit_given and m.kind == "money":
            recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, g, "unmapped"))
            continue
        ok = abs(m.value - g) <= _tol(m, g)
        err = None if ok else _classify(m, m.value, metric, period, gold)
        recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, g, "correct" if ok else "incorrect",
                                   err, "value", assumed))
    return recs
