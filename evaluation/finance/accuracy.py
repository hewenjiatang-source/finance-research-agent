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


# ---------------------------------------------------------------------------------------------
# Mapping hygiene. Principle: when unsure, leave a number *unmapped*; never guess and call it wrong.
# Tightened after a hand audit of real 10-K reports (Apple/Microsoft/JPMorgan), where 66 of 66 "errors" were mapping
# bugs, not report errors: segment/non-GAAP/adjustment numbers mapped to consolidated metrics, and the prior-year
# number in "A in FY2024, against B in FY2023" given A's period.
# ---------------------------------------------------------------------------------------------
_COMPARATOR = re.compile(r"\b(?:against|versus|vs\.?|compared\s+(?:with|to)|from|than|and|prior|previous|last\s+year)\b", re.I)
_NON_CONSOLIDATED = re.compile(
    r"non-gaap|adjusted|managed\s+basis|\bfte\b|taxable-equivalent|pro\s?forma|segment|excluding|\bex-|"
    r"of\s+total\s+(?:liabilities|assets|deposits|revenue)|loans-to-deposits|per\s+employee", re.I)
_DELTA_BEFORE = re.compile(
    r"\b(?:charge|adjustments?|gain|impact|benefit|rose|fell|grew|increased?|decreased?|declined?|reduced?|added|by|up|down|plus)"
    r"\s+(?:of\s+|by\s+)?(?:approximately\s+|about\s+|roughly\s+)?(?:usd\s*|us\$|\$)?\s*$", re.I)
_QUALIFIER_OK = {
    "total", "net", "reported", "gaap", "consolidated", "annual", "full-year", "fiscal", "year", "the", "our", "company",
    "firm", "group", "diluted", "basic", "of", "in", "for", "and", "its", "their", "with", "a", "an", "was", "were", "is",
    "are", "to", "that", "as", "by", "at", "on", "from", "versus", "against", "vs", "compared", "than", "fy", "quarterly",
    "operating", "gross", "usd", "per",
}
_GROWTH_CUE = re.compile(r"(?:up|down|increased?|decreased?|rose|fell|grew|grow(?:th)?|declined?|decline|change|"
                         r"higher|lower|上升|下降|增长|减少|同比)\b|[+\-]\s*$|year-over-year|yoy|同比|增长|下降", re.I)
_DECLINE_CUE = re.compile(r"(?:down|decreased?|fell|declined?|decline|lower|drop(?:ped)?|下降|下滑|减少)\b|(?:下降|下滑|减少)", re.I)
_MARGIN_CUE = re.compile(r"margin|of\s+(?:total\s+)?(?:net\s+)?(?:sales|revenues?)|as\s+a\s+percent|利润率|毛利率|净利率|占比|占营收", re.I)


_ATTACHED = re.compile(
    r"[\s\)\],;:]*(?:(?:in|at|for|during|of|on|as\s+of|ended|ending)\s+)*"
    r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s*)?(?:fiscal\s+(?:year\s+)?)?\(?", re.I)
_PCT_OF_SALES = re.compile(r"\(?-?[\d.,]+%\s+of\s+(?:total\s+)?(?:net\s+)?(?:sales|revenues?)\)?", re.I)
_LABEL = re.compile(r"^\s*([A-Z][^:|\[\]$%]{0,60}?):\s")
_GAP_OK = {
    "was", "were", "is", "are", "rose", "fell", "grew", "up", "down", "increase", "increased", "decrease", "decreased",
    "decline", "declined", "growth", "grow", "change", "changed", "against", "versus", "vs", "in", "to", "from", "of", "by",
    "the", "a", "an", "and", "at", "per", "cent", "percent", "or", "about", "approximately", "compared", "with", "than",
    "higher", "lower", "yoy", "year-over-year", "year", "over", "million", "billion", "trillion", "usd", "fiscal", "ended",
    "for", "as", "reported", "total", "net", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
    "dec", "june", "july", "august", "september", "october", "november", "december", "january", "february", "march",
    "april", "stated", "states", "mdna", "md", "a", "its", "this", "that", "mn", "bn", "m", "b", "yearend", "year-end",
}


def _scoped_label(clause: str, aliases: list[str]) -> bool:
    """"Greater China: Net sales declined 8%": a leading label naming no metric scopes the clause to a sub-entity."""
    m = _LABEL.match(clause)
    if not m:
        return False
    label = m.group(1)
    return not _metric_hits(label) and not any(a.lower() in label.lower() for a in aliases)


def _gap_clean(gap: str) -> bool:
    for w in re.findall(r"[A-Za-z][A-Za-z&'’\-]*", gap):
        lw = w.lower()
        if lw in _GAP_OK or re.fullmatch(r"fy\d*", lw) or lw.endswith("'s"):
            continue
        return False
    return True


def _period_hits(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for rx in _PERIOD_RES:
        for m in rx.finditer(text):
            hits.append((m.start(), f"FY{m.group(1)}"))
        if hits and rx is not _PERIOD_RES[-1]:
            break
    return sorted(hits)


def _shift(period: str, delta: int) -> str | None:
    if period.startswith("FY") and period[2:6].isdigit():
        return f"FY{int(period[2:6]) + delta}"
    return None


def _metric_hits(text: str) -> list[tuple[int, int, str]]:
    """All non-overlapping metric alias hits (start, end, key), longest alias first; sorted by position."""
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
            ok = not alias.isascii() or (
                (i == 0 or not low[i - 1].isalnum()) and (j >= len(low) or not low[j].isalnum() or low[j] == "s"))
            if ok and not any(a < j and i < b for a, b in taken):
                taken.append((i, j))
                hits.append((i, j, key))
            start = j
    return sorted(hits)


def _qualified(text: str, i: int, aliases: list[str]) -> bool:
    """True if the metric alias at text[i:] is a plain (consolidated) mention, not "Gaming revenue" / "Americas net sales"."""
    before = text[:i].rstrip()
    if not before or before[-1] in ":|(,;—–-[":
        return True
    m = re.search(r"([A-Za-z][A-Za-z&'’\-]*)$", before)
    if not m:
        return True  # preceded by a digit / symbol / CJK character
    w = m.group(1).lower().rstrip("'’s") if m.group(1).lower().endswith(("'s", "’s")) else m.group(1).lower()
    if w in _QUALIFIER_OK or w.startswith("fy") or w in {a.lower() for a in aliases}:
        return True
    return any(w == x.lower() or w in x.lower().split() for x in aliases)


def _pick_metric(text: str, m: Mention, gold: Gold, allow_after: bool) -> tuple[str | None, int | None, int | None]:
    """Metric that "owns" mention m: the last alias before it (nearest-by-distance misfires in "A vs B, and Y was C")."""
    hits = _metric_hits(text)
    before = [h for h in hits if h[1] <= m.start]
    if before:
        h = before[-1]
        return (h[2], h[0], h[1]) if _qualified(text, h[0], gold.aliases) else (None, None, None)
    if allow_after:
        after = [h for h in hits if h[0] >= m.end and h[0] - m.end <= 25]
        if after and _qualified(text, after[0][0], gold.aliases):
            return after[0][2], after[0][0], after[0][1]
    return None, None, None


def _bind_periods(group: list[Mention], gold: Gold, default_period: str | None) -> dict[int, tuple[str | None, bool]]:
    """Period of each mention in one clause -> {id(m): (period, assumed)}.

    1) a period token directly after the number (no other number in between), else directly before it;
    2) "A, against B in FY2023": A (no token) is the year after B; "A in FY2024 ... against B": B is the year before A;
    3) otherwise fall back to the latest period in the sentence, else the default (latest gold period, flagged as assumed).
    """
    ms = sorted(group, key=lambda x: x.start)
    text = ms[0].clause
    hits = _period_hits(text)
    out: dict[int, tuple[str | None, bool]] = {}
    explicit: dict[int, str] = {}
    for i, m in enumerate(ms):
        nxt = ms[i + 1].start if i + 1 < len(ms) else len(text) + 1
        prv = ms[i - 1].end if i else -1
        after = [h for h in hits if m.end <= h[0] < nxt and _ATTACHED.fullmatch(text[m.end:h[0]])]
        before = [h for h in hits if max(prv, 0) <= h[0] < m.start]
        if after:
            explicit[i] = after[0][1]
        elif before:
            explicit[i] = before[-1][1]

    def joined(a: Mention, b: Mention) -> bool:
        gap = re.sub(r"\([^()]*\)?", " ", text[a.end:b.start])  # drop parentheticals such as "(46.2% of net sales)"
        return bool(_COMPARATOR.search(gap)) and not _metric_hits(gap)

    for i, m in enumerate(ms):
        if i in explicit:
            out[id(m)] = (explicit[i], False)
            continue
        if i + 1 < len(ms) and (i + 1) in explicit and joined(m, ms[i + 1]):
            nxt_p = _shift(explicit[i + 1], +1)
            if nxt_p in gold.periods:
                out[id(m)] = (nxt_p, True)
                continue
        j = next((k for k in range(i - 1, -1, -1) if ms[k].kind == m.kind), None)  # nearest earlier mention of the same kind
        if j is not None and joined(ms[j], m) and out.get(id(ms[j]), (None, 0))[0]:
            prv_p = _shift(out[id(ms[j])][0], -1)
            if prv_p in gold.periods:
                out[id(m)] = (prv_p, True)
                continue
        if i > 0 and (i - 1) in explicit and re.match(r"^\W*\band\b", text[ms[i - 1].end:m.start], re.I) \
                and _metric_hits(text[ms[i - 1].end:m.start]):
            out[id(m)] = (explicit[i - 1], True)  # "FY2023 net sales (A) and operating income (B)": same period
            continue
        p = find_period(m.sentence, None, latest=True)
        out[id(m)] = (p, False) if p else (default_period, True)
    return out


def _local_window(text: str, m: Mention, before: int = 28, after: int = 28) -> tuple[str, str]:
    return text[max(0, m.start - before): m.start], text[m.end: m.end + after]


def check_accuracy(mentions: list[Mention], gold: Gold) -> list[AccuracyRecord]:
    recs: list[AccuracyRecord] = []
    default_period = gold.latest_period()
    # periods are bound per clause (the unit that carries one citation group), not per sentence
    groups: dict[tuple[str, str], list[Mention]] = {}
    for m in mentions:
        if m.kind != "plain":
            groups.setdefault((m.sentence, m.clause), []).append(m)
    bound: dict[int, tuple[str | None, bool]] = {}
    for g in groups.values():
        if g[0].row or g[0].col:
            continue
        bound.update(_bind_periods(g, gold, default_period))

    metric_of: dict[int, str | None] = {}

    def unm(m, metric=None, period=None, exp=None, kind="value"):
        recs.append(AccuracyRecord(m.raw, m.sentence, metric, period, m.kind, m.value, exp, "unmapped", None, kind))

    for m in mentions:
        if m.kind == "plain":
            continue
        sent = m.sentence
        table = bool(m.row or m.col)
        ctx_text = m.clause or sent
        # ---- metric ----
        if table:
            hits = _metric_hits(m.row)
            metric = hits[0][2] if hits and _qualified(m.row, hits[0][0], gold.aliases) else None
            m_beg = m_end = None
        else:
            metric, m_beg, m_end = _pick_metric(ctx_text, m, gold, allow_after=m.kind != "percent")
            if metric is None and ctx_text != sent:
                # the clause carries no metric of its own: use the sentence (split clauses share one subject)
                i0 = sent.find(ctx_text)
                if i0 >= 0:
                    shifted = Mention(m.raw, m.value, m.kind, m.decimals, m.scale, m.start + i0, m.end + i0, sent)
                    metric, m_beg, m_end = _pick_metric(sent, shifted, gold, allow_after=False)
        if not table and m.kind != "percent":
            prev_money = [x for x in mentions if x.clause == m.clause and x.kind == m.kind and x.end <= m.start]
            if prev_money:
                pm = max(prev_money, key=lambda x: x.end)
                gap = _PCT_OF_SALES.sub("", ctx_text[pm.end:m.start])
                if _COMPARATOR.search(gap) and not _metric_hits(gap) and id(pm) in metric_of:
                    metric = metric_of[id(pm)] or metric  # "A (46.2% of net sales), against B": B is the same metric as A
        if metric is None:
            unm(m)
            continue
        metric_of[id(m)] = metric
        if not table and _scoped_label(ctx_text, gold.aliases):
            unm(m, metric)
            continue
        if not table and _NON_CONSOLIDATED.search(ctx_text):
            unm(m, metric)
            continue
        # ---- period ----
        if table:
            period, assumed = find_period(m.col) if m.col else None, False
            period = period or find_period(sent, None, latest=m.kind == "percent")
            if period is None:
                period, assumed = default_period, True
        else:
            period, assumed = bound.get(id(m), (default_period, True))
        if period is None or period not in gold.periods:
            unm(m, metric, period)
            continue

        if m.kind == "percent":
            bef, aft = _local_window(ctx_text, m, 28, 32) if not table else (f"{m.row} {m.col}", f"{m.col}")
            growth = bool(_GROWTH_CUE.search(bef) or _GROWTH_CUE.search(aft[:24]))
            margin = bool(_MARGIN_CUE.search(bef[-18:]) or _MARGIN_CUE.search(aft[:24])) and not (m.col and _GROWTH.search(m.col))
            owner = ctx_text[m_beg:m_end].lower() if (m_beg is not None and m_end is not None) else ""
            if owner and re.search(r"margin|利润率|毛利率|净利率", owner) and not growth:
                margin = True
            if table:
                margin = bool(_MARGIN_CUE.search(f"{m.row} {m.col}")) and not (m.col and _GROWTH.search(m.col))
                growth = bool(_GROWTH.search(f"{m.row} {m.col}"))
            if growth and margin and not table:
                margin = False  # "gross margin ... up 17%": a growth rate of the metric, not its margin
            if not (margin or growth):
                unm(m, metric, period)
                continue
            if not table and m_end is not None and m.start - m_end > 120:
                unm(m, metric, period)
                continue
            kind = "margin" if margin else "growth"
            if not table and m_end is not None and not _gap_clean(ctx_text[m_end:m.start]):
                unm(m, metric, period)  # another subject sits between the metric name and the percent ("..., NII +4%")
                continue
            if kind == "growth" and not table:  # growth is stated for the latest period ("up 8.2% from FY2022")
                lp = find_period(sent, None, latest=True)
                if lp:
                    period, assumed = lp, False
                    if period not in gold.periods:
                        unm(m, metric, period)
                        continue
            exp = _expected_derived(metric, period, gold, margin)
            if exp is None:
                unm(m, metric, period, None, kind)
                continue
            neg = bool(_DECLINE_CUE.search(bef[-20:])) if not table else bool(_DECLINE.search(f"{m.row} {m.col} {sent}"))
            v = -abs(m.value) if (kind == "growth" and neg and m.value > 0) else m.value
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
        kind_eff = m.kind
        if kind_eff == "money" and spec.get("per_share") and m.scale == 1.0 and m.decimals >= 1 and abs(m.value) < 1000:
            kind_eff = "per_share"  # "3.71美元" next to an EPS alias is a per-share amount
        if (kind_eff == "per_share") != bool(spec.get("per_share")):
            unm(m, metric, period)
            continue
        if not table:
            pre_txt = ctx_text[:m.start]
            if _DELTA_BEFORE.search(pre_txt.lower()):
                unm(m, metric, period)  # "rose $33.2 billion", "charge of $10.2 billion": an amount of change, not the metric
                continue
            if m_end is not None and m.start - m_end > 70 and not (
                    _COMPARATOR.search(ctx_text[m_end:m.start]) and len([x for x in mentions if x.clause == m.clause]) > 1):
                unm(m, metric, period)  # too far from the metric name: ownership unclear
                continue
        gv = gold.value(period, metric)
        if gv is None:
            unm(m, metric, period)
            continue
        if not m.unit_given and m.kind == "money":
            recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, gv, "unmapped"))
            continue
        ok = abs(m.value - gv) <= _tol(m, gv)
        err = None if ok else _classify(m, m.value, metric, period, gold)
        recs.append(AccuracyRecord(m.raw, sent, metric, period, m.kind, m.value, gv, "correct" if ok else "incorrect",
                                   err, "value", assumed))
    return recs
