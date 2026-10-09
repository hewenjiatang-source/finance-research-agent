"""Extract context-bearing numeric mentions (Mention) from report text.

Key point: the same number can be written 48,250.0 million / $48.25 billion / 482.5亿美元;
the evaluation must first normalize to base units, then compare with "the precision as written" (rounding-aware).
Chinese unit tokens (亿, 万, 美元 ...) are intentional: they support Chinese-language reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Mention", "extract_mentions", "split_units", "CITE_RE", "find_cites"]

CITE_RE = re.compile(r"\[(\d{1,4})\]")

_SCALES = {
    "thousand": 1e3, "k": 1e3, "千": 1e3,
    "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6, "百万": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9, "十亿": 1e9,
    "trillion": 1e12, "tn": 1e12, "万亿": 1e12,
    "万": 1e4, "亿": 1e8,
}
_UNIT_ALT = "万亿|十亿|百万|trillion|billion|million|thousand|bn|mn|mm|tn|万|亿|千|k|m|b"
_CURRENCY = r"(?:USD|US\$|美元|dollars?|RMB|CNY|人民币|HKD|港元|港币|元)"

# number + optional unit + optional currency; the $ / US$ prefix is handled separately
_NUM = r"(?P<num>-?\(?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|-?\(?\d+(?:\.\d+)?\)?)"
_MENTION_RE = re.compile(
    r"(?P<pre>US\$|\$|¥|￥)?\s?" + _NUM +
    r"(?P<pct>\s?%|\s?个百分点|\s?pp\b|\s?percentage points?)?"
    r"(?:\s?(?P<unit>" + _UNIT_ALT + r")(?![A-Za-z]))?"
    r"(?:\s?(?P<cur>" + _CURRENCY + r"))?"
    r"(?P<ps>\s?(?:per share|/share|每股))?",
    re.I,
)
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


@dataclass
class Mention:
    raw: str
    value: float            # normalized base value: amounts in the currency's base unit, percent = the percentage itself (8.2), per-share = USD
    kind: str               # money | percent | per_share | count | plain
    decimals: int           # decimal places as written -> precision
    scale: float            # unit multiplier as written (million = 1e6)
    start: int
    end: int
    sentence: str = ""
    cites: list[int] = field(default_factory=list)
    unit_given: bool = False
    row: str = ""           # first table column (metric name); empty outside tables
    col: str = ""           # table column header (period); empty outside tables
    clause: str = ""        # text the mention was extracted from; start/end are offsets into it

    @property
    def half_ulp(self) -> float:
        """Half of the written precision: 48,250.0 million -> 0.05 million = 5e4."""
        s = self.scale if self.kind in ("money", "count") else 1.0
        return 0.5 * (10 ** -self.decimals) * s


def split_units(text: str) -> str:
    return text


def find_cites(text: str) -> list[int]:
    return [int(x) for x in CITE_RE.findall(text)]


def _to_float(num: str) -> tuple[float, int]:
    neg = num.startswith("-") or (num.startswith("(") and num.endswith(")"))
    clean = num.strip("()-").replace(",", "")
    dec = len(clean.split(".")[1]) if "." in clean else 0
    v = float(clean)
    return (-v if neg else v), dec


def extract_mentions(text: str, sentence: str = "", base: int = 0) -> list[Mention]:
    """Extract numeric mentions from text. Numbers inside citation ids [n], years, ordinals and dates are skipped."""
    out: list[Mention] = []
    cite_spans = [(m.start(), m.end()) for m in CITE_RE.finditer(text)]
    for m in _MENTION_RE.finditer(text):
        if any(a <= m.start("num") < b for a, b in cite_spans):
            continue
        num = m.group("num")
        if not any(ch.isdigit() for ch in num):
            continue
        v, dec = _to_float(num)
        pre, pct, unit, cur, ps = m.group("pre"), m.group("pct"), m.group("unit"), m.group("cur"), m.group("ps")
        s = text
        # years / small unitless integers / date fragments / ids glued to letters (Q2, FY2023, 10-K) -> skip
        before = s[max(0, m.start("num") - 3): m.start("num")]
        after = s[m.end("num"): m.end("num") + 2]
        if re.search(r"[A-Za-z]$", before) or re.match(r"^[-/]\d", after):
            continue
        if re.match(r"^(年|财年|季度|月|日|号|Q\d)", s[m.end("num"): m.end("num") + 3]) and not (unit or pct or pre):
            continue
        if _YEAR_RE.match(num) and not (unit or pct or pre or cur):
            continue
        if not (unit or pct or pre or cur or ps) and "," not in num and dec == 0 and abs(v) < 100:
            continue  # an isolated small integer ("3 drivers") is not financial data

        scale = 1.0
        if unit:
            scale = _SCALES.get(unit.lower(), _SCALES.get(unit, 1.0))
        if pct:
            kind = "percent"
            value = v
            scale = 1.0
        elif ps or (pre and not unit and dec == 2 and abs(v) < 1000):
            kind, value = "per_share", v
        elif unit or pre or cur:
            kind, value = "money", v * scale
        else:
            kind, value = "plain", v
        out.append(Mention(
            raw=m.group(0).strip(), value=value, kind=kind, decimals=dec, scale=scale,
            start=base + m.start(), end=base + m.end(), sentence=sentence or text,
            unit_given=bool(unit or cur or pre or pct or ps), clause=text,
        ))
    return out
