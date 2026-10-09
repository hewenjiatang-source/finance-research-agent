"""Citation verification — the deterministic rule layer.

For every numeric claim in the report it answers three questions:
  1. Does the cited id exist? (dangling: the id is not in the evidence ledger — a typical hallucinated citation)
  2. Does the cited source really contain the number? (supported / misattributed / unsupported)
  3. For an uncited number, can it at least be found in some evidence? (uncited_grounded / uncited_ungrounded)

Matching rule (see evidence_candidates): the number and its unit are normalized together and compared "precision-aware";
a bare number is only multiplied by a scale when the evidence declares "in millions/thousands", so a x1000 scale error is not mistaken for grounded.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .numbers import Mention, extract_mentions

__all__ = ["CitationRecord", "check_citations", "evidence_candidates"]

_RAW_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])-?\d+(?:\.\d+)?")
_DECLARED = [
    (re.compile(r"in\s+thousands|\(thousands\)|千美元|千元|单位[:：]?\s*千", re.I), 1e3),
    (re.compile(r"in\s+millions|\(millions\)|百万美元|百万元|单位[:：]?\s*百万", re.I), 1e6),
    (re.compile(r"in\s+billions|\(billions\)|十亿美元|单位[:：]?\s*十亿", re.I), 1e9),
]
_MAX_TOKENS = 20000


@dataclass
class CitationRecord:
    raw: str
    sentence: str
    cites: list[int]
    status: str   # supported | misattributed | unsupported | dangling | uncited_grounded | uncited_ungrounded | derived_unverified
    found_in: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


def evidence_candidates(text: str) -> list[tuple[float, float]]:
    """Evidence text -> [(normalized value, half_ulp)]. Includes: mentions with units, bare numbers x1, bare numbers x declared scales."""
    out: list[tuple[float, float]] = []
    scales = [s for rx, s in _DECLARED if rx.search(text)]
    for m in extract_mentions(text):
        if m.kind in ("money", "per_share", "percent", "count"):
            out.append((m.value, m.half_ulp))
    seen = 0
    for t in _RAW_NUM.finditer(text):
        seen += 1
        if seen > _MAX_TOKENS:
            break
        raw = t.group(0)
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        dec = len(raw.split(".")[1]) if "." in raw else 0
        h = 0.5 * 10 ** -dec
        out.append((v, h))
        for s in scales:
            out.append((v * s, h * s))
    return out


def _match(m: Mention, cands: list[tuple[float, float]]) -> bool:
    for v, h in cands:
        if abs(m.value - v) <= max(m.half_ulp, h) + 1e-9 * max(abs(v), 1.0):
            return True
    return False


def _derived_ok(m: Mention, evidence: list[dict]) -> bool:
    """Derived numbers such as percentages: the result of a calculator evidence, or recomputable from two numbers in structured evidence (XBRL / computation)."""
    pool: list[float] = []
    for e in evidence:
        if e["kind"] == "computation" and _match(m, evidence_candidates(e["text"])):
            return True
        if e["kind"] in ("xbrl_facts", "computation"):
            pool += [v for v, _ in evidence_candidates(e["text"]) if v and abs(v) >= 1e3]
    pool = sorted(set(pool))[:200]
    tol = m.half_ulp + 0.02
    for a in pool:
        for b in pool:
            if b and (abs((a / b - 1) * 100 - m.value) <= tol or abs(a / b * 100 - m.value) <= tol):
                return True
    return False


def check_citations(units, evidence: list[dict]) -> list[CitationRecord]:
    by_id = {e["id"]: e for e in evidence}
    cache: dict[int, list[tuple[float, float]]] = {}

    def cands(eid: int):
        if eid not in cache:
            cache[eid] = evidence_candidates(by_id[eid]["text"])
        return cache[eid]

    recs: list[CitationRecord] = []
    for u in units:
        for m in u.mentions:
            if m.kind == "plain":
                continue
            cited = [c for c in u.cites if c in by_id]
            missing = [c for c in u.cites if c not in by_id]
            if u.cites and not cited:
                recs.append(CitationRecord(m.raw, m.sentence, u.cites, "dangling", []))
                continue
            if cited:
                if any(_match(m, cands(c)) for c in cited):
                    status, found = "supported", [c for c in cited if _match(m, cands(c))]
                elif m.kind == "percent" and _derived_ok(m, [by_id[c] for c in cited]):
                    status, found = "supported", []
                else:
                    elsewhere = [e for e in by_id if e not in cited and _match(m, cands(e))]
                    if elsewhere:
                        status, found = "misattributed", elsewhere
                    elif m.kind == "percent":
                        status, found = "derived_unverified", []
                    else:
                        status, found = "unsupported", []
                if missing and status == "supported":
                    status = "supported"  # partially dangling citations are counted separately via dangling_ids
                recs.append(CitationRecord(m.raw, m.sentence, u.cites, status, found))
            else:
                found = [e for e in by_id if _match(m, cands(e))]
                recs.append(CitationRecord(m.raw, m.sentence, [], "uncited_grounded" if found else "uncited_ungrounded", found))
    return recs
