"""引用核对（citation verification）—— 确定性规则层。

对报告里每条"带数字的断言"回答三个问题：
  1. 引用编号存在吗？（dangling：编号不在证据账本里 —— 典型幻觉引用）
  2. 被引来源里真的有这个数吗？（supported / misattributed / unsupported）
  3. 没有引用的数字，是不是至少能在某条证据里找到？（uncited_grounded / uncited_ungrounded）

匹配规则（见 _candidates）：数字与单位一起归一后做"精度感知"比较；
证据文本声明了 "in millions/thousands" 时，才允许裸数字乘以对应倍数，因此 ×1000 的量级错误不会被误判为有据。
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
    """证据文本 → [(归一值, half_ulp)]。含：带单位的提及、裸数字×1、裸数字×声明的倍数。"""
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
    """百分比等派生数：来自计算器证据的结果，或能由结构化证据（XBRL/计算）中两个数复算。"""
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
                    status = "supported"  # 部分引用悬空由 dangling_ids 单独统计
                recs.append(CitationRecord(m.raw, m.sentence, u.cites, status, found))
            else:
                found = [e for e in by_id if _match(m, cands(e))]
                recs.append(CitationRecord(m.raw, m.sentence, [], "uncited_grounded" if found else "uncited_ungrounded", found))
    return recs
