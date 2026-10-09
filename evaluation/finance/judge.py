"""可选的 LLM 判官层：只做规则做不了的事 —— 非数字断言是否被引用来源蕴含。

设计取舍（写进报告的局限）:
  * 默认关闭；规则层（numbers/citations/accuracy）已覆盖数字类断言，且零成本、可复现。
  * 判官模型应与被测模型不同或更强（configs 里默认 claude-opus-5-5 当判官、sonnet 当被测），
    避免同一模型评自己的"自偏好"；不能完全消除，因此只报告判官一致率，不把它混进硬指标。
  * 只给判官看"与断言最相关的证据窗口"，不是整篇证据，控制成本，也避免长上下文稀释。
  * 结果按 (断言, 证据窗口) 哈希缓存，重复评测不重复花钱。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

__all__ = ["ClaimJudge", "best_window", "VERDICTS"]

VERDICTS = ["entailed", "partial", "not_supported", "contradicted"]

SYSTEM = (
    "You are a strict fact-checking auditor for financial research reports. Given ONE claim from a report and the "
    "EVIDENCE it cites, decide whether the evidence supports the claim. Judge only from the evidence text; do not use "
    "outside knowledge. Pay attention to numbers, units, periods and entities.\n"
    'Return ONLY JSON: {"verdict": "entailed|partial|not_supported|contradicted", "reason": "<one sentence>"}\n'
    "entailed = fully supported; partial = part of the claim is supported, part is missing or imprecise; "
    "not_supported = the evidence neither supports nor contradicts; contradicted = the evidence says otherwise."
)

_TOKEN = re.compile(r"[A-Za-z]{3,}|\d[\d,\.]*|[一-鿿]{2,}")


def best_window(text: str, claim: str, size: int = 1800) -> str:
    """从证据中取与断言词汇重叠最多的窗口。"""
    if len(text) <= size:
        return text
    toks = {t.lower() for t in _TOKEN.findall(claim)}
    step = size // 2
    best, best_score = 0, -1
    for i in range(0, max(len(text) - step, 1), step):
        w = text[i: i + size].lower()
        score = sum(1 for t in toks if t in w)
        if score > best_score:
            best, best_score = i, score
    return text[best: best + size]


class ClaimJudge:
    def __init__(self, policy: Any, cache: dict | None = None, window: int = 1800) -> None:
        self.policy = policy
        self.cache = cache if cache is not None else {}
        self.window = window
        self.calls = 0

    def judge(self, claim: str, evidence_texts: list[str]) -> dict:
        windows = [best_window(t, claim, self.window) for t in evidence_texts]
        key = hashlib.sha1(json.dumps([claim, windows], ensure_ascii=False).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        user = "CLAIM:\n" + claim + "\n\nEVIDENCE:\n" + "\n---\n".join(windows)
        self.calls += 1
        try:
            resp = self.policy([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            out = self._parse(resp.get("content", "") or "")
        except Exception as e:  # 判官失败不能拖垮评测
            out = {"verdict": "error", "reason": f"{type(e).__name__}: {e}"}
        self.cache[key] = out
        return out

    @staticmethod
    def _parse(content: str) -> dict:
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return {"verdict": "error", "reason": "no JSON in judge reply"}
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"verdict": "error", "reason": "invalid JSON"}
        v = str(d.get("verdict", "")).strip().lower()
        return {"verdict": v if v in VERDICTS else "error", "reason": str(d.get("reason", ""))[:300]}

    def judge_report(self, units, evidence: list[dict], max_claims: int = 40) -> dict:
        """只审"有引用、且不含数字提及"的断言（数字类已由规则层确定性核对）。"""
        by_id = {e["id"]: e for e in evidence}
        items = []
        for u in units:
            if u.mentions or not u.cites or len(u.text) < 25:
                continue
            texts = [by_id[c]["text"] for c in u.cites if c in by_id]
            if texts:
                items.append((u.text, texts))
        items = items[:max_claims]
        results = [dict(claim=c, **self.judge(c, t)) for c, t in items]
        n = len(results)
        counts = {v: sum(1 for r in results if r["verdict"] == v) for v in VERDICTS + ["error"]}
        ok = counts["entailed"]
        return {
            "n_claims": n, "counts": counts,
            "entailed_rate": ok / n if n else None,
            "contradicted_rate": counts["contradicted"] / n if n else None,
            "details": results,
        }
