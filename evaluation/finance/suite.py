"""评测套件：加载用例 → 评测一批报告 → 汇总（合并计数 + Wilson 区间）→ Markdown 报告。"""
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
    L = ["# 财报研究 Agent 评测报告", "", f"评测用例 {agg['n_evaluated']}/{agg['n_cases']}"]
    if agg.get("missing"):
        L.append(f"（缺失: {', '.join(agg['missing'])}）")
    if "citation" in agg:
        c, a = agg["citation"], agg["accuracy"]
        ci = lambda d: f" (95% CI {_pct(d['ci95'][0])}–{_pct(d['ci95'][1])})" if d.get("ci95") else ""  # noqa: E731
        L += ["", "## 引用核对", "",
              f"- 引用精度（被引来源确含该数字）: **{_pct(c['precision']['value'])}**{ci(c['precision'])}，n={c['precision']['n']}",
              f"- 引用覆盖（数字中带引用的比例）: {_pct(c['coverage']['value'])}",
              f"- 无引用且无证据可查的数字占比（疑似凭记忆）: {_pct(c['uncited_ungrounded_rate']['value'])}",
              f"- 含悬空引用编号的报告数: {c['reports_with_dangling_ids']}",
              f"- 明细: {c['counts']}", "", "## 数据准确性（对 SEC XBRL 金标准）", "",
              f"- 准确率: **{_pct(a['accuracy']['value'])}**{ci(a['accuracy'])}，可映射数字 n={a['accuracy']['n']}，映射率 {_pct(a['mapping_rate'])}",
              f"- 核心指标召回（营收/净利润/EPS 是否写对）: {_pct(a['headline_recall_mean'])}",
              f"- 错误分类: {a['errors']}", f"- 触发硬性标记的报告: {agg['reports_with_hard_flags']}"]
    L += ["", "## 逐用例", "", "| case | 引用精度 | 准确率 | 悬空引用 | 硬性标记 |", "|---|---|---|---|---|"]
    for r in results:
        if r.get("status") != "ok":
            L.append(f"| {r['id']} | {r['status']} | | | |")
            continue
        L.append(f"| {r['id']} | {_pct(r['citation']['citation_precision'])} | {_pct(r.get('accuracy', {}).get('accuracy'))} "
                 f"| {r['citation']['dangling_ids'] or '-'} | {', '.join(r['hard_flags']) or '-'} |")
    if meta:
        cl = meta["clean"]
        L += ["", "## 评测器自检（元评测：向合成干净报告注入已知错误）", "",
              f"- 干净报告误报率: {_pct(cl['false_positive_rate'])}（{cl['reports_with_false_flags']}/{cl['n_reports']} 份），"
              f"数字映射率 {_pct(cl['mapping_rate'])}", "", "| 注入的错误 | n | 检出率 | 归类正确率 |", "|---|---|---|---|"]
        for k, v in meta["perturbations"].items():
            L.append(f"| {k} | {v['n']} | {_pct(v['detection_rate'])} | {_pct(v['type_accuracy'])} |")
        L += ["", "> 注入错误是人造的、报告句式规整，检出率是评测器的**上界**；真实模型的错误更杂。"]
    L += ["", "## 方法局限", "",
          "- 数字→(指标, 期间) 的映射是启发式（别名表 + 就近原则），映射不了的数字计入『未映射』而不是『正确』。",
          "- 金标准目前只覆盖 XBRL 标准概念；MD&A 里的分部数据、非 GAAP 指标、指引、日期不在自动核对范围内。",
          "- 引用核对在『子句』粒度判断：同一子句挂多个引用时，只要其中之一含该数字就算支持。",
          "- 金标准取数与 Agent 工具共用 select_period 代码；取数逻辑本身的正确性由人工核对的 truth.json 回归测试单独保证。",
          "- LLM 判官（若开启）只评非数字断言，且存在模型自偏好，不并入硬指标。"]
    return "\n".join(L) + "\n"
