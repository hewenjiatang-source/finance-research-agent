"""金标准（gold）：来自 SEC XBRL 申报数据，而不是 LLM 生成。

两种来源：
  * ``gold_from_companyfacts`` —— 用与工具同一套 ``select_period`` 逻辑从 companyfacts 生成；
    注意这里与被测 Agent 共用取数代码，只能证明"Agent 抄对了结构化数据"，
    不能证明取数逻辑本身正确 —— 所以 truth.json（人工核对的夹具真值）单独作为回归基准。
  * ``load_gold`` —— 读取人工核对过的 JSON：{"2023": {"revenue": 4.825e10, ...}, "2022": {...}}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.finance.xbrl import METRICS, select_period

__all__ = ["Gold", "gold_from_companyfacts", "load_gold"]


@dataclass
class Gold:
    company: str                                   # 主名称
    aliases: list[str] = field(default_factory=list)  # 报告里可能出现的称呼（含 ticker）
    periods: dict[str, dict[str, float]] = field(default_factory=dict)  # "FY2023" -> {metric: value}
    period_ends: dict[str, str] = field(default_factory=dict)

    def value(self, period: str, metric: str) -> float | None:
        return self.periods.get(period, {}).get(metric)

    def latest_period(self) -> str | None:
        return max(self.periods, key=lambda p: int(p[2:6]) if p.startswith("FY") else 0) if self.periods else None

    def prior_period(self, period: str) -> str | None:
        if period.startswith("FY") and period[2:6].isdigit():
            prev = f"FY{int(period[2:6]) - 1}"
            return prev if prev in self.periods else None
        return None


def gold_from_companyfacts(cf: dict, fiscal_year: int, aliases: list[str] | None = None) -> Gold:
    g = Gold(company=cf.get("entityName", ""), aliases=list(aliases or []) + [cf.get("entityName", "")])
    for metric in METRICS:
        for fv in select_period(cf, metric, fiscal_year, "FY"):
            g.periods.setdefault(fv.period_label, {})[metric] = fv.value
            g.period_ends[fv.period_label] = fv.end
    return g


def load_gold(path: str | Path, company: str = "", aliases: list[str] | None = None) -> Gold:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    periods = {(k if str(k).startswith("FY") else f"FY{k}"): v for k, v in raw.items() if isinstance(v, dict)}
    return Gold(company=company, aliases=list(aliases or []) + ([company] if company else []), periods=periods)
