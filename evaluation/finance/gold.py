"""Gold standard: comes from SEC XBRL filing data, not from an LLM.

Two sources:
  * ``gold_from_companyfacts`` — generated from companyfacts with the same ``select_period`` logic the tools use;
    note this shares the extraction code with the agent under test, so it only proves the agent copied the structured data faithfully,
    not that the extraction logic itself is right — hence truth.json (hand-checked fixture truth) is kept as a separate regression baseline.
  * ``load_gold`` — reads a hand-checked JSON: {"2023": {"revenue": 4.825e10, ...}, "2022": {...}}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.finance.xbrl import METRICS, select_period

__all__ = ["Gold", "gold_from_companyfacts", "load_gold"]


@dataclass
class Gold:
    company: str                                   # primary name
    aliases: list[str] = field(default_factory=list)  # names the report may use (including the ticker)
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
