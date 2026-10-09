"""Agent subclasses for the finance scenario: override only the prompt / tool recommendations / evidence catalog, not the framework's main loop."""
from __future__ import annotations

import re
from typing import Any

from ..agents.researcher import ResearcherAgent
from ..agents.summarizer import SummarizerAgent
from ..orchestrator.schemas import AgentResult, AgentStatus, ResearchReport, SubTask
from ..planner.planner import Planner
from .evidence import EvidenceLedger
from .prompts import (
    DOC_KEYWORDS,
    NON_US_KEYWORDS,
    NUMERIC_KEYWORDS,
    planner_prompt,
    researcher_system_prompt,
    summarizer_system_prompt,
)

__all__ = ["FinanceResearcherAgent", "FinancePlanner", "FinanceSummarizerAgent"]


def _has(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


class FinanceResearcherAgent(ResearcherAgent):
    def __init__(self, name, policy, tools=None, max_turns: int = 10, language: str = "zh", max_tool_calls: int = 8):
        super().__init__(name, policy, tools, max_turns=max_turns)
        self.language = language
        self.max_tool_calls = max_tool_calls

    def _system_prompt(self) -> str:
        return researcher_system_prompt(self.language, self.max_tool_calls)

    def _is_non_searchable(self, task: SubTask, context: dict) -> bool:
        # in the finance scenario "direct analysis" would allow answering from memory, so it must be off
        return False

    def _recommend(self, task: SubTask) -> list[str]:
        d = task.description or ""
        names = set(self.tool_map)
        if _has(d, NON_US_KEYWORDS):
            rec = ["web_search", "browser"]
        else:
            rec = []
            if _has(d, NUMERIC_KEYWORDS):
                rec += ["sec_facts"]
            if _has(d, DOC_KEYWORDS):
                rec += ["sec_filings", "sec_filing"]
            if not rec:
                rec = ["sec_filings", "sec_facts"]
            rec += ["calculator"]
        return [t for t in rec if t in names] or [t for t in ("web_search",) if t in names]

    def _build_task_prompt(self, task: SubTask, context: dict) -> str:
        rec = self._recommend(task)
        lines = [
            f"## Task: {task.description}",
            f"Type: {task.task_type.value}",
            f"Expected output: {task.expected_type}",
            "",
            f"## RECOMMENDED TOOLS (in priority order): {', '.join(rec)}",
            "",
            "## INSTRUCTIONS:",
            "1. Identify the company (name + ticker) and the exact fiscal period from the task.",
            "2. Retrieve evidence with the recommended tools. Every claim you write must cite the "
            "`evidence_id` returned by a tool, as [id].",
            "3. Use `calculator` for every derived number; cite the calculator id and the input ids.",
            "4. If something cannot be found, list it under Gaps. Do NOT estimate or use memory.",
            "5. DO NOT greet the user or ask clarifying questions — execute immediately.",
        ]
        if task.search_hints:
            lines.insert(1, f"Search hints: {', '.join(task.search_hints)}")
        parts = []
        for key in task.context_keys or []:
            if key in context:
                parts.append(f"- {key}: {context[key]}")
        if parts:
            lines.append("\n## Context from earlier tasks:")
            lines.extend(parts)
        return "\n".join(lines)

    def _fallback_tool(self, task: SubTask) -> str:
        d = task.description or ""
        names = set(self.tool_map)
        if _has(d, NON_US_KEYWORDS) and "web_search" in names:
            return "web_search"
        if _has(d, NUMERIC_KEYWORDS) and "sec_facts" in names:
            return "sec_facts"
        if "sec_filings" in names:
            return "sec_filings"
        return "web_search"

    def _is_tool_failure_explanation(self, content: str) -> bool:
        # a long report may legitimately say "unable to retrieve" in its Gaps section; only very short replies count as a tool failure
        if not content or len(content) > 600:
            return False
        return super()._is_tool_failure_explanation(content)


class FinancePlanner(Planner):
    def __init__(self, policy, budget_tracker=None, max_sub_tasks: int = 8) -> None:
        super().__init__(policy, budget_tracker)
        self.max_sub_tasks = max_sub_tasks

    def _build_prompt(self, query: str, memory: str) -> str:
        has_memory = bool(memory and memory.strip() and memory != "None")
        return planner_prompt(query, memory if has_memory else "None", self.max_sub_tasks)


_CITE_RE = re.compile(r"\[(\d+)\]")


class FinanceSummarizerAgent(SummarizerAgent):
    """Summarizer: puts the evidence catalog into the prompt so [n] matches the ledger ids; the report carries the ledger's sources."""

    def __init__(self, name, policy, tools=None, ledger: EvidenceLedger | None = None, language: str = "zh",
                 catalog_chars: int = 240):
        super().__init__(name, policy, tools)
        self.ledger = ledger
        self.language = language
        self.catalog_chars = catalog_chars

    def _system_prompt(self) -> str:
        return summarizer_system_prompt(self.language)

    def _catalog(self) -> str:
        if not self.ledger or not len(self.ledger):
            return "(empty — no tool evidence was collected; do not cite any source)"
        rows = []
        for e in self.ledger.to_list():
            snippet = " ".join(e["text"].split())[: self.catalog_chars]
            rows.append(f"[{e['id']}] ({e['kind']}) {e['title']} | {e['url']}\n    {snippet}")
        return "\n".join(rows)

    def _build_synthesis_prompt(self, query: str, results: list[AgentResult]) -> str:
        ordered = sorted(results, key=lambda r: r.confidence, reverse=True)
        parts = [f"# Research Question\n{query}\n", "# Evidence Catalog (cite ONLY these ids)\n" + self._catalog() + "\n",
                 f"# Sub-task Findings ({len(results)} total)\n"]
        for i, r in enumerate(ordered, 1):
            icon = "OK" if r.status == AgentStatus.SUCCESS else "FAILED"
            parts.append(f"## Finding {i} [{icon}] (confidence {r.confidence:.2f}) task={r.task_id}\n{r.output}\n")
        parts.append(
            "\n# Instructions\n"
            "1. Write the report directly (no preamble). Use only facts from the findings above.\n"
            "2. Cite with [n] from the Evidence Catalog after every sentence/table cell containing a figure or claim.\n"
            "3. State fiscal period, period end, units and GAAP basis for every figure.\n"
            "4. Put anything missing or conflicting under 'Data notes and limitations'.\n"
            "5. End with: Overall Confidence: X.XX"
        )
        return "\n".join(parts)

    def _parse_report(self, query: str, content: str, results: list[AgentResult]) -> ResearchReport:
        report = super()._parse_report(query, content, results)
        if self.ledger is not None:
            report.sources = self.ledger.to_sources()
            report.evidence = self.ledger.to_list()
        return report
