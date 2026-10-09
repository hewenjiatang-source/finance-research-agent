"""财报场景端到端（假模型 + 合成 SEC 夹具）：规划 → 研究员取数 → 合成 → 证据账本/侧车文件。"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import aiohttp  # noqa: F401
except ImportError:
    sys.modules["aiohttp"] = types.ModuleType("aiohttp")

from src.core.runner import _format_report, save_report  # noqa: E402
from src.finance.agents import FinancePlanner, FinanceResearcherAgent, FinanceSummarizerAgent  # noqa: E402
from src.finance.evidence import EvidenceLedger, LedgerTool  # noqa: E402
from src.orchestrator.agent_pool import AgentPool  # noqa: E402
from src.orchestrator.orchestrator import Orchestrator  # noqa: E402
from src.planner.budget_tracker import BudgetTracker  # noqa: E402
from src.tools import CalculatorTool  # noqa: E402
from src.tools.sec_edgar import SecClient, create_sec_tools  # noqa: E402
from tests.test_finance_tools import fetcher  # noqa: E402

PLAN = {"sub_tasks": [
    {"task_id": "task_1", "task_type": "search",
     "description": "Pull FY2023 and FY2022 revenue and net income for Acme Corp (ACME) from SEC XBRL data",
     "dependencies": [], "search_hints": ["ACME"]},
    {"task_id": "task_2", "task_type": "analyze",
     "description": "Compute FY2023 revenue growth for Acme Corp (ACME)",
     "dependencies": ["task_1"], "context_keys": []},
]}


class ScriptedPolicy:
    """按 system prompt 区分角色的假 Claude。无状态：只看 messages。"""

    def __init__(self) -> None:
        self.tools = None
        self.calls: list[str] = []

    def set_tools(self, schemas):
        self.tools = schemas

    def __call__(self, messages):
        system = messages[0]["content"]
        if "planning assistant" in system:
            self.calls.append("planner")
            return {"role": "assistant", "content": json.dumps(PLAN), "tool_calls": []}
        if "report writer" in system:
            self.calls.append("summarizer")
            catalog = messages[1]["content"]
            ids = re.findall(r"^\[(\d+)\] \(xbrl_facts\)", catalog, re.M)
            i = ids[0] if ids else "99"
            body = (f"## Summary\nAcme FY2023 revenue was 48,250.0 million USD [{i}].\n"
                    f"Net income FY2023 was 7,425.0 million USD [{i}].\nOverall Confidence: 0.80")
            return {"role": "assistant", "content": body, "tool_calls": []}
        # researcher
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        if not tool_msgs:
            self.calls.append("researcher-call")
            call = {"id": "tc1", "type": "function", "function": {
                "name": "sec_facts",
                "arguments": json.dumps({"company": "ACME", "fiscal_year": 2023, "metrics": ["revenue", "net_income"]})}}
            return {"role": "assistant", "content": "", "tool_calls": [call]}
        res = json.loads(tool_msgs[-1]["content"].split("\n\n[SYSTEM")[0])
        eid = res["evidence_id"]
        self.calls.append("researcher-final")
        return {"role": "assistant", "tool_calls": [], "content":
                f"Findings:\n- revenue = 48,250.0 million USD, FY2023 [{eid}]\nConfidence: 0.9"}


class TestFinanceE2E(unittest.TestCase):
    def test_pipeline(self):
        policy = ScriptedPolicy()
        ledger = EvidenceLedger()
        client = SecClient(user_agent="Test test@example.com", fetcher=fetcher, min_interval=0)
        tools = [LedgerTool(t, ledger) for t in create_sec_tools(client) + [CalculatorTool()]]
        planner = FinancePlanner(policy, BudgetTracker())
        pool = AgentPool(policy_factory=lambda: policy, tools_factory=lambda: list(tools), researcher_cls=FinanceResearcherAgent)
        orch = Orchestrator(
            planner=planner, agent_pool=pool, budget_tracker=BudgetTracker(), summarizer_policy=policy,
            summarizer_factory=lambda p, t: FinanceSummarizerAgent("summarizer", p, t, ledger=ledger, language="en"),
        )
        from src.orchestrator.schemas import RunConfig

        report = asyncio.run(orch.run("Acme FY2023 revenue", config=RunConfig(enable_adversarial=False)))
        self.assertIn("[1]", report.content)
        self.assertEqual(len(report.evidence), len(ledger))
        self.assertEqual(report.evidence[0]["kind"], "xbrl_facts")
        self.assertIn("48250000000", report.evidence[0]["text"])
        self.assertEqual(report.sources[0]["id"], 1)

        md = _format_report(report, 1.0)
        self.assertIn("[1] [", md)  # 参考来源编号 == evidence id
        with tempfile.TemporaryDirectory() as d:
            path = save_report(md, "acme", d, evidence=report.evidence)
            side = Path(path[:-3] + ".evidence.json")
            self.assertTrue(side.exists())
            self.assertEqual(json.loads(side.read_text())["schema"], "evidence/v1")

    def test_english_report_labels(self):
        from src.orchestrator.schemas import ResearchReport

        r = ResearchReport(query="q", content="body [1]", sources=[{"id": 1, "url": "u", "title": "t", "snippet": "s"}], confidence=0.5)
        md = _format_report(r, 1.0, "en")
        self.assertIn("## Metadata", md)
        self.assertIn("## References", md)
        self.assertIn("[1] [t](u)", md)

    def test_researcher_prompt_is_finance_specific(self):
        a = FinanceResearcherAgent("r", ScriptedPolicy(), [t for t in create_sec_tools(SecClient(fetcher=fetcher))])
        p = a._system_prompt()
        self.assertIn("evidence_id", p)
        self.assertFalse(a._is_non_searchable(PLAN_TASK, {"query": "我的朋友"}))
        self.assertEqual(a._fallback_tool(PLAN_TASK), "sec_facts")


from src.orchestrator.schemas import SubTask, TaskType  # noqa: E402

PLAN_TASK = SubTask(task_id="t", task_type=TaskType.SEARCH, description="Get revenue and net income for ACME")

if __name__ == "__main__":
    unittest.main()
