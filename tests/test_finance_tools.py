"""XBRL 取数 / SEC 工具 / 证据账本 的离线测试（使用合成夹具，不联网）。"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:  # 沙箱里可能没有 aiohttp；本测试不需要真实网络栈，只需让包可导入
    import aiohttp  # noqa: F401
except ImportError:
    sys.modules["aiohttp"] = types.ModuleType("aiohttp")

from src.finance.evidence import EvidenceLedger, LedgerTool  # noqa: E402
from src.finance.xbrl import resolve_metric, select_period  # noqa: E402
from src.tools.sec_edgar import (  # noqa: E402
    SecClient, SecFactsTool, SecFilingsTool, SecFilingTool, extract_section, html_to_text,
)

FIX = ROOT / "tests" / "fixtures" / "finance"
CF = json.loads((FIX / "acme_companyfacts.json").read_text(encoding="utf-8"))
TRUTH = json.loads((FIX / "truth.json").read_text(encoding="utf-8"))


def fetcher(url: str) -> str:
    if url.endswith("company_tickers.json"):
        return (FIX / "company_tickers.json").read_text(encoding="utf-8")
    if "submissions/CIK0001234567" in url:
        return (FIX / "acme_submissions.json").read_text(encoding="utf-8")
    if "companyfacts/CIK0001234567" in url:
        return (FIX / "acme_companyfacts.json").read_text(encoding="utf-8")
    if url.endswith("acme-20230930.htm"):
        return (FIX / "acme_10k_fy2023.html").read_text(encoding="utf-8")
    raise urllib.error.HTTPError(url, 404, "not found", None, None)  # type: ignore[arg-type]


def run(coro):
    return asyncio.run(coro)


class TestXbrl(unittest.TestCase):
    def test_annual_current_and_prior(self):
        cur, prior = select_period(CF, "revenue", 2023, "FY")
        self.assertEqual(cur.value, TRUTH["2023"]["revenue"])  # 不是 10-K/A 的 99,999,000,000
        self.assertEqual(cur.period_label, "FY2023")
        self.assertEqual(cur.end, "2023-09-30")
        self.assertEqual(cur.form, "10-K")
        self.assertEqual(prior.value, TRUTH["2022"]["revenue"])
        self.assertEqual(prior.period_label, "FY2022")

    def test_amendment_and_later_filing_not_mixed_in(self):
        # FY2024 10-K 重复披露了 FY2023 数值（fy=2024），不应影响 FY2023 的选取
        cur, _ = select_period(CF, "net_income", 2023, "FY")
        self.assertEqual(cur.accn, "0001234567-23-000010")

    def test_instant_metric(self):
        cur, prior = select_period(CF, "total_assets", 2023, "FY")
        self.assertEqual(cur.value, TRUTH["2023"]["total_assets"])
        self.assertIsNone(cur.start)
        self.assertEqual(prior.value, TRUTH["2022"]["total_assets"])

    def test_quarter_excludes_cumulative_rows(self):
        cur, prior = select_period(CF, "revenue", 2024, "Q1")
        self.assertEqual(cur.value, 12_300_000_000)  # 不是 9 个月累计的 36,000,000,000
        self.assertEqual(prior.value, 11_500_000_000)
        self.assertEqual(cur.period_label, "Q1 FY2024")

    def test_missing_returns_empty(self):
        self.assertEqual(select_period(CF, "revenue", 2019, "FY"), [])
        self.assertEqual(select_period(CF, "not_a_metric", 2023, "FY"), [])

    def test_resolve_metric_aliases(self):
        self.assertEqual(resolve_metric("Net Income"), "net_income")
        self.assertEqual(resolve_metric("NetIncomeLoss"), "net_income")
        self.assertEqual(resolve_metric("营业收入"), "revenue")
        self.assertEqual(resolve_metric("operating cash flow"), "operating_cash_flow")
        self.assertIsNone(resolve_metric("foo"))


class TestSecTools(unittest.TestCase):
    def setUp(self):
        self.client = SecClient(fetcher=fetcher)

    def test_filings_filtered_and_urls_built(self):
        res = run(SecFilingsTool(self.client).execute("ACME", form_types=["10-K"], limit=5))
        self.assertTrue(res["ok"])
        self.assertEqual([f["form"] for f in res["filings"]], ["10-K", "10-K", "10-K"])
        fy23 = [f for f in res["filings"] if f["period"] == "2023-09-30"][0]
        self.assertEqual(fy23["url"], "https://www.sec.gov/Archives/edgar/data/1234567/000123456723000010/acme-20230930.htm")

    def test_company_resolution_by_name_and_cik(self):
        self.assertEqual(self.client.resolve_company("acme corp")["ticker"], "ACME")  # 名称命中最短标题
        self.assertEqual(self.client.resolve_company("1234567")["cik"], 1234567)
        self.assertIsNone(self.client.resolve_company("zzz-unknown"))

    def test_facts_tool_default_metrics_and_provenance(self):
        res = run(SecFactsTool(self.client).execute("ACME", 2023))
        self.assertTrue(res["ok"])
        by = {(f["metric"], f["period_label"]): f for f in res["facts"]}
        self.assertEqual(by[("revenue", "FY2023")]["value"], 48_250_000_000)
        self.assertEqual(by[("net_income", "FY2022")]["value"], 6_610_000_000)
        self.assertEqual(by[("eps_diluted", "FY2023")]["value"], 3.71)
        self.assertIn("accn", by[("revenue", "FY2023")])
        self.assertIn("billion", by[("revenue", "FY2023")]["display"])
        self.assertNotIn("error", res)

    def test_facts_tool_failure_has_no_error_key(self):
        # 返回 error 键会让 ResearcherAgent 直接判子任务失败；这里必须是 ok=False
        res = run(SecFactsTool(self.client).execute("ACME", 2001))
        self.assertFalse(res["ok"])
        self.assertNotIn("error", res)
        res = run(SecFactsTool(self.client).execute("NOSUCH", 2023))
        self.assertFalse(res["ok"])
        self.assertNotIn("error", res)

    def test_facts_tool_alias_and_unknown_metric(self):
        res = run(SecFactsTool(self.client).execute("ACME", 2023, metrics=["营业收入", "bogus_metric"]))
        self.assertTrue(res["ok"])
        self.assertEqual({f["metric"] for f in res["facts"]}, {"revenue"})
        self.assertEqual(res["unknown_metrics"], ["bogus_metric"])

    def test_filing_reader_section_skips_toc_and_renders_tables(self):
        url = "https://www.sec.gov/Archives/edgar/data/1234567/000123456723000010/acme-20230930.htm"
        res = run(SecFilingTool(self.client).execute(url, section="7"))
        self.assertTrue(res["ok"], res)
        self.assertIn("increase of 8.2%", res["text"])
        self.assertNotIn("Risk Factors", res["text"][:200])
        res8 = run(SecFilingTool(self.client).execute(url, section="Item 8"))
        self.assertIn("Net sales | $48,250 | $44,600", res8["text"])
        self.assertNotIn("HIDDEN_XBRL_NOISE", res8["text"])

    def test_filing_reader_find_and_limits(self):
        url = "https://www.sec.gov/Archives/edgar/data/1234567/000123456723000010/acme-20230930.htm"
        res = run(SecFilingTool(self.client).execute(url, find=["diluted share"], max_chars=300))
        self.assertTrue(res["ok"])
        self.assertLessEqual(len(res["text"]), 300)
        self.assertTrue(res["truncated"])
        miss = run(SecFilingTool(self.client).execute(url, section="9B"))
        self.assertFalse(miss["ok"])

    def test_filing_reader_rejects_non_sec_urls(self):
        for bad in ("https://evil.example.com/a.htm", "http://www.sec.gov/a.htm", "file:///etc/passwd"):
            res = run(SecFilingTool(self.client).execute(bad))
            self.assertFalse(res["ok"], bad)

    def test_not_configured_message(self):
        c = SecClient(user_agent=None)
        c.user_agent = None
        res = run(SecFactsTool(c).execute("ACME", 2023))
        self.assertFalse(res["ok"])
        self.assertIn("SEC_USER_AGENT", res["message"])

    def test_html_to_text_and_extract_section_direct(self):
        text = html_to_text((FIX / "acme_10k_fy2023.html").read_text(encoding="utf-8"))
        sec = extract_section(text, "1A")
        self.assertIn("supply chain", sec)
        self.assertIsNone(extract_section(text, "99"))


class TestLedger(unittest.TestCase):
    def test_dedup_and_ids_are_stable(self):
        led = EvidenceLedger()
        a = led.add("web_page", "http://x/1", "t", "hello")
        b = led.add("web_page", "http://x/2", "t", "hello")
        a2 = led.add("web_page", "http://x/1", "t", "hello")
        self.assertEqual((a, b, a2), (1, 2, 1))
        led.reset()
        self.assertEqual(len(led), 0)

    def test_ledger_tool_registers_sec_facts_and_calculator(self):
        led = EvidenceLedger()
        facts = LedgerTool(SecFactsTool(SecClient(fetcher=fetcher)), led)
        res = run(facts.execute("ACME", 2023))
        eid = res["evidence_id"]
        ev = led.get(eid)
        self.assertEqual(ev.kind, "xbrl_facts")
        self.assertIn("value 48250000000 USD", ev.text)  # 评测靠这行文本核对数字
        self.assertEqual(facts.get_openai_tool_schema()["function"]["name"], "sec_facts")

        from src.tools.calculator import CalculatorTool
        calc = LedgerTool(CalculatorTool(), led)
        out = run(calc.execute("48250/44600-1"))
        self.assertEqual(led.get(out["evidence_id"]).kind, "computation")
        self.assertIn("48250/44600-1 =", led.get(out["evidence_id"]).text)
        # 位置参数（Blue Agent 的调用方式）同样可用
        out2 = run(calc.execute("1+1"))
        self.assertIn("1+1 = 2", led.get(out2["evidence_id"]).text)

    def test_ledger_tool_wraps_browser_string_into_dict(self):
        class FakeBrowser:
            name = "browser"
            description = "d"

            def get_openai_tool_schema(self):
                return {"type": "function", "function": {"name": "browser"}}

            async def execute(self, url, max_chars=8000):
                return "Press release\nRevenue was $48.25 billion."

        led = EvidenceLedger()
        res = run(LedgerTool(FakeBrowser(), led).execute("https://ir.acme.example/pr"))
        self.assertEqual(res["evidence_id"], 1)
        self.assertEqual(led.get(1).url, "https://ir.acme.example/pr")
        self.assertEqual(led.get(1).title, "Press release")

    def test_ledger_roundtrip_to_file(self):
        import tempfile
        led = EvidenceLedger()
        led.add("web_page", "http://x", "t", "内容")
        with tempfile.TemporaryDirectory() as d:
            path = led.save(Path(d) / "r.evidence.json", query="q")
            loaded = EvidenceLedger.load(path)
        self.assertEqual(loaded[0]["text"], "内容")


if __name__ == "__main__":
    unittest.main()
