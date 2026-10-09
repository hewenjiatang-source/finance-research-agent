"""财报评测模块的单元测试（全部离线）。"""
from __future__ import annotations

import copy
import json
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

from evaluation.finance.accuracy import check_accuracy  # noqa: E402
from evaluation.finance.citations import check_citations, evidence_candidates  # noqa: E402
from evaluation.finance.claims import parse_report  # noqa: E402
from evaluation.finance.core import evaluate_report  # noqa: E402
from evaluation.finance.gold import load_gold  # noqa: E402
from evaluation.finance.judge import ClaimJudge, best_window  # noqa: E402
from evaluation.finance.numbers import extract_mentions  # noqa: E402
from evaluation.finance.perturb import meta_eval, synth_case  # noqa: E402
from evaluation.finance.suite import aggregate, build_case, evaluate_dir, render_markdown, wilson  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "finance"
CF = json.loads((FIX / "acme_companyfacts.json").read_text(encoding="utf-8"))
GOLD = load_gold(FIX / "truth.json", "Acme Corp", ["ACME"])


def acc(text):
    recs = []
    for u in parse_report(text):
        recs += check_accuracy(u.mentions, GOLD)
    return recs


class TestNumbers(unittest.TestCase):
    def vals(self, t):
        return [(m.value, m.kind) for m in extract_mentions(t)]

    def test_units_normalised(self):
        self.assertEqual(self.vals("48,250.0 million USD"), [(4.825e10, "money")])
        self.assertEqual(self.vals("$48.25 billion"), [(4.825e10, "money")])
        self.assertEqual(self.vals("482.5亿美元"), [(4.825e10, "money")])
        self.assertEqual(self.vals("diluted EPS $3.71"), [(3.71, "per_share")])
        self.assertEqual(self.vals("up 8.2%"), [(8.2, "percent")])

    def test_ignores_years_ids_and_small_ints(self):
        self.assertEqual(self.vals("In FY2023 (Q2) the 10-K filed 2023-09-30 listed 3 drivers in 2023年"), [])

    def test_precision_is_kept(self):
        m = extract_mentions("48,250.0 million USD")[0]
        self.assertAlmostEqual(m.half_ulp, 5e4)


class TestAccuracy(unittest.TestCase):
    def test_correct_in_any_unit_spelling(self):
        for t in ["FY2023 revenue was 48,250.0 million USD.", "FY2023 revenue was $48.25 billion.", "2023财年营收为482.5亿美元。"]:
            r = acc(t)
            self.assertEqual([x.status for x in r], ["correct"], t)

    def test_rounding_aware(self):
        self.assertEqual(acc("FY2023 revenue was about 48.3 billion USD.")[0].status, "correct")  # 48.25 四舍五入到 1 位小数
        self.assertEqual(acc("FY2023 net income was 7.4 billion USD.")[0].status, "correct")  # 7.425→7.4
        self.assertEqual(acc("FY2023 revenue was 48.4 billion USD.")[0].error_type, "imprecise")  # 超出写出精度但 <1%
        self.assertEqual(acc("FY2023 net income was 7.5 billion USD.")[0].status, "incorrect")

    def test_error_taxonomy(self):
        self.assertEqual(acc("FY2023 revenue was 48,250.0 billion USD.")[0].error_type, "scale_error")
        self.assertEqual(acc("FY2023 revenue was 44,600.0 million USD.")[0].error_type, "wrong_period")
        self.assertEqual(acc("FY2023 revenue was 7,425.0 million USD.")[0].error_type, "wrong_metric")
        self.assertEqual(acc("FY2023 net income was 9,999.0 million USD.")[0].error_type, "wrong_value")

    def test_growth_margin_and_sign(self):
        self.assertEqual(acc("FY2023 revenue grew 8.2% year-over-year.")[0].status, "correct")
        self.assertEqual(acc("FY2023 revenue declined 8.2% year-over-year.")[0].error_type, "sign_error")
        self.assertEqual(acc("FY2023 gross margin was 43.7%.")[0].status, "correct")
        self.assertEqual(acc("FY2023 gross margin was 47.0%.")[0].status, "incorrect")

    def test_period_taken_from_table_header(self):
        t = "| Metric | FY2023 | FY2022 |\n|---|---|---|\n| Revenue | 48,250.0 million USD | 44,600.0 million USD |\n"
        self.assertEqual([x.status for x in acc(t)], ["correct", "correct"])
        t2 = t.replace("FY2023 | FY2022", "FY2022 | FY2023")
        self.assertEqual([x.error_type for x in acc(t2)], ["wrong_period", "wrong_period"])

    def test_unmapped_is_not_counted_correct(self):
        r = acc("Headcount was 164,000 employees and tax rate was 15.9%.")
        self.assertTrue(all(x.status == "unmapped" for x in r))


class TestCitations(unittest.TestCase):
    EV = [
        {"id": 1, "kind": "xbrl_facts", "url": "u", "title": "t", "text": "revenue | FY2023 | value 48250000000 USD (48,250.0 million USD)\nnet_income | value 7425000000 USD"},
        {"id": 2, "kind": "sec_filing_text", "url": "u", "title": "t", "text": "(in millions)\nNet sales 48,250 44,600\nOperating income 9,870"},
    ]

    def st(self, text):
        return [r.status for r in check_citations(parse_report(text), self.EV)]

    def test_statuses(self):
        self.assertEqual(self.st("Revenue was 48,250.0 million USD [1]."), ["supported"])
        self.assertEqual(self.st("Revenue was 48,250.0 million USD [99]."), ["dangling"])
        self.assertEqual(self.st("Operating income was 9,870.0 million USD [1]."), ["misattributed"])
        self.assertEqual(self.st("Operating income was 5,555.0 million USD [1]."), ["unsupported"])
        self.assertEqual(self.st("Revenue was 48,250.0 million USD."), ["uncited_grounded"])
        self.assertEqual(self.st("Revenue was 12,345.0 million USD."), ["uncited_ungrounded"])

    def test_declared_scale_allows_bare_numbers_but_not_x1000(self):
        self.assertEqual(self.st("Operating income was 9,870 million USD [2]."), ["supported"])
        self.assertEqual(self.st("Operating income was 9,870 billion USD [2]."), ["unsupported"])
        self.assertNotIn(9.87e12, [round(v) for v, _ in evidence_candidates(self.EV[1]["text"])])

    def test_clause_level_attribution(self):
        # 同一句里前一半引 [1]、后一半引 [2]：错配要能被发现
        s = self.st("Revenue was 48,250.0 million USD [1], and operating income was 9,870.0 million USD [1].")
        self.assertEqual(s, ["supported", "misattributed"])


class TestJudge(unittest.TestCase):
    def test_parse_cache_and_scope(self):
        calls = []

        def policy(msgs):
            calls.append(1)
            return {"content": 'sure ```{"verdict": "entailed", "reason": "ok"}```'}

        ev = [{"id": 1, "kind": "sec_filing_text", "url": "", "title": "", "text": "Management attributes growth to iPhone demand."}]
        rep = "Management attributes the growth to strong iPhone demand across regions [1].\nRevenue was 5.0 billion USD [1]."
        j = ClaimJudge(policy)
        out = j.judge_report(parse_report(rep), ev)
        self.assertEqual(out["n_claims"], 1)  # 含数字的句子交给规则层
        self.assertEqual(out["entailed_rate"], 1.0)
        j.judge_report(parse_report(rep), ev)
        self.assertEqual(len(calls), 1)  # 缓存命中

    def test_failure_is_contained(self):
        def boom(msgs):
            raise RuntimeError("down")

        r = ClaimJudge(boom).judge("claim text here", ["evidence"])
        self.assertEqual(r["verdict"], "error")

    def test_best_window_prefers_overlap(self):
        text = "x" * 3000 + " Management attributes growth to iPhone demand " + "y" * 3000
        self.assertIn("iPhone", best_window(text, "growth driven by iPhone demand", 800))


class TestMetaEval(unittest.TestCase):
    @staticmethod
    def cases():
        out = []
        for k, name in [(1, "Acme Corp"), (0.37, "Beta Inc"), (2.9, "Gamma Ltd")]:
            c = copy.deepcopy(CF)
            c["entityName"] = name
            for con in c["facts"]["us-gaap"].values():
                for unit, rows in con["units"].items():
                    for r in rows:
                        if unit == "USD":
                            r["val"] = round(r["val"] * k / 1e6) * 1e6
                        elif unit == "USD/shares":
                            r["val"] = round(r["val"] * k, 2)
            for lang in ("en", "zh"):
                out.append(synth_case(c, 2023, name[:4].upper(), lang))
        return out

    def test_no_false_positives_and_known_errors_detected(self):
        res = meta_eval(self.cases())
        self.assertEqual(res["clean"]["false_positive_rate"], 0.0)
        self.assertGreaterEqual(res["clean"]["mapping_rate"], 0.9)
        for name, v in res["perturbations"].items():
            self.assertGreater(v["n"], 0, name)
            floor = 0.5 if name == "dropped_citation" else 1.0  # 见 README：逗号并列子句共享引用时无法判定漏引
            self.assertGreaterEqual(v["detection_rate"], floor, name)
            self.assertGreaterEqual(v["type_accuracy"], floor, name)


class TestSuite(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(9, 10)
        self.assertTrue(0.55 < lo < 0.6 and 0.97 < hi < 1.0)
        self.assertIsNone(wilson(0, 0))

    def test_end_to_end_dir(self):
        case = build_case("acme-fy2023", "q", CF, 2023, "ACME")
        self.assertEqual(case["gold"]["FY2023"]["revenue"], 48250000000)
        sc = synth_case(CF, 2023, "ACME", "en")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "acme-fy2023.md").write_text("# 研究报告：q\n\n---\n\n" + sc.report + "\n---\n\n## 参考来源\n\n1. [x](u) — 99,999\n", encoding="utf-8")
            (d / "acme-fy2023.evidence.json").write_text(json.dumps({"schema": "evidence/v1", "evidence": sc.evidence}), encoding="utf-8")
            res = evaluate_dir(d, [case, {**case, "id": "absent"}])
            self.assertEqual(res[0]["status"], "ok")
            self.assertEqual(res[1]["status"], "missing_report")
            self.assertEqual(res[0]["hard_flags"], [])  # 参考来源里的数字不被当作断言
            agg = aggregate(res)
            self.assertEqual(agg["accuracy"]["accuracy"]["value"], 1.0)
            self.assertEqual(agg["citation"]["precision"]["value"], 1.0)
            self.assertEqual(agg["accuracy"]["headline_recall_mean"], 1.0)
            self.assertIn("引用精度", render_markdown(agg, res))

    def test_flags_on_bad_report(self):
        sc = synth_case(CF, 2023, "ACME", "en")
        bad = sc.report.replace("48,250.0 million USD [1], up", "48,250.0 billion USD [99], up")
        r = evaluate_report(bad, sc.evidence, sc.gold)
        self.assertIn("scale_error", r["hard_flags"])
        self.assertIn("dangling_citation", r["hard_flags"])

    def test_build_cases_script_with_fixture_fetcher(self):
        from scripts.build_finance_cases import build
        from src.tools.sec_edgar import SecClient
        from tests.test_finance_tools import fetcher

        out = build([{"id": "a", "ticker": "ACME", "fiscal_year": 2023, "query": "q"},
                     {"id": "b", "ticker": "ACME", "fiscal_year": 1999, "query": "q"}],
                    SecClient(user_agent="T t@example.com", fetcher=fetcher, min_interval=0))
        self.assertEqual([c["id"] for c in out], ["a"])


if __name__ == "__main__":
    unittest.main()
