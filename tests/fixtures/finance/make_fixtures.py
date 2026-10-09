"""Generate SYNTHETIC SEC fixtures (fictional company Acme Corp; numbers are made up, for tests only).

Deliberately plants pitfalls found in real XBRL, to verify the data-selection logic:
  * "wrong" values in a 10-K/A amendment (must be excluded)
  * 9-month cumulative values in a 10-Q (same accn as the quarter, different duration)
  * the next year's 10-K repeating the prior-year values (different fy, must not be taken as current)
  * current and prior comparative values in the same filing (same fy, different end)

Run: python tests/fixtures/finance/make_fixtures.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent
CIK = 1234567

# gold (ground truth), shared by evaluation and tests
TRUTH = {
    2023: dict(revenue=48_250_000_000, gross_profit=21_100_000_000, operating_income=9_870_000_000,
               net_income=7_425_000_000, eps_diluted=3.71, diluted_shares=2_001_000_000,
               operating_cash_flow=10_600_000_000, capex=2_150_000_000, rd_expense=4_120_000_000,
               total_assets=91_300_000_000, total_liabilities=55_200_000_000,
               stockholders_equity=36_100_000_000, cash_and_equivalents=14_800_000_000),
    2022: dict(revenue=44_600_000_000, gross_profit=19_050_000_000, operating_income=8_720_000_000,
               net_income=6_610_000_000, eps_diluted=3.28, diluted_shares=2_015_000_000,
               operating_cash_flow=9_400_000_000, capex=1_980_000_000, rd_expense=3_760_000_000,
               total_assets=85_700_000_000, total_liabilities=53_100_000_000,
               stockholders_equity=32_600_000_000, cash_and_equivalents=12_900_000_000),
}
CONCEPT = dict(
    revenue=("RevenueFromContractWithCustomerExcludingAssessedTax", "USD", "flow"),
    gross_profit=("GrossProfit", "USD", "flow"),
    operating_income=("OperatingIncomeLoss", "USD", "flow"),
    net_income=("NetIncomeLoss", "USD", "flow"),
    eps_diluted=("EarningsPerShareDiluted", "USD/shares", "flow"),
    diluted_shares=("WeightedAverageNumberOfDilutedSharesOutstanding", "shares", "flow"),
    operating_cash_flow=("NetCashProvidedByUsedInOperatingActivities", "USD", "flow"),
    capex=("PaymentsToAcquirePropertyPlantAndEquipment", "USD", "flow"),
    rd_expense=("ResearchAndDevelopmentExpense", "USD", "flow"),
    total_assets=("Assets", "USD", "instant"),
    total_liabilities=("Liabilities", "USD", "instant"),
    stockholders_equity=("StockholdersEquity", "USD", "instant"),
    cash_and_equivalents=("CashAndCashEquivalentsAtCarryingValue", "USD", "instant"),
)

ACCN_10K_22 = "0001234567-22-000011"
ACCN_10K_23 = "0001234567-23-000010"
ACCN_10KA_23 = "0001234567-24-000002"   # 10-K/A, value deliberately "wrong"
ACCN_10K_24 = "0001234567-24-000009"
ACCN_10Q_Q1_24 = "0001234567-24-000003"


def add(facts, metric, val, end, start, form, accn, filed, fy, fp):
    concept, unit, kind = CONCEPT[metric]
    row = {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed}
    if kind == "flow":
        row["start"] = start
    facts.setdefault(concept, {"units": {}})["units"].setdefault(unit, []).append(row)


def build_companyfacts():
    f = {}
    for m in CONCEPT:
        # FY2022 10-K: current period FY2022
        add(f, m, TRUTH[2022][m], "2022-09-30", "2021-10-01", "10-K", ACCN_10K_22, "2022-11-03", 2022, "FY")
        # FY2023 10-K: current FY2023 + prior comparative FY2022 (fy is 2023 for both)
        add(f, m, TRUTH[2023][m], "2023-09-30", "2022-10-01", "10-K", ACCN_10K_23, "2023-11-02", 2023, "FY")
        add(f, m, TRUTH[2022][m], "2022-09-30", "2021-10-01", "10-K", ACCN_10K_23, "2023-11-02", 2023, "FY")
        # FY2024 10-K repeats FY2023 (fy=2024)
        add(f, m, TRUTH[2023][m], "2023-09-30", "2022-10-01", "10-K", ACCN_10K_24, "2024-11-01", 2024, "FY")
    # 10-K/A: changes FY2023 revenue to a wrong value (must be excluded)
    add(f, "revenue", 99_999_000_000, "2023-09-30", "2022-10-01", "10-K/A", ACCN_10KA_23, "2024-02-01", 2023, "FY")
    # 10-Q Q1 FY2024: single quarter, 3 months (end 2023-12-31)
    add(f, "revenue", 12_300_000_000, "2023-12-31", "2023-10-01", "10-Q", ACCN_10Q_Q1_24, "2024-02-02", 2024, "Q1")
    add(f, "revenue", 11_500_000_000, "2022-12-31", "2022-10-01", "10-Q", ACCN_10Q_Q1_24, "2024-02-02", 2024, "Q1")
    # the same 10-Q also has a 9-month cumulative "bad" row (wrong duration, must be excluded)
    add(f, "revenue", 36_000_000_000, "2023-12-31", "2023-04-01", "10-Q", ACCN_10Q_Q1_24, "2024-02-02", 2024, "Q1")
    return {"cik": CIK, "entityName": "Acme Corp", "facts": {"us-gaap": f}}


def build_submissions():
    rows = [
        ("10-K", "2024-11-01", "2024-09-30", ACCN_10K_24, "acme-20240930.htm", "10-K"),
        ("10-Q", "2024-02-02", "2023-12-31", ACCN_10Q_Q1_24, "acme-20231231.htm", "10-Q"),
        ("8-K", "2023-11-02", "", "0001234567-23-000011", "acme-8k.htm", "Earnings release"),
        ("10-K", "2023-11-02", "2023-09-30", ACCN_10K_23, "acme-20230930.htm", "10-K"),
        ("10-K", "2022-11-03", "2022-09-30", ACCN_10K_22, "acme-20220930.htm", "10-K"),
    ]
    recent = {k: [] for k in ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument", "primaryDocDescription")}
    for form, filed, period, accn, doc, desc in rows:
        recent["form"].append(form); recent["filingDate"].append(filed); recent["reportDate"].append(period)
        recent["accessionNumber"].append(accn); recent["primaryDocument"].append(doc); recent["primaryDocDescription"].append(desc)
    return {"cik": str(CIK), "name": "Acme Corp", "filings": {"recent": recent, "files": []}}


HTML_10K = """<html><head><title>acme-20230930</title></head><body>
<ix:header><ix:hidden>HIDDEN_XBRL_NOISE 999999</ix:hidden></ix:header>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p>FORM 10-K  For the fiscal year ended September 30, 2023  Acme Corp</p>
<table>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>
<tr><td>Item 7.</td><td>Management's Discussion and Analysis of Financial Condition and Results of Operations</td><td>22</td></tr>
<tr><td>Item 8.</td><td>Financial Statements and Supplementary Data</td><td>31</td></tr>
</table>
<p>Item 1. Business</p>
<p>Acme Corp designs and sells industrial automation equipment and related software subscriptions to customers worldwide.
The Company's fiscal year ends on September 30. This section describes the segments, products, competition and employees of the Company in detail.</p>
<p>Item 1A. Risk Factors</p>
<p>The Company's business is subject to supply chain constraints, currency fluctuations and changes in customer capital spending. Any of these could adversely affect results of operations and financial condition. Investors should carefully consider the risks described below before making an investment decision in the Company's securities.</p>
<p>Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations</p>
<p>Total net sales were $48.25 billion for fiscal 2023, an increase of 8.2% compared to $44.60 billion in fiscal 2022. The increase was driven primarily by growth in software subscriptions and higher unit shipments of automation controllers.</p>
<p>Gross margin was 43.7% in fiscal 2023, compared to 42.7% in fiscal 2022. Operating income was $9.87 billion, and net income was $7.43 billion, or $3.71 per diluted share.</p>
<p>Research and development expense was $4.12 billion, and cash generated by operating activities was $10.60 billion. The Company believes its existing cash and cash equivalents of $14.80 billion will be sufficient to meet its liquidity needs for at least the next twelve months.</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<table>
<tr><td>(in millions, except per share data)</td><td>2023</td><td>2022</td></tr>
<tr><td>Net sales</td><td>$</td><td>48,250</td><td>$</td><td>44,600</td></tr>
<tr><td>Gross profit</td><td>21,100</td><td>19,050</td></tr>
<tr><td>Operating income</td><td>9,870</td><td>8,720</td></tr>
<tr><td>Net income</td><td>$</td><td>7,425</td><td>$</td><td>6,610</td></tr>
<tr><td>Diluted earnings per share</td><td>$</td><td>3.71</td><td>$</td><td>3.28</td></tr>
</table>
<p>The accompanying notes are an integral part of these consolidated financial statements and should be read together with them for a complete understanding of the Company's results and financial position at year end.</p>
</body></html>"""


def main():
    (OUT / "acme_companyfacts.json").write_text(json.dumps(build_companyfacts(), indent=1), encoding="utf-8")
    (OUT / "acme_submissions.json").write_text(json.dumps(build_submissions(), indent=1), encoding="utf-8")
    (OUT / "company_tickers.json").write_text(json.dumps({"0": {"cik_str": CIK, "ticker": "ACME", "title": "Acme Corp"},
                                                         "1": {"cik_str": 7654321, "ticker": "ACMX", "title": "Acme Exploration Holdings Inc"}}, indent=1), encoding="utf-8")
    (OUT / "acme_10k_fy2023.html").write_text(HTML_10K, encoding="utf-8")
    (OUT / "truth.json").write_text(json.dumps({str(k): v for k, v in TRUTH.items()}, indent=1), encoding="utf-8")
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main()
