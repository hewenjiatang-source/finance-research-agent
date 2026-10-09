# Finance Research Agent

**A Claude-powered multi-agent research system for SEC filings and earnings reports, with built-in citation verification and data-accuracy evaluation.**

Forked from [qiqihezh/deepresearch-agent](https://github.com/qiqihezh/deepresearch-agent) (MIT). The original Chinese README is kept as [README.zh-CN.md](README.zh-CN.md).

What this fork changes:

1. **LLM backend → Claude** (Anthropic Messages API), configurable per module.
2. **Scenario → financial-filings research**: SEC EDGAR tools, XBRL numbers, an evidence ledger, finance-specific prompts.
3. **Own evaluation** in `evaluation/finance/`: citation verification, data accuracy against XBRL ground truth, and a meta-evaluation that measures the evaluator itself.

---

## Quick start

```bash
git clone https://github.com/hewenjiatang-source/finance-research-agent.git
cd finance-research-agent
git checkout finance-research

pip install -r requirements.txt
cp .env.template .env      # set ANTHROPIC_API_KEY, SEC_USER_AGENT ("Your Name you@example.com"), search key

python scripts/run_finance.py "Compare Apple's FY2023 and FY2022 revenue, net income and diluted EPS, and summarize the main drivers"
# -> outputs/finance/report_*.md  and  report_*.evidence.json  (evidence sidecar used by the evaluation)
```

Useful flags: `--no-adversarial` skips the Red/Blue loop (cheaper). Config lives in `configs/finance.yaml` (extends `configs/default.yaml`).

## Architecture

The upstream engine is a hand-written asyncio orchestrator:

```
Query -> Planner (JSON DAG) -> Orchestrator (topological, concurrent dispatch)
      -> Researcher agents (tool loop) -> Summarizer -> Red/Blue adversarial loop -> report
```

Upstream modules: Orchestrator (9-state machine), Planner (replanning), Compressor (embedding-based context compression), shared Memory Store (SQLite), Red/Blue adversarial loop, and an evolution stub.

### What was added or changed

| Area | What |
|---|---|
| Claude backend | `src/models/claude_policy.py` maps the OpenAI-style message contract to the Claude Messages API: `tool_use`/`tool_result` pairing repair, thinking-block round-trip, retries, automatic streaming for large outputs. `configs/default.yaml` defaults to `backend: claude`; each module can pick its own model (e.g. haiku for the compressor, opus for the judge). |
| SEC tools | `src/tools/sec_edgar.py`: `sec_filings` (filing index), `sec_facts` (structured XBRL numbers with period and accession provenance), `sec_filing` (10-K/10-Q sections and keyword windows). Numbers come from structured data instead of the model "reading" HTML tables. |
| Evidence ledger | `src/finance/evidence.py`: every tool call registers its raw output and returns a stable `evidence_id`. Researchers, summarizer, Red/Blue and the evaluation all use the same `[n]`. |
| Finance agents/prompts | `src/finance/{agents,prompts}.py`: no filling gaps from memory, every figure carries period and basis, derived numbers go through the calculator, "not found in retrieved sources" instead of guessing. |
| Framework fix | The upstream Red/Blue loop only reviewed the first 4,000 characters and Blue overwrote the full report with a truncated rewrite. Review length is now configurable and fixed text is spliced back with the unreviewed tail. |
| XBRL pitfalls handled | `fy/fp` describe the filing, not the value's period; 10-K/A exclusion; cumulative 6M/9M rows in 10-Qs; duration filters; instant vs. flow metrics. |

## Evaluation (`evaluation/finance/`)

```bash
python scripts/run_finance_eval.py --meta-only                       # evaluator self-check, fully offline
python scripts/build_finance_cases.py                                # build cases + XBRL gold from SEC (needs network)
python scripts/run_finance_eval.py --cases evaluation/finance/cases/real_cases.json --run --reports outputs/finance_eval
python scripts/run_finance_eval.py --cases ... --reports outputs/finance_eval   # evaluate existing reports offline
```

1. **Citation verification**: dangling citation ids; cited source does not contain the number (`misattributed` / `unsupported`); uncited numbers that no evidence supports (likely from memory). Attribution is done per clause. Bare numbers only match evidence that declares its unit ("in millions"), so ×1000 scale errors are not hidden.
2. **Data accuracy**: numbers are normalized (million/billion/亿) and compared against SEC XBRL gold **at the precision the report wrote** (48.3 billion is a valid rounding of 48.25 billion). Errors are classified as `scale_error`, `wrong_period`, `wrong_metric`, `sign_error`, `imprecise`, `wrong_value`. Growth rates and margins are recomputed from gold.
3. **Meta-evaluation**: known errors (9 types) are injected into clean synthetic reports built from gold; the evaluator's detection rate, error-type accuracy and false-positive rate are reported.
4. **Optional LLM judge** (`--judge`): checks entailment of non-numeric claims only; cached; kept out of the hard metrics because of self-preference bias (use a different/stronger model than the one under test).

## Real-run results (Apple, Microsoft, JPMorgan — latest 10-K, Claude as the model)

One run per company (FY2024 10-K; `outputs/real_eval/`), evaluated with `evaluation/finance/`. **Small sample (3 reports, 197 numeric claims); read the numbers as a smoke test, not a benchmark.**

| | Apple | Microsoft | JPMorgan | Total |
|---|---|---|---|---|
| Numeric claims | 75 | 79 | 43 | 197 |
| Cited, and the cited source contains the number (`supported`) | 37 | 74 | 39 | **150 / 156 cited = 96.2%** |
| Cited but source lacks it (`misattributed` + `unsupported`) | 1 | 3 | 2 | 6 |
| Share of numbers carrying a citation | 51% | 97% | 95% | 79.2% |
| Uncited, not in any evidence (`uncited_ungrounded`) | 1 | 0 | 0 | 1 of 41 uncited = 2.4% |

**Data accuracy.** 70 numbers were mapped automatically to an XBRL gold value (revenue, net income, EPS, gross/operating profit, cash flow, total assets ...); **all 70 match the filing at the precision the report wrote**. The other 127 are not auto-checked: segment figures, non-GAAP numbers, percentages derived by the agent, and numbers whose (metric, period) the evaluator will not guess. I also checked 12 headline figures by hand against the 10-K: Apple net sales 391,035 / net income 93,736 / operating income 123,216 / gross profit 180,683 / diluted EPS 6.08 ($M); Microsoft revenue 245,122 / operating income 109,433 / net income 88,136; JPMorgan revenue 177,556 / net income 58,471 / diluted EPS 19.75 / total assets 4,002,814 — all correct.

**What went wrong** (this is the useful part):
- *The evaluator was the weakest link, not the agent.* The first real-data pass reported 42.6% accuracy; every "error" I traced was an evaluator mapping bug (wrong period for "A in FY2024, against B in FY2023", a prior-year figure bound to the current year, a Unicode minus sign, segment revenue mapped to total revenue, percent changes mapped to the wrong base). They are fixed, with regression tests built from the real sentences, and the evaluator now prefers "unmapped" to guessing. The 70/70 above is therefore partly a statement about what the evaluator is willing to judge.
- *Tool bug found by the real run:* the XBRL evidence text rounded per-share values to an integer (`value 20 USD/shares (19.75 ...)`), which can make a faithful EPS look unsupported. Fixed in `src/finance/evidence.py`.
- *Citation errors (6):* the agent cited an XBRL item for a growth rate it had computed itself, and cited `[1]` (an FY2025 filing index) for FY2024 figures. Derived values (growth rates, margins computed with the calculator) usually have no evidence id of their own, so they cannot be traced to a source.
- *Agent behavior:* with no search key the web-search tool failed and the agent fell back to SEC tools only; some calculator calls failed; there is no independent second source for a figure, and the Red/Blue adversarial loop was off.

## Known limitations

- Three large US filers, one run each — no variance estimate, no small-cap, no restatement or non-calendar-year edge cases beyond Apple/Microsoft fiscal years. HK/A-share filings go through web search and have no structured gold.
- Number → (metric, period) mapping is heuristic. MD&A segment data, non-GAAP measures, guidance and dates are not auto-checked; the share of mapped numbers (35–60% depending on how much of the XBRL gold is available) is the real coverage limit of the accuracy metric.
- The gold is built with the same `select_period` code as the agent's tool, so it proves the agent copied structured data faithfully, not that the extraction logic is right; a hand-checked `truth.json` regression test covers that separately.
- Meta-evaluation uses synthetic reports with regular phrasing, so its detection rates are an upper bound (e.g. "dropped citation" detection is ~50% because comma-joined clauses can share a citation).
- The optional LLM judge (`--judge`) was not exercised on real data.

## Test status

`python -m unittest tests.test_claude_policy tests.test_finance_tools tests.test_finance_e2e tests.test_finance_eval` (78 tests, offline): Claude message/tool conversion, XBRL period selection, SEC tools, evidence ledger, the whole pipeline with a scripted fake model and synthetic SEC fixtures (fictional "Acme Corp"), every evaluation layer, the meta-evaluation (zero false positives on clean reports, mapping rate ≥ 0.9), and regression tests from real report sentences.

## Repository layout (new parts)

```
src/models/claude_policy.py     Claude backend
src/tools/sec_edgar.py          SEC EDGAR tools
src/finance/                    XBRL registry, evidence ledger, finance agents and prompts
configs/finance.yaml            finance scenario config
evaluation/finance/             numbers, claims, citations, accuracy, judge, perturb (meta-eval), suite
scripts/run_finance.py          run one query
scripts/run_finance_eval.py     evaluate / meta-evaluate
scripts/build_finance_cases.py  build cases with XBRL gold
tests/                          offline tests + synthetic fixtures
```

## License

MIT. Original work © its contributors; see [README.zh-CN.md](README.zh-CN.md) for the upstream project description.
