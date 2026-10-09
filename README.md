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

## Verification status (please read)

- **Verified offline** (`python -m unittest tests.test_claude_policy tests.test_finance_tools tests.test_finance_e2e tests.test_finance_eval`, 73 tests): Claude message/tool conversion, XBRL period selection, SEC tools, evidence ledger, the whole pipeline with a scripted fake model and synthetic SEC fixtures, every evaluation layer, and the meta-evaluation.
- **Not verified**: real Anthropic API calls, live EDGAR access, LLM-judge quality. The development environment had no API key and no access to data.sec.gov. The fixture company "Acme Corp" is fictional. Spot-check one or two real companies by hand before trusting aggregate numbers.
- **Meta-evaluation caveats**: injected errors are synthetic and the report phrasing is regular, so detection rates are an upper bound. On "dropped citation" detection is only ~50% because comma-joined clauses can share one citation.
- The number → (metric, period) mapping is heuristic; unmappable numbers are counted as "unmapped", never as "correct". MD&A segment data, non-GAAP measures, guidance and dates are not auto-checked yet. HK/A-share filings go through web search and have no structured gold.
- The gold is built with the same `select_period` code as the agent's tool, so it proves the agent copied structured data faithfully, not that the extraction logic is right; that is covered separately by a hand-checked `truth.json` regression test.

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
