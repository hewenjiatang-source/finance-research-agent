#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/core/runner.py
================================================================================
Core run logic of the DeepResearch Agent.

This module holds the core functions that initialize all modules and run the complete research flow,
called uniformly by scripts/ and evaluation/, so that evaluation/ does not depend back on scripts/.

Public interface:
    - load_config(config_path) -> dict
    - initialize_modules(config) -> dict
    - run_research(query, config, modules) -> str
    - save_report(report, query, output_dir) -> str
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# add the project root to sys.path so the src package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO") -> None:
    """Configure the global log format and level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """
    Load a YAML configuration file.

    If no path is given, configs/default.yaml is loaded by default.
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # supports `extends: default.yaml`: load the parent config first, then recursively merge this file (dicts deep-merge, everything else overrides)
    parent = config.pop("extends", None)
    if parent:
        parent_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(config_path), parent)
        config = _deep_merge(load_config(parent_path), config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def is_finance(config: dict) -> bool:
    return (config.get("domain") or "").lower() == "finance"


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------
def _create_tools_factory(config: dict, ledger=None):
    """Create the tool factory, returning the list of tools available to Agents."""
    tools_cfg = config.get("tools", {})
    mock_mode = tools_cfg.get("web_search", {}).get("mock_mode", True)

    from src.tools import (
        WebSearchTool,
        MockWebSearchTool,
        ArxivReaderTool,
        BrowserTool,
        MockBrowserTool,
        FileReaderTool,
        CodeSandboxTool,
        CalculatorTool,
        NotepadTool,
    )

    tools = {}

    # 1. web_search
    if mock_mode:
        tools["web_search"] = MockWebSearchTool()
    else:
        tools["web_search"] = WebSearchTool()

    # 2. browser
    if mock_mode:
        tools["browser"] = MockBrowserTool()
    else:
        tools["browser"] = BrowserTool()

    # 3. arxiv_reader
    tools["arxiv_reader"] = ArxivReaderTool(use_mock=mock_mode)

    # 4. file_reader (no directory restriction)
    tools["file_reader"] = FileReaderTool(allowed_base_dir=None)

    # 5. code_sandbox
    tools["code_sandbox"] = CodeSandboxTool(use_mock=mock_mode)

    # 6. calculator
    tools["calculator"] = CalculatorTool()

    # 7. notepad
    tools["notepad"] = NotepadTool()

    # finance scenario: drop the academic-paper tool, add the SEC tools, and wrap everything with the evidence ledger
    if is_finance(config):
        from src.finance.evidence import LedgerTool
        from src.tools.sec_edgar import SecClient, create_sec_tools

        tools.pop("arxiv_reader", None)
        fin_cfg = config.get("finance", {})
        client = SecClient(user_agent=fin_cfg.get("sec_user_agent") or None)
        for t in create_sec_tools(client):
            tools[t.name] = t
        if ledger is not None:
            tools = {k: LedgerTool(v, ledger) for k, v in tools.items()}

    # return a list (AgentPool and the Agent constructors need a list)
    return list(tools.values())


# ---------------------------------------------------------------------------
# Module initialization
# ---------------------------------------------------------------------------
def initialize_modules(config: dict, session_id: str = "") -> dict[str, Any]:
    """
    Initialize all core modules from the configuration.

    Args:
        config: global configuration dict.
        session_id: session id, used for session isolation in the memory store.

    Returns a dict containing the module instances.
    """
    logger = logging.getLogger("runner")
    logger.info("Initializing core modules...")

    modules: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Multi-backend LLM initialization (reads config from .env + configs/default.yaml)
    # ------------------------------------------------------------------
    from src.models.model_router import ModelRouter

    model_cfg = config.get("model", {})
    default_backend = model_cfg.get("backend", "vllm")
    backend_mapping = model_cfg.get("backend_mapping", {})
    backend_sampling = model_cfg.get("backend_sampling", {})

    # helper: get the sampling-parameter overrides by module name
    def _get_sampling_kwargs(module_name: str, backend_name: str) -> dict:
        """Merge backend-wide defaults + module-level overrides."""
        kwargs = {}
        # 1. backend-wide defaults
        if backend_name in backend_sampling:
            kwargs.update(backend_sampling[backend_name])
        # 2. module-level overrides (higher priority)
        module_overrides = backend_sampling.get("modules", {}).get(module_name, {})
        kwargs.update(module_overrides)
        return kwargs

    # default backend (shared by all modules)
    default_kwargs = _get_sampling_kwargs("default", default_backend)
    default_policy = ModelRouter.create_backend(default_backend, **default_kwargs)
    modules["default_policy"] = default_policy
    logger.info(f"[LLM] Default backend loaded: {default_backend} ({default_kwargs})")

    # multi-backend division of labor: different modules use different backends + different sampling parameters
    for module_name, backend_name in backend_mapping.items():
        kwargs = _get_sampling_kwargs(module_name, backend_name)
        modules[f"{module_name}_policy"] = ModelRouter.create_backend(backend_name, **kwargs)
        logger.info(f"[LLM] {module_name} → backend={backend_name}, sampling={kwargs}")

    # if no division of labor is configured, every module falls back to default_policy
    # ------------------------------------------------------------------

    # M2: Adaptive Planner (the Orchestrator depends on the Planner, so initialize it first)
    from src.planner.planner import Planner
    from src.planner.budget_tracker import BudgetTracker

    planner_policy = modules.get("planner_policy", default_policy)
    budget_tracker = BudgetTracker()
    finance = is_finance(config)
    fin_cfg = config.get("finance", {}) if finance else {}
    lang = fin_cfg.get("language", "zh")
    ledger = None
    if finance:
        from src.finance.agents import FinancePlanner
        from src.finance.evidence import EvidenceLedger

        ledger = EvidenceLedger()
        modules["ledger"] = ledger
        planner = FinancePlanner(
            policy=planner_policy, budget_tracker=budget_tracker,
            max_sub_tasks=config.get("orchestrator", {}).get("max_sub_questions", 8),
        )
    else:
        planner = Planner(policy=planner_policy, budget_tracker=budget_tracker)
    modules["planner"] = planner
    logger.info("[M2] Planner module initialized")

    # M3: Context Compressor
    from src.compressor.compressor import ContextCompressor

    compressor_policy = modules.get("compressor_policy", default_policy)
    compressor_cfg = config.get("compressor", {})
    compressor = ContextCompressor(
        llm_policy=compressor_policy,
        budget=compressor_cfg.get("max_context_length", 16000),
        output_reserve=compressor_cfg.get("output_reserve_tokens", 2048),
    )
    modules["compressor"] = compressor
    logger.info("[M3] Compressor module initialized")

    # M4: Shared Memory Store
    from src.memory.memory_store import SharedMemoryStore

    memory_cfg = config.get("memory", {})
    memory_store = SharedMemoryStore(
        db_path=memory_cfg.get("db_path", "data/memory.db"),
        session_id=session_id,
    )
    modules["memory_store"] = memory_store
    logger.info(f"[M4] Memory Store module initialized (session={session_id})")

    # Tools (real tools or Mock tools)
    tools_list = _create_tools_factory(config, ledger)
    modules["tools"] = tools_list
    logger.info(f"Tools module initialized ({len(tools_list)} tools)")

    # M5: Red-Blue Adversarial Loop (create it first, then inject it into the Orchestrator)
    from src.adversarial.loop import AdversarialLoop
    from src.adversarial.red_agent import RedAgent
    from src.adversarial.blue_agent import BlueAgent

    red_policy = modules.get("red_agent_policy", default_policy)
    blue_policy = modules.get("blue_agent_policy", default_policy)
    adversarial_cfg = config.get("adversarial", {})

    if finance:
        from src.finance.prompts import BLUE_FINANCE_EXTRA, RED_FINANCE_EXTRA

        review = dict(max_report_chars=adversarial_cfg.get("max_report_chars", 60000),
                      max_sources=adversarial_cfg.get("max_sources", 60))
        red_agent = RedAgent(policy=red_policy, max_tokens=4096, extra_system=RED_FINANCE_EXTRA, **review)
        blue_agent = BlueAgent(policy=blue_policy, tools=tools_list, max_tokens=16384,
                               extra_system=BLUE_FINANCE_EXTRA, **review)
    else:
        red_agent = RedAgent(policy=red_policy)
        blue_agent = BlueAgent(policy=blue_policy, tools=tools_list)
    adversarial_loop = AdversarialLoop(
        red_agent=red_agent,
        blue_agent=blue_agent,
        policy=modules.get("judge_policy", default_policy),
        max_rounds=adversarial_cfg.get("max_rounds", 3),
        score_threshold=adversarial_cfg.get("score_threshold", 8.0),
        delta_threshold=adversarial_cfg.get("delta_threshold", 0.3),
    )
    modules["adversarial"] = adversarial_loop
    logger.info("[M5] Adversarial module initialized")

    # M1: Multi-Agent Orchestrator
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.agent_pool import AgentPool

    researcher_cls = None
    summarizer_factory = None
    if finance:
        import functools

        from src.finance.agents import FinanceResearcherAgent, FinanceSummarizerAgent

        researcher_cls = functools.partial(
            FinanceResearcherAgent, language=lang, max_tool_calls=fin_cfg.get("max_tool_calls", 8),
            max_turns=fin_cfg.get("researcher_max_turns", 12),
        )
        summarizer_factory = lambda policy, tools: FinanceSummarizerAgent(  # noqa: E731
            name="summarizer", policy=policy, tools=tools, ledger=ledger, language=lang
        )
    agent_pool = AgentPool(
        policy_factory=lambda: modules.get("solver_policy", default_policy),
        tools_factory=lambda: list(modules["tools"]),
        max_idle=3,
        researcher_cls=researcher_cls,
    )
    modules["agent_pool"] = agent_pool

    orchestrator = Orchestrator(
        planner=planner,
        agent_pool=agent_pool,
        budget_tracker=budget_tracker,
        compressor=compressor,
        adversarial_loop=adversarial_loop,
        memory_store=memory_store,
        summarizer_policy=modules.get("summarizer_policy", default_policy),
        summarizer_factory=summarizer_factory,
    )
    modules["orchestrator"] = orchestrator
    logger.info("[M1] Orchestrator module initialized")

    # M6: Self-Evolution Engine (reserved, disabled by default)
    if config.get("evolution", {}).get("enabled", False):
        logger.info("[M6] Evolution module enabled (reserved interface)")
    else:
        logger.info("[M6] Evolution module disabled")

    return modules


# ---------------------------------------------------------------------------
# Main research flow
# ---------------------------------------------------------------------------
async def run_research_full(query: str, config: dict, modules: dict[str, Any]):
    """
    Run the complete research flow.

    Flow:
        1. the Orchestrator calls the Planner to decompose the question into a sub-task DAG
        2. the Orchestrator schedules the sub-agents in the AgentPool, in parallel / serially
        3. sub-agents call Tools to retrieve information and produce sub-reports
        4. the Compressor manages long context
        5. Memory stores intermediate results
        6. the Adversarial Loop optimizes the report over several adversarial rounds (if enabled)
        7. output the final research report

    Args:
        query: the research question entered by the user.
        config: global configuration dict.
        modules: dict of initialized module instances.

    Returns:
        (final research report as Markdown text, ResearchReport). In the finance scenario ResearchReport.evidence is a snapshot of the evidence ledger.
    """
    import asyncio

    logger = logging.getLogger("runner")
    logger.info(f"Starting research, query: {query[:80]}...")

    start_time = time.time()

    ledger = modules.get("ledger")
    if ledger is not None:
        ledger.reset()  # the ledger is per run; running several queries concurrently on the same modules is not supported

    # Step 1-3: the Orchestrator does planning, scheduling, collection and synthesis internally
    orchestrator = modules["orchestrator"]
    from src.orchestrator.schemas import RunConfig

    run_cfg = RunConfig(
        max_concurrent=config.get("orchestrator", {}).get("max_concurrent", 5),
        global_timeout_seconds=config.get("orchestrator", {}).get("global_timeout_seconds", 600),
        max_replan_rounds=config.get("orchestrator", {}).get("max_replan_rounds", 3),
        max_sub_questions=config.get("orchestrator", {}).get("max_sub_questions", 8),
        enable_adversarial=config.get("adversarial", {}).get("enabled", True),
        enable_evolution=config.get("evolution", {}).get("enabled", False),
    )

    report = await orchestrator.run(query, config=run_cfg)
    logger.info(
        f"[Orchestrator] Report generated | confidence={report.confidence:.2f} | "
        f"searches={report.num_searches} | replans={report.num_replan} | adversarial rounds={report.adversarial_rounds}"
    )

    if ledger is not None:  # Blue may retrieve new evidence inside the adversarial loop; the final ledger is authoritative
        report.evidence = ledger.to_list()
        report.sources = ledger.to_sources()

    # Step 4/5: evolution optimization (if enabled and trained)
    if run_cfg.enable_evolution:
        logger.info("[Evolution] Evolution optimization enabled (reserved interface)")
    else:
        logger.info("[Evolution] Evolution optimization skipped")

    # close the WebSearchTool connection pool
    from src.tools.web_search import WebSearchTool
    await WebSearchTool.close_session()

    elapsed = time.time() - start_time
    logger.info(f"Research finished, elapsed: {elapsed:.2f} s")

    # assemble the final output
    final_report = _format_report(report, elapsed, config.get("finance", {}).get("language", "zh") if is_finance(config) else "zh")
    return final_report, report


async def run_research(query: str, config: dict, modules: dict[str, Any]) -> str:
    """Backward-compatible interface: returns only the Markdown text. Use run_research_full to get the evidence ledger."""
    text, _ = await run_research_full(query, config, modules)
    return text


_LABELS = {
    "zh": dict(title="研究报告：", meta="元信息", conf="置信度", searches="搜索轮数", replans="重规划次数",
               rounds="对抗轮数", elapsed="总耗时", secs="秒", refs="参考来源", unk="未知标题"),
    "en": dict(title="Research report: ", meta="Metadata", conf="Confidence", searches="Search calls",
               replans="Replans", rounds="Adversarial rounds", elapsed="Elapsed", secs="s", refs="References",
               unk="Untitled"),
}


def _format_report(report, elapsed: float, lang: str = "zh") -> str:
    """Format a ResearchReport as Markdown text (lang: zh | en; in the finance scenario finance.language decides)."""
    L = _LABELS.get(lang, _LABELS["zh"])
    content = report.content or ""

    # unify confidence: if the body has the LLM's self-rated "overall confidence", replace it with the actually computed value, to avoid inconsistency
    content = re.sub(
        r"(整体置信度|Overall Confidence|置信度)[:：]\s*0?\.\d+",
        f"\\1: {report.confidence:.2f}",
        content,
        flags=re.I,
    )

    lines = [
        f"# {L['title']}{report.query}",
        "",
        "---",
        "",
        content,
        "",
        "---",
        "",
        f"## {L['meta']}",
        "",
        f"- **{L['conf']}**: {report.confidence:.2f}",
        f"- **{L['searches']}**: {report.num_searches}",
        f"- **{L['replans']}**: {report.num_replan}",
        f"- **{L['rounds']}**: {report.adversarial_rounds}",
        f"- **{L['elapsed']}**: {elapsed:.2f} {L['secs']}",
        "",
    ]

    if report.sources:
        lines.append(f"## {L['refs']}")
        lines.append("")
        for i, src in enumerate(report.sources, 1):
            title = src.get("title", L["unk"])
            url = src.get("url", "")
            snippet = " ".join(src.get("snippet", "").split())[:160]
            if "id" in src:  # finance scenario: number == evidence_id, matching the [n] in the body
                lines.append(f"[{src['id']}] [{title}]({url}) — {snippet}")
            else:
                lines.append(f"{i}. [{title}]({url}) — {snippet}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------
def save_report(report: str, query: str, output_dir: str = "outputs/reports", evidence: list[dict] | None = None) -> str:
    """
    Save the research report to a file.

    File name format: report_YYYYMMDD_HHMMSS_<first 20 chars of query>.md
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() or c in "_-" else "_" for c in query[:20])
    filename = f"report_{timestamp}_{safe_query}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    if evidence:  # evidence sidecar file: lets the evaluation replay citation verification / data accuracy offline
        from src.finance.evidence import EVIDENCE_SCHEMA
        import json

        with open(filepath[:-3] + ".evidence.json", "w", encoding="utf-8") as f:
            json.dump({"schema": EVIDENCE_SCHEMA, "query": query, "evidence": evidence}, f, ensure_ascii=False, indent=1)

    return filepath
