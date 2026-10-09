#!/usr/bin/env python3
"""
Continuous integration test script - automated end-to-end validation of the DeepResearch Agent

Test goals:
  1. All tools (web_search, arxiv_reader, calculator, file_reader, etc.) are actually invoked
  2. No fake/empty report is produced when subtasks fail
  3. The report is based on search results rather than pure LLM fabrication
  4. Each module (Planner/Compressor/Memory/Adversarial) works as expected
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Force unbuffered output so logs are written in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Test case design: each query focuses on a different tool combination and capability
# ---------------------------------------------------------------------------
TEST_QUERIES: list[dict] = [
    {
        "id": "T1",
        "name": "Combined search + paper retrieval",
        "query": (
            "2024年至2025年大模型Agent技术方向与后端系统开发方向的对比研究："
            "检索两个领域的最新学术论文发表数量与趋势变化、"
            "工业界代表性落地案例与市场规模数据"
        ),
        "expected_tools": ["web_search", "arxiv_reader"],
        "min_success_rate": 0.5,
        "min_report_length": 1500,
    },
    {
        "id": "T2",
        "name": "Search + calculator tool",
        "query": (
            "计算Transformer模型在7B、13B、70B三种参数量下的训练FLOPs和推理显存占用，"
            "并检索2024年至2025年主流大模型（如GPT-4、Claude 3、Gemini、DeepSeek）的实际部署成本和推理延迟数据"
        ),
        "expected_tools": ["web_search", "calculator"],
        "min_success_rate": 0.4,
        "min_report_length": 1500,
    },
    {
        "id": "T3",
        "name": "Deep search + adversarial refinement",
        "query": (
            "中国新能源汽车行业2024年至2025年的市场份额变化、主要品牌销量排名、"
            "电池技术路线（磷酸铁锂vs三元锂）的技术对比与成本分析"
        ),
        "expected_tools": ["web_search"],
        "min_success_rate": 0.5,
        "min_report_length": 2000,
    },
    {
        "id": "T4",
        "name": "Dedicated paper retrieval",
        "query": (
            "检索近一年（2024年至2025年）关于RLHF（基于人类反馈的强化学习）的顶级会议论文，"
            "统计NeurIPS、ICML、ICLR各会议收录数量，并分析该领域的技术演进趋势和核心作者"
        ),
        "expected_tools": ["arxiv_reader", "web_search"],
        "min_success_rate": 0.4,
        "min_report_length": 1500,
    },
    {
        "id": "T5",
        "name": "Cross-domain trend analysis",
        "query": (
            "对比分析2024年至2025年生成式AI在医疗健康领域和金融投资领域的应用进展："
            "检索各领域的代表性产品、监管政策变化、以及商业化落地案例"
        ),
        "expected_tools": ["web_search", "arxiv_reader"],
        "min_success_rate": 0.5,
        "min_report_length": 2000,
    },
]


def clean_env():
    """Clean up the database and old logs."""
    db_path = PROJECT_ROOT / "data" / "memory.db"
    if db_path.exists():
        db_path.unlink()
    for f in (PROJECT_ROOT / "outputs" / "reports").glob("report_*.md"):
        f.unlink()
    print("[Test] Environment cleaned up")


def run_single_test(test_case: dict) -> dict:
    """Run a single test and return the analysis result."""
    test_id = test_case["id"]
    query = test_case["query"]
    log_file = PROJECT_ROOT / "outputs" / f"test_{test_id}.log"

    print(f"\n{'='*60}")
    print(f"[Test {test_id}] {test_case['name']}")
    print(f"Query: {query[:60]}...")
    print(f"{'='*60}")

    clean_env()

    start = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_research.py"),
            "--query", query,
            "--output_dir", str(PROJECT_ROOT / "outputs" / "reports" / test_id),
        ],
        capture_output=True,
        text=True,
        timeout=900,  # 15-minute timeout
    )
    elapsed = time.time() - start

    # Save log
    log_file.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    # Extract key metrics
    result = analyze_log(proc.stdout, test_case)
    result["elapsed"] = elapsed
    result["returncode"] = proc.returncode

    # Check report file
    report_dir = PROJECT_ROOT / "outputs" / "reports" / test_id
    reports = list(report_dir.glob("*.md")) if report_dir.exists() else []
    if reports:
        report_text = reports[0].read_text(encoding="utf-8")
        result["report_path"] = str(reports[0])
        result["report_length"] = len(report_text)
        result["report_has_content"] = len(report_text) > test_case["min_report_length"]
        result["is_empty_md"] = "Research failed" in report_text or result["report_length"] < 500
    else:
        result["report_path"] = None
        result["report_length"] = 0
        result["report_has_content"] = False
        result["is_empty_md"] = True

    return result


def analyze_log(stdout: str, test_case: dict) -> dict:
    """Extract key metrics from stdout."""
    metrics = {
        "subtask_total": 0,
        "subtask_success": 0,
        "subtask_failed": 0,
        "success_rate": 0.0,
        "num_searches": 0,
        "num_replan": 0,
        "adversarial_rounds": 0,
        "confidence": 0.0,
        "has_bogus_output": False,
        "issues": [],
    }

    # Subtask success rate
    m = re.search(r"子任务完成:\s*(\d+)/(\d+)\s*成功\s*\((\d+)\s*失败\)", stdout)
    if m:
        metrics["subtask_success"] = int(m.group(1))
        metrics["subtask_total"] = int(m.group(2))
        metrics["subtask_failed"] = int(m.group(3))
        if metrics["subtask_total"] > 0:
            metrics["success_rate"] = metrics["subtask_success"] / metrics["subtask_total"]

    # Meta information
    m = re.search(r"置信度=(\d+\.?\d*)", stdout)
    if m:
        metrics["confidence"] = float(m.group(1))

    m = re.search(r"搜索轮数=(\d+)", stdout)
    if m:
        metrics["num_searches"] = int(m.group(1))

    m = re.search(r"重规划=(\d+)", stdout)
    if m:
        metrics["num_replan"] = int(m.group(1))

    m = re.search(r"对抗轮数=(\d+)", stdout)
    if m:
        metrics["adversarial_rounds"] = int(m.group(1))

    # Bug detection
    if metrics["success_rate"] == 0.0 and metrics["num_searches"] == 0:
        metrics["has_bogus_output"] = True
        metrics["issues"].append("All subtasks failed and search rounds is 0, but a report may have been produced")

    if "Research failed" in stdout and metrics["report_length"] == 0:
        metrics["issues"].append("Report marked as failed and empty - this is the correct behavior")

    # Check whether the expected tools were used (judged indirectly via search rounds)
    if metrics["num_searches"] == 0 and "web_search" in test_case["expected_tools"]:
        metrics["issues"].append("Expected web_search to be used, but search rounds is 0")

    return metrics


def print_summary(results: list[dict]):
    """Print the test summary."""
    print("\n" + "=" * 70)
    print("Test summary")
    print("=" * 70)

    total_pass = 0
    for r in results:
        tc = r["test_case"]
        passed = (
            r["success_rate"] >= tc["min_success_rate"]
            and r.get("report_has_content", False)
            and not r.get("is_empty_md", True)
            and not r["has_bogus_output"]
        )
        total_pass += int(passed)

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n[{tc['id']}] {tc['name']} — {status}")
        print(f"  Subtasks: {r['subtask_success']}/{r['subtask_total']} succeeded ({r['success_rate']:.0%})")
        print(f"  Search rounds: {r['num_searches']} | Replans: {r['num_replan']} | Adversarial: {r['adversarial_rounds']}")
        print(f"  Confidence: {r['confidence']:.2f} | Report length: {r.get('report_length', 0)} chars")
        print(f"  Elapsed: {r['elapsed']:.1f}s")
        if r["issues"]:
            print(f"  Issues: {'; '.join(r['issues'])}")

    print(f"\nTotal: {total_pass}/{len(results)} passed")


def main():
    results = []
    for tc in TEST_QUERIES:
        try:
            r = run_single_test(tc)
            r["test_case"] = tc
            results.append(r)
        except subprocess.TimeoutExpired:
            print(f"[Test {tc['id']}] Timeout (15 minutes)")
            results.append({
                "test_case": tc,
                "success_rate": 0,
                "has_bogus_output": True,
                "issues": ["Timeout"],
                "elapsed": 900,
            })
        except Exception as e:
            print(f"[Test {tc['id']}] Exception: {e}")
            results.append({
                "test_case": tc,
                "success_rate": 0,
                "has_bogus_output": True,
                "issues": [f"Exception: {e}"],
                "elapsed": 0,
            })

    print_summary(results)

    # Save detailed results
    summary_path = PROJECT_ROOT / "outputs" / "test_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nDetailed results saved: {summary_path}")


if __name__ == "__main__":
    main()
