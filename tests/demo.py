#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — DeepResearch Agent tool layer demo script
================================================================================

Purpose:
  1. Verify in one step that all 7 tools work
  2. Show how the tools collaborate across a multi-round research flow
  3. Serve as an interview demo: interviewers can run this script directly to see the results

Run modes:
  - Real mode (default): call real APIs / run locally
  - Mock mode (debugging): no API key needed, uses preset data

How to switch: modify the USE_MOCK variable below

Usage:
    python demo.py                    # Real mode (default)
    # Or modify USE_MOCK = False and run   # Mock mode
================================================================================
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Add the project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Run mode switch
# =============================================================================
# Set to True to use Mock mode (no API key needed, for quick demos)
# Set to False to use real mode (requires the corresponding API keys)
USE_MOCK = False

# =============================================================================
# Quick Mock mode switch (keep False for normal runs, set True for debugging / offline environments)
# =============================================================================
# USE_MOCK = False   # <-- uncomment this line to switch to Mock mode (no API key needed)


# =============================================================================
# Tool factory: create real or Mock tools according to USE_MOCK
# =============================================================================
def create_tools(mock_mode: bool = USE_MOCK):
    """Create tool instances.

    Args:
        mock_mode: True uses Mock tools, False uses real tools.

    Returns:
        dict: Mapping from tool name to instance.
    """
    from src.tools import (
        WebSearchTool, MockWebSearchTool,
        ArxivReaderTool, CodeSandboxTool,
        BrowserTool, MockBrowserTool,
        FileReaderTool, CalculatorTool, NotepadTool,
    )

    tools = {}

    # 1. web_search
    if mock_mode:
        tools["web_search"] = MockWebSearchTool()
    else:
        # Real mode: WebSearchTool automatically reads SERPAPI_KEY from .env / .env.local
        tools["web_search"] = WebSearchTool()

    # 2. browser
    if mock_mode:
        tools["browser"] = MockBrowserTool()
    else:
        tools["browser"] = BrowserTool()

    # 3. arxiv_reader(mock mode controlled via the use_mock parameter)
    tools["arxiv_reader"] = ArxivReaderTool(use_mock=mock_mode)

    # 4. file_reader(always runs locally, directory is unrestricted in the demo)
    tools["file_reader"] = FileReaderTool(allowed_base_dir=None)

    # 5. code_sandbox(mock mode controlled via the use_mock parameter)
    tools["code_sandbox"] = CodeSandboxTool(use_mock=mock_mode)

    # 6. calculator(always runs locally)
    tools["calculator"] = CalculatorTool()

    # 7. notepad(always runs locally)
    tools["notepad"] = NotepadTool()

    return tools


# =============================================================================
# Tool schema printing
# =============================================================================
def print_tool_schemas(tools: dict) -> None:
    """Print the OpenAI Function Calling Schema of all tools."""
    print("\n" + "=" * 70)
    print("[Tool Schema List] ({} tools in total)".format(len(tools)))
    print("=" * 70)

    for name, tool in tools.items():
        schema = tool.get_openai_tool_schema()
        func = schema["function"]
        print(f"\n🔧 {func['name']}")
        print(f"   Description: {func['description'][:80]}...")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        req = params.get("required", [])
        for pname, pdef in props.items():
            marker = "*" if pname in req else " "
            desc = pdef.get("description", "")[:50]
            print(f"   {marker} {pname}: {desc}")


# =============================================================================
# Individual tool functionality tests
# =============================================================================
async def test_tools_individually(tools: dict) -> list[str]:
    """Test the core functionality of each tool one by one."""
    errors: list[str] = []

    print("\n" + "=" * 70)
    print("[Single Tool Functionality Tests]")
    print("=" * 70)

    # 1. web_search
    print("\n📡 web_search: searching 'transformer architecture' ...")
    try:
        r = await asyncio.wait_for(
            tools["web_search"].execute("transformer architecture", top_n=3),
            timeout=10,
        )
        results = r.get("results", [])
        print(f"   ✓ returned {len(results)} results")
        for i, item in enumerate(results[:2], 1):
            print(f"     {i}. {item.get('title', 'N/A')[:50]}")
    except asyncio.TimeoutError:
        errors.append("web_search: request timed out (slow network or API not responding)")
        print(f"   ✗ Timeout: search request did not return within 10 seconds")
    except Exception as e:
        errors.append(f"web_search: {e}")
        print(f"   ✗ Failed: {e}")

    # 2. browser
    test_url = (
        "https://example.com/ai-report-2024"
        if USE_MOCK
        else "https://httpbin.org/html"
    )
    print(f"\n🌐 browser: opening {test_url} ...")
    try:
        r = await asyncio.wait_for(
            tools["browser"].execute(test_url, max_chars=500),
            timeout=8,
        )
        preview = r[:120].replace("\n", " ")
        print(f"   ✓ extracted {len(r)} characters: {preview}...")
    except asyncio.TimeoutError:
        errors.append("browser: request timed out (slow page load)")
        print(f"   ✗ Timeout: page request did not return within 8 seconds")
    except Exception as e:
        errors.append(f"browser: {e}")
        print(f"   ✗ Failed: {e}")

    # 3. arxiv_reader
    print("\n📄 arxiv_reader: querying 'attention mechanism' ...")
    try:
        r = await asyncio.wait_for(
            tools["arxiv_reader"].execute(query="attention mechanism", max_results=2),
            timeout=12,
        )
        papers = r.get("papers", [])
        print(f"   ✓ returned {len(papers)} papers")
        for i, p in enumerate(papers[:2], 1):
            print(f"     {i}. {p.get('title', 'N/A')[:50]}")
    except asyncio.TimeoutError:
        errors.append("arxiv_reader: request timed out (slow ArXiv API response)")
        print(f"   ✗ Timeout: ArXiv request did not return within 12 seconds")
    except Exception as e:
        errors.append(f"arxiv_reader: {e}")
        print(f"   ✗ Failed: {e}")

    # 4. file_reader (test with a temporary file)
    print("\n📁 file_reader: reading a temporary test file ...")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "2024 Global AI Investment Report\n"
                "======================\n"
                "US: USD 30 billion\n"
                "China: USD 6 billion\n"
                "UK: USD 1.5 billion\n"
            )
            tmp_path = f.name

        r = await tools["file_reader"].execute(tmp_path)
        preview = r.split("\n")[-4:]  # take the last few lines of data
        print(f"   ✓ Read succeeded")
        for line in preview:
            print(f"     {line}")
    except Exception as e:
        errors.append(f"file_reader: {e}")
        print(f"   ✗ Failed: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # 5. calculator
    print("\n🧮 calculator: computing '(300 + 60 + 15) * 0.15' ...")
    try:
        r = await tools["calculator"].execute("(300 + 60 + 15) * 0.15")
        print(f"   ✓ Result: {r}")
    except Exception as e:
        errors.append(f"calculator: {e}")
        print(f"   ✗ Failed: {e}")

    # 6. code_sandbox
    print("\n💻 code_sandbox: running 'sum([300, 60, 15])' ...")
    try:
        r = await tools["code_sandbox"].execute("sum([300, 60, 15])")
        print(f"   ✓ stdout: {r.get('stdout', '').strip()}")
        print(f"   ✓ return_value: {r.get('return_value')}")
    except Exception as e:
        errors.append(f"code_sandbox: {e}")
        print(f"   ✗ Failed: {e}")

    # 7. notepad
    print("\n📝 notepad: write note → read note → search notes ...")
    try:
        pad = tools["notepad"]
        await pad.execute(
            action="write",
            content="US AI investment in 2024 was about USD 30 billion",
            category="conclusion",
            source="Sample data",
        )
        await pad.execute(
            action="write",
            content="Verify the official source of the China data",
            category="todo",
        )
        r = await pad.execute(action="read", max_entries=5)
        print(f"   ✓ Note recorded")
        # Print only the first-line summary
        first_line = r.split("\n")[0]
        print(f"     {first_line}")
    except Exception as e:
        errors.append(f"notepad: {e}")
        print(f"   ✗ Failed: {e}")

    return errors


# =============================================================================
# Simulate a complete research flow
# =============================================================================
async def simulate_research_flow(tools: dict) -> None:
    """Simulate a complete multi-round research flow to show tool collaboration."""

    print("\n" + "=" * 70)
    print("[Simulated Research Flow]")
    print("=" * 70)
    print("Research topic: Analysis of global generative AI investment scale in 2024")
    print("=" * 70)

    notepad = tools["notepad"]

    # ---- Round 1: Breadth search ----
    print("\n🔍 Round 1: Breadth search (web_search)")
    print("   Query: 'global generative AI investment 2024'")
    try:
        r = await asyncio.wait_for(
            tools["web_search"].execute(
                "global generative AI investment 2024", top_n=5
            ),
            timeout=10,
        )
        results = r.get("results", [])
        print(f"   Got {len(results)} search results")
        for item in results[:2]:
            print(f"   - {item.get('title', 'N/A')[:55]}")
            # Note the key link to read with browser in the next round
            if "example.com" in item.get("url", "") or USE_MOCK:
                await notepad.execute(
                    action="write",
                    content=f"Link to read: {item.get('url')} — {item.get('title')}",
                    category="todo",
                )
    except asyncio.TimeoutError:
        print("   Timeout: search request did not return within 10 seconds")
    except Exception as e:
        print(f"   Failed: {e}")

    # ---- Round 2: Deep reading ----
    print("\n📖 Round 2: Deep reading (browser)")
    url_to_read = (
        "https://example.com/ai-report-2024"
        if USE_MOCK
        else "https://httpbin.org/html"
    )
    print(f"   Opening: {url_to_read}")
    try:
        r = await asyncio.wait_for(
            tools["browser"].execute(url_to_read, max_chars=1000),
            timeout=8,
        )
        # Extract key figures (simplified: find lines containing "$" or "亿")
        lines = r.split("\n")
        key_lines = [l for l in lines if any(k in l for k in ["亿", "billion", "$", "投资"])][:3]
        if key_lines:
            print("   Extracted key information:")
            for line in key_lines:
                print(f"     · {line[:70]}")
            await notepad.execute(
                action="write",
                content=f"Extracted from {url_to_read}: " + "; ".join(key_lines[:2]),
                category="conclusion",
                source=url_to_read,
            )
        else:
            print("   No obvious numeric information extracted")
    except Exception as e:
        print(f"   Failed: {e}")

    # ---- Round 3: Numeric verification ----
    print("\n🧮 Round 3: Numeric verification (calculator)")
    print("   Computing: US 30 billion + China 6 billion + UK 1.5 billion = ?")
    try:
        r = await tools["calculator"].execute("300 + 60 + 15")
        total = r
        print(f"   Combined investment of three countries: {total} (units of 100 million USD)")
        await notepad.execute(
            action="write",
            content=f"Total 2024 generative AI investment (US+China+UK): {total} (units of 100 million USD)",
            category="conclusion",
            source="calculator summary",
        )
    except Exception as e:
        print(f"   Failed: {e}")

    # ---- Round 4: Complex computation (code_sandbox) ----
    print("\n💻 Round 4: Complex computation (code_sandbox)")
    print("   Computing: US share = 300 / 375 * 100")
    try:
        code = "total = 300 + 60 + 15; us_ratio = 300 / total * 100; print(f'{us_ratio:.1f}%')"
        r = await tools["code_sandbox"].execute(code)
        stdout = r.get("stdout", "").strip()
        print(f"   US investment share: {stdout}")
    except Exception as e:
        print(f"   Failed: {e}")

    # ---- Round 5: Academic paper cross-validation ----
    print("\n📄 Round 5: Academic validation (arxiv_reader)")
    print("   Query: 'generative AI venture capital survey'")
    try:
        r = await asyncio.wait_for(
            tools["arxiv_reader"].execute(
                query="generative AI venture capital", max_results=2
            ),
            timeout=12,
        )
        papers = r.get("papers", [])
        print(f"   Found {len(papers)} related papers")
        for p in papers[:2]:
            print(f"   - {p.get('title', 'N/A')[:55]}")
    except Exception as e:
        print(f"   Failed: {e}")

    # ---- Final: Review notes ----
    print("\n📝 Final: Review research notes (notepad)")
    try:
        r = await notepad.execute(action="read", max_entries=10)
        # Print only the summary
        lines = r.split("\n")
        for line in lines[:6]:
            print(f"   {line}")
        if len(lines) > 6:
            print(f"   ... ({len(lines) - 6} more lines)")
    except Exception as e:
        print(f"   Failed: {e}")

    print("\n" + "=" * 70)
    print("[Research Flow Simulation Complete]")
    print("=" * 70)


# =============================================================================
# Output saving
# =============================================================================
import io
from datetime import datetime


class _TeeOutput:
    """Write to both the screen and an in-memory buffer."""
    def __init__(self) -> None:
        self.stdout = sys.stdout
        self.buf = io.StringIO()

    def write(self, text: str) -> None:
        self.stdout.write(text)
        self.buf.write(text)

    def flush(self) -> None:
        self.stdout.flush()

    def getvalue(self) -> str:
        return self.buf.getvalue()


def _save_outputs(tools: dict, errors: list[str], log_text: str) -> dict:
    """Save run outputs to the outputs/demo/ directory."""
    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files_saved = {}

    # 1. Full run log
    log_path = out_dir / f"demo_log_{timestamp}.txt"
    log_path.write_text(log_text, encoding="utf-8")
    files_saved["Run log"] = str(log_path)

    # 2. Notepad notes export
    notepad = tools.get("notepad")
    if notepad and hasattr(notepad, "to_dict"):
        notes = notepad.to_dict()
        if notes:
            note_path = out_dir / f"notepad_{timestamp}.json"
            import json
            note_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
            files_saved["Research notes"] = str(note_path)

    # 3. Concise summary report
    summary_lines = [
        "# DeepResearch Agent — Demo Run Report",
        "",
        f"**Run time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run mode**: {'Mock' if USE_MOCK else 'Real'}",
        f"**Tool count**: 7",
        f"**Error count**: {len(errors)}",
        "",
        "## Tool Status",
        "",
        "| Tool | Status |",
        "|------|------|",
    ]
    tool_status = {
        "web_search": "Error" if any("web_search" in e for e in errors) else "OK",
        "browser": "Error" if any("browser" in e for e in errors) else "OK",
        "arxiv_reader": "Error" if any("arxiv_reader" in e for e in errors) else "OK",
        "file_reader": "Error" if any("file_reader" in e for e in errors) else "OK",
        "calculator": "Error" if any("calculator" in e for e in errors) else "OK",
        "code_sandbox": "Error" if any("code_sandbox" in e for e in errors) else "OK",
        "notepad": "Error" if any("notepad" in e for e in errors) else "OK",
    }
    for name, status in tool_status.items():
        icon = "❌" if status == "Error" else "✅"
        summary_lines.append(f"| {name} | {icon} {status} |")

    if errors:
        summary_lines.extend(["", "## Error Details", ""])
        for err in errors:
            summary_lines.append(f"- {err}")

    summary_lines.extend([
        "",
        "## Output Files",
        "",
    ])
    for desc, path in files_saved.items():
        summary_lines.append(f"- **{desc}**: `{path}`")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    files_saved["Summary report"] = str(summary_path)

    return files_saved


# =============================================================================
# Main function
# =============================================================================
async def main() -> None:
    tee = _TeeOutput()
    sys.stdout = tee

    print("=" * 70)
    print("DeepResearch Agent — Tool Layer Demo Script")
    print("=" * 70)
    print(f"Run mode: {'Mock (simulated data)' if USE_MOCK else 'Real (real APIs / local execution)'}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 70)

    # Create tools
    try:
        tools = create_tools()
    except Exception as e:
        print(f"\n❌ Tool initialization failed: {e}")
        print("\nHint: to debug in Mock mode, modify USE_MOCK at the top of the script")
        sys.stdout = tee.stdout
        sys.exit(1)

    # Print schema
    print_tool_schemas(tools)

    # Single tool tests
    errors = await test_tools_individually(tools)

    # Simulate a complete research flow
    await simulate_research_flow(tools)

    # Final summary
    print("\n" + "=" * 70)
    if errors:
        print(f"⚠️  Demo finished, {len(errors)} tool(s) had errors:")
        for err in errors:
            print(f"   - {err}")
        print("\nHint:")
        print("   · If an API key is missing, set the environment variable (e.g. SERPAPI_KEY)")
        print("   · Or modify USE_MOCK to switch to Mock mode")
    else:
        print("✅ All 7 tools ran successfully!")
    print("=" * 70)

    # Save output files
    sys.stdout = tee.stdout
    files_saved = _save_outputs(tools, errors, tee.getvalue())

    print("\n" + "=" * 70)
    print("[Output Files Saved]")
    print("=" * 70)
    for desc, path in files_saved.items():
        print(f"  📄 {desc}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
