#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_repl.py
================================================================================
DeepResearch Agent interactive single-session REPL script.

Features:
  1. List existing sessions on startup; supports creating a new one or resuming
  2. Ask questions continuously within a single process, sharing one Orchestrator + Memory Store
  3. All data is stored in SQLite, isolated per session
  4. Type q/quit/exit to leave; Ctrl+C interrupts gracefully

Usage:
    python scripts/run_repl.py [--config path/to/config.yaml]
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.runner import initialize_modules, load_config, run_research, save_report, setup_logging
from src.memory.memory_store import SharedMemoryStore


def list_sessions(db_path: str) -> list[dict]:
    """List all sessions in the database."""
    if not os.path.exists(db_path):
        return []
    store = SharedMemoryStore(db_path=db_path, session_id="")
    return store.list_sessions()


def print_help() -> None:
    print("""
Available commands:
  <any question>  run deep research
  ls          show the number of memories stored in the current session
  sessions    list all sessions
  save        save the last report to a file
  help        show this help
  q / quit / exit  exit the REPL
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepResearch Agent interactive REPL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--session_id", type=str, default=None, help="Specify session_id directly, skipping interactive selection")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("repl")

    config = load_config(args.config)
    db_path = config.get("memory", {}).get("db_path", "data/memory.db")

    # ------------------------------------------------------------------
    # Session selection
    # ------------------------------------------------------------------
    if args.session_id:
        session_id = args.session_id
        print(f"[REPL] Session specified: {session_id}")
    else:
        sessions = list_sessions(db_path)

        print("=" * 50)
        print("DeepResearch Agent interactive REPL")
        print("=" * 50)

        if sessions:
            print("\nExisting sessions:")
            for i, s in enumerate(sessions, 1):
                ts = datetime.fromtimestamp(s["last_update"]).strftime("%Y-%m-%d %H:%M")
                print(f"  [{i}] {s['session_id']:25s} ({s['count']:3d} memories, last updated {ts})")
            print("  [N] New session")
            choice = input("\nChoose (number or N): ").strip()
            if choice.lower() == "n":
                session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                try:
                    idx = int(choice) - 1
                    session_id = sessions[idx]["session_id"]
                except (ValueError, IndexError):
                    print("Invalid choice, creating a new session")
                    session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            print("\nNo existing sessions, creating a new one...")
            session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n[REPL] Current session: {session_id}")

    # ------------------------------------------------------------------
    # Initialize modules (once only, reused for the whole REPL lifetime)
    # ------------------------------------------------------------------
    print("[REPL] Initializing modules...")
    modules = initialize_modules(config, session_id=session_id)
    print(f"[REPL] Modules initialized; type 'help' for commands, 'q' to quit\n")

    last_report: str | None = None

    # ------------------------------------------------------------------
    # REPL loop
    # ------------------------------------------------------------------
    while True:
        try:
            query = input(f"[{session_id}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[REPL] Interrupt received, exiting...")
            break

        if not query:
            continue

        cmd = query.lower()

        if cmd in ("q", "quit", "exit"):
            break
        elif cmd == "help":
            print_help()
            continue
        elif cmd == "ls":
            count = len(modules["memory_store"])
            print(f"  Current session '{session_id}' has {count} memories")
            continue
        elif cmd == "sessions":
            all_sessions = list_sessions(db_path)
            if not all_sessions:
                print("  No sessions")
            else:
                for s in all_sessions:
                    ts = datetime.fromtimestamp(s["last_update"]).strftime("%Y-%m-%d %H:%M")
                    marker = " <- current" if s["session_id"] == session_id else ""
                    print(f"  {s['session_id']:25s} ({s['count']:3d} memories) {ts}{marker}")
            continue
        elif cmd == "save":
            if last_report:
                filepath = save_report(last_report, "repl_report", "outputs/reports")
                print(f"  Report saved: {filepath}")
            else:
                print("  No report to save")
            continue

        # ------------------------------------------------------------------
        # Run deep research
        # ------------------------------------------------------------------
        print(f"[REPL] Researching: {query[:60]}...")
        start = time.time()
        try:
            report = asyncio.run(run_research(query, config, modules))
            elapsed = time.time() - start
            last_report = report

            # Parse metadata (extracted from the end of the report)
            confidence = 0.0
            num_searches = 0
            for line in report.splitlines():
                if "**置信度**:" in line:
                    try:
                        confidence = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                if "**搜索轮数**:" in line:
                    try:
                        num_searches = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

            print(f"\n  ✓ Report complete | {len(report)} chars | confidence {confidence:.2f} | "
                  f"searches {num_searches} rounds | elapsed {elapsed:.1f}s")
            print(f"  Type 'save' to save the report, 'ls' to show the current session's memory count\n")

        except Exception as e:
            logger.exception("Research execution failed")
            print(f"\n  ✗ Execution failed: {e}\n")

    # ------------------------------------------------------------------
    # Exit cleanup
    # ------------------------------------------------------------------
    print(f"\n[REPL] Data for session '{session_id}' persisted to {db_path}")
    print("[REPL] Next time, resume it with the --session_id argument or pick it from the interactive menu.")
    print("[REPL] Goodbye!")


if __name__ == "__main__":
    main()
