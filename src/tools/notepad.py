"""
Notepad tool (NotepadTool)

Rationale:
  Deep research is a long-horizon, multi-round process. After 10+ rounds of search an Agent easily "forgets" early conclusions
  or falls into repeated searches. NotepadTool provides a persistent "scratch pad" so the Agent can:

  1. Record intermediate conclusions (e.g. "Company A 2024 revenue = 30 billion, source: filing page 3")
  2. Record hypotheses to verify ("need to confirm whether Company B also launched a similar product")
  3. Record search strategy ("already searched X and Y, next search Z")
  4. Read the notes in later rounds to avoid duplicated work

Difference from Memory Store (M4):
  - Memory Store: structured, de-duplicated, contradiction-detecting, used to share information across Agents
  - Notepad: unstructured, personal, temporary, the "thinking scratch" of a single Agent

Design points:
  - Pure in-memory implementation (session level), not persisted
  - Supports CRUD: write / read / list / clear
  - Each note carries a timestamp and category (conclusion / todo / question / source)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


__all__ = ["NotepadTool", "NotepadEntry"]


@dataclass
class NotepadEntry:
    """A single note."""
    content: str
    category: str  # "conclusion" | "todo" | "question" | "source" | "strategy"
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # optional: the source/basis of this note


class NotepadTool:
    """Notepad tool: the Agent's scratch pad."""

    name: str = "notepad"
    description: str = (
        "A personal notepad for the agent to record intermediate thoughts, conclusions, "
        "and todo items during long-horizon research. Use this to avoid forgetting key "
        "findings or repeating searches. "
        "Input: {'action': str, 'content': str(optional), 'category': str(optional), ...}. "
        "Actions: write, read, list_categories, clear, search."
    )

    def __init__(self) -> None:
        self._notes: list[NotepadEntry] = []

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to perform: write, read, list_categories, clear, search",
                            "enum": ["write", "read", "list_categories", "clear", "search"],
                        },
                        "content": {
                            "type": "string",
                            "description": "Content for write action",
                        },
                        "category": {
                            "type": "string",
                            "description": "Category for write/read/clear: conclusion, todo, question, source, strategy",
                        },
                        "source": {
                            "type": "string",
                            "description": "Optional source annotation for write action",
                        },
                        "keyword": {
                            "type": "string",
                            "description": "Search keyword for search action",
                        },
                        "max_entries": {
                            "type": "integer",
                            "description": "Maximum entries to return for read/search",
                            "default": 10,
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    async def execute(self, action: str, **kwargs) -> str:
        """Unified entry point: dispatch to the specific method by action."""
        import asyncio
        await asyncio.sleep(0)

        if action == "write":
            return await self.write(**kwargs)
        if action == "read":
            return await self.read(**kwargs)
        if action == "list_categories":
            return await self.list_categories(**kwargs)
        if action == "clear":
            return await self.clear(**kwargs)
        if action == "search":
            return await self.search(**kwargs)
        return f"[Notepad Error] Unknown action: {action}. Supported: write, read, list_categories, clear, search."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write(self, content: str, category: str = "conclusion", source: str = "") -> str:
        import asyncio
        await asyncio.sleep(0)

        """Write a note.

        Args:
            content: note content.
            category: note type. Suggested values:
                      - "conclusion": a confirmed conclusion
                      - "todo": a to-do item (needs later verification)
                      - "question": an unanswered question
                      - "source": a record of an important source
                      - "strategy": search strategy/plan
            source: optional source annotation.

        Returns:
            Confirmation message.
        """
        entry = NotepadEntry(content=content, category=category, source=source)
        self._notes.append(entry)
        return f"[Notepad] Written ({category}): {content[:80]}{'...' if len(content) > 80 else ''}"

    async def read(self, category: str | None = None, max_entries: int = 10) -> str:
        import asyncio
        await asyncio.sleep(0)

        """Read notes.

        Args:
            category: only read notes of this type. None reads all.
            max_entries: maximum number of entries returned (newest first).

        Returns:
            Formatted list of notes.
        """
        notes = self._notes
        if category:
            notes = [n for n in notes if n.category == category]

        if not notes:
            cat_hint = f' in category "{category}"' if category else ""
            return f"[Notepad] No notes found{cat_hint}."

        # Newest first
        notes = sorted(notes, key=lambda n: n.timestamp, reverse=True)[:max_entries]

        lines = [f"=== Notepad ({len(notes)} entries) ==="]
        for i, n in enumerate(notes, 1):
            time_str = time.strftime("%H:%M:%S", time.localtime(n.timestamp))
            source_hint = f" [src: {n.source}]" if n.source else ""
            lines.append(f"{i}. [{n.category}] {time_str}{source_hint}\n   {n.content}")
        return "\n".join(lines)

    async def list_categories(self) -> str:
        import asyncio
        await asyncio.sleep(0)

        """List all note categories and their counts."""
        from collections import Counter
        counts = Counter(n.category for n in self._notes)
        if not counts:
            return "[Notepad] No notes."
        lines = ["=== Notepad Categories ==="]
        for cat, cnt in counts.most_common():
            lines.append(f"  {cat}: {cnt}")
        return "\n".join(lines)

    async def clear(self, category: str | None = None) -> str:
        import asyncio
        await asyncio.sleep(0)

        """Clear notes.

        Args:
            category: only clear this type. None clears all.
        """
        if category is None:
            count = len(self._notes)
            self._notes.clear()
            return f"[Notepad] Cleared all {count} notes."

        before = len(self._notes)
        self._notes = [n for n in self._notes if n.category != category]
        removed = before - len(self._notes)
        return f"[Notepad] Cleared {removed} notes in category '{category}'."

    async def search(self, keyword: str, max_entries: int = 5) -> str:
        import asyncio
        await asyncio.sleep(0)

        """Search note content.

        Args:
            keyword: search keyword.
            max_entries: maximum number of entries returned.
        """
        matches = [n for n in self._notes if keyword.lower() in n.content.lower()]
        if not matches:
            return f"[Notepad] No notes matching '{keyword}'."

        matches = sorted(matches, key=lambda n: n.timestamp, reverse=True)[:max_entries]
        lines = [f"=== Notepad Search: '{keyword}' ({len(matches)} matches) ==="]
        for i, n in enumerate(matches, 1):
            time_str = time.strftime("%H:%M:%S", time.localtime(n.timestamp))
            lines.append(f"{i}. [{n.category}] {time_str}\n   {n.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization (for saving trajectories)
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict]:
        """Export as a list of dicts."""
        return [
            {
                "content": n.content,
                "category": n.category,
                "timestamp": n.timestamp,
                "source": n.source,
            }
            for n in self._notes
        ]
