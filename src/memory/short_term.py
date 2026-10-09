"""
Short-term Memory module: session-level temporary storage

Design decisions:
1. A pure in-memory structure (list + dict), no persistence, for the lowest latency
2. Each message keeps metadata (timestamp, token_count, etc.) to support later compression decisions
3. Supports a role-merge check (consecutive same-role messages can be merged to reduce context fragmentation)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Turn:
    """Record of a single conversation turn."""

    turn_id: str
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """
    Session-level short-term memory.

    Maintains the full conversation history of the current session, supporting filtering by role, merging of consecutive messages,
    and a fast estimate of the total token count (character-count heuristic).
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        Initialize short-term memory.

        Args:
            session_id: unique session identifier; a UUID is generated when None
        """
        self.session_id = session_id or str(uuid.uuid4())
        self._turns: list[Turn] = []
        self._created_at = time.time()

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Turn:
        """
        Add one conversation turn.

        Args:
            role: the role, usually "system" / "user" / "assistant" / "tool"
            content: the message content
            metadata: extra metadata (e.g. token_count, tool_call_id)

        Returns:
            the created Turn object
        """
        turn = Turn(
            turn_id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._turns.append(turn)
        return turn

    def get_history(
        self,
        roles: Optional[list[str]] = None,
        last_n: Optional[int] = None,
    ) -> list[Turn]:
        """
        Get the conversation history.

        Args:
            roles: filter by role; None means no filtering
            last_n: return only the last N; None means all

        Returns:
            list of Turns (in chronological order)
        """
        turns = self._turns
        if roles:
            turns = [t for t in turns if t.role in roles]
        if last_n is not None:
            turns = turns[-last_n:]
        return turns

    def get_history_as_dicts(
        self,
        roles: Optional[list[str]] = None,
        last_n: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Convert the history into a list of OpenAI-style dicts, ready to pass to the LLM.

        Returns:
            list of {"role": ..., "content": ...}
        """
        return [
            {"role": t.role, "content": t.content, **t.metadata}
            for t in self.get_history(roles=roles, last_n=last_n)
        ]

    def clear(self) -> None:
        """Clear all conversation records of the current session."""
        self._turns.clear()

    def estimate_tokens(self) -> int:
        """
        Estimate the tokens taken by the current history.

        Uses the characters / 3.5 heuristic (an empirical value for mixed Chinese/English).
        A more exact count needs tiktoken, but the heuristic is enough to decide compression triggers.
        """
        total_chars = sum(len(t.content) for t in self._turns)
        # add 10% overhead for role labels and formatting
        return int(total_chars / 3.5 * 1.1)

    def estimate_chars(self) -> int:
        """Estimate the total character count of the current history."""
        return sum(len(t.content) for t in self._turns)

    def last_turn(self) -> Optional[Turn]:
        """Return the latest record, or None if there is none."""
        return self._turns[-1] if self._turns else None

    def __len__(self) -> int:
        return len(self._turns)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(session_id={self.session_id[:8]}, "
            f"turns={len(self._turns)}, est_tokens={self.estimate_tokens()})"
        )
