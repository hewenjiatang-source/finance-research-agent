"""
Sliding Window Compressor module: FIFO truncation of old messages

Design decisions:
1. Reuses the core idea of project one's VLLMPolicy._truncate_messages, extracted into a standalone module
2. Improvements: (a) supports token estimation (not just characters), (b) a finer role-retention policy
3. Truncation granularity is "drop whole messages", to avoid cutting in the middle of a message and breaking its meaning
4. In extreme cases the last message is truncated at content level as a fallback, but at least 500 characters are kept
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SlidingWindowCompressor:
    """
    Sliding-window truncator.

    Drops old messages FIFO, keeping the system prompt and recent interaction first;
    suited to fast degradation when the session history exceeds the context budget.
    """

    def __init__(
        self,
        max_tokens: int = 12000,
        char_per_token: float = 3.5,
        min_recent_turns: int = 3,
        min_last_msg_chars: int = 500,
    ) -> None:
        """
        Initialize the sliding-window truncator.

        Args:
            max_tokens: maximum allowed tokens
            char_per_token: characters-per-token conversion ratio (about 3.0-4.0 for mixed Chinese/English)
            min_recent_turns: minimum number of non-system messages to keep
            min_last_msg_chars: minimum characters of the last message kept in extreme cases
        """
        self.max_tokens = max_tokens
        self.char_per_token = char_per_token
        self.min_recent_turns = min_recent_turns
        self.min_last_msg_chars = min_last_msg_chars
        self._last_truncated = False
        self._last_stats: dict[str, Any] = {}

    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Run sliding-window truncation.

        Args:
            messages: list of OpenAI-format messages, each dict with role/content

        Returns:
            the truncated list of messages
        """
        self._last_truncated = False
        max_chars = int(self.max_tokens * self.char_per_token)
        result = self._truncate_messages(messages, max_chars)
        return result

    def _truncate_messages(
        self,
        messages: list[dict[str, Any]],
        max_chars: int,
    ) -> list[dict[str, Any]]:
        """
        Core truncation logic.

        Steps:
        1. separate system messages from the others
        2. if the total character count is within the limit, return directly
        3. drop from the oldest messages until the character count fits or only min_recent_turns remain
        4. in extreme cases truncate the content of the last message
        """
        system_msgs = [
            m for m in messages if isinstance(m, dict) and m.get("role") == "system"
        ]
        other_msgs = [
            m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")
        ]

        before_chars = self._count_chars(messages)
        if before_chars <= max_chars:
            self._last_truncated = False
            self._last_stats = {
                "before_chars": before_chars,
                "after_chars": before_chars,
                "removed_turns": 0,
                "truncated": False,
            }
            return messages

        self._last_truncated = True
        logger.info(
            f"[SlidingWindow] Triggered: {before_chars} chars > {max_chars} threshold. "
            f"n_msgs={len(messages)}"
        )

        kept = list(other_msgs)
        removed_count = 0
        while len(kept) > self.min_recent_turns:
            removed = kept.pop(0)
            removed_count += 1
            after_chars = self._count_chars(system_msgs + kept)
            if after_chars <= max_chars:
                logger.info(
                    f"[SlidingWindow] Reduced to {after_chars} chars, "
                    f"kept {len(kept)} non-system msgs"
                )
                self._last_stats = {
                    "before_chars": before_chars,
                    "after_chars": after_chars,
                    "removed_turns": removed_count,
                    "truncated": True,
                }
                return system_msgs + kept

        # extreme case: still over the limit even with only system + the last N kept
        after_chars = self._count_chars(system_msgs + kept)
        if after_chars > max_chars and kept:
            last_msg = kept[-1]
            excess = after_chars - max_chars
            content = str(last_msg.get("content", ""))
            new_len = max(
                len(content) - excess - 100,
                self.min_last_msg_chars,
            )
            if new_len < len(content):
                last_msg["content"] = content[:new_len] + "\n[CONTENT_TRUNCATED]"
            final_chars = self._count_chars(system_msgs + kept)
            logger.info(
                f"[SlidingWindow] Content-truncated last msg to {new_len} chars. "
                f"Final: {final_chars}"
            )
            self._last_stats = {
                "before_chars": before_chars,
                "after_chars": final_chars,
                "removed_turns": removed_count,
                "truncated": True,
            }
            return system_msgs + kept

        self._last_stats = {
            "before_chars": before_chars,
            "after_chars": after_chars,
            "removed_turns": removed_count,
            "truncated": True,
        }
        return system_msgs + kept

    def _count_chars(self, messages: list[dict[str, Any]]) -> int:
        """Compute the total character count of a message list, including content/tool_calls/tool metadata."""
        total = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            total += len(str(m.get("content", "")))
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    total += len(str(func.get("arguments", "")))
                    total += len(str(func.get("name", "")))
            if m.get("role") == "tool":
                total += len(str(m.get("tool_call_id", "")))
                total += len(str(m.get("name", "")))
        return total

    def get_stats(self) -> dict[str, Any]:
        """Return the statistics of the most recent compression."""
        return dict(self._last_stats)

    def was_truncated(self) -> bool:
        """Return whether the most recent compression caused truncation."""
        return self._last_truncated
