"""
M6 self-evolution engine — Symbolic Learning (prompt self-optimization)

SymbolicLearner extracts systematic error patterns from failed trajectories, generates improvement instructions and updates prompts.
It has version management and automatic rollback to keep prompt evolution safe.

Design decisions:
1. Error-pattern extraction: an LLM analyzes failed trajectories and summarizes common error types.
2. Prompt optimization: turn error patterns into concrete prompt-improvement instructions (e.g. "add an XX constraint").
3. Version management: keep the latest 10 prompt versions, enabling quick rollback.
4. Automatic rollback: revert to the previous version when a new prompt causes a performance drop >5%.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any


__all__ = ["SymbolicLearner"]


# ============================================================================
# Prompt templates
# ============================================================================

SYSTEM_SYMBOLIC = (
    "You are a Prompt Engineering expert. Your task is to analyze failed trajectories of an AI Agent, "
    "extract systematic error patterns, and generate precise prompt-improvement instructions."
)

PROMPT_EXTRACT_PATTERNS = """Analyze the following failed trajectories and extract systematic error patterns.

Requirements:
1. Focus only on recurring error types (ignore one-off errors).
2. Each error pattern must include: error description, frequency, root-cause analysis.
3. Sort by severity (severe -> minor).

Output in the following JSON format:
{
  "patterns": [
    {
      "pattern_name": "string",
      "description": "string",
      "frequency": "high|medium|low",
      "root_cause": "string",
      "suggested_fix": "string"
    }
  ]
}

--- List of failed trajectories ---
{trajectories}
"""

PROMPT_OPTIMIZE_PROMPT = """Optimize the given prompts according to the following error patterns.

Optimization principles:
1. Keep the prompt's core goal unchanged.
2. For each error pattern, add concrete constraints or examples.
3. Avoid overly long prompts (keep within 2000 tokens).
4. Output the complete optimized prompt.

Output in the following JSON format:
{
  "optimized_prompts": {
    "prompt_name": "string",   // e.g. "system_prompt"
    "new_content": "string",
    "changes": ["string"]      // list of change descriptions
  }
}

--- Error patterns ---
{patterns}

--- Current prompts ---
{current_prompts}
"""


# ============================================================================
# SymbolicLearner implementation
# ============================================================================

class SymbolicLearner:
    """Prompt self-optimizer: symbolic learning based on failed trajectories.

    Attributes:
        policy: a VLLMPolicy instance.
        max_versions: maximum number of prompt versions kept.
        rollback_threshold: performance-drop threshold (ratio) that triggers rollback.
        _prompt_versions: prompt version history stack.
        _performance_history: list of performance records, used for rollback decisions.
    """

    def __init__(
        self,
        policy,
        max_versions: int = 10,
        rollback_threshold: float = 0.05,
    ):
        self.policy = policy
        self.max_versions = max_versions
        self.rollback_threshold = rollback_threshold
        self._prompt_versions: list[dict[str, str]] = []
        self._performance_history: list[dict[str, float]] = []

    async def optimize_prompts(
        self,
        failed_trajectories: list[dict[str, Any]],
        current_prompts: dict[str, str],
    ) -> dict[str, str]:
        """Extract error patterns from failed trajectories and optimize prompts.

        Flow:
        1. Compress failed trajectories into a text summary.
        2. Call the LLM to extract systematic error patterns.
        3. Call the LLM to optimize prompts based on the error patterns.
        4. Save the current version onto the history stack.

        Args:
            failed_trajectories: list of failed trajectories, each in the collect() output format.
            current_prompts: dict of prompts currently in use, keyed by prompt name.

        Returns:
            Dict of optimized prompts.
        """
        if not failed_trajectories:
            return copy.deepcopy(current_prompts)

        # Step 1: compress failed trajectories
        traj_text = self._compress_trajectories(failed_trajectories)

        # Step 2: extract error patterns
        patterns = await self._extract_patterns(traj_text)
        if not patterns:
            return copy.deepcopy(current_prompts)

        # Step 3: optimize prompts
        new_prompts = await self._generate_optimized_prompts(patterns, current_prompts)

        # Step 4: save the version
        self._save_version(current_prompts)

        return new_prompts

    def rollback_if_needed(
        self,
        new_prompts: dict[str, str],
        performance: dict[str, float],
    ) -> dict[str, str]:
        """Check performance and roll back to the previous version if it dropped beyond the threshold.

        Args:
            new_prompts: the newly applied prompts.
            performance: current-round performance metrics; must contain the "avg_score" key.

        Returns:
            The previous version's prompts if rolled back, otherwise new_prompts.
        """
        self._performance_history.append(performance)

        if len(self._performance_history) < 2:
            return new_prompts

        prev_perf = self._performance_history[-2]
        curr_perf = self._performance_history[-1]

        prev_score = prev_perf.get("avg_score", 0.0)
        curr_score = curr_perf.get("avg_score", 0.0)

        if prev_score > 0.0:
            drop = (prev_score - curr_score) / prev_score
            if drop > self.rollback_threshold:
                # Trigger rollback
                rolled_back = self._rollback_one()
                if rolled_back is not None:
                    # Roll back performance_history
                    self._performance_history.pop()
                    return rolled_back

        return new_prompts

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compress_trajectories(self, trajectories: list[dict[str, Any]]) -> str:
        """Compress failed trajectories into a text summary, keeping the prompt length under control."""
        parts = []
        for i, traj in enumerate(trajectories[:20]):  # take at most 20
            query = traj.get("query", "")
            final_score = traj.get("final_score", 0.0)
            num_searches = traj.get("num_searches", 0)
            content_preview = traj.get("report_content", "")[:300]
            parts.append(
                f"[Case {i+1}] query={query}, score={final_score}, searches={num_searches}\n"
                f"Content preview: {content_preview}\n"
            )
        return "\n".join(parts)

    async def _extract_patterns(self, traj_text: str) -> list[dict[str, str]]:
        """Call the LLM to extract systematic error patterns."""
        prompt = PROMPT_EXTRACT_PATTERNS.format(trajectories=traj_text)
        messages = [
            {"role": "system", "content": SYSTEM_SYMBOLIC},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.policy(messages)
            raw = resp.content or ""
            data = self._parse_json(raw)
            return data.get("patterns", [])
        except Exception:
            return []

    async def _generate_optimized_prompts(
        self,
        patterns: list[dict[str, str]],
        current_prompts: dict[str, str],
    ) -> dict[str, str]:
        """Generate optimized prompts from the error patterns."""
        patterns_text = json.dumps(patterns, ensure_ascii=False, indent=2)
        current_text = json.dumps(current_prompts, ensure_ascii=False, indent=2)
        prompt = PROMPT_OPTIMIZE_PROMPT.format(
            patterns=patterns_text,
            current_prompts=current_text,
        )
        messages = [
            {"role": "system", "content": SYSTEM_SYMBOLIC},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.policy(messages)
            raw = resp.content or ""
            data = self._parse_json(raw)
            # Parse the optimized prompts
            optimized = data.get("optimized_prompts", {})
            if isinstance(optimized, dict) and "new_content" in optimized:
                # Single-prompt optimization
                name = optimized.get("prompt_name", "system_prompt")
                result = copy.deepcopy(current_prompts)
                result[name] = optimized["new_content"]
                return result
            elif isinstance(optimized, list):
                # Multi-prompt optimization
                result = copy.deepcopy(current_prompts)
                for item in optimized:
                    name = item.get("prompt_name", "system_prompt")
                    result[name] = item.get("new_content", result.get(name, ""))
                return result
            return copy.deepcopy(current_prompts)
        except Exception:
            return copy.deepcopy(current_prompts)

    def _save_version(self, prompts: dict[str, str]) -> None:
        """Save the current prompt version onto the history stack, evicting the oldest when over the limit."""
        self._prompt_versions.append(copy.deepcopy(prompts))
        while len(self._prompt_versions) > self.max_versions:
            self._prompt_versions.pop(0)

    def _rollback_one(self) -> dict[str, str] | None:
        """Roll back to the previous version."""
        if len(self._prompt_versions) < 2:
            return None
        # The current version is already at the top of the stack (saved by _save_version)
        # Rollback = discard the latest version and return the second-to-last
        self._prompt_versions.pop()
        return copy.deepcopy(self._prompt_versions[-1])

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Robust JSON parsing."""
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        import re

        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                return json.loads(m.strip())
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass
        return {}

    def save_versions_to_disk(self, dir_path: str) -> None:
        """Persist the version history to disk."""
        os.makedirs(dir_path, exist_ok=True)
        for i, version in enumerate(self._prompt_versions):
            path = os.path.join(dir_path, f"prompt_v{i:03d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(version, f, ensure_ascii=False, indent=2)

    def load_versions_from_disk(self, dir_path: str) -> None:
        """Load the version history from disk."""
        if not os.path.isdir(dir_path):
            return
        files = sorted(
            [f for f in os.listdir(dir_path) if f.endswith(".json")],
            key=lambda x: int(x.split("_v")[1].split(".")[0]),
        )
        self._prompt_versions = []
        for fname in files:
            with open(os.path.join(dir_path, fname), "r", encoding="utf-8") as f:
                self._prompt_versions.append(json.load(f))
