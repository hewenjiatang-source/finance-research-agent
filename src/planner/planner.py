"""
Adaptive Planner

An LLM-driven planner that decomposes a research question into a structured sub-task DAG.
Core capabilities:
  - initial planning: generate a DAG of 3-8 sub-questions
  - incremental re-planning: keep successful results with confidence≥0.6, modify only the failed sub-questions
  - robust JSON parsing: tolerates noise such as markdown code blocks and extra newlines
"""
from __future__ import annotations

import json
import re
from typing import Any

from .dag import DAG, DAGCycleError
from .budget_tracker import BudgetTracker
from ..orchestrator.schemas import SubTask, TaskType, AgentResult
from ..utils.tracing import trace_chain


__all__ = ["Planner", "PlanParseError"]


class PlanParseError(Exception):
    """Raised when the planning result cannot be parsed."""
    pass


# ============================================================================
# Prompt constants
# ============================================================================

INITIAL_PLAN_PROMPT = """\
You are an expert research planner. Your task is to decompose a complex research question into a directed acyclic graph (DAG) of sub-tasks.

## Input
Research Question: {query}

## Output Format
Return a JSON object with this exact structure (no markdown, no extra text):
{{
  "sub_tasks": [
    {{
      "task_id": "task_1",
      "task_type": "search",
      "description": "What is ...",
      "dependencies": [],
      "context_keys": [],
      "timeout_seconds": 120,
      "priority": 1,
      "expected_type": "factual",
      "search_hints": ["keyword1", "keyword2"]
    }}
  ]
}}

## Rules
1. task_type must be one of: search, analyze, verify
2. dependencies must reference existing task_id values
3. The graph must be a DAG (no cycles)
4. Generate 3 to 8 sub_tasks
5. More fundamental/information-gathering tasks should have fewer dependencies
6. Verification tasks should depend on analysis tasks
7. Use concise but clear descriptions
8. CRITICAL — RELEVANCE CONSTRAINT: Each sub-task description MUST directly address the research question. If the user asks about 'internship/job application', do NOT generate tasks about 'technology trends', 'annual news summary', or 'science breakthroughs'.
9. The search_hints field MUST contain keywords directly from the query. Do NOT invent unrelated keywords.
10. Prefer specific, actionable queries over broad, vague ones.

## Anti-examples (DO NOT do this)
- Query: "How to find an internship at a big tech company" → BAD tasks: "2025 technology trends", "annual science news", "latest AI breakthroughs"
- Query: "How to prepare for post-training LLM engineer internship" → GOOD tasks: "Big tech post-training intern JD requirements", "LLM post-training intern interview experience", "Resume tips for LLM algorithm intern"

## Context (if any)
{memory_context}
"""

REPLAN_PROMPT = """\
You are an expert research planner. Some sub-tasks failed and need to be re-planned.

## Original Question
{query}

## Failed Tasks
{failed_tasks_json}

## Successful Results to Preserve (confidence >= 0.6)
{preserved_results_json}

## Reason for Failure
{reason}

## Output Format
Return a JSON object with new sub_tasks. You may:
1. Modify failed tasks (new task_id, same or different description)
2. Add new tasks to fill gaps
3. Remove tasks that are no longer needed
4. Keep dependencies consistent

Structure:
{{
  "sub_tasks": [...]
}}

Only return the JSON. No markdown, no extra text.
"""


class Planner:
    """Adaptive planner.

    Attributes:
        policy: VLLMPolicy instance used to call the LLM.
        budget_tracker: optional budget tracker that monitors the token consumption of the planning phase.
    """

    def __init__(self, policy, budget_tracker: BudgetTracker | None = None) -> None:
        self.policy = policy
        self.budget_tracker = budget_tracker or BudgetTracker()
        self._last_raw_json: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @trace_chain(name="planner.generate_plan", tags=["m2", "planner"])
    def generate_plan(self, query: str, memory_context: str = "") -> DAG:
        """Generate the initial execution plan (DAG).

        Args:
            query: the original research question.
            memory_context: historical context (an empty string for the first plan).

        Returns:
            DAG: the sub-task dependency graph.

        Raises:
            PlanParseError: raised when the LLM output cannot be parsed into a valid DAG.
        """
        prompt = self._build_prompt(query, memory_context)
        messages = [
            {"role": "system", "content": "You are a research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.policy(messages)
        except RuntimeError as e:
            raise PlanParseError(f"LLM call failed during planning: {e}") from e

        content = response.get("content", "") or ""
        self._last_raw_json = content
        # estimate the planning token consumption
        self.budget_tracker.track(len(content) // 3)

        return self._parse_plan(content)

    @trace_chain(name="planner.replan", tags=["m2", "planner"])
    def replan(
        self,
        query: str,
        failed_tasks: list[SubTask],
        existing_results: list[AgentResult],
        reason: str,
    ) -> DAG:
        """Incremental re-planning: keep high-confidence results, modify failed tasks.

        Args:
            query: the original research question.
            failed_tasks: list of SubTasks that failed.
            existing_results: all historical execution results.
            reason: description of the failure reasons.

        Returns:
            DAG: the new execution plan.
        """
        # select results to keep (confidence >= 0.6 and status SUCCESS)
        preserved = [
            {
                "task_id": r.task_id,
                "output": str(r.output)[:500] if r.output else "",
                "confidence": r.confidence,
            }
            for r in existing_results
            if r.status.value == "success" and r.confidence >= 0.6
        ]

        failed_json = json.dumps(
            [{"task_id": t.task_id, "description": t.description, "type": t.task_type.value} for t in failed_tasks],
            ensure_ascii=False,
            indent=2,
        )
        preserved_json = json.dumps(preserved, ensure_ascii=False, indent=2)

        prompt = REPLAN_PROMPT.format(
            query=query,
            failed_tasks_json=failed_json,
            preserved_results_json=preserved_json,
            reason=reason,
        )
        messages = [
            {"role": "system", "content": "You are a research planning assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.policy(messages)
        except RuntimeError as e:
            raise PlanParseError(f"LLM call failed during replanning: {e}") from e

        content = response.get("content", "") or ""
        self._last_raw_json = content
        self.budget_tracker.track(len(content) // 3)

        return self._parse_plan(content)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, memory: str) -> str:
        """Build the initial planning prompt."""
        # on a first run (no history), tell the Planner to decompose more aggressively
        has_memory = bool(memory and memory.strip() and memory != "None")
        if not has_memory:
            extra_hint = (
                "\n## Note\n"
                "No previous research memory is available for this topic. "
                "Please be MORE AGGRESSIVE in decomposition: generate 6-10 sub_tasks to thoroughly cover the topic, "
                "rather than the usual 3-5. Each sub-task should focus on a distinct angle or data source.\n"
                "IMPORTANT: Each sub-task description must directly reflect the user's original intent. "
                "If the user asks about 'internship application strategies', do NOT generate tasks about '2025 tech trends' or 'annual science summary'."
            )
        else:
            extra_hint = (
                "\n## Note\n"
                "Use the preserved successful results above to inform new sub-tasks. "
                "New tasks should fill gaps and avoid duplicating existing coverage."
            )
        return INITIAL_PLAN_PROMPT.format(query=query, memory_context=memory or "None") + extra_hint

    def _parse_plan(self, json_str: str) -> DAG:
        """Robust JSON parsing: handles noise such as markdown code blocks and extra newlines.

        Parsing strategy:
          1. first try json.loads directly
          2. on failure, extract the markdown code block content
          3. clean up common noise (trailing commas, comments, etc.)
          4. verify the DAG is acyclic
        """
        raw = json_str.strip()

        # try to extract a markdown code block
        if raw.startswith("```"):
            # strip the first line ```json or ```
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # try to extract the content between ```json...``` (even when it is not at the beginning)
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()

        # try to find the outermost JSON object directly
        if not raw.startswith("{"):
            obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1).strip()

        # clean trailing commas (JSON does not allow trailing commas)
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        # parse the JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            # last attempt: repair line by line (strip comments)
            cleaned_lines = []
            for line in raw.splitlines():
                # strip // comments
                if "//" in line:
                    line = line[: line.index("//")]
                cleaned_lines.append(line)
            try:
                data = json.loads("\n".join(cleaned_lines))
            except json.JSONDecodeError:
                raise PlanParseError(
                    f"Failed to parse planner output as JSON. Raw snippet: {json_str[:500]}"
                ) from e

        if not isinstance(data, dict) or "sub_tasks" not in data:
            raise PlanParseError(f"Planner output missing 'sub_tasks' key. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        sub_tasks_raw = data["sub_tasks"]
        if not isinstance(sub_tasks_raw, list):
            raise PlanParseError(f"'sub_tasks' must be a list, got {type(sub_tasks_raw)}")

        dag = DAG()
        for item in sub_tasks_raw:
            task = self._deserialize_subtask(item)
            dag.add_node(task.task_id)

        # second pass: add edges
        for item in sub_tasks_raw:
            task_id = item.get("task_id", "")
            for dep in item.get("dependencies", []):
                if not dag.has_node(dep):
                    # the dependency points to a missing task, create a placeholder node
                    dag.add_node(dep)
                dag.add_edge(dep, task_id)  # dep -> task_id (task_id depends on dep)

        # verify acyclic
        try:
            dag.topological_sort()
        except DAGCycleError as e:
            raise PlanParseError(f"Planner generated a cyclic graph: {e}") from e

        return dag

    def _deserialize_subtask(self, item: dict[str, Any]) -> SubTask:
        """Deserialize a JSON dict into a SubTask."""
        task_type_str = item.get("task_type", "search")
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.SEARCH  # degrade to the default

        return SubTask(
            task_id=item.get("task_id", "unknown"),
            task_type=task_type,
            description=item.get("description", ""),
            dependencies=list(item.get("dependencies", [])),
            context_keys=list(item.get("context_keys", [])),
            timeout_seconds=int(item.get("timeout_seconds", 120)),
            priority=int(item.get("priority", 1)),
            expected_type=item.get("expected_type", "factual"),
            search_hints=list(item.get("search_hints", [])),
        )

    def get_task_map_from_dag(self, dag: DAG, raw_json: str) -> dict[str, SubTask]:
        """Rebuild the task_id -> SubTask mapping from the DAG and the raw JSON.

        Usually called by the orchestrator after generate_plan.
        """
        # reuse the parsing logic in _parse_plan but return a mapping
        # re-parse raw_json here to get the full SubTask info
        raw = raw_json.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if code_block_match:
            raw = code_block_match.group(1).strip()
        if not raw.startswith("{"):
            obj_match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if obj_match:
                raw = obj_match.group(1).strip()
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}

        sub_tasks_raw = data.get("sub_tasks", [])
        return {item.get("task_id", f"task_{i}"): self._deserialize_subtask(item)
                for i, item in enumerate(sub_tasks_raw)}
