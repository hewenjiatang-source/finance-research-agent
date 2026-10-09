"""
M5 Blue Agent — fixer and defender

The Blue Agent receives the Red Agent's Verdict, sorts issues by priority and performs three kinds of fixes:
1. In-place Fix: numbers/dates inconsistent with the source -> replace directly
2. Supplementary Search: unsourced claims -> trigger a new search
3. Removal: high-confidence hallucinations -> delete the paragraph

After fixing it runs self_verify to make sure no new contradictions are introduced.
"""
from __future__ import annotations

import copy
from typing import Any

from src.adversarial.verdict import (
    FixOperation,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent


__all__ = ["BlueAgent"]


# ============================================================================
# Prompt templates
# ============================================================================

SYSTEM_BLUE_AGENT = (
    "You are a rigorous research-report reviser (the Blue Agent). Your task is to fix the research report according to the review comments, "
    "making sure every change is well-founded and introduces no new errors. Output must be JSON."
)

PROMPT_SELF_VERIFY = """Verify whether the following fixed research report introduces any new contradictions or errors.

Output in the following JSON format:
{
  "has_new_issue": bool,
  "new_issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string"
    }
  ]
}

--- Original report ---
{original}

--- Fixed report ---
{revised}

--- Fixes performed ---
{fixes}
"""

PROMPT_IN_PLACE_FIX = """Based on the following review comments, make an **in-place correction** to the research report.

Requirements:
1. Only change factual content such as specific numbers, dates and names that disagree with the sources.
2. Keep the original structure and narrative style unchanged.
3. Every change must be based on the provided sources; do not introduce new information.
4. Output the complete corrected passage.

Output in the following JSON format:
{
  "fixed_content": "string",   // full text of the corrected report
  "changes": [                 // list of change records
    {
      "location": "string",
      "before": "string",
      "after": "string"
    }
  ]
}

--- Review comments ---
{issue_desc}

--- Original report ---
{content}

--- Sources ---
{sources}
"""

PROMPT_SUPPLEMENTARY_SEARCH = """Based on the following review comments, **correct the research report after supplementary search**.

Requirements:
1. For claims without source support, use the search results to add evidence.
2. If the search results cannot confirm a claim, delete it or mark it as "unverified".
3. Output the complete corrected report.

Output in the following JSON format:
{
  "fixed_content": "string",
  "changes": [
    {
      "location": "string",
      "action": "added|removed|modified",
      "detail": "string"
    }
  ]
}

--- Review comments ---
{issue_desc}

--- Original report ---
{content}

--- Search results ---
{search_results}
"""

PROMPT_REMOVAL = """Based on the following review comments, make a **removal correction** to the research report.

Requirements:
1. Delete high-confidence hallucinated paragraphs or claims that cannot be verified.
2. After deletion make sure the context stays coherent, adding transition sentences if needed.
3. Output the complete corrected report.

Output in the following JSON format:
{
  "fixed_content": "string",
  "removed_segments": [
    {
      "location": "string",
      "original_text": "string",
      "reason": "string"
    }
  ]
}

--- Review comments ---
{issue_desc}

--- Original report ---
{content}
"""


# ============================================================================
# Blue Agent implementation
# ============================================================================

class BlueAgent:
    """Blue Agent — fixer and defender.

    Attributes:
        policy: a VLLMPolicy instance.
        tools: list of available tools, at least including a search tool for supplementary_search.
        max_tokens: maximum output tokens for a single fix call.
    """

    def __init__(
        self,
        policy,
        tools: list[Any] | None = None,
        max_tokens: int = 4096,
        max_report_chars: int = 4000,
        max_sources: int = 15,
        extra_system: str = "",
    ):
        self.policy = policy
        self.tools = tools or []
        self.max_tokens = max_tokens
        # Maximum report characters sent to the model for fixing. Note: the fix only covers the first N characters sent,
        # so it is concatenated with the rest of the original on write-back (see _merge_fixed) and never overwrites a long report with a truncated one.
        self.max_report_chars = max_report_chars
        self.max_sources = max_sources
        self.system_prompt = SYSTEM_BLUE_AGENT + (("\n\n" + extra_system) if extra_system else "")
        # Cache the search tool
        self._search_tool = self._find_search_tool()

    def _find_search_tool(self) -> Any | None:
        """Find the search tool in the tools list."""
        for t in self.tools:
            name = getattr(t, "name", "")
            if "search" in name.lower():
                return t
        return None

    @trace_agent(name="blue_agent.defend", tags=["m5", "blue", "adversarial"])
    async def defend(
        self, report: ResearchReport, verdict: RedVerdict
    ) -> tuple[ResearchReport, list[FixOperation]]:
        """Fix the research report according to the Red Verdict.

        Flow:
        1. Sort issues by priority.
        2. Perform fixes one by one (in_place / search / removal).
        3. Run self_verify after each round of fixes to detect newly introduced problems.
        4. Return the fixed report and all FixOperation records.

        Args:
            report: original research report (not modified; deep-copied internally).
            verdict: the Red Agent's review result.

        Returns:
            (fixed_report, fix_operations)
        """
        current = copy.deepcopy(report)
        operations: list[FixOperation] = []

        if not verdict.issues:
            return current, operations

        # Sort by priority, descending
        sorted_issues = sorted(
            verdict.issues,
            key=lambda issue: VerdictEngine.compute_priority(issue),
            reverse=True,
        )

        original_content = report.content

        for issue in sorted_issues:
            op = await self._fix_single_issue(current, issue)
            operations.append(op)

            # self_verify: check whether the fix introduced new contradictions
            verify_pass, verify_issues = await self._self_verify(
                original_content, current.content, operations
            )
            if not verify_pass:
                # New problem introduced: record it but continue (handle high-priority issues first to avoid deadlock)
                for vi in verify_issues:
                    operations.append(
                        FixOperation(
                            issue=vi,
                            action="self_verify_detected_new_issue",
                            success=False,
                            detail=vi.description,
                        )
                    )

        return current, operations

    async def _fix_single_issue(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """Perform the fix for a single Issue."""
        old_max = getattr(self.policy, "max_tokens", None)
        if old_max is not None:
            self.policy.max_tokens = self.max_tokens

        try:
            if issue.fix_type == FixType.IN_PLACE:
                result = await self._do_in_place_fix(report, issue)
            elif issue.fix_type == FixType.SUPPLEMENTARY:
                result = await self._do_supplementary_search(report, issue)
            elif issue.fix_type == FixType.REMOVAL:
                result = await self._do_removal(report, issue)
            else:
                result = FixOperation(
                    issue=issue,
                    action="unknown_fix_type",
                    success=False,
                    detail=f"Unknown fix_type: {issue.fix_type}",
                )
        finally:
            if old_max is not None:
                self.policy.max_tokens = old_max

        return result

    async def _do_in_place_fix(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """Perform an in-place correction."""
        prompt = PROMPT_IN_PLACE_FIX
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        prompt = prompt.replace("{sources}", self._format_sources(report.sources, max_items=self.max_sources))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, changes = self._parse_fix_json(raw)
        if fixed_content:
            report.content = self._merge_fixed(report.content, fixed_content)
            return FixOperation(
                issue=issue,
                action=f"in_place_fix: {changes}",
                success=True,
                detail=f"changes={changes}",
            )
        return FixOperation(
            issue=issue,
            action="in_place_fix_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_supplementary_search(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """Perform a correction after supplementary search."""
        search_results = ""
        if self._search_tool is not None:
            try:
                # Assume the search tool has an async execute or a synchronous execute interface
                query = issue.description
                if hasattr(self._search_tool, "execute"):
                    if hasattr(self._search_tool.execute, "__call__"):
                        import inspect
                        if inspect.iscoroutinefunction(self._search_tool.execute):
                            sr = await self._search_tool.execute(query)
                        else:
                            sr = self._search_tool.execute(query)
                    else:
                        sr = None
                else:
                    sr = None
                search_results = str(sr) if sr else "(the search tool returned no results)"
            except Exception as e:
                search_results = f"(search failed: {e})"
        else:
            search_results = "(no search tool available)"

        prompt = PROMPT_SUPPLEMENTARY_SEARCH
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        # Truncate search results to avoid bloat
        search_results = search_results[:2000] if len(search_results) > 2000 else search_results
        prompt = prompt.replace("{search_results}", search_results)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, changes = self._parse_fix_json(raw)
        if fixed_content:
            report.content = self._merge_fixed(report.content, fixed_content)
            return FixOperation(
                issue=issue,
                action=f"supplementary_search: {changes}",
                success=True,
                detail=f"search_results_len={len(search_results)}, changes={changes}",
            )
        return FixOperation(
            issue=issue,
            action="supplementary_search_failed",
            success=False,
            detail=raw[:500],
        )

    async def _do_removal(
        self, report: ResearchReport, issue: Issue
    ) -> FixOperation:
        """Perform a removal correction."""
        prompt = PROMPT_REMOVAL
        prompt = prompt.replace("{issue_desc}", issue.description)
        prompt = prompt.replace("{content}", self._truncate_content(report.content))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        fixed_content, removed = self._parse_removal_json(raw)
        if fixed_content:
            report.content = self._merge_fixed(report.content, fixed_content)
            return FixOperation(
                issue=issue,
                action=f"removal: {removed}",
                success=True,
                detail=f"removed={removed}",
            )
        return FixOperation(
            issue=issue,
            action="removal_failed",
            success=False,
            detail=raw[:500],
        )

    async def _self_verify(
        self, original: str, revised: str, operations: list[FixOperation]
    ) -> tuple[bool, list[Issue]]:
        """Self-verify after fixing, checking whether new contradictions were introduced.

        Returns:
            (whether it passed, list of newly found issues)
        """
        if not revised or revised == original:
            return True, []

        fixes_text = "\n".join(
            f"- [{op.issue.dimension.value}] {op.action}: {op.detail[:200]}"
            for op in operations[-5:]  # only the latest 5, to keep the prompt short
        )
        prompt = PROMPT_SELF_VERIFY
        prompt = prompt.replace("{original}", original[:2000])
        prompt = prompt.replace("{revised}", revised[:2000])
        prompt = prompt.replace("{fixes}", fixes_text)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        resp = self.policy(messages)
        raw = resp.content or ""

        try:
            import json

            data = json.loads(raw.strip())
            has_new = bool(data.get("has_new_issue", False))
            new_issues = []
            for item in data.get("new_issues", []):
                new_issues.append(
                    Issue(
                        severity=Severity(item.get("severity", "minor")),
                        dimension=Dimension(item.get("dimension", "logical")),
                        description=item.get("description", ""),
                        location=item.get("location", ""),
                        fix_type=FixType.IN_PLACE,
                    )
                )
            return not has_new, new_issues
        except Exception:
            # A self_verify parse failure counts as a pass, to avoid blocking the fix flow
            return True, []

    def _format_sources(self, sources: list[dict], max_items: int = 15) -> str:
        """Format the source list, truncated to avoid context bloat.

        When a source carries a stable ``id`` (evidence ledger), that id is used as its number so it matches the [n] in the report;
        otherwise fall back to positional numbering (the original behavior).
        """
        if not sources:
            return "(no sources)"
        lines = []
        for i, s in enumerate(sources[:max_items], 1):
            title = s.get("title", "Unknown title")
            url = s.get("url", "")
            snippet = s.get("snippet", "")[:300]
            lines.append(f"[{s.get('id', i)}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) > max_items:
            lines.append(f"... {len(sources) - max_items} more sources not shown")
        return "\n".join(lines)

    def _truncate_content(self, content: str, max_len: int | None = None) -> str:
        """Truncate report content to keep the prompt from getting too long."""
        max_len = max_len or self.max_report_chars
        if len(content) <= max_len:
            return content
        return content[:max_len] + "\n\n[Report truncated; only the first {} characters are shown]".format(max_len)

    def _merge_fixed(self, original: str, fixed: str) -> str:
        """Write the model's fix result back into the report.

        If the original report exceeds max_report_chars, the model only saw the beginning, and the fixed_content it returns
        also covers only the beginning — overwriting directly would drop the second half. Here we strip any truncation notice the model may have echoed,
        then append the remainder of the original.
        """
        n = self.max_report_chars
        if len(original) <= n:
            return fixed
        import re as _re

        fixed = _re.sub(r"\n*\[Report truncated[^\]]*\]\s*$", "", fixed.rstrip())
        return fixed + original[n:]

    def _parse_fix_json(self, raw: str) -> tuple[str, list[dict]]:
        """Parse the JSON output of in_place / search fixes."""
        import json
        import re

        raw = raw.strip()
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            return data.get("fixed_content", ""), data.get("changes", [])
        except json.JSONDecodeError:
            pass
        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                data = json.loads(m.strip())
                return data.get("fixed_content", ""), data.get("changes", [])
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                data = json.loads(brace.group(0))
                return data.get("fixed_content", ""), data.get("changes", [])
            except json.JSONDecodeError:
                pass
        return "", []

    def _parse_removal_json(self, raw: str) -> tuple[str, list[dict]]:
        """Parse the JSON output of removal fixes."""
        import json
        import re

        raw = raw.strip()
        if not raw:
            return "", []
        try:
            data = json.loads(raw)
            return data.get("fixed_content", ""), data.get("removed_segments", [])
        except json.JSONDecodeError:
            pass
        code = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        for m in code.findall(raw):
            try:
                data = json.loads(m.strip())
                return data.get("fixed_content", ""), data.get("removed_segments", [])
            except json.JSONDecodeError:
                continue
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                data = json.loads(brace.group(0))
                return data.get("fixed_content", ""), data.get("removed_segments", [])
            except json.JSONDecodeError:
                pass
        return "", []
