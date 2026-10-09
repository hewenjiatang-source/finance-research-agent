"""
M5 Red Agent — five-dimension adversarial attacker

The Red Agent's job is to "attack" a research report along several dimensions to find factual errors, hallucinations, logical contradictions,
source-credibility problems and coverage gaps. Each dimension has its own prompt template to keep the evaluation careful and thorough.

Design decisions:
1. Each dimension calls the LLM independently, so one overlong prompt does not scatter the model's attention.
2. Output is required to be structured JSON, lowering the parse-failure rate.
3. A dimension whose output fails to parse gets a conservative score (5.0) and its raw output is recorded, so the adversarial loop does not crash.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.adversarial.verdict import (
    Dimension,
    FixType,
    Issue,
    RedVerdict,
    Severity,
    VerdictEngine,
)
from src.orchestrator.schemas import ResearchReport
from src.utils.tracing import trace_agent


__all__ = ["RedAgent"]


# ============================================================================
# Prompt templates — each dimension is independent, detailed and runnable
# ============================================================================

SYSTEM_RED_AGENT = (
    "You are an extremely strict research-report reviewer (the Red Agent). Your task is to critically and deeply review research reports "
    "and find all factual errors, hallucinations, logical gaps, source defects and coverage gaps. You must score based on objective evidence "
    "and must not relax your standards because a report is well written. Scoring is strict — most research reports default to 5-6 rather than 8-9. "
    "Output must be strict JSON."
)

# --- Dimension 1: fact checking ---
PROMPT_FACTUAL = """Score the following research report on **fact checking**.

Scoring rubric (0-10):
- 10: every verifiable fact (numbers, dates, names, institutions, statistics) is backed by a reliable source and matches it exactly.
- 7-9: a few non-core facts lack a direct source, or there are slight numeric deviations (<5%).
- 4-6: clear factual errors (wrong dates, wrongly cited data) exist but the core argument still holds.
- 1-3: multiple core facts are wrong, seriously damaging the report's credibility.
- 0: a large number of facts are completely wrong; the report is essentially unreliable.

Review requirements:
1. Extract the factual claims (numbers, dates, ratios, rankings, etc.) one by one.
2. Compare each claim against the provided sources.
3. Flag claims that are inconsistent or cannot be verified.

Output in the following JSON format (no extra text):
{
  "score": float,           // 0-10
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",   // specific description of the problem
      "location": "string",      // where the problem is, e.g. "paragraph 3" or a citation mark
      "fix_type": "in_place|search|removal",
      "evidence": "string"       // supporting evidence or the original source text
    }
  ]
}

--- Research report ---
Query: {query}

Content:
{content}

--- Source list ---
{sources}
"""

# --- Dimension 2: hallucination detection ---
PROMPT_HALLUCINATION = """Score the following research report on **hallucination detection**.

Scoring rubric (0-10):
- 10: every piece of information in the report finds clear support in the sources; no hallucination.
- 7-9: a few "reasonable inferences" are not explicitly marked as inferences and may mislead readers.
- 4-6: clear unsourced statements, especially specific numbers, event details or causal relationships.
- 1-3: many paragraphs contain unsourced information; some appears fabricated by the model.
- 0: the report is full of model hallucinations and has almost no credible content.

Review requirements:
1. Check paragraph by paragraph for claims not supported by the sources.
2. Pay special attention to: specific numbers, exact dates, direct quotes, causal relationships, rankings.
3. Distinguish "reasonable inference" from "unsupported assertion": inferences should be clearly marked.

Output in the following JSON format (no extra text):
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- Research report ---
Query: {query}

Content:
{content}

--- Source list ---
{sources}
"""

# --- Dimension 3: logical consistency ---
PROMPT_LOGICAL = """Score the following research report on **logical consistency**.

Scoring rubric (0-10):
- 10: the argument chain is complete, premises and conclusions agree, no contradictory statements.
- 7-9: a few inferences are slightly jumpy but do not affect the overall conclusion.
- 4-6: internal contradictions (e.g. A earlier, not-A later) or causal fallacies exist.
- 1-3: multiple logical breaks and self-contradictions; the core argument is not self-consistent.
- 0: the logic is chaotic and the argument is entirely unreliable.

Review requirements:
1. Check for contradictory statements.
2. Check whether causal relationships are reasonable (avoid post hoc / reversed causation).
3. Check whether inferring a population from a sample is over-generalized.
4. Check whether the baselines in comparative statements are consistent.

Output in the following JSON format (no extra text):
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- Research report ---
Query: {query}

Content:
{content}
"""

# --- Dimension 4: source credibility ---
PROMPT_SOURCE_CREDIBILITY = """Score the **source credibility** of the following research report.

Scoring rubric (0-10):
- 10: all sources are highly authoritative primary materials (government sites, top journals, official financial filings) and timely.
- 7-9: mostly authoritative secondary materials; a few sources are slightly dated but not core data.
- 4-6: low-authority sources (anonymous forums, unverified self-media) mixed in without cross-verification.
- 1-3: relies mainly on low-quality sources, or has circular citations.
- 0: no sources, or the sources are entirely untrustworthy.

Review requirements:
1. Assess each source's domain authority (.gov / .edu / top media / self-media / unknown).
2. Assess the content type (primary data / analytical reporting / editorial / user-generated content).
3. Assess timeliness: in fast-moving fields (technology, stock markets), more than 1 year old is stale.
4. Check primacy: prefer primary data; secondary analysis must cite the original source.

Output in the following JSON format (no extra text):
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- Research report ---
Query: {query}

Content:
{content}

--- Source list ---
{sources}
"""

# --- Dimension 5: coverage completeness ---
PROMPT_COVERAGE = """Score the **coverage completeness** of the following research report.

Scoring rubric (0-10):
- 10: fully covers every sub-topic the query requires, no important omissions, pros and cons presented in balance, and the discussion of each sub-topic rests on relevant search results.
- 7-9: covers the main sub-topics; a few peripheral perspectives are missing but the core conclusion is unaffected. Search results are largely relevant to the query.
- 4-6: omits key sub-topics implied by the query, or presents only one side. Some search results may be irrelevant to the query.
- 1-3: seriously off-topic (e.g. search content unrelated to the query topic) or many sub-topics uncovered.
- 0: does not answer the query at all.

Review requirements:
1. Break the query down into a list of sub-topics that should be covered.
2. Check one by one whether each sub-topic is adequately discussed in the report.
3. Check for obvious bias (presenting only the pro side and ignoring the con side).
4. Check temporal coverage (historical background, current state, future trends, depending on what the query needs).
5. CRITICAL: check whether the report's sources (search sources) are relevant to the query topic. If the sources are all unrelated pages (e.g. searching "internships" returns "technology trends"), flag it as a major/critical issue and state that the search content does not match the query intent.

Output in the following JSON format (no extra text):
{
  "score": float,
  "issues": [
    {
      "severity": "critical|major|minor",
      "description": "string",
      "location": "string",
      "fix_type": "in_place|search|removal",
      "evidence": "string"
    }
  ]
}

--- Original question ---
{query}

--- Research report ---
{content}
"""

# Dimension -> prompt mapping
DIMENSION_PROMPTS: dict[Dimension, str] = {
    Dimension.FACTUAL: PROMPT_FACTUAL,
    Dimension.HALLUCINATION: PROMPT_HALLUCINATION,
    Dimension.LOGICAL: PROMPT_LOGICAL,
    Dimension.SOURCE_CREDIBILITY: PROMPT_SOURCE_CREDIBILITY,
    Dimension.COVERAGE: PROMPT_COVERAGE,
}


# ============================================================================
# Red Agent implementation
# ============================================================================

class RedAgent:
    """Red Agent — five-dimension adversarial attacker.

    Attributes:
        policy: a VLLMPolicy instance providing LLM-call capability.
        max_tokens: maximum output tokens for a single dimension evaluation.
    """

    def __init__(
        self,
        policy,
        max_tokens: int = 2048,
        max_report_chars: int = 4000,
        max_sources: int = 15,
        extra_system: str = "",
    ):
        """Initialize the Red Agent.

        Args:
            policy: any object implementing __call__(messages: list) -> OpenAICompatibleDict.
            max_tokens: maximum output length for each dimension evaluation.
            max_report_chars: maximum report characters submitted for review (default 4000 keeps the original behavior;
                              raise it for financial-filing scenarios, otherwise only the start of the report is reviewed).
            max_sources: upper bound on the number of sources submitted for review.
            extra_system: domain review instructions appended to the system prompt (e.g. rules for checking filing figures).
        """
        self.policy = policy
        self.max_tokens = max_tokens
        self.max_report_chars = max_report_chars
        self.max_sources = max_sources
        self.system_prompt = SYSTEM_RED_AGENT + (("\n\n" + extra_system) if extra_system else "")

    @trace_agent(name="red_agent.attack", tags=["m5", "red", "adversarial"])
    async def attack(self, report: ResearchReport) -> RedVerdict:
        """Run the five-dimension attack on a research report.

        Flow:
        1. Call the five dimension prompts (awaited sequentially but can be gathered externally).
        2. Parse each dimension's JSON output and extract scores and issues.
        3. Aggregate into a RedVerdict.

        Args:
            report: the research report to review.

        Returns:
            RedVerdict: contains the five dimension scores, overall_score and the issues list.
        """
        dimension_scores: dict[Dimension, float] = {}
        all_issues: list[Issue] = []
        raw_feedbacks: list[str] = []

        # Truncate the report to keep a single prompt within the context limit
        limit = self.max_report_chars
        content_truncated = report.content[:limit] if len(report.content) > limit else report.content
        if len(report.content) > limit:
            content_truncated += f"\n\n[Report truncated; only the first {limit} characters are shown]"

        sources_text = self._format_sources(report.sources, max_items=self.max_sources)

        for dim, prompt_template in DIMENSION_PROMPTS.items():
            # Use safe substitution so a { in report.content/sources_text is not misparsed by format
            prompt = prompt_template
            prompt = prompt.replace("{query}", report.query)
            prompt = prompt.replace("{content}", content_truncated)
            prompt = prompt.replace("{sources}", sources_text)
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]

            try:
                # Temporarily raise max_tokens to fit long output
                old_max = getattr(self.policy, "max_tokens", None)
                if old_max is not None:
                    self.policy.max_tokens = self.max_tokens
                resp = self.policy(messages)
                if old_max is not None:
                    self.policy.max_tokens = old_max

                raw = resp.content or ""
                raw_feedbacks.append(f"[{dim.value}]\n{raw}\n")
                score, issues = self._parse_json_output(raw, dim)
                dimension_scores[dim] = score
                all_issues.extend(issues)
            except Exception as e:
                # On parse or call failure return a conservative score so the loop does not crash
                dimension_scores[dim] = 5.0
                raw_feedbacks.append(f"[{dim.value}]\nERROR: {e}\n")
                all_issues.append(
                    Issue(
                        severity=Severity.MINOR,
                        dimension=dim,
                        description=f"Red Agent parse failure: {e}",
                        location="",
                        fix_type=FixType.IN_PLACE,
                    )
                )

        overall = VerdictEngine.compute_overall(dimension_scores)
        return RedVerdict(
            dimension_scores=dimension_scores,
            overall_score=overall,
            issues=all_issues,
            raw_feedback="\n".join(raw_feedbacks),
        )

    def _format_sources(self, sources: list[dict], max_items: int = 15) -> str:
        """Format the source list as text for the prompt. Truncated to avoid context bloat."""
        if not sources:
            return "(no sources)"
        lines = []
        for i, s in enumerate(sources[:max_items], 1):
            title = s.get("title", "Unknown title")
            url = s.get("url", "")
            snippet = s.get("snippet", "")[:300]  # truncate snippet
            lines.append(f"[{s.get('id', i)}] {title}\nURL: {url}\nSnippet: {snippet}\n")
        if len(sources) > max_items:
            lines.append(f"... {len(sources) - max_items} more sources not shown")
        return "\n".join(lines)

    def _parse_json_output(self, raw: str, dimension: Dimension) -> tuple[float, list[Issue]]:
        """Parse the model's JSON output and extract the score and issues.

        Tolerance strategy:
        1. First try to extract the first JSON object from the whole output.
        2. If that fails, try a regex for a ```json ... ``` block.
        3. If that still fails, try to repair common JSON errors (trailing commas, single quotes).
        4. If that still fails, return the conservative score 5.0 and empty issues.

        Args:
            raw: raw model output text.
            dimension: the dimension being parsed, used to build Issues.

        Returns:
            (score, issues_list)
        """
        raw = raw.strip()
        if not raw:
            return 5.0, []

        # Attempt 1: parse the whole text directly
        try:
            data = json.loads(raw)
            return self._extract_from_dict(data, dimension)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract a ```json code block
        code_block_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        matches = code_block_pattern.findall(raw)
        for m in matches:
            try:
                data = json.loads(m.strip())
                return self._extract_from_dict(data, dimension)
            except json.JSONDecodeError:
                continue

        # Attempt 3: extract the first { ... } block (possibly nested)
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return self._extract_from_dict(data, dimension)
            except json.JSONDecodeError:
                pass

        # Attempt 4: retry after repairing common JSON errors
        fixed = self._fix_common_json_errors(raw)
        if fixed:
            for candidate in [fixed, fixed[fixed.find("{"):fixed.rfind("}")+1]]:
                try:
                    data = json.loads(candidate)
                    return self._extract_from_dict(data, dimension)
                except json.JSONDecodeError:
                    continue

        # All failed: return conservatively
        return 5.0, []

    def _fix_common_json_errors(self, raw: str) -> str | None:
        """Repair common JSON format errors."""
        # Extract the outermost braces' content
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = raw[start:end+1]
        
        # Fix 1: remove comments
        text = re.sub(r"//.*?\n", "\n", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        
        # Fix 2: single quotes -> double quotes
        text = text.replace("'", '"')
        
        # Fix 3: remove trailing commas (after the last element of an object/array)
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        
        # Fix 4: fix unescaped newlines inside strings (simple heuristic)
        # A line containing "..." with a newline inside may break parsing
        
        return text

    def _extract_from_dict(self, data: dict, dimension: Dimension) -> tuple[float, list[Issue]]:
        """Extract the score and issues from the parsed dict."""
        score = float(data.get("score", 5.0))
        score = max(0.0, min(10.0, score))

        # Fault-tolerant severity/fix_type mapping (Chinese aliases kept intentionally)
        sev_map = {
            "critical": Severity.CRITICAL, "严重": Severity.CRITICAL,
            "major": Severity.MAJOR, "重要": Severity.MAJOR, "较大": Severity.MAJOR,
            "minor": Severity.MINOR, "轻微": Severity.MINOR, "一般": Severity.MINOR,
        }
        fix_map = {
            "in_place": FixType.IN_PLACE, "就地修复": FixType.IN_PLACE, "修正": FixType.IN_PLACE,
            "search": FixType.SUPPLEMENTARY, "supplementary": FixType.SUPPLEMENTARY, "补充搜索": FixType.SUPPLEMENTARY, "搜索": FixType.SUPPLEMENTARY,
            "removal": FixType.REMOVAL, "删除": FixType.REMOVAL, "移除": FixType.REMOVAL,
        }

        issues: list[Issue] = []
        raw_issues = data.get("issues", [])
        # Tolerate issues being a string rather than a list
        if isinstance(raw_issues, str):
            raw_issues = []
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            try:
                sev_raw = str(item.get("severity", "minor")).lower().strip()
                fix_raw = str(item.get("fix_type", "in_place")).lower().strip()
                sev = sev_map.get(sev_raw, Severity.MINOR)
                fix = fix_map.get(fix_raw, FixType.IN_PLACE)
                issues.append(
                    Issue(
                        severity=sev,
                        dimension=dimension,
                        description=str(item.get("description", "")),
                        location=str(item.get("location", "")),
                        fix_type=fix,
                        evidence=str(item.get("evidence", "")),
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
        return score, issues
