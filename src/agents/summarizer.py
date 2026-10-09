"""
Synthesis agent (SummarizerAgent)

Synthesizes the results of several SubTasks into a structured research report.
Unlike ResearcherAgent's multi-turn tool calling, the Summarizer is a single-turn long-context generation task:
  - sort all sub-results by confidence and concatenate them into the context
  - call the LLM once to generate a Markdown report
  - extract the cited sources and compute the overall confidence
"""
from __future__ import annotations

import json
import re
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus, ResearchReport
from ..utils.tracing import trace_agent


__all__ = ["SummarizerAgent"]


class SummarizerAgent(BaseAgent):
    """Synthesis agent: synthesizes sub-task results into the final research report.

    Attributes:
        max_output_tokens: maximum tokens for report generation (controlled via policy.max_tokens).
    """

    def __init__(self, name: str, policy, tools: list | None = None) -> None:
        super().__init__(name, policy, tools)

    @trace_agent(name="summarizer.run", tags=["agent", "summarizer"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """Run the synthesis task.

        Args:
            task: usually a special "synthesize" task.
            context: global context, must contain the "results" and "query" keys.
                results: list[AgentResult]
                query: str, the original research question

        Returns:
            AgentResult whose output field is a ResearchReport instance.
        """
        query = context.get("query", "")
        results: list[AgentResult] = context.get("results", [])

        if not results:
            report = ResearchReport(
                query=query,
                content="No sub-task results available to synthesize.",
                confidence=0.0,
            )
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=report,
                trajectory=[],
                token_usage=0,
                confidence=0.0,
            )

        # build the synthesis prompt
        prompt = self._build_synthesis_prompt(query, results)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            # synthesis needs no tool calls; temporarily disable tools so the model does not enter tool-calling mode
            old_tools = getattr(self.policy, "tools", None)
            self.policy.tools = None
            response = self.policy(messages)
            self.policy.tools = old_tools
        except RuntimeError as e:
            return AgentResult(
                task_id=task.task_id,
                status=AgentStatus.FAILED,
                output=str(e),
                trajectory=[{"error": str(e)}],
                token_usage=0,
                confidence=0.0,
            )

        content = response.get("content", "") or ""
        token_usage = len(content) // 3  # rough estimate

        # parse the report content, extract sources and confidence
        report = self._parse_report(query, content, results)

        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.SUCCESS,
            output=report,
            trajectory=[{"role": "assistant", "content": content}],
            token_usage=token_usage,
            confidence=report.confidence,
        )

    def _system_prompt(self) -> str:
        return (
            "You are an expert research synthesizer. "
            "Your task is to integrate multiple research findings into a coherent, well-structured report. "
            "Use Markdown formatting. Cite sources explicitly. "
            "The report body MUST be at least 3000 Chinese characters (or 2000 English words) long. "
            "Write in depth: include background, key findings, detailed analysis, comparisons, and implications. "
            "DO NOT describe what you will do — directly output the synthesized report. "
            "At the end, provide an overall confidence score (0-1) and a summary of key sources."
        )

    def _build_synthesis_prompt(self, query: str, results: list[AgentResult]) -> str:
        """Build the synthesis prompt, sorting results by descending confidence."""
        sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)

        parts = [
            f"# Research Question\n{query}\n",
            f"# Sub-task Results ({len(results)} total)\n",
        ]
        for i, r in enumerate(sorted_results, 1):
            status_icon = "✓" if r.status == AgentStatus.SUCCESS else "✗"
            parts.append(
                f"## Result {i} [{status_icon}] (confidence: {r.confidence:.2f})\n"
                f"Task: {r.task_id}\n"
                f"Output:\n{r.output}\n"
            )

        parts.append(
            "\n# Instructions\n"
            "1. Directly write the synthesized report based on the findings above. Do NOT say 'I will synthesize'.\n"
            "2. The report MUST be comprehensive and detailed (at least 3000 Chinese characters or 2000 English words).\n"
            "3. Structure: Executive Summary → Background → Key Findings (with details) → Analysis → Comparisons → Implications → Conclusion.\n"
            "4. Resolve any contradictions between sources.\n"
            "5. Explicitly list all sources cited.\n"
            "6. End with: Overall Confidence: X.XX"
        )
        return "\n".join(parts)

    def _parse_report(self, query: str, content: str, results: list[AgentResult]) -> ResearchReport:
        """Parse a ResearchReport from the LLM output and calibrate confidence by the sub-task success rate."""
        # 1. extract the LLM's self-rated confidence from the text
        llm_confidence = 0.5
        m = re.search(r"[Oo]verall\s+[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)", content)
        if m:
            try:
                llm_confidence = float(m.group(1))
            except ValueError:
                pass

        # 2. compute an objective confidence from the sub-task success rate
        total = len(results)
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)
        success_rate = success / max(total, 1)

        # 3. combined confidence = LLM self-rating x sqrt(success rate) (lowers the weight of the success rate)
        confidence = llm_confidence * (success_rate ** 0.5)
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        # collect sources (extracted from the trajectories of the sub-results)
        sources: list[dict] = []
        for r in results:
            if r.status != AgentStatus.SUCCESS:
                continue
        # simple heuristic: extract urls from the tool results in the trajectory
            for step in r.trajectory:
                if step.get("role") == "tool" and isinstance(step.get("result"), dict):
                    res = step["result"]
                    if "results" in res and isinstance(res["results"], list):
                        for item in res["results"]:
                            if isinstance(item, dict) and "url" in item:
                                sources.append({
                                    "url": item["url"],
                                    "title": item.get("title", ""),
                                    "snippet": item.get("snippet", ""),
                                    "task_id": r.task_id,
                                })
                    elif "papers" in res and isinstance(res["papers"], list):
                        for paper in res["papers"]:
                            if isinstance(paper, dict) and "pdf_url" in paper:
                                sources.append({
                                    "url": paper["pdf_url"],
                                    "title": paper.get("title", ""),
                                    "snippet": paper.get("summary", "")[:200],
                                    "task_id": r.task_id,
                                })

        # deduplicate
        seen = set()
        unique_sources = []
        for s in sources:
            key = s["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        # count the actual tool calls (across the trajectories of all sub-tasks)
        num_searches = sum(
            len([t for t in r.trajectory if t.get("role") == "tool"])
            for r in results
        )

        return ResearchReport(
            query=query,
            content=content,
            sources=unique_sources,
            confidence=confidence,
            num_searches=num_searches,
        )
