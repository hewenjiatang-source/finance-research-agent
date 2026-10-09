"""
Researcher agent (ResearcherAgent)

Executes search and analysis SubTasks with a multi-turn tool-calling loop.
A simplified version of project one's ToolAgentLoop:
  - single trajectory, no batching
  - supports 7 tools: web_search, arxiv_reader, code_sandbox, browser,
    file_reader, calculator, notepad
  - LLM calls go through the VLLMPolicy
  - after tool results are written back, it continues automatically until the model stops calling tools or max_turns is reached
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from .base_agent import BaseAgent
from ..orchestrator.schemas import SubTask, AgentResult, AgentStatus
from ..utils.tracing import trace_agent


__all__ = ["ResearcherAgent"]


class ResearcherAgent(BaseAgent):
    """Researcher agent: handles search, analysis and verification tasks.

    Available tools (7):
      - web_search:   web search, returns title / link / snippet
      - browser:      web page reader, opens a URL and extracts the body text
      - arxiv_reader: ArXiv paper metadata retrieval
      - file_reader:  local file reading (.txt/.md/.pdf/.csv/.json/.docx)
      - code_sandbox: Python code sandbox execution
      - calculator:   lightweight math (faster and safer than the sandbox)
      - notepad:      scratch notes (record intermediate conclusions / todos / search strategy)

    Attributes:
        max_turns: maximum number of interaction turns, to prevent infinite loops.
        tool_map: mapping from tool name to tool instance.
    """

    def __init__(
        self,
        name: str,
        policy,
        tools: list | None = None,
        max_turns: int = 10,
    ) -> None:
        super().__init__(name, policy, tools)
        self.max_turns = max_turns
        self.tool_map: dict[str, Any] = {t.name: t for t in (tools or [])}

    @trace_agent(name="researcher.run", tags=["agent", "researcher"])
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """Run a Researcher task.

        Flow:
          1. build the initial system + user messages
          2. loop calling the policy and parsing tool_calls
          3. execute the tools and append the results as tool messages
          4. until there are no tool_calls or max_turns is reached
        """
        trajectory: list[dict] = []
        total_tokens: int = 0

        # build the task description
        task_desc = self._build_task_prompt(task, context)

        # feasibility check: if the task obviously cannot be answered by web search, analyze directly from known information
        if self._is_non_searchable(task, context):
            messages = [
                {"role": "system", "content": self._system_prompt_direct_analysis()},
                {"role": "user", "content": task_desc},
            ]
            try:
                response = self.policy(messages)
                content = response.get("content", "") or ""
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=[{"role": "assistant", "content": content}],
                    token_usage=len(content) // 3,
                    confidence=self._extract_confidence(content),
                )
            except Exception as e:
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=f"Direct analysis failed: {e}",
                    trajectory=[{"error": str(e)}],
                    token_usage=0,
                    confidence=0.0,
                )

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task_desc},
        ]

        # if the policy supports setting tools, register the available tools
        if hasattr(self.policy, "set_tools") and self.tools:
            schemas = [t.get_openai_tool_schema() for t in self.tools]
            self.policy.set_tools(schemas)

        # decide the fallback tool by task type (subclasses may override _fallback_tool)
        fallback_tool = self._fallback_tool(task)

        for turn in range(self.max_turns):
            # Fallback: if last turn had no tool_calls, force a search instruction
            if turn > 0 and messages and messages[-1].get("role") == "assistant":
                last_tool_calls = messages[-1].get("tool_calls", [])
                if not last_tool_calls:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You did not use any tools. "
                            f"You MUST call the '{fallback_tool}' tool now to search for information. "
                            f"Do not write a summary without searching first."
                        ),
                    })

            try:
                # run the synchronous policy in a thread pool, so it does not block the asyncio event loop
                response = await asyncio.to_thread(self.policy, messages)
            except RuntimeError as e:
                # fatal errors such as context length exceeded
                trajectory.append({"turn": turn, "error": str(e)})
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.FAILED,
                    output=str(e),
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=0.0,
                )

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []

            trajectory.append({
                "turn": turn,
                "role": "assistant",
                "content": content,
                "tool_calls": [dict(tc) for tc in tool_calls],
            })

            # estimate tokens (simplified: characters / 3)
            total_tokens += len(json.dumps(messages, ensure_ascii=False)) // 3

            # no tool calls -> task finished
            if not tool_calls:
                # plan B: detect whether the LLM reply is an obvious explanation of a tool failure
                if self._is_tool_failure_explanation(content):
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=content,
                        trajectory=trajectory,
                        token_usage=total_tokens,
                        confidence=0.0,
                    )
                confidence = self._extract_confidence(content)
                return AgentResult(
                    task_id=task.task_id,
                    status=AgentStatus.SUCCESS,
                    output=content,
                    trajectory=trajectory,
                    token_usage=total_tokens,
                    confidence=confidence,
                )

            # execute the tool calls
            tool_results = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                result = await self._execute_tool(tool_name, args)

                # plan B: detect whether the tool result contains an error field
                if isinstance(result, dict) and result.get("error"):
                    error_msg = result["error"]
                    trajectory.append({
                        "turn": turn,
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "name": tool_name,
                        "error": error_msg,
                    })
                    return AgentResult(
                        task_id=task.task_id,
                        status=AgentStatus.FAILED,
                        output=f"Tool '{tool_name}' failed: {error_msg}",
                        trajectory=trajectory,
                        token_usage=total_tokens,
                        confidence=0.0,
                    )

                tool_results.append({
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": result,
                })
                trajectory.append({
                    "turn": turn,
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "name": tool_name,
                    "result": result,
                })

            # detect whether the search results are all empty (the tool returned but with no useful content)
            all_empty = True
            for tr in tool_results:
                if tr["name"] == "web_search":
                    res = tr["result"]
                    if isinstance(res, dict) and res.get("results"):
                        for r in res["results"]:
                            if r.get("snippet", "").strip():
                                all_empty = False
                                break
            
            # if searched 2+ rounds or the results are all empty, force a summary
            search_count = sum(1 for t in trajectory if t.get("role") == "tool" and t.get("name") == "web_search")
            force_summary = False
            if search_count >= 2:
                force_summary = True
            if all_empty and tool_results:
                force_summary = True

            # append the assistant message and tool results to messages
            assistant_msg = {
                "role": "assistant",
                "content": content,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [dict(tc) for tc in tool_calls]
            # keep reasoning_content (DeepSeek reasoning models need it passed back)
            if response.get("reasoning_content"):
                assistant_msg["reasoning_content"] = response["reasoning_content"]
            messages.append(assistant_msg)

            for tr in tool_results:
                msg_content = json.dumps(tr["result"], ensure_ascii=False, default=str)
                # if a summary is forced, append a notice to the tool result
                if force_summary:
                    msg_content += "\n\n[SYSTEM NOTICE] You have already searched enough. Write your final summary NOW. Do NOT call any more tools."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": msg_content,
                })

        # reached max_turns
        return AgentResult(
            task_id=task.task_id,
            status=AgentStatus.TIMEOUT,
            output="Reached max_turns without final answer.",
            trajectory=trajectory,
            token_usage=total_tokens,
            confidence=0.0,
        )

    def _fallback_tool(self, task: SubTask) -> str:
        """Name of the tool the model is forced to use when it called none in a turn."""
        desc_lower = (task.description or "").lower()
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]  # Chinese keywords are intentional: they match Chinese-language tasks
        return "arxiv_reader" if any(kw in desc_lower for kw in academic_keywords) else "web_search"

    def _system_prompt(self) -> str:
        return (
            "You are a meticulous research assistant. "
            "Your job is to gather and analyze information using the RIGHT tool for each task. "
            "\n\nAVAILABLE TOOLS:\n"
            "- web_search: General web search for news, market data, industry reports, current events. "
            "  Use this as the FIRST tool for most tasks.\n"
            "- arxiv_reader: Academic paper search (ArXiv / Semantic Scholar). "
            "  USE when the task involves: papers, publications, academic research, citation counts.\n"
            "- browser: Open a URL and extract full webpage text. "
            "  USE after web_search when search results are too short and you need to read the original article in depth.\n"
            "- code_sandbox: Execute Python code for calculations, data processing, simulations. "
            "  USE when the task requires: computing FLOPs, memory usage, statistical analysis, data transformation.\n"
            "- calculator: Quick math evaluation (+, -, *, /, sqrt, log, mean, etc.). "
            "  USE for simple calculations instead of code_sandbox.\n"
            "- notepad: Write/read intermediate notes to avoid forgetting findings during multi-step research. "
            "  USE to record key numbers, conclusions, or next search queries.\n"
            "- file_reader: Read local files (.txt, .md, .pdf, .csv, .json, .docx). "
            "  USE only when the task explicitly references a local file path.\n"
            "\nIMPORTANT RULES:\n"
            "1. You MUST use a tool to find factual information. Do NOT answer from your own knowledge.\n"
            "2. Choose the RIGHT tool based on the task type. You can use MULTIPLE tools in sequence.\n"
            "3. For most research tasks, START with web_search or arxiv_reader.\n"
            "4. If search results are too short, use browser to read the full article.\n"
            "5. If the task involves numbers/calculations, use calculator or code_sandbox.\n"
            "6. You may call tools AT MOST 2 times total. After that you MUST summarize.\n"
            "7. Only after gathering information, provide a concise summary with a confidence score (0-1).\n"
            "8. NEVER greet the user or ask what they want to search — just execute immediately.\n"
            "9. If you have already performed 2 tool calls, do NOT call more — write the final summary now."
        )

    def _system_prompt_direct_analysis(self) -> str:
        return (
            "You are a thoughtful analyst. "
            "The user has asked a question that cannot be answered by web search "
            "(e.g., analyzing a specific private individual, personal advice, or subjective judgment). "
            "Your job is to provide a reasoned analysis based ONLY on the information already provided in the context. "
            "Do NOT make up facts. Clearly state what is known, what can be reasonably inferred, and what remains unknown. "
            "End with a confidence score (0-1)."
        )

    def _is_non_searchable(self, task: SubTask, context: dict) -> bool:
        """Heuristically decide whether a task cannot be answered by web search."""
        desc = (task.description or "").lower()
        query = context.get("query", "").lower()
        combined = desc + " " + query

        # Pattern 1: analyzing / judging a specific private individual (name + descriptive analysis)
        if "朋友" in combined or "同学" in combined or "同事" in combined:  # Chinese keywords intentional (friend / classmate / colleague)
            if any(w in combined for w in ["分析", "评价", "是什么样", "性格", "人品"]):
                return True

        # Pattern 2: subjective advice (based on personal circumstances)
        if any(w in combined for w in ["建议我", "我该怎么", "适合我吗", "要不要"]):  # Chinese phrases intentional ("advise me", "what should I do", ...)
            if "朋友" in combined or "我" in query:
                return True

        # Pattern 3: clearly personal-privacy analysis
        if "叫" in combined and any(w in combined for w in ["分析", "评价", "是什么样"]):
            return True

        return False

    def _build_task_prompt(self, task: SubTask, context: dict) -> str:
        """Build the user prompt from the SubTask and the global context."""
        desc_lower = (task.description or "").lower()
        
        # smart tool recommendation: match keywords in the task description
        tool_recommendations = []
        
        # academic papers
        academic_keywords = ["论文", "paper", "publication", "学术", "arxiv", "neurips", "icml", "iclr", "scholar", "citation", "文献"]
        if any(kw in desc_lower for kw in academic_keywords):
            tool_recommendations.append("arxiv_reader")
        
        # calculation / math
        calc_keywords = ["计算", "flops", "显存", "内存", "参数量", "延迟", "成本", "公式", "数值", "统计", "数学", "公式", "推导"]
        if any(kw in desc_lower for kw in calc_keywords):
            tool_recommendations.append("calculator")
            tool_recommendations.append("code_sandbox")
        
        # deep reading (needs the original text)
        browser_keywords = ["详细", "原文", "全文", "深度", "详细内容", "网页内容", "文章正文"]
        if any(kw in desc_lower for kw in browser_keywords):
            tool_recommendations.append("browser")
        
        # files
        file_keywords = ["文件", "文档", "dataset", "数据集", "pdf", "csv", "json"]
        if any(kw in desc_lower for kw in file_keywords):
            tool_recommendations.append("file_reader")
        
        # decide the primary tool: academic papers prefer arxiv_reader, everything else starts with web_search
        is_academic = "arxiv_reader" in tool_recommendations
        if is_academic:
            # academic paper task: arxiv_reader first, web_search as a backup
            tool_recommendations = ["arxiv_reader"] + [t for t in tool_recommendations if t != "arxiv_reader"]
        elif not tool_recommendations:
            tool_recommendations.insert(0, "web_search")
        
        primary_tool = tool_recommendations[0]
        secondary_tools = tool_recommendations[1:]
        
        lines = [
            f"## Task: {task.description}",
            f"Type: {task.task_type.value}",
            f"Expected output: {task.expected_type}",
            "",
            f"## RECOMMENDED TOOLS (in priority order): {', '.join(tool_recommendations)}",
        ]
        
        if secondary_tools:
            lines.append(f"Start with '{primary_tool}'. If the task involves numbers/calculations, also use {', '.join(secondary_tools)}.")
        else:
            lines.append(f"Use '{primary_tool}' to gather information.")
        
        lines.extend([
            "",
            "## INSTRUCTIONS:",
            f"1. First, call the '{primary_tool}' tool with a relevant query to gather information.",
            "2. Review the results.",
            f"3. If needed, call '{primary_tool}' ONE MORE time with a refined query.",
            "   You may call tools AT MOST 2 times total. After the 2nd call, you MUST write the final summary.",
            "4. If search results are too short, you may use 'browser' to read the full article (counts as 1 tool call).",
            "5. If calculations are needed, use 'calculator' or 'code_sandbox' (counts as 1 tool call).",
            "6. Finally, summarize your findings in English with a confidence score (0-1).",
            "7. DO NOT greet the user or ask clarifying questions — just execute immediately.",
            "8. IMPORTANT: Your query MUST directly address the task description.",
        ])
        if task.search_hints:
            lines.insert(1, f"Search hints (MUST use these as primary keywords): {', '.join(task.search_hints)}")
        if task.context_keys:
            ctx_parts = []
            for key in task.context_keys:
                if key in context:
                    ctx_parts.append(f"- {key}: {context[key]}")
            if ctx_parts:
                lines.append("\n## Context:")
                lines.extend(ctx_parts)
        return "\n".join(lines)

    async def _execute_tool(self, tool_name: str, args: dict) -> dict:
        """Call the concrete tool instance."""
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return {"error": f"Tool '{tool_name}' not found"}
        try:
            return await tool.execute(**args)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _is_tool_failure_explanation(self, content: str) -> bool:
        """Detect whether an LLM reply is an explanation of a tool failure rather than a real research result.

        Common patterns: quota exhausted, cannot connect, cannot search, etc.
        """
        if not content:
            return False
        c = content.lower()
        failure_keywords = [
            "无法通过", "无法执行", "无法使用", "无法获取", "无法访问",
            "额度已用尽", "配额已用完", "额度已用完", "搜索配额",
            "cannot search", "unable to search", "quota exceeded",
            "api key", "额度不足", "余额不足", "余额为", "余额：0",
            "网络错误", "连接失败", "无法连接到",
        ]
        return any(kw in c for kw in failure_keywords)

    def _extract_confidence(self, content: str) -> float:
        """Try to extract a confidence score from the output text."""
        import re
        # match "Confidence: 0.85" or "置信度: 0.85"
        patterns = [
            r"[Cc]onfidence[:\s]+(0\.\d+|1\.0|1)",
            r"置信度[:\s]+(0\.\d+|1\.0|1)",
        ]
        for pat in patterns:
            m = re.search(pat, content)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        # default to medium confidence
        return 0.6
