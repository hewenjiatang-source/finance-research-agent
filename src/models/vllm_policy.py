"""
VLLM Policy — OpenAI API wrapper

Reuses the project-one implementation, adding from __future__ import annotations to keep Python 3.10+ compatible.
The interface stays exactly the same:
  - __call__(messages) -> OpenAICompatibleDict
  - set_tools(tools)
  - _truncate_messages(messages, max_chars)
"""
from __future__ import annotations

import json
import re
from typing import Optional


__all__ = ["VLLMPolicy", "OpenAICompatibleDict"]


# regex: extract tool instructions when Qwen emits chatter outside the tags
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# [quality filter] the assistant must never output these template markers
# if detected, the whole trajectory is marked contaminated (reusing the was_truncated channel)
FORBIDDEN_TEMPLATE_TOKENS = ["</tool_response>", "<tool_response>"]


# universal compatibility class: lets a dict support .content and .tool_calls access
class OpenAICompatibleDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class VLLMPolicy:
    """VLLM Policy: wraps OpenAI-compatible APIs (vLLM / OpenAI).

    Core capabilities:
      - message format cleaning and merging (prevents vLLM 400)
      - proactive truncation (keep system + recent interaction, drop old turns)
      - tool-call parsing (native + regex fallback)
      - error classification (context overflow raises, other errors return a fake assistant)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        tools: Optional[list[dict]] = None,
    ):
        from openai import OpenAI  # lazy import: openai only needs to be installed when using OpenAI-compatible backends

        raw_client = OpenAI(base_url=base_url, api_key=api_key)
        # if LangSmith tracing is on, automatically wrap the client to trace all LLM calls
        from ..utils.tracing import maybe_wrap_openai_client
        self.client = maybe_wrap_openai_client(raw_client)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tools = tools
        # [contamination flag] once proactive truncation has happened, the whole trajectory is void
        self.was_truncated = False

    def set_tools(self, tools: list[dict]) -> None:
        """Register available tools (OpenAI function calling schema)."""
        self.tools = tools

    def _truncate_messages(self, messages: list, max_chars: int = 35000) -> list:
        """Proactive truncation: keep system + recent interaction, gradually dropping old turns.

        Threshold 35000 chars ≈ 11-12K content tokens (ratio 2.5-3.0 + overhead + tool metadata).
        Truncation means "drop old turns" rather than cutting content, to avoid slicing a message in the middle and breaking its meaning.
        """
        system_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        other_msgs = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]

        def _count_chars(msgs):
            total = 0
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                # 1. content character count
                total += len(str(m.get("content", "")))
                # 2. arguments + name in the assistant message's tool_calls (a big token consumer that was previously missed)
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        func = tc.get("function", {})
                        total += len(str(func.get("arguments", "")))
                        total += len(str(func.get("name", "")))
                # 3. tool message metadata (short but counted too)
                if m.get("role") == "tool":
                    total += len(str(m.get("tool_call_id", "")))
                    total += len(str(m.get("name", "")))
            return total

        before_chars = _count_chars(messages)
        if before_chars <= max_chars:
            return messages

        self.was_truncated = True
        print(f"[TRUNCATE] Triggered: {before_chars} chars > {max_chars} threshold. n_msgs={len(messages)}")
        print(f"[TRUNCATE] System msgs: {len(system_msgs)}, Other msgs: {len(other_msgs)}")

        # Strategy: drop old messages starting from the head of other_msgs, keep the recent interaction
        # but always keep at least system + the last 3 (otherwise context is lost entirely)
        # Key: an assistant(tool_calls) must not be separated from the tool messages right after it
        kept = list(other_msgs)
        while len(kept) > 3:
            removed = kept.pop(0)
            # if an assistant with tool_calls is dropped, the consecutive tool messages after it must be dropped too
            if isinstance(removed, dict) and removed.get("role") == "assistant" and removed.get("tool_calls"):
                while kept and isinstance(kept[0], dict) and kept[0].get("role") == "tool":
                    kept.pop(0)
            after_chars = _count_chars(system_msgs + kept)
            if after_chars <= max_chars:
                print(f"[TRUNCATE] Reduced to {after_chars} chars, kept {len(kept)} non-system msgs")
                return system_msgs + kept

        # Extreme case: still over the threshold even with only system + the last 3 kept
        # apply content-level truncation to the last message (the newest interaction) as a fallback
        after_chars = _count_chars(system_msgs + kept)
        if after_chars > max_chars and kept:
            # truncate the content of the last message (usually a very long tool result)
            last_msg = kept[-1]
            excess = after_chars - max_chars
            content = str(last_msg.get("content", ""))
            new_len = max(len(content) - excess - 100, 500)  # leave a 100-char buffer, keep at least 500
            last_msg["content"] = content[:new_len] + "\n[CONTENT_TRUNCATED]"
            final_chars = _count_chars(system_msgs + kept)
            print(f"[TRUNCATE] Content-truncated last msg to {new_len} chars. Final: {final_chars}")
            return system_msgs + kept

        return system_msgs + kept

    def __call__(self, messages: list) -> OpenAICompatibleDict:
        """Call the LLM and return an OpenAI-compatible message.

        Args:
            messages: list of OpenAI-format messages.

        Returns:
            OpenAICompatibleDict: with the role, content and tool_calls fields.
        """
        # 1. deep-clean the message format
        sanitized = []
        for m in messages:
            role, content = "user", ""
            if isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content", "")
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                # fix the core error: handle tuples like ['observation', '...']
                role = "user" if m[0] in ["observation", "user"] else "assistant"
                content = str(m[1])

            # filter out leaked internal Task object info so it does not disturb the model
            if "task=Task(" in str(content):
                continue

            new_msg = {"role": role, "content": str(content)}
            # keep assistant tool_calls and tool metadata, otherwise vLLM returns 400
            if role == "assistant" and m.get("tool_calls"):
                new_msg["tool_calls"] = m["tool_calls"]
            # keep reasoning_content (needed by DeepSeek reasoning models)
            if role == "assistant" and m.get("reasoning_content"):
                new_msg["reasoning_content"] = m["reasoning_content"]
            if role == "tool":
                new_msg["tool_call_id"] = m.get("tool_call_id", "")
                new_msg["name"] = m.get("name", "")

            # merge consecutive same-role messages to prevent vLLM 400 errors
            # but messages carrying tool_calls / tool_call_id cannot be merged, or those fields would be lost
            can_merge = (
                sanitized
                and sanitized[-1]["role"] == role
                and role in ("user", "assistant")
                and "tool_calls" not in sanitized[-1]
                and "tool_calls" not in new_msg
                and "tool_call_id" not in new_msg
            )
            if can_merge:
                sanitized[-1]["content"] += "\n" + str(content)
            else:
                sanitized.append(new_msg)

        # 2. proactive truncation (quality filter under the 16K constraint)
        # threshold 12-13K content tokens ≈ 40000 chars (ratio 2.8-3.2 + overhead)
        sanitized = self._truncate_messages(sanitized, max_chars=35000)

        # 3. send the request
        kwargs = dict(
            model=self.model_name,
            messages=sanitized,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        try:
            resp = self.client.chat.completions.create(**kwargs)
            raw_msg = resp.choices[0].message
            content = raw_msg.content or ""

            # 4. [FORBIDDEN] detect whether the assistant output template markers it should not
            for forbidden in FORBIDDEN_TEMPLATE_TOKENS:
                if forbidden in content:
                    print(f"[FORBIDDEN] Detected '{forbidden}' in assistant content, marking trajectory as contaminated")
                    self.was_truncated = True
                    break

            # 5. parse tool calls (with regex fallback)
            final_tool_calls = []
            if raw_msg.tool_calls:
                for tc in raw_msg.tool_calls:
                    final_tool_calls.append(OpenAICompatibleDict(
                        id=tc.id, type="function",
                        function=OpenAICompatibleDict(name=tc.function.name, arguments=tc.function.arguments)
                    ))
            elif "<tool_call>" in content:
                matches = TOOL_CALL_PATTERN.findall(content)
                for i, m_str in enumerate(matches):
                    try:
                        d = json.loads(m_str.strip())
                        final_tool_calls.append(OpenAICompatibleDict(
                            id=f"manual_{i}", type="function",
                            function=OpenAICompatibleDict(name=d.get("name"), arguments=json.dumps(d.get("arguments", {})))
                        ))
                    except Exception:
                        continue

            # 6. return the universal object
            result = OpenAICompatibleDict(role="assistant", content=content, tool_calls=final_tool_calls)
            if getattr(raw_msg, "reasoning_content", None):
                result["reasoning_content"] = raw_msg.reasoning_content
            return result

        except Exception as e:
            err_str = str(e)
            err_lower = err_str.lower()
            print(f"Policy Error: {err_str}")

            # context overflow: deterministic error, abort the trajectory at once (do not waste more sampling)
            if "maximum context length" in err_lower or "context length" in err_lower:
                n_msgs = len(messages)
                total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
                raise RuntimeError(
                    f"[CONTEXT_LENGTH_EXCEEDED] n_msgs={n_msgs}, est_chars={total_chars}: {err_str}"
                ) from e

            # other errors (network jitter, vLLM temporarily busy, etc.): return a fake assistant so the trajectory can continue
            return OpenAICompatibleDict(
                role="assistant",
                content=f"Error: {err_str}",
                tool_calls=[]
            )
