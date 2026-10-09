"""
Claude Policy — Anthropic Messages API 封装

与 VLLMPolicy 保持相同的对外契约，使上层模块（Planner / Researcher / Summarizer /
Red-Blue / Compressor / Judge）无需任何改动即可切换到 Claude：

  - ``__call__(messages) -> OpenAICompatibleDict``  （role / content / tool_calls）
  - ``set_tools(tools)``  接受 OpenAI function-calling schema
  - ``tools`` / ``was_truncated`` 属性
  - 上下文超限抛 ``RuntimeError("[CONTEXT_LENGTH_EXCEEDED] ...")``

内部负责 OpenAI 消息格式 <-> Anthropic Messages 格式互转：

  * system 消息抽出为顶层 ``system`` 参数
  * assistant.tool_calls  -> ``tool_use`` 内容块
  * role=tool 消息        -> ``tool_result`` 块，同一轮的多个结果合并进 **一条** user 消息
  * 修复孤儿 tool_result / 缺失 tool_result（截断或重试后常见，会导致 400）
  * 空文本块、首条非 user、末条为 assistant 等 API 不接受的形态

设计取舍（均为刻意为之）:
  1. ``anthropic`` SDK 懒加载，也可以通过 ``client=`` 注入（单测用假 client）。
  2. 只发送 ``temperature``，默认不发送 ``top_p``：部分模型不允许同时指定两者。
     若 API 因采样参数返回 400，自动去掉采样参数重试一次并记住。
  3. 不可重试的确定性错误（401/403/404）直接抛 RuntimeError，不像旧后端那样
     返回"假 assistant"继续空转；瞬时错误（限流/过载/网络）保持旧行为返回
     ``Error: ...`` 让上层有机会继续。SDK 自带指数退避重试（max_retries）。
  4. 若模型返回 thinking 块，原样序列化到 ``reasoning_content``（带前缀），
     下一轮回传时还原为完整 content blocks，保证带工具的多轮思考不会 400。
     ResearcherAgent 已经会把 reasoning_content 透传回消息历史，无需改动。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Optional

from .vllm_policy import OpenAICompatibleDict, VLLMPolicy


__all__ = [
    "ClaudePolicy",
    "to_anthropic_messages",
    "to_anthropic_tools",
    "DEFAULT_CLAUDE_MODEL",
]

logger = logging.getLogger(__name__)

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"

# thinking 块载体前缀：放在 reasoning_content 里跨轮回传
_CARRIER_PREFIX = "__claude_blocks__:"
_PLACEHOLDER = "(no content)"
_ID_BAD_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
# 超过该 max_tokens 改用流式，避免 SDK 对超长非流式请求的限制
_STREAM_THRESHOLD = 16384


# ===========================================================================
# 格式转换（纯函数，便于单测）
# ===========================================================================

def _clean_id(raw: Any, fallback: str) -> str:
    s = _ID_BAD_CHARS.sub("_", str(raw or "")).strip("_")
    return s or fallback


def _block_to_dict(block: Any) -> dict:
    """把 SDK 返回的内容块（pydantic 对象或 dict）转成可序列化 dict。"""
    if isinstance(block, dict):
        return dict(block)
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    return {k: v for k, v in vars(block).items() if not k.startswith("_")}


def _normalize_messages(messages: list) -> list[dict]:
    """把调用方传入的各种形态消息归一为 OpenAI 风格 dict 列表。

    沿用 VLLMPolicy 的容错：支持 ``("observation", text)`` 元组，
    过滤环境泄露的 ``task=Task(`` 内容。
    """
    out: list[dict] = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            extra = m
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            role = "user" if m[0] in ("observation", "user") else "assistant"
            content = str(m[1])
            extra = {}
        else:
            continue

        if "task=Task(" in str(content):
            continue

        norm: dict[str, Any] = {"role": role, "content": "" if content is None else str(content)}
        if role == "assistant":
            if extra.get("tool_calls"):
                norm["tool_calls"] = extra["tool_calls"]
            if extra.get("reasoning_content"):
                norm["reasoning_content"] = extra["reasoning_content"]
        if role == "tool":
            norm["tool_call_id"] = extra.get("tool_call_id", "")
            norm["name"] = extra.get("name", "")
        out.append(norm)
    return out


def to_anthropic_tools(tools: Optional[list[dict]]) -> list[dict]:
    """OpenAI function schema -> Anthropic tools。已是 Anthropic 格式的原样放行。"""
    converted: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if "input_schema" in t and "name" in t:
            converted.append(t)
            continue
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        schema = fn.get("parameters") or {"type": "object", "properties": {}}
        converted.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": schema,
            }
        )
    return converted


def _parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except (TypeError, ValueError):
        return {}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text if text and text.strip() else _PLACEHOLDER}


def to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI 风格消息 -> (system, anthropic messages)。

    输入需已经过 ``_normalize_messages``。输出满足 Messages API 的结构约束：
      * 首条为 user，user/assistant 严格交替（同角色相邻自动合并）
      * 每个 tool_use 在紧随其后的 user 消息里有且仅有一个对应 tool_result，且置于最前
      * 不含空文本块；末条不是 assistant（避免预填充，新模型不支持）
    """
    system_parts: list[str] = []
    turns: list[dict] = []  # {"role": ..., "content": [blocks]}
    pending_ids: list[str] = []  # 上一条 assistant 中尚未被 tool 消息认领的 tool_use id
    auto_idx = 0

    def push(role: str, blocks: list[dict]) -> None:
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""

        if role == "system":
            if content.strip():
                system_parts.append(content)
            continue

        if role == "assistant":
            carrier = m.get("reasoning_content") or ""
            blocks: list[dict] | None = None
            if isinstance(carrier, str) and carrier.startswith(_CARRIER_PREFIX):
                try:
                    restored = json.loads(carrier[len(_CARRIER_PREFIX):])
                    if isinstance(restored, list) and restored:
                        blocks = restored
                except ValueError:
                    blocks = None

            if blocks is None:
                blocks = []
                if content.strip():
                    blocks.append({"type": "text", "text": content})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    auto_idx += 1
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": _clean_id(tc.get("id"), f"call_{auto_idx}"),
                            "name": fn.get("name", ""),
                            "input": _parse_arguments(fn.get("arguments")),
                        }
                    )
                if not blocks:
                    blocks = [_text_block("")]

            pending_ids = [b["id"] for b in blocks if b.get("type") == "tool_use"]
            push("assistant", blocks)
            continue

        if role == "tool":
            tid = _clean_id(m.get("tool_call_id"), "")
            if not tid and pending_ids:
                tid = pending_ids[0]  # 缺 id 时按顺序认领
            if tid in pending_ids:
                pending_ids.remove(tid)
            body = content if content.strip() else _PLACEHOLDER
            push("user", [{"type": "tool_result", "tool_use_id": tid, "content": body}])
            continue

        # user 及其他未知角色
        push("user", [_text_block(content)])

    # ---- 修复 tool_use / tool_result 配对 ----
    repaired: list[dict] = []
    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            prev_ids: list[str] = []
            if repaired and repaired[-1]["role"] == "assistant":
                prev_ids = [b["id"] for b in repaired[-1]["content"] if b.get("type") == "tool_use"]
            results = [b for b in turn["content"] if b.get("type") == "tool_result"]
            others = [b for b in turn["content"] if b.get("type") != "tool_result"]
            # 丢弃孤儿 / 重复的 tool_result
            seen: set[str] = set()
            kept: list[dict] = []
            for r in results:
                rid = r.get("tool_use_id")
                if rid in prev_ids and rid not in seen:
                    kept.append(r)
                    seen.add(rid)
            # 为缺失结果补占位，保持 tool_use 顺序
            ordered: list[dict] = []
            by_id = {r["tool_use_id"]: r for r in kept}
            for pid in prev_ids:
                ordered.append(
                    by_id.get(pid)
                    or {
                        "type": "tool_result",
                        "tool_use_id": pid,
                        "content": "[tool result unavailable]",
                        "is_error": True,
                    }
                )
            new_content = ordered + others
            if not new_content:
                new_content = [_text_block("")]
            repaired.append({"role": "user", "content": new_content})
        else:
            repaired.append(turn)

        # assistant 带 tool_use 但后面没有 user 消息 -> 补一条
        if turn["role"] == "assistant":
            has_next = i + 1 < len(turns)
            tu = [b["id"] for b in turn["content"] if b.get("type") == "tool_use"]
            if tu and not has_next:
                repaired.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": pid,
                                "content": "[tool result unavailable]",
                                "is_error": True,
                            }
                            for pid in tu
                        ],
                    }
                )

    if not repaired:
        repaired = [{"role": "user", "content": [_text_block("")]}]
    if repaired[0]["role"] != "user":
        repaired.insert(0, {"role": "user", "content": [_text_block("(conversation start)")]})
    if repaired[-1]["role"] == "assistant":
        repaired.append({"role": "user", "content": [_text_block("Please continue.")]})

    return "\n\n".join(system_parts), repaired


# ===========================================================================
# Policy
# ===========================================================================

class ClaudePolicy:
    """Claude 后端策略（Anthropic Messages API）。"""

    # 复用旧后端的"丢弃旧轮次、保留 system+最近交互"截断逻辑（只依赖 self.was_truncated）
    _truncate_messages = VLLMPolicy._truncate_messages

    def __init__(
        self,
        model_name: str = DEFAULT_CLAUDE_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
        tools: Optional[list[dict]] = None,
        max_retries: int = 3,
        timeout: float = 600.0,
        max_input_chars: int = 600_000,
        send_top_p: bool = False,
        extra_params: Optional[dict] = None,
        client: Any = None,
        **ignored: Any,
    ) -> None:
        if ignored:
            logger.debug("ClaudePolicy 忽略未使用参数: %s", sorted(ignored))
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.send_top_p = send_top_p
        self.max_tokens = int(max_tokens)
        self.tools = tools
        self.max_input_chars = int(max_input_chars)
        self.extra_params = dict(extra_params or {})
        self.was_truncated = False
        self.last_stop_reason: Optional[str] = None
        self.usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self._lock = threading.Lock()
        self._omit_sampling = False

        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - 取决于环境
                raise ImportError(
                    "使用 Claude 后端需要安装 anthropic SDK: pip install anthropic"
                ) from e
            kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": timeout}
            if api_key:
                kwargs["api_key"] = api_key  # 未提供则由 SDK 读取 ANTHROPIC_API_KEY
            if base_url:
                kwargs["base_url"] = base_url
            raw_client = anthropic.Anthropic(**kwargs)
            from ..utils.tracing import maybe_wrap_anthropic_client

            client = maybe_wrap_anthropic_client(raw_client)
        self.client = client

    # ------------------------------------------------------------------
    def set_tools(self, tools: list[dict]) -> None:
        """注册可用工具（OpenAI function-calling schema，内部转换）。"""
        self.tools = tools

    # ------------------------------------------------------------------
    def _build_request(self, system: str, msgs: list[dict], tools: Optional[list[dict]]) -> dict:
        req: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": msgs,
        }
        if system:
            req["system"] = system
        if not self._omit_sampling:
            if self.temperature is not None:
                req["temperature"] = self.temperature
            if self.send_top_p and self.top_p is not None:
                req["top_p"] = self.top_p
        anth_tools = to_anthropic_tools(tools)
        if anth_tools:
            req["tools"] = anth_tools
            req["tool_choice"] = {"type": "auto"}  # any/tool 在新模型上会 400，只用 auto
        req.update(self.extra_params)
        return req

    def _send(self, req: dict) -> Any:
        if req["max_tokens"] > _STREAM_THRESHOLD and hasattr(self.client.messages, "stream"):
            with self.client.messages.stream(**req) as stream:
                return stream.get_final_message()
        return self.client.messages.create(**req)

    def _record_usage(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        with self._lock:
            self.usage["calls"] += 1
            for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                self.usage[key] += int(getattr(u, key, 0) or 0)

    # ------------------------------------------------------------------
    def __call__(self, messages: list) -> OpenAICompatibleDict:
        norm = _normalize_messages(messages)
        norm = self._truncate_messages(norm, max_chars=self.max_input_chars)
        system, msgs = to_anthropic_messages(norm)
        tools = self.tools  # 读一次，避免并发修改
        req = self._build_request(system, msgs, tools)

        try:
            try:
                resp = self._send(req)
            except Exception as e:
                if self._is_sampling_error(e) and not self._omit_sampling:
                    logger.warning("模型拒绝采样参数，去掉 temperature/top_p 后重试: %s", e)
                    self._omit_sampling = True
                    req.pop("temperature", None)
                    req.pop("top_p", None)
                    resp = self._send(req)
                else:
                    raise
            return self._parse_response(resp)

        except RuntimeError:
            raise
        except Exception as e:
            err = str(e)
            low = err.lower()
            status = getattr(e, "status_code", None)
            logger.error("Claude API error (status=%s): %s", status, err)

            if (
                "prompt is too long" in low
                or "context window" in low
                or "context length" in low
                or status == 413
            ):
                total = sum(len(str(m.get("content", ""))) for m in norm)
                raise RuntimeError(
                    f"[CONTEXT_LENGTH_EXCEEDED] n_msgs={len(norm)}, est_chars={total}: {err}"
                ) from e
            if status in (401, 403, 404):
                raise RuntimeError(f"[CLAUDE_API_FATAL status={status}] {err}") from e

            # 瞬时错误（SDK 已按 max_retries 退避重试过）：保持旧后端行为
            return OpenAICompatibleDict(role="assistant", content=f"Error: {err}", tool_calls=[])

    # ------------------------------------------------------------------
    @staticmethod
    def _is_sampling_error(e: Exception) -> bool:
        if getattr(e, "status_code", None) != 400:
            return False
        low = str(e).lower()
        return "temperature" in low or "top_p" in low

    def _parse_response(self, resp: Any) -> OpenAICompatibleDict:
        self._record_usage(resp)
        stop = getattr(resp, "stop_reason", None)
        self.last_stop_reason = stop

        if stop == "model_context_window_exceeded":
            raise RuntimeError("[CONTEXT_LENGTH_EXCEEDED] model_context_window_exceeded")
        if stop == "max_tokens":
            logger.warning("Claude 输出触达 max_tokens=%s，内容可能被截断", self.max_tokens)

        text_parts: list[str] = []
        tool_calls: list[OpenAICompatibleDict] = []
        raw_blocks: list[dict] = []
        has_thinking = False

        for block in getattr(resp, "content", None) or []:
            bd = _block_to_dict(block)
            btype = bd.get("type")
            raw_blocks.append(bd)
            if btype == "text":
                text_parts.append(bd.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    OpenAICompatibleDict(
                        id=bd.get("id", ""),
                        type="function",
                        function=OpenAICompatibleDict(
                            name=bd.get("name", ""),
                            arguments=json.dumps(bd.get("input", {}), ensure_ascii=False),
                        ),
                    )
                )
            elif btype in ("thinking", "redacted_thinking"):
                has_thinking = True

        content = "".join(text_parts)
        if not content and not tool_calls:
            content = f"[Claude returned no content; stop_reason={stop}]"

        result = OpenAICompatibleDict(role="assistant", content=content, tool_calls=tool_calls)
        if has_thinking:
            result["reasoning_content"] = _CARRIER_PREFIX + json.dumps(raw_blocks, ensure_ascii=False)
        return result
