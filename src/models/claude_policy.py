"""
Claude Policy — wrapper for the Anthropic Messages API

Keeps the same external contract as VLLMPolicy, so upper-layer modules (Planner / Researcher / Summarizer /
Red-Blue / Compressor / Judge) can switch to Claude without any change:

  - ``__call__(messages) -> OpenAICompatibleDict``  (role / content / tool_calls)
  - ``set_tools(tools)``  accepts OpenAI function-calling schemas
  - ``tools`` / ``was_truncated`` attributes
  - context overflow raises ``RuntimeError("[CONTEXT_LENGTH_EXCEEDED] ...")``

Internally converts between the OpenAI message format and the Anthropic Messages format:

  * system messages are pulled out into the top-level ``system`` parameter
  * assistant.tool_calls  -> ``tool_use`` content blocks
  * role=tool messages    -> ``tool_result`` blocks; several results of one turn are merged into **one** user message
  * repairs orphan / missing tool_result (common after truncation or retries, would cause a 400)
  * shapes the API rejects: empty text blocks, first message not user, last message assistant, etc.

Design trade-offs (all deliberate):
  1. The ``anthropic`` SDK is lazily imported and can also be injected via ``client=`` (tests use a fake client).
  2. Only ``temperature`` is sent, ``top_p`` is not sent by default: some models do not allow both.
     If the API returns 400 because of sampling parameters, retry once without them and remember that.
  3. Non-retryable deterministic errors (401/403/404) raise RuntimeError directly, instead of
     returning a "fake assistant" and spinning like the old backend; transient errors (rate limit / overload / network) keep the old behavior of returning
     an ``Error: ...`` message so the upper layer can carry on. The SDK has its own exponential-backoff retries (max_retries).
  4. If the model returns thinking blocks, they are serialized verbatim into ``reasoning_content`` (with a prefix);
     on the next turn they are restored to full content blocks, so multi-turn thinking with tools does not 400.
     ResearcherAgent already passes reasoning_content through to the message history, so nothing needs to change.
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

# thinking-block carrier prefix: placed in reasoning_content to travel across turns
_CARRIER_PREFIX = "__claude_blocks__:"
_PLACEHOLDER = "(no content)"
_ID_BAD_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
# above this max_tokens, switch to streaming to avoid the SDK's limit on very long non-streaming requests
_STREAM_THRESHOLD = 16384


# ===========================================================================
# Format conversion (pure functions, easy to unit test)
# ===========================================================================

def _clean_id(raw: Any, fallback: str) -> str:
    s = _ID_BAD_CHARS.sub("_", str(raw or "")).strip("_")
    return s or fallback


def _block_to_dict(block: Any) -> dict:
    """Convert a content block returned by the SDK (pydantic object or dict) into a serializable dict."""
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
    """Normalize messages of various shapes passed by the caller into a list of OpenAI-style dicts.

    Keeps VLLMPolicy's tolerance: supports ``("observation", text)`` tuples and
    filters out leaked ``task=Task(`` environment content.
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
    """OpenAI function schema -> Anthropic tools. Schemas already in Anthropic format pass through unchanged."""
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
    """OpenAI-style messages -> (system, anthropic messages).

    Input must already have gone through ``_normalize_messages``. The output satisfies the structural constraints of the Messages API:
      * the first message is user, user/assistant strictly alternate (adjacent same-role messages are merged)
      * every tool_use has exactly one matching tool_result in the user message right after it, placed first
      * no empty text blocks; the last message is not assistant (avoids prefill, which newer models do not support)
    """
    system_parts: list[str] = []
    turns: list[dict] = []  # {"role": ..., "content": [blocks]}
    pending_ids: list[str] = []  # tool_use ids in the previous assistant message not yet claimed by a tool message
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
                tid = pending_ids[0]  # when the id is missing, claim in order
            if tid in pending_ids:
                pending_ids.remove(tid)
            body = content if content.strip() else _PLACEHOLDER
            push("user", [{"type": "tool_result", "tool_use_id": tid, "content": body}])
            continue

        # user and other unknown roles
        push("user", [_text_block(content)])

    # ---- repair tool_use / tool_result pairing ----
    repaired: list[dict] = []
    for i, turn in enumerate(turns):
        if turn["role"] == "user":
            prev_ids: list[str] = []
            if repaired and repaired[-1]["role"] == "assistant":
                prev_ids = [b["id"] for b in repaired[-1]["content"] if b.get("type") == "tool_use"]
            results = [b for b in turn["content"] if b.get("type") == "tool_result"]
            others = [b for b in turn["content"] if b.get("type") != "tool_result"]
            # drop orphan / duplicate tool_results
            seen: set[str] = set()
            kept: list[dict] = []
            for r in results:
                rid = r.get("tool_use_id")
                if rid in prev_ids and rid not in seen:
                    kept.append(r)
                    seen.add(rid)
            # add a placeholder for missing results, keeping tool_use order
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

        # assistant has tool_use but no user message follows -> add one
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
    """Claude backend policy (Anthropic Messages API)."""

    # reuse the old backend's "drop old turns, keep system + recent interaction" truncation logic (depends only on self.was_truncated)
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
            logger.debug("ClaudePolicy ignoring unused parameters: %s", sorted(ignored))
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
            except ImportError as e:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The Claude backend needs the anthropic SDK: pip install anthropic"
                ) from e
            kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": timeout}
            if api_key:
                kwargs["api_key"] = api_key  # if not provided, the SDK reads ANTHROPIC_API_KEY
            if base_url:
                kwargs["base_url"] = base_url
            raw_client = anthropic.Anthropic(**kwargs)
            from ..utils.tracing import maybe_wrap_anthropic_client

            client = maybe_wrap_anthropic_client(raw_client)
        self.client = client

    # ------------------------------------------------------------------
    def set_tools(self, tools: list[dict]) -> None:
        """Register available tools (OpenAI function-calling schema, converted internally)."""
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
            req["tool_choice"] = {"type": "auto"}  # any/tool return 400 on newer models, use auto only
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
        tools = self.tools  # read once, to avoid concurrent modification
        req = self._build_request(system, msgs, tools)

        try:
            try:
                resp = self._send(req)
            except Exception as e:
                if self._is_sampling_error(e) and not self._omit_sampling:
                    logger.warning("Model rejected sampling parameters; retrying without temperature/top_p: %s", e)
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

            # transient error (the SDK already retried with backoff per max_retries): keep the old backend's behavior
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
            logger.warning("Claude output hit max_tokens=%s; content may be truncated", self.max_tokens)

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
