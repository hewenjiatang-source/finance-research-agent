"""Unit tests for ClaudePolicy / ModelRouter (no network and no anthropic SDK installation required).

Run: python -m unittest discover -s tests -p "test_*.py" -v
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.claude_policy import (  # noqa: E402
    ClaudePolicy,
    _CARRIER_PREFIX,
    _normalize_messages,
    to_anthropic_messages,
    to_anthropic_tools,
)


# --------------------------------------------------------------------------
# fake client
# --------------------------------------------------------------------------
class APIError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


def text_resp(text="ok", stop="end_turn", **usage):
    return NS(
        content=[NS(type="text", text=text)],
        stop_reason=stop,
        usage=NS(input_tokens=usage.get("i", 10), output_tokens=usage.get("o", 5)),
    )


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.stream_calls: list[dict] = []

    def _next(self):
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def create(self, **kw):
        self.calls.append(kw)
        return self._next()

    def stream(self, **kw):
        self.stream_calls.append(kw)
        outer = self

        class Ctx:
            def __enter__(self_inner):
                return NS(get_final_message=outer._next)

            def __exit__(self_inner, *a):
                return False

        return Ctx()


def make_policy(responses, **kw):
    fm = FakeMessages(responses)
    pol = ClaudePolicy(client=NS(messages=fm), **kw)
    return pol, fm


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search the web",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def tc(id_, name, args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class TestConversion(unittest.TestCase):
    def conv(self, msgs):
        return to_anthropic_messages(_normalize_messages(msgs))

    def test_system_extracted_and_user_kept(self):
        system, msgs = self.conv(
            [{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"}, {"role": "user", "content": "hi"}]
        )
        self.assertEqual(system, "S1\n\nS2")
        self.assertEqual(msgs, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])

    def test_tools_converted(self):
        out = to_anthropic_tools([TOOL_SCHEMA])
        self.assertEqual(out[0]["name"], "web_search")
        self.assertEqual(out[0]["input_schema"]["required"], ["query"])
        # Already in Anthropic format: pass through unchanged
        native = {"name": "x", "description": "d", "input_schema": {"type": "object", "properties": {}}}
        self.assertEqual(to_anthropic_tools([native]), [native])
        self.assertEqual(to_anthropic_tools(None), [])

    def test_parallel_tool_results_merge_into_single_user_message(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "let me search", "tool_calls": [tc("a", "web_search", {"query": "q1"}), tc("b", "web_search", {"query": "q2"})]},
            {"role": "tool", "tool_call_id": "a", "content": "R1"},
            {"role": "tool", "tool_call_id": "b", "content": "R2"},
            {"role": "user", "content": "now summarize"},
        ]
        _, out = self.conv(msgs)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual([b["type"] for b in out[1]["content"]], ["text", "tool_use", "tool_use"])
        self.assertEqual(out[1]["content"][1]["input"], {"query": "q1"})
        last = out[2]["content"]
        self.assertEqual([b["type"] for b in last], ["tool_result", "tool_result", "text"])
        self.assertEqual([b["tool_use_id"] for b in last[:2]], ["a", "b"])

    def test_tool_result_ordering_fixed_when_user_text_precedes(self):
        # User text appears before the tool message -> tool_result must still come first
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [tc("a", "web_search", {"query": "q"})]},
            {"role": "user", "content": "extra"},
            {"role": "tool", "tool_call_id": "a", "content": "R"},
        ]
        _, out = self.conv(msgs)
        self.assertEqual([b["type"] for b in out[2]["content"]], ["tool_result", "text"])

    def test_orphan_tool_result_dropped_and_missing_result_synthesized(self):
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "tool", "tool_call_id": "ghost", "content": "orphan"},
            {"role": "assistant", "content": "", "tool_calls": [tc("a", "web_search", {}), tc("b", "web_search", {})]},
            {"role": "tool", "tool_call_id": "b", "content": "only b"},
        ]
        _, out = self.conv(msgs)
        # orphan is dropped, user text is kept
        self.assertEqual(out[0]["content"], [{"type": "text", "text": "task"}])
        results = out[2]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["a", "b"])
        self.assertTrue(results[0].get("is_error"))
        self.assertEqual(results[1]["content"], "only b")

    def test_dangling_tool_use_at_end_gets_results_appended(self):
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "", "tool_calls": [tc("a", "web_search", {})]},
        ]
        _, out = self.conv(msgs)
        self.assertEqual(out[-1]["role"], "user")
        self.assertEqual(out[-1]["content"][0]["tool_use_id"], "a")

    def test_id_less_tool_messages_claim_pending_ids_in_order(self):
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "", "tool_calls": [tc("a", "f", {}), tc("b", "f", {})]},
            {"role": "tool", "content": "R1"},
            {"role": "tool", "content": "R2"},
        ]
        _, out = self.conv(msgs)
        self.assertEqual([(r["tool_use_id"], r["content"]) for r in out[2]["content"]], [("a", "R1"), ("b", "R2")])

    def test_ids_sanitized_consistently(self):
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "", "tool_calls": [tc("call.1/x", "f", {})]},
            {"role": "tool", "tool_call_id": "call.1/x", "content": "R"},
        ]
        _, out = self.conv(msgs)
        self.assertEqual(out[1]["content"][0]["id"], out[2]["content"][0]["tool_use_id"])
        self.assertRegex(out[1]["content"][0]["id"], r"^[a-zA-Z0-9_-]+$")

    def test_empty_blocks_get_placeholder(self):
        msgs = [{"role": "user", "content": "   "}, {"role": "assistant", "content": ""}, {"role": "user", "content": "x"}]
        _, out = self.conv(msgs)
        for m in out:
            for b in m["content"]:
                if b["type"] == "text":
                    self.assertTrue(b["text"].strip())

    def test_first_assistant_and_trailing_assistant_repaired(self):
        _, out = self.conv([{"role": "assistant", "content": "hello"}])
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual(out[-1]["content"][0]["text"], "Please continue.")

    def test_tuple_messages_and_task_leak_filtered(self):
        msgs = [("observation", "obs text"), ("assistant", "reply"), {"role": "user", "content": "leak task=Task(id=1)"}]
        _, out = self.conv(msgs)
        self.assertEqual(out[0]["content"][0]["text"], "obs text")
        flat = json.dumps(out)
        self.assertNotIn("task=Task(", flat)

    def test_thinking_carrier_roundtrip(self):
        blocks = [
            {"type": "thinking", "thinking": "hmm", "signature": "sig123"},
            {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "q"}},
        ]
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "", "reasoning_content": _CARRIER_PREFIX + json.dumps(blocks)},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "R"},
        ]
        _, out = self.conv(msgs)
        self.assertEqual(out[1]["content"], blocks)  # restored as-is, including signature
        self.assertEqual(out[2]["content"][0]["tool_use_id"], "toolu_1")


class TestPolicyCall(unittest.TestCase):
    def test_basic_request_shape(self):
        pol, fm = make_policy([text_resp("hello")], model_name="claude-sonnet-5-5", temperature=0.3, top_p=0.9, max_tokens=777)
        out = pol([{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}])
        req = fm.calls[0]
        self.assertEqual(out["content"], "hello")
        self.assertEqual(out.content, "hello")  # attribute-style access (OpenAICompatibleDict contract)
        self.assertEqual(out["tool_calls"], [])
        self.assertEqual(req["model"], "claude-sonnet-5-5")
        self.assertEqual(req["system"], "sys")
        self.assertEqual(req["temperature"], 0.3)
        self.assertEqual(req["max_tokens"], 777)
        self.assertNotIn("top_p", req)  # top_p is not sent by default
        self.assertNotIn("tools", req)

    def test_top_p_only_when_explicitly_enabled(self):
        pol, fm = make_policy([text_resp()], top_p=0.9, send_top_p=True)
        pol([{"role": "user", "content": "q"}])
        self.assertEqual(fm.calls[0]["top_p"], 0.9)

    def test_tool_use_response_mapped_to_openai_tool_calls(self):
        resp = NS(
            content=[
                NS(type="text", text="searching"),
                NS(type="tool_use", id="toolu_9", name="web_search", input={"query": "AAPL 10-K"}),
            ],
            stop_reason="tool_use",
            usage=NS(input_tokens=1, output_tokens=1),
        )
        pol, fm = make_policy([resp])
        pol.set_tools([TOOL_SCHEMA])
        out = pol([{"role": "user", "content": "go"}])
        req = fm.calls[0]
        self.assertEqual(req["tools"][0]["name"], "web_search")
        self.assertEqual(req["tool_choice"], {"type": "auto"})
        call = out["tool_calls"][0]
        self.assertEqual(call["id"], "toolu_9")
        self.assertEqual(call["function"]["name"], "web_search")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "AAPL 10-K"})
        self.assertEqual(out["content"], "searching")

    def test_summarizer_disables_tools_by_assigning_none(self):
        pol, fm = make_policy([text_resp()])
        pol.set_tools([TOOL_SCHEMA])
        pol.tools = None  # what SummarizerAgent does
        pol([{"role": "user", "content": "q"}])
        self.assertNotIn("tools", fm.calls[0])

    def test_thinking_blocks_exposed_via_reasoning_content(self):
        resp = NS(
            content=[
                NS(type="thinking", thinking="plan", signature="s1"),
                NS(type="tool_use", id="t1", name="web_search", input={"query": "x"}),
            ],
            stop_reason="tool_use",
            usage=NS(input_tokens=1, output_tokens=1),
        )
        pol, _ = make_policy([resp])
        out = pol([{"role": "user", "content": "q"}])
        rc = out["reasoning_content"]
        self.assertTrue(rc.startswith(_CARRIER_PREFIX))
        restored = json.loads(rc[len(_CARRIER_PREFIX):])
        self.assertEqual(restored[0]["signature"], "s1")

    def test_usage_accumulates(self):
        pol, _ = make_policy([text_resp(i=100, o=20), text_resp(i=50, o=10)])
        pol([{"role": "user", "content": "a"}])
        pol([{"role": "user", "content": "b"}])
        self.assertEqual(pol.usage["input_tokens"], 150)
        self.assertEqual(pol.usage["output_tokens"], 30)
        self.assertEqual(pol.usage["calls"], 2)

    def test_sampling_param_rejection_retries_without_and_remembers(self):
        err = APIError("temperature is not supported for this model", 400)
        pol, fm = make_policy([err, text_resp("fine"), text_resp("again")], temperature=0.5)
        out = pol([{"role": "user", "content": "q"}])
        self.assertEqual(out["content"], "fine")
        self.assertIn("temperature", fm.calls[0])
        self.assertNotIn("temperature", fm.calls[1])
        pol([{"role": "user", "content": "q2"}])
        self.assertNotIn("temperature", fm.calls[2])  # already remembered

    def test_context_overflow_raises_runtime_error(self):
        pol, _ = make_policy([APIError("prompt is too long: 1200000 tokens > 1000000 maximum", 400)])
        with self.assertRaises(RuntimeError) as cm:
            pol([{"role": "user", "content": "q"}])
        self.assertIn("[CONTEXT_LENGTH_EXCEEDED]", str(cm.exception))

    def test_stop_reason_context_exceeded_raises(self):
        pol, _ = make_policy([text_resp("x", stop="model_context_window_exceeded")])
        with self.assertRaises(RuntimeError):
            pol([{"role": "user", "content": "q"}])

    def test_auth_error_is_fatal_not_swallowed(self):
        pol, _ = make_policy([APIError("invalid x-api-key", 401)])
        with self.assertRaises(RuntimeError) as cm:
            pol([{"role": "user", "content": "q"}])
        self.assertIn("CLAUDE_API_FATAL", str(cm.exception))

    def test_transient_error_returns_error_assistant(self):
        pol, _ = make_policy([APIError("overloaded", 529)])
        out = pol([{"role": "user", "content": "q"}])
        self.assertTrue(out["content"].startswith("Error:"))
        self.assertEqual(out["tool_calls"], [])

    def test_large_max_tokens_uses_streaming(self):
        pol, fm = make_policy([text_resp("long")], max_tokens=32000)
        out = pol([{"role": "user", "content": "q"}])
        self.assertEqual(out["content"], "long")
        self.assertEqual(len(fm.stream_calls), 1)
        self.assertEqual(len(fm.calls), 0)

    def test_empty_response_gets_marker_text(self):
        pol, _ = make_policy([NS(content=[], stop_reason="refusal", usage=None)])
        out = pol([{"role": "user", "content": "q"}])
        self.assertIn("refusal", out["content"])

    def test_old_turns_dropped_when_input_too_long(self):
        pol, fm = make_policy([text_resp()], max_input_chars=2000)
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(10):
            msgs.append({"role": "user", "content": f"u{i} " + "x" * 400})
            msgs.append({"role": "assistant", "content": f"a{i} " + "y" * 400})
        msgs.append({"role": "user", "content": "final question"})
        pol(msgs)
        self.assertTrue(pol.was_truncated)
        sent = json.dumps(fm.calls[0]["messages"])
        self.assertIn("final question", sent)
        self.assertNotIn("u0 ", sent)


class TestRouter(unittest.TestCase):
    """Verify the env -> ClaudePolicy mapping using a fake anthropic module."""

    def setUp(self):
        self.created: list[dict] = []
        outer = self

        class FakeAnthropic:
            def __init__(self, **kw):
                outer.created.append(kw)
                self.messages = FakeMessages([text_resp()])

        self._saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=FakeAnthropic)
        self._env = {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "ANTHROPIC_BASE_URL")}
        import src.utils.env_config as ec

        ec._ENV_LOADED = True  # do not read the local .env
        from src.models.model_router import ModelRouter

        ModelRouter.clear_cache()
        self.Router = ModelRouter

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = self._saved
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.Router.clear_cache()

    def test_claude_backend_from_env(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        os.environ["CLAUDE_MODEL"] = "claude-haiku-5-5"
        pol = self.Router.create_backend("claude", temperature=0.1, max_tokens=123, top_p=1.0)
        self.assertIsInstance(pol, ClaudePolicy)
        self.assertEqual(pol.model_name, "claude-haiku-5-5")
        self.assertEqual(pol.max_tokens, 123)
        self.assertEqual(self.created[0]["api_key"], "sk-test")

    def test_module_level_model_override(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        pol = self.Router.create_backend("claude", model_name="claude-opus-5-5")
        self.assertEqual(pol.model_name, "claude-opus-5-5")

    def test_anthropic_alias_and_cache(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        a = self.Router.create_backend("anthropic", temperature=0.2, tools=None)
        b = self.Router.create_backend("anthropic", temperature=0.2, tools=None)
        self.assertIs(a, b)

    def test_missing_key_raises_value_error(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("CLAUDE_API_KEY", None)
        os.environ.pop("ANTHROPIC_BASE_URL", None)
        with self.assertRaises(ValueError):
            self.Router.create_backend("claude")


if __name__ == "__main__":
    unittest.main()
