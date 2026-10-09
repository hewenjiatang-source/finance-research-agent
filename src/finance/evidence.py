"""
证据账本（Evidence Ledger）

问题：Summarizer 只能看到各子任务的文字输出，看不到原始工具结果；
      如果引用编号是事后统一编的，研究员在子任务里写的 [n] 就对不上。

做法：每次工具调用时立即把「原文」登记进账本，并分配一个全局稳定的整数 ``evidence_id``，
      随工具结果一起返回给模型。研究员 → 合成器 → 对抗环 → 评测全程使用同一套编号：

    研究员:  "FY2023 净销售额 3,832.85 亿美元 [3]"      (3 = sec_facts 返回的 evidence_id)
    评测:    ledger[3].text 里能否找到 383,285 百万 ?    (引用核对 / 数据准确性)

账本只在内存里累积，运行结束后落盘为 ``<report>.evidence.json`` 侧车文件，
评测可以完全离线重放（无需再次联网/调用模型）。
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Evidence", "EvidenceLedger", "LedgerTool", "register_result", "EVIDENCE_SCHEMA"]

EVIDENCE_SCHEMA = "evidence/v1"

# 位置参数 -> 参数名（Blue Agent 会用位置参数调用搜索工具）
_POSITIONAL = {
    "web_search": ["query", "top_n"],
    "browser": ["url", "max_chars"],
    "sec_filing": ["url", "section", "max_chars"],
    "calculator": ["expression"],
    "code_sandbox": ["code", "timeout"],
    "file_reader": ["file_path"],
}


@dataclass
class Evidence:
    id: int
    kind: str  # web_snippet | web_page | sec_filing_index | xbrl_facts | sec_filing_text | computation | local_file
    url: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


class EvidenceLedger:
    """线程安全的证据账本，id 从 1 连续递增（与报告里的 [n] 一一对应）。"""

    def __init__(self, max_text_chars: int = 400_000) -> None:
        self.max_text_chars = max_text_chars
        self._items: list[Evidence] = []
        self._index: dict[str, int] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._index.clear()

    def add(self, kind: str, url: str, title: str, text: str, meta: dict | None = None) -> int:
        """登记一条证据，返回 id。同一来源 + 同一内容重复登记会复用旧 id。"""
        text = (text or "")[: self.max_text_chars]
        key = hashlib.sha1(f"{kind}\x00{url}\x00{text}".encode("utf-8", "ignore")).hexdigest()
        with self._lock:
            if key in self._index:
                return self._index[key]
            eid = len(self._items) + 1
            self._items.append(Evidence(eid, kind, url or "", (title or "").strip()[:300], text, dict(meta or {})))
            self._index[key] = eid
            return eid

    def get(self, eid: int) -> Evidence | None:
        with self._lock:
            return self._items[eid - 1] if 1 <= eid <= len(self._items) else None

    def __len__(self) -> int:
        return len(self._items)

    def to_list(self) -> list[dict]:
        with self._lock:
            return [asdict(e) for e in self._items]

    def to_sources(self, snippet_chars: int = 300) -> list[dict]:
        """兼容旧的 ``ResearchReport.sources`` 结构（位置序号 == evidence id）。"""
        return [
            {"id": e["id"], "url": e["url"], "title": e["title"] or e["kind"],
             "snippet": e["text"][:snippet_chars], "kind": e["kind"]}
            for e in self.to_list()
        ]

    def save(self, path: str | Path, query: str = "") -> str:
        payload = {"schema": EVIDENCE_SCHEMA, "query": query, "evidence": self.to_list()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return str(p)

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data["evidence"] if isinstance(data, dict) else data


# ---------------------------------------------------------------------------
# 工具结果 -> 账本
# ---------------------------------------------------------------------------

def _first_line(text: str, limit: int = 120) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


def _render_filings(res: dict) -> str:
    lines = []
    for f in res.get("filings", []):
        lines.append(
            f"{f.get('form')} | filed {f.get('filed')} | period {f.get('period')} | "
            f"accn {f.get('accession')} | {f.get('url')}"
        )
    return "\n".join(lines)


def _render_facts(res: dict) -> str:
    lines = []
    for f in res.get("facts", []):
        lines.append(
            f"{f.get('metric')} ({f.get('concept')}) | {f.get('period_label')} | "
            f"period {f.get('start') or 'instant'}..{f.get('end')} | "
            f"value {f.get('value'):.0f} {f.get('unit')} ({f.get('display')}) | "
            f"{f.get('form')} filed {f.get('filed')} | accn {f.get('accn')}"
            if isinstance(f.get("value"), (int, float))
            else json.dumps(f, ensure_ascii=False)
        )
    return "\n".join(lines)


def register_result(ledger: EvidenceLedger, tool_name: str, args: dict, result: Any) -> Any:
    """把一次工具调用的结果登记进账本，并把 evidence_id 注入返回值。

    注意：返回的 dict 里 **不使用 "error" 键** —— ResearcherAgent 看到 error 键会直接判整个子任务失败，
    而 SEC 查不到公司之类的错误模型本可以自行换个写法重试。
    """
    if tool_name == "web_search" and isinstance(result, dict):
        items = result.get("results")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("url"):
                    it["evidence_id"] = ledger.add("web_snippet", it["url"], it.get("title", ""), it.get("snippet", ""))
        return result

    if tool_name == "browser":
        if isinstance(result, str):
            if result.startswith("[Browser"):  # 错误/警告
                return result
            url = args.get("url", "")
            eid = ledger.add("web_page", url, _first_line(result), result)
            return {"evidence_id": eid, "url": url, "text": result}
        return result

    if tool_name == "sec_filings" and isinstance(result, dict) and result.get("filings"):
        first = result["filings"][0].get("url", "")
        co = result.get("company", {})
        result["evidence_id"] = ledger.add(
            "sec_filing_index", first, f"SEC filings index — {co.get('name', '')}", _render_filings(result), {"company": co}
        )
        return result

    if tool_name == "sec_facts" and isinstance(result, dict) and result.get("facts"):
        co = result.get("company", {})
        result["evidence_id"] = ledger.add(
            "xbrl_facts", result.get("source_url", ""), f"XBRL company facts — {co.get('name', '')}",
            _render_facts(result), {"company": co},
        )
        return result

    if tool_name == "sec_filing" and isinstance(result, dict) and result.get("text"):
        result["evidence_id"] = ledger.add(
            "sec_filing_text", result.get("url", ""), result.get("title", ""), result["text"],
            {"section": result.get("section"), "truncated": result.get("truncated")},
        )
        return result

    if tool_name == "calculator" and isinstance(result, str):
        if result.startswith("[Calculator Error]"):
            return result
        expr = args.get("expression", "")
        eid = ledger.add("computation", "", f"calc: {expr}"[:120], f"{expr} = {result}")
        return {"evidence_id": eid, "expression": expr, "result": result}

    if tool_name == "code_sandbox" and isinstance(result, dict):
        out = result.get("stdout") or result.get("output") or ""
        if out and not result.get("error"):
            code = args.get("code", "")
            result["evidence_id"] = ledger.add("computation", "", "code_sandbox run", f"{code}\n---\n{out}")
        return result

    if tool_name == "file_reader" and isinstance(result, str) and not result.startswith("[FileReader Error]"):
        path = args.get("file_path", "")
        eid = ledger.add("local_file", path, _first_line(result), result)
        return {"evidence_id": eid, "path": path, "text": result}

    return result


class LedgerTool:
    """包装任意工具：执行后把结果登记进账本，并在结果里带回 evidence_id。

    对 Agent 完全透明：name / description / schema 与被包装工具一致。
    """

    def __init__(self, tool: Any, ledger: EvidenceLedger) -> None:
        self._tool = tool
        self._ledger = ledger
        self.name = tool.name
        self.description = getattr(tool, "description", "")

    def get_openai_tool_schema(self) -> dict:
        return self._tool.get_openai_tool_schema()

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        result = await self._tool.execute(*args, **kwargs)
        named = dict(kwargs)
        for pname, val in zip(_POSITIONAL.get(self.name, []), args):
            named.setdefault(pname, val)
        try:
            return register_result(self._ledger, self.name, named, result)
        except Exception:  # 账本登记失败不能影响研究流程
            return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._tool, item)
