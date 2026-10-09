"""
Evidence Ledger

Problem: the Summarizer only sees the text output of each sub-task, not the raw tool results;
      if citation numbers were assigned after the fact, the [n] a researcher wrote in a sub-task would not line up.

Approach: at every tool call, register the raw output in the ledger at once and assign a globally stable integer ``evidence_id``,
      returned to the model together with the tool result. Researcher -> summarizer -> adversarial loop -> evaluation all use the same numbering:

    researcher:  "FY2023 net sales $383,285 million [3]"      (3 = evidence_id returned by sec_facts)
    evaluation:  can ledger[3].text be found to contain 383,285 million ?    (citation verification / data accuracy)

The ledger accumulates in memory only; at the end of a run it is saved as a ``<report>.evidence.json`` sidecar,
so the evaluation can replay completely offline (no network or model calls needed).
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

# positional argument -> parameter name (the Blue Agent calls search tools with positional arguments)
_POSITIONAL = {
    "web_search": ["query", "top_n"],
    "browser": ["url", "max_chars"],
    "sec_filing": ["url", "section", "max_chars"],
    "calculator": ["expression"],
    "code_sandbox": ["code", "timeout"],
    "file_reader": ["file_path"],
}


def _raw(v) -> str:
    """Full-precision raw value; per-share values must not be rounded to an integer."""
    v = float(v)
    return f"{v:.0f}" if v.is_integer() else f"{v:.6g}"


@dataclass
class Evidence:
    id: int
    kind: str  # web_snippet | web_page | sec_filing_index | xbrl_facts | sec_filing_text | computation | local_file
    url: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)


class EvidenceLedger:
    """Thread-safe evidence ledger; ids increase consecutively from 1 (matching the [n] in the report one-to-one)."""

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
        """Register one piece of evidence and return its id. Registering the same source + same content again reuses the old id."""
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
        """Compatible with the old ``ResearchReport.sources`` structure (positional index == evidence id)."""
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
# Tool result -> ledger
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
            f"value {_raw(f.get('value'))} {f.get('unit')} ({f.get('display')}) | "
            f"{f.get('form')} filed {f.get('filed')} | accn {f.get('accn')}"
            if isinstance(f.get("value"), (int, float))
            else json.dumps(f, ensure_ascii=False)
        )
    return "\n".join(lines)


def register_result(ledger: EvidenceLedger, tool_name: str, args: dict, result: Any) -> Any:
    """Register the result of a tool call in the ledger and inject the evidence_id into the return value.

    Note: the returned dict does **not use an "error" key** — when ResearcherAgent sees an error key it fails the whole sub-task,
    whereas an error such as "company not found in SEC" is something the model can retry with a different spelling.
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
            if result.startswith("[Browser"):  # error / warning
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
    """Wrap any tool: after execution, register the result in the ledger and return the evidence_id in the result.

    Fully transparent to the agent: name / description / schema are identical to the wrapped tool.
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
        except Exception:  # a ledger registration failure must not affect the research flow
            return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._tool, item)
