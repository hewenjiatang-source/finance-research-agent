"""Split a Markdown report into the smallest checkable units (sentences / table cells) with their citation ids."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .numbers import CITE_RE, Mention, extract_mentions, find_cites

__all__ = ["Unit", "parse_report", "strip_boilerplate"]

_TAIL_HEADINGS = re.compile(r"^\s{0,3}#{1,6}\s*(元信息|参考来源|参考文献|references|sources|metadata|meta)\b", re.I | re.M)
_TITLE_RE = re.compile(r"^# .*\n", re.M)
_SENT_END = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s)")
_LEAD_CITES = re.compile(r"^\s*((?:\[\d{1,4}\]\s*)+)")


@dataclass
class Unit:
    text: str
    cites: list[int]
    mentions: list[Mention] = field(default_factory=list)
    table: bool = False
    row: str = ""
    col: str = ""
    heading: str = ""


def strip_boilerplate(report: str) -> str:
    """Drop the metadata / reference sections after the body (system-generated, not claims made by the model)."""
    m = _TAIL_HEADINGS.search(report)
    body = report[: m.start()] if m else report
    body = re.sub(r"\n-{3,}\s*$", "", body.rstrip())
    return body


def _split_sentences(line: str) -> list[str]:
    parts = [p for p in _SENT_END.split(line) if p and p.strip()]
    merged: list[str] = []
    for p in parts:
        lead = _LEAD_CITES.match(p)
        if lead and merged:  # a citation after "。" (e.g. "。[3]") belongs to the previous sentence
            merged[-1] = merged[-1].rstrip() + " " + lead.group(1).strip()
            p = p[lead.end():]
            if not p.strip():
                continue
        merged.append(p.strip())
    return merged


_CLAUSE_SPLIT = re.compile(r"(?<=\])\s*[，,;；]\s*(?=\S)")


def _split_clauses(sentence: str) -> list[str]:
    if len(CITE_RE.findall(sentence)) < 2:
        return [sentence]
    parts = _CLAUSE_SPLIT.split(sentence)
    # a lone "[2]" citation group right after a citation is not its own clause (the lookbehind avoids it); merge fragments with no digits and no citation
    out: list[str] = []
    for p in parts:
        if out and not find_cites(p) and not re.search(r"\d", p):
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2 and line.strip().startswith("|")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_report(report: str) -> list[Unit]:
    report = report.replace("\u2212", "-")  # U+2212 MINUS SIGN: without this "−3.36%" would be read as +3.36%
    body = strip_boilerplate(report)
    units: list[Unit] = []
    lines = body.splitlines()
    heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("#"):
            heading = line.strip("# ").strip()
            i += 1
            continue
        if _is_table_row(line):
            block = []
            while i < len(lines) and _is_table_row(lines[i]):
                block.append(lines[i])
                i += 1
            units.extend(_table_units(block, heading))
            continue
        text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line).replace("**", "").strip()
        if text and not re.fullmatch(r"[-=_*\s]{3,}", text):
            for s in _split_sentences(text):
                # Citation scope = the text before the citation group: "…[1], …[2]" within a sentence is split into two clauses checked separately;
                # the clause still takes its metric/period context from the whole sentence (mention.sentence)
                for clause in _split_clauses(s):
                    units.append(Unit(clause, find_cites(clause), extract_mentions(clause, sentence=s), heading=heading))
        i += 1
    for u in units:
        for m in u.mentions:
            m.cites = list(u.cites)
            m.row, m.col = u.row, u.col
    return units


def _table_units(block: list[str], heading: str) -> list[Unit]:
    rows = [_cells(r) for r in block if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
    if len(rows) < 2:
        return []
    header, data = rows[0], rows[1:]
    out: list[Unit] = []
    for r in data:
        label = CITE_RE.sub("", r[0]).replace("**", "").strip()
        row_cites = sorted({c for cell in r for c in find_cites(cell)})
        for j, cell in enumerate(r[1:], 1):
            col = header[j] if j < len(header) else ""
            cites = find_cites(cell) or row_cites
            clean = cell.replace("**", "")
            ms = extract_mentions(clean, sentence=f"{label} | {col} | {clean}")
            if not ms:
                continue
            u = Unit(f"{label} | {col} | {clean}", cites, ms, table=True, row=label, col=col, heading=heading)
            out.append(u)
    return out
