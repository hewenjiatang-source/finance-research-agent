"""
SEC EDGAR toolset (financial-filings research scenario)

  sec_filings : list a company's 10-K / 10-Q / 8-K filings (with filing date, period, accession, document URL)
  sec_facts   : read structured XBRL financial data (revenue, net income, EPS, cash flow, balance sheet ...),
                with form / filed / accession provenance. Suited to "fetching numbers"; less error-prone than copying from the text.
  sec_filing  : read the filing text, extracting a section by Item (e.g. 7=MD&A, 1A=Risk Factors, 8=Financial Statements);
                tables are rendered row by row as "line item | current | prior", which makes numbers easy to check.

Compliance and robustness:
  * SEC requires automated requests to carry contact info in the User-Agent -> env var SEC_USER_AGENT ("name email"), otherwise 403.
  * Rate limit: by default <= ~8 req/s (SEC's limit is 10 req/s); exponential-backoff retries on 429/5xx.
  * ``sec_filing`` may only access sec.gov domains, so it cannot be tricked into fetching arbitrary URLs.
  * On failure the tools return ``{"ok": False, "message": ...}`` instead of ``{"error": ...}``:
    ResearcherAgent fails the whole sub-task when it sees an error key, whereas errors here (company not found, wrong fiscal year)
    can be fixed by the model itself by changing parameters and retrying.
  * The HTTP layer is injectable (``fetcher=``), so tests need no network.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable
from urllib.parse import urlparse

from ..finance.xbrl import METRICS, format_value, resolve_metric, select_period
from ..utils.env_config import get_env

__all__ = [
    "SecClient",
    "SecFilingsTool",
    "SecFactsTool",
    "SecFilingTool",
    "create_sec_tools",
    "html_to_text",
    "extract_section",
]

_ALLOWED_HOSTS = {"www.sec.gov", "sec.gov", "data.sec.gov"}
_DEFAULT_CORE_METRICS = [
    "revenue", "gross_profit", "operating_income", "net_income",
    "eps_diluted", "operating_cash_flow", "total_assets",
]


# ===========================================================================
# HTTP client
# ===========================================================================

class SecClient:
    """SEC client with rate limiting, retries and caching. fetcher(url) -> str is injectable."""

    def __init__(
        self,
        user_agent: str | None = None,
        fetcher: Callable[[str], str] | None = None,
        min_interval: float = 0.125,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.user_agent = user_agent or get_env("SEC_USER_AGENT")
        self._fetcher = fetcher
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._last = 0.0
        self._cache: dict[str, str] = {}

    def configured(self) -> bool:
        return self._fetcher is not None or bool(self.user_agent)

    def get_text(self, url: str) -> str:
        if url in self._cache:
            return self._cache[url]
        if self._fetcher is not None:
            body = self._fetcher(url)
        else:
            body = self._http_get(url)
        self._cache[url] = body
        return body

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_text(url))

    def _http_get(self, url: str) -> str:
        if not self.user_agent:
            raise PermissionError("SEC_USER_AGENT is not configured (format: 'name email'); SEC rejects automated requests without contact info")
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            with self._lock:  # global rate limit
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
            req = urllib.request.Request(
                url, headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    return raw.decode("utf-8", errors="replace")
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last_err  # pragma: no cover

    # ---- business-level helpers ------------------------------------------------------
    def resolve_company(self, query: str) -> dict | None:
        """ticker / CIK / company name -> {cik, ticker, name}."""
        q = (query or "").strip()
        if not q:
            return None
        if q.isdigit():
            cik = int(q)
            name = ""
            try:
                name = self.get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").get("name", "")
            except Exception:
                pass
            return {"cik": cik, "ticker": "", "name": name}

        table = self.get_json("https://www.sec.gov/files/company_tickers.json")
        rows = list(table.values()) if isinstance(table, dict) else list(table)
        up = q.upper()
        for r in rows:
            if str(r.get("ticker", "")).upper() == up:
                return {"cik": int(r["cik_str"]), "ticker": r["ticker"], "name": r.get("title", "")}
        low = q.casefold()
        cands = [r for r in rows if low in str(r.get("title", "")).casefold()]
        if cands:
            r = min(cands, key=lambda r: len(r.get("title", "")))
            return {"cik": int(r["cik_str"]), "ticker": r.get("ticker", ""), "name": r.get("title", "")}
        return None


# ===========================================================================
# HTML -> text (tables rendered row by row)
# ===========================================================================

_ATTACH_LEFT = {")", "%", ")%"}
_ATTACH_RIGHT = {"$", "("}


def _merge_cells(cells: list[str]) -> list[str]:
    out: list[str] = []
    pending = ""
    for c in cells:
        c = c.strip()
        if not c:
            continue
        if c in _ATTACH_RIGHT:
            pending += c
            continue
        if c in _ATTACH_LEFT and out:
            out[-1] += c
            continue
        out.append(pending + c)
        pending = ""
    if pending:
        out.append(pending)
    return out


def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup, NavigableString
    except ImportError:  # fallback: crudely strip tags
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"</(tr|p|div|br|h\d)>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"[ \t\xa0]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(["ix:header"]):  # hidden inline-XBRL header, all noise
        tag.decompose()
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        row = " | ".join(_merge_cells(cells))
        tr.replace_with(NavigableString(f"\n{row}\n" if row else "\n"))
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


_ITEM_RE = re.compile(r"(?im)^\s*item\s+(\d{1,2}[AB]?)\s*[\.\:\-—–]?\s*(.{0,160})$")


def extract_section(text: str, item: str, min_body_chars: int = 1) -> str | None:
    """Extract a section by 10-K Item number (e.g. '7' / '1A' / 'Item 8').

    The table of contents has the same Item headings, but its "body" is the next TOC entry (another Item right after the heading line)
    and is very short. For each candidate start we take text up to the "next different Item heading" and pick the longest, which skips the TOC naturally;
    we also require at least ``min_body_chars`` characters of body after the heading line - when there is only a TOC and no real section, None is returned,
    while genuinely short sections such as "Item 1B. Unresolved Staff Comments / None." can still be found.
    """
    target = re.sub(r"(?i)^item\s*", "", item.strip()).upper().rstrip(".")
    matches = [(m.start(), m.group(1).upper()) for m in _ITEM_RE.finditer(text)]
    best: tuple[int, int] | None = None
    for i, (pos, num) in enumerate(matches):
        if num != target:
            continue
        end = len(text)
        for pos2, num2 in matches[i + 1:]:
            if num2 != target:
                end = pos2
                break
        nl = text.find("\n", pos)
        heading_end = nl if 0 <= nl < end else end
        if len(text[heading_end:end].strip()) < min_body_chars:
            continue  # TOC entry: no body after the heading
        if best is None or (end - pos) > (best[1] - best[0]):
            best = (pos, end)
    if best is None:
        return None
    return text[best[0]:best[1]].strip()


def _keyword_windows(text: str, keywords: list[str], width: int = 1500, top: int = 3) -> str:
    low = text.lower()
    scored: list[tuple[int, int]] = []
    for kw in keywords:
        start = 0
        k = kw.lower()
        while True:
            i = low.find(k, start)
            if i < 0:
                break
            scored.append((i, 1))
            start = i + len(k)
            if len(scored) > 2000:
                break
    if not scored:
        return ""
    # score by window: the more keywords hit in a window the better
    buckets: dict[int, int] = {}
    for pos, w in scored:
        buckets[pos // width] = buckets.get(pos // width, 0) + w
    best = sorted(buckets, key=lambda b: (-buckets[b], b))[:top]
    parts = []
    for b in sorted(best):
        s = max(0, b * width - width // 4)
        parts.append(text[s:s + width + width // 2])
    return "\n...\n".join(parts)


# ===========================================================================
# Tool base class
# ===========================================================================

class _SecTool:
    name = ""
    description = ""
    parameters: dict = {}

    def __init__(self, client: SecClient | None = None) -> None:
        self.client = client or SecClient()

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def _fail(self, message: str) -> dict:
        return {"ok": False, "message": message}

    def _not_configured(self) -> dict:
        return self._fail("SEC_USER_AGENT is not configured: set SEC_USER_AGENT=\"name email\" in .env")


class SecFilingsTool(_SecTool):
    name = "sec_filings"
    description = (
        "List SEC EDGAR filings (10-K annual, 10-Q quarterly, 8-K events, 20-F, etc.) for a US-listed company. "
        "Returns form, filing date, report period end, accession number and the document URL "
        "(read it with sec_filing). Use this first to find which filing contains the numbers you need. "
        "Each result has an evidence_id you can cite."
    )
    parameters = {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Ticker (AAPL), CIK number, or company name"},
            "form_types": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter by form, e.g. ['10-K','10-Q']. Default ['10-K','10-Q']",
            },
            "limit": {"type": "integer", "description": "Max filings to return (default 8)", "default": 8},
        },
        "required": ["company"],
    }

    async def execute(self, company: str, form_types: list[str] | None = None, limit: int = 8) -> dict:
        if not self.client.configured():
            return self._not_configured()
        try:
            return await asyncio.to_thread(self._run, company, form_types, limit)
        except Exception as e:
            return self._fail(f"{type(e).__name__}: {e}")

    def _run(self, company: str, form_types: list[str] | None, limit: int) -> dict:
        info = self.client.resolve_company(company)
        if info is None:
            return self._fail(f"Company '{company}' not found; try a ticker or CIK instead")
        sub = self.client.get_json(f"https://data.sec.gov/submissions/CIK{info['cik']:010d}.json")
        recent = sub.get("filings", {}).get("recent", {})
        forms = {f.upper() for f in (form_types or ["10-K", "10-Q"])}
        out: list[dict] = []
        n = len(recent.get("form", []))
        for i in range(n):
            form = recent["form"][i]
            if form.upper() not in forms:
                continue
            accn = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            out.append(
                {
                    "form": form,
                    "filed": recent["filingDate"][i],
                    "period": recent.get("reportDate", [""] * n)[i],
                    "accession": accn,
                    "url": f"https://www.sec.gov/Archives/edgar/data/{info['cik']}/{accn.replace('-', '')}/{doc}",
                    "description": recent.get("primaryDocDescription", [""] * n)[i],
                }
            )
            if len(out) >= max(1, int(limit)):
                break
        if not out:
            return self._fail(f"{info['name']} has no {sorted(forms)} among its recent filings (older filings are not in the recent list)")
        return {"ok": True, "company": {**info, "name": sub.get("name", info["name"])}, "filings": out}


class SecFactsTool(_SecTool):
    name = "sec_facts"
    description = (
        "Get structured XBRL financial data for a US-listed company straight from SEC filings: "
        "current period and prior-year comparative values, each with form/filed/accession provenance. "
        "PREFER THIS over reading filing text for numbers. "
        "Valid metrics: " + ", ".join(METRICS) + ". "
        "fiscal_period is FY (annual, from 10-K) or Q1/Q2/Q3 (single quarter, from 10-Q). "
        "Values are in raw units (USD, USD/shares, shares); the 'display' field is human-readable. "
        "Result has an evidence_id you can cite."
    )
    parameters = {
        "type": "object",
        "properties": {
            "company": {"type": "string", "description": "Ticker, CIK, or company name"},
            "fiscal_year": {"type": "integer", "description": "Fiscal year as labeled in the filing (e.g. 2023)"},
            "fiscal_period": {"type": "string", "enum": ["FY", "Q1", "Q2", "Q3"], "default": "FY"},
            "metrics": {
                "type": "array", "items": {"type": "string"},
                "description": "Metric names; default = revenue, gross_profit, operating_income, net_income, "
                               "eps_diluted, operating_cash_flow, total_assets",
            },
        },
        "required": ["company", "fiscal_year"],
    }

    async def execute(
        self, company: str, fiscal_year: int, fiscal_period: str = "FY", metrics: list[str] | None = None
    ) -> dict:
        if not self.client.configured():
            return self._not_configured()
        try:
            return await asyncio.to_thread(self._run, company, int(fiscal_year), fiscal_period, metrics)
        except Exception as e:
            return self._fail(f"{type(e).__name__}: {e}")

    def _run(self, company: str, fy: int, fp: str, metrics: list[str] | None) -> dict:
        info = self.client.resolve_company(company)
        if info is None:
            return self._fail(f"Company '{company}' not found; try a ticker or CIK instead")
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{info['cik']:010d}.json"
        cf = self.client.get_json(url)

        wanted: list[str] = []
        unknown: list[str] = []
        for m in metrics or _DEFAULT_CORE_METRICS:
            key = resolve_metric(m)
            (wanted if key else unknown).append(key or m)

        facts: list[dict] = []
        missing: list[str] = []
        for key in dict.fromkeys(wanted):
            vals = select_period(cf, key, fy, fp)
            if not vals:
                missing.append(key)
                continue
            for v in vals:
                d = v.to_dict()
                d["display"] = format_value(v.value, v.unit)
                facts.append(d)
        if not facts:
            return self._fail(
                f"{info['name']} FY{fy} {fp} returned no metrics (the fiscal-year label may differ; try fiscal_year±1, "
                f"or read the text with sec_filings + sec_filing). unknown_metrics={unknown}"
            )
        res: dict[str, Any] = {
            "ok": True,
            "company": {**info, "name": cf.get("entityName", info["name"])},
            "source_url": url,
            "facts": facts,
            "missing_metrics": missing,
        }
        if unknown:
            res["unknown_metrics"] = unknown
        return res


class SecFilingTool(_SecTool):
    name = "sec_filing"
    description = (
        "Read the text of an SEC filing document (URL from sec_filings). Tables are rendered row-by-row "
        "as 'line item | current | prior'. Optionally extract one section with `section` "
        "(10-K items: '1' Business, '1A' Risk Factors, '7' MD&A, '7A' Market Risk, '8' Financial Statements) "
        "or pull the passages most relevant to `find` keywords. Only sec.gov URLs are allowed. "
        "Result has an evidence_id you can cite."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Filing document URL on sec.gov"},
            "section": {"type": "string", "description": "10-K/10-Q item number, e.g. '7' or '1A'"},
            "find": {"type": "array", "items": {"type": "string"}, "description": "Keywords to locate passages"},
            "max_chars": {"type": "integer", "description": "Max characters returned (default 20000)", "default": 20000},
        },
        "required": ["url"],
    }

    async def execute(
        self, url: str, section: str | None = None, find: list[str] | None = None, max_chars: int = 20000
    ) -> dict:
        host = (urlparse(url).hostname or "").lower()
        if not url.startswith("https://") or host not in _ALLOWED_HOSTS:
            return self._fail("Only filings under https://www.sec.gov/ may be read")
        if not self.client.configured():
            return self._not_configured()
        try:
            return await asyncio.to_thread(self._run, url, section, find, int(max_chars))
        except Exception as e:
            return self._fail(f"{type(e).__name__}: {e}")

    def _run(self, url: str, section: str | None, find: list[str] | None, max_chars: int) -> dict:
        html = self.client.get_text(url)
        text = html_to_text(html)
        total = len(text)
        label = "full document (head)"
        body = text

        if section:
            sec = extract_section(text, section)
            if sec is None:
                return self._fail(
                    f"Could not locate Item {section} in this document ({total} characters). "
                    f"Try find=[keywords], or read the beginning without section."
                )
            body, label = sec, f"Item {section.upper().replace('ITEM', '').strip()}"
        if find:
            win = _keyword_windows(body, find)
            if not win:
                return self._fail(f"No keyword hit in the document: {find}")
            body, label = win, f"{label} / keyword passages {find}"

        truncated = len(body) > max_chars
        if truncated:
            body = body[:max_chars]
        return {
            "ok": True,
            "url": url,
            "title": f"SEC filing {url.rsplit('/', 1)[-1]} — {label}",
            "section": label,
            "chars_total": total,
            "truncated": truncated,
            "text": body,
        }


def create_sec_tools(client: SecClient | None = None) -> list[_SecTool]:
    client = client or SecClient()
    return [SecFilingsTool(client), SecFactsTool(client), SecFilingTool(client)]
