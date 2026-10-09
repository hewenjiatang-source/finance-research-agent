"""
Web search tool — multi-backend: SerpAPI / Bing / Bocha AI / Metaso AI

Rationale:
  Switch backends via SEARCH_BACKEND in .env with zero source changes.

Backend comparison:
  - serpapi: 100 free searches per month, most complete results (Google data), reachable in mainland China
  - bing:    Microsoft search API, stable in mainland China, needs an Azure subscription key
  - bocha:   Bocha AI search, most complete mainland-China index, optimized for AI Agents
  - metaso:  Metaso AI search, strong Chinese semantics, has a multi-round research mode
"""
from __future__ import annotations

import asyncio
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from ..utils.env_config import get_env

__all__ = ["WebSearchTool", "MockWebSearchTool", "BaseWebSearchTool"]


class BaseWebSearchTool(ABC):
    """Abstract base class of the web search tool."""

    name: str = "web_search"
    description: str = (
        "Search the web for information. "
        "Supports SerpAPI / Bing / Bocha AI (bocha) / Metaso AI (metaso) backends. "
        "Input: {'query': str, 'top_n': int(optional, default=5)}. "
        "Output: list of {'title': str, 'url': str, 'snippet': str}."
    )

    @abstractmethod
    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        """Run a search and return results."""
        pass

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"},
                        "top_n": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }


class MockWebSearchTool(BaseWebSearchTool):
    """Mock search tool: for tests and demos without a network."""

    def __init__(self, delay_ms: tuple[int, int] = (50, 200)) -> None:
        self.delay_ms = delay_ms

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000.0)

        query_lower = query.lower()
        mock_db: dict[str, list[dict]] = {
            "transformer": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "snippet": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                },
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "url": "https://arxiv.org/abs/1810.04805",
                    "snippet": "BERT obtains new state-of-the-art results on eleven natural language processing tasks.",
                },
            ],
            "llm": [
                {
                    "title": "Large Language Models: A Survey",
                    "url": "https://arxiv.org/abs/2303.18223",
                    "snippet": "This survey reviews the recent advances in large language models, including pre-training, adaptation, and applications.",
                },
            ],
            "python": [
                {
                    "title": "Python Documentation",
                    "url": "https://docs.python.org/3/",
                    "snippet": "Official Python programming language documentation.",
                },
            ],
        }

        results: list[dict] = []
        for keyword, entries in mock_db.items():
            if keyword in query_lower:
                results.extend(entries)

        seen = set()
        unique = []
        for r in results:
            key = r["url"]
            if key not in seen:
                seen.add(key)
                unique.append(r)
        results = unique[:top_n]

        if not results:
            results = [
                {
                    "title": f"Mock result for '{query}'",
                    "url": "https://example.com/mock",
                    "snippet": "This is a mock search result for testing purposes.",
                }
            ]

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }


class WebSearchTool(BaseWebSearchTool):
    """Real web search tool: SerpAPI and Bing Search API backends.

    Configuration is read from .env / .env.local first:
      - SEARCH_BACKEND: backend choice, one of "serpapi" | "bing" (default serpapi)
      - SERPAPI_KEY / SERPAPI_ENDPOINT: SerpAPI configuration
      - BING_SEARCH_KEY / BING_SEARCH_ENDPOINT: Bing API configuration
    """

    _session: aiohttp.ClientSession | None = None

    def __init__(self, backend: str | None = None, api_key: str | None = None, api_endpoint: str | None = None) -> None:
        self.backend = (backend or get_env("SEARCH_BACKEND", "serpapi")).lower().strip()

        # SerpAPI configuration
        self.serpapi_key = api_key or get_env("SERPAPI_KEY")
        self.serpapi_endpoint = api_endpoint or get_env("SERPAPI_ENDPOINT", "https://serpapi.com/search")

        # Bing API configuration
        self.bing_key = api_key or get_env("BING_SEARCH_KEY")
        self.bing_endpoint = api_endpoint or get_env("BING_SEARCH_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search")

        # Bocha AI configuration
        self.bocha_key = api_key or get_env("BOCHA_API_KEY")
        self.bocha_endpoint = api_endpoint or get_env("BOCHA_API_ENDPOINT", "https://api.bochaai.com/v1/web-search")

        # Metaso AI configuration
        self.metaso_key = api_key or get_env("METASO_API_KEY")
        self.metaso_endpoint = api_endpoint or get_env("METASO_API_ENDPOINT", "https://metaso.cn/api/open/search/v2")

    def _get_session(self) -> aiohttp.ClientSession:
        """Get the reused ClientSession, avoiding a new connection for every search."""
        if WebSearchTool._session is None or WebSearchTool._session.closed:
            WebSearchTool._session = aiohttp.ClientSession(
                headers={"Accept-Encoding": "gzip, deflate"}
            )
        return WebSearchTool._session

    @classmethod
    async def close_session(cls) -> None:
        """Close the class-level shared session. Should be called before program exit."""
        if cls._session is not None and not cls._session.closed:
            await cls._session.close()
            cls._session = None

    def __del__(self):
        """On destruction try to close the session (fallback for synchronous environments)."""
        if WebSearchTool._session is not None and not WebSearchTool._session.closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.close_session())
            except RuntimeError:
                # No running event loop, ignore
                pass

    async def execute(self, query: str, top_n: int = 5) -> dict[str, Any]:
        if self.backend == "bing":
            return await self._bing_execute(query, top_n)
        if self.backend == "bocha":
            return await self._bocha_execute(query, top_n)
        if self.backend == "metaso":
            return await self._metaso_execute(query, top_n)
        return await self._serpapi_execute(query, top_n)

    async def _serpapi_execute(self, query: str, top_n: int) -> dict[str, Any]:
        if not self.serpapi_key:
            raise RuntimeError(
                "WebSearchTool (serpapi backend) requires an API key.\n"
                "Set SERPAPI_KEY in .env or .env.local,\n"
                "or pass it to the constructor: WebSearchTool(api_key='your_key')\n"
                "For mock mode, explicitly use MockWebSearchTool()"
            )

        params = {
            "q": query,
            "num": top_n,
            "api_key": self.serpapi_key,
            "engine": "google",
            "gl": "us",
            "hl": "en",
        }

        try:
            session = self._get_session()
            async with session.get(
                self.serpapi_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("error", f"HTTP {resp.status}")
                        return {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": f"SerpAPI error: {error_msg}",
                        }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"SerpAPI network error: {e}",
            }

        # Parse the SerpAPI response
        organic = data.get("organic_results", [])
        results = []
        for item in organic[:top_n]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "serpapi",
        }

    async def _bing_execute(self, query: str, top_n: int) -> dict[str, Any]:
        if not self.bing_key:
            raise RuntimeError(
                "WebSearchTool (bing backend) requires an API key.\n"
                "Set BING_SEARCH_KEY in .env or .env.local,\n"
                "or create a Bing Search v7 resource in the Azure Portal to get a key.\n"
                "For mock mode, explicitly use MockWebSearchTool()"
            )

        headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
        params = {"q": query, "count": top_n, "mkt": "en-US"}

        try:
            session = self._get_session()
            async with session.get(
                self.bing_endpoint,
                params=params,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                    data = await resp.json()
                    if resp.status != 200:
                        error_msg = data.get("message", f"HTTP {resp.status}")
                        return {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": f"Bing API error: {error_msg}",
                        }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"Bing API network error: {e}",
            }

        # Parse the Bing response
        web_pages = data.get("webPages", {}).get("value", [])
        results = []
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bing",
        }

    async def _bocha_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Bocha AI search backend.

        Docs: https://open.bochaai.com
        Features: most complete mainland-China web index, optimized for AI Agents and RAG, returns structured summaries.
        """
        if not self.bocha_key:
            raise RuntimeError(
                "WebSearchTool (bocha backend) requires an API key.\n"
                "Set BOCHA_API_KEY in .env or .env.local,\n"
                "or register at https://open.bochaai.com to get one.\n"
                "For mock mode, explicitly use MockWebSearchTool()"
            )

        payload = {
            "query": query,
            "summary": True,
            "freshness": "noLimit",
            "count": top_n,
        }
        headers = {
            "Authorization": f"Bearer {self.bocha_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.bocha_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    error_msg = data.get("message", f"HTTP {resp.status}")
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "error": f"Bocha AI error: {error_msg}",
                    }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"Bocha AI network error: {e}",
            }

        # Parse the Bocha response — compatible with the structures returned by both the web-search and ai-search endpoints
        results: list[dict] = []

        # Structure A: /v1/web-search -> data.webPages.value[]
        web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
        for item in web_pages[:top_n]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        # Structure B: /v1/ai-search -> data.messages[] content contains citations
        if not results:
            messages = data.get("data", {}).get("messages", [])
            for msg in messages[:top_n]:
                content = msg.get("content", "")
                if content:
                    results.append({
                        "title": msg.get("role", "Citation")[:30],
                        "url": "",
                        "snippet": content[:500],
                    })

        # De-duplicate: different URLs of the same article (mobile/PC/reposts) would be treated as separate results
        results = self._deduplicate_results(results)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "bocha",
        }

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """De-duplicate search results based on the normalized URL and cleaned title."""
        from urllib.parse import urlparse
        import re

        seen_keys: set[str] = set()
        unique: list[dict] = []

        for r in results:
            raw_url = r.get("url", "")
            raw_title = r.get("title", "").strip()

            # --- URL normalization ---
            try:
                parsed = urlparse(raw_url)
                netloc = parsed.netloc.lower()
                path = parsed.path.lower().rstrip("/")

                # Strip the mobile prefix
                for prefix in ("m.", "wap.", "mobile.", "app."):
                    if netloc.startswith(prefix):
                        netloc = netloc[len(prefix):]
                        break
                # Strip the www prefix
                if netloc.startswith("www."):
                    netloc = netloc[4:]

                # For common news/blog sites keep only domain + path (drop query parameters)
                normalized_url = f"{netloc}{path}"
            except Exception:
                normalized_url = raw_url.lower().strip()

            # --- Title cleaning ---
            # Strip common source suffixes such as " - Huxiu", "_CSDN Blog", "| Everyone Is a Product Manager" (the Chinese outlet names in the regex below are intentional)
            cleaned_title = re.sub(
                r"[_\-\s|]*(CSDN博客|虎嗅网|人人都是产品经理|36氪|知乎|搜狐|新浪|网易|腾讯|今日头条|飞书云文档|简书|豆瓣|百度文库|原创力文档|道客巴巴|豆丁网|MBA智库文档|外唐智库|未来智库|中研网|中商产业研究院|三个皮匠报告|book118\.com|doc88\.com|docin\.com|mbalib\.com|askci\.com|chinairn\.com|vzkoo\.com|waitang\.com|sgpjbg\.com|toutiao\.com|sohu\.com|sina\.com|163\.com|qq\.com|ifeng\.com|huxiu\.com|36kr\.com|woshipm\.com|csdn\.net|zhihu\.com|juejin\.cn|segmentfault\.com|cnblogs\.com|简书|知乎专栏|百家号|大鱼号|企鹅号|新浪看点|一点资讯|趣头条|东方财富|雪球|同花顺|财联社|华尔街见闻|界面新闻|澎湃|新京报|南方周末|财新|第一财经|经济观察网|21世纪经济报道|新浪财经|腾讯财经|网易财经|凤凰财经|和讯网|中金在线|东方财富网|中国证券报|上海证券报|证券时报|证券日报|每日经济新闻|第一财经日报|经济参考报|人民日报|新华社|央视新闻|中央广播电视总台|中国日报|环球时报|参考消息|瞭望|半月谈|求是|学习强国|新华网|人民网|中国网|国际在线|中国新闻网|环球网等?)",
                "",
                raw_title,
                flags=re.IGNORECASE,
            ).strip()
            # Then strip trailing " - ", " | ", "_"
            cleaned_title = re.sub(r"[_\-\s|]+$", "", cleaned_title).strip()

            # --- De-dup key: prefer the URL; use the cleaned title when the URL is empty ---
            key = normalized_url if normalized_url else cleaned_title.lower()
            if not key:
                unique.append(r)
                continue

            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(r)

        return unique

    async def _metaso_execute(self, query: str, top_n: int) -> dict[str, Any]:
        """Metaso AI search backend.

        Docs: https://metaso.cn/open
        Features: strong Chinese semantic search, supports detail / concise / research modes.
        """
        if not self.metaso_key:
            raise RuntimeError(
                "WebSearchTool (metaso backend) requires an API key.\n"
                "Set METASO_API_KEY in .env or .env.local,\n"
                "or register at https://metaso.cn/open to get one.\n"
                "For mock mode, explicitly use MockWebSearchTool()"
            )

        payload = {
            "question": query,
            "lang": "zh",
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.metaso_key}",
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            async with session.post(
                self.metaso_endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if resp.status != 200 or data.get("errCode"):
                    error_msg = data.get("errMsg", f"HTTP {resp.status}")
                    return {
                        "query": query,
                        "results": [],
                        "total": 0,
                        "error": f"Metaso AI error: {error_msg}",
                    }
        except Exception as e:
            return {
                "query": query,
                "results": [],
                "total": 0,
                "error": f"Metaso AI network error: {e}",
            }

        # Parse the Metaso response
        results: list[dict] = []
        result_data = data.get("data", {})

        # 1. Prefer the text field (the full answer compiled by Metaso AI) as a high-value result
        text = result_data.get("text", "")
        if text:
            results.append({
                "title": "Metaso AI search summary",
                "url": "",
                "snippet": text[:1500],  # give enough context for the LLM to summarize directly
            })

        # 2. Append the reference list (for provenance)
        refs = result_data.get("references", [])
        for item in refs[:top_n]:
            snippet_parts = []
            if item.get("title"):
                snippet_parts.append(item["title"])
            if item.get("article_type"):
                snippet_parts.append(f"Type: {item['article_type']}")
            if item.get("date"):
                snippet_parts.append(f"Date: {item['date']}")
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": " | ".join(snippet_parts)[:500],
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "source": "metaso",
        }
