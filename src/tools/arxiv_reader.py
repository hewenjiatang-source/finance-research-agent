"""
Paper reading tool (ArxivReaderTool) — multi-backend: ArXiv / Semantic Scholar / OpenAlex

Rationale:
  The ArXiv API is unreliable from mainland China (frequent timeouts/disconnects).
  The Semantic Scholar API is reachable there and free; a key gives a higher rate limit.
  The OpenAlex API is reachable there, completely free, needs no key, and covers 200M+ papers.
  Switch backends via ARXIV_READER_BACKEND in .env with zero source changes.

Backend comparison:
  - arxiv:              most complete paper library, but needs a VPN in mainland China
  - semantic_scholar:   reachable in mainland China, covers 200M+ papers, includes citation data
  - openalex:           reachable in mainland China, completely free with no key, rich metadata
"""
from __future__ import annotations

import asyncio
import json
import random
import xml.etree.ElementTree as ET
from typing import Any

import aiohttp

from ..utils.env_config import get_env

__all__ = ["ArxivReaderTool"]


class ArxivReaderTool:
    """Paper-reading tool: ArXiv / Semantic Scholar / OpenAlex backends.

    Configuration is read from .env / .env.local first:
      - ARXIV_READER_BACKEND: backend choice, one of "arxiv" | "semantic_scholar" | "openalex" (default semantic_scholar)
      - ARXIV_API_ENDPOINT:    ArXiv API endpoint (usually no need to change)
      - SEMANTIC_SCHOLAR_API_KEY: Semantic Scholar API key (free to request, optional)
      - OPENALEX_EMAIL:        optional OpenAlex email (raises the rate limit; recommended)
    """

    name: str = "arxiv_reader"
    description: str = (
        "Read paper metadata from academic databases. "
        "Supports ArXiv, Semantic Scholar, and OpenAlex backends. "
        "Input: {'paper_id': str(optional), 'query': str(optional), 'max_results': int(default=3)}. "
        "Output: list of paper metadata dicts."
    )

    def __init__(self, backend: str | None = None, use_mock: bool = False, delay_ms: tuple[int, int] = (50, 200)) -> None:
        self.backend = (backend or get_env("ARXIV_READER_BACKEND", "semantic_scholar")).lower().strip()
        self.use_mock = use_mock
        self.delay_ms = delay_ms

        # ArXiv configuration
        self.arxiv_base_url = get_env("ARXIV_API_ENDPOINT", "http://export.arxiv.org/api/query")

        # Semantic Scholar configuration
        self.ss_api_key = get_env("SEMANTIC_SCHOLAR_API_KEY")
        self.ss_base_url = "https://api.semanticscholar.org/graph/v1"

        # OpenAlex configuration
        self.openalex_email = get_env("OPENALEX_EMAIL", "")
        self.openalex_base_url = "https://api.openalex.org"

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paper_id": {
                            "type": "string",
                            "description": "ArXiv paper ID or Semantic Scholar paper ID, e.g. '1706.03762'",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 3,
                        },
                    },
                    "anyOf": [{"required": ["paper_id"]}, {"required": ["query"]}],
                },
            },
        }

    async def execute(
        self, paper_id: str | None = None, query: str | None = None, max_results: int = 3
    ) -> dict[str, Any]:
        if self.use_mock:
            return await self._mock_execute(paper_id, query, max_results)

        if self.backend == "semantic_scholar":
            return await self._semantic_scholar_execute(paper_id, query, max_results)
        if self.backend == "openalex":
            return await self._openalex_execute(paper_id, query, max_results)
        return await self._arxiv_execute(paper_id, query, max_results)

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------
    async def _mock_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000.0)

        mock_papers = [
            {
                "id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
                "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                "published": "2017-06-12",
                "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
                "source": "arxiv_mock",
            },
            {
                "id": "1810.04805",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
                "summary": "We introduce a new language representation model called BERT...",
                "published": "2018-10-11",
                "pdf_url": "https://arxiv.org/pdf/1810.04805.pdf",
                "source": "arxiv_mock",
            },
            {
                "id": "2303.18223",
                "title": "Large Language Models: A Survey",
                "authors": ["Wayne Xin Zhao", "Kun Zhou", "Junyi Li"],
                "summary": "This survey reviews the recent advances in large language models...",
                "published": "2023-03-31",
                "pdf_url": "https://arxiv.org/pdf/2303.18223.pdf",
                "source": "arxiv_mock",
            },
        ]

        if paper_id:
            papers = [p for p in mock_papers if p["id"] == paper_id]
        else:
            q = (query or "").lower()
            papers = [p for p in mock_papers if q in p["title"].lower() or q in p["summary"].lower()]

        return {
            "source": "arxiv_mock",
            "query": query or paper_id,
            "papers": papers[:max_results],
        }

    # ------------------------------------------------------------------
    # ArXiv backend (most complete library; needs a VPN in mainland China)
    # ------------------------------------------------------------------
    async def _arxiv_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        if paper_id:
            search_query = f"id:{paper_id}"
        else:
            search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.arxiv_base_url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    text = await resp.text()
        except Exception as e:
            return {
                "source": "arxiv_api",
                "query": query or paper_id,
                "papers": [],
                "error": f"ArXiv API network error (access from mainland China may need a VPN). Consider switching to the semantic_scholar backend: "
                        f"set ARXIV_READER_BACKEND=semantic_scholar in .env. Original error: {e}",
            }

        # Parse Atom XML
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            preview = text[:200].replace("\n", " ")
            return {
                "source": "arxiv_api",
                "query": query or paper_id,
                "papers": [],
                "error": f"The ArXiv API returned content that could not be parsed. The service may be temporarily unavailable or there may be a network problem. "
                        f"Content preview: {preview}... (original error: {e})",
            }

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            paper = {
                "id": (entry.find("atom:id", ns).text or "").split("/")[-1],
                "title": (entry.find("atom:title", ns).text or "").strip().replace("\n", " "),
                "summary": (entry.find("atom:summary", ns).text or "").strip(),
                "published": entry.find("atom:published", ns).text or "",
                "pdf_url": "",
                "source": "arxiv_api",
            }
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    paper["pdf_url"] = link.get("href", "")
                    break
            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text or "")
            paper["authors"] = authors
            papers.append(paper)

        return {
            "source": "arxiv_api",
            "query": query or paper_id,
            "papers": papers,
        }

    # ------------------------------------------------------------------
    # Semantic Scholar backend (reachable in mainland China, free)
    # Request a key: https://www.semanticscholar.org/product/api#api-key-form
    # ------------------------------------------------------------------
    async def _semantic_scholar_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        headers = {}
        if self.ss_api_key:
            headers["x-api-key"] = self.ss_api_key

        try:
            if paper_id:
                # Query directly by ID
                url = f"{self.ss_base_url}/paper/{paper_id}"
                params = {"fields": "title,authors,year,abstract,url,citationCount"}
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "semantic_scholar",
                                "query": paper_id,
                                "papers": [],
                                "error": f"Semantic Scholar API error: {data.get('message', resp.status)}",
                            }
                        paper = self._ss_paper_to_dict(data)
                        return {
                            "source": "semantic_scholar",
                            "query": paper_id,
                            "papers": [paper],
                        }
            else:
                # Search query
                url = f"{self.ss_base_url}/paper/search"
                params = {
                    "query": query,
                    "fields": "title,authors,year,abstract,url,citationCount",
                    "limit": max_results,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "semantic_scholar",
                                "query": query,
                                "papers": [],
                                "error": f"Semantic Scholar API error: {data.get('message', resp.status)}",
                            }
                        papers = [self._ss_paper_to_dict(p) for p in data.get("data", [])]
                        return {
                            "source": "semantic_scholar",
                            "query": query,
                            "papers": papers,
                        }
        except Exception as e:
            return {
                "source": "semantic_scholar",
                "query": query or paper_id,
                "papers": [],
                "error": f"Semantic Scholar network error: {e}",
            }

    @staticmethod
    def _ss_paper_to_dict(data: dict) -> dict:
        """Convert raw Semantic Scholar data to the unified format."""
        authors = []
        for a in data.get("authors", [])[:10]:
            name = a.get("name", "")
            if name:
                authors.append(name)

        return {
            "id": data.get("paperId", "")[:20],
            "title": data.get("title", ""),
            "authors": authors,
            "summary": data.get("abstract", "") or "",
            "published": str(data.get("year", "")),
            "pdf_url": data.get("url", ""),
            "source": "semantic_scholar",
            "citation_count": data.get("citationCount"),
        }

    # ------------------------------------------------------------------
    # OpenAlex backend (reachable in mainland China, completely free, no key)
    # Docs: https://docs.openalex.org/
    # ------------------------------------------------------------------
    async def _openalex_execute(
        self, paper_id: str | None, query: str | None, max_results: int
    ) -> dict[str, Any]:
        headers = {
            "User-Agent": "deep-research-agent",
            "Accept-Encoding": "gzip, deflate",  # avoid brotli decoding problems
        }
        if self.openalex_email:
            headers["mailto"] = self.openalex_email

        try:
            if paper_id:
                # Query directly by ID (OpenAlex ID or DOI supported)
                url = f"{self.openalex_base_url}/works/{paper_id}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "openalex",
                                "query": paper_id,
                                "papers": [],
                                "error": f"OpenAlex API error: {data.get('message', resp.status)}",
                            }
                        paper = self._openalex_paper_to_dict(data)
                        return {
                            "source": "openalex",
                            "query": paper_id,
                            "papers": [paper],
                        }
            else:
                # Search query
                url = f"{self.openalex_base_url}/works"
                params = {
                    "search": query,
                    "per-page": max_results,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            return {
                                "source": "openalex",
                                "query": query,
                                "papers": [],
                                "error": f"OpenAlex API error: {data.get('message', resp.status)}",
                            }
                        papers = [self._openalex_paper_to_dict(r) for r in data.get("results", [])]
                        return {
                            "source": "openalex",
                            "query": query,
                            "papers": papers,
                        }
        except Exception as e:
            return {
                "source": "openalex",
                "query": query or paper_id,
                "papers": [],
                "error": f"OpenAlex network error: {e}",
            }

    @staticmethod
    def _openalex_paper_to_dict(data: dict) -> dict:
        """Convert raw OpenAlex data to the unified format."""
        authors = []
        for a in data.get("authorships", [])[:10]:
            author_info = a.get("author", {})
            name = author_info.get("display_name", "")
            if name:
                authors.append(name)

        # OpenAlex's abstract is an inverted index; simply leave it empty or take it from the summary
        summary = ""
        ab = data.get("abstract_inverted_index")
        if ab:
            # Restore the inverted index to approximate text (ordering by frequency is imprecise; simple concatenation here)
            words = []
            for word, positions in ab.items():
                for pos in positions:
                    while len(words) <= pos:
                        words.append("")
                    words[pos] = word
            summary = " ".join(words)

        # PDF link
        pdf_url = ""
        oa = data.get("open_access", {})
        if oa:
            pdf_url = oa.get("oa_url", "") or oa.get("pdf_url", "")
        if not pdf_url:
        # Try taking it from best_oa_location
            loc = data.get("best_oa_location", {})
            if loc:
                pdf_url = loc.get("pdf_url", "") or loc.get("landing_page_url", "")

        return {
            "id": (data.get("id") or "").split("/")[-1],
            "title": data.get("display_name", ""),
            "authors": authors,
            "summary": summary,
            "published": str(data.get("publication_year", "")),
            "pdf_url": pdf_url,
            "source": "openalex",
            "citation_count": data.get("cited_by_count"),
        }
