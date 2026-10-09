"""
Browser tool (BrowserTool) — full-text web page reader

Rationale:
  web_search only returns search snippets (usually 100-200 characters), while deep research needs the original page text.
  BrowserTool: open URL -> extract main text -> strip ads/navigation -> return structured text.

Relationship with web_search:
  web_search: "find links that may contain information"
  browser: "read the specific content behind this link"
  They are upstream/downstream of each other, not duplicates.

Implementation notes:
  - Fetch asynchronously with aiohttp so the event loop is not blocked
  - Extract main text with BeautifulSoup (remove noise tags such as script/style/nav)
  - Automatically truncate overlong pages (keep the first N paragraphs to prevent token explosion)
  - Supports retries and graceful error degradation
"""
from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any

import aiohttp


__all__ = ["BrowserTool", "MockBrowserTool"]

# Default upper bound on kept body characters (keeps overlong pages from filling the context)
_DEFAULT_MAX_CHARS = 8000

# HTML tags that usually contain the main text
_CONTENT_TAGS = ["article", "main", "section", "div"]
# Noise tags (removed outright)
_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe", "svg"]


class BaseBrowserTool(ABC):
    """Base class of the browser tool."""

    name: str = "browser"
    description: str = (
        "Open a URL and extract the main article text. "
        "Use this after web_search when you need to read the full content of a webpage. "
        "Input: {'url': str, 'max_chars': int(optional, default=8000)}. "
        "Output: extracted text content."
    )

    @abstractmethod
    async def execute(self, url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        """Open a URL and extract the main text.

        Args:
            url: the web page address to visit.
            max_chars: maximum characters returned; truncated beyond this.

        Returns:
            The extracted main text.
        """
        ...

    def get_openai_tool_schema(self) -> dict:
        """Return the schema in OpenAI Function Calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to open and read",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return (default: 8000)",
                            "default": 8000,
                        },
                    },
                    "required": ["url"],
                },
            },
        }


class BrowserTool(BaseBrowserTool):
    """Real browser tool: async HTTP fetching + main-text extraction.

    Configuration is read from .env / .env.local first; constructor arguments only override.
    Supported environment variables:
      - BROWSER_TIMEOUT: HTTP request timeout in seconds (default 15)
      - BROWSER_USER_AGENT: custom User-Agent
    """

    def __init__(self, timeout: int | None = None, user_agent: str | None = None) -> None:
        from ..utils.env_config import get_env, get_env_int

        self.timeout = timeout or get_env_int("BROWSER_TIMEOUT", 15)
        self.user_agent = user_agent or get_env(
            "BROWSER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )

    async def execute(self, url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        if not url.startswith(("http://", "https://")):
            return f"[Browser Error] Invalid URL: {url}. URL must start with http:// or https://"

        try:
            html = await self._fetch(url)
            text = self._extract_text(html)
            text = self._clean_text(text)

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[CONTENT_TRUNCATED: {len(text)} chars total, showing first {max_chars}]"

            return text if text else "[Browser Warning] No meaningful content extracted from the page."

        except aiohttp.ClientError as e:
            return f"[Browser Error] Network error: {type(e).__name__}: {e}"
        except Exception as e:
            return f"[Browser Error] Unexpected: {type(e).__name__}: {e}"

    async def _fetch(self, url: str) -> str:
        """Asynchronously fetch the page HTML."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={"User-Agent": self.user_agent},
        ) as session:
            async with session.get(url, allow_redirects=True) as resp:
                resp.raise_for_status()
                # Try to auto-detect the encoding
                charset = resp.charset or "utf-8"
                return await resp.text(encoding=charset)

    def _extract_text(self, html: str) -> str:
        """Extract the main text from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            # Fallback: simple regex extraction
            return self._fallback_extract(html)

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise tags
        for tag_name in _NOISE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Strategy 1: find an article or main tag (common in semantic HTML)
        for tag_name in ["article", "main"]:
            tag = soup.find(tag_name)
            if tag:
                return tag.get_text(separator="\n", strip=True)

        # Strategy 2: find the longest div (heuristic: the main text is usually in the longest div)
        best_div = None
        best_len = 0
        for div in soup.find_all("div"):
            text_len = len(div.get_text(strip=True))
            if text_len > best_len:
                best_len = text_len
                best_div = div

        if best_div and best_len > 200:
            return best_div.get_text(separator="\n", strip=True)

        # Strategy 3: fall back to the whole body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

    def _fallback_extract(self, html: str) -> str:
        """Fallback extraction when BeautifulSoup is unavailable."""
        # Remove script/style content
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove all tags, keep the text
        text = re.sub(r"<[^>]+>", "\n", html)
        return self._clean_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean the extracted text."""
        # Merge extra newlines
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        # Strip leading/trailing whitespace on each line
        lines = [line.strip() for line in text.splitlines()]
        # Filter empty and too-short lines (usually navigation items)
        lines = [line for line in lines if len(line) > 3]
        return "\n".join(lines)


class MockBrowserTool(BaseBrowserTool):
    """Mock browser tool: for debugging without a network."""

    _MOCK_PAGES: dict[str, str] = {
        "https://example.com/ai-report-2024": """
2024 Global Artificial Intelligence Development Report

Summary
In 2024 the global AI industry entered a boom period. Large language models passed the trillion-parameter mark,
and multimodal capabilities improved markedly. China and the US continue to lead in both foundational AI research and deployment.

1. Market size
According to IDC, the global AI market reached USD 554.0 billion in 2024, up 38.2% year over year.
Generative AI accounted for 28%, about USD 155.1 billion.

2. Technical progress
1. Large language models: GPT-4o, Claude 3.5, Gemini 1.5 Pro and others made breakthroughs in multimodal reasoning
2. Code generation: GitHub Copilot has more than 5 million monthly active developers
3. Scientific discovery: AlphaFold 3 predicts the structures of almost all biomolecules

3. Key players
- OpenAI: valued at USD 157 billion, annualized revenue USD 3.4 billion
- Anthropic: valued at USD 40 billion, Claude series growing rapidly
- Google DeepMind: Gemini integrated across the product line
- Baidu: ERNIE Bot users exceed 300 million

4. Policy and regulation
The EU AI Act formally entered into force in August 2024, the world's first comprehensive law regulating AI.

Sources: IDC, OpenAI Blog, Anthropic official announcements
        """.strip(),
        "https://example.com/quantum-computing": """
Quantum computing: latest progress (2024)

In December 2024 IBM released the Condor quantum processor with 1,121 qubits,
the largest publicly available quantum processor.

The Google Quantum AI team reached a key milestone in surface-code error correction,
with the logical error rate falling below the physical error rate for the first time.

The team of Pan Jianwei at the University of Science and Technology of China achieved quantum computational advantage with 255 photons.

Sources: IBM Research Blog, Nature, USTC official website
        """.strip(),
    }

    async def execute(self, url: str, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
        await asyncio.sleep(0.1)  # simulate network latency

        # Exact match
        if url in self._MOCK_PAGES:
            content = self._MOCK_PAGES[url]
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n[CONTENT_TRUNCATED]"
            return content

        # Fuzzy match: return a generic mock based on URL keywords
        if "wikipedia" in url.lower():
            return f"[Mock Browser] Wikipedia page for {url}\n\nThis is a mock Wikipedia article. In production, BrowserTool would fetch the real Wikipedia content."

        return f"[Mock Browser] Fetched {url}\n\nThis is a generic mock page. Content would be extracted from the real webpage in production mode."


def get_browser_tool(mock_mode: bool = False, **kwargs) -> BaseBrowserTool:
    """Factory function: return a BrowserTool or MockBrowserTool according to configuration."""
    if mock_mode:
        return MockBrowserTool()
    return BrowserTool(**kwargs)
