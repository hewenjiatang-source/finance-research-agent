"""Tools subpackage: wrappers for external capabilities (search, paper reading, code execution, browsing, file reading, calculation, notes, ...)."""
from __future__ import annotations

from .web_search import WebSearchTool, MockWebSearchTool, BaseWebSearchTool
from .arxiv_reader import ArxivReaderTool
from .code_sandbox import CodeSandboxTool
from .browser import BrowserTool, MockBrowserTool, BaseBrowserTool, get_browser_tool
from .file_reader import FileReaderTool
from .calculator import CalculatorTool
from .notepad import NotepadTool, NotepadEntry
from .sec_edgar import SecClient, SecFilingsTool, SecFactsTool, SecFilingTool, create_sec_tools

__all__ = [
    # Search and reading
    "WebSearchTool",
    "MockWebSearchTool",
    "BaseWebSearchTool",
    "ArxivReaderTool",
    "BrowserTool",
    "MockBrowserTool",
    "BaseBrowserTool",
    "get_browser_tool",
    "FileReaderTool",
    # Calculation and execution
    "CodeSandboxTool",
    "CalculatorTool",
    # Helpers
    "NotepadTool",
    "NotepadEntry",
    # Financial filings research (SEC EDGAR)
    "SecClient",
    "SecFilingsTool",
    "SecFactsTool",
    "SecFilingTool",
    "create_sec_tools",
]
