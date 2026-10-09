"""
File reader (FileReaderTool)

Rationale:
  Deep research often needs to handle user-uploaded documents (PDF reports, CSV datasets, Markdown notes).
  FileReaderTool reads local files and converts them to a text format the LLM can consume.

Supported formats:
  - .txt, .md, .markdown -> read directly
  - .pdf -> extract text (PyPDF2 / pdfplumber fallback)
  - .csv, .json -> read and format a summary
  - .docx -> extracted with python-docx (optional dependency)

Security design:
  - Only files under the specified directory may be read (sandbox mode)
  - File size limit (default 10MB)
  - Never executes any code in files
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


__all__ = ["FileReaderTool"]

# Default allowed file extensions
_SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".csv", ".json", ".docx"}
# Default file size limit (bytes)
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class FileReaderTool:
    """File reader: reads local files and returns structured text."""

    name: str = "file_reader"
    description: str = (
        "Read a local file and return its content as text. "
        "Supports: .txt, .md, .pdf, .csv, .json, .docx. "
        "Use this when the user references an uploaded document or dataset. "
        "Input: {'file_path': str}. Output: file content as formatted text."
    )

    # Sentinel object: distinguishes "argument not passed" from "explicitly passed None"
    _UNSET = object()

    def __init__(
        self,
        allowed_base_dir: str | None = _UNSET,
        max_file_size: int | None = _UNSET,
    ) -> None:
        """
        Args:
            allowed_base_dir: root directory that may be read.
                              None means unrestricted (strongly recommended to set in production).
                              When not passed, FILE_READER_ALLOWED_BASE_DIR is read from .env.
            max_file_size: maximum file size in bytes; larger files are rejected.
                           When not passed, FILE_READER_MAX_FILE_SIZE is read from .env.
        """
        from ..utils.env_config import get_env, get_env_int

        if allowed_base_dir is not FileReaderTool._UNSET:
            self.allowed_base_dir = Path(allowed_base_dir).resolve() if allowed_base_dir else None
        else:
            env_dir = get_env("FILE_READER_ALLOWED_BASE_DIR")
            self.allowed_base_dir = Path(env_dir).resolve() if env_dir else None

        if max_file_size is not FileReaderTool._UNSET:
            self.max_file_size = max_file_size if max_file_size is not None else _MAX_FILE_SIZE
        else:
            self.max_file_size = get_env_int("FILE_READER_MAX_FILE_SIZE", _MAX_FILE_SIZE)

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file to read",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        }

    async def execute(self, file_path: str) -> str:
        """Read a file and return its content.

        Args:
            file_path: file path (absolute or relative).

        Returns:
            Text representation of the file content.
        """
        # Simulate async IO (file reading is IO-bound, but synchronous operations are fast enough)
        import asyncio
        await asyncio.sleep(0)

        try:
            path = Path(file_path).resolve()
        except Exception as e:
            return f"[FileReader Error] Invalid path: {e}"

        # Security check 1: directory restriction
        if self.allowed_base_dir and not str(path).startswith(str(self.allowed_base_dir)):
            return (
                f"[FileReader Error] Access denied: {path} is outside the allowed directory "
                f"{self.allowed_base_dir}."
            )

        # Security check 2: file existence
        if not path.exists():
            return f"[FileReader Error] File not found: {path}"
        if not path.is_file():
            return f"[FileReader Error] Not a file: {path}"

        # Security check 3: extension
        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            return (
                f"[FileReader Error] Unsupported file type: {ext}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTS))}"
            )

        # Security check 4: file size
        size = path.stat().st_size
        if size > self.max_file_size:
            return (
                f"[FileReader Error] File too large: {size} bytes "
                f"(max allowed: {self.max_file_size} bytes)."
            )

        # Read the file
        try:
            return self._read_by_ext(path, ext)
        except Exception as e:
            return f"[FileReader Error] Failed to read {path}: {type(e).__name__}: {e}"

    def _read_by_ext(self, path: Path, ext: str) -> str:
        """Choose the reading strategy by extension."""
        if ext in (".txt", ".md", ".markdown"):
            return self._read_text(path)
        if ext == ".pdf":
            return self._read_pdf(path)
        if ext == ".csv":
            return self._read_csv(path)
        if ext == ".json":
            return self._read_json(path)
        if ext == ".docx":
            return self._read_docx(path)
        return f"[FileReader Error] No reader implemented for {ext}"

    @staticmethod
    def _read_text(path: Path) -> str:
        """Read a plain text file."""
        content = path.read_text(encoding="utf-8", errors="replace")
        # Add a file metadata header
        return f"[File: {path.name}]\n[Size: {len(content)} chars]\n\n{content}"

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Read a PDF file."""
        # Try pdfplumber first (preserves tables better)
        try:
            import pdfplumber
            texts = []
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        texts.append(f"--- Page {i} ---\n{text}")
            full = "\n\n".join(texts)
            return f"[File: {path.name}]\n[Pages: {len(pdf.pages)}]\n\n{full}"
        except ImportError:
            pass

        # Fall back to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            texts = []
            for i, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    texts.append(f"--- Page {i} ---\n{text}")
            full = "\n\n".join(texts)
            return f"[File: {path.name}]\n[Pages: {len(reader.pages)}]\n\n{full}"
        except ImportError:
            return (
                f"[FileReader Error] Cannot read PDF: {path.name}. "
                f"Please install pdfplumber or PyPDF2: pip install pdfplumber"
            )

    @staticmethod
    def _read_csv(path: Path, preview_rows: int = 20) -> str:
        """Read a CSV file and return a structured summary."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            shape = df.shape
            dtypes = df.dtypes.to_dict()
            head = df.head(preview_rows).to_string(index=False)
            summary = (
                f"[File: {path.name}]\n"
                f"[Shape: {shape[0]} rows × {shape[1]} columns]\n"
                f"[Columns: {list(df.columns)}]\n"
                f"[Dtypes: {dtypes}]\n\n"
                f"--- First {preview_rows} rows ---\n{head}"
            )
            if shape[0] > preview_rows:
                summary += f"\n\n[Note: {shape[0] - preview_rows} more rows not shown]"
            return summary
        except ImportError:
            return "[FileReader Error] pandas required for CSV. pip install pandas"

    @staticmethod
    def _read_json(path: Path, max_depth: int = 3) -> str:
        """Read a JSON file and return a formatted summary."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Compute basic statistics
        def _summarize(obj, depth: int = 0) -> str:
            if depth > max_depth:
                return "..."
            if isinstance(obj, dict):
                items = []
                for k, v in list(obj.items())[:10]:
                    items.append(f"  {k}: {_summarize(v, depth + 1)}")
                if len(obj) > 10:
                    items.append(f"  ... ({len(obj) - 10} more keys)")
                return "{\n" + "\n".join(items) + "\n}"
            if isinstance(obj, list):
                if len(obj) == 0:
                    return "[]"
                sample = _summarize(obj[0], depth + 1)
                return f"[{len(obj)} items, e.g.: {sample}]"
            return repr(obj)

        summary = _summarize(data)
        type_name = type(data).__name__
        return f"[File: {path.name}]\n[Type: {type_name}]\n\n{summary}"

    @staticmethod
    def _read_docx(path: Path) -> str:
        """Read a Word document."""
        try:
            from docx import Document
            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n\n".join(paragraphs)
            return f"[File: {path.name}]\n[Paragraphs: {len(paragraphs)}]\n\n{content}"
        except ImportError:
            return (
                f"[FileReader Error] Cannot read DOCX: {path.name}. "
                f"Please install python-docx: pip install python-docx"
            )
