"""
LLM Summarizer module: hierarchical summarization

Design decisions:
1. Two-level summarization: per-document summary → aggregate summary, to avoid quality loss from feeding too much at once
2. The prompt templates strictly require keeping numbers, source citations and expressions of uncertainty (e.g. "about", "may", "reportedly")
3. Single-document summaries are capped by max_length; the aggregate summary may be longer to keep cross-document links
4. Reuses project one's VLLMPolicy as the LLM backend, keeping the interface consistent
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# single-document summary prompt
_DOCUMENT_SUMMARY_PROMPT = """Summarize the following document. Requirements:
1. Keep all key numbers, dates and statistics
2. Keep source citations and author information
3. Keep expressions of uncertainty (e.g. "about", "may", "reportedly", "preliminary results show")
4. The summary must not exceed {max_length} characters
5. Write in the same language as the document

Document:
{doc}

Summary:"""

# aggregate summary prompt
_AGGREGATE_SUMMARY_PROMPT = """Merge the following document summaries into one coherent overview. Requirements:
1. Merge duplicated information and keep complementary content across documents
2. Keep all key numbers, dates and statistics
3. Keep source citations
4. Keep expressions of uncertainty
5. If documents contradict each other, list the different views separately and attribute each to its source
6. The total length must not exceed {max_length} characters
7. Write in the same language as the documents

User query context: {query}

List of document summaries:
{docs}

Overview:"""


class LLMSummarizer:
    """
    LLM hierarchical summarizer.

    Calls the VLLMPolicy for single-document and aggregate summaries;
    the prompts are specially designed to keep information-dense content.
    """

    def __init__(self, llm_policy: Any) -> None:
        """
        Initialize the LLM summarizer.

        Args:
            llm_policy: a VLLMPolicy instance or any object implementing the __call__(messages) interface
        """
        self.llm_policy = llm_policy

    def summarize_document(
        self,
        doc: str,
        query: str = "",
        max_length: int = 500,
    ) -> str:
        """
        Single-document summary.

        Args:
            doc: the original document text
            query: the current query (optional, for context understanding)
            max_length: maximum summary length in characters

        Returns:
            the summary text
        """
        if not doc or len(doc) < max_length:
            # the document is already short, return it directly (with simple cleanup)
            return doc.strip()

        prompt = _DOCUMENT_SUMMARY_PROMPT.format(doc=doc, max_length=max_length)
        if query:
            prompt = f"User query: {query}\n\n" + prompt

        try:
            resp = self.llm_policy([{"role": "user", "content": prompt}])
            summary = str(resp.content or "").strip()
            if not summary:
                logger.warning("LLM returned empty summary, returning truncated original.")
                return doc[:max_length] + "\n[TRUNCATED]"
            return summary
        except Exception as e:
            logger.error(f"LLM summarize_document failed: {e}")
            # Fallback: truncate the original text directly
            return doc[:max_length] + "\n[TRUNCATED]"

    def summarize_documents(
        self,
        docs: list[str],
        query: str = "",
        max_length: int = 800,
    ) -> str:
        """
        Aggregate summary of several documents.

        Flow:
        1. first summarize each document on its own (if the document is long)
        2. aggregate all the single-document summaries into the final overview

        Args:
            docs: list of documents
            query: the current query
            max_length: maximum length of the final overview in characters

        Returns:
            the aggregate summary text
        """
        if not docs:
            return ""
        if len(docs) == 1:
            return self.summarize_document(docs[0], query, max_length)

        # phase 1: per-document summaries
        single_summaries: list[str] = []
        for i, doc in enumerate(docs, 1):
            # keep single-document summaries short, so the aggregation input does not explode
            single_max = max(200, max_length // len(docs))
            summary = self.summarize_document(doc, query, max_length=single_max)
            single_summaries.append(f"[Document {i}]\n{summary}")

        # phase 2: aggregate summary
        combined_text = "\n\n".join(single_summaries)
        prompt = _AGGREGATE_SUMMARY_PROMPT.format(
            query=query,
            docs=combined_text,
            max_length=max_length,
        )

        try:
            resp = self.llm_policy([{"role": "user", "content": prompt}])
            aggregate = str(resp.content or "").strip()
            if not aggregate:
                logger.warning("LLM returned empty aggregate summary, concatenating singles.")
                return "\n\n".join(single_summaries)
            return aggregate
        except Exception as e:
            logger.error(f"LLM summarize_documents failed: {e}")
            return "\n\n".join(single_summaries)

    def get_stats(self, original_docs: list[str], summary: str) -> dict[str, Any]:
        """
        Return summary statistics.

        Args:
            original_docs: list of original documents
            summary: the summary result

        Returns:
            a stats dict
        """
        total_orig = sum(len(d) for d in original_docs)
        comp_len = len(summary)
        ratio = comp_len / max(total_orig, 1)
        return {
            "compression_ratio": round(ratio, 3),
            "original_chars": total_orig,
            "summary_chars": comp_len,
            "n_documents": len(original_docs),
        }
