from __future__ import annotations

import logging

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lightweight reranker that uses lexical scoring instead of a heavy cross-encoder model.
    This avoids loading sentence_transformers which exceeds Render free tier memory limits.
    """

    def rerank(self, query: str, docs_with_scores: list[tuple[Document, float]], top_n: int):
        if not docs_with_scores:
            return []
        terms = {term.lower() for term in query.split() if len(term) > 2}
        reranked = []
        for doc, retrieval_score in docs_with_scores:
            text = doc.page_content.lower()
            lexical_hits = sum(1 for term in terms if term in text)
            normalized = min(1.0, lexical_hits * 0.1)
            final_score = max(0.0, min(1.0, (0.65 * (retrieval_score or 0.0)) + (0.35 * normalized)))
            reranked.append((doc, final_score, normalized))
        return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_n]
