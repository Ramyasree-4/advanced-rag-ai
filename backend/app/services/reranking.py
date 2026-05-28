from __future__ import annotations

from langchain_core.documents import Document


class LightweightReranker:
    """Fast lexical reranker used after dense retrieval for portfolio-friendly local operation."""

    def rerank(self, query: str, docs_with_scores: list[tuple[Document, float]], top_n: int) -> list[tuple[Document, float]]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored: list[tuple[Document, float]] = []
        for doc, score in docs_with_scores:
            text = doc.page_content.lower()
            lexical_hits = sum(1 for term in terms if term in text)
            blended = (score or 0.0) + min(0.2, lexical_hits * 0.025)
            scored.append((doc, max(0.0, min(1.0, blended))))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]
