from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document


class HybridLexicalRetriever:
    """Small BM25-style lexical retriever for hybrid dense + keyword search."""

    def rank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        candidates = [doc for doc in documents if self._matches(doc, document_ids, filters)]
        if not candidates:
            return []

        tokenized_docs = [self._tokens(doc.page_content) for doc in candidates]
        query_terms = self._tokens(query)
        if not query_terms:
            return []

        avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
        doc_freq = Counter(term for tokens in tokenized_docs for term in set(tokens))
        raw_scores = [
            self._bm25_score(query_terms, tokens, doc_freq, len(candidates), avg_len)
            for tokens in tokenized_docs
        ]
        max_score = max(raw_scores) or 1.0
        scored = [
            (doc, max(0.0, min(0.85, score / max_score)))
            for doc, score in zip(candidates, raw_scores)
            if score > 0
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _bm25_score(
        self,
        query_terms: list[str],
        doc_terms: list[str],
        doc_freq: Counter[str],
        corpus_size: int,
        avg_len: float,
    ) -> float:
        k1 = 1.5
        b = 0.75
        term_freq = Counter(doc_terms)
        score = 0.0
        doc_len = len(doc_terms) or 1
        for term in query_terms:
            if term not in term_freq:
                continue
            idf = math.log(1 + (corpus_size - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            numerator = term_freq[term] * (k1 + 1)
            denominator = term_freq[term] + k1 * (1 - b + b * doc_len / max(1.0, avg_len))
            score += idf * numerator / denominator
        return score

    def _tokens(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2]

    def _matches(self, doc: Document, document_ids: list[str] | None, filters: dict[str, Any] | None) -> bool:
        if document_ids and doc.metadata.get("document_id") not in document_ids:
            return False
        for key, value in (filters or {}).items():
            if value not in (None, "") and doc.metadata.get(key) != value:
                return False
        return True
