from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.cross_encoder_model_name)


class CrossEncoderReranker:
    def rerank(self, query: str, docs_with_scores: list[tuple[Document, float]], top_n: int):
        if not docs_with_scores:
            return []
        try:
            model = get_cross_encoder()
            pairs = [(query, doc.page_content[:1800]) for doc, _ in docs_with_scores]
            raw_scores = [float(score) for score in model.predict(pairs)]
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            spread = max(max_score - min_score, 1e-9)
            reranked = []
            for (doc, retrieval_score), raw_score in zip(docs_with_scores, raw_scores):
                normalized = (raw_score - min_score) / spread
                final_score = max(0.0, min(1.0, (0.35 * retrieval_score) + (0.65 * normalized)))
                reranked.append((doc, final_score, normalized))
            return sorted(reranked, key=lambda item: item[1], reverse=True)[:top_n]
        except Exception as exc:
            logger.warning("Cross-encoder reranking unavailable, using retrieval order: %s", exc)
            return [(doc, score, None) for doc, score in docs_with_scores[:top_n]]
