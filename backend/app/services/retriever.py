from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.errors import RetrievalError
from app.models.schemas import Citation
from app.services.cross_encoder_reranker import CrossEncoderReranker
from app.services.hybrid_retrieval import HybridLexicalRetriever
from app.services.retrieval_router import retrieval_router
from app.services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = ChromaVectorStore()
        self.lexical_retriever = HybridLexicalRetriever()
        self.reranker = CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Document], list[Citation], list[dict[str, Any]]]:
        try:
            vector_filter = self._build_filter(document_ids, filters)
            route = retrieval_router.route(query)
            vector_results = self.vector_store.similarity_search_with_score(
                query,
                k=self.settings.retrieval_fetch_k,
                filters=vector_filter,
            )
            lexical_results = self.lexical_retriever.rank(
                query=query,
                documents=self.vector_store.all_documents(),
                top_k=self.settings.retrieval_fetch_k,
                document_ids=document_ids,
                filters=filters,
            )
            fused = self._rrf_fuse(vector_results, lexical_results, route)
            filtered = [(doc, score) for doc, score in fused if score >= self.settings.score_threshold]
            reranked = self.reranker.rerank(query, filtered or fused, self.settings.rerank_top_n)
            docs = [doc for doc, _, _ in reranked[: self.settings.retrieval_k]]
            citations = [self._citation(doc, score) for doc, score, _ in reranked[: self.settings.retrieval_k]]
            debug = (
                [{"stage": "router", "strategy": route["strategy"], "vector_weight": route["vector_weight"], "lexical_weight": route["lexical_weight"]}]
                + [self._debug_row(doc, score, "vector") for doc, score in vector_results[: self.settings.retrieval_k]]
                + [self._debug_row(doc, score, "lexical") for doc, score in lexical_results[: self.settings.retrieval_k]]
                + [self._debug_row(doc, score, "rrf_hybrid") for doc, score in fused[: self.settings.retrieval_k]]
                + [self._rerank_debug_row(doc, score, cross_score) for doc, score, cross_score in reranked]
            )
            return docs, citations, debug
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise RetrievalError(str(exc)) from exc

    def _build_filter(self, document_ids: list[str] | None, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if document_ids:
            clauses.append({"document_id": {"$in": document_ids}})
        for key, value in (filters or {}).items():
            if value is not None and value != "":
                clauses.append({key: value})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _citation(self, doc: Document, score: float | None) -> Citation:
        return Citation(
            document_id=str(doc.metadata.get("document_id", "")),
            filename=str(doc.metadata.get("filename", "unknown")),
            page=doc.metadata.get("page"),
            chunk_id=str(doc.metadata.get("chunk_id", "")),
            score=score,
            excerpt=doc.page_content[:360],
        )

    def _rrf_fuse(
        self,
        vector_results: list[tuple[Document, float]],
        lexical_results: list[tuple[Document, float]],
        route: dict[str, float | str],
    ) -> list[tuple[Document, float]]:
        by_chunk: dict[str, dict[str, Any]] = {}
        k = 60.0
        vector_weight = float(route["vector_weight"])
        lexical_weight = float(route["lexical_weight"])

        for rank, (doc, score) in enumerate(vector_results, start=1):
            chunk_id = str(doc.metadata.get("chunk_id"))
            by_chunk[chunk_id] = {
                "doc": doc,
                "vector": score,
                "lexical": 0.0,
                "rrf": vector_weight / (k + rank),
            }
        for rank, (doc, score) in enumerate(lexical_results, start=1):
            chunk_id = str(doc.metadata.get("chunk_id"))
            if chunk_id not in by_chunk:
                by_chunk[chunk_id] = {
                    "doc": doc,
                    "vector": 0.0,
                    "lexical": score,
                    "rrf": lexical_weight / (k + rank),
                }
            else:
                by_chunk[chunk_id]["lexical"] = score
                by_chunk[chunk_id]["rrf"] += lexical_weight / (k + rank)

        max_rrf = max((float(item["rrf"]) for item in by_chunk.values()), default=1.0)
        fused: list[tuple[Document, float]] = []
        for item in by_chunk.values():
            rrf_score = float(item["rrf"]) / max_rrf
            score = (0.7 * rrf_score) + (0.2 * float(item["vector"])) + (0.1 * float(item["lexical"]))
            fused.append((item["doc"], max(0.0, min(1.0, score))))
        return sorted(fused, key=lambda item: item[1], reverse=True)

    def _debug_row(self, doc: Document, score: float | None, stage: str) -> dict[str, Any]:
        return {
            "stage": stage,
            "chunk_id": doc.metadata.get("chunk_id"),
            "filename": doc.metadata.get("filename"),
            "page": doc.metadata.get("page"),
            "score": score,
            "chars": len(doc.page_content),
        }

    def _rerank_debug_row(self, doc: Document, score: float | None, cross_score: float | None) -> dict[str, Any]:
        row = self._debug_row(doc, score, "cross_encoder_rerank")
        row["cross_encoder_score"] = cross_score
        return row
