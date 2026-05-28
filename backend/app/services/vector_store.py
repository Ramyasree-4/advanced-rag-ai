from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

from app.core.config import get_settings
from app.services.embeddings import get_embedding_model


class ChromaVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = Chroma(
            collection_name="advancedrag_documents",
            embedding_function=get_embedding_model(),
            persist_directory=str(Path(self.settings.chroma_persist_directory)),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[Document]) -> list[str]:
        # Filter out None and complex metadata values that ChromaDB cannot store
        chunks = filter_complex_metadata(chunks)
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        self.store.add_documents(chunks, ids=ids)
        return ids

    def similarity_search_with_score(self, query: str, k: int, filters: dict | None = None):
        results = self.store.similarity_search_with_score(query, k=k, filter=filters or None)
        return [(doc, self._cosine_distance_to_similarity(distance)) for doc, distance in results]

    def all_documents(self) -> list[Document]:
        result = self.store.get(include=["documents", "metadatas"])
        texts = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        return [Document(page_content=text, metadata=metadata or {}) for text, metadata in zip(texts, metadatas)]

    def as_retriever(self, k: int, filters: dict | None = None):
        return self.store.as_retriever(search_kwargs={"k": k, "filter": filters or None})

    def delete_document(self, document_id: str) -> None:
        self.store.delete(where={"document_id": document_id})

    def _cosine_distance_to_similarity(self, distance: float) -> float:
        similarity = 1.0 - float(distance)
        return max(0.0, min(1.0, similarity))
