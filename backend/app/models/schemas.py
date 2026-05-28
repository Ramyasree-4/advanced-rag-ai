from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    content_type: str
    pages: int | None = None
    chunks: int = 0
    uploaded_at: datetime


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int | None = None
    chunk_id: str
    score: float | None = None
    excerpt: str


class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    citations: list[Citation] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str
    message: str
    document_ids: list[str] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True
    debug: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    rewritten_query: str
    retrieval_debug: list[dict[str, Any]] = Field(default_factory=list)
    quality_evaluation: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)


class RetrieverSettings(BaseModel):
    k: int
    fetch_k: int
    score_threshold: float
    rerank_top_n: int
