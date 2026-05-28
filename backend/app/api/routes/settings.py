from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import RetrieverSettings

router = APIRouter()


@router.get("/retriever", response_model=RetrieverSettings)
def retriever_settings() -> RetrieverSettings:
    settings = get_settings()
    return RetrieverSettings(
        k=settings.retrieval_k,
        fetch_k=settings.retrieval_fetch_k,
        score_threshold=settings.score_threshold,
        rerank_top_n=settings.rerank_top_n,
    )
