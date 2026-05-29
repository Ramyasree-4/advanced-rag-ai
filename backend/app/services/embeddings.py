from __future__ import annotations

from functools import lru_cache

from langchain_mistralai import MistralAIEmbeddings

from app.core.config import get_settings


@lru_cache
def get_embedding_model() -> MistralAIEmbeddings:
    settings = get_settings()
    return MistralAIEmbeddings(
        model="mistral-embed",
        api_key=settings.mistral_api_key,
    )
