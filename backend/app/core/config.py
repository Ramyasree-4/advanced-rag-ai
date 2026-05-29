from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AdvancedRAG AI"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ]
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                return json.loads(v)
            # comma-separated plain string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    mistral_api_key: str | None = None
    mistral_model: str = "mistral-large-latest"
    mistral_temperature: float = 0.2
    mistral_max_tokens: int = 1200

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "advancedrag-ai"

    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_persist_directory: Path = Path("./storage/chroma")
    upload_directory: Path = Path("./storage/uploads")
    max_upload_mb: int = 50

    retrieval_k: int = 6
    retrieval_fetch_k: int = 50
    rerank_top_n: int = 6
    score_threshold: float = 0.25
    chunk_min_chars: int = 700
    chunk_max_chars: int = 1600
    chunk_overlap: int = 180
    context_token_budget: int = 4200

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    settings.upload_directory.mkdir(parents=True, exist_ok=True)
    return settings
