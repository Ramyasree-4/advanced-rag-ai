from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.errors import DocumentProcessingError
from app.models.schemas import DocumentMetadata
from app.services.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class IngestedDocument:
    metadata: DocumentMetadata
    documents: list[Document]


class DocumentIngestionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.preprocessor = TextPreprocessor()

    async def save_upload(self, file: UploadFile) -> Path:
        content = await file.read()
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise DocumentProcessingError(f"File exceeds {self.settings.max_upload_mb}MB upload limit")

        suffix = Path(file.filename or "document").suffix.lower()
        document_id = self._document_id(content, file.filename or "document")
        safe_name = f"{document_id}{suffix or '.txt'}"
        target = self.settings.upload_directory / safe_name
        target.write_bytes(content)
        return target

    def load(self, path: Path, original_filename: str, content_type: str) -> IngestedDocument:
        try:
            loader = self._loader_for(path)
            raw_documents = loader.load()
            documents = self._normalize_documents(raw_documents, original_filename, content_type)
            metadata = DocumentMetadata(
                document_id=path.stem,
                filename=original_filename,
                content_type=content_type,
                pages=len(raw_documents) if path.suffix.lower() == ".pdf" else None,
                chunks=0,
                uploaded_at=datetime.utcnow(),
            )
            return IngestedDocument(metadata=metadata, documents=documents)
        except Exception as exc:
            logger.exception("Document ingestion failed for %s", original_filename)
            raise DocumentProcessingError(str(exc)) from exc

    def _loader_for(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return PyPDFLoader(str(path))
        if suffix in {".txt", ".md", ".csv"}:
            return TextLoader(str(path), encoding="utf-8")
        return UnstructuredFileLoader(str(path))

    def _normalize_documents(self, docs: list[Document], filename: str, content_type: str) -> list[Document]:
        normalized: list[Document] = []
        for index, doc in enumerate(docs, start=1):
            text = self.preprocessor.clean(doc.page_content)
            if not text:
                continue
            metadata = {
                **doc.metadata,
                "filename": filename,
                "content_type": content_type,
                "page": doc.metadata.get("page", index - 1) + 1 if isinstance(doc.metadata.get("page", index), int) else index,
            }
            normalized.append(Document(page_content=text, metadata=metadata))
        return normalized

    def _document_id(self, content: bytes, filename: str) -> str:
        digest = hashlib.sha256(content + filename.encode("utf-8")).hexdigest()[:16]
        return f"doc_{digest}_{uuid4().hex[:8]}"
