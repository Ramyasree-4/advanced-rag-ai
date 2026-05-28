from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.errors import DocumentProcessingError
from app.models.schemas import DocumentMetadata
from app.services.chunking import AdaptiveChunker
from app.services.ingestion import DocumentIngestionService
from app.services.vector_store import ChromaVectorStore

router = APIRouter()
metadata_file = Path("./storage/documents.json")
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=DocumentMetadata)
async def upload_document(file: UploadFile = File(...)) -> DocumentMetadata:
    ingestion = DocumentIngestionService()
    try:
        path = await ingestion.save_upload(file)
        ingested = ingestion.load(path, file.filename or path.name, file.content_type or "application/octet-stream")
        chunks = AdaptiveChunker().split(ingested.documents, ingested.metadata.document_id)
        ChromaVectorStore().add_documents(chunks)
        metadata = ingested.metadata.model_copy(update={"chunks": len(chunks)})
        _save_metadata(metadata)
        return metadata
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected upload failure for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@router.get("", response_model=list[DocumentMetadata])
def list_documents() -> list[DocumentMetadata]:
    return sorted(_load_metadata(), key=lambda doc: doc.uploaded_at, reverse=True)


@router.delete("/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    vector_status = "deleted"
    try:
        ChromaVectorStore().delete_document(document_id)
    except Exception as exc:
        vector_status = "cleanup_skipped"
        logger.warning("Vector cleanup failed for %s: %s", document_id, exc)
    docs = [doc for doc in _load_metadata() if doc.document_id != document_id]
    _write_metadata(docs)
    return {"status": "deleted", "document_id": document_id, "vector_status": vector_status}


def _load_metadata() -> list[DocumentMetadata]:
    if not metadata_file.exists():
        return []
    return [DocumentMetadata(**item) for item in json.loads(metadata_file.read_text(encoding="utf-8"))]


def _save_metadata(metadata: DocumentMetadata) -> None:
    docs = [doc for doc in _load_metadata() if doc.document_id != metadata.document_id]
    docs.append(metadata)
    _write_metadata(docs)


def _write_metadata(docs: list[DocumentMetadata]) -> None:
    get_settings().upload_directory.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    sorted_docs = sorted(docs, key=lambda doc: doc.uploaded_at, reverse=True)
    metadata_file.write_text(json.dumps([doc.model_dump(mode="json") for doc in sorted_docs], indent=2), encoding="utf-8")
