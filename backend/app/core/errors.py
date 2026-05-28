from __future__ import annotations


class AdvancedRAGError(Exception):
    """Base exception for domain-specific failures."""


class DocumentProcessingError(AdvancedRAGError):
    """Raised when a document cannot be extracted or chunked."""


class RetrievalError(AdvancedRAGError):
    """Raised when retrieval cannot be completed."""


class GenerationError(AdvancedRAGError):
    """Raised when the LLM call fails."""
