from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

from app.core.config import get_settings


class AdaptiveChunker:
    def __init__(self) -> None:
        self.settings = get_settings()

    def split(self, documents: list[Document], document_id: str) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            chunk_size = self._chunk_size_for(doc.page_content)
            chunk_profile = self._profile_for(doc.page_content)
            structural_docs = self._structure_aware_documents(doc)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=min(self.settings.chunk_overlap, chunk_size // 4),
                separators=["\n## ", "\n# ", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            )
            page_chunks = splitter.split_documents(structural_docs)
            for index, chunk in enumerate(page_chunks):
                chunk.metadata.update(
                    {
                        "document_id": document_id,
                        "chunk_index": index,
                        "chunk_id": f"{document_id}:{chunk.metadata.get('page', 1)}:{index}",
                        "char_count": len(chunk.page_content),
                        "chunk_size_target": chunk_size,
                        "chunk_profile": chunk_profile,
                        "adaptive_overlap": min(self.settings.chunk_overlap, chunk_size // 4),
                        "structure_aware": True,
                        "section_title": chunk.metadata.get("section_title"),
                    }
                )
            chunks.extend(page_chunks)
        return chunks

    def _chunk_size_for(self, text: str) -> int:
        length = len(text)
        if length < 3_000:
            return self.settings.chunk_min_chars
        if length > 25_000:
            return self.settings.chunk_max_chars
        return int((self.settings.chunk_min_chars + self.settings.chunk_max_chars) / 2)

    def _structure_aware_documents(self, doc: Document) -> list[Document]:
        sections = self._split_sections(doc.page_content)
        if len(sections) <= 1:
            return [doc]
        return [
            Document(
                page_content=section_text,
                metadata={**doc.metadata, "section_title": section_title},
            )
            for section_title, section_text in sections
            if section_text.strip()
        ]

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        heading_pattern = re.compile(
            r"(?m)^(#{1,3}\s+.+|[A-Z][A-Za-z0-9 ,:&/\-]{3,80}:?|(?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z0-9 ,:&/\-]{3,80})$"
        )
        matches = list(heading_pattern.finditer(text))
        if len(matches) < 2:
            return [("body", text)]
        sections: list[tuple[str, str]] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            title = match.group(0).strip("# :")
            sections.append((title, text[start:end].strip()))
        if matches[0].start() > 0:
            sections.insert(0, ("introduction", text[: matches[0].start()].strip()))
        return sections

    def _profile_for(self, text: str) -> str:
        length = len(text)
        if length < 3_000:
            return "short_document_precision"
        if length > 25_000:
            return "long_document_recall"
        return "balanced_document"
