"""Text chunking utilities for RAG ingestion."""

from .document import Document, DocumentChunk


class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[DocumentChunk]:
        text = " ".join(document.content.split())
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append(DocumentChunk(
                    content=content,
                    source=document.source,
                    metadata={**document.metadata, "chunk_index": chunk_index},
                ))
                chunk_index += 1
            if end == len(text):
                break
            start = end - self.overlap
        return chunks
