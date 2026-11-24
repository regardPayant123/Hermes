"""Retrieval-Augmented Generation pipeline."""

from dataclasses import replace

from ..memory.interface import MemoryRecord
from ..memory.long_term.interface import LongTermMemory
from .document import Document, RetrievedChunk
from .chunker import TextChunker


class RAGPipeline:
    def __init__(
        self,
        long_term_memory: LongTermMemory,
        chunker: TextChunker | None = None,
    ):
        self.long_term_memory = long_term_memory
        self.chunker = chunker or TextChunker()

    async def ingest(self, document: Document) -> list[str]:
        record_ids = []
        for chunk in self.chunker.split(document):
            record = MemoryRecord(
                content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "source": document.source,
                    "document_type": "rag_chunk",
                },
            )
            record_ids.append(await self.long_term_memory.add(record))
        return record_ids

    async def retrieve(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        records = await self.long_term_memory.search(query, limit=limit)
        query_terms = {term.lower() for term in query.lower().split() if term}
        chunks = []
        for record in records:
            text = record.content.lower()
            matches = sum(term in text for term in query_terms)
            score = matches / len(query_terms) if query_terms else 0.0
            chunks.append(RetrievedChunk(
                content=record.content,
                source=str(record.metadata.get("source", "")),
                score=score,
                metadata=record.metadata,
            ))
        return chunks

    async def build_context(self, query: str, limit: int = 5) -> str:
        chunks = await self.retrieve(query, limit=limit)
        if not chunks:
            return ""

        sections = []
        for index, chunk in enumerate(chunks, start=1):
            source = f" [{chunk.source}]" if chunk.source else ""
            sections.append(f"Source {index}{source}:\n{chunk.content}")
        return "\n\n".join(sections)
