"""Embedding-backed Retrieval-Augmented Generation pipeline."""

from .document import Document, RetrievedChunk
from .chunker import TextChunker
from .vector_store import FaissSQLiteVectorStore


class RAGPipeline:
    def __init__(
        self,
        vector_store: FaissSQLiteVectorStore,
        chunker: TextChunker | None = None,
        retrieval_limit: int = 5,
    ):
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero")
        self.vector_store = vector_store
        self.chunker = chunker or TextChunker()
        self.retrieval_limit = retrieval_limit

    async def ingest(self, document: Document) -> list[str]:
        chunks = self.chunker.split(document)
        return await self.vector_store.add_many(chunks)

    async def retrieve(self, query: str, limit: int | None = None) -> list[RetrievedChunk]:
        return await self.vector_store.search(
            query,
            limit=self.retrieval_limit if limit is None else limit,
        )

    async def build_context(self, query: str, limit: int | None = None) -> str:
        chunks = await self.retrieve(query, limit=limit)
        if not chunks:
            return ""

        sections = []
        for index, chunk in enumerate(chunks, start=1):
            source = f" [{chunk.source}]" if chunk.source else ""
            sections.append(f"Source {index}{source}:\n{chunk.content}")
        return "\n\n".join(sections)
