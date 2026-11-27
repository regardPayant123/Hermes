"""Persistent FAISS vector index with SQLite chunk and metadata storage."""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from .document import DocumentChunk, RetrievedChunk
from .embeddings import EmbeddingProvider, SentenceTransformerEmbedder


class FaissSQLiteVectorStore:
    def __init__(
        self,
        database_path: str = "data/rag/chunks.sqlite3",
        index_path: str | None = None,
        embedding_model: str = "intfloat/multilingual-e5-small",
        device: str | None = "cpu",
        *,
        embedder: EmbeddingProvider | None = None,
    ):
        self.database_path = Path(database_path)
        self.index_path = Path(index_path) if index_path else self.database_path.with_suffix(".faiss")
        self.embedder = embedder or SentenceTransformerEmbedder(embedding_model, device)
        self.embedding_model = getattr(self.embedder, "model_name", embedding_model)
        self._lock = asyncio.Lock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL
                )"""
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            stored_model = connection.execute(
                "SELECT value FROM settings WHERE key='embedding_model'"
            ).fetchone()
            if stored_model and stored_model["value"] != self.embedding_model:
                record_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                if record_count:
                    raise ValueError(
                        "This vector store uses a different embedding model; choose a new database path or re-embed its documents"
                    )
            connection.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('embedding_model', ?)",
                (self.embedding_model,),
            )

    @staticmethod
    def _dependencies():
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Install knowledge dependencies with: pip install -r requirements-knowledge.txt"
            ) from exc
        return faiss, np

    def _load_or_rebuild_index(self, dimension: int, faiss, np):
        with self._connect() as connection:
            expected_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        if self.index_path.is_file():
            try:
                index = faiss.read_index(str(self.index_path))
                if index.d == dimension and index.ntotal == expected_count:
                    return index
            except RuntimeError:
                pass

        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        if expected_count:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT id, embedding, dimension FROM chunks ORDER BY id"
                ).fetchall()
            if any(row["dimension"] != dimension for row in rows):
                raise ValueError("Stored embeddings do not match the configured embedding dimension")
            vectors = np.vstack([
                np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
            ])
            ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
            index.add_with_ids(vectors, ids)
        self._save_index(index, faiss)
        return index

    def _save_index(self, index, faiss) -> None:
        temporary_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        faiss.write_index(index, str(temporary_path))
        temporary_path.replace(self.index_path)

    @staticmethod
    def _normalize_vectors(vectors: list[list[float]], np):
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("embedder must return a non-empty 2D vector matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embeddings cannot be zero vectors")
        return np.ascontiguousarray(matrix / norms, dtype=np.float32)

    async def add_many(self, chunks: list[DocumentChunk]) -> list[str]:
        if not chunks:
            return []
        vectors = await self.embedder.embed_documents([chunk.content for chunk in chunks])
        faiss, np = self._dependencies()
        matrix = self._normalize_vectors(vectors, np)
        if matrix.shape[0] != len(chunks):
            raise ValueError("embedder returned a different number of vectors than input chunks")

        async with self._lock:
            return await asyncio.to_thread(self._add_many, chunks, matrix, faiss, np)

    def _add_many(self, chunks: list[DocumentChunk], matrix, faiss, np) -> list[str]:
        index = self._load_or_rebuild_index(matrix.shape[1], faiss, np)
        records = []
        for chunk, vector in zip(chunks, matrix):
            records.append((
                chunk.content,
                chunk.source,
                json.dumps(chunk.metadata, ensure_ascii=False),
                vector.tobytes(),
                int(matrix.shape[1]),
            ))

        with self._connect() as connection:
            ids = []
            for record in records:
                cursor = connection.execute(
                    "INSERT INTO chunks (content, source, metadata, embedding, dimension) VALUES (?, ?, ?, ?, ?)",
                    record,
                )
                ids.append(cursor.lastrowid)

        index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))
        self._save_index(index, faiss)
        return [str(record_id) for record_id in ids]

    async def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        vector = await self.embedder.embed_query(query)
        faiss, np = self._dependencies()
        matrix = self._normalize_vectors([vector], np)
        async with self._lock:
            return await asyncio.to_thread(self._search, matrix, limit, faiss, np)

    def _search(self, matrix, limit: int, faiss, np) -> list[RetrievedChunk]:
        index = self._load_or_rebuild_index(matrix.shape[1], faiss, np)
        if not index.ntotal:
            return []
        scores, identifiers = index.search(matrix, limit)
        ids = [int(identifier) for identifier in identifiers[0] if identifier >= 0]
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, content, source, metadata FROM chunks WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        results = []
        for identifier, score in zip(identifiers[0], scores[0]):
            row = by_id.get(int(identifier))
            if row is None:
                continue
            results.append(RetrievedChunk(
                content=row["content"],
                source=row["source"],
                score=float(score),
                metadata=json.loads(row["metadata"]),
            ))
        return results

    async def clear(self) -> None:
        faiss, _ = self._dependencies()
        async with self._lock:
            await asyncio.to_thread(self._clear, faiss)

    def _clear(self, faiss) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")
        if self.index_path.exists():
            self.index_path.unlink()