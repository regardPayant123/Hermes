"""Embedding providers for semantic retrieval."""

import asyncio
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbedder(EmbeddingProvider):
    """SentenceTransformers embeddings with E5 query/document prefixes."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = "cpu",
    ):
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install knowledge dependencies with: pip install -r requirements-knowledge.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, [f"passage: {text}" for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query cannot be empty")
        vectors = await asyncio.to_thread(self._encode, [f"query: {text}"])
        return vectors[0]