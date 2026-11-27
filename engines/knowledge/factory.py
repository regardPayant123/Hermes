"""Configuration factory for memory and RAG components."""

import json
from pathlib import Path
from typing import Any


class KnowledgeFactory:
    @staticmethod
    def load_config(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Knowledge configuration not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Knowledge configuration must contain a JSON object")
        return config

    @classmethod
    def from_json(cls, path: str | Path):
        config = cls.load_config(path)
        short_options = config.get("short_term", {})
        long_options = config.get("long_term", {})
        rag_options = config.get("rag", {})
        if not all(isinstance(options, dict) for options in (short_options, long_options, rag_options)):
            raise ValueError("short_term, long_term, and rag configuration must be JSON objects")
        vector_options = rag_options.get("vector_store", {})
        chunker_options = rag_options.get("chunker", {})
        if not isinstance(vector_options, dict) or not isinstance(chunker_options, dict):
            raise ValueError("rag.vector_store and rag.chunker must be JSON objects")

        from .memory.short_term.buffer import BufferShortTermMemory
        from .memory.long_term.json_store import JsonLongTermMemory
        from .rag.chunker import TextChunker
        from .rag.rag import RAGPipeline
        from .rag.vector_store import FaissSQLiteVectorStore

        short_term = BufferShortTermMemory(**short_options)
        long_term = JsonLongTermMemory(**long_options)
        chunker = TextChunker(**chunker_options)
        vector_store = FaissSQLiteVectorStore(**vector_options)
        rag = RAGPipeline(
            vector_store,
            chunker,
            retrieval_limit=rag_options.get("retrieval_limit", 5),
        )
        return {
            "short_term": short_term,
            "long_term": long_term,
            "vector_store": vector_store,
            "rag": rag,
        }
