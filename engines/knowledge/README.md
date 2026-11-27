# Knowledge and RAG

Install the optional local vector-search dependencies with:

```bash
pip install -r requirements-knowledge.txt
```

The RAG pipeline uses `intfloat/multilingual-e5-small` by default, with E5 query/document prefixes and normalized vectors. FAISS performs cosine-similarity search through an inner-product index. SQLite stores chunk text, source, metadata, model identity, and normalized vectors so the FAISS index can be rebuilt after an interrupted write.

Configure the SQLite path, FAISS index path, embedding model, and device under `rag.vector_store` in `configs/knowledge.json`. Embedding weights are loaded lazily on the first ingestion or retrieval. The first use may download the model; set a local SentenceTransformers model path and configure offline model loading when downloads are not desired.

Document RAG is separate from personal long-term memory: `long_term` remains the JSON-backed memory store, while `rag.vector_store` indexes document chunks. The vector store accepts an injected `EmbeddingProvider`, which allows another embedding model or a hosted embedding service to be added without changing the RAG pipeline.

```python
from engines.knowledge.factory import KnowledgeFactory
from engines.knowledge.rag import Document

knowledge = KnowledgeFactory.from_json("configs/knowledge.json")
await knowledge["rag"].ingest(Document(
    content="Hermes stores searchable document chunks.",
    source="notes.md",
))
context = await knowledge["rag"].build_context("Where are document chunks stored?")
```