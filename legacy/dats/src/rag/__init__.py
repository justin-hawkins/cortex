"""
RAG (Retrieval-Augmented Generation) module for DATS.

Provides LightRAG integration for context retrieval and embedding management.
"""

from src.rag.client import (
    EmbeddingClient,
    FileVectorStore,
    RAGDocument,
    SearchResult,
)
from src.rag.embedding_trigger import (
    EmbeddingQueue,
    EmbeddingTrigger,
    PendingDocument,
)
from src.rag.query import (
    QueryResult,
    QueryStrategy,
    RAGQueryEngine,
)

__all__ = [
    # Client
    "EmbeddingClient",
    "FileVectorStore",
    "RAGDocument",
    "SearchResult",
    # Embedding Trigger
    "EmbeddingQueue",
    "EmbeddingTrigger",
    "PendingDocument",
    # Query
    "QueryResult",
    "QueryStrategy",
    "RAGQueryEngine",
]