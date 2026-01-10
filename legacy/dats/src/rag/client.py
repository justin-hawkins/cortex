"""
RAG client for DATS.

Provides embedding generation and file-based vector storage.
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RAGDocument:
    """Document stored in the RAG system."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Source tracking
    task_id: Optional[str] = None
    provenance_id: Optional[str] = None
    project_id: Optional[str] = None
    domain: Optional[str] = None
    
    # Document type
    doc_type: str = "output"  # output, context, reference
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    embedded_at: Optional[datetime] = None
    
    # Content hash for deduplication
    content_hash: Optional[str] = None

    def __post_init__(self):
        """Compute content hash if not provided."""
        if self.content and not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "task_id": self.task_id,
            "provenance_id": self.provenance_id,
            "project_id": self.project_id,
            "domain": self.domain,
            "doc_type": self.doc_type,
            "created_at": self.created_at.isoformat(),
            "embedded_at": self.embedded_at.isoformat() if self.embedded_at else None,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGDocument":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            task_id=data.get("task_id"),
            provenance_id=data.get("provenance_id"),
            project_id=data.get("project_id"),
            domain=data.get("domain"),
            doc_type=data.get("doc_type", "output"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            embedded_at=datetime.fromisoformat(data["embedded_at"]) if data.get("embedded_at") else None,
            content_hash=data.get("content_hash"),
        )


@dataclass
class SearchResult:
    """Result from a vector search."""

    document: RAGDocument
    score: float
    rank: int


class EmbeddingClient:
    """
    Client for generating embeddings via Ollama.

    Uses mxbai-embed-large:335m model for high-quality embeddings.
    Server configuration is centralized in servers.yaml.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
    ):
        """
        Initialize embedding client.

        Args:
            endpoint: Ollama API endpoint (defaults to settings.rag_embedding_endpoint)
            model_name: Embedding model name (defaults to settings.rag_embedding_model)
            timeout: Request timeout in seconds
        """
        # Import settings lazily to avoid circular imports
        from src.config.settings import get_settings
        settings = get_settings()
        
        if endpoint is None:
            endpoint = settings.rag_embedding_endpoint
        if model_name is None:
            model_name = settings.rag_embedding_model
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_dim: Optional[int] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            List of embedding floats
        """
        client = await self._get_client()

        response = await client.post(
            f"{self.endpoint}/api/embeddings",
            json={
                "model": self.model_name,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding", [])
        
        # Cache embedding dimension
        if embedding and self._embedding_dim is None:
            self._embedding_dim = len(embedding)
            logger.info(f"Embedding dimension: {self._embedding_dim}")

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Note: Ollama doesn't have native batch embedding, so we process sequentially.
        For production, consider batching with asyncio.gather.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = await self.embed(text)
            embeddings.append(embedding)
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (default for mxbai-embed-large)."""
        return self._embedding_dim or 1024

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple character-based estimation.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Average of ~4 characters per token for English text
        return len(text) // 4


class FileVectorStore:
    """
    File-based vector storage for RAG documents.

    Stores embeddings as numpy arrays and metadata as JSON.
    Uses cosine similarity for search.
    """

    def __init__(self, storage_path: str):
        """
        Initialize file vector store.

        Args:
            storage_path: Base directory for storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage files
        self._index_path = self.storage_path / "index.json"
        self._embeddings_path = self.storage_path / "embeddings.npy"
        self._documents_path = self.storage_path / "documents.json"

        # In-memory cache
        self._documents: dict[str, RAGDocument] = {}
        self._embeddings: Optional[np.ndarray] = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}

        # Load existing data
        self._load()

    def _load(self):
        """Load existing data from disk."""
        # Load index
        if self._index_path.exists():
            with open(self._index_path) as f:
                index = json.load(f)
                self._id_to_idx = index.get("id_to_idx", {})
                self._idx_to_id = {int(k): v for k, v in index.get("idx_to_id", {}).items()}

        # Load documents
        if self._documents_path.exists():
            with open(self._documents_path) as f:
                docs_data = json.load(f)
                for doc_id, doc_dict in docs_data.items():
                    self._documents[doc_id] = RAGDocument.from_dict(doc_dict)

        # Load embeddings
        if self._embeddings_path.exists():
            self._embeddings = np.load(self._embeddings_path)
            logger.info(f"Loaded {len(self._embeddings)} embeddings from disk")

    def _save(self):
        """Save data to disk."""
        # Save index
        index = {
            "id_to_idx": self._id_to_idx,
            "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
            "updated_at": datetime.utcnow().isoformat(),
        }
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

        # Save documents
        docs_data = {doc_id: doc.to_dict() for doc_id, doc in self._documents.items()}
        with open(self._documents_path, "w") as f:
            json.dump(docs_data, f, indent=2)

        # Save embeddings
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.save(self._embeddings_path, self._embeddings)

    def add(
        self,
        document: RAGDocument,
        embedding: list[float],
    ) -> str:
        """
        Add a document with its embedding.

        Args:
            document: Document to store
            embedding: Document embedding

        Returns:
            Document ID
        """
        # Check for duplicate by content hash
        for existing_doc in self._documents.values():
            if existing_doc.content_hash == document.content_hash:
                logger.debug(f"Duplicate document detected: {document.content_hash[:16]}")
                return existing_doc.id

        # Add to documents
        document.embedded_at = datetime.utcnow()
        self._documents[document.id] = document

        # Add to embeddings
        embedding_array = np.array(embedding, dtype=np.float32)
        
        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = embedding_array.reshape(1, -1)
        else:
            self._embeddings = np.vstack([self._embeddings, embedding_array])

        # Update index
        idx = len(self._id_to_idx)
        self._id_to_idx[document.id] = idx
        self._idx_to_id[idx] = document.id

        # Persist
        self._save()

        logger.info(f"Added document {document.id} (total: {len(self._documents)})")
        return document.id

    def add_batch(
        self,
        documents: list[RAGDocument],
        embeddings: list[list[float]],
    ) -> list[str]:
        """
        Add multiple documents with embeddings.

        Args:
            documents: Documents to store
            embeddings: Document embeddings

        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")

        ids = []
        new_embeddings = []
        start_idx = len(self._id_to_idx)

        for doc, emb in zip(documents, embeddings):
            # Check for duplicate
            is_duplicate = False
            for existing_doc in self._documents.values():
                if existing_doc.content_hash == doc.content_hash:
                    ids.append(existing_doc.id)
                    is_duplicate = True
                    break

            if not is_duplicate:
                doc.embedded_at = datetime.utcnow()
                self._documents[doc.id] = doc
                
                idx = start_idx + len(new_embeddings)
                self._id_to_idx[doc.id] = idx
                self._idx_to_id[idx] = doc.id
                
                new_embeddings.append(emb)
                ids.append(doc.id)

        # Add new embeddings
        if new_embeddings:
            new_array = np.array(new_embeddings, dtype=np.float32)
            if self._embeddings is None or len(self._embeddings) == 0:
                self._embeddings = new_array
            else:
                self._embeddings = np.vstack([self._embeddings, new_array])

        # Persist
        self._save()

        logger.info(f"Added {len(new_embeddings)} documents (total: {len(self._documents)})")
        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        domain: Optional[str] = None,
        project_id: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Uses cosine similarity.

        Args:
            query_embedding: Query vector
            k: Number of results
            domain: Optional domain filter
            project_id: Optional project filter
            doc_type: Optional document type filter

        Returns:
            List of search results
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        # Compute cosine similarities
        query_array = np.array(query_embedding, dtype=np.float32)
        query_norm = query_array / (np.linalg.norm(query_array) + 1e-8)
        
        embeddings_norm = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1]

        # Filter and collect results
        results = []
        for idx in top_indices:
            if len(results) >= k:
                break

            idx = int(idx)
            doc_id = self._idx_to_id.get(idx)
            if not doc_id:
                continue

            doc = self._documents.get(doc_id)
            if not doc:
                continue

            # Apply filters
            if domain and doc.domain != domain:
                continue
            if project_id and doc.project_id != project_id:
                continue
            if doc_type and doc.doc_type != doc_type:
                continue

            results.append(SearchResult(
                document=doc,
                score=float(similarities[idx]),
                rank=len(results) + 1,
            ))

        return results

    def get(self, document_id: str) -> Optional[RAGDocument]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document if found
        """
        return self._documents.get(document_id)

    def delete(self, document_id: str) -> bool:
        """
        Delete a document.

        Note: This marks the document as deleted but doesn't compact storage.
        Call compact() periodically to reclaim space.

        Args:
            document_id: Document to delete

        Returns:
            True if deleted
        """
        if document_id not in self._documents:
            return False

        # Remove from documents
        del self._documents[document_id]

        # Mark index entry as deleted (set to None)
        if document_id in self._id_to_idx:
            idx = self._id_to_idx[document_id]
            del self._id_to_idx[document_id]
            if idx in self._idx_to_id:
                del self._idx_to_id[idx]

        self._save()
        logger.info(f"Deleted document {document_id}")
        return True

    def delete_by_provenance(self, provenance_ids: list[str]) -> int:
        """
        Delete all documents associated with provenance IDs.

        Used for invalidation when outputs become tainted.

        Args:
            provenance_ids: List of provenance IDs to invalidate

        Returns:
            Number of documents deleted
        """
        provenance_set = set(provenance_ids)
        to_delete = [
            doc_id for doc_id, doc in self._documents.items()
            if doc.provenance_id in provenance_set
        ]

        for doc_id in to_delete:
            self.delete(doc_id)

        logger.info(f"Deleted {len(to_delete)} documents for {len(provenance_ids)} provenance IDs")
        return len(to_delete)

    def compact(self):
        """
        Compact storage by rebuilding embeddings array.

        Removes gaps from deleted documents.
        """
        if not self._documents:
            self._embeddings = None
            self._id_to_idx = {}
            self._idx_to_id = {}
            self._save()
            return

        # Rebuild embeddings for existing documents
        new_embeddings = []
        new_id_to_idx = {}
        new_idx_to_id = {}

        for doc_id in self._documents:
            if doc_id in self._id_to_idx:
                old_idx = self._id_to_idx[doc_id]
                if self._embeddings is not None and old_idx < len(self._embeddings):
                    new_idx = len(new_embeddings)
                    new_embeddings.append(self._embeddings[old_idx])
                    new_id_to_idx[doc_id] = new_idx
                    new_idx_to_id[new_idx] = doc_id

        self._embeddings = np.array(new_embeddings, dtype=np.float32) if new_embeddings else None
        self._id_to_idx = new_id_to_idx
        self._idx_to_id = new_idx_to_id

        self._save()
        logger.info(f"Compacted storage: {len(self._documents)} documents")

    def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_documents": len(self._documents),
            "total_embeddings": len(self._embeddings) if self._embeddings is not None else 0,
            "embedding_dimension": self._embeddings.shape[1] if self._embeddings is not None and len(self._embeddings) > 0 else None,
            "domains": list(set(d.domain for d in self._documents.values() if d.domain)),
            "projects": list(set(d.project_id for d in self._documents.values() if d.project_id)),
            "storage_path": str(self.storage_path),
        }