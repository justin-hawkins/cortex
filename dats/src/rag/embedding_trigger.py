"""
Embedding trigger for DATS.

Provides event-driven batch embedding for task outputs.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.config.settings import get_settings
from src.rag.client import EmbeddingClient, FileVectorStore, RAGDocument

logger = logging.getLogger(__name__)


@dataclass
class PendingDocument:
    """Document pending embedding."""

    document: RAGDocument
    queued_at: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0  # Higher = more urgent

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document": self.document.to_dict(),
            "queued_at": self.queued_at.isoformat(),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingDocument":
        """Create from dictionary."""
        return cls(
            document=RAGDocument.from_dict(data["document"]),
            queued_at=datetime.fromisoformat(data["queued_at"]) if data.get("queued_at") else datetime.utcnow(),
            priority=data.get("priority", 0),
        )


@dataclass
class EmbeddingQueue:
    """Queue of documents pending embedding."""

    pending: list[PendingDocument] = field(default_factory=list)
    last_processed: Optional[datetime] = None
    total_processed: int = 0
    total_failed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pending": [p.to_dict() for p in self.pending],
            "last_processed": self.last_processed.isoformat() if self.last_processed else None,
            "total_processed": self.total_processed,
            "total_failed": self.total_failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingQueue":
        """Create from dictionary."""
        return cls(
            pending=[PendingDocument.from_dict(p) for p in data.get("pending", [])],
            last_processed=datetime.fromisoformat(data["last_processed"]) if data.get("last_processed") else None,
            total_processed=data.get("total_processed", 0),
            total_failed=data.get("total_failed", 0),
        )


class EmbeddingTrigger:
    """
    Event-driven embedding trigger.

    Batches documents and triggers embedding based on:
    - Batch size threshold (default: 10 items)
    - Time interval (default: 5 minutes)
    - Manual trigger after QA pass

    Also handles invalidation for tainted outputs.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        batch_interval: Optional[int] = None,
        batch_size_threshold: Optional[int] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize embedding trigger.

        Args:
            storage_path: Base path for RAG storage (defaults to settings.rag_storage_path)
            batch_interval: Seconds between batch processing (defaults to settings.rag_batch_interval)
            batch_size_threshold: Number of items to trigger immediate batch (defaults to settings.rag_batch_size)
            embedding_endpoint: Ollama endpoint for embeddings (defaults to settings.rag_embedding_endpoint)
            embedding_model: Embedding model name (defaults to settings.rag_embedding_model)
        """
        settings = get_settings()
        
        # Use settings as defaults - server config is centralized in servers.yaml
        self.storage_path = Path(storage_path or settings.rag_storage_path)
        self.batch_interval = batch_interval if batch_interval is not None else settings.rag_batch_interval
        self.batch_size_threshold = batch_size_threshold if batch_size_threshold is not None else settings.rag_batch_size
        
        # Queue storage
        self._queue_path = self.storage_path / "embedding_queue.json"
        self._queue = self._load_queue()
        
        # Clients (lazy initialized)
        self._embedding_client: Optional[EmbeddingClient] = None
        self._vector_store: Optional[FileVectorStore] = None
        self._embedding_endpoint = embedding_endpoint or settings.rag_embedding_endpoint
        self._embedding_model = embedding_model or settings.rag_embedding_model

    def _load_queue(self) -> EmbeddingQueue:
        """Load queue from disk."""
        if self._queue_path.exists():
            with open(self._queue_path) as f:
                data = json.load(f)
                return EmbeddingQueue.from_dict(data)
        return EmbeddingQueue()

    def _save_queue(self):
        """Save queue to disk."""
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._queue_path, "w") as f:
            json.dump(self._queue.to_dict(), f, indent=2)

    def _get_embedding_client(self) -> EmbeddingClient:
        """Get or create embedding client."""
        if self._embedding_client is None:
            self._embedding_client = EmbeddingClient(
                endpoint=self._embedding_endpoint,
                model_name=self._embedding_model,
            )
        return self._embedding_client

    def _get_vector_store(self) -> FileVectorStore:
        """Get or create vector store."""
        if self._vector_store is None:
            self._vector_store = FileVectorStore(str(self.storage_path))
        return self._vector_store

    def add_pending(
        self,
        document: RAGDocument,
        priority: int = 0,
    ):
        """
        Add a document to the pending queue.

        Args:
            document: Document to queue for embedding
            priority: Priority level (higher = more urgent)
        """
        pending = PendingDocument(
            document=document,
            priority=priority,
        )
        self._queue.pending.append(pending)
        self._save_queue()

        logger.info(f"Queued document {document.id} for embedding (queue size: {len(self._queue.pending)})")

        # Check if batch threshold met
        if len(self._queue.pending) >= self.batch_size_threshold:
            logger.info(f"Batch threshold reached ({self.batch_size_threshold}), triggering immediate processing")

    def add_task_output(
        self,
        task_id: str,
        provenance_id: str,
        content: str,
        project_id: Optional[str] = None,
        domain: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Queue task output for embedding.

        Convenience method for adding completed task outputs.

        Args:
            task_id: Task ID that produced the output
            provenance_id: Provenance record ID
            content: Output content to embed
            project_id: Associated project
            domain: Task domain
            metadata: Additional metadata
        """
        document = RAGDocument(
            content=content,
            task_id=task_id,
            provenance_id=provenance_id,
            project_id=project_id,
            domain=domain,
            doc_type="output",
            metadata=metadata or {},
        )
        self.add_pending(document, priority=1)

    def should_process(self) -> bool:
        """
        Check if batch processing should run.

        Returns:
            True if threshold or interval conditions are met
        """
        if not self._queue.pending:
            return False

        # Check size threshold
        if len(self._queue.pending) >= self.batch_size_threshold:
            return True

        # Check time interval
        if self._queue.last_processed:
            elapsed = datetime.utcnow() - self._queue.last_processed
            if elapsed >= timedelta(seconds=self.batch_interval):
                return True
        else:
            # Never processed, check if any items are old enough
            oldest = min(p.queued_at for p in self._queue.pending)
            elapsed = datetime.utcnow() - oldest
            if elapsed >= timedelta(seconds=self.batch_interval):
                return True

        return False

    async def process_batch(self, max_items: Optional[int] = None) -> dict[str, Any]:
        """
        Process pending documents.

        Args:
            max_items: Maximum items to process (None = all)

        Returns:
            Processing result statistics
        """
        if not self._queue.pending:
            return {"processed": 0, "failed": 0, "remaining": 0}

        # Sort by priority (highest first) then by queue time
        self._queue.pending.sort(
            key=lambda p: (-p.priority, p.queued_at)
        )

        # Get items to process
        to_process = self._queue.pending[:max_items] if max_items else self._queue.pending
        remaining = self._queue.pending[len(to_process):] if max_items else []

        client = self._get_embedding_client()
        store = self._get_vector_store()

        processed = 0
        failed = 0
        documents = []
        embeddings = []

        try:
            # Generate embeddings
            for pending in to_process:
                try:
                    embedding = await client.embed(pending.document.content)
                    documents.append(pending.document)
                    embeddings.append(embedding)
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to embed document {pending.document.id}: {e}")
                    failed += 1

            # Store in vector store
            if documents:
                store.add_batch(documents, embeddings)
                logger.info(f"Stored {len(documents)} embeddings")

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            failed += len(to_process) - processed

        # Update queue
        self._queue.pending = remaining
        self._queue.last_processed = datetime.utcnow()
        self._queue.total_processed += processed
        self._queue.total_failed += failed
        self._save_queue()

        result = {
            "processed": processed,
            "failed": failed,
            "remaining": len(remaining),
            "total_processed": self._queue.total_processed,
            "total_failed": self._queue.total_failed,
        }
        logger.info(f"Batch processing complete: {result}")
        return result

    async def invalidate(self, provenance_ids: list[str]) -> int:
        """
        Invalidate embeddings for tainted outputs.

        Removes documents from vector store and any pending queue items.

        Args:
            provenance_ids: Provenance IDs to invalidate

        Returns:
            Number of documents removed
        """
        provenance_set = set(provenance_ids)
        
        # Remove from pending queue
        original_pending = len(self._queue.pending)
        self._queue.pending = [
            p for p in self._queue.pending
            if p.document.provenance_id not in provenance_set
        ]
        removed_pending = original_pending - len(self._queue.pending)
        
        if removed_pending:
            self._save_queue()
            logger.info(f"Removed {removed_pending} pending items for invalidation")

        # Remove from vector store
        store = self._get_vector_store()
        removed_stored = store.delete_by_provenance(provenance_ids)

        total_removed = removed_pending + removed_stored
        logger.info(f"Invalidated {total_removed} documents ({removed_pending} pending, {removed_stored} stored)")
        return total_removed

    def queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            "pending_count": len(self._queue.pending),
            "last_processed": self._queue.last_processed.isoformat() if self._queue.last_processed else None,
            "total_processed": self._queue.total_processed,
            "total_failed": self._queue.total_failed,
            "should_process": self.should_process(),
            "batch_size_threshold": self.batch_size_threshold,
            "batch_interval_seconds": self.batch_interval,
        }

    async def close(self):
        """Clean up resources."""
        if self._embedding_client:
            await self._embedding_client.close()


# Celery task wrappers
# These are defined here to avoid circular imports with the main tasks module

def _get_trigger() -> EmbeddingTrigger:
    """Get embedding trigger instance using centralized settings."""
    # All settings come from settings.py which references servers.yaml
    return EmbeddingTrigger()


def trigger_embedding_for_task(
    task_id: str,
    provenance_id: str,
    content: str,
    project_id: Optional[str] = None,
    domain: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
):
    """
    Queue task output for embedding (synchronous wrapper).

    Call this after successful QA validation.

    Args:
        task_id: Task ID
        provenance_id: Provenance record ID
        content: Output content
        project_id: Project ID
        domain: Task domain
        metadata: Additional metadata
    """
    trigger = _get_trigger()
    trigger.add_task_output(
        task_id=task_id,
        provenance_id=provenance_id,
        content=content,
        project_id=project_id,
        domain=domain,
        metadata=metadata,
    )


def process_embedding_batch_sync() -> dict[str, Any]:
    """
    Process embedding batch (synchronous wrapper).

    Call this from Celery beat or manually.

    Returns:
        Processing statistics
    """
    trigger = _get_trigger()
    
    # Run async in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(trigger.process_batch())
    finally:
        loop.run_until_complete(trigger.close())

    return result


def invalidate_embeddings_sync(provenance_ids: list[str]) -> int:
    """
    Invalidate embeddings (synchronous wrapper).

    Call this when outputs become tainted.

    Args:
        provenance_ids: Provenance IDs to invalidate

    Returns:
        Number of documents removed
    """
    trigger = _get_trigger()
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        result = loop.run_until_complete(trigger.invalidate(provenance_ids))
    finally:
        loop.run_until_complete(trigger.close())

    return result
