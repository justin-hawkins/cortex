"""
RAG query engine for DATS.

Provides context retrieval for task execution.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.config.settings import get_settings
from src.rag.client import EmbeddingClient, FileVectorStore, RAGDocument, SearchResult

logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Query strategies for context retrieval."""

    SEMANTIC = "semantic"  # Pure embedding similarity
    KEYWORD = "keyword"  # Text-based filtering (future)
    HYBRID = "hybrid"  # Combination of both (future)
    DOMAIN_SPECIFIC = "domain_specific"  # Domain-aware retrieval


@dataclass
class QueryResult:
    """Result from a RAG query."""

    documents: list[RAGDocument] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    token_count_estimate: int = 0
    query_strategy: str = "semantic"
    query_text: str = ""
    filters_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def total_documents(self) -> int:
        """Total documents retrieved."""
        return len(self.documents)

    @property
    def avg_score(self) -> float:
        """Average relevance score."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def to_context_string(self, max_tokens: Optional[int] = None) -> str:
        """
        Convert to context string for prompt injection.

        Args:
            max_tokens: Maximum tokens to include

        Returns:
            Formatted context string
        """
        if not self.documents:
            return ""

        context_parts = []
        current_tokens = 0

        for doc, score in zip(self.documents, self.scores):
            # Estimate tokens for this document
            doc_tokens = len(doc.content) // 4

            if max_tokens and current_tokens + doc_tokens > max_tokens:
                break

            # Format document with metadata
            header = f"[Source: {doc.doc_type}"
            if doc.domain:
                header += f", Domain: {doc.domain}"
            header += f", Relevance: {score:.2f}]"

            context_parts.append(f"{header}\n{doc.content}")
            current_tokens += doc_tokens

        return "\n\n---\n\n".join(context_parts)


class RAGQueryEngine:
    """
    Query engine for RAG context retrieval.

    Retrieves relevant context for task execution based on
    query strategy and domain-specific filtering.
    """

    # Domain-specific query parameters
    DOMAIN_CONFIGS = {
        "code-general": {
            "k": 5,
            "min_score": 0.6,
            "doc_types": ["output", "reference"],
        },
        "code-vision": {
            "k": 3,
            "min_score": 0.65,
            "doc_types": ["output", "reference"],
        },
        "code-embedded": {
            "k": 4,
            "min_score": 0.6,
            "doc_types": ["output", "reference"],
        },
        "documentation": {
            "k": 6,
            "min_score": 0.55,
            "doc_types": ["output", "context", "reference"],
        },
        "ui-design": {
            "k": 4,
            "min_score": 0.6,
            "doc_types": ["output", "reference"],
        },
        "default": {
            "k": 5,
            "min_score": 0.5,
            "doc_types": ["output", "context", "reference"],
        },
    }

    def __init__(
        self,
        storage_path: Optional[str] = None,
        embedding_endpoint: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize query engine.

        Args:
            storage_path: Path to RAG storage (defaults to settings.rag_storage_path)
            embedding_endpoint: Ollama endpoint for embeddings (defaults to settings.rag_embedding_endpoint)
            embedding_model: Embedding model name (defaults to settings.rag_embedding_model)
        """
        settings = get_settings()

        # Use settings as defaults - server config is centralized in servers.yaml
        self._storage_path = storage_path or settings.rag_storage_path
        self._embedding_endpoint = embedding_endpoint or settings.rag_embedding_endpoint
        self._embedding_model = embedding_model or settings.rag_embedding_model

        # Lazy initialized clients
        self._embedding_client: Optional[EmbeddingClient] = None
        self._vector_store: Optional[FileVectorStore] = None

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
            self._vector_store = FileVectorStore(self._storage_path)
        return self._vector_store

    async def query(
        self,
        query_text: str,
        strategy: QueryStrategy = QueryStrategy.SEMANTIC,
        k: int = 5,
        min_score: float = 0.5,
        max_tokens: Optional[int] = None,
        domain: Optional[str] = None,
        project_id: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
    ) -> QueryResult:
        """
        Query for relevant context.

        Args:
            query_text: Query text
            strategy: Query strategy
            k: Number of results
            min_score: Minimum relevance score
            max_tokens: Maximum tokens to retrieve
            domain: Filter by domain
            project_id: Filter by project
            doc_types: Filter by document types

        Returns:
            QueryResult with matching documents
        """
        if strategy == QueryStrategy.DOMAIN_SPECIFIC:
            return await self._query_domain_specific(
                query_text=query_text,
                domain=domain or "default",
                project_id=project_id,
                max_tokens=max_tokens,
            )

        # Generate query embedding
        client = self._get_embedding_client()
        query_embedding = await client.embed(query_text)

        # Search vector store
        store = self._get_vector_store()

        # If filtering by doc_type, we need to search more and filter
        search_k = k * 3 if doc_types else k

        results = store.search(
            query_embedding=query_embedding,
            k=search_k,
            domain=domain,
            project_id=project_id,
        )

        # Filter by doc_types if specified
        if doc_types:
            results = [r for r in results if r.document.doc_type in doc_types]

        # Filter by min_score and limit to k
        filtered_results: list[SearchResult] = []
        for result in results:
            if result.score >= min_score:
                filtered_results.append(result)
                if len(filtered_results) >= k:
                    break

        # Build result
        documents = [r.document for r in filtered_results]
        scores = [r.score for r in filtered_results]

        # Estimate token count
        token_count = sum(len(doc.content) // 4 for doc in documents)

        # Apply max_tokens limit if specified
        if max_tokens and token_count > max_tokens:
            documents, scores, token_count = self._truncate_to_tokens(
                documents, scores, max_tokens
            )

        return QueryResult(
            documents=documents,
            scores=scores,
            token_count_estimate=token_count,
            query_strategy=strategy.value,
            query_text=query_text,
            filters_applied={
                "domain": domain,
                "project_id": project_id,
                "doc_types": doc_types,
                "min_score": min_score,
            },
        )

    async def _query_domain_specific(
        self,
        query_text: str,
        domain: str,
        project_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> QueryResult:
        """
        Query with domain-specific configuration.

        Args:
            query_text: Query text
            domain: Domain name
            project_id: Optional project filter
            max_tokens: Maximum tokens

        Returns:
            QueryResult with domain-appropriate documents
        """
        # Get domain config
        config = self.DOMAIN_CONFIGS.get(domain, self.DOMAIN_CONFIGS["default"])

        return await self.query(
            query_text=query_text,
            strategy=QueryStrategy.SEMANTIC,
            k=config["k"],
            min_score=config["min_score"],
            max_tokens=max_tokens,
            domain=domain if domain != "default" else None,
            project_id=project_id,
            doc_types=config["doc_types"],
        )

    async def query_for_task(
        self,
        task_data: dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> QueryResult:
        """
        Query for task-relevant context.

        Builds a query from task data and retrieves appropriate context.

        Args:
            task_data: Task configuration
            max_tokens: Maximum tokens for context

        Returns:
            QueryResult with task-relevant documents
        """
        # Extract task information
        description = task_data.get("description", "")
        domain = task_data.get("domain", "code-general")
        project_id = task_data.get("project_id")

        # Build query text from task
        query_parts = []

        if description:
            query_parts.append(description)

        # Add input content if available
        inputs = task_data.get("inputs", [])
        for inp in inputs[:2]:  # Limit to first 2 inputs
            if isinstance(inp, dict):
                content = inp.get("content", "")[:500]  # Limit content
                if content:
                    query_parts.append(content)
            elif isinstance(inp, str):
                query_parts.append(inp[:500])

        query_text = "\n".join(query_parts)

        if not query_text:
            # No query content, return empty result
            return QueryResult(query_strategy="domain_specific")

        return await self._query_domain_specific(
            query_text=query_text,
            domain=domain,
            project_id=project_id,
            max_tokens=max_tokens,
        )

    def _truncate_to_tokens(
        self,
        documents: list[RAGDocument],
        scores: list[float],
        max_tokens: int,
    ) -> tuple[list[RAGDocument], list[float], int]:
        """
        Truncate documents to fit within token limit.

        Args:
            documents: Documents to truncate
            scores: Corresponding scores
            max_tokens: Maximum tokens

        Returns:
            Tuple of (truncated_docs, truncated_scores, actual_tokens)
        """
        truncated_docs = []
        truncated_scores = []
        current_tokens = 0

        for doc, score in zip(documents, scores):
            doc_tokens = len(doc.content) // 4

            if current_tokens + doc_tokens > max_tokens:
                # Check if we can fit a partial document
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only include if we can fit meaningful content
                    # Truncate content
                    truncated_content = doc.content[: remaining_tokens * 4]
                    truncated_doc = RAGDocument(
                        id=doc.id,
                        content=truncated_content + "...[truncated]",
                        metadata=doc.metadata,
                        task_id=doc.task_id,
                        provenance_id=doc.provenance_id,
                        project_id=doc.project_id,
                        domain=doc.domain,
                        doc_type=doc.doc_type,
                    )
                    truncated_docs.append(truncated_doc)
                    truncated_scores.append(score)
                    current_tokens += remaining_tokens
                break

            truncated_docs.append(doc)
            truncated_scores.append(score)
            current_tokens += doc_tokens

        return truncated_docs, truncated_scores, current_tokens

    async def get_context_for_worker(
        self,
        task_data: dict[str, Any],
        safe_working_limit: int,
        context_budget_ratio: float = 0.3,
    ) -> str:
        """
        Get formatted context string for worker execution.

        Respects token budget for the model's context window.

        Args:
            task_data: Task configuration
            safe_working_limit: Safe working token limit for model
            context_budget_ratio: Fraction of limit to use for RAG context

        Returns:
            Formatted context string for prompt injection
        """
        # Calculate token budget for RAG context
        max_tokens = int(safe_working_limit * context_budget_ratio)

        result = await self.query_for_task(
            task_data=task_data,
            max_tokens=max_tokens,
        )

        if not result.documents:
            return ""

        context = result.to_context_string(max_tokens=max_tokens)

        logger.info(
            f"Retrieved RAG context: {result.total_documents} documents, "
            f"~{result.token_count_estimate} tokens"
        )

        return context

    async def close(self):
        """Clean up resources."""
        if self._embedding_client:
            await self._embedding_client.close()


# Convenience functions for synchronous usage

def get_query_engine() -> RAGQueryEngine:
    """Get configured query engine instance using centralized settings."""
    # All settings come from settings.py which references servers.yaml
    return RAGQueryEngine()
