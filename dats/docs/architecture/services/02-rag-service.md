# RAG Service

> **DATS Microservice** - Context Retrieval & Embedding Management  
> Priority: P1 (Second to Extract)  
> Team: Data  
> Status: Planned

---

## Overview

### Purpose

The RAG Service provides context retrieval and embedding management for DATS. It wraps LightRAG to enable:

- **Context Retrieval**: Query relevant context for task execution
- **Embedding Management**: Index approved outputs for future retrieval
- **Invalidation**: Remove embeddings when artifacts are tainted
- **Hybrid Search**: Combine vector similarity with keyword matching

### Current State (Monolith)

```
src/rag/
├── __init__.py
├── client.py           # LightRAG client wrapper
├── embedding_trigger.py # Embedding on approval
└── query.py            # RAGQueryEngine
```

**Problems:**
- Embedded in worker execution flow
- No independent scaling (GPU-bound)
- Invalidation tightly coupled to cascade logic

---

## API Specification

### Base URL

```
http://rag-service:8000/api/v1
```

### Endpoints

#### POST /query

Query for relevant context.

**Request:**
```json
{
  "query": "How to implement fibonacci in Python?",
  "project_id": "proj-123",
  "domain": "code-general",
  "max_tokens": 2000,
  "search_mode": "hybrid",
  "filters": {
    "verified_only": true,
    "exclude_tainted": true
  }
}
```

**Response:**
```json
{
  "id": "query-550e8400-e29b-41d4",
  "query": "How to implement fibonacci in Python?",
  "context": "## Previous Implementation\n\n```python\ndef fibonacci(n)...",
  "sources": [
    {
      "artifact_id": "art-123",
      "provenance_id": "prov-456",
      "relevance_score": 0.92,
      "snippet": "def fibonacci(n: int) -> int:..."
    }
  ],
  "tokens_used": 1500,
  "search_mode": "hybrid",
  "metadata": {
    "vector_results": 3,
    "keyword_results": 2,
    "latency_ms": 250
  }
}
```

#### POST /embed

Index content for future retrieval.

**Request:**
```json
{
  "content": "def fibonacci(n: int) -> int:\n    ...",
  "metadata": {
    "artifact_id": "art-789",
    "provenance_id": "prov-012",
    "task_id": "task-345",
    "project_id": "proj-123",
    "domain": "code-general",
    "doc_type": "output",
    "verified": true
  }
}
```

**Response:**
```json
{
  "id": "emb-550e8400-e29b-41d4",
  "artifact_id": "art-789",
  "status": "indexed",
  "chunks": 3,
  "tokens": 450,
  "latency_ms": 1200
}
```

#### DELETE /embeddings/{artifact_id}

Invalidate embeddings for an artifact.

**Response:**
```json
{
  "artifact_id": "art-789",
  "removed_chunks": 3,
  "status": "invalidated"
}
```

#### POST /embeddings/batch-invalidate

Invalidate multiple artifacts (for cascade).

**Request:**
```json
{
  "artifact_ids": ["art-789", "art-790", "art-791"],
  "reason": "upstream_taint",
  "cascade_id": "cascade-123"
}
```

**Response:**
```json
{
  "invalidated": 3,
  "chunks_removed": 9,
  "cascade_id": "cascade-123"
}
```

#### GET /health

Health check with LightRAG status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "lightrag": {
    "status": "connected",
    "index_size": 15000,
    "embedding_model": "nomic-embed-text"
  },
  "embedding_queue": {
    "pending": 5,
    "processing": 1
  }
}
```

---

## Events

### Subscribed Events

| Event | Source | Action |
|-------|--------|--------|
| `task.validated` | QA Service | Embed approved output |
| `artifact.tainted` | Cascade Service | Invalidate embeddings |

### Published Events

| Event | Trigger | Data |
|-------|---------|------|
| `embedding.completed` | After indexing | `{artifact_id, chunks, status}` |
| `embedding.invalidated` | After removal | `{artifact_ids, cascade_id}` |

---

## Data Models

### QueryRequest

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class SearchMode(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class QueryRequest(BaseModel):
    """Request for context retrieval."""
    
    query: str = Field(..., description="Natural language query")
    project_id: Optional[str] = None
    domain: Optional[str] = None
    max_tokens: int = Field(2000, ge=100, le=10000)
    search_mode: SearchMode = SearchMode.HYBRID
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "verified_only": True,
            "exclude_tainted": True,
        }
    )
```

### QueryResponse

```python
class SourceReference(BaseModel):
    """Reference to a source document."""
    
    artifact_id: str
    provenance_id: str
    relevance_score: float
    snippet: str

class QueryResponse(BaseModel):
    """Response from context query."""
    
    id: str
    query: str
    context: str
    sources: List[SourceReference]
    tokens_used: int
    search_mode: SearchMode
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### EmbedRequest

```python
class EmbedRequest(BaseModel):
    """Request to index content."""
    
    content: str = Field(..., min_length=10)
    metadata: Dict[str, Any] = Field(
        ...,
        description="Required: artifact_id, provenance_id, project_id"
    )
```

---

## Architecture

### Internal Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       RAG SERVICE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP Requests                              │
│  │  Router     │◄─── RabbitMQ Events                            │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                    QUERY ENGINE                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐       │   │
│  │  │  Vector  │  │ Keyword  │  │  Result Merger   │       │   │
│  │  │  Search  │  │  Search  │  │  (RRF ranking)   │       │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   EMBEDDING PIPELINE                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐       │   │
│  │  │ Chunker  │─▶│ Embedder │─▶│   Index Writer   │       │   │
│  │  └──────────┘  └──────────┘  └──────────────────┘       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     LIGHTRAG CORE                         │   │
│  │  ┌──────────────────┐  ┌──────────────────────────┐      │   │
│  │  │   Vector Store   │  │   Knowledge Graph (Neo4j) │      │   │
│  │  │   (Qdrant/FAISS) │  │   or Graph-in-memory      │      │   │
│  │  └──────────────────┘  └──────────────────────────┘      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    EVENT HANDLERS                         │   │
│  │  - task.validated → embed output                          │   │
│  │  - artifact.tainted → invalidate embeddings               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/rag-service.yaml
lightrag:
  working_dir: /data/lightrag
  embedding_model: nomic-embed-text
  embedding_endpoint: http://model-gateway:8000/api/v1
  
  # Vector store
  vector_store:
    type: qdrant  # or faiss for local
    host: qdrant
    port: 6333
    collection: dats_embeddings
    
  # Chunking
  chunking:
    strategy: semantic  # or fixed, sentence
    max_chunk_size: 500
    overlap: 50

query:
  default_mode: hybrid
  default_max_tokens: 2000
  result_limit: 10
  reranking:
    enabled: true
    model: cross-encoder/ms-marco-MiniLM-L-6-v2
    
embedding:
  batch_size: 10
  queue_max_size: 1000
  retry_attempts: 3
  
events:
  rabbitmq:
    host: rabbitmq
    port: 5672
    exchange: task.events
    queue: rag-service-events
```

---

## Migration Path

### Phase 1: Extract RAG Logic (Week 1)

1. Create `services/rag-service/` directory:
   ```
   services/rag-service/
   ├── Dockerfile
   ├── requirements.txt
   ├── src/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── config.py
   │   ├── routers/
   │   │   ├── __init__.py
   │   │   ├── query.py
   │   │   ├── embed.py
   │   │   └── health.py
   │   ├── services/
   │   │   ├── __init__.py
   │   │   ├── query_engine.py
   │   │   ├── embedding_pipeline.py
   │   │   └── invalidation.py
   │   ├── lightrag/
   │   │   ├── __init__.py
   │   │   └── client.py
   │   └── events/
   │       ├── __init__.py
   │       ├── handlers.py
   │       └── publisher.py
   └── tests/
   ```

2. Move existing code:
   - `src/rag/client.py` → `services/rag-service/src/lightrag/client.py`
   - `src/rag/query.py` → `services/rag-service/src/services/query_engine.py`
   - `src/rag/embedding_trigger.py` → `services/rag-service/src/services/embedding_pipeline.py`

### Phase 2: Add HTTP Client (Week 2)

```python
# dats_common/clients/rag.py
import httpx
from typing import Optional, Dict, Any, List

class RAGClient:
    """Client for RAG Service."""
    
    def __init__(
        self,
        base_url: str = "http://rag-service:8000/api/v1",
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def query(
        self,
        query: str,
        project_id: Optional[str] = None,
        domain: Optional[str] = None,
        max_tokens: int = 2000,
        search_mode: str = "hybrid",
    ) -> Dict[str, Any]:
        """Query for relevant context."""
        response = await self.client.post(
            f"{self.base_url}/query",
            json={
                "query": query,
                "project_id": project_id,
                "domain": domain,
                "max_tokens": max_tokens,
                "search_mode": search_mode,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def embed(
        self,
        content: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Index content for retrieval."""
        response = await self.client.post(
            f"{self.base_url}/embed",
            json={
                "content": content,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def invalidate(
        self,
        artifact_id: str,
    ) -> Dict[str, Any]:
        """Invalidate embeddings for an artifact."""
        response = await self.client.delete(
            f"{self.base_url}/embeddings/{artifact_id}",
        )
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        await self.client.aclose()
```

### Phase 3: Add Event Handlers (Week 2)

```python
# src/events/handlers.py
import pika
import json
from src.services.embedding_pipeline import EmbeddingPipeline
from src.services.invalidation import InvalidationService

class EventHandlers:
    def __init__(self, embedding_pipeline: EmbeddingPipeline, invalidation: InvalidationService):
        self.embedding = embedding_pipeline
        self.invalidation = invalidation
    
    async def handle_task_validated(self, message: dict):
        """Handle task.validated event - embed approved output."""
        artifact_id = message["data"]["artifact_id"]
        content = message["data"]["content"]
        metadata = message["data"]["metadata"]
        
        await self.embedding.embed(content, metadata)
    
    async def handle_artifact_tainted(self, message: dict):
        """Handle artifact.tainted event - invalidate embeddings."""
        artifact_ids = message["data"].get("artifact_ids", [message["data"]["artifact_id"]])
        cascade_id = message["data"].get("cascade_id")
        
        await self.invalidation.batch_invalidate(artifact_ids, cascade_id)
```

### Phase 4: Update Workers (Week 3)

Update `src/queue/tasks.py` to use RAG client:

```python
# Before (direct import)
from src.rag.query import RAGQueryEngine
engine = RAGQueryEngine()
context = await engine.get_context_for_worker(task_data, ...)

# After (service client)
from dats_common.clients.rag import RAGClient
client = RAGClient()
result = await client.query(
    query=task_data["description"],
    project_id=task_data.get("project_id"),
    domain=task_data.get("domain"),
)
context = result["context"]
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for LightRAG
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dats-common
COPY dats-common /tmp/dats-common
RUN pip install /tmp/dats-common

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Create data directory
RUN mkdir -p /data/lightrag

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Scaling Considerations

### Why RAG Service Scales Differently

- **GPU-Bound**: Embedding generation benefits from GPU
- **Memory-Intensive**: Vector index in memory
- **I/O Heavy**: Disk access for large indices

### Scaling Strategy

```yaml
# docker-compose.override.yml (for GPU)
services:
  rag-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### Separate Read/Write

For high throughput:
- **Query replicas**: Multiple read-only instances
- **Single writer**: One instance handles embeddings
- **Shared index**: Qdrant cluster for distributed vector store

---

## Observability

### Metrics

```
# Query performance
rag_query_latency_seconds{mode="hybrid", quantile="0.95"}
rag_query_results_count{mode="hybrid"}
rag_query_tokens_used{project_id="proj-123"}

# Embedding performance
rag_embedding_latency_seconds{quantile="0.95"}
rag_embedding_chunks_created_total
rag_embedding_queue_size

# Invalidation
rag_invalidation_count_total{reason="taint"}
rag_invalidation_chunks_removed_total

# Index health
rag_index_size_total
rag_index_memory_bytes
```

### Traces

```python
@tracer.start_as_current_span("rag.query")
async def query(self, request: QueryRequest):
    span = trace.get_current_span()
    span.set_attribute("rag.query.mode", request.search_mode)
    span.set_attribute("rag.query.max_tokens", request.max_tokens)
    
    # ... query logic ...
    
    span.set_attribute("rag.query.results_count", len(results))
    span.set_attribute("rag.query.tokens_used", tokens_used)
```

---

## Testing

### Unit Tests

```python
# tests/test_query_engine.py
import pytest
from src.services.query_engine import QueryEngine

@pytest.fixture
def engine():
    return QueryEngine(config_path="tests/fixtures/config.yaml")

@pytest.mark.asyncio
async def test_hybrid_search(engine):
    result = await engine.query(
        query="fibonacci implementation",
        search_mode="hybrid",
    )
    assert result.context != ""
    assert len(result.sources) > 0

@pytest.mark.asyncio
async def test_filters_exclude_tainted(engine):
    result = await engine.query(
        query="test query",
        filters={"exclude_tainted": True},
    )
    for source in result.sources:
        assert source.artifact_id not in TAINTED_ARTIFACTS
```

### Integration Tests

```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_embed_then_query():
    # Embed content
    embed_response = await client.post(
        "/api/v1/embed",
        json={
            "content": "def hello(): return 'world'",
            "metadata": {"artifact_id": "test-123", "project_id": "proj-1"},
        },
    )
    assert embed_response.status_code == 200
    
    # Query for it
    query_response = await client.post(
        "/api/v1/query",
        json={"query": "hello function"},
    )
    assert query_response.status_code == 200
    assert "hello" in query_response.json()["context"]
```

---

## Dependencies

```
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.5.0

# LightRAG
lightrag>=0.1.0
qdrant-client>=1.7.0
sentence-transformers>=2.2.0

# Message queue
pika>=1.3.0
aio-pika>=9.3.0

# Observability
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-fastapi>=0.42b0
prometheus-client>=0.19.0
```

---

## Environment Variables

```bash
# LightRAG
LIGHTRAG_WORKING_DIR=/data/lightrag
EMBEDDING_MODEL=nomic-embed-text

# Model Gateway (for embeddings)
MODEL_GATEWAY_URL=http://model-gateway:8000/api/v1

# Vector store
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION=dats_embeddings

# RabbitMQ
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672

# Telemetry
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
OTEL_SERVICE_NAME=rag-service
```

---

## Success Criteria

- [ ] Query latency P95 < 500ms
- [ ] Embedding throughput > 10 docs/second
- [ ] Zero stale results after invalidation
- [ ] Event handlers process within 5 seconds
- [ ] Index survives service restart
- [ ] Metrics visible in Grafana

---

*Last updated: January 2026*