# File: docs/architecture/services/02-rag-service.md
# RAG Service

> **Priority**: P-DEFERRED | **Team**: Data | **Status**: Deferred
>
> Context retrieval and embedding management via LightRAG.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

**Note**: RAG is not core functionality. Extract only after Phase 5 is complete.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/rag-service/contracts/openapi.yaml` |
| AsyncAPI | `services/rag-service/contracts/asyncapi.yaml` |

---

## Purpose

- **Context Retrieval**: Query relevant context for task execution
- **Embedding Management**: Index approved outputs for future retrieval
- **Invalidation**: Remove embeddings when artifacts are tainted
- **Hybrid Search**: Combine vector similarity with keyword matching

---

## API Endpoints

### POST /query

Query for relevant context.

```json
// Request
{"query": "How to implement fibonacci?", "project_id": "proj-123", "max_tokens": 2000, "search_mode": "hybrid"}

// Response
{"context": "## Previous Implementation\n...", "sources": [{"artifact_id": "art-123", "relevance_score": 0.92}], "tokens_used": 1500}
```

### POST /embed

Index content for future retrieval.

```json
// Request
{"content": "def fibonacci(n)...", "metadata": {"artifact_id": "art-789", "project_id": "proj-123"}}

// Response
{"status": "indexed", "chunks": 3, "tokens": 450}
```

### DELETE /embeddings/{artifact_id}

Invalidate embeddings for an artifact.

### POST /embeddings/batch-invalidate

Invalidate multiple artifacts (for cascade operations).

---

## Events

| Subscribes | Source | Action |
|------------|--------|--------|
| `task.validated` | QA Service | Embed approved output |
| `artifact.tainted` | Cascade Service | Invalidate embeddings |

| Publishes | Trigger |
|-----------|---------|
| `embedding.completed` | After indexing |
| `embedding.invalidated` | After removal |

---

## Configuration

```yaml
# config/rag-service.yaml
lightrag:
  working_dir: /data/lightrag
  embedding_model: mxbai-embed-large:335m
  embedding_endpoint: http://192.168.1.12:11434
  vector_store:
    type: qdrant
    host: qdrant
    port: 6333

query:
  default_mode: hybrid
  default_max_tokens: 2000
```

---

## Scaling Notes

- **GPU-Bound**: Embedding generation benefits from GPU
- **Memory-Intensive**: Vector index in memory
- For high throughput: separate read replicas from single writer

---

## Migration Path

1. **Week 1**: Create service, move LightRAG wrapper from `src/rag/`
2. **Week 2**: Add HTTP client to dats-common
3. **Week 3**: Update workers to use RAG client

---

## Environment Variables

```bash
LIGHTRAG_WORKING_DIR=/data/lightrag
EMBEDDING_MODEL=mxbai-embed-large:335m
OLLAMA_EMBEDDING_URL=http://192.168.1.12:11434
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

---

## Success Criteria

- [ ] Query latency P95 < 500ms
- [ ] Embedding throughput > 10 docs/second
- [ ] Zero stale results after invalidation
- [ ] Index survives service restart
