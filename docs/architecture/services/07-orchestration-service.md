# File: docs/architecture/services/07-orchestration-service.md
# Orchestration Service

> **Priority**: P4 (Last to Extract) | **Team**: Platform | **Status**: Planned
>
> Task lifecycle management, routing, status tracking, and provenance.

**Common patterns**: See [SERVICE_COMMON.md](../_shared/SERVICE_COMMON.md) for folder structure, Dockerfile, testing, and observability patterns.

---

## Contracts

| Type | Location |
|------|----------|
| OpenAPI | `services/orchestration-service/contracts/openapi.yaml` |
| AsyncAPI | `services/orchestration-service/contracts/asyncapi.yaml` |

---

## Purpose

- **Task Lifecycle**: Submit, route, track, complete
- **Status Management**: Track task state across services
- **Provenance Storage**: Maintain execution history (PostgreSQL)
- **External API**: Main entry point for CLI and external systems

---

## API Endpoints

### POST /tasks

Submit a new task.

```json
// Request
{"description": "Create a Python fibonacci function", "project_id": "proj-123", "options": {"decompose": true}}

// Response
{"task_id": "task-456", "status": "pending", "estimated_completion": "2026-01-09T18:02:00Z"}
```

### GET /tasks/{task_id}

Get task status with artifacts and provenance.

### POST /tasks/{task_id}/cancel

Cancel a running task.

### GET /tasks

List tasks with filtering by project, status, pagination.

### GET /projects/{project_id}/status

Get project-level status summary.

### GET /provenance/{id}

Get provenance record for an artifact.

---

## Events

| Subscribes | Source | Action |
|------------|--------|--------|
| `task.output.created` | Worker Service | Update status, trigger QA |
| `task.validated` | QA Service | Mark complete, store artifact |
| `task.rejected` | QA Service | Handle retry or fail |
| `rollback.requested` | Cascade Service | Process rollback |

| Publishes | Trigger |
|-----------|---------|
| `task.created` | New submission |
| `task.ready.{tier}` | After analysis |
| `task.cancelled` | Cancel request |

---

## Database Schema

```sql
-- Provenance records
CREATE TABLE provenance_records (
    id UUID PRIMARY KEY,
    task_id VARCHAR(255) NOT NULL,
    project_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    model_used VARCHAR(100),
    tier VARCHAR(20),
    domain VARCHAR(50),
    tokens_input INTEGER,
    tokens_output INTEGER,
    is_tainted BOOLEAN DEFAULT FALSE,
    outputs JSONB,
    created_at TIMESTAMP NOT NULL
);

-- Dependency graph
CREATE TABLE artifact_dependencies (
    artifact_id UUID NOT NULL,
    depends_on UUID NOT NULL,
    dependency_type VARCHAR(50)
);
```

---

## Configuration

```yaml
# config/orchestration-service.yaml
task_management:
  default_priority: normal
  max_concurrent_per_project: 10
  timeout_default_seconds: 600

routing:
  agent_service_url: http://agent-service:8000/api/v1
  rag_service_url: http://rag-service:8000/api/v1

storage:
  database_url: postgresql://dats:secret@postgres:5432/dats
```

---

## Migration Path

1. **Week 1**: Set up PostgreSQL, migrate JSON provenance to database
2. **Week 2**: Replace Celery with RabbitMQ publishers
3. **Week 3-4**: Extract routing logic, API routes from `tasks.py`

---

## Environment Variables

```bash
DATABASE_URL=postgresql://dats:secret@postgres:5432/dats
AGENT_SERVICE_URL=http://agent-service:8000/api/v1
RAG_SERVICE_URL=http://rag-service:8000/api/v1
RABBITMQ_HOST=192.168.1.49
REDIS_URL=redis://192.168.1.44:6379/0
```

---

## Success Criteria

- [ ] Task submission latency P95 < 500ms
- [ ] Status queries P95 < 100ms
- [ ] Zero lost tasks during migration
- [ ] Provenance query across 100k records < 1s
