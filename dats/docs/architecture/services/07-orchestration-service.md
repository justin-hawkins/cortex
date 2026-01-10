# Orchestration Service

> **DATS Microservice** - Task Lifecycle Management  
> Priority: P4 (Last to Extract)  
> Team: Platform  
> Status: Planned

---

## Overview

### Purpose

The Orchestration Service is the central coordinator for DATS:

- **Task Lifecycle**: Submit, route, track, complete
- **Status Management**: Track task state across services
- **Provenance Storage**: Maintain execution history
- **External API**: Main entry point for CLI and external systems

### Current State (Monolith)

```
src/pipeline/
├── __init__.py
└── orchestrator.py    # AgentPipeline

src/queue/
├── __init__.py
├── celery_app.py
└── tasks.py           # ~800 lines (to be split)

src/api/
├── app.py
└── routes/
    ├── tasks.py
    ├── projects.py
    └── ...
```

---

## API Specification

### Base URL

```
http://orchestration-service:8000/api/v1
```

### Endpoints

#### POST /tasks

Submit a new task.

**Request:**
```json
{
  "description": "Create a Python function that calculates fibonacci numbers",
  "project_id": "proj-123",
  "priority": "normal",
  "options": {
    "decompose": true,
    "qa_profile": "consensus"
  }
}
```

**Response:**
```json
{
  "task_id": "task-456",
  "project_id": "proj-123",
  "status": "pending",
  "created_at": "2026-01-09T18:00:00Z",
  "estimated_completion": "2026-01-09T18:02:00Z"
}
```

#### GET /tasks/{task_id}

Get task status.

**Response:**
```json
{
  "task_id": "task-456",
  "project_id": "proj-123",
  "status": "completed",
  "mode": "new_project",
  "domain": "code-general",
  "tier": "small",
  "started_at": "2026-01-09T18:00:05Z",
  "completed_at": "2026-01-09T18:00:55Z",
  "execution_time_ms": 50000,
  "artifacts": [
    {"id": "art-789", "type": "code", "path": "fibonacci.py"}
  ],
  "provenance_id": "prov-012"
}
```

#### POST /tasks/{task_id}/cancel

Cancel a running task.

#### GET /tasks

List tasks with filtering.

**Query Parameters:**
- `project_id`: Filter by project
- `status`: Filter by status (pending, running, completed, failed)
- `limit`, `offset`: Pagination

#### GET /projects/{project_id}/status

Get project-level status summary.

#### GET /provenance/{id}

Get provenance record.

#### GET /health

Service health check.

---

## Events

### Published Events

| Event | Trigger | Data |
|-------|---------|------|
| `task.created` | New submission | `{task_id, project_id, description}` |
| `task.ready.{tier}` | After analysis | `{task_id, domain, tier, context}` |
| `task.cancelled` | Cancel request | `{task_id, reason}` |

### Subscribed Events

| Event | Source | Action |
|-------|--------|--------|
| `task.output.created` | Worker Service | Update status, trigger QA |
| `task.validated` | QA Service | Mark complete, store artifact |
| `task.rejected` | QA Service | Handle retry or fail |
| `task.failed` | Any | Update status, notify |
| `rollback.requested` | Cascade Service | Process rollback |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION SERVICE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP Requests (CLI, external)              │
│  │  Router     │◄─── RabbitMQ Events                            │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                  TASK MANAGER                             │   │
│  │  - Create tasks                                           │   │
│  │  - Track status                                           │   │
│  │  - Handle state transitions                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  ROUTING ENGINE                           │   │
│  │  - Call Agent Service for analysis                        │   │
│  │  - Determine tier and domain                              │   │
│  │  - Publish task.ready events                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  PROVENANCE STORE                         │   │
│  │  - PostgreSQL for provenance records                      │   │
│  │  - Dependency graph                                       │   │
│  │  - Taint status                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐        │
│  │   Agent     │  │    RAG      │  │   Event Bus      │        │
│  │   Client    │  │   Client    │  │   (RabbitMQ)     │        │
│  └─────────────┘  └─────────────┘  └──────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

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
  provenance_table: provenance_records
  
events:
  rabbitmq:
    host: rabbitmq
    port: 5672
    exchanges:
      - task.events
      - cascade.events
      - qa.events
    queues:
      - orchestration-events
```

---

## Database Schema

```sql
-- Provenance records (migrated from JSON files)
CREATE TABLE provenance_records (
    id UUID PRIMARY KEY,
    task_id VARCHAR(255) NOT NULL,
    project_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    
    -- Execution details
    model_used VARCHAR(100),
    worker_id VARCHAR(100),
    tier VARCHAR(20),
    domain VARCHAR(50),
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Metrics
    tokens_input INTEGER,
    tokens_output INTEGER,
    execution_time_ms INTEGER,
    confidence DECIMAL(3,2),
    
    -- Taint tracking
    is_tainted BOOLEAN DEFAULT FALSE,
    tainted_at TIMESTAMP,
    tainted_reason TEXT,
    is_suspect BOOLEAN DEFAULT FALSE,
    
    -- Content
    outputs JSONB,
    inputs_consumed JSONB,
    verification JSONB,
    
    -- Indexes
    INDEX idx_task_id (task_id),
    INDEX idx_project_id (project_id),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Dependency graph for cascade
CREATE TABLE artifact_dependencies (
    id SERIAL PRIMARY KEY,
    artifact_id UUID NOT NULL,
    depends_on UUID NOT NULL,
    dependency_type VARCHAR(50),
    FOREIGN KEY (artifact_id) REFERENCES provenance_records(id),
    FOREIGN KEY (depends_on) REFERENCES provenance_records(id)
);
```

---

## Migration Path

### Phase 1: Database Migration (Week 1)

1. Set up PostgreSQL schema
2. Migrate JSON provenance files to database
3. Update provenance storage to use database
4. Maintain JSON export for compatibility

### Phase 2: Event Bus Integration (Week 2)

1. Replace Celery with RabbitMQ publishers
2. Update task status to event-driven
3. Add event handlers for service responses

### Phase 3: Service Extraction (Week 3-4)

1. Create `services/orchestration-service/`
2. Move routing logic from `tasks.py`
3. Extract API routes
4. Update CLI to use service

---

## Success Criteria

- [ ] Task submission latency P95 < 500ms
- [ ] Status queries P95 < 100ms
- [ ] Zero lost tasks during migration
- [ ] Provenance query across 100k records < 1s
- [ ] Event processing within 2 seconds

---

*Last updated: January 2026*