# Worker Service

> **DATS Microservice** - Code Generation Execution  
> Priority: P3  
> Team: AI/ML  
> Status: Planned

---

## Overview

### Purpose

The Worker Service executes code generation tasks:

- **Domain-Specific Workers**: code-general, code-vision, code-embedded, documentation, ui-design
- **Context Integration**: Receives context from RAG Service
- **Output Production**: Generates artifacts for QA validation

### Current State (Monolith)

```
src/workers/
├── __init__.py
├── base.py           # BaseWorker
├── code_general.py
├── code_vision.py
├── code_embedded.py
├── documentation.py
└── ui_design.py
```

---

## API Specification

### Base URL

```
http://worker-service:8000/api/v1
```

### Endpoints

#### POST /execute

Execute a task (synchronous for small tasks).

**Request:**
```json
{
  "task_id": "task-123",
  "description": "Create fibonacci function",
  "domain": "code-general",
  "tier": "small",
  "context": "## Previous Implementation\n...",
  "project_id": "proj-456"
}
```

**Response:**
```json
{
  "task_id": "task-123",
  "artifact_id": "art-789",
  "status": "completed",
  "output": {
    "content": "def fibonacci(n)...",
    "artifacts": [
      {"type": "code", "language": "python", "path": "fibonacci.py"}
    ]
  },
  "tokens_used": 450,
  "execution_time_ms": 25000
}
```

#### GET /health

Health check with worker availability.

---

## Events

### Subscribed Events

| Event | Source | Action |
|-------|--------|--------|
| `task.ready.{tier}` | Orchestration | Execute task |

### Published Events

| Event | Trigger | Data |
|-------|---------|------|
| `task.output.created` | Generation complete | `{task_id, artifact_id, output}` |
| `task.failed` | Generation failed | `{task_id, error}` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WORKER SERVICE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP (sync requests)                       │
│  │  Router     │◄─── RabbitMQ (async tasks)                     │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                  WORKER DISPATCHER                        │   │
│  │  - Route to domain worker                                 │   │
│  │  - Handle tier-based queuing                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    DOMAIN WORKERS                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │   │
│  │  │  Code    │  │  Code    │  │  Code    │              │   │
│  │  │ General  │  │  Vision  │  │ Embedded │              │   │
│  │  └──────────┘  └──────────┘  └──────────┘              │   │
│  │  ┌──────────┐  ┌──────────┐                            │   │
│  │  │  Docs    │  │   UI     │                            │   │
│  │  │          │  │  Design  │                            │   │
│  │  └──────────┘  └──────────┘                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐       │
│  │Model Gateway│  │ RAG Service │  │  Prompt Renderer  │       │
│  │   Client    │  │   Client    │  │                   │       │
│  └─────────────┘  └─────────────┘  └───────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/worker-service.yaml
workers:
  code-general:
    prompt_template: workers/code_general.md
    default_tier: small
    
  code-vision:
    prompt_template: workers/code_vision.md
    default_tier: large
    
  code-embedded:
    prompt_template: workers/code_embedded.md
    default_tier: small

queues:
  tiny:
    concurrency: 10
    timeout_seconds: 60
  small:
    concurrency: 5
    timeout_seconds: 120
  large:
    concurrency: 2
    timeout_seconds: 300
  frontier:
    concurrency: 1
    timeout_seconds: 600

model_gateway:
  url: http://model-gateway:8000/api/v1
  
rag_service:
  url: http://rag-service:8000/api/v1

events:
  rabbitmq:
    host: rabbitmq
    port: 5672
    exchange: task.events
```

---

## Success Criteria

- [ ] P95 execution time within tier SLA
- [ ] Zero lost tasks from queue
- [ ] Output validates against schema
- [ ] RAG context integration working

---

*Last updated: January 2026*