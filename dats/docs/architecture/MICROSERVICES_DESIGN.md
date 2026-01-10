# DATS Microservices Architecture Design

> **Distributed Agentic Task System** - Microservices Architecture Specification
> 
> Version: 1.0.0  
> Last Updated: January 2026  
> Status: Draft

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Target Architecture](#target-architecture)
4. [Service Definitions](#service-definitions)
5. [Communication Patterns](#communication-patterns)
6. [Shared Infrastructure](#shared-infrastructure)
7. [Migration Strategy](#migration-strategy)
8. [Deployment](#deployment)
9. [Appendices](#appendices)

---

## Executive Summary

> **Note**: Infrastructure endpoints (Ollama, vLLM, RabbitMQ, Redis) are defined in 
> [`servers.yaml`](servers.yaml). All service configurations reference those centralized definitions.

### Purpose

This document defines the microservices architecture for DATS, transforming the current monolithic POC into a scalable, maintainable, and team-distributable system.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Message Bus** | RabbitMQ | Rich routing, dead-letter queues, mature ecosystem |
| **API Format** | REST + OpenAPI | Developer familiarity, tooling; gRPC-ready for future |
| **Deployment** | Docker Compose → K8s | Start simple, scale when needed |
| **Shared Code** | Python package (`dats-common`) | Type sharing, avoid duplication |
| **Repo Strategy** | Monorepo → Multi-repo | Start unified, split as teams form |

### Service Inventory

| Service | Priority | Team Ownership | Status |
|---------|----------|----------------|--------|
| Model Gateway | P0 | Platform | Planned |
| Agent Service | P1 | AI/ML | Planned |
| Worker Service | P1 | AI/ML | Planned |
| Orchestration Service | P2 | Platform | Planned |
| Cascade Service | P3 | Reliability | Planned |
| QA Service | P3 | Quality | Planned |
| RAG Service | P-DEFERRED | Data | Deferred (not core) |

---

## Current State Analysis

### Monolith Pain Points

1. **Large Files**: `tasks.py` (~800 lines) handles execution, cascade, validation, embedding
2. **Tight Coupling**: Direct imports between layers prevent independent deployment
3. **Mixed Patterns**: Sync/async gymnastics (`asyncio.get_event_loop()`) throughout
4. **Single Deployment**: All components deploy together, blocking independent releases
5. **Scaling Limitations**: Cannot scale workers independently of orchestration

### Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT MONOLITHIC DATS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌────────────┐    ┌─────────────────┐   │
│  │   CLI   │───▶│   FastAPI   │───▶│  Pipeline  │───▶│  Celery Queue   │   │
│  │ or API  │    │   /api/v1   │    │Orchestrator│    │  (Redis/RMQ)    │   │
│  └─────────┘    └─────────────┘    └────────────┘    └─────────────────┘   │
│                                           │                    │            │
│                                           ▼                    ▼            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TIGHTLY COUPLED LAYERS                            │   │
│  │  Agents ←→ Workers ←→ Models ←→ Storage ←→ RAG ←→ Cascade ←→ QA    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Problem: Everything imports everything. Cannot deploy or scale separately. │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code to Extract

| Current Location | Target Service | Lines | Complexity |
|------------------|----------------|-------|------------|
| `src/models/*.py` | Model Gateway | ~400 | Low |
| `src/rag/*.py` | RAG Service | ~300 | Medium |
| `src/cascade/*.py` | Cascade Service | ~500 | Medium |
| `src/qa/*.py` | QA Service | ~600 | High |
| `src/agents/*.py` | Agent Service | ~800 | High |
| `src/workers/*.py` | Worker Service | ~500 | Medium |
| `src/pipeline/*.py` + `src/queue/tasks.py` | Orchestration | ~1200 | High |

---

## Target Architecture

### High-Level View

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATS MICROSERVICES                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌────────────────┐                                                             │
│  │   API Gateway  │  (Traefik / NGINX)                                          │
│  │   /api/v1/*    │                                                             │
│  └───────┬────────┘                                                             │
│          │                                                                       │
│  ┌───────┴────────────────────────────────────────────────────────────────┐     │
│  │                        MESSAGE BUS (RabbitMQ)                           │     │
│  │  Exchanges: task.events, cascade.events, qa.events, model.events       │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│          │                                                                       │
│  ┌───────┴───────┬──────────────┬──────────────┬──────────────┬────────────┐   │
│  │               │              │              │              │            │   │
│  ▼               ▼              ▼              ▼              ▼            ▼   │
│ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│ │Orchestr-│ │  Agent   │ │  Worker  │ │   RAG    │ │ Cascade  │ │   QA    │  │
│ │ation    │ │ Service  │ │ Service  │ │ Service  │ │ Service  │ │ Service │  │
│ │ Service │ │          │ │          │ │          │ │          │ │         │  │
│ └────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘  │
│      │           │            │            │            │            │       │
│  ┌───┴───────────┴────────────┴────────────┴────────────┴────────────┴───┐   │
│  │                     MODEL GATEWAY SERVICE                              │   │
│  │  Unified LLM interface: Ollama, OpenAI, Anthropic, vLLM               │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │                       STORAGE SERVICES                                  │     │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │     │
│  │  │  Provenance   │  │ Work Product  │  │   LightRAG    │               │     │
│  │  │   (Postgres)  │  │   (S3/Minio)  │  │   (Vector)    │               │     │
│  │  └───────────────┘  └───────────────┘  └───────────────┘               │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Communication Matrix

| From \ To | Model Gateway | RAG | Cascade | QA | Agent | Worker | Orchestration |
|-----------|---------------|-----|---------|----|----|--------|---------------|
| **Orchestration** | - | HTTP | Event | Event | HTTP | Event | - |
| **Agent** | HTTP | - | - | - | - | - | Event |
| **Worker** | HTTP | HTTP | - | - | - | - | Event |
| **QA** | HTTP | - | Event | - | - | - | Event |
| **Cascade** | - | HTTP | - | - | - | - | Event |
| **RAG** | - | - | - | - | - | - | - |
| **Model Gateway** | - | - | - | - | - | - | - |

**Legend:**
- **HTTP**: Synchronous REST API call
- **Event**: Asynchronous message via RabbitMQ
- **-**: No direct communication

---

## Service Definitions

Each service has a dedicated design document:

| Service | Document | Description |
|---------|----------|-------------|
| Model Gateway | [01-model-gateway.md](services/01-model-gateway.md) | Unified LLM interface |
| RAG Service | [02-rag-service.md](services/02-rag-service.md) | Context retrieval & embedding |
| Cascade Service | [03-cascade-service.md](services/03-cascade-service.md) | Failure propagation & rollback |
| QA Service | [04-qa-service.md](services/04-qa-service.md) | Output validation & review |
| Agent Service | [05-agent-service.md](services/05-agent-service.md) | Task analysis & decomposition |
| Worker Service | [06-worker-service.md](services/06-worker-service.md) | Code generation execution |
| Orchestration Service | [07-orchestration-service.md](services/07-orchestration-service.md) | Task lifecycle management |

### Service Summary

#### Model Gateway (Priority: P0)
- **Purpose**: Abstract LLM providers behind unified interface
- **API**: `POST /generate`, `GET /models`, `GET /health`
- **Stateless**: Yes
- **Scales**: Horizontally

#### RAG Service (Priority: P1)
- **Purpose**: Context retrieval and embedding management
- **API**: `POST /query`, `POST /embed`, `DELETE /embeddings/{id}`
- **State**: LightRAG vector store
- **Scales**: Vertically (GPU-bound)

#### Cascade Service (Priority: P2)
- **Purpose**: Taint propagation, revalidation, rollback
- **API**: `POST /taint`, `GET /impact/{id}`, `POST /rollback`
- **Subscribes**: `task.rejected`, `security.alert`
- **Publishes**: `cascade.started`, `artifact.tainted`

#### QA Service (Priority: P2)
- **Purpose**: Output validation, human review
- **API**: `POST /validate`, `GET /reviews/{id}`, `POST /approve`
- **Subscribes**: `task.output.created`
- **Publishes**: `task.validated`, `task.rejected`

#### Agent Service (Priority: P3)
- **Purpose**: Coordinator, Decomposer, Complexity Estimator
- **API**: `POST /analyze`, `POST /decompose`, `POST /estimate`
- **Stateless**: Yes (uses Model Gateway)
- **Scales**: Horizontally

#### Worker Service (Priority: P3)
- **Purpose**: Execute code generation tasks
- **Subscribes**: `task.ready.{tier}`
- **Publishes**: `task.output.created`
- **May Split**: By domain (code-general, code-vision, etc.)

#### Orchestration Service (Priority: P4)
- **Purpose**: Task lifecycle, routing, status
- **API**: `POST /tasks`, `GET /tasks/{id}`, `POST /cancel`
- **Publishes**: `task.created`, `task.ready`
- **Subscribes**: `task.completed`, `task.failed`

---

## Communication Patterns

### Synchronous (REST API)

Used for:
- Request/response operations (submit task, get status)
- Operations requiring immediate confirmation
- Service health checks

```yaml
# OpenAPI pattern for all services
openapi: 3.0.3
info:
  title: DATS {Service} API
  version: 1.0.0
servers:
  - url: http://{service}:8000/api/v1
paths:
  /health:
    get:
      operationId: healthCheck
      responses:
        '200':
          description: Service healthy
```

### Asynchronous (RabbitMQ)

Used for:
- Long-running operations
- Fire-and-forget notifications
- Event-driven workflows

#### Exchange Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     RabbitMQ Exchanges                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐                                        │
│  │  task.events        │  (topic exchange)                      │
│  │  ─────────────────  │                                        │
│  │  task.created       │ → Orchestration publishes              │
│  │  task.ready.{tier}  │ → Worker subscribes by tier            │
│  │  task.output.created│ → QA subscribes                        │
│  │  task.completed     │ → Orchestration subscribes             │
│  │  task.failed        │ → Orchestration subscribes             │
│  │  task.validated     │ → Orchestration, RAG subscribe         │
│  │  task.rejected      │ → Cascade subscribes                   │
│  └─────────────────────┘                                        │
│                                                                  │
│  ┌─────────────────────┐                                        │
│  │  cascade.events     │  (topic exchange)                      │
│  │  ─────────────────  │                                        │
│  │  cascade.started    │ → Monitoring subscribes                │
│  │  artifact.tainted   │ → RAG subscribes (invalidate)          │
│  │  artifact.suspect   │ → QA subscribes (revalidate)           │
│  │  rollback.requested │ → Orchestration subscribes             │
│  └─────────────────────┘                                        │
│                                                                  │
│  ┌─────────────────────┐                                        │
│  │  qa.events          │  (topic exchange)                      │
│  │  ─────────────────  │                                        │
│  │  review.requested   │ → Human review UI subscribes           │
│  │  review.completed   │ → Orchestration subscribes             │
│  └─────────────────────┘                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Message Schema (CloudEvents)

All messages follow CloudEvents specification:

```json
{
  "specversion": "1.0",
  "type": "dats.task.created",
  "source": "/orchestration",
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "time": "2026-01-09T12:00:00Z",
  "datacontenttype": "application/json",
  "data": {
    "task_id": "abc123",
    "project_id": "proj-456",
    "tier": "small",
    "domain": "code-general"
  }
}
```

---

## Shared Infrastructure

### dats-common Package

Shared Python package installed by all services:

```
dats-common/
├── pyproject.toml
├── src/
│   └── dats_common/
│       ├── __init__.py
│       ├── models/           # Pydantic models
│       │   ├── task.py       # TaskRequest, TaskResponse
│       │   ├── provenance.py # ProvenanceRecord
│       │   └── events.py     # CloudEvent wrappers
│       ├── clients/          # Service clients
│       │   ├── model_gateway.py
│       │   ├── rag.py
│       │   └── rabbitmq.py
│       ├── telemetry/        # OpenTelemetry setup
│       │   ├── config.py
│       │   └── decorators.py
│       └── config/           # Shared settings
│           └── base.py
```

### Container Base Image

```dockerfile
# dats-base/Dockerfile
FROM python:3.11-slim

# Common dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi \
    pika  # RabbitMQ client

# Install dats-common
COPY dats-common /tmp/dats-common
RUN pip install /tmp/dats-common

WORKDIR /app
```

### Docker Compose (Development)

```yaml
# docker-compose.yml
# NOTE: This uses external infrastructure defined in servers.yaml
# - RabbitMQ: 192.168.1.49:5672
# - Redis: 192.168.1.44:6379
# - Ollama (CPU/coding): 192.168.1.11:11434
# - Ollama (GPU/general): 192.168.1.12:11434
# - vLLM: 192.168.1.11:8000

version: '3.8'

services:
  # Local Infrastructure (for development - production uses servers.yaml endpoints)
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: dats
      POSTGRES_USER: dats
      POSTGRES_PASSWORD: dats_secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  jaeger:
    image: jaegertracing/all-in-one:1.50
    ports:
      - "16686:16686"
      - "4317:4317"

  # Services
  model-gateway:
    build: ./services/model-gateway
    ports:
      - "8001:8000"
    environment:
      # From servers.yaml endpoints
      - OLLAMA_CPU_LARGE=http://192.168.1.11:11434
      - OLLAMA_GPU_GENERAL=http://192.168.1.12:11434
      - VLLM_ENDPOINT=http://192.168.1.11:8000/v1
      - ANTHROPIC_ENDPOINT=https://api.anthropic.com/v1
      # From servers.yaml infrastructure
      - REDIS_URL=redis://192.168.1.44:6379/0
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317

  rag-service:
    build: ./services/rag-service
    ports:
      - "8002:8000"
    environment:
      # From servers.yaml defaults.embedding
      - EMBEDDING_MODEL=mxbai-embed-large:335m
      - OLLAMA_EMBEDDING_URL=http://192.168.1.12:11434
      # From servers.yaml infrastructure.rabbitmq
      - RABBITMQ_HOST=192.168.1.49
      - RABBITMQ_PORT=5672
    depends_on:
      - model-gateway

  cascade-service:
    build: ./services/cascade-service
    ports:
      - "8003:8000"
    environment:
      # From servers.yaml infrastructure
      - RABBITMQ_HOST=192.168.1.49
      - RABBITMQ_PORT=5672
      - REDIS_URL=redis://192.168.1.44:6379/0
    depends_on:
      - postgres

  qa-service:
    build: ./services/qa-service
    ports:
      - "8004:8000"
    environment:
      - RABBITMQ_HOST=192.168.1.49
      - RABBITMQ_PORT=5672
    depends_on:
      - model-gateway

  agent-service:
    build: ./services/agent-service
    ports:
      - "8005:8000"
    depends_on:
      - model-gateway

  worker-service:
    build: ./services/worker-service
    ports:
      - "8006:8000"
    environment:
      - RABBITMQ_HOST=192.168.1.49
      - RABBITMQ_PORT=5672
    depends_on:
      - model-gateway
      - rag-service

  orchestration-service:
    build: ./services/orchestration-service
    ports:
      - "8000:8000"
    environment:
      - RABBITMQ_HOST=192.168.1.49
      - RABBITMQ_PORT=5672
      - REDIS_URL=redis://192.168.1.44:6379/0
    depends_on:
      - postgres
      - agent-service

volumes:
  postgres_data:
  minio_data:
```

---

## Migration Strategy

### Phase 1: Interface Extraction (Week 1-2)

**Goal**: Define clear service boundaries without breaking existing code.

1. Create `dats-common` package with shared types
2. Define Protocol classes for each service interface
3. Add internal HTTP clients that wrap current direct calls
4. No deployment changes - all still runs as monolith

### Phase 2: Model Gateway Extraction (Week 3-4)

**Goal**: First real microservice - lowest risk.

1. Create `services/model-gateway/` with FastAPI app
2. Implement `/generate` endpoint wrapping existing clients
3. Add Docker setup and health checks
4. Update `dats-common` with `ModelGatewayClient`
5. Update agents/workers to use client instead of direct import
6. Deploy alongside monolith, switch traffic gradually

### Phase 3: Event Bus Introduction (Week 5-6)

**Goal**: Replace Celery with RabbitMQ for task execution.

1. Set up RabbitMQ with exchanges defined above
2. Create publisher/subscriber wrappers in `dats-common`
3. Update Orchestration to publish task events
4. Update Workers to subscribe to task queues
5. Run Celery and RabbitMQ in parallel during transition
6. Remove Celery once stable

### Phase 4: Core Services Extraction (Week 7-12)

Extract in order:
1. Agent Service (Week 7-8) - Task analysis and decomposition
2. Worker Service (Week 9-10) - Code generation execution
3. Orchestration Service (Week 11-12) - Task lifecycle management

### Phase 5: Supporting Services (Week 13-16)

Extract as needed:
1. Cascade Service (Week 13-14) - Failure propagation
2. QA Service (Week 15-16) - Output validation

### Phase DEFERRED: RAG Service

**Goal**: Isolate LightRAG for independent scaling (after core is stable).

The RAG Service is **not part of core functionality** and should be extracted only after the main pipeline is working as microservices. Current RAG integration can continue as direct library calls.

When ready to extract:
1. Create `services/rag-service/`
2. Move embedding and query logic
3. Keep vector store data in dedicated volume
4. Update workers to use RAG client

---

## Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f orchestration-service

# Scale workers
docker-compose up -d --scale worker-service=3
```

### Production (Future: Kubernetes)

Directory structure prepared for K8s:

```
deploy/
├── docker-compose.yml      # Development
├── k8s/
│   ├── base/              # Kustomize base
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   └── services/
│   │       ├── model-gateway/
│   │       ├── rag-service/
│   │       └── ...
│   └── overlays/
│       ├── dev/
│       ├── staging/
│       └── prod/
```

---

## Appendices

### A. Architecture Decision Records

| ADR | Decision | Document |
|-----|----------|----------|
| 001 | Monorepo initially, split later | [001-repo-strategy.md](decisions/001-repo-strategy.md) |
| 002 | RabbitMQ over Redis Streams | [002-message-bus.md](decisions/002-message-bus.md) |
| 003 | REST first, gRPC later | [003-api-format.md](decisions/003-api-format.md) |
| 004 | Service boundaries | [004-service-boundaries.md](decisions/004-service-boundaries.md) |

### B. API Contracts

OpenAPI specifications:
- [contracts/openapi/model-gateway.yaml](contracts/openapi/model-gateway.yaml)
- [contracts/openapi/orchestration.yaml](contracts/openapi/orchestration.yaml)
- [contracts/openapi/rag.yaml](contracts/openapi/rag.yaml)

Event schemas:
- [contracts/events/task-events.yaml](contracts/events/task-events.yaml)
- [contracts/events/cascade-events.yaml](contracts/events/cascade-events.yaml)

### C. Related Documents

- [ARCHITECTURE_KNOWLEDGE_GRAPH.md](../ARCHITECTURE_KNOWLEDGE_GRAPH.md) - Current monolith reference
- [telemetry-plan.md](../telemetry-plan.md) - Observability strategy

---

*This document is the source of truth for DATS microservices architecture. Updates require review by architecture team.*