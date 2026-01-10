# Cascade Service

> **DATS Microservice** - Failure Propagation & Rollback  
> Priority: P2  
> Team: Reliability  
> Status: Planned

---

## Overview

### Purpose

The Cascade Service manages failure propagation, taint tracking, revalidation, and rollback for DATS. It ensures that when a task output is rejected or found faulty, all dependent artifacts are properly handled.

- **Taint Propagation**: Mark artifacts and their dependents as tainted/suspect
- **Revalidation**: Queue and process suspect artifacts for re-verification
- **Rollback**: Restore to previous known-good checkpoints
- **Impact Analysis**: Calculate blast radius of failures

### Current State (Monolith)

```
src/cascade/
├── __init__.py
├── detector.py      # CascadeDetector
├── taint.py         # TaintPropagator
├── revalidation.py  # RevalidationQueue, RevalidationEvaluator
└── rollback.py      # RollbackManager
```

**Problems:**
- Tightly coupled to provenance storage
- Cascade tasks mixed with execution tasks in `tasks.py`
- No independent scaling for revalidation processing

---

## API Specification

### Base URL

```
http://cascade-service:8000/api/v1
```

### Endpoints

#### POST /taint

Initiate taint propagation for an artifact.

**Request:**
```json
{
  "artifact_id": "art-123",
  "reason": "qa_failure",
  "source_id": "task-456",
  "severity": "high",
  "metadata": {
    "qa_score": 0.3,
    "issues": ["security_vulnerability", "logic_error"]
  }
}
```

**Response:**
```json
{
  "cascade_id": "cascade-789",
  "artifact_id": "art-123",
  "status": "propagating",
  "tainted_count": 1,
  "suspect_count": 5,
  "revalidation_queued": 5,
  "estimated_impact": {
    "total_artifacts": 6,
    "projects_affected": 1,
    "tasks_affected": 3
  }
}
```

#### GET /impact/{artifact_id}

Analyze impact if artifact were tainted.

**Response:**
```json
{
  "artifact_id": "art-123",
  "impact": {
    "direct_dependents": 3,
    "transitive_dependents": 12,
    "projects": ["proj-1", "proj-2"],
    "critical_paths": [
      ["art-123", "art-456", "art-789"]
    ]
  },
  "recommendation": "propagate",
  "risk_level": "medium"
}
```

#### POST /rollback

Rollback to a checkpoint.

**Request:**
```json
{
  "checkpoint_id": "chk-123",
  "reason": "cascade_threshold_exceeded",
  "dry_run": false
}
```

**Response:**
```json
{
  "rollback_id": "rb-456",
  "checkpoint_id": "chk-123",
  "status": "completed",
  "artifacts_reverted": 15,
  "tasks_to_requeue": 8,
  "dry_run": false
}
```

#### POST /checkpoints

Create a checkpoint.

**Request:**
```json
{
  "project_id": "proj-123",
  "description": "Pre-refactor checkpoint",
  "trigger": "manual"
}
```

#### GET /checkpoints/{project_id}

List checkpoints for a project.

#### GET /revalidation/queue

Get revalidation queue status.

**Response:**
```json
{
  "pending": 25,
  "processing": 3,
  "completed_today": 150,
  "failed_today": 5,
  "queue_health": "healthy"
}
```

#### GET /health

Service health check.

---

## Events

### Subscribed Events

| Event | Source | Action |
|-------|--------|--------|
| `task.rejected` | QA Service | Detect cascade, propagate taint |
| `security.alert` | External | Detect security-triggered cascade |
| `review.rejected` | Human Review | Detect human-rejection cascade |

### Published Events

| Event | Trigger | Data |
|-------|---------|------|
| `cascade.started` | Taint initiated | `{cascade_id, artifact_id, reason}` |
| `artifact.tainted` | After taint | `{artifact_id, cascade_id, reason}` |
| `artifact.suspect` | Dependent flagged | `{artifact_id, cascade_id, source_id}` |
| `revalidation.needed` | Suspect queued | `{artifact_id, priority}` |
| `rollback.requested` | Threshold exceeded | `{checkpoint_id, reason}` |
| `cascade.completed` | Processing done | `{cascade_id, stats}` |

---

## Data Models

### TaintRequest

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaintRequest(BaseModel):
    """Request to taint an artifact."""
    
    artifact_id: str = Field(..., description="Artifact to taint")
    reason: str = Field(..., description="Reason for tainting")
    source_id: Optional[str] = Field(None, description="What caused the taint")
    severity: Severity = Severity.MEDIUM
    metadata: Optional[Dict[str, Any]] = None
```

### CascadeResult

```python
class ImpactEstimate(BaseModel):
    total_artifacts: int
    projects_affected: int
    tasks_affected: int

class CascadeResult(BaseModel):
    """Result of cascade propagation."""
    
    cascade_id: str
    artifact_id: str
    status: str  # propagating, completed, failed
    tainted_count: int
    suspect_count: int
    revalidation_queued: int
    estimated_impact: ImpactEstimate
    errors: List[str] = Field(default_factory=list)
```

### RollbackRequest

```python
class RollbackRequest(BaseModel):
    """Request to rollback to checkpoint."""
    
    checkpoint_id: str
    reason: str
    dry_run: bool = False
```

---

## Architecture

### Internal Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     CASCADE SERVICE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                                                │
│  │  FastAPI    │◄─── HTTP Requests                              │
│  │  Router     │◄─── RabbitMQ Events                            │
│  └──────┬──────┘                                                │
│         │                                                        │
│  ┌──────┴──────────────────────────────────────────────────┐   │
│  │                  CASCADE DETECTOR                         │   │
│  │  - Analyze trigger (QA failure, rejection, security)     │   │
│  │  - Determine severity and scope                          │   │
│  │  - Decide: propagate, rollback, or manual review         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  TAINT PROPAGATOR                         │   │
│  │  - Mark source as tainted                                 │   │
│  │  - Walk provenance graph for dependents                   │   │
│  │  - Mark dependents as suspect                             │   │
│  │  - Publish events for each affected artifact              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                REVALIDATION MANAGER                       │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │    Queue     │  │  Evaluator   │  │  Processor   │   │   │
│  │  │  (Priority)  │  │  (via QA)    │  │  (Workers)   │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  ROLLBACK MANAGER                         │   │
│  │  - Checkpoint management                                  │   │
│  │  - Rollback execution                                     │   │
│  │  - Task requeue coordination                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 PROVENANCE CLIENT                         │   │
│  │  - Query provenance graph (via Orchestration/Storage)    │   │
│  │  - Update taint status                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
# config/cascade-service.yaml
cascade:
  max_depth: 10  # Maximum propagation depth
  batch_size: 50  # Artifacts to process per batch
  
thresholds:
  # When to recommend rollback
  suspect_count: 100
  cascade_depth: 5
  project_impact_ratio: 0.3  # 30% of project affected
  
revalidation:
  enabled: true
  queue_max_size: 1000
  priority_factors:
    - recency: 0.3
    - depth: 0.3
    - project_importance: 0.4
  retry_attempts: 3
  
rollback:
  enabled: true
  require_approval: true
  max_rollback_age_days: 30
  
events:
  rabbitmq:
    host: rabbitmq
    port: 5672
    exchanges:
      - task.events
      - cascade.events
    queue: cascade-service-events
```

---

## Migration Path

### Phase 1: Extract Cascade Logic (Week 1)

1. Create `services/cascade-service/` directory
2. Move code from `src/cascade/`:
   - `detector.py` → `src/services/detector.py`
   - `taint.py` → `src/services/taint_propagator.py`
   - `revalidation.py` → `src/services/revalidation.py`
   - `rollback.py` → `src/services/rollback.py`

3. Extract cascade-related tasks from `tasks.py`:
   - `propagate_taint`
   - `queue_revalidation`
   - `process_revalidation_batch`
   - `detect_cascade`
   - `execute_rollback`
   - `create_checkpoint`

### Phase 2: Add Provenance Client (Week 2)

Since Cascade needs provenance graph access:

```python
# dats_common/clients/provenance.py
class ProvenanceClient:
    """Client for provenance operations."""
    
    async def get_dependents(self, artifact_id: str) -> List[str]:
        """Get artifacts that depend on this one."""
        pass
    
    async def mark_tainted(self, artifact_id: str, reason: str) -> None:
        """Mark artifact as tainted."""
        pass
    
    async def mark_suspect(self, artifact_id: str, source_id: str) -> None:
        """Mark artifact as suspect."""
        pass
```

### Phase 3: Event-Driven Integration (Week 2)

```python
# src/events/handlers.py
class CascadeEventHandlers:
    async def handle_task_rejected(self, message: dict):
        """QA rejected a task - detect cascade."""
        artifact_id = message["data"]["artifact_id"]
        qa_result = message["data"]["qa_result"]
        
        scenario = await self.detector.detect_from_qa_failure(
            artifact_id=artifact_id,
            qa_result=qa_result,
        )
        
        if scenario.recommended_action == "propagate":
            await self.propagator.taint_artifact(
                artifact_id=artifact_id,
                reason=qa_result.get("reason", "qa_failure"),
            )
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dats-common /tmp/dats-common
RUN pip install /tmp/dats-common

COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Observability

### Metrics

```
# Cascade operations
cascade_started_total{trigger="qa_failure"}
cascade_completed_total{status="success"}
cascade_duration_seconds{quantile="0.95"}

# Taint tracking
taint_propagation_depth{cascade_id="..."}
artifacts_tainted_total
artifacts_suspect_total

# Revalidation
revalidation_queue_size
revalidation_processed_total{result="passed"}
revalidation_processed_total{result="failed"}
revalidation_latency_seconds{quantile="0.95"}

# Rollback
rollback_executed_total{trigger="threshold"}
rollback_artifacts_reverted_total
```

---

## Dependencies

```
# requirements.txt
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
pydantic>=2.5.0

# Message queue
pika>=1.3.0
aio-pika>=9.3.0

# Database
asyncpg>=0.29.0
sqlalchemy>=2.0.0

# Observability
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
prometheus-client>=0.19.0
```

---

## Success Criteria

- [ ] Taint propagation completes in < 30 seconds for 100 artifacts
- [ ] Revalidation queue processes 50 items/minute
- [ ] Zero missed dependents in propagation
- [ ] Rollback restores to checkpoint correctly
- [ ] Events published within 1 second of state change

---

*Last updated: January 2026*