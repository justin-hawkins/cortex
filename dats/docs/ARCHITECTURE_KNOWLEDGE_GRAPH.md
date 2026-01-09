# DATS Architecture Knowledge Graph

> **Distributed Agentic Task System** - A comprehensive map of functions, classes, connections, and data flows for LLM agent navigation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Data Flow](#core-data-flow)
3. [Module Dependency Map](#module-dependency-map)
4. [Class Hierarchy & Relationships](#class-hierarchy--relationships)
5. [Key Integration Points](#key-integration-points)
6. [Celery Task Reference](#celery-task-reference)
7. [API Routes Reference](#api-routes-reference)
8. [CLI Commands Reference](#cli-commands-reference)
9. [File Index by Concern](#file-index-by-concern)
10. [Quick Lookup Tables](#quick-lookup-tables)

---

## System Overview

DATS orchestrates AI agents to decompose, execute, and validate complex software development tasks. The system uses a tiered model architecture (tiny → small → large → frontier) with Celery for distributed task execution.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATS ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────────┐    ┌────────────┐    ┌─────────────────┐   │
│  │   CLI   │───▶│   FastAPI   │───▶│  Pipeline  │───▶│  Celery Queue   │   │
│  │ or API  │    │   /api/v1   │    │Orchestrator│    │  (Redis/RMQ)    │   │
│  └─────────┘    └─────────────┘    └────────────┘    └─────────────────┘   │
│                                           │                    │            │
│                                           ▼                    ▼            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         AGENT LAYER                                 │    │
│  │  ┌────────────┐  ┌───────────┐  ┌──────────────────┐  ┌──────────┐ │    │
│  │  │Coordinator │─▶│Decomposer │─▶│ComplexityEstimator│─▶│QAReviewer│ │    │
│  │  └────────────┘  └───────────┘  └──────────────────┘  └──────────┘ │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                           │                                 │
│                                           ▼                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        WORKER LAYER                                 │    │
│  │  ┌───────────┐  ┌──────────┐  ┌────────────┐  ┌─────────────────┐  │    │
│  │  │CodeGeneral│  │CodeVision│  │CodeEmbedded│  │ Documentation   │  │    │
│  │  └───────────┘  └──────────┘  └────────────┘  └─────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                           │                                 │
│                                           ▼                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        MODEL LAYER                                  │    │
│  │  ┌────────────┐  ┌────────────────────┐  ┌───────────────────┐     │    │
│  │  │OllamaClient│  │OpenAICompatibleClient│  │ AnthropicClient │     │    │
│  │  └────────────┘  └────────────────────┘  └───────────────────┘     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                           │                                 │
│                                           ▼                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       STORAGE LAYER                                 │    │
│  │  ┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │    │
│  │  │ProvenanceTracker │  │WorkProductStore │  │  LightRAG/RAG   │    │    │
│  │  └──────────────────┘  └─────────────────┘  └─────────────────┘    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Data Flow

### Task Submission → Execution Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         TASK EXECUTION FLOW                               │
└──────────────────────────────────────────────────────────────────────────┘

1. USER REQUEST
   │
   ▼
2. AgentPipeline.process_request()              [src/pipeline/orchestrator.py]
   │
   ├──▶ 3. Coordinator.analyze_task()           [src/agents/coordinator.py]
   │         └── Determines: mode, needs_decomposition, domain
   │
   ├──▶ 4. Decomposer.decompose_recursive()     [src/agents/decomposer.py]
   │         └── Breaks task into atomic subtasks (if needed)
   │
   ├──▶ 5. ComplexityEstimator.estimate()       [src/agents/complexity_estimator.py]
   │         └── Routes to tier: tiny/small/large/frontier
   │
   └──▶ 6. Celery Queue                         [src/queue/tasks.py]
            │
            ▼
7. _async_execute()                             [src/queue/tasks.py]
   │
   ├──▶ RAGQueryEngine.get_context_for_worker() [src/rag/query.py]
   │
   ├──▶ Worker.execute()                        [src/workers/*.py]
   │         └── Uses BaseModelClient.generate()
   │
   ├──▶ ProvenanceTracker.complete_record()     [src/storage/provenance.py]
   │
   └──▶ WorkProductStore.store()                [src/storage/work_product.py]
            │
            ▼
8. validate_task()                              [src/queue/tasks.py]
   │
   └──▶ QAReviewer.review()                     [src/agents/qa_reviewer.py]
            │
            ▼
9. EmbeddingTrigger.embed_text()                [src/rag/embedding_trigger.py]
   │
   ▼
10. RESULT STORED
```

### Cascade Failure Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      CASCADE FAILURE HANDLING                             │
└──────────────────────────────────────────────────────────────────────────┘

QA FAILURE / HUMAN REJECTION
   │
   ▼
detect_cascade()                                [src/queue/tasks.py]
   │
   └──▶ CascadeDetector                         [src/cascade/detector.py]
            │
            ▼
propagate_taint()                               [src/queue/tasks.py]
   │
   └──▶ TaintPropagator.taint_artifact()        [src/cascade/taint.py]
            │
            ├──▶ ProvenanceGraph.find_dependents()  [src/storage/provenance.py]
            │
            └──▶ ProvenanceTracker.mark_tainted()   [src/storage/provenance.py]
                 ProvenanceTracker.mark_suspect()
                      │
                      ▼
queue_revalidation()                            [src/queue/tasks.py]
   │
   └──▶ RevalidationQueue.add()                 [src/cascade/revalidation.py]
            │
            ▼
process_revalidation_batch()                    [src/queue/tasks.py]
   │
   └──▶ RevalidationEvaluator.evaluate()        [src/cascade/revalidation.py]
            │
            ├── PASS ──▶ ProvenanceTracker.clear_suspect()
            │
            └── FAIL ──▶ propagate_taint() (recursive)
                              │
                              ▼
                    (If thresholds exceeded)
                              │
                    execute_rollback()          [src/queue/tasks.py]
                         │
                         └──▶ RollbackManager   [src/cascade/rollback.py]
```

---

## Module Dependency Map

### Hierarchical View

```
src/
├── pipeline/                    # ENTRY POINT
│   └── orchestrator.py         
│       ├── uses: agents.Coordinator
│       ├── uses: agents.Decomposer
│       ├── uses: agents.ComplexityEstimator
│       ├── uses: storage.ProvenanceTracker
│       ├── uses: config.routing
│       └── uses: telemetry.config
│
├── agents/                      # ORCHESTRATION LAYER
│   ├── base.py                 # BaseAgent (abstract)
│   │   ├── uses: config.routing.RoutingConfig
│   │   ├── uses: models.*.Client
│   │   ├── uses: prompts.renderer
│   │   └── uses: telemetry.llm_tracer
│   │
│   ├── coordinator.py          # Task analysis & mode detection
│   ├── decomposer.py           # Task breakdown
│   ├── complexity_estimator.py # Tier routing
│   ├── qa_reviewer.py          # Output validation
│   └── merge_coordinator.py    # Result merging
│
├── workers/                     # EXECUTION LAYER
│   ├── base.py                 # BaseWorker (abstract)
│   │   ├── uses: config.routing.RoutingConfig
│   │   ├── uses: models.*.Client
│   │   ├── uses: prompts.renderer
│   │   └── uses: telemetry.llm_tracer
│   │
│   ├── code_general.py         # domain: code-general
│   ├── code_vision.py          # domain: code-vision
│   ├── code_embedded.py        # domain: code-embedded
│   ├── documentation.py        # domain: documentation
│   └── ui_design.py            # domain: ui-design
│
├── models/                      # LLM CLIENT LAYER
│   ├── base.py                 # BaseModelClient (abstract)
│   ├── ollama_client.py        # Ollama API
│   ├── openai_client.py        # OpenAI-compatible API
│   ├── anthropic_client.py     # Anthropic API
│   └── mock_client.py          # Testing
│
├── queue/                       # DISTRIBUTED EXECUTION
│   ├── celery_app.py           # Celery configuration
│   └── tasks.py                # All Celery tasks
│       ├── uses: workers.*
│       ├── uses: agents.*
│       ├── uses: storage.*
│       ├── uses: rag.*
│       └── uses: cascade.*
│
├── storage/                     # PERSISTENCE LAYER
│   ├── provenance.py           # ProvenanceTracker, ProvenanceGraph
│   │   └── uses: telemetry
│   └── work_product.py         # WorkProductStore
│
├── config/                      # CONFIGURATION
│   ├── settings.py             # Environment settings (pydantic)
│   └── routing.py              # Model tier routing
│
├── api/                         # HTTP API
│   ├── app.py                  # FastAPI application
│   ├── dependencies.py         # Dependency injection
│   ├── schemas.py              # Pydantic models
│   └── routes/                 # API endpoints
│       ├── tasks.py
│       ├── review.py
│       ├── projects.py
│       ├── monitoring.py
│       └── provenance.py
│
├── cli/                         # COMMAND LINE
│   ├── main.py                 # Click application
│   ├── output.py               # Formatting utilities
│   └── commands/
│       ├── submit.py
│       ├── status.py
│       ├── review.py
│       ├── project.py
│       ├── config.py
│       └── monitoring.py
│
├── qa/                          # QUALITY ASSURANCE
│   ├── pipeline.py             # QAPipeline
│   ├── profiles.py             # QA profile definitions
│   ├── results.py              # Result dataclasses
│   ├── human_review.py         # Human-in-the-loop
│   └── validators/
│       ├── base.py
│       ├── consensus.py
│       ├── adversarial.py
│       ├── security.py
│       ├── testing.py
│       └── documentation.py
│
├── merge/                       # CONFLICT RESOLUTION
│   ├── coordinator.py          # MergeCoordinator
│   ├── detector.py             # ConflictDetector
│   ├── classifier.py           # ConflictClassifier
│   ├── strategies.py           # ResolutionStrategySelector
│   ├── models.py               # Data models
│   ├── config.py               # MergeConfig
│   └── resolvers/
│       ├── base.py
│       ├── textual.py
│       ├── semantic.py
│       └── architectural.py
│
├── cascade/                     # FAILURE HANDLING
│   ├── detector.py             # CascadeDetector
│   ├── taint.py                # TaintPropagator
│   ├── revalidation.py         # RevalidationQueue, RevalidationEvaluator
│   └── rollback.py             # RollbackManager
│
├── rag/                         # CONTEXT RETRIEVAL
│   ├── client.py               # LightRAG client
│   ├── query.py                # RAGQueryEngine
│   └── embedding_trigger.py    # EmbeddingTrigger
│
├── prompts/                     # PROMPT MANAGEMENT
│   ├── loader.py               # Load from prompts/ directory
│   └── renderer.py             # PromptRenderer (Jinja2)
│
└── telemetry/                   # OBSERVABILITY
    ├── config.py               # OpenTelemetry setup
    ├── decorators.py           # @traced decorator
    ├── context.py              # Distributed trace context
    └── llm_tracer.py           # LLM-specific tracing
```

---

## Class Hierarchy & Relationships

### Agent Classes

| Class | File | Extends | Key Methods | Purpose |
|-------|------|---------|-------------|---------|
| `BaseAgent` | `agents/base.py` | `ABC` | `execute()`, `get_model_client()`, `_render_prompt()` | Abstract base for all agents |
| `Coordinator` | `agents/coordinator.py` | `BaseAgent` | `analyze_task()` | Analyzes requests, determines mode |
| `Decomposer` | `agents/decomposer.py` | `BaseAgent` | `decompose()`, `decompose_recursive()` | Breaks tasks into subtasks |
| `ComplexityEstimator` | `agents/complexity_estimator.py` | `BaseAgent` | `estimate()` | Routes to model tier |
| `QAReviewer` | `agents/qa_reviewer.py` | `BaseAgent` | `review()` | Validates worker output |
| `MergeCoordinator` | `agents/merge_coordinator.py` | `BaseAgent` | `merge()` | Merges subtask results |

### Worker Classes

| Class | File | Extends | Domain | Purpose |
|-------|------|---------|--------|---------|
| `BaseWorker` | `workers/base.py` | `ABC` | - | Abstract base for all workers |
| `CodeGeneralWorker` | `workers/code_general.py` | `BaseWorker` | `code-general` | General code generation |
| `CodeVisionWorker` | `workers/code_vision.py` | `BaseWorker` | `code-vision` | Computer vision code |
| `CodeEmbeddedWorker` | `workers/code_embedded.py` | `BaseWorker` | `code-embedded` | Embedded systems code |
| `DocumentationWorker` | `workers/documentation.py` | `BaseWorker` | `documentation` | Documentation generation |
| `UIDesignWorker` | `workers/ui_design.py` | `BaseWorker` | `ui-design` | UI/UX code generation |

### Model Client Classes

| Class | File | Extends | Type | Purpose |
|-------|------|---------|------|---------|
| `BaseModelClient` | `models/base.py` | `ABC` | - | Abstract LLM client interface |
| `OllamaClient` | `models/ollama_client.py` | `BaseModelClient` | `ollama` | Ollama API client |
| `OpenAICompatibleClient` | `models/openai_client.py` | `BaseModelClient` | `openai_compatible` | OpenAI/vLLM client |
| `AnthropicClient` | `models/anthropic_client.py` | `BaseModelClient` | `anthropic` | Anthropic API client |
| `MockModelClient` | `models/mock_client.py` | `BaseModelClient` | `mock` | Testing client |

### Storage Classes

| Class | File | Purpose |
|-------|------|---------|
| `ProvenanceTracker` | `storage/provenance.py` | Track task execution lineage |
| `ProvenanceRecord` | `storage/provenance.py` | Single execution record |
| `ProvenanceGraph` | `storage/provenance.py` | DAG operations for impact analysis |
| `TaintInfo` | `storage/provenance.py` | Taint status tracking |
| `WorkProductStore` | `storage/work_product.py` | Artifact storage |

### Configuration Classes

| Class | File | Purpose |
|-------|------|---------|
| `Settings` | `config/settings.py` | Environment configuration |
| `RoutingConfig` | `config/routing.py` | Model tier routing |
| `ModelTier` | `config/routing.py` | Tier definition (tiny/small/large/frontier) |
| `ModelConfig` | `config/routing.py` | Individual model configuration |
| `AgentRouting` | `config/routing.py` | Agent-to-tier mapping |

---

## Key Integration Points

### 1. Agent → Model Client Flow

```python
# In BaseAgent.get_model_client()
RoutingConfig.get_agent_routing(agent_name)  # Get routing for agent
    → RoutingConfig.get_tier(tier_name)       # Get tier config
    → ModelTier.get_primary_model()           # Get model config
    → BaseAgent._create_client(model_config)  # Create appropriate client
```

**Files involved:**
- `src/agents/base.py` - `get_model_client()`, `_create_client()`
- `src/config/routing.py` - `RoutingConfig`, `ModelTier`, `ModelConfig`

### 2. Pipeline → Celery Queue Flow

```python
# In AgentPipeline
AgentPipeline.process_request()
    → _coordinate()           # Coordinator.analyze_task()
    → _decompose_recursive()  # Decomposer.decompose()
    → _estimate_and_queue()   # ComplexityEstimator.estimate()
        → _queue_to_celery()  # execute_task.delay(task_data, tier)
```

**Files involved:**
- `src/pipeline/orchestrator.py` - `AgentPipeline`
- `src/queue/tasks.py` - `execute_task`

### 3. Worker Execution Flow

```python
# In queue/tasks.py
_async_execute(task_data, tier)
    → RAGQueryEngine.get_context_for_worker()  # Get RAG context
    → _get_worker_for_domain(domain)            # Select worker
    → Worker.execute()                          # Execute task
    → ProvenanceTracker.complete_record()       # Record provenance
    → WorkProductStore.store()                  # Store artifacts
```

**Files involved:**
- `src/queue/tasks.py` - `_async_execute()`, `_get_worker_for_domain()`
- `src/workers/*.py` - Worker implementations
- `src/rag/query.py` - `RAGQueryEngine`
- `src/storage/provenance.py` - `ProvenanceTracker`
- `src/storage/work_product.py` - `WorkProductStore`

### 4. Provenance Graph Operations

```python
# In ProvenanceTracker
ProvenanceTracker.impact_analysis(artifact_id)
    → ProvenanceGraph.find_dependents(artifact_id)  # Get dependents
    → ProvenanceGraph.find_path(from, to)           # Trace lineage

ProvenanceTracker.root_cause_analysis(artifact_id)
    → ProvenanceGraph.find_dependencies()           # Get inputs
    → Trace back to tainted source
```

**Files involved:**
- `src/storage/provenance.py` - `ProvenanceTracker`, `ProvenanceGraph`

### 5. Cascade Failure Handling

```python
# Trigger cascade
detect_cascade(artifact_id, trigger_type, details)
    → CascadeDetector.detect_from_qa_failure()
    → propagate_taint.delay(artifact_id, reason)

# Propagate taint
propagate_taint(artifact_id, reason)
    → TaintPropagator.taint_artifact()
    → ProvenanceTracker.mark_tainted()
    → ProvenanceTracker.mark_suspect()  # For dependents
    → queue_revalidation.delay()        # For suspect artifacts

# Process revalidation
process_revalidation_batch()
    → RevalidationEvaluator.evaluate()
    → Clear suspect OR propagate taint
```

**Files involved:**
- `src/queue/tasks.py` - Celery task orchestration
- `src/cascade/detector.py` - `CascadeDetector`
- `src/cascade/taint.py` - `TaintPropagator`
- `src/cascade/revalidation.py` - `RevalidationQueue`, `RevalidationEvaluator`
- `src/cascade/rollback.py` - `RollbackManager`

---

## Celery Task Reference

| Task Name | Function | Trigger | Description |
|-----------|----------|---------|-------------|
| `execute_task` | `execute_task()` | Pipeline queue | Routes to tier-specific task |
| `execute_tiny` | `execute_tiny()` | execute_task | Execute on tiny tier |
| `execute_small` | `execute_small()` | execute_task | Execute on small tier |
| `execute_large` | `execute_large()` | execute_task | Execute on large tier |
| `execute_frontier` | `execute_frontier()` | execute_task | Execute on frontier tier |
| `decompose_task` | `decompose_task()` | Manual | Decompose complex task |
| `validate_task` | `validate_task()` | After execution | Run QA validation |
| `trigger_embedding` | `_trigger_embedding()` | After QA pass | Index in RAG |
| `merge_results` | `merge_results()` | After subtasks | Merge subtask outputs |
| `process_pipeline` | `process_pipeline()` | API/CLI | Full pipeline entry point |
| `propagate_taint` | `propagate_taint()` | Cascade detection | Propagate taint through graph |
| `queue_revalidation` | `queue_revalidation()` | Taint propagation | Queue suspect for revalidation |
| `process_revalidation_batch` | `process_revalidation_batch()` | Celery beat | Process pending revalidations |
| `detect_cascade` | `detect_cascade()` | QA failure/rejection | Detect cascade scenario |
| `execute_rollback` | `execute_rollback()` | Manual/threshold | Rollback to checkpoint |
| `create_checkpoint` | `create_checkpoint()` | Manual/milestone | Create rollback point |

---

## API Routes Reference

| Endpoint | Method | Handler | Purpose |
|----------|--------|---------|---------|
| `/api/v1/tasks` | POST | `tasks.submit_task` | Submit new task |
| `/api/v1/tasks/{id}` | GET | `tasks.get_task` | Get task status |
| `/api/v1/tasks/{id}/cancel` | POST | `tasks.cancel_task` | Cancel task |
| `/api/v1/review/{id}` | GET | `review.get_review` | Get QA review |
| `/api/v1/review/{id}/approve` | POST | `review.approve` | Approve output |
| `/api/v1/review/{id}/reject` | POST | `review.reject` | Reject output |
| `/api/v1/projects` | GET | `projects.list_projects` | List projects |
| `/api/v1/projects/{id}` | GET | `projects.get_project` | Get project |
| `/api/v1/monitoring/health` | GET | `monitoring.health` | Health check |
| `/api/v1/monitoring/metrics` | GET | `monitoring.metrics` | System metrics |
| `/api/v1/provenance/{id}` | GET | `provenance.get_record` | Get provenance |
| `/api/v1/provenance/{id}/lineage` | GET | `provenance.get_lineage` | Get artifact lineage |

---

## CLI Commands Reference

| Command | File | Description |
|---------|------|-------------|
| `dats submit` | `cli/commands/submit.py` | Submit task |
| `dats status` | `cli/commands/status.py` | Check task status |
| `dats review` | `cli/commands/review.py` | Review/approve outputs |
| `dats project` | `cli/commands/project.py` | Project management |
| `dats config` | `cli/commands/config.py` | Configuration |
| `dats monitoring` | `cli/commands/monitoring.py` | System monitoring |

---

## File Index by Concern

### "I need to add a new worker domain"

1. Create worker class in `src/workers/new_worker.py`
   - Extend `BaseWorker`
   - Set `worker_name` and `domain` class attributes
   - Implement `_render_prompt()`
2. Add to `src/workers/__init__.py`
3. Add prompt template to `prompts/workers/new_domain.md`
4. Update `_get_worker_for_domain()` in `src/queue/tasks.py`

### "I need to add a new agent"

1. Create agent class in `src/agents/new_agent.py`
   - Extend `BaseAgent`
   - Set `agent_name` class attribute
   - Implement `_render_prompt()`
2. Add to `src/agents/__init__.py`
3. Add prompt template to `prompts/agents/new_agent.md`
4. Add routing config in `prompts/schemas/routing_config.yaml`

### "I need to modify model routing"

1. Edit `prompts/schemas/routing_config.yaml`
   - Add/modify model tiers
   - Add/modify agent routing
2. Classes affected:
   - `src/config/routing.py` - `RoutingConfig`, `ModelTier`

### "I need to add a new Celery task"

1. Add task function in `src/queue/tasks.py`
   - Decorate with `@app.task(base=DATSTask, ...)`
2. Add task name to any triggering code

### "I need to add a new API endpoint"

1. Add route in appropriate `src/api/routes/*.py`
2. Add schemas in `src/api/schemas.py` if needed
3. Router is already included in `src/api/app.py`

### "I need to modify provenance tracking"

1. Main file: `src/storage/provenance.py`
   - `ProvenanceRecord` - record structure
   - `ProvenanceTracker` - CRUD operations
   - `ProvenanceGraph` - DAG operations

### "I need to modify cascade/taint handling"

1. Detection: `src/cascade/detector.py` - `CascadeDetector`
2. Propagation: `src/cascade/taint.py` - `TaintPropagator`
3. Revalidation: `src/cascade/revalidation.py` - `RevalidationQueue`
4. Rollback: `src/cascade/rollback.py` - `RollbackManager`
5. Celery tasks: `src/queue/tasks.py` - cascade task functions

### "I need to modify QA validation"

1. Profiles: `src/qa/profiles.py`
2. Validators: `src/qa/validators/*.py`
3. Pipeline: `src/qa/pipeline.py`
4. Human review: `src/qa/human_review.py`

### "I need to add telemetry/observability"

1. Configuration: `src/telemetry/config.py`
2. Decorators: `src/telemetry/decorators.py`
3. LLM tracing: `src/telemetry/llm_tracer.py`
4. Distributed context: `src/telemetry/context.py`

---

## Quick Lookup Tables

### Model Tiers

| Tier | Context Window | Safe Limit | Example Models |
|------|---------------|------------|----------------|
| `tiny` | 8K | 6K | gemma3:4b |
| `small` | 16K | 12K | gemma3:12b |
| `large` | 32K | 22K | qwen3-coder, gpt-oss |
| `frontier` | 200K | 150K | claude-sonnet-4 |

### Agent → Tier Mapping

| Agent | Preferred Tier | Purpose |
|-------|---------------|---------|
| `coordinator` | large | Task analysis needs reasoning |
| `decomposer` | large | Complex task breakdown |
| `complexity_estimator` | small | Quick classification |
| `qa_reviewer` | large | Output validation |
| `merge_coordinator` | large | Conflict resolution |

### Worker → Domain Mapping

| Worker | Domain | File |
|--------|--------|------|
| CodeGeneralWorker | `code-general` | `workers/code_general.py` |
| CodeVisionWorker | `code-vision` | `workers/code_vision.py` |
| CodeEmbeddedWorker | `code-embedded` | `workers/code_embedded.py` |
| DocumentationWorker | `documentation` | `workers/documentation.py` |
| UIDesignWorker | `ui-design` | `workers/ui_design.py` |

### Verification Status Values

| Status | Meaning |
|--------|---------|
| `pending` | Not yet verified |
| `passed` | QA passed |
| `failed` | QA failed |
| `human_approved` | Human approved |
| `tainted` | Marked as bad |
| `suspect` | Potentially affected by upstream taint |

### Task Mode Values

| Mode | Description |
|------|-------------|
| `new_project` | Create new project |
| `modify` | Modify existing code |
| `fix_bug` | Bug fix |
| `refactor` | Code refactoring |
| `documentation` | Documentation task |
| `testing` | Test writing |

---

## External Dependencies

### Prompt Templates Location

```
prompts/
├── agents/
│   ├── coordinator.md
│   ├── decomposer.md
│   ├── complexity_estimator.md
│   ├── qa_reviewer.md
│   └── merge_coordinator.md
├── workers/
│   ├── code_general.md
│   ├── code_vision.md
│   ├── code_embedded.md
│   ├── documentation.md
│   └── ui_design.md
└── schemas/
    ├── routing_config.yaml    # Model routing configuration
    ├── task_schema.yaml       # Task data schema
    └── provenance_schema.yaml # Provenance record schema
```

### Data Storage Locations

```
data/
├── provenance/           # Provenance JSON records
├── rag/                  # LightRAG working directory
└── work_products/        # Artifact storage
    ├── .artifact_index.json
    └── artifacts/
```

---

## Key Dataclasses Summary

### From `pipeline/orchestrator.py`
- `SubtaskResult` - Result of subtask execution
- `PipelineResult` - Full pipeline result

### From `agents/base.py`
- `AgentContext` - Agent execution context
- `AgentResult` - Agent execution result

### From `workers/base.py`
- `WorkerContext` - Worker execution context
- `WorkerResult` - Worker execution result

### From `models/base.py`
- `ModelResponse` - LLM response with tokens/metadata

### From `storage/provenance.py`
- `ProvenanceRecord` - Full provenance record
- `ArtifactRef` - Reference to output artifact
- `InputRef` - Reference to input artifact
- `ExecutionContext` - Model/worker/prompt info
- `VerificationInfo` - QA verification status
- `TaintInfo` - Taint tracking info
- `TaintEvent` - Audit record for taint
- `Checkpoint` - Rollback point
- `ImpactReport` - Impact analysis result
- `RootCauseReport` - Root cause analysis result

### From `config/routing.py`
- `RoutingConfig` - Complete routing config
- `ModelTier` - Tier configuration
- `ModelConfig` - Individual model config
- `AgentRouting` - Agent-to-tier mapping

---

*Last updated: January 2026*