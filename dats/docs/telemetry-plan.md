# DATS Comprehensive Telemetry Plan

## Objective
Add telemetry spans for each stage of the pipeline to provide complete visibility into the data flow and processing at each subsystem.

## Current State
Based on the E2E test, we currently capture:
- `pipeline.process_request` - main entry point
- `pipeline.coordinate` - coordination phase
- `pipeline.estimate_and_queue` - complexity estimation and queuing
- `rag.get_context` - RAG context retrieval
- `worker.{domain}.execute` - worker execution
- `llm.{provider}.generate` - LLM calls (newly added with prompt/response)

## Missing Telemetry Stages

Based on the architecture diagram, we need spans for:

### 1. COORDINATOR Agent
**File:** `dats/src/agents/coordinator.py`
- [ ] `coordinator.analyze_request` - Initial request analysis
- [ ] `coordinator.determine_mode` - Mode selection (simple/complex)
- [ ] `coordinator.identify_checkpoints` - Checkpoint identification
- **Attributes:** mode, checkpoint_count, request_complexity

### 2. DECOMPOSER Agent
**File:** `dats/src/agents/decomposer.py`
- [ ] `decomposer.break_task` - Main decomposition
- [ ] `decomposer.create_subtask` - Per-subtask creation
- [ ] `decomposer.check_atomicity` - Atomicity check
- **Attributes:** subtask_count, depth_level, is_atomic

### 3. COMPLEXITY ESTIMATOR Agent
**File:** `dats/src/agents/complexity_estimator.py`
- [ ] `estimator.analyze_complexity` - Complexity analysis
- [ ] `estimator.assign_tier` - Tier assignment
- **Attributes:** estimated_complexity, assigned_tier, reasoning

### 4. CELERY QUEUES
**File:** `dats/src/queue/tasks.py`
- [ ] `queue.enqueue_task` - Task queued
- [ ] `queue.dequeue_task` - Task picked up
- [ ] `queue.task_waiting` - Time in queue
- **Attributes:** queue_name, tier, wait_time_ms, position_in_queue

### 5. WORKERS (Domain-Specific)
**Files:** `dats/src/workers/code_general.py`, etc.
- [ ] `worker.{domain}.receive_task` - Task received
- [ ] `worker.{domain}.check_complexity` - Can worker handle it?
- [ ] `worker.{domain}.refuse_task` - If too complex
- [ ] `worker.{domain}.execute` - Main execution (exists)
- [ ] `worker.{domain}.produce_output` - Output generated
- **Attributes:** domain, can_handle, refusal_reason, output_type, artifact_count

### 6. QA PIPELINE
**File:** `dats/src/qa/pipeline.py`
- [ ] `qa.pipeline.start` - QA validation starts
- [ ] `qa.consensus.validate` - Consensus validation
- [ ] `qa.adversarial.validate` - Adversarial testing
- [ ] `qa.security.validate` - Security checks
- [ ] `qa.human.request_review` - Human review request
- [ ] `qa.pipeline.complete` - Final verdict
- **Attributes:** validators_run, pass_count, fail_count, overall_result, needs_human_review

### 7. EMBEDDING TRIGGER
**File:** `dats/src/rag/embedding_trigger.py`
- [ ] `embedding.trigger.start` - Trigger activated
- [ ] `embedding.generate` - Embedding generation
- [ ] `embedding.store` - Storage to vector DB
- **Attributes:** content_length, embedding_dimension, storage_target

### 8. PROVENANCE TRACKING
**File:** `dats/src/storage/provenance.py`
- [ ] `provenance.create_record` - New record created
- [ ] `provenance.link_parent` - Parent linkage
- [ ] `provenance.save` - Persistence
- **Attributes:** provenance_id, parent_ids, task_id, project_id

### 9. MERGE COORDINATOR
**File:** `dats/src/merge/coordinator.py`
- [ ] `merge.detect_conflicts` - Conflict detection
- [ ] `merge.classify_conflict` - Conflict classification
- [ ] `merge.resolve_conflict` - Resolution strategy
- [ ] `merge.apply_resolution` - Applied merge
- **Attributes:** conflict_count, conflict_types, resolution_strategy, success

### 10. WORK PRODUCT STORE
**File:** `dats/src/storage/work_product.py`
- [ ] `workproduct.create_artifact` - Artifact creation
- [ ] `workproduct.save` - Save to store
- [ ] `workproduct.index` - Index update
- **Attributes:** artifact_id, artifact_type, file_path, size_bytes

## Implementation Approach

### Phase 1: Add Missing Span Decorators
Each subsystem will get the appropriate `@trace_async` or `@trace_sync` decorator with:
- Span name following the pattern `{subsystem}.{operation}`
- Relevant attributes for the operation
- Events for input/output data where applicable

### Phase 2: Add Stage-Level Input/Output Events
Similar to the LLM tracer, add span events for:
- `{stage}.input` - What data came into this stage
- `{stage}.output` - What data came out of this stage
- `{stage}.decision` - Key decisions made at this stage

### Phase 3: Update Grafana Dashboard
Add new dashboard panels for:
- Pipeline stage timeline
- Subsystem breakdown
- QA pass/fail rates
- Queue wait times
- Merge conflict statistics

## Files to Modify

| File | Changes |
|------|---------|
| `dats/src/agents/coordinator.py` | Add telemetry decorators and events |
| `dats/src/agents/decomposer.py` | Add telemetry decorators and events |
| `dats/src/agents/complexity_estimator.py` | Add telemetry decorators and events |
| `dats/src/queue/tasks.py` | Add queue telemetry spans |
| `dats/src/workers/base.py` | Enhance worker telemetry |
| `dats/src/qa/pipeline.py` | Add QA pipeline spans |
| `dats/src/qa/validators/*.py` | Add per-validator spans |
| `dats/src/rag/embedding_trigger.py` | Add embedding trigger spans |
| `dats/src/storage/provenance.py` | Add provenance tracking spans |
| `dats/src/merge/coordinator.py` | Add merge coordination spans |
| `dats/src/storage/work_product.py` | Add work product spans |
| `dats/grafana/dashboards/dats-observability.json` | Add new panels |

## Span Attribute Standards

All spans should include:
- `dats.task.id` - Current task ID
- `dats.project.id` - Current project ID
- `dats.stage` - Pipeline stage name
- `dats.{stage}.input_summary` - Brief input description
- `dats.{stage}.output_summary` - Brief output description
- Timing is automatic via OpenTelemetry

## Span Event Standards

Input/output events should capture:
- Timestamp (automatic)
- Event name: `{stage}.{event_type}`
- Attributes with the actual data (truncated if too large)