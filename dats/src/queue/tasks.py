"""
Celery task definitions for DATS.

Provides task execution functions for each model tier.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from celery import Task
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

from src.queue.celery_app import app
from src.storage.provenance import ProvenanceTracker
from src.storage.work_product import WorkProductStore
from src.telemetry.config import get_tracer
from src.telemetry.context import extract_task_context, add_task_context

logger = logging.getLogger(__name__)

# Global instances for task execution
_provenance_tracker: Optional[ProvenanceTracker] = None
_work_product_store: Optional[WorkProductStore] = None


def get_provenance_tracker() -> ProvenanceTracker:
    """Get or create provenance tracker singleton."""
    global _provenance_tracker
    if _provenance_tracker is None:
        from src.config.settings import get_settings
        settings = get_settings()
        storage_path = getattr(settings, "provenance_path", "data/provenance")
        _provenance_tracker = ProvenanceTracker(storage_path=storage_path)
    return _provenance_tracker


def get_work_product_store() -> WorkProductStore:
    """Get or create work product store singleton."""
    global _work_product_store
    if _work_product_store is None:
        from src.config.settings import get_settings
        settings = get_settings()
        storage_path = getattr(settings, "work_product_path", "data/work_products")
        _work_product_store = WorkProductStore(base_path=storage_path)
    return _work_product_store


class DATSTask(Task):
    """Base task class with error handling and logging."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(
            f"Task {task_id} failed: {exc}",
            extra={
                "task_id": task_id,
                "args": args,
                "kwargs": kwargs,
                "exception": str(exc),
            },
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Handle task success."""
        logger.info(
            f"Task {task_id} completed successfully",
            extra={"task_id": task_id},
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handle task retry."""
        logger.warning(
            f"Task {task_id} retrying: {exc}",
            extra={"task_id": task_id, "exception": str(exc)},
        )


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.execute_task")
def execute_task(
    self,
    task_data: dict[str, Any],
    tier: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generic task executor that routes to the appropriate tier.

    Args:
        task_data: Task configuration from task_schema
        tier: Optional tier override

    Returns:
        Execution result dictionary
    """
    # Determine tier from task data or use override
    execution_tier = tier or task_data.get("routing", {}).get("tier", "small")

    # Route to appropriate tier-specific task
    tier_tasks = {
        "tiny": execute_tiny,
        "small": execute_small,
        "large": execute_large,
        "frontier": execute_frontier,
    }

    tier_task = tier_tasks.get(execution_tier)
    if not tier_task:
        raise ValueError(f"Unknown tier: {execution_tier}")

    # Execute synchronously for now (tier tasks handle async internally)
    return tier_task.delay(task_data).get()


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.execute_tiny")
def execute_tiny(self, task_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute task using tiny tier model (gemma3:4b).

    Args:
        task_data: Task configuration

    Returns:
        Execution result dictionary
    """
    return _execute_with_tier(task_data, "tiny")


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.execute_small")
def execute_small(self, task_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute task using small tier model (gemma3:12b).

    Args:
        task_data: Task configuration

    Returns:
        Execution result dictionary
    """
    return _execute_with_tier(task_data, "small")


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.execute_large")
def execute_large(self, task_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute task using large tier model (qwen3-coder, gpt-oss).

    Args:
        task_data: Task configuration

    Returns:
        Execution result dictionary
    """
    return _execute_with_tier(task_data, "large")


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.execute_frontier")
def execute_frontier(self, task_data: dict[str, Any]) -> dict[str, Any]:
    """
    Execute task using frontier tier model (claude-sonnet-4).

    Args:
        task_data: Task configuration

    Returns:
        Execution result dictionary
    """
    return _execute_with_tier(task_data, "frontier")


def _execute_with_tier(task_data: dict[str, Any], tier: str) -> dict[str, Any]:
    """
    Internal function to execute a task with a specific tier.

    This is a stub implementation. Full implementation will:
    1. Load the appropriate model client
    2. Load and render the prompt template
    3. Execute the model generation
    4. Process and validate the response
    5. Create provenance record
    6. Return the result

    Args:
        task_data: Task configuration
        tier: Model tier to use

    Returns:
        Execution result dictionary
    """
    task_id = task_data.get("id", str(uuid.uuid4()))
    task_type = task_data.get("type", "execute")
    domain = task_data.get("domain", "code-general")
    
    # Extract trace context from task data for distributed tracing
    parent_context = extract_task_context(task_data)
    tracer = get_tracer("dats.queue.tasks")
    
    with tracer.start_as_current_span(
        f"execute_task.{tier}",
        context=parent_context,
        kind=SpanKind.CONSUMER,
        attributes={
            "dats.task.id": task_id,
            "dats.task.type": task_type,
            "dats.task.tier": tier,
            "dats.task.domain": domain,
        },
    ) as span:
        logger.info(
            f"Executing task {task_id} on tier {tier}",
            extra={
                "task_id": task_id,
                "tier": tier,
                "type": task_type,
                "domain": domain,
            },
        )

        # Run async execution in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_async_execute(task_data, tier))
            span.set_status(Status(StatusCode.OK))
            
            # Add result info to span
            if result.get("status") == "completed":
                provenance = result.get("provenance", {})
                span.set_attribute("dats.tokens.input", provenance.get("tokens_input", 0))
                span.set_attribute("dats.tokens.output", provenance.get("tokens_output", 0))
                span.set_attribute("dats.execution_time_ms", provenance.get("execution_time_ms", 0))
            
            return result
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


async def _async_execute(task_data: dict[str, Any], tier: str) -> dict[str, Any]:
    """
    Full async task execution implementation.

    Executes the complete task workflow:
    1. Get RAG context for the task
    2. Select appropriate worker based on domain
    3. Execute with the model
    4. Record provenance
    5. Store work products
    6. Return results

    Args:
        task_data: Task configuration
        tier: Model tier

    Returns:
        Execution result with output and provenance
    """
    from src.config.routing import get_routing_config
    from src.rag.query import RAGQueryEngine

    task_id = task_data.get("id", str(uuid.uuid4()))
    task_type = task_data.get("type", "execute")
    domain = task_data.get("domain", "code-general")
    project_id = task_data.get("project_id", "")
    description = task_data.get("description", "")
    
    tracer = get_tracer("dats.queue.tasks")

    logger.info(
        f"Starting task execution: {task_id}",
        extra={"task_id": task_id, "tier": tier, "domain": domain},
    )

    # Get routing config
    routing_config = get_routing_config()
    model_tier = routing_config.get_tier(tier)

    if not model_tier:
        raise ValueError(f"Unknown tier: {tier}")

    # Get primary model for tier
    model_config = model_tier.get_primary_model()
    if not model_config:
        raise ValueError(f"No models configured for tier: {tier}")

    started_at = datetime.utcnow()

    # Initialize provenance tracking
    provenance_tracker = get_provenance_tracker()
    provenance_record = provenance_tracker.create_record(
        task_id=task_id,
        project_id=project_id,
        model_used=model_config.name,
        worker_id=domain,
    )

    try:
        # Step 1: Get RAG context
        rag_context = ""
        with tracer.start_as_current_span(
            "rag.get_context",
            attributes={"dats.task.id": task_id, "dats.task.domain": domain},
        ) as rag_span:
            try:
                rag_engine = RAGQueryEngine()
                rag_context = await rag_engine.get_context_for_worker(
                    task_data=task_data,
                    safe_working_limit=model_tier.safe_working_limit,
                    context_budget_ratio=0.3,
                )
                await rag_engine.close()
                rag_span.set_attribute("dats.rag.context_length", len(rag_context))
                rag_span.set_status(Status(StatusCode.OK))
            except Exception as e:
                logger.warning(f"RAG context retrieval failed: {e}")
                rag_span.set_status(Status(StatusCode.ERROR, str(e)))
                # Continue without RAG context

        # Add RAG context to task data
        if rag_context:
            task_data["lightrag_context"] = rag_context

        # Step 2: Get appropriate worker
        worker = _get_worker_for_domain(domain, tier)

        # Step 3: Execute with worker
        with tracer.start_as_current_span(
            f"worker.{domain}.execute",
            attributes={
                "dats.task.id": task_id,
                "dats.worker.domain": domain,
                "dats.model.name": model_config.name,
            },
        ) as worker_span:
            worker_result = await worker.execute(task_data)
            
            if worker_result.success:
                worker_span.set_status(Status(StatusCode.OK))
                worker_span.set_attribute("dats.worker.execution_time_ms", worker_result.execution_time_ms)
            else:
                worker_span.set_status(Status(StatusCode.ERROR, worker_result.error or "Unknown error"))

        if not worker_result.success:
            raise Exception(f"Worker execution failed: {worker_result.error}")

        # Step 4: Process output
        content = worker_result.content
        artifacts = worker_result.artifacts

        # Step 5: Store work products
        work_store = get_work_product_store()
        stored_artifacts = []

        for artifact in artifacts:
            stored = work_store.store(
                content=artifact.get("content", ""),
                artifact_type=artifact.get("type", "code"),
                language=artifact.get("language", "text"),
                metadata={
                    "task_id": task_id,
                    "project_id": project_id,
                    "domain": domain,
                },
            )
            stored_artifacts.append({
                "artifact_id": stored.id,
                "type": artifact.get("type", "code"),
                "language": artifact.get("language", "text"),
                "checksum": stored.checksum,
            })

        # Step 6: Complete provenance record
        tokens_input = 0
        tokens_output = 0
        if worker_result.model_response:
            tokens_input = worker_result.model_response.tokens_input
            tokens_output = worker_result.model_response.tokens_output

        provenance_tracker.complete_record(
            record_id=provenance_record.id,
            outputs=stored_artifacts,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            confidence=0.8,  # Default confidence
        )

        completed_at = datetime.utcnow()

        result = {
            "task_id": task_id,
            "status": "completed",
            "tier": tier,
            "model": model_config.name,
            "type": task_type,
            "domain": domain,
            "output": {
                "content": content,
                "artifacts": stored_artifacts,
            },
            "provenance": {
                "id": provenance_record.id,
                "model_used": model_config.name,
                "prompt_template": worker_result.prompt_version,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "execution_time_ms": worker_result.execution_time_ms,
            },
        }

        logger.info(
            f"Task {task_id} completed successfully",
            extra={
                "task_id": task_id,
                "status": "completed",
                "tokens_total": tokens_input + tokens_output,
                "execution_time_ms": worker_result.execution_time_ms,
            },
        )

        return result

    except Exception as e:
        logger.error(
            f"Task {task_id} execution failed: {e}",
            extra={"task_id": task_id, "error": str(e)},
        )

        # Update provenance with failure
        provenance_tracker.update_verification(
            record_id=provenance_record.id,
            status="failed",
            details={"error": str(e)},
        )

        return {
            "task_id": task_id,
            "status": "failed",
            "tier": tier,
            "model": model_config.name,
            "type": task_type,
            "domain": domain,
            "error": str(e),
            "provenance": {
                "id": provenance_record.id,
                "model_used": model_config.name,
                "started_at": started_at.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
            },
        }


def _get_worker_for_domain(domain: str, tier: Optional[str] = None):
    """
    Get the appropriate worker for a domain.

    Args:
        domain: Task domain (code-general, documentation, etc.)
        tier: Optional tier override

    Returns:
        Configured worker instance
    """
    from src.workers.code_general import CodeGeneralWorker
    from src.workers.code_vision import CodeVisionWorker
    from src.workers.code_embedded import CodeEmbeddedWorker
    from src.workers.documentation import DocumentationWorker
    from src.workers.ui_design import UIDesignWorker

    worker_map = {
        "code-general": CodeGeneralWorker,
        "code-vision": CodeVisionWorker,
        "code-embedded": CodeEmbeddedWorker,
        "documentation": DocumentationWorker,
        "ui-design": UIDesignWorker,
    }

    worker_class = worker_map.get(domain, CodeGeneralWorker)
    return worker_class(model_tier=tier)


# Task for decomposition
@app.task(base=DATSTask, bind=True, name="src.queue.tasks.decompose_task")
def decompose_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
    """
    Decompose a complex task into subtasks.

    Uses the decomposer agent to break down work.

    Args:
        task_data: Parent task configuration

    Returns:
        Dictionary with subtasks
    """
    from src.agents.decomposer import Decomposer

    # Run async decomposition in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _decompose():
        decomposer = Decomposer()
        try:
            result = await decomposer.decompose_recursive(task_data)
            return {
                "parent_task_id": task_data.get("id"),
                "subtasks": result,
                "status": "decomposed",
            }
        finally:
            await decomposer.close()

    return loop.run_until_complete(_decompose())


# Task for validation
@app.task(base=DATSTask, bind=True, name="src.queue.tasks.validate_task")
def validate_task(
    self,
    task_id: str,
    result: dict[str, Any],
    qa_profile: str = "consensus",
) -> dict[str, Any]:
    """
    Validate task output using QA agent.

    Args:
        task_id: ID of task to validate
        result: Task execution result
        qa_profile: QA profile to use

    Returns:
        Validation result
    """
    from src.agents.qa_reviewer import QAReviewer

    # Run async validation in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _validate():
        reviewer = QAReviewer()
        try:
            # Prepare task data for QA
            task_data = {
                "id": task_id,
                "output": result.get("output", {}),
                "domain": result.get("domain", "code-general"),
                "qa_profile": qa_profile,
            }

            review_result = await reviewer.review(task_data)

            # Update provenance with verification status
            if review_result.get("status") == "approved":
                provenance_tracker = get_provenance_tracker()
                provenance_id = result.get("provenance", {}).get("id")
                if provenance_id:
                    provenance_tracker.update_verification(
                        record_id=provenance_id,
                        status="verified",
                        details=review_result,
                    )

                # Trigger embedding for approved outputs
                _trigger_embedding.delay(task_id, result)

            return {
                "task_id": task_id,
                "validation_status": review_result.get("status", "pending"),
                "qa_profile": qa_profile,
                "issues": review_result.get("issues", []),
                "score": review_result.get("score", 0.0),
            }
        finally:
            await reviewer.close()

    return loop.run_until_complete(_validate())


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.trigger_embedding")
def _trigger_embedding(self, task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """
    Trigger embedding generation for approved task output.

    Args:
        task_id: Task ID
        result: Task execution result

    Returns:
        Embedding status
    """
    from src.rag.embedding_trigger import EmbeddingTrigger

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _embed():
        trigger = EmbeddingTrigger()
        try:
            output = result.get("output", {})
            content = output.get("content", "")
            artifacts = output.get("artifacts", [])

            # Embed main content
            if content:
                await trigger.embed_text(
                    content=content,
                    task_id=task_id,
                    doc_type="output",
                    domain=result.get("domain", "code-general"),
                )

            # Embed artifacts
            for artifact in artifacts:
                artifact_content = artifact.get("content", "")
                if artifact_content:
                    await trigger.embed_text(
                        content=artifact_content,
                        task_id=task_id,
                        doc_type="output",
                        domain=result.get("domain", "code-general"),
                    )

            return {
                "task_id": task_id,
                "status": "embedded",
                "artifacts_count": len(artifacts),
            }
        finally:
            await trigger.close()

    return loop.run_until_complete(_embed())


# Task for merging results
@app.task(base=DATSTask, bind=True, name="src.queue.tasks.merge_results")
def merge_results(
    self,
    parent_task_id: str,
    subtask_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge results from subtasks.

    Uses merge coordinator agent.

    Args:
        parent_task_id: ID of parent task
        subtask_results: Results from subtasks

    Returns:
        Merged result
    """
    from src.agents.merge_coordinator import MergeCoordinator

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _merge():
        coordinator = MergeCoordinator()
        try:
            merge_data = {
                "id": parent_task_id,
                "subtask_results": subtask_results,
            }

            result = await coordinator.merge(merge_data)

            return {
                "parent_task_id": parent_task_id,
                "merged_output": result.get("merged_output", {}),
                "status": "merged",
                "conflicts": result.get("conflicts", []),
            }
        finally:
            await coordinator.close()

    return loop.run_until_complete(_merge())


# ===== CASCADE FAILURE HANDLING TASKS =====


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.propagate_taint")
def propagate_taint(
    self,
    artifact_id: str,
    reason: str,
    source_id: str = "",
    cascade_id: str = "",
) -> dict[str, Any]:
    """
    Propagate taint through the provenance graph.

    Marks the source artifact as tainted and all dependents as suspect.

    Args:
        artifact_id: Artifact to taint
        reason: Reason for tainting
        source_id: ID of what caused the taint
        cascade_id: ID grouping related events

    Returns:
        Propagation result
    """
    from src.cascade.taint import TaintPropagator
    from src.rag.embedding_trigger import invalidate_embeddings_sync

    logger.info(f"Starting taint propagation for artifact {artifact_id}")

    provenance_tracker = get_provenance_tracker()
    propagator = TaintPropagator(provenance_tracker)

    result = propagator.taint_artifact(
        artifact_id=artifact_id,
        reason=reason,
        source_id=source_id,
        cascade_id=cascade_id,
    )

    # Invalidate embeddings for tainted/suspect artifacts
    if result.provenance_ids_to_invalidate:
        try:
            removed = invalidate_embeddings_sync(result.provenance_ids_to_invalidate)
            logger.info(f"Invalidated {removed} embeddings during taint propagation")
        except Exception as e:
            logger.error(f"Failed to invalidate embeddings: {e}")
            result.errors.append(f"Embedding invalidation failed: {e}")

    # Queue revalidation tasks for suspect artifacts
    if result.revalidation_artifact_ids:
        for suspect_id in result.revalidation_artifact_ids:
            queue_revalidation.delay(
                artifact_id=suspect_id,
                taint_source_id=artifact_id,
                taint_reason=reason,
                cascade_id=result.cascade_id,
            )
        result.revalidation_queued = len(result.revalidation_artifact_ids)

    logger.info(
        f"Taint propagation complete: {result.tainted_count} tainted, "
        f"{result.suspect_count} suspect, {result.revalidation_queued} revalidations queued"
    )

    return result.to_dict()


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.queue_revalidation")
def queue_revalidation(
    self,
    artifact_id: str,
    taint_source_id: str,
    taint_reason: str,
    cascade_id: str = "",
    cascade_depth: int = 1,
) -> dict[str, Any]:
    """
    Queue a revalidation task for a suspect artifact.

    Args:
        artifact_id: Suspect artifact to revalidate
        taint_source_id: What tainted artifact caused this
        taint_reason: Why the source was tainted
        cascade_id: Cascade grouping ID
        cascade_depth: Current depth in cascade

    Returns:
        Queued task info
    """
    from src.cascade.revalidation import RevalidationTask, RevalidationQueue
    from src.config.settings import get_settings

    settings = get_settings()
    provenance_tracker = get_provenance_tracker()

    # Get provenance record
    record = provenance_tracker.get_producer(artifact_id)
    if not record:
        return {"error": f"No provenance record for artifact {artifact_id}"}

    # Create revalidation task
    task = RevalidationTask(
        suspect_artifact_id=artifact_id,
        suspect_provenance_id=record.id,
        project_id=record.project_id,
        taint_source_artifact_id=taint_source_id,
        taint_reason=taint_reason,
        cascade_id=cascade_id,
        cascade_depth=cascade_depth,
    )

    # Add to queue
    queue = RevalidationQueue(
        storage_path=getattr(settings, "provenance_path", "data/provenance"),
    )
    task_id = queue.add(task)

    logger.info(f"Queued revalidation task {task_id} for artifact {artifact_id}")

    return {
        "task_id": task_id,
        "artifact_id": artifact_id,
        "cascade_id": cascade_id,
        "cascade_depth": cascade_depth,
        "status": "queued",
    }


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.process_revalidation_batch")
def process_revalidation_batch(
    self,
    max_batch: int = 10,
) -> dict[str, Any]:
    """
    Process a batch of pending revalidation tasks.

    This should be called periodically by Celery beat.

    Args:
        max_batch: Maximum tasks to process

    Returns:
        Processing statistics
    """
    from src.cascade.revalidation import (
        RevalidationQueue,
        RevalidationEvaluator,
        process_revalidation_queue,
    )
    from src.cascade.taint import TaintPropagator
    from src.config.settings import get_settings

    settings = get_settings()
    provenance_tracker = get_provenance_tracker()
    work_product_store = get_work_product_store()

    # Initialize components
    queue = RevalidationQueue(
        storage_path=getattr(settings, "provenance_path", "data/provenance"),
    )
    
    evaluator = RevalidationEvaluator(
        provenance_tracker=provenance_tracker,
        work_product_store=work_product_store,
        model_endpoint=getattr(settings, "vllm_endpoint", "http://192.168.1.11:8000/v1"),
        model_name="openai/gpt-oss-20b",
    )
    
    propagator = TaintPropagator(provenance_tracker)

    # Run async processing
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _process():
        try:
            return await process_revalidation_queue(
                queue=queue,
                evaluator=evaluator,
                taint_propagator=propagator,
                max_batch=max_batch,
            )
        finally:
            await evaluator.close()

    result = loop.run_until_complete(_process())

    # Check if thresholds exceeded and trigger rollback check
    if queue.exceeds_thresholds():
        logger.warning("Revalidation queue exceeds thresholds, considering rollback")
        # Could trigger rollback assessment here

    return result


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.detect_cascade")
def detect_cascade(
    self,
    artifact_id: str,
    trigger_type: str,
    trigger_details: dict[str, Any],
) -> dict[str, Any]:
    """
    Detect and assess a potential cascade scenario.

    Args:
        artifact_id: Artifact that triggered detection
        trigger_type: Type of trigger (qa_failure, human_rejection, etc.)
        trigger_details: Details about the trigger

    Returns:
        Cascade scenario if detected
    """
    from src.cascade.detector import CascadeDetector, CascadeTrigger

    provenance_tracker = get_provenance_tracker()
    detector = CascadeDetector(provenance_tracker)

    scenario = None
    
    if trigger_type == "qa_failure":
        scenario = detector.detect_from_qa_failure(
            task_id=trigger_details.get("task_id", ""),
            artifact_id=artifact_id,
            qa_result=trigger_details.get("qa_result", {}),
        )
    elif trigger_type == "human_rejection":
        scenario = detector.detect_from_human_rejection(
            task_id=trigger_details.get("task_id", ""),
            artifact_id=artifact_id,
            rejection_reason=trigger_details.get("reason", ""),
            reviewer_id=trigger_details.get("reviewer_id", ""),
        )
    elif trigger_type == "manual_taint":
        scenario = detector.detect_from_manual_taint(
            artifact_id=artifact_id,
            reason=trigger_details.get("reason", "Manual taint request"),
            requested_by=trigger_details.get("requested_by", ""),
        )
    elif trigger_type == "security_issue":
        scenario = detector.detect_from_security_issue(
            artifact_id=artifact_id,
            vulnerability=trigger_details.get("vulnerability", ""),
            severity=trigger_details.get("severity", "high"),
            cve_id=trigger_details.get("cve_id", ""),
        )

    if not scenario:
        return {"detected": False, "artifact_id": artifact_id}

    # If cascade detected, trigger propagation based on recommendation
    if scenario.recommended_action == "propagate":
        propagate_taint.delay(
            artifact_id=artifact_id,
            reason=scenario.reason,
            cascade_id=scenario.id,
        )
    elif scenario.recommended_action == "rollback":
        # Recommend rollback but don't auto-trigger
        logger.warning(f"Cascade scenario {scenario.id} recommends rollback")

    return {
        "detected": True,
        "scenario": scenario.to_dict(),
        "should_pause": detector.should_pause_execution(scenario),
        "affected_tasks": detector.get_affected_tasks(scenario),
    }


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.execute_rollback")
def execute_rollback(
    self,
    checkpoint_id: str,
    trigger: str = "human_request",
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Execute a rollback to a checkpoint.

    Args:
        checkpoint_id: Checkpoint to rollback to
        trigger: What triggered the rollback
        dry_run: If True, only calculate impact

    Returns:
        Rollback result
    """
    from src.cascade.rollback import RollbackManager, RollbackTrigger
    from src.config.settings import get_settings

    settings = get_settings()
    provenance_tracker = get_provenance_tracker()

    rollback_manager = RollbackManager(
        provenance_tracker=provenance_tracker,
        storage_path=getattr(settings, "provenance_path", "data/provenance"),
    )

    trigger_enum = RollbackTrigger(trigger)
    result = rollback_manager.rollback_to_checkpoint(
        checkpoint_id=checkpoint_id,
        trigger=trigger_enum,
        dry_run=dry_run,
    )

    if result.success and not dry_run:
        # Queue re-execution of affected tasks
        for task_id in result.details.get("tasks_to_requeue", []):
            logger.info(f"Would requeue task {task_id} after rollback")
            # In practice, you would look up and requeue the original task

    return result.to_dict()


@app.task(base=DATSTask, bind=True, name="src.queue.tasks.create_checkpoint")
def create_checkpoint(
    self,
    project_id: str,
    description: str = "",
    trigger: str = "manual",
) -> dict[str, Any]:
    """
    Create a checkpoint for a project.

    Args:
        project_id: Project to checkpoint
        description: Optional description
        trigger: What triggered the checkpoint

    Returns:
        Created checkpoint info
    """
    from src.cascade.rollback import RollbackManager
    from src.config.settings import get_settings

    settings = get_settings()
    provenance_tracker = get_provenance_tracker()

    rollback_manager = RollbackManager(
        provenance_tracker=provenance_tracker,
        storage_path=getattr(settings, "provenance_path", "data/provenance"),
    )

    checkpoint = rollback_manager.create_checkpoint(
        project_id=project_id,
        description=description,
        trigger=trigger,
    )

    return checkpoint.to_dict()


# Task for pipeline processing
@app.task(base=DATSTask, bind=True, name="src.queue.tasks.process_pipeline")
def process_pipeline(
    self,
    user_request: str,
    project_id: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Process a user request through the full agent pipeline.

    This is the main entry point for new requests.

    Args:
        user_request: Natural language request
        project_id: Optional project ID
        context: Optional additional context

    Returns:
        Pipeline processing result
    """
    from src.pipeline.orchestrator import AgentPipeline

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def _process():
        pipeline = AgentPipeline(use_celery=True)
        try:
            result = await pipeline.process_request(
                user_request=user_request,
                project_id=project_id,
                context=context,
            )

            return {
                "task_id": result.task_id,
                "project_id": result.project_id,
                "status": result.status.value,
                "mode": result.mode.value,
                "tier": result.tier,
                "output": result.output,
                "artifacts": result.artifacts,
                "subtasks": [
                    {
                        "id": s.subtask_id,
                        "status": s.status.value,
                        "tier": s.tier,
                    }
                    for s in result.subtasks
                ],
                "provenance_ids": result.provenance_ids,
                "error": result.error,
            }
        finally:
            await pipeline.close()

    return loop.run_until_complete(_process())
