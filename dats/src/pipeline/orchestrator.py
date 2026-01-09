"""
Agent Pipeline Orchestrator for DATS.

Coordinates the full agent pipeline:
Coordinator → Decomposer → Complexity Estimator → Worker routing.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from opentelemetry.trace import Status, StatusCode

from src.agents.coordinator import Coordinator
from src.agents.decomposer import Decomposer
from src.agents.complexity_estimator import ComplexityEstimator
from src.config.routing import get_routing_config
from src.storage.provenance import ProvenanceTracker
from src.telemetry.config import get_tracer
from src.telemetry.context import add_task_context

logger = logging.getLogger(__name__)


class TaskMode(Enum):
    """Mode of operation for a task."""

    NEW_PROJECT = "new_project"
    MODIFY = "modify"
    FIX_BUG = "fix_bug"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    UNKNOWN = "unknown"


class TaskStatus(Enum):
    """Status of a task in the pipeline."""

    PENDING = "pending"
    ANALYZING = "analyzing"
    DECOMPOSING = "decomposing"
    ESTIMATING = "estimating"
    QUEUED = "queued"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubtaskResult:
    """Result of a subtask execution."""

    subtask_id: str
    status: TaskStatus
    tier: str
    output: Optional[str] = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    provenance_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result from pipeline processing."""

    task_id: str
    project_id: str
    status: TaskStatus
    mode: TaskMode
    tier: Optional[str] = None
    output: Optional[str] = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    subtasks: list[SubtaskResult] = field(default_factory=list)
    provenance_ids: list[str] = field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def execution_time_ms(self) -> Optional[int]:
        """Calculate execution time in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class AgentPipeline:
    """
    Main pipeline orchestrator for DATS.

    Handles the full flow from user request to task execution:
    1. Coordinator analyzes request and determines mode
    2. Decomposer breaks down complex tasks
    3. Complexity Estimator routes each subtask to appropriate tier
    4. Tasks are queued to Celery for execution
    """

    def __init__(
        self,
        provenance_path: Optional[str] = None,
        use_celery: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            provenance_path: Path for provenance storage
            use_celery: Whether to use Celery for task queuing
        """
        self._coordinator = Coordinator()
        self._decomposer = Decomposer()
        self._complexity_estimator = ComplexityEstimator()
        self._provenance = ProvenanceTracker(storage_path=provenance_path)
        self._routing_config = get_routing_config()
        self._use_celery = use_celery

    async def process_request(
        self,
        user_request: str,
        project_id: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Process a user request through the full pipeline.

        Args:
            user_request: Natural language request from user
            project_id: Optional project identifier
            context: Optional additional context

        Returns:
            PipelineResult with execution status and outputs
        """
        started_at = datetime.utcnow()
        task_id = str(uuid.uuid4())
        project_id = project_id or str(uuid.uuid4())
        
        tracer = get_tracer("dats.pipeline")

        logger.info(
            f"Processing request: {user_request[:100]}...",
            extra={"task_id": task_id, "project_id": project_id},
        )

        # Create the root span for the entire pipeline
        with tracer.start_as_current_span(
            "pipeline.process_request",
            attributes={
                "dats.task.id": task_id,
                "dats.project.id": project_id,
                "dats.request.length": len(user_request),
            },
        ) as pipeline_span:
            try:
                # Step 1: Coordinator analyzes the request
                with tracer.start_as_current_span(
                    "pipeline.coordinate",
                    attributes={"dats.task.id": task_id},
                ) as coord_span:
                    coordination_result = await self._coordinate(
                        user_request=user_request,
                        task_id=task_id,
                        project_id=project_id,
                        context=context,
                    )
                    
                    if coordination_result.get("success"):
                        coord_span.set_attribute("dats.coordination.mode", coordination_result.get("mode", "unknown"))
                        coord_span.set_attribute("dats.coordination.needs_decomposition", coordination_result.get("needs_decomposition", False))
                        coord_span.set_status(Status(StatusCode.OK))
                    else:
                        coord_span.set_status(Status(StatusCode.ERROR, coordination_result.get("error", "Unknown error")))

                if not coordination_result.get("success", False):
                    pipeline_span.set_status(Status(StatusCode.ERROR, "Coordination failed"))
                    return PipelineResult(
                        task_id=task_id,
                        project_id=project_id,
                        status=TaskStatus.FAILED,
                        mode=TaskMode.UNKNOWN,
                        error=coordination_result.get("error", "Coordination failed"),
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                    )

                mode = self._parse_mode(coordination_result.get("mode", "unknown"))
                task_data = coordination_result.get("task_data", {})
                task_data["id"] = task_id
                task_data["project_id"] = project_id
                task_data["description"] = user_request
                
                pipeline_span.set_attribute("dats.pipeline.mode", mode.value)

                # Step 2: Check if decomposition is needed
                needs_decomposition = coordination_result.get("needs_decomposition", False)

                if needs_decomposition:
                    # Step 3: Decompose the task
                    with tracer.start_as_current_span(
                        "pipeline.decompose",
                        attributes={"dats.task.id": task_id},
                    ) as decompose_span:
                        subtasks = await self._decompose_recursive(task_data)
                        decompose_span.set_attribute("dats.decomposition.subtask_count", len(subtasks) if subtasks else 0)
                        
                        if subtasks:
                            decompose_span.set_status(Status(StatusCode.OK))
                        else:
                            decompose_span.set_status(Status(StatusCode.ERROR, "No subtasks produced"))

                    if not subtasks:
                        pipeline_span.set_status(Status(StatusCode.ERROR, "Decomposition failed"))
                        return PipelineResult(
                            task_id=task_id,
                            project_id=project_id,
                            status=TaskStatus.FAILED,
                            mode=mode,
                            error="Decomposition produced no subtasks",
                            started_at=started_at,
                            completed_at=datetime.utcnow(),
                        )

                    # Step 4: Estimate complexity and queue each subtask
                    subtask_results = []
                    with tracer.start_as_current_span(
                        "pipeline.queue_subtasks",
                        attributes={
                            "dats.task.id": task_id,
                            "dats.subtask.count": len(subtasks),
                        },
                    ):
                        for subtask in subtasks:
                            result = await self._estimate_and_queue(subtask)
                            subtask_results.append(result)

                    pipeline_span.set_attribute("dats.pipeline.subtasks_queued", len(subtask_results))
                    pipeline_span.set_status(Status(StatusCode.OK))
                    
                    return PipelineResult(
                        task_id=task_id,
                        project_id=project_id,
                        status=TaskStatus.QUEUED,
                        mode=mode,
                        subtasks=subtask_results,
                        provenance_ids=[r.provenance_id for r in subtask_results if r.provenance_id],
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                    )

                else:
                    # Simple task - estimate and queue directly
                    with tracer.start_as_current_span(
                        "pipeline.estimate_and_queue",
                        attributes={"dats.task.id": task_id},
                    ) as queue_span:
                        result = await self._estimate_and_queue(task_data)
                        queue_span.set_attribute("dats.task.tier", result.tier)
                        queue_span.set_status(Status(StatusCode.OK) if result.status != TaskStatus.FAILED else Status(StatusCode.ERROR))

                    pipeline_span.set_attribute("dats.pipeline.tier", result.tier)
                    pipeline_span.set_status(Status(StatusCode.OK))
                    
                    return PipelineResult(
                        task_id=task_id,
                        project_id=project_id,
                        status=result.status,
                        mode=mode,
                        tier=result.tier,
                        output=result.output,
                        artifacts=result.artifacts,
                        provenance_ids=[result.provenance_id] if result.provenance_id else [],
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                    )

            except Exception as e:
                logger.error(f"Pipeline error: {e}", extra={"task_id": task_id})
                pipeline_span.set_status(Status(StatusCode.ERROR, str(e)))
                pipeline_span.record_exception(e)
                return PipelineResult(
                    task_id=task_id,
                    project_id=project_id,
                    status=TaskStatus.FAILED,
                    mode=TaskMode.UNKNOWN,
                    error=str(e),
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

    async def _coordinate(
        self,
        user_request: str,
        task_id: str,
        project_id: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Run the coordinator to analyze the request.

        Args:
            user_request: User's natural language request
            task_id: Task identifier
            project_id: Project identifier
            context: Optional additional context

        Returns:
            Coordination result with mode and task data
        """
        task_data = {
            "id": task_id,
            "project_id": project_id,
            "description": user_request,
            "inputs": [],
        }

        if context:
            task_data["inputs"].append({
                "type": "context",
                "content": context,
            })

        result = await self._coordinator.analyze_task(task_data)

        if result.get("status") == "analyzed":
            recommendation = result.get("recommendation", {})
            return {
                "success": True,
                "mode": recommendation.get("mode", "unknown"),
                "needs_decomposition": recommendation.get("needs_decomposition", False),
                "task_data": {
                    "domain": recommendation.get("domain", "code-general"),
                    "acceptance_criteria": recommendation.get("acceptance_criteria", ""),
                    "estimated_complexity": recommendation.get("complexity", "medium"),
                    "qa_profile": recommendation.get("qa_profile", "consensus"),
                },
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Unknown coordination error"),
            }

    async def _decompose_recursive(
        self,
        task_data: dict[str, Any],
        max_depth: int = 5,
        current_depth: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Recursively decompose a task until all subtasks are atomic.

        Args:
            task_data: Task to decompose
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            List of atomic subtasks
        """
        if current_depth >= max_depth:
            logger.warning(f"Max decomposition depth reached for task {task_data.get('id')}")
            return [task_data]

        result = await self._decomposer.decompose(task_data)

        if result.get("status") != "decomposed":
            logger.error(f"Decomposition failed: {result.get('error')}")
            return [task_data]

        subtasks = result.get("subtasks", [])

        if not subtasks:
            # No subtasks means this task is already atomic
            return [task_data]

        atomic_tasks = []
        for subtask in subtasks:
            # Add parent task info
            subtask["parent_task_id"] = task_data.get("id")
            subtask["project_id"] = task_data.get("project_id")
            subtask["id"] = subtask.get("id", str(uuid.uuid4()))

            # Check if subtask is atomic
            if self._is_atomic(subtask):
                atomic_tasks.append(subtask)
            else:
                # Recursively decompose
                sub_subtasks = await self._decompose_recursive(
                    subtask,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
                atomic_tasks.extend(sub_subtasks)

        return atomic_tasks

    def _is_atomic(self, subtask: dict[str, Any]) -> bool:
        """
        Check if a subtask is atomic (single worker executable).

        Criteria:
        - Has a single domain
        - Estimated tokens < tier context limit
        - Marked as atomic by decomposer
        - No explicit sub-components

        Args:
            subtask: Subtask to check

        Returns:
            True if subtask is atomic
        """
        # If explicitly marked as atomic
        if subtask.get("is_atomic", False):
            return True

        # If marked as needing further decomposition
        if subtask.get("needs_decomposition", False):
            return False

        # Check complexity estimate
        complexity = subtask.get("complexity", "medium")
        if complexity in ["tiny", "small"]:
            return True

        # Check if it has subtasks defined
        if subtask.get("subtasks"):
            return False

        # Default: assume atomic if we get here
        return True

    async def _estimate_and_queue(
        self,
        task_data: dict[str, Any],
    ) -> SubtaskResult:
        """
        Estimate complexity and queue task to appropriate tier.

        Args:
            task_data: Task to estimate and queue

        Returns:
            SubtaskResult with queuing status
        """
        task_id = task_data.get("id", str(uuid.uuid4()))

        # Step 1: Estimate complexity
        estimation = await self._complexity_estimator.estimate(task_data)

        tier = estimation.get("recommended_tier", "small")
        qa_profile = estimation.get("qa_profile", "consensus")

        # Update task data with estimation
        task_data["routing"] = {
            "tier": tier,
            "estimated_tokens": estimation.get("estimated_tokens", 5000),
            "confidence": estimation.get("confidence", 0.5),
        }
        task_data["qa_profile"] = qa_profile

        # Step 2: Queue the task
        if self._use_celery:
            return await self._queue_to_celery(task_data, tier)
        else:
            return await self._execute_sync(task_data, tier)

    async def _queue_to_celery(
        self,
        task_data: dict[str, Any],
        tier: str,
    ) -> SubtaskResult:
        """
        Queue task to Celery for async execution.

        Args:
            task_data: Task to queue
            tier: Target tier

        Returns:
            SubtaskResult with queued status
        """
        from src.queue.tasks import execute_task

        task_id = task_data.get("id", str(uuid.uuid4()))
        
        # Add trace context to task data for distributed tracing
        task_data = add_task_context(task_data)

        # Queue the task
        try:
            result = execute_task.delay(task_data, tier)

            logger.info(
                f"Task {task_id} queued to tier {tier}",
                extra={"task_id": task_id, "tier": tier, "celery_id": result.id},
            )

            return SubtaskResult(
                subtask_id=task_id,
                status=TaskStatus.QUEUED,
                tier=tier,
            )

        except Exception as e:
            logger.error(f"Failed to queue task {task_id}: {e}")
            return SubtaskResult(
                subtask_id=task_id,
                status=TaskStatus.FAILED,
                tier=tier,
                error=str(e),
            )

    async def _execute_sync(
        self,
        task_data: dict[str, Any],
        tier: str,
    ) -> SubtaskResult:
        """
        Execute task synchronously (for testing without Celery).

        Args:
            task_data: Task to execute
            tier: Target tier

        Returns:
            SubtaskResult with execution result
        """
        from src.queue.tasks import _async_execute

        task_id = task_data.get("id", str(uuid.uuid4()))

        try:
            result = await _async_execute(task_data, tier)

            # Use provenance record created by _async_execute - don't create duplicate!
            # The provenance ID is returned in the result
            provenance_id = result.get("provenance", {}).get("id", "")

            return SubtaskResult(
                subtask_id=task_id,
                status=TaskStatus.COMPLETED,
                tier=tier,
                output=result.get("output", {}).get("content", ""),
                artifacts=result.get("output", {}).get("artifacts", []),
                provenance_id=provenance_id,
            )

        except Exception as e:
            logger.error(f"Sync execution failed for task {task_id}: {e}")
            return SubtaskResult(
                subtask_id=task_id,
                status=TaskStatus.FAILED,
                tier=tier,
                error=str(e),
            )

    def _parse_mode(self, mode_str: str) -> TaskMode:
        """Parse mode string to TaskMode enum."""
        mode_map = {
            "new_project": TaskMode.NEW_PROJECT,
            "modify": TaskMode.MODIFY,
            "fix_bug": TaskMode.FIX_BUG,
            "refactor": TaskMode.REFACTOR,
            "documentation": TaskMode.DOCUMENTATION,
            "testing": TaskMode.TESTING,
        }
        return mode_map.get(mode_str.lower(), TaskMode.UNKNOWN)

    async def close(self):
        """Clean up resources."""
        await self._coordinator.close()
        await self._decomposer.close()
        await self._complexity_estimator.close()


async def process_request(
    user_request: str,
    project_id: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    use_celery: bool = True,
) -> PipelineResult:
    """
    Convenience function to process a request through the pipeline.

    Args:
        user_request: Natural language request
        project_id: Optional project identifier
        context: Optional additional context
        use_celery: Whether to use Celery queuing

    Returns:
        PipelineResult with execution status
    """
    pipeline = AgentPipeline(use_celery=use_celery)
    try:
        return await pipeline.process_request(
            user_request=user_request,
            project_id=project_id,
            context=context,
        )
    finally:
        await pipeline.close()
