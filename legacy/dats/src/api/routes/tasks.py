"""
Task submission and status API routes.

Handles task lifecycle: submit, status, tree, stream, cancel.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import (
    get_pipeline,
    get_provenance_tracker,
    verify_api_key,
)
from src.api.schemas import (
    CurrentActivity,
    HumanActionNeeded,
    Links,
    ReviewType,
    SubtaskProgress,
    TaskError,
    TaskListItem,
    TaskListResponse,
    TaskMode,
    TaskOutput,
    TaskStatus,
    TaskStatusResponse,
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskTreeNode,
    TaskTreeResponse,
)
from src.config.settings import get_settings
from src.pipeline.orchestrator import AgentPipeline, TaskStatus as PipelineStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def _map_pipeline_status(status: PipelineStatus) -> TaskStatus:
    """Map pipeline status to API status."""
    mapping = {
        PipelineStatus.PENDING: TaskStatus.QUEUED,
        PipelineStatus.ANALYZING: TaskStatus.DECOMPOSING,
        PipelineStatus.DECOMPOSING: TaskStatus.DECOMPOSING,
        PipelineStatus.ESTIMATING: TaskStatus.DECOMPOSING,
        PipelineStatus.QUEUED: TaskStatus.QUEUED,
        PipelineStatus.EXECUTING: TaskStatus.IN_PROGRESS,
        PipelineStatus.VALIDATING: TaskStatus.VALIDATING,
        PipelineStatus.COMPLETED: TaskStatus.COMPLETED,
        PipelineStatus.FAILED: TaskStatus.FAILED,
    }
    return mapping.get(status, TaskStatus.QUEUED)


@router.post(
    "/tasks",
    response_model=TaskSubmitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new task",
    description="Submit a new task/request for processing",
)
async def submit_task(
    request: TaskSubmitRequest,
    api_key: Optional[str] = Depends(verify_api_key),
    pipeline: AgentPipeline = Depends(get_pipeline),
):
    """
    Submit a new task for processing.
    
    The task will be analyzed, potentially decomposed, and queued for execution.
    """
    settings = get_settings()
    project_id = request.project_id or settings.default_project
    
    # Build context from request
    context = {}
    if request.constitution_override:
        context["constitution"] = request.constitution_override
    if request.metadata:
        context["metadata"] = request.metadata.model_dump()
    context["priority"] = request.priority.value
    context["mode"] = request.mode.value
    
    try:
        # Process through pipeline
        result = await pipeline.process_request(
            user_request=request.description,
            project_id=project_id,
            context=context,
        )
        
        # Build response
        base_url = settings.api_url
        return TaskSubmitResponse(
            task_id=result.task_id,
            status=_map_pipeline_status(result.status),
            mode=TaskMode(request.mode.value),
            created_at=result.started_at or datetime.utcnow(),
            estimated_completion=None,  # TODO: Add estimation
            links=Links(
                self=f"{base_url}/api/v1/tasks/{result.task_id}",
                status=f"{base_url}/api/v1/tasks/{result.task_id}/status",
                stream=f"{base_url}/api/v1/tasks/{result.task_id}/stream",
            ),
        )
        
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SUBMISSION_FAILED",
                "message": f"Failed to submit task: {str(e)}",
            },
        )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task details",
    description="Get detailed information about a task",
)
async def get_task(
    task_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get detailed information about a task."""
    provenance_tracker = get_provenance_tracker()
    
    # Get provenance records for this task
    records = provenance_tracker.get_by_task(task_id)
    
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TASK_NOT_FOUND",
                "message": f"Task with ID {task_id} not found",
            },
        )
    
    # Aggregate status from records
    # For now, use the first record as the main task
    main_record = records[0]
    
    # Count subtask statuses
    completed = sum(1 for r in records if r.verification_status == "passed")
    failed = sum(1 for r in records if r.verification_status == "failed")
    pending = sum(1 for r in records if r.verification_status == "pending")
    
    # Determine overall status
    if failed > 0:
        task_status = TaskStatus.FAILED
    elif completed == len(records):
        task_status = TaskStatus.COMPLETED
    elif pending > 0:
        task_status = TaskStatus.IN_PROGRESS
    else:
        task_status = TaskStatus.QUEUED
    
    # Build outputs
    outputs = []
    for record in records:
        for output in record.outputs:
            outputs.append(TaskOutput(
                artifact_id=output.artifact_id,
                type=output.type.value if hasattr(output.type, 'value') else str(output.type),
                location=output.location,
            ))
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_status,
        mode=TaskMode.AUTONOMOUS,  # TODO: Store mode in provenance
        progress=SubtaskProgress(
            total_subtasks=len(records),
            completed_subtasks=completed,
            failed_subtasks=failed,
            pending_review=0,  # TODO: Track pending reviews
        ),
        current_activity=None,  # TODO: Track current activity
        outputs=outputs,
        errors=[],
        human_actions_needed=[],
        created_at=main_record.created_at or datetime.utcnow(),
        updated_at=main_record.completed_at,
    )


@router.get(
    "/tasks/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="Get task status",
    description="Get current status of a task",
)
async def get_task_status(
    task_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get current status of a task (same as get_task)."""
    return await get_task(task_id, api_key)


@router.get(
    "/tasks/{task_id}/tree",
    response_model=TaskTreeResponse,
    summary="Get task decomposition tree",
    description="Get the decomposition tree for a task",
)
async def get_task_tree(
    task_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get the decomposition tree for a task."""
    provenance_tracker = get_provenance_tracker()
    
    # Get task tree from provenance graph
    tree_data = provenance_tracker.graph.get_task_tree(task_id)
    
    if not tree_data.get("nodes"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "TASK_NOT_FOUND",
                "message": f"Task tree for {task_id} not found",
            },
        )
    
    # Build tree structure from flat nodes
    nodes_by_id = {}
    for node_data in tree_data.get("nodes", []):
        node = TaskTreeNode(
            task_id=node_data.get("task_id", ""),
            description=node_data.get("task_description", ""),
            status=TaskStatus.COMPLETED,  # TODO: Map from provenance
            tier=node_data.get("model_used"),
            children=[],
        )
        nodes_by_id[node.task_id] = node
    
    # Find root (task with given ID)
    root = nodes_by_id.get(task_id)
    if not root:
        # If no exact match, use first node as root
        root = TaskTreeNode(
            task_id=task_id,
            description="Root task",
            status=TaskStatus.COMPLETED,
            children=list(nodes_by_id.values()),
        )
    
    return TaskTreeResponse(root=root)


@router.post(
    "/tasks/{task_id}/cancel",
    summary="Cancel a task",
    description="Cancel a running or queued task",
)
async def cancel_task(
    task_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Cancel a running or queued task."""
    # TODO: Implement task cancellation via Celery
    # For now, just acknowledge the request
    logger.info(f"Cancel requested for task {task_id}")
    
    return {
        "task_id": task_id,
        "status": "cancellation_requested",
        "message": "Task cancellation has been requested",
    }


@router.get(
    "/projects/{project_id}/tasks",
    response_model=TaskListResponse,
    summary="List tasks in project",
    description="List all tasks in a project",
)
async def list_project_tasks(
    project_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[TaskStatus] = Query(default=None, alias="status"),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """List all tasks in a project."""
    provenance_tracker = get_provenance_tracker()
    
    # Get all records for project
    records = provenance_tracker.get_by_project(project_id)
    
    # Group by task_id to get unique tasks
    tasks_by_id: dict[str, Any] = {}
    for record in records:
        if record.task_id not in tasks_by_id:
            tasks_by_id[record.task_id] = {
                "task_id": record.task_id,
                "description": "",  # TODO: Store task description
                "status": TaskStatus.COMPLETED,
                "mode": TaskMode.AUTONOMOUS,
                "created_at": record.created_at or datetime.utcnow(),
            }
    
    # Convert to list and apply pagination
    all_tasks = list(tasks_by_id.values())
    total = len(all_tasks)
    
    start = (page - 1) * page_size
    end = start + page_size
    page_tasks = all_tasks[start:end]
    
    return TaskListResponse(
        tasks=[TaskListItem(**t) for t in page_tasks],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/tasks/{task_id}/stream",
    summary="Stream task updates",
    description="Server-sent events stream for task status updates",
)
async def stream_task_updates(
    request: Request,
    task_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """
    Stream task status updates via Server-Sent Events.
    
    The client should listen for events like:
    - status_update: Task status changed
    - subtask_complete: A subtask finished
    - review_needed: Human review is required
    - task_complete: Task finished successfully
    - task_failed: Task failed
    """
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events for task updates."""
        provenance_tracker = get_provenance_tracker()
        last_status = None
        last_check = datetime.utcnow()
        
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
            
            # Get current task status
            records = provenance_tracker.get_by_task(task_id)
            
            if not records:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "type": "error",
                        "task_id": task_id,
                        "message": "Task not found",
                    }),
                }
                break
            
            # Calculate current status
            completed = sum(1 for r in records if r.verification_status == "passed")
            failed = sum(1 for r in records if r.verification_status == "failed")
            
            if failed > 0:
                current_status = "failed"
            elif completed == len(records):
                current_status = "completed"
            else:
                current_status = "in_progress"
            
            # Send update if status changed
            if current_status != last_status:
                event_type = f"task_{current_status}" if current_status in ("completed", "failed") else "status_update"
                
                yield {
                    "event": event_type,
                    "data": json.dumps({
                        "type": event_type,
                        "task_id": task_id,
                        "status": current_status,
                        "progress": {
                            "total": len(records),
                            "completed": completed,
                            "failed": failed,
                        },
                    }),
                }
                
                last_status = current_status
                
                # Stop streaming if task is complete or failed
                if current_status in ("completed", "failed"):
                    break
            
            # Wait before next check
            await asyncio.sleep(2)
    
    return EventSourceResponse(event_generator())