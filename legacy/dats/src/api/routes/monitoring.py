"""
Monitoring and health API routes.

Handles health checks, worker status, queue depths, and failure information.
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_provenance_tracker, verify_api_key
from src.api.schemas import (
    FailureInfo,
    FailuresResponse,
    HealthResponse,
    QueueInfo,
    QueuesResponse,
    WorkerInfo,
    WorkersResponse,
)
from src.api.app import API_VERSION
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and component status",
)
async def health_check():
    """Check API health and component status."""
    settings = get_settings()
    
    # Check component health
    components = {}
    
    # Check Redis (result backend)
    try:
        import redis
        r = redis.from_url(settings.result_backend_url)
        r.ping()
        components["redis"] = "healthy"
    except Exception:
        components["redis"] = "unhealthy"
    
    # Check RabbitMQ (broker)
    try:
        # Simple connectivity check
        components["rabbitmq"] = "healthy"  # TODO: Add actual check
    except Exception:
        components["rabbitmq"] = "unhealthy"
    
    # Check provenance storage
    try:
        provenance_tracker = get_provenance_tracker()
        components["provenance"] = "healthy"
    except Exception:
        components["provenance"] = "unhealthy"
    
    # Determine overall status
    all_healthy = all(s == "healthy" for s in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=API_VERSION,
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get(
    "/workers",
    response_model=WorkersResponse,
    summary="Get worker status",
    description="Get status of all Celery workers",
)
async def get_workers(
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get status of all Celery workers."""
    workers = []
    
    try:
        from src.queue.celery_app import app as celery_app
        
        # Get active workers
        inspect = celery_app.control.inspect()
        active_workers = inspect.active() or {}
        stats = inspect.stats() or {}
        
        for worker_name, tasks in active_workers.items():
            # Get worker stats
            worker_stats = stats.get(worker_name, {})
            
            # Determine tier from worker name (e.g., celery@tier-small-1)
            tier = "unknown"
            if "tiny" in worker_name:
                tier = "tiny"
            elif "small" in worker_name:
                tier = "small"
            elif "large" in worker_name:
                tier = "large"
            elif "frontier" in worker_name:
                tier = "frontier"
            
            workers.append(WorkerInfo(
                worker_id=worker_name,
                status="active" if tasks else "idle",
                tier=tier,
                current_task=tasks[0]["id"] if tasks else None,
                tasks_completed=worker_stats.get("total", {}).get("completed", 0),
                last_heartbeat=datetime.utcnow(),  # TODO: Get actual heartbeat
            ))
            
    except Exception as e:
        logger.warning(f"Failed to get worker status: {e}")
    
    return WorkersResponse(
        workers=workers,
        total_active=len([w for w in workers if w.status == "active"]),
    )


@router.get(
    "/queues",
    response_model=QueuesResponse,
    summary="Get queue depths",
    description="Get current depth of all task queues",
)
async def get_queues(
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get current depth of all task queues."""
    queues = []
    
    # Define expected queues
    queue_names = ["celery", "tiny", "small", "large", "frontier"]
    
    try:
        from src.queue.celery_app import app as celery_app
        
        inspect = celery_app.control.inspect()
        active_queues = inspect.active_queues() or {}
        
        # Count consumers per queue
        consumers_per_queue = {}
        for worker_queues in active_queues.values():
            for q in worker_queues:
                name = q.get("name", "unknown")
                consumers_per_queue[name] = consumers_per_queue.get(name, 0) + 1
        
        # Get queue lengths from broker (RabbitMQ)
        # TODO: Implement actual queue depth check
        for name in queue_names:
            queues.append(QueueInfo(
                name=name,
                depth=0,  # TODO: Get actual depth
                consumers=consumers_per_queue.get(name, 0),
            ))
            
    except Exception as e:
        logger.warning(f"Failed to get queue info: {e}")
        # Return placeholder data
        for name in queue_names:
            queues.append(QueueInfo(
                name=name,
                depth=0,
                consumers=0,
            ))
    
    return QueuesResponse(queues=queues)


@router.get(
    "/failures",
    response_model=FailuresResponse,
    summary="Get recent failures",
    description="Get list of recent task failures",
)
async def get_failures(
    last: int = Query(default=10, ge=1, le=100),
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Get list of recent task failures."""
    provenance_tracker = get_provenance_tracker()
    
    # Get all records and filter for failures
    all_records = provenance_tracker.get_all_records()
    
    failures = []
    for record in all_records:
        if record.verification_status == "failed" or record.is_tainted():
            failures.append(FailureInfo(
                task_id=record.task_id,
                error=record.taint.tainted_reason if record.is_tainted() else "Verification failed",
                failed_at=record.completed_at or record.created_at or datetime.utcnow(),
                tier=record.execution.model if hasattr(record, 'execution') else None,
            ))
    
    # Sort by date (most recent first) and limit
    failures.sort(key=lambda x: x.failed_at, reverse=True)
    failures = failures[:last]
    
    return FailuresResponse(
        failures=failures,
        total=len(failures),
    )


@router.get(
    "/metrics",
    summary="Get Prometheus metrics",
    description="Get metrics in Prometheus format",
)
async def get_metrics():
    """Get metrics in Prometheus format."""
    provenance_tracker = get_provenance_tracker()
    
    # Collect metrics
    all_records = provenance_tracker.get_all_records()
    
    total_tasks = len(set(r.task_id for r in all_records))
    completed_records = sum(1 for r in all_records if r.verification_status == "passed")
    failed_records = sum(1 for r in all_records if r.verification_status == "failed")
    pending_records = sum(1 for r in all_records if r.verification_status == "pending")
    tainted_records = sum(1 for r in all_records if r.is_tainted())
    
    # Format as Prometheus text format
    metrics = []
    metrics.append(f"# HELP dats_tasks_total Total number of tasks")
    metrics.append(f"# TYPE dats_tasks_total counter")
    metrics.append(f"dats_tasks_total {total_tasks}")
    
    metrics.append(f"# HELP dats_records_total Total provenance records by status")
    metrics.append(f"# TYPE dats_records_total counter")
    metrics.append(f'dats_records_total{{status="completed"}} {completed_records}')
    metrics.append(f'dats_records_total{{status="failed"}} {failed_records}')
    metrics.append(f'dats_records_total{{status="pending"}} {pending_records}')
    metrics.append(f'dats_records_total{{status="tainted"}} {tainted_records}')
    
    return "\n".join(metrics)