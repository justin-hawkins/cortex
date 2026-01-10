"""
Pydantic schemas for DATS API.

Defines request and response models for all API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ===== Enums =====


class TaskMode(str, Enum):
    """Mode of task execution."""

    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"


class TaskStatus(str, Enum):
    """Status of a task."""

    QUEUED = "queued"
    DECOMPOSING = "decomposing"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Priority level for tasks."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ReviewType(str, Enum):
    """Type of human review."""

    QA = "qa"
    ARCHITECTURE = "architecture"
    AMBIGUITY = "ambiguity"
    APPROVAL = "approval"


class ReviewStatus(str, Enum):
    """Status of a review request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


# ===== Common Models =====


class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        default=None, description="Additional error context"
    )
    task_id: Optional[str] = Field(
        default=None, description="Related task ID if applicable"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail


class Links(BaseModel):
    """HATEOAS links for resources."""

    self: str = Field(..., description="Link to this resource")
    status: Optional[str] = Field(default=None, description="Link to status endpoint")
    stream: Optional[str] = Field(default=None, description="Link to SSE stream")


class TaskMetadata(BaseModel):
    """User-provided metadata for tasks."""

    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    requester: Optional[str] = Field(default=None, description="Requester identifier")


# ===== Task Schemas =====


class TaskSubmitRequest(BaseModel):
    """Request to submit a new task."""

    description: str = Field(..., description="The work to be done")
    project_id: Optional[str] = Field(
        default=None, description="Project ID (uses default if not set)"
    )
    mode: TaskMode = Field(default=TaskMode.AUTONOMOUS, description="Execution mode")
    constitution_override: Optional[dict[str, Any]] = Field(
        default=None, description="Optional per-request standards override"
    )
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Priority")
    metadata: Optional[TaskMetadata] = Field(
        default=None, description="Optional user metadata"
    )


class TaskSubmitResponse(BaseModel):
    """Response after submitting a task."""

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current status")
    mode: TaskMode = Field(..., description="Execution mode")
    created_at: datetime = Field(..., description="Creation timestamp")
    estimated_completion: Optional[datetime] = Field(
        default=None, description="Rough completion estimate"
    )
    links: Links = Field(..., description="Related links")


class SubtaskProgress(BaseModel):
    """Progress information for subtasks."""

    total_subtasks: int = Field(default=0, description="Total number of subtasks")
    completed_subtasks: int = Field(default=0, description="Completed subtasks")
    failed_subtasks: int = Field(default=0, description="Failed subtasks")
    pending_review: int = Field(default=0, description="Subtasks pending review")


class CurrentActivity(BaseModel):
    """Current activity information."""

    description: str = Field(..., description="What's currently happening")
    worker: Optional[str] = Field(default=None, description="Worker handling this")
    started_at: Optional[datetime] = Field(
        default=None, description="When activity started"
    )


class TaskOutput(BaseModel):
    """Output artifact from a task."""

    artifact_id: str = Field(..., description="Artifact identifier")
    type: str = Field(..., description="Type of artifact")
    location: str = Field(..., description="Storage location")


class TaskError(BaseModel):
    """Error information for failed tasks."""

    task_id: str = Field(..., description="Task that failed")
    error: str = Field(..., description="Error message")


class HumanActionNeeded(BaseModel):
    """Information about required human action."""

    review_id: str = Field(..., description="Review request ID")
    type: ReviewType = Field(..., description="Type of review needed")
    summary: str = Field(..., description="Brief summary of what's needed")


class TaskStatusResponse(BaseModel):
    """Detailed task status response."""

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Current status")
    mode: TaskMode = Field(..., description="Execution mode")
    progress: SubtaskProgress = Field(..., description="Subtask progress")
    current_activity: Optional[CurrentActivity] = Field(
        default=None, description="Current activity if in progress"
    )
    outputs: list[TaskOutput] = Field(
        default_factory=list, description="Completed outputs"
    )
    errors: list[TaskError] = Field(
        default_factory=list, description="Errors if failed"
    )
    human_actions_needed: list[HumanActionNeeded] = Field(
        default_factory=list, description="Pending human actions"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(
        default=None, description="Last update timestamp"
    )


class TaskListItem(BaseModel):
    """Summary item for task listing."""

    task_id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description (truncated)")
    status: TaskStatus = Field(..., description="Current status")
    mode: TaskMode = Field(..., description="Execution mode")
    created_at: datetime = Field(..., description="Creation timestamp")


class TaskListResponse(BaseModel):
    """Response for listing tasks."""

    tasks: list[TaskListItem] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total count")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Items per page")


class TaskTreeNode(BaseModel):
    """Node in task decomposition tree."""

    task_id: str = Field(..., description="Task identifier")
    description: str = Field(..., description="Task description")
    status: TaskStatus = Field(..., description="Status")
    tier: Optional[str] = Field(default=None, description="Model tier")
    children: list["TaskTreeNode"] = Field(
        default_factory=list, description="Child subtasks"
    )


class TaskTreeResponse(BaseModel):
    """Response for task tree endpoint."""

    root: TaskTreeNode = Field(..., description="Root task node")


# ===== Review Schemas =====


class AutomatedQAResult(BaseModel):
    """Results from automated QA."""

    issues: list[dict[str, Any]] = Field(
        default_factory=list, description="Detected issues"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations"
    )


class ReviewOption(BaseModel):
    """Option for decision reviews."""

    id: str = Field(..., description="Option identifier")
    description: str = Field(..., description="Option description")
    implications: str = Field(..., description="Implications of this choice")


class ReviewContext(BaseModel):
    """Context for a review request."""

    task_description: str = Field(..., description="Original task description")
    output_summary: str = Field(..., description="Summary of output to review")
    specific_questions: list[str] = Field(
        default_factory=list, description="Specific questions for reviewer"
    )


class ReviewResponse(BaseModel):
    """Detailed review information."""

    review_id: str = Field(..., description="Unique review identifier")
    task_id: str = Field(..., description="Related task ID")
    type: ReviewType = Field(..., description="Type of review")
    status: ReviewStatus = Field(..., description="Current status")
    context: ReviewContext = Field(..., description="Review context")
    automated_results: Optional[AutomatedQAResult] = Field(
        default=None, description="Automated QA results if available"
    )
    options: list[ReviewOption] = Field(
        default_factory=list, description="Options for decision reviews"
    )
    recommendation: Optional[str] = Field(
        default=None, description="System's recommendation"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    timeout_at: Optional[datetime] = Field(
        default=None, description="Expiration timestamp"
    )


class ReviewListItem(BaseModel):
    """Summary item for review listing."""

    review_id: str = Field(..., description="Review identifier")
    task_id: str = Field(..., description="Related task ID")
    type: ReviewType = Field(..., description="Type of review")
    status: ReviewStatus = Field(..., description="Current status")
    summary: str = Field(..., description="Brief summary")
    created_at: datetime = Field(..., description="Creation timestamp")
    timeout_at: Optional[datetime] = Field(default=None, description="Expiration")


class ReviewListResponse(BaseModel):
    """Response for listing reviews."""

    reviews: list[ReviewListItem] = Field(..., description="List of reviews")
    total: int = Field(..., description="Total count")


class ReviewApproveRequest(BaseModel):
    """Request to approve a review."""

    comments: Optional[str] = Field(default=None, description="Optional comments")


class ReviewRejectRequest(BaseModel):
    """Request to reject a review."""

    reason: str = Field(..., description="Reason for rejection")


class ReviewRequestChangesRequest(BaseModel):
    """Request to request changes."""

    guidance: str = Field(..., description="Guidance for changes")
    required_changes: list[str] = Field(
        default_factory=list, description="Specific required changes"
    )


class ReviewActionResponse(BaseModel):
    """Response after review action."""

    review_id: str = Field(..., description="Review identifier")
    status: ReviewStatus = Field(..., description="New status")
    message: str = Field(..., description="Confirmation message")


# ===== Project Schemas =====


class ProjectConfig(BaseModel):
    """Project configuration."""

    qa_profile: str = Field(default="consensus", description="QA profile to use")
    default_mode: TaskMode = Field(default=TaskMode.AUTONOMOUS, description="Default mode")
    constitution: Optional[dict[str, Any]] = Field(
        default=None, description="Project constitution/standards"
    )


class ProjectCreateRequest(BaseModel):
    """Request to create a project."""

    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(default=None, description="Project description")
    config: Optional[ProjectConfig] = Field(
        default=None, description="Project configuration"
    )
    repo_url: Optional[str] = Field(
        default=None, description="Git repository URL"
    )


class ProjectResponse(BaseModel):
    """Project information response."""

    id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(default=None, description="Description")
    config: ProjectConfig = Field(..., description="Configuration")
    repo_url: Optional[str] = Field(default=None, description="Repository URL")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update")
    task_count: int = Field(default=0, description="Number of tasks")


class ProjectListResponse(BaseModel):
    """Response for listing projects."""

    projects: list[ProjectResponse] = Field(..., description="List of projects")
    total: int = Field(..., description="Total count")


class ProjectUpdateRequest(BaseModel):
    """Request to update a project."""

    description: Optional[str] = Field(default=None, description="New description")
    config: Optional[ProjectConfig] = Field(default=None, description="New configuration")


class ConstitutionSetRequest(BaseModel):
    """Request to set project constitution."""

    constitution: dict[str, Any] = Field(..., description="Constitution content")


# ===== Monitoring Schemas =====


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    components: dict[str, str] = Field(
        default_factory=dict, description="Component statuses"
    )


class WorkerInfo(BaseModel):
    """Worker information."""

    worker_id: str = Field(..., description="Worker identifier")
    status: str = Field(..., description="Worker status")
    tier: str = Field(..., description="Model tier")
    current_task: Optional[str] = Field(
        default=None, description="Currently processing task"
    )
    tasks_completed: int = Field(default=0, description="Total completed tasks")
    last_heartbeat: Optional[datetime] = Field(
        default=None, description="Last heartbeat timestamp"
    )


class WorkersResponse(BaseModel):
    """Response for workers endpoint."""

    workers: list[WorkerInfo] = Field(..., description="List of workers")
    total_active: int = Field(..., description="Active worker count")


class QueueInfo(BaseModel):
    """Queue information."""

    name: str = Field(..., description="Queue name")
    depth: int = Field(..., description="Current queue depth")
    consumers: int = Field(..., description="Number of consumers")


class QueuesResponse(BaseModel):
    """Response for queues endpoint."""

    queues: list[QueueInfo] = Field(..., description="List of queues")


class FailureInfo(BaseModel):
    """Failure information."""

    task_id: str = Field(..., description="Failed task ID")
    error: str = Field(..., description="Error message")
    failed_at: datetime = Field(..., description="Failure timestamp")
    tier: Optional[str] = Field(default=None, description="Model tier")


class FailuresResponse(BaseModel):
    """Response for failures endpoint."""

    failures: list[FailureInfo] = Field(..., description="List of failures")
    total: int = Field(..., description="Total count")


# ===== Provenance Schemas =====


class ProvenanceRecord(BaseModel):
    """Provenance record information."""

    id: str = Field(..., description="Record identifier")
    task_id: str = Field(..., description="Task ID")
    project_id: str = Field(..., description="Project ID")
    model_used: str = Field(..., description="Model used")
    worker_id: str = Field(..., description="Worker ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    verification_status: str = Field(..., description="Verification status")
    is_tainted: bool = Field(default=False, description="Taint status")


class ProvenanceResponse(BaseModel):
    """Response for provenance endpoint."""

    record: ProvenanceRecord = Field(..., description="Provenance record")


class DependencyInfo(BaseModel):
    """Dependency/dependent information."""

    artifact_id: str = Field(..., description="Artifact ID")
    provenance_id: str = Field(..., description="Provenance record ID")
    relationship: str = Field(..., description="Relationship type")


class DependentsResponse(BaseModel):
    """Response for dependents endpoint."""

    artifact_id: str = Field(..., description="Source artifact ID")
    dependents: list[DependencyInfo] = Field(..., description="Dependent artifacts")
    total: int = Field(..., description="Total count")


class DependenciesResponse(BaseModel):
    """Response for dependencies endpoint."""

    artifact_id: str = Field(..., description="Source artifact ID")
    dependencies: list[DependencyInfo] = Field(..., description="Dependencies")
    total: int = Field(..., description="Total count")


class ImpactAnalysisResponse(BaseModel):
    """Response for impact analysis endpoint."""

    artifact_id: str = Field(..., description="Analyzed artifact ID")
    direct_dependents: list[str] = Field(..., description="Direct dependents")
    transitive_dependents: list[str] = Field(..., description="All transitive dependents")
    total_affected: int = Field(..., description="Total affected count")
    max_depth: int = Field(..., description="Maximum cascade depth")


# ===== WebSocket/SSE Schemas =====


class WSSubscribe(BaseModel):
    """WebSocket subscribe message."""

    type: str = Field(default="subscribe", description="Message type")
    task_id: str = Field(..., description="Task to subscribe to")


class WSUnsubscribe(BaseModel):
    """WebSocket unsubscribe message."""

    type: str = Field(default="unsubscribe", description="Message type")
    task_id: str = Field(..., description="Task to unsubscribe from")


class SSEEvent(BaseModel):
    """Server-sent event."""

    type: str = Field(..., description="Event type")
    task_id: str = Field(..., description="Related task ID")
    data: dict[str, Any] = Field(..., description="Event data")


class StatusUpdateEvent(SSEEvent):
    """Status update event."""

    type: str = Field(default="status_update")
    status: TaskStatus = Field(..., description="New status")
    details: Optional[dict[str, Any]] = Field(default=None, description="Extra details")


class SubtaskCompleteEvent(SSEEvent):
    """Subtask completion event."""

    type: str = Field(default="subtask_complete")
    subtask_id: str = Field(..., description="Completed subtask ID")
    result: str = Field(..., description="Result (passed/failed)")


class ReviewNeededEvent(SSEEvent):
    """Review needed event."""

    type: str = Field(default="review_needed")
    review_id: str = Field(..., description="Review request ID")


class TaskCompleteEvent(SSEEvent):
    """Task completion event."""

    type: str = Field(default="task_complete")
    outputs: list[TaskOutput] = Field(..., description="Task outputs")


class TaskFailedEvent(SSEEvent):
    """Task failed event."""

    type: str = Field(default="task_failed")
    error: str = Field(..., description="Error message")


# Allow recursive model for TaskTreeNode
TaskTreeNode.model_rebuild()