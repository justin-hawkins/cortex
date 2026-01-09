"""
Human review handling for DATS QA.

Manages the queue, notification, and response handling for
outputs that require human review.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class HumanReviewStatus(str, Enum):
    """Status of a human review request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    DELEGATED = "delegated"
    ESCALATED = "escalated"
    EXPIRED = "expired"


class HumanReviewPriority(str, Enum):
    """Priority level for human review."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HumanReviewRequest:
    """Request for human review."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    project_id: str = ""

    # What to review
    task_description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    work_output: str = ""
    output_type: str = "code"

    # Context for the reviewer
    automated_qa_results: Optional[dict[str, Any]] = None
    decision_points: list[str] = field(default_factory=list)
    questions_for_reviewer: list[str] = field(default_factory=list)
    rag_context: Optional[str] = None

    # Metadata
    priority: HumanReviewPriority = HumanReviewPriority.NORMAL
    timeout_hours: int = 24
    reminder_hours: int = 4
    assigned_reviewers: list[str] = field(default_factory=list)
    required_approvals: int = 1

    # State
    status: HumanReviewStatus = HumanReviewStatus.PENDING
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamps."""
        now = datetime.utcnow()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        if self.expires_at is None:
            self.expires_at = now + timedelta(hours=self.timeout_hours)

    def is_expired(self) -> bool:
        """Check if the review request has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def needs_reminder(self) -> bool:
        """Check if a reminder should be sent."""
        if self.status != HumanReviewStatus.PENDING:
            return False
        if self.created_at is None or self.expires_at is None:
            return False

        # Check if we're past the reminder threshold
        reminder_time = self.created_at + timedelta(hours=self.reminder_hours)
        return datetime.utcnow() > reminder_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "project_id": self.project_id,
            "task_description": self.task_description,
            "acceptance_criteria": self.acceptance_criteria,
            "work_output": self.work_output,
            "output_type": self.output_type,
            "automated_qa_results": self.automated_qa_results,
            "decision_points": self.decision_points,
            "questions_for_reviewer": self.questions_for_reviewer,
            "rag_context": self.rag_context,
            "priority": self.priority.value,
            "timeout_hours": self.timeout_hours,
            "reminder_hours": self.reminder_hours,
            "assigned_reviewers": self.assigned_reviewers,
            "required_approvals": self.required_approvals,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanReviewRequest":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            project_id=data.get("project_id", ""),
            task_description=data.get("task_description", ""),
            acceptance_criteria=data.get("acceptance_criteria", []),
            work_output=data.get("work_output", ""),
            output_type=data.get("output_type", "code"),
            automated_qa_results=data.get("automated_qa_results"),
            decision_points=data.get("decision_points", []),
            questions_for_reviewer=data.get("questions_for_reviewer", []),
            rag_context=data.get("rag_context"),
            priority=HumanReviewPriority(data.get("priority", "normal")),
            timeout_hours=data.get("timeout_hours", 24),
            reminder_hours=data.get("reminder_hours", 4),
            assigned_reviewers=data.get("assigned_reviewers", []),
            required_approvals=data.get("required_approvals", 1),
            status=HumanReviewStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )


@dataclass
class HumanReviewResponse:
    """Response from a human reviewer."""

    request_id: str
    reviewer_id: str
    status: HumanReviewStatus
    comments: str = ""
    required_changes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    delegate_to: Optional[str] = None
    escalation_reason: Optional[str] = None
    responded_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize timestamp."""
        if self.responded_at is None:
            self.responded_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "reviewer_id": self.reviewer_id,
            "status": self.status.value,
            "comments": self.comments,
            "required_changes": self.required_changes,
            "suggestions": self.suggestions,
            "delegate_to": self.delegate_to,
            "escalation_reason": self.escalation_reason,
            "responded_at": self.responded_at.isoformat()
            if self.responded_at
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanReviewResponse":
        """Create from dictionary."""
        return cls(
            request_id=data.get("request_id", ""),
            reviewer_id=data.get("reviewer_id", ""),
            status=HumanReviewStatus(data.get("status", "pending")),
            comments=data.get("comments", ""),
            required_changes=data.get("required_changes", []),
            suggestions=data.get("suggestions", []),
            delegate_to=data.get("delegate_to"),
            escalation_reason=data.get("escalation_reason"),
            responded_at=datetime.fromisoformat(data["responded_at"])
            if data.get("responded_at")
            else None,
        )


# Type for notification callback
NotificationCallback = Callable[[HumanReviewRequest], None]


class HumanReviewQueue:
    """
    Queue for managing human review requests.

    Provides methods for creating, tracking, and responding to
    human review requests with notification and timeout handling.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        notification_callback: Optional[NotificationCallback] = None,
    ):
        """
        Initialize human review queue.

        Args:
            storage_path: Optional path for file-based storage
            notification_callback: Optional callback for notifications
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.notification_callback = notification_callback
        self._requests: dict[str, HumanReviewRequest] = {}
        self._responses: dict[str, list[HumanReviewResponse]] = {}

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_request(
        self,
        task_id: str,
        project_id: str,
        task_description: str,
        work_output: str,
        acceptance_criteria: list[str] = None,
        output_type: str = "code",
        automated_qa_results: Optional[dict[str, Any]] = None,
        decision_points: list[str] = None,
        questions: list[str] = None,
        priority: HumanReviewPriority = HumanReviewPriority.NORMAL,
        timeout_hours: int = 24,
        assigned_reviewers: list[str] = None,
    ) -> HumanReviewRequest:
        """
        Create a new human review request.

        Args:
            task_id: Associated task ID
            project_id: Associated project ID
            task_description: Description of the task
            work_output: The output to review
            acceptance_criteria: Criteria for acceptance
            output_type: Type of output (code, documentation, etc.)
            automated_qa_results: Results from automated QA
            decision_points: Key decisions for reviewer
            questions: Specific questions for reviewer
            priority: Priority level
            timeout_hours: Hours until timeout
            assigned_reviewers: Specific reviewers to assign

        Returns:
            Created HumanReviewRequest
        """
        request = HumanReviewRequest(
            task_id=task_id,
            project_id=project_id,
            task_description=task_description,
            work_output=work_output,
            acceptance_criteria=acceptance_criteria or [],
            output_type=output_type,
            automated_qa_results=automated_qa_results,
            decision_points=decision_points or [],
            questions_for_reviewer=questions or [],
            priority=priority,
            timeout_hours=timeout_hours,
            assigned_reviewers=assigned_reviewers or [],
        )

        self._requests[request.id] = request
        self._responses[request.id] = []

        # Persist if storage configured
        if self.storage_path:
            self._save_request(request)

        # Send notification
        if self.notification_callback:
            try:
                self.notification_callback(request)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

        logger.info(
            f"Created human review request {request.id} for task {task_id}"
        )
        return request

    def get_request(self, request_id: str) -> Optional[HumanReviewRequest]:
        """
        Get a review request by ID.

        Args:
            request_id: Request ID

        Returns:
            HumanReviewRequest if found
        """
        if request_id in self._requests:
            return self._requests[request_id]

        if self.storage_path:
            return self._load_request(request_id)

        return None

    def submit_response(
        self,
        request_id: str,
        reviewer_id: str,
        status: HumanReviewStatus,
        comments: str = "",
        required_changes: list[str] = None,
        suggestions: list[str] = None,
        delegate_to: Optional[str] = None,
        escalation_reason: Optional[str] = None,
    ) -> HumanReviewResponse:
        """
        Submit a human review response.

        Args:
            request_id: ID of the request being responded to
            reviewer_id: ID of the reviewer
            status: Review verdict
            comments: Reviewer comments
            required_changes: Required changes for revision
            suggestions: Optional suggestions
            delegate_to: ID of reviewer to delegate to
            escalation_reason: Reason for escalation

        Returns:
            Created HumanReviewResponse
        """
        request = self.get_request(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        response = HumanReviewResponse(
            request_id=request_id,
            reviewer_id=reviewer_id,
            status=status,
            comments=comments,
            required_changes=required_changes or [],
            suggestions=suggestions or [],
            delegate_to=delegate_to,
            escalation_reason=escalation_reason,
        )

        # Store response
        if request_id not in self._responses:
            self._responses[request_id] = []
        self._responses[request_id].append(response)

        # Update request status
        request.status = status
        request.updated_at = datetime.utcnow()

        # Handle delegation
        if status == HumanReviewStatus.DELEGATED and delegate_to:
            if delegate_to not in request.assigned_reviewers:
                request.assigned_reviewers.append(delegate_to)
            request.status = HumanReviewStatus.PENDING

        # Persist
        if self.storage_path:
            self._save_request(request)
            self._save_response(response)

        logger.info(
            f"Human review response submitted for {request_id}: {status.value}"
        )
        return response

    def get_pending_reviews(
        self,
        reviewer_id: Optional[str] = None,
        project_id: Optional[str] = None,
        priority: Optional[HumanReviewPriority] = None,
    ) -> list[HumanReviewRequest]:
        """
        Get pending review requests.

        Args:
            reviewer_id: Filter by assigned reviewer
            project_id: Filter by project
            priority: Filter by priority

        Returns:
            List of pending requests
        """
        results = []

        for request in self._requests.values():
            if request.status != HumanReviewStatus.PENDING:
                continue

            if reviewer_id and reviewer_id not in request.assigned_reviewers:
                if request.assigned_reviewers:  # Has specific assignees
                    continue

            if project_id and request.project_id != project_id:
                continue

            if priority and request.priority != priority:
                continue

            results.append(request)

        # Sort by priority and creation time
        priority_order = {
            HumanReviewPriority.CRITICAL: 0,
            HumanReviewPriority.HIGH: 1,
            HumanReviewPriority.NORMAL: 2,
            HumanReviewPriority.LOW: 3,
        }
        results.sort(
            key=lambda r: (
                priority_order.get(r.priority, 4),
                r.created_at or datetime.utcnow(),
            )
        )

        return results

    def check_expired_reviews(
        self,
        escalate: bool = True,
    ) -> list[HumanReviewRequest]:
        """
        Check for and handle expired review requests.

        Args:
            escalate: Whether to escalate expired reviews

        Returns:
            List of expired requests
        """
        expired = []

        for request in self._requests.values():
            if request.status == HumanReviewStatus.PENDING and request.is_expired():
                expired.append(request)

                if escalate:
                    request.status = HumanReviewStatus.ESCALATED
                else:
                    request.status = HumanReviewStatus.EXPIRED

                request.updated_at = datetime.utcnow()

                if self.storage_path:
                    self._save_request(request)

        if expired:
            logger.warning(f"Found {len(expired)} expired human review requests")

        return expired

    def get_responses(self, request_id: str) -> list[HumanReviewResponse]:
        """
        Get all responses for a request.

        Args:
            request_id: Request ID

        Returns:
            List of responses
        """
        return self._responses.get(request_id, [])

    def is_approved(self, request_id: str) -> bool:
        """
        Check if a request has been approved.

        Args:
            request_id: Request ID

        Returns:
            True if approved
        """
        request = self.get_request(request_id)
        if not request:
            return False

        if request.status == HumanReviewStatus.APPROVED:
            return True

        # Check if we have enough approvals
        responses = self.get_responses(request_id)
        approvals = sum(
            1
            for r in responses
            if r.status == HumanReviewStatus.APPROVED
        )

        return approvals >= request.required_approvals

    def _save_request(self, request: HumanReviewRequest):
        """Save request to disk."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"request_{request.id}.json"
        with open(file_path, "w") as f:
            json.dump(request.to_dict(), f, indent=2)

    def _load_request(self, request_id: str) -> Optional[HumanReviewRequest]:
        """Load request from disk."""
        if not self.storage_path:
            return None

        file_path = self.storage_path / f"request_{request_id}.json"
        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)
            request = HumanReviewRequest.from_dict(data)
            self._requests[request.id] = request
            return request

    def _save_response(self, response: HumanReviewResponse):
        """Save response to disk."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"response_{response.request_id}_{response.reviewer_id}.json"
        with open(file_path, "w") as f:
            json.dump(response.to_dict(), f, indent=2)

    def format_review_summary(self, request: HumanReviewRequest) -> str:
        """
        Format a review request for human-readable display.

        Args:
            request: Review request to format

        Returns:
            Formatted summary string
        """
        sections = [
            f"# Human Review Request",
            f"**Task ID:** {request.task_id}",
            f"**Priority:** {request.priority.value.upper()}",
            f"**Created:** {request.created_at}",
            f"**Expires:** {request.expires_at}",
            "",
            "## Task Description",
            request.task_description,
            "",
        ]

        if request.acceptance_criteria:
            sections.append("## Acceptance Criteria")
            for criterion in request.acceptance_criteria:
                sections.append(f"- {criterion}")
            sections.append("")

        if request.decision_points:
            sections.append("## Key Decisions Needed")
            for point in request.decision_points:
                sections.append(f"- {point}")
            sections.append("")

        if request.questions_for_reviewer:
            sections.append("## Questions for Reviewer")
            for question in request.questions_for_reviewer:
                sections.append(f"- {question}")
            sections.append("")

        if request.automated_qa_results:
            sections.append("## Automated QA Results")
            sections.append("```json")
            sections.append(json.dumps(request.automated_qa_results, indent=2))
            sections.append("```")
            sections.append("")

        sections.append("## Work Output to Review")
        sections.append(f"```{request.output_type}")
        sections.append(request.work_output)
        sections.append("```")

        return "\n".join(sections)