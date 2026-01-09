"""
QA Result structures for DATS.

Defines the data structures for QA validation results, issues,
and aggregated verdicts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import uuid


class QAVerdict(str, Enum):
    """Overall QA verdict for a validation run."""

    PASS = "PASS"
    FAIL = "FAIL"
    NEEDS_REVISION = "NEEDS_REVISION"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    ESCALATE = "ESCALATE"


class IssueSeverity(str, Enum):
    """Severity level of a QA issue."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"


class IssueCategory(str, Enum):
    """Category of a QA issue."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    COMPLETENESS = "completeness"


class ProfileVerdict(str, Enum):
    """Verdict from a single profile validation."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class QAIssue:
    """Individual issue found during QA validation."""

    severity: IssueSeverity
    category: IssueCategory
    description: str
    location: str = ""
    recommendation: str = ""
    reviewer_id: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "location": self.location,
            "recommendation": self.recommendation,
            "reviewer_id": self.reviewer_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAIssue":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            severity=IssueSeverity(data.get("severity", "minor")),
            category=IssueCategory(data.get("category", "correctness")),
            description=data.get("description", ""),
            location=data.get("location", ""),
            recommendation=data.get("recommendation", ""),
            reviewer_id=data.get("reviewer_id", ""),
        )

    def is_blocking(self) -> bool:
        """Check if this issue should block approval."""
        return self.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR)


@dataclass
class ProfileResult:
    """Result from a single QA profile validation."""

    profile: str
    verdict: ProfileVerdict
    confidence: float = 0.0
    issues: list[QAIssue] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    reviewer_ids: list[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile": self.profile,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "issues": [issue.to_dict() for issue in self.issues],
            "details": self.details,
            "reviewer_ids": self.reviewer_ids,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileResult":
        """Create from dictionary."""
        return cls(
            profile=data.get("profile", ""),
            verdict=ProfileVerdict(data.get("verdict", "fail")),
            confidence=data.get("confidence", 0.0),
            issues=[QAIssue.from_dict(i) for i in data.get("issues", [])],
            details=data.get("details", {}),
            reviewer_ids=data.get("reviewer_ids", []),
            duration_ms=data.get("duration_ms", 0),
        )

    def has_critical_issues(self) -> bool:
        """Check if any critical issues were found."""
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    def has_major_issues(self) -> bool:
        """Check if any major issues were found."""
        return any(i.severity == IssueSeverity.MAJOR for i in self.issues)

    def count_issues_by_severity(self, severity: IssueSeverity) -> int:
        """Count issues of a specific severity."""
        return sum(1 for i in self.issues if i.severity == severity)


@dataclass
class RevisionGuidance:
    """Guidance for revising a failed output."""

    required_changes: list[str] = field(default_factory=list)
    suggested_improvements: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    additional_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "required_changes": self.required_changes,
            "suggested_improvements": self.suggested_improvements,
            "focus_areas": self.focus_areas,
            "additional_context": self.additional_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RevisionGuidance":
        """Create from dictionary."""
        return cls(
            required_changes=data.get("required_changes", []),
            suggested_improvements=data.get("suggested_improvements", []),
            focus_areas=data.get("focus_areas", []),
            additional_context=data.get("additional_context", ""),
        )

    def is_empty(self) -> bool:
        """Check if there's any guidance."""
        return not (
            self.required_changes
            or self.suggested_improvements
            or self.focus_areas
        )


@dataclass
class QAMetadata:
    """Metadata about the QA validation run."""

    reviewer_ids: list[str] = field(default_factory=list)
    duration_ms: int = 0
    tokens_consumed: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    escalation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reviewer_ids": self.reviewer_ids,
            "duration_ms": self.duration_ms,
            "tokens_consumed": self.tokens_consumed,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "escalation_reason": self.escalation_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAMetadata":
        """Create from dictionary."""
        return cls(
            reviewer_ids=data.get("reviewer_ids", []),
            duration_ms=data.get("duration_ms", 0),
            tokens_consumed=data.get("tokens_consumed", 0),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            retry_count=data.get("retry_count", 0),
            escalation_reason=data.get("escalation_reason", ""),
        )


@dataclass
class QAResult:
    """
    Complete QA validation result.

    Aggregates results from all profile validations and provides
    the final verdict with guidance for next steps.
    """

    task_id: str
    verdict: QAVerdict
    profile_results: list[ProfileResult] = field(default_factory=list)
    issues: list[QAIssue] = field(default_factory=list)
    revision_guidance: Optional[RevisionGuidance] = None
    metadata: Optional[QAMetadata] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "verdict": self.verdict.value,
            "profile_results": [pr.to_dict() for pr in self.profile_results],
            "issues": [issue.to_dict() for issue in self.issues],
            "revision_guidance": self.revision_guidance.to_dict()
            if self.revision_guidance
            else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAResult":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            task_id=data.get("task_id", ""),
            verdict=QAVerdict(data.get("verdict", "FAIL")),
            profile_results=[
                ProfileResult.from_dict(pr) for pr in data.get("profile_results", [])
            ],
            issues=[QAIssue.from_dict(i) for i in data.get("issues", [])],
            revision_guidance=RevisionGuidance.from_dict(data["revision_guidance"])
            if data.get("revision_guidance")
            else None,
            metadata=QAMetadata.from_dict(data["metadata"])
            if data.get("metadata")
            else None,
        )

    def is_approved(self) -> bool:
        """Check if the output is approved."""
        return self.verdict == QAVerdict.PASS

    def needs_revision(self) -> bool:
        """Check if revision is needed."""
        return self.verdict == QAVerdict.NEEDS_REVISION

    def needs_human_review(self) -> bool:
        """Check if human review is needed."""
        return self.verdict in (QAVerdict.NEEDS_HUMAN, QAVerdict.ESCALATE)

    def get_critical_issues(self) -> list[QAIssue]:
        """Get all critical severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    def get_blocking_issues(self) -> list[QAIssue]:
        """Get all issues that block approval."""
        return [i for i in self.issues if i.is_blocking()]

    def get_issues_by_category(self, category: IssueCategory) -> list[QAIssue]:
        """Get issues of a specific category."""
        return [i for i in self.issues if i.category == category]

    @classmethod
    def create_pass(
        cls,
        task_id: str,
        profile_results: list[ProfileResult],
        metadata: Optional[QAMetadata] = None,
    ) -> "QAResult":
        """Create a passing result."""
        all_issues = []
        for pr in profile_results:
            all_issues.extend(pr.issues)

        return cls(
            task_id=task_id,
            verdict=QAVerdict.PASS,
            profile_results=profile_results,
            issues=all_issues,
            metadata=metadata,
        )

    @classmethod
    def create_fail(
        cls,
        task_id: str,
        profile_results: list[ProfileResult],
        revision_guidance: Optional[RevisionGuidance] = None,
        metadata: Optional[QAMetadata] = None,
    ) -> "QAResult":
        """Create a failing result."""
        all_issues = []
        for pr in profile_results:
            all_issues.extend(pr.issues)

        return cls(
            task_id=task_id,
            verdict=QAVerdict.FAIL,
            profile_results=profile_results,
            issues=all_issues,
            revision_guidance=revision_guidance,
            metadata=metadata,
        )

    @classmethod
    def create_needs_revision(
        cls,
        task_id: str,
        profile_results: list[ProfileResult],
        revision_guidance: RevisionGuidance,
        metadata: Optional[QAMetadata] = None,
    ) -> "QAResult":
        """Create a needs-revision result."""
        all_issues = []
        for pr in profile_results:
            all_issues.extend(pr.issues)

        return cls(
            task_id=task_id,
            verdict=QAVerdict.NEEDS_REVISION,
            profile_results=profile_results,
            issues=all_issues,
            revision_guidance=revision_guidance,
            metadata=metadata,
        )

    @classmethod
    def create_needs_human(
        cls,
        task_id: str,
        profile_results: list[ProfileResult],
        reason: str = "",
        metadata: Optional[QAMetadata] = None,
    ) -> "QAResult":
        """Create a needs-human-review result."""
        all_issues = []
        for pr in profile_results:
            all_issues.extend(pr.issues)

        if metadata:
            metadata.escalation_reason = reason

        return cls(
            task_id=task_id,
            verdict=QAVerdict.NEEDS_HUMAN,
            profile_results=profile_results,
            issues=all_issues,
            metadata=metadata,
        )

    @classmethod
    def create_escalate(
        cls,
        task_id: str,
        profile_results: list[ProfileResult],
        reason: str,
        metadata: Optional[QAMetadata] = None,
    ) -> "QAResult":
        """Create an escalation result."""
        all_issues = []
        for pr in profile_results:
            all_issues.extend(pr.issues)

        if metadata:
            metadata.escalation_reason = reason

        return cls(
            task_id=task_id,
            verdict=QAVerdict.ESCALATE,
            profile_results=profile_results,
            issues=all_issues,
            metadata=metadata,
        )