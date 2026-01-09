"""
Data models for merge coordination.

Defines the core data structures for conflict detection,
classification, and resolution.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ConflictType(str, Enum):
    """Type of conflict detected."""

    TEXTUAL = "textual"  # Same file/lines modified
    SEMANTIC = "semantic"  # Different approaches to same problem
    ARCHITECTURAL = "architectural"  # Incompatible design assumptions


class ConflictSeverity(str, Enum):
    """Severity level of a conflict."""

    TRIVIAL = "trivial"  # Non-overlapping, auto-mergeable
    MODERATE = "moderate"  # Overlapping but semantically mergeable
    SIGNIFICANT = "significant"  # Requires careful consideration
    FUNDAMENTAL = "fundamental"  # Architectural incompatibility


class ResolutionStrategy(str, Enum):
    """Strategy for resolving a conflict."""

    AUTO_MERGE = "auto_merge"  # Standard three-way merge
    SEMANTIC_MERGE = "semantic_merge"  # LLM-assisted merge
    REDESIGN = "redesign"  # Re-decomposition needed
    HUMAN_DECISION = "human_decision"  # Escalate to human


class ResolutionStatus(str, Enum):
    """Status of conflict resolution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass
class InvolvedTask:
    """Task involved in a conflict."""

    task_id: str
    output_artifact_id: str
    completed_at: Optional[datetime] = None
    description: Optional[str] = None
    output_summary: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "output_artifact_id": self.output_artifact_id,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "description": self.description,
            "output_summary": self.output_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InvolvedTask":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            output_artifact_id=data.get("output_artifact_id", ""),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            description=data.get("description"),
            output_summary=data.get("output_summary"),
        )


@dataclass
class ConflictRegion:
    """Region of conflict within an artifact (for textual conflicts)."""

    start_line: int
    end_line: int
    ours_content: str = ""
    theirs_content: str = ""
    base_content: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "ours_content": self.ours_content,
            "theirs_content": self.theirs_content,
            "base_content": self.base_content,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictRegion":
        """Create from dictionary."""
        return cls(
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            ours_content=data.get("ours_content", ""),
            theirs_content=data.get("theirs_content", ""),
            base_content=data.get("base_content", ""),
            description=data.get("description", ""),
        )


@dataclass
class AffectedArtifact:
    """Artifact affected by a conflict."""

    artifact_id: str
    location: str  # File path or identifier
    artifact_type: str = "code"  # code, document, config, etc.
    conflict_regions: list[ConflictRegion] = field(default_factory=list)
    ours_checksum: Optional[str] = None
    theirs_checksum: Optional[str] = None
    base_checksum: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "location": self.location,
            "artifact_type": self.artifact_type,
            "conflict_regions": [r.to_dict() for r in self.conflict_regions],
            "ours_checksum": self.ours_checksum,
            "theirs_checksum": self.theirs_checksum,
            "base_checksum": self.base_checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AffectedArtifact":
        """Create from dictionary."""
        return cls(
            artifact_id=data.get("artifact_id", ""),
            location=data.get("location", ""),
            artifact_type=data.get("artifact_type", "code"),
            conflict_regions=[
                ConflictRegion.from_dict(r)
                for r in data.get("conflict_regions", [])
            ],
            ours_checksum=data.get("ours_checksum"),
            theirs_checksum=data.get("theirs_checksum"),
            base_checksum=data.get("base_checksum"),
        )


@dataclass
class ConflictClassification:
    """Classification of a conflict."""

    type: ConflictType
    severity: ConflictSeverity
    auto_resolvable: bool
    confidence: float
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "auto_resolvable": self.auto_resolvable,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictClassification":
        """Create from dictionary."""
        return cls(
            type=ConflictType(data.get("type", "textual")),
            severity=ConflictSeverity(data.get("severity", "moderate")),
            auto_resolvable=data.get("auto_resolvable", False),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class RedesignRecommendation:
    """Recommendation for redesign when conflicts are architectural."""

    problem: str
    suggested_approach: str
    tasks_to_invalidate: list[str] = field(default_factory=list)
    context_for_redecomposition: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem": self.problem,
            "suggested_approach": self.suggested_approach,
            "tasks_to_invalidate": self.tasks_to_invalidate,
            "context_for_redecomposition": self.context_for_redecomposition,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedesignRecommendation":
        """Create from dictionary."""
        return cls(
            problem=data.get("problem", ""),
            suggested_approach=data.get("suggested_approach", ""),
            tasks_to_invalidate=data.get("tasks_to_invalidate", []),
            context_for_redecomposition=data.get("context_for_redecomposition", ""),
        )


@dataclass
class HumanDecisionRequest:
    """Request for human decision on a conflict."""

    question: str
    options: list[dict[str, str]] = field(default_factory=list)  # [{option, implications}]
    recommendation: str = ""
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "options": self.options,
            "recommendation": self.recommendation,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanDecisionRequest":
        """Create from dictionary."""
        return cls(
            question=data.get("question", ""),
            options=data.get("options", []),
            recommendation=data.get("recommendation", ""),
            context=data.get("context", ""),
        )


@dataclass
class MergedOutput:
    """Output from a successful merge."""

    artifact_type: str
    content: str
    merge_notes: str = ""
    artifacts_produced: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "content": self.content,
            "merge_notes": self.merge_notes,
            "artifacts_produced": self.artifacts_produced,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergedOutput":
        """Create from dictionary."""
        return cls(
            artifact_type=data.get("artifact_type", "code"),
            content=data.get("content", ""),
            merge_notes=data.get("merge_notes", ""),
            artifacts_produced=data.get("artifacts_produced", []),
        )


@dataclass
class ConflictResolution:
    """Resolution of a conflict."""

    strategy: ResolutionStrategy
    status: ResolutionStatus
    resolved_by: Optional[str] = None  # model_id or human_id
    resolution_artifact_id: Optional[str] = None

    # Resolution outcomes (one of these will be populated)
    merged_output: Optional[MergedOutput] = None
    redesign_recommendation: Optional[RedesignRecommendation] = None
    human_decision_request: Optional[HumanDecisionRequest] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "status": self.status.value,
            "resolved_by": self.resolved_by,
            "resolution_artifact_id": self.resolution_artifact_id,
            "merged_output": self.merged_output.to_dict() if self.merged_output else None,
            "redesign_recommendation": self.redesign_recommendation.to_dict()
            if self.redesign_recommendation
            else None,
            "human_decision_request": self.human_decision_request.to_dict()
            if self.human_decision_request
            else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictResolution":
        """Create from dictionary."""
        return cls(
            strategy=ResolutionStrategy(data.get("strategy", "auto_merge")),
            status=ResolutionStatus(data.get("status", "pending")),
            resolved_by=data.get("resolved_by"),
            resolution_artifact_id=data.get("resolution_artifact_id"),
            merged_output=MergedOutput.from_dict(data["merged_output"])
            if data.get("merged_output")
            else None,
            redesign_recommendation=RedesignRecommendation.from_dict(
                data["redesign_recommendation"]
            )
            if data.get("redesign_recommendation")
            else None,
            human_decision_request=HumanDecisionRequest.from_dict(
                data["human_decision_request"]
            )
            if data.get("human_decision_request")
            else None,
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
        )


@dataclass
class ConflictAudit:
    """Audit information for a conflict."""

    detection_method: str
    resolution_attempts: int = 0
    tokens_consumed: int = 0
    escalation_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection_method": self.detection_method,
            "resolution_attempts": self.resolution_attempts,
            "tokens_consumed": self.tokens_consumed,
            "escalation_history": self.escalation_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictAudit":
        """Create from dictionary."""
        return cls(
            detection_method=data.get("detection_method", ""),
            resolution_attempts=data.get("resolution_attempts", 0),
            tokens_consumed=data.get("tokens_consumed", 0),
            escalation_history=data.get("escalation_history", []),
        )


@dataclass
class ConflictRecord:
    """
    Complete record of a conflict.

    This is the main data structure representing a conflict throughout
    its lifecycle from detection through resolution.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    detected_at: datetime = field(default_factory=datetime.utcnow)
    project_id: str = ""

    # What's in conflict
    involved_tasks: list[InvolvedTask] = field(default_factory=list)
    affected_artifacts: list[AffectedArtifact] = field(default_factory=list)

    # Classification
    classification: Optional[ConflictClassification] = None

    # Resolution
    resolution: Optional[ConflictResolution] = None

    # Audit trail
    audit: Optional[ConflictAudit] = None

    # Context for resolution
    common_parent_task_id: Optional[str] = None
    lightrag_context: Optional[str] = None

    def is_resolved(self) -> bool:
        """Check if conflict has been resolved."""
        if not self.resolution:
            return False
        return self.resolution.status == ResolutionStatus.RESOLVED

    def is_escalated(self) -> bool:
        """Check if conflict has been escalated."""
        if not self.resolution:
            return False
        return self.resolution.status == ResolutionStatus.ESCALATED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "detected_at": self.detected_at.isoformat(),
            "project_id": self.project_id,
            "involved_tasks": [t.to_dict() for t in self.involved_tasks],
            "affected_artifacts": [a.to_dict() for a in self.affected_artifacts],
            "classification": self.classification.to_dict() if self.classification else None,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "audit": self.audit.to_dict() if self.audit else None,
            "common_parent_task_id": self.common_parent_task_id,
            "lightrag_context": self.lightrag_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictRecord":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            detected_at=datetime.fromisoformat(data["detected_at"])
            if data.get("detected_at")
            else datetime.utcnow(),
            project_id=data.get("project_id", ""),
            involved_tasks=[
                InvolvedTask.from_dict(t) for t in data.get("involved_tasks", [])
            ],
            affected_artifacts=[
                AffectedArtifact.from_dict(a) for a in data.get("affected_artifacts", [])
            ],
            classification=ConflictClassification.from_dict(data["classification"])
            if data.get("classification")
            else None,
            resolution=ConflictResolution.from_dict(data["resolution"])
            if data.get("resolution")
            else None,
            audit=ConflictAudit.from_dict(data["audit"]) if data.get("audit") else None,
            common_parent_task_id=data.get("common_parent_task_id"),
            lightrag_context=data.get("lightrag_context"),
        )


@dataclass
class ConflictBatch:
    """Batch of related conflicts for group resolution."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conflicts: list[ConflictRecord] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    window_seconds: float = 30.0
    project_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "created_at": self.created_at.isoformat(),
            "window_seconds": self.window_seconds,
            "project_id": self.project_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConflictBatch":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            conflicts=[
                ConflictRecord.from_dict(c) for c in data.get("conflicts", [])
            ],
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.utcnow(),
            window_seconds=data.get("window_seconds", 30.0),
            project_id=data.get("project_id", ""),
        )