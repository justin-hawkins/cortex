"""
Cascade scenario detection for DATS.

Detects when cascade failure handling is needed based on various triggers
such as QA failures, human review rejections, and downstream task failures.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.storage.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class CascadeTrigger(str, Enum):
    """What triggered the cascade detection."""

    QA_FAILURE = "qa_failure"  # QA failed on previously-passed output
    HUMAN_REJECTION = "human_rejection"  # Human review rejected
    DOWNSTREAM_FAILURE = "downstream_failure"  # Task failed due to bad input
    MANUAL_TAINT = "manual_taint"  # Explicit taint request
    REVALIDATION_FAILURE = "revalidation_failure"  # Revalidation found issue
    SECURITY_ISSUE = "security_issue"  # Security vulnerability discovered


class CascadeSeverity(str, Enum):
    """Severity of the cascade scenario."""

    LOW = "low"  # Limited impact, few dependents
    MEDIUM = "medium"  # Moderate impact
    HIGH = "high"  # Significant impact, many dependents
    CRITICAL = "critical"  # Architectural or widespread issue


@dataclass
class CascadeScenario:
    """Description of a detected cascade scenario."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: CascadeTrigger = CascadeTrigger.MANUAL_TAINT
    severity: CascadeSeverity = CascadeSeverity.LOW
    
    # What was affected
    source_artifact_id: str = ""
    source_provenance_id: str = ""
    source_task_id: str = ""
    project_id: str = ""
    
    # Impact assessment
    direct_dependents: list[str] = field(default_factory=list)
    transitive_dependents: list[str] = field(default_factory=list)
    total_affected: int = 0
    max_depth: int = 0
    
    # Context
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Recommendations
    recommended_action: str = "propagate"  # propagate, revalidate, rollback

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "trigger": self.trigger.value,
            "severity": self.severity.value,
            "source_artifact_id": self.source_artifact_id,
            "source_provenance_id": self.source_provenance_id,
            "source_task_id": self.source_task_id,
            "project_id": self.project_id,
            "direct_dependents": self.direct_dependents,
            "transitive_dependents": self.transitive_dependents,
            "total_affected": self.total_affected,
            "max_depth": self.max_depth,
            "reason": self.reason,
            "details": self.details,
            "detected_at": self.detected_at.isoformat(),
            "recommended_action": self.recommended_action,
        }


@dataclass
class CascadeMetrics:
    """Metrics about the cascade detection."""

    estimated_revalidation_count: int = 0
    estimated_cascade_depth: int = 0
    estimated_rollback_cost: int = 0  # Number of tasks to re-run
    checkpoint_available: bool = False
    nearest_checkpoint_age_hours: float = 0.0
    
    # Thresholds (for comparison)
    max_revalidation_depth: int = 5
    max_revalidation_count: int = 50

    def exceeds_thresholds(self) -> bool:
        """Check if metrics exceed configured thresholds."""
        return (
            self.estimated_cascade_depth > self.max_revalidation_depth
            or self.estimated_revalidation_count > self.max_revalidation_count
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "estimated_revalidation_count": self.estimated_revalidation_count,
            "estimated_cascade_depth": self.estimated_cascade_depth,
            "estimated_rollback_cost": self.estimated_rollback_cost,
            "checkpoint_available": self.checkpoint_available,
            "nearest_checkpoint_age_hours": self.nearest_checkpoint_age_hours,
            "max_revalidation_depth": self.max_revalidation_depth,
            "max_revalidation_count": self.max_revalidation_count,
            "exceeds_thresholds": self.exceeds_thresholds(),
        }


class CascadeDetector:
    """
    Detect cascade failure scenarios.
    
    Analyzes provenance graphs to determine impact of failures
    and recommend appropriate response strategies.
    """

    def __init__(
        self,
        provenance_tracker: ProvenanceTracker,
        max_revalidation_depth: int = 5,
        max_revalidation_count: int = 50,
    ):
        """
        Initialize cascade detector.
        
        Args:
            provenance_tracker: Tracker for provenance records
            max_revalidation_depth: Max depth before recommending rollback
            max_revalidation_count: Max count before recommending rollback
        """
        self.tracker = provenance_tracker
        self.max_revalidation_depth = max_revalidation_depth
        self.max_revalidation_count = max_revalidation_count

    def detect_from_qa_failure(
        self,
        task_id: str,
        artifact_id: str,
        qa_result: dict[str, Any],
    ) -> Optional[CascadeScenario]:
        """
        Detect cascade from QA failure on previously-passed output.
        
        Args:
            task_id: Task that failed QA
            artifact_id: Artifact that failed
            qa_result: QA result details
            
        Returns:
            CascadeScenario if cascade needed, None otherwise
        """
        record = self.tracker.get_producer(artifact_id)
        if not record:
            logger.warning(f"No provenance record found for artifact {artifact_id}")
            return None

        # Only trigger cascade if it was previously passed
        if record.verification.status not in (
            VerificationStatus.PASSED,
            VerificationStatus.HUMAN_APPROVED,
        ):
            logger.debug(f"Artifact {artifact_id} was not previously passed, no cascade needed")
            return None

        return self._build_scenario(
            trigger=CascadeTrigger.QA_FAILURE,
            artifact_id=artifact_id,
            record=record,
            reason=f"QA failure: {qa_result.get('reason', 'unknown')}",
            details={"qa_result": qa_result},
        )

    def detect_from_human_rejection(
        self,
        task_id: str,
        artifact_id: str,
        rejection_reason: str,
        reviewer_id: str = "",
    ) -> Optional[CascadeScenario]:
        """
        Detect cascade from human review rejection.
        
        Args:
            task_id: Task that was rejected
            artifact_id: Artifact that was rejected
            rejection_reason: Why it was rejected
            reviewer_id: Who rejected it
            
        Returns:
            CascadeScenario if cascade needed
        """
        record = self.tracker.get_producer(artifact_id)
        if not record:
            logger.warning(f"No provenance record found for artifact {artifact_id}")
            return None

        return self._build_scenario(
            trigger=CascadeTrigger.HUMAN_REJECTION,
            artifact_id=artifact_id,
            record=record,
            reason=rejection_reason,
            details={"reviewer_id": reviewer_id},
        )

    def detect_from_downstream_failure(
        self,
        failed_task_id: str,
        failed_artifact_id: str,
        suspected_input_id: str,
        failure_details: dict[str, Any],
    ) -> Optional[CascadeScenario]:
        """
        Detect cascade from downstream task failure due to bad input.
        
        Args:
            failed_task_id: Task that failed
            failed_artifact_id: Output that failed (if any)
            suspected_input_id: Input artifact suspected of causing failure
            failure_details: Details about the failure
            
        Returns:
            CascadeScenario if the suspected input is indeed problematic
        """
        input_record = self.tracker.get_producer(suspected_input_id)
        if not input_record:
            return None

        return self._build_scenario(
            trigger=CascadeTrigger.DOWNSTREAM_FAILURE,
            artifact_id=suspected_input_id,
            record=input_record,
            reason=f"Downstream task {failed_task_id} failed due to this input",
            details={
                "failed_task_id": failed_task_id,
                "failed_artifact_id": failed_artifact_id,
                "failure_details": failure_details,
            },
        )

    def detect_from_manual_taint(
        self,
        artifact_id: str,
        reason: str,
        requested_by: str = "",
    ) -> Optional[CascadeScenario]:
        """
        Detect cascade from manual taint request.
        
        Args:
            artifact_id: Artifact to taint
            reason: Why it's being tainted
            requested_by: Who requested the taint
            
        Returns:
            CascadeScenario for the taint
        """
        record = self.tracker.get_producer(artifact_id)
        if not record:
            logger.warning(f"No provenance record found for artifact {artifact_id}")
            return None

        return self._build_scenario(
            trigger=CascadeTrigger.MANUAL_TAINT,
            artifact_id=artifact_id,
            record=record,
            reason=reason,
            details={"requested_by": requested_by},
        )

    def detect_from_security_issue(
        self,
        artifact_id: str,
        vulnerability: str,
        severity: str = "high",
        cve_id: str = "",
    ) -> Optional[CascadeScenario]:
        """
        Detect cascade from discovered security vulnerability.
        
        Args:
            artifact_id: Affected artifact
            vulnerability: Description of vulnerability
            severity: Severity level
            cve_id: Optional CVE identifier
            
        Returns:
            CascadeScenario with elevated severity
        """
        record = self.tracker.get_producer(artifact_id)
        if not record:
            return None

        scenario = self._build_scenario(
            trigger=CascadeTrigger.SECURITY_ISSUE,
            artifact_id=artifact_id,
            record=record,
            reason=f"Security vulnerability: {vulnerability}",
            details={
                "vulnerability": vulnerability,
                "severity": severity,
                "cve_id": cve_id,
            },
        )

        # Security issues get elevated severity
        if scenario and severity in ("high", "critical"):
            scenario.severity = CascadeSeverity.CRITICAL

        return scenario

    def estimate_impact(self, artifact_id: str) -> CascadeMetrics:
        """
        Estimate impact of tainting an artifact.
        
        Args:
            artifact_id: Artifact to analyze
            
        Returns:
            CascadeMetrics with estimated costs
        """
        impact = self.tracker.impact_analysis(artifact_id)
        
        # Check for available checkpoints
        record = self.tracker.get_producer(artifact_id)
        checkpoints = []
        checkpoint_age = 0.0
        
        if record:
            checkpoints = self.tracker.get_checkpoints(record.project_id)
            valid_checkpoints = [c for c in checkpoints if c.is_valid]
            if valid_checkpoints:
                # Get age of nearest checkpoint
                nearest = max(valid_checkpoints, key=lambda c: c.created_at)
                age_delta = datetime.utcnow() - nearest.created_at
                checkpoint_age = age_delta.total_seconds() / 3600

        return CascadeMetrics(
            estimated_revalidation_count=impact.total_affected,
            estimated_cascade_depth=impact.max_depth,
            estimated_rollback_cost=impact.total_affected + 1,  # +1 for the source
            checkpoint_available=len([c for c in checkpoints if c.is_valid]) > 0,
            nearest_checkpoint_age_hours=checkpoint_age,
            max_revalidation_depth=self.max_revalidation_depth,
            max_revalidation_count=self.max_revalidation_count,
        )

    def _build_scenario(
        self,
        trigger: CascadeTrigger,
        artifact_id: str,
        record: ProvenanceRecord,
        reason: str,
        details: dict[str, Any],
    ) -> CascadeScenario:
        """Build a cascade scenario from detection data."""
        # Get impact analysis
        impact = self.tracker.impact_analysis(artifact_id)
        
        # Determine severity based on impact
        severity = self._determine_severity(impact.total_affected, impact.max_depth)
        
        # Determine recommended action
        metrics = self.estimate_impact(artifact_id)
        if metrics.exceeds_thresholds():
            recommended_action = "rollback"
        elif impact.total_affected > 10:
            recommended_action = "revalidate"
        else:
            recommended_action = "propagate"

        return CascadeScenario(
            trigger=trigger,
            severity=severity,
            source_artifact_id=artifact_id,
            source_provenance_id=record.id,
            source_task_id=record.task_id,
            project_id=record.project_id,
            direct_dependents=impact.direct_dependents,
            transitive_dependents=impact.transitive_dependents,
            total_affected=impact.total_affected,
            max_depth=impact.max_depth,
            reason=reason,
            details=details,
            recommended_action=recommended_action,
        )

    def _determine_severity(
        self,
        affected_count: int,
        max_depth: int,
    ) -> CascadeSeverity:
        """Determine severity based on impact metrics."""
        if affected_count > 50 or max_depth > 5:
            return CascadeSeverity.CRITICAL
        elif affected_count > 20 or max_depth > 3:
            return CascadeSeverity.HIGH
        elif affected_count > 5 or max_depth > 1:
            return CascadeSeverity.MEDIUM
        else:
            return CascadeSeverity.LOW

    def should_pause_execution(self, scenario: CascadeScenario) -> bool:
        """
        Determine if task execution should be paused.
        
        Args:
            scenario: Detected cascade scenario
            
        Returns:
            True if execution should be paused
        """
        # Pause on critical severity
        if scenario.severity == CascadeSeverity.CRITICAL:
            return True
            
        # Pause on security issues
        if scenario.trigger == CascadeTrigger.SECURITY_ISSUE:
            return True
            
        # Pause if recommended action is rollback
        if scenario.recommended_action == "rollback":
            return True
            
        return False

    def get_affected_tasks(self, scenario: CascadeScenario) -> list[str]:
        """
        Get list of task IDs affected by a cascade.
        
        Args:
            scenario: Cascade scenario
            
        Returns:
            List of affected task IDs
        """
        task_ids = set()
        
        for artifact_id in scenario.transitive_dependents:
            prov_id = self.tracker.graph.get_provenance_for_artifact(artifact_id)
            if prov_id:
                record = self.tracker.get_record(prov_id)
                if record:
                    task_ids.add(record.task_id)
                    
        return list(task_ids)