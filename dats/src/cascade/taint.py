"""
Taint propagation for DATS.

Handles the propagation of taint status through the provenance graph,
marking downstream artifacts as suspect and triggering revalidation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.storage.provenance import ProvenanceTracker, TaintEvent
from src.cascade.detector import CascadeScenario

logger = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """Result of taint propagation."""

    cascade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_artifact_id: str = ""
    
    # What was affected
    tainted_count: int = 0
    suspect_count: int = 0
    tainted_artifact_ids: list[str] = field(default_factory=list)
    suspect_artifact_ids: list[str] = field(default_factory=list)
    
    # What needs revalidation
    revalidation_queued: int = 0
    revalidation_artifact_ids: list[str] = field(default_factory=list)
    
    # Depth tracking
    max_depth_reached: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Errors
    errors: list[str] = field(default_factory=list)
    
    # Provenance IDs to invalidate from embeddings
    provenance_ids_to_invalidate: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cascade_id": self.cascade_id,
            "source_artifact_id": self.source_artifact_id,
            "tainted_count": self.tainted_count,
            "suspect_count": self.suspect_count,
            "tainted_artifact_ids": self.tainted_artifact_ids,
            "suspect_artifact_ids": self.suspect_artifact_ids,
            "revalidation_queued": self.revalidation_queued,
            "revalidation_artifact_ids": self.revalidation_artifact_ids,
            "max_depth_reached": self.max_depth_reached,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
            "provenance_ids_to_invalidate": self.provenance_ids_to_invalidate,
        }


class TaintPropagator:
    """
    Propagate taint through the provenance graph.
    
    When an artifact is tainted, this class:
    1. Marks the artifact as tainted in provenance
    2. Finds all downstream dependents
    3. Marks them as "suspect" (not definitively tainted)
    4. Queues revalidation tasks for each suspect
    5. Notifies embedding system to invalidate
    """

    def __init__(
        self,
        provenance_tracker: ProvenanceTracker,
        max_propagation_depth: int = 10,
        pause_on_propagate: bool = True,
    ):
        """
        Initialize taint propagator.
        
        Args:
            provenance_tracker: Tracker for provenance records
            max_propagation_depth: Maximum depth to propagate taint
            pause_on_propagate: Whether to pause new task execution
        """
        self.tracker = provenance_tracker
        self.max_propagation_depth = max_propagation_depth
        self.pause_on_propagate = pause_on_propagate

    def taint_artifact(
        self,
        artifact_id: str,
        reason: str,
        source_id: str = "",
        cascade_id: str = "",
    ) -> PropagationResult:
        """
        Taint an artifact and propagate to dependents.
        
        Args:
            artifact_id: Artifact to taint
            reason: Reason for tainting
            source_id: ID of what caused the taint (if cascade)
            cascade_id: ID grouping related taint events
            
        Returns:
            PropagationResult with details of what was affected
        """
        cascade_id = cascade_id or str(uuid.uuid4())
        result = PropagationResult(
            cascade_id=cascade_id,
            source_artifact_id=artifact_id,
        )

        try:
            # Mark source as tainted
            record = self.tracker.get_producer(artifact_id)
            if not record:
                result.errors.append(f"No provenance record for artifact {artifact_id}")
                return result

            self.tracker.mark_tainted(
                record_id=record.id,
                reason=reason,
                source_id=source_id,
                cascade_id=cascade_id,
                cascade_depth=0,
            )
            result.tainted_count = 1
            result.tainted_artifact_ids.append(artifact_id)
            result.provenance_ids_to_invalidate.append(record.id)

            # Propagate to dependents
            self._propagate_suspect(
                artifact_id=artifact_id,
                cascade_id=cascade_id,
                current_depth=1,
                result=result,
            )

            result.completed_at = datetime.utcnow()
            logger.info(
                f"Taint propagation complete: {result.tainted_count} tainted, "
                f"{result.suspect_count} suspect, cascade_id={cascade_id}"
            )

        except Exception as e:
            logger.error(f"Error during taint propagation: {e}")
            result.errors.append(str(e))

        return result

    def propagate_from_scenario(
        self,
        scenario: CascadeScenario,
    ) -> PropagationResult:
        """
        Propagate taint based on a detected cascade scenario.
        
        Args:
            scenario: Detected cascade scenario
            
        Returns:
            PropagationResult with details
        """
        return self.taint_artifact(
            artifact_id=scenario.source_artifact_id,
            reason=scenario.reason,
            cascade_id=scenario.id,
        )

    def _propagate_suspect(
        self,
        artifact_id: str,
        cascade_id: str,
        current_depth: int,
        result: PropagationResult,
    ):
        """
        Recursively propagate suspect status to dependents.
        
        Args:
            artifact_id: Tainted/suspect artifact to propagate from
            cascade_id: Cascade identifier
            current_depth: Current depth in cascade
            result: Result object to update
        """
        if current_depth > self.max_propagation_depth:
            logger.warning(
                f"Max propagation depth {self.max_propagation_depth} reached, "
                f"stopping at artifact {artifact_id}"
            )
            return

        # Get direct dependents
        dependents = self.tracker.graph.find_dependents(artifact_id, transitive=False)
        
        for dependent_id in dependents:
            # Skip if already processed in this cascade
            if dependent_id in result.tainted_artifact_ids:
                continue
            if dependent_id in result.suspect_artifact_ids:
                continue

            # Get provenance record
            record = self.tracker.get_producer(dependent_id)
            if not record:
                continue

            # Skip if already tainted
            if record.is_tainted():
                continue

            # Mark as suspect
            try:
                self.tracker.mark_suspect(
                    record_id=record.id,
                    source_artifact_id=artifact_id,
                    cascade_id=cascade_id,
                    cascade_depth=current_depth,
                )
                result.suspect_count += 1
                result.suspect_artifact_ids.append(dependent_id)
                result.revalidation_artifact_ids.append(dependent_id)
                result.provenance_ids_to_invalidate.append(record.id)

                if current_depth > result.max_depth_reached:
                    result.max_depth_reached = current_depth

                logger.debug(
                    f"Marked artifact {dependent_id} as suspect "
                    f"(depth={current_depth}, source={artifact_id})"
                )

                # Continue propagation
                self._propagate_suspect(
                    artifact_id=dependent_id,
                    cascade_id=cascade_id,
                    current_depth=current_depth + 1,
                    result=result,
                )

            except Exception as e:
                error_msg = f"Failed to mark {dependent_id} as suspect: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)

    def escalate_suspect_to_taint(
        self,
        artifact_id: str,
        reason: str = "Revalidation confirmed issue",
        cascade_id: str = "",
    ) -> PropagationResult:
        """
        Escalate a suspect artifact to tainted and continue propagation.
        
        Called when revalidation determines the output is invalid.
        
        Args:
            artifact_id: Suspect artifact to escalate
            reason: Reason for escalation
            cascade_id: Existing cascade ID
            
        Returns:
            PropagationResult for continued propagation
        """
        result = PropagationResult(
            cascade_id=cascade_id or str(uuid.uuid4()),
            source_artifact_id=artifact_id,
        )

        record = self.tracker.get_producer(artifact_id)
        if not record:
            result.errors.append(f"No provenance record for {artifact_id}")
            return result

        # Get current cascade depth from taint event
        events = self.tracker.get_taint_events(artifact_id=artifact_id)
        current_depth = 0
        if events:
            # Find the suspect event
            for event in events:
                if event.action == "suspect" and event.cascade_id == cascade_id:
                    current_depth = event.cascade_depth
                    break

        # Mark as tainted (this clears suspect status)
        self.tracker.mark_tainted(
            record_id=record.id,
            reason=reason,
            source_id=record.taint.suspect_source,
            cascade_id=result.cascade_id,
            cascade_depth=current_depth,
        )
        result.tainted_count = 1
        result.tainted_artifact_ids.append(artifact_id)
        result.provenance_ids_to_invalidate.append(record.id)

        # Continue propagation from this point
        self._propagate_suspect(
            artifact_id=artifact_id,
            cascade_id=result.cascade_id,
            current_depth=current_depth + 1,
            result=result,
        )

        result.completed_at = datetime.utcnow()
        return result

    def clear_suspect_status(
        self,
        artifact_id: str,
        reason: str = "Revalidation passed",
    ) -> bool:
        """
        Clear suspect status from an artifact after successful revalidation.
        
        This stops the cascade on this branch - dependents of this artifact
        do not need revalidation since this output was verified as valid.
        
        Args:
            artifact_id: Artifact to clear
            reason: Reason for clearing
            
        Returns:
            True if cleared, False if not found or not suspect
        """
        record = self.tracker.get_producer(artifact_id)
        if not record:
            return False

        if not record.is_suspect():
            return False

        try:
            self.tracker.clear_suspect(record.id, reason)
            logger.info(f"Cleared suspect status from artifact {artifact_id}: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear suspect status: {e}")
            return False

    def get_suspect_artifacts(self, project_id: str) -> list[str]:
        """
        Get all suspect artifacts for a project.
        
        Args:
            project_id: Project to query
            
        Returns:
            List of suspect artifact IDs
        """
        suspect_ids = []
        records = self.tracker.get_by_project(project_id)
        
        for record in records:
            if record.is_suspect():
                suspect_ids.extend(record.get_output_artifact_ids())
                
        return suspect_ids

    def get_cascade_summary(self, cascade_id: str) -> dict[str, Any]:
        """
        Get summary of a cascade propagation.
        
        Args:
            cascade_id: Cascade to summarize
            
        Returns:
            Summary dictionary
        """
        events = self.tracker.get_taint_events(cascade_id=cascade_id)
        
        if not events:
            return {"cascade_id": cascade_id, "found": False}

        tainted = [e for e in events if e.action == "taint"]
        suspect = [e for e in events if e.action == "suspect"]
        cleared = [e for e in events if e.action == "clear"]

        max_depth = max((e.cascade_depth for e in events), default=0)
        start_time = min((e.timestamp for e in events), default=datetime.utcnow())
        end_time = max((e.timestamp for e in events), default=datetime.utcnow())

        return {
            "cascade_id": cascade_id,
            "found": True,
            "total_events": len(events),
            "tainted_count": len(tainted),
            "suspect_count": len(suspect),
            "cleared_count": len(cleared),
            "max_depth": max_depth,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
        }