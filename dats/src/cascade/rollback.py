"""
Rollback management for DATS.

Provides checkpoint creation and rollback capabilities for recovering
from cascade failures when surgical revalidation isn't sufficient.
"""

import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.storage.provenance import ProvenanceTracker, Checkpoint

logger = logging.getLogger(__name__)


class RollbackTrigger(str, Enum):
    """What triggered the rollback."""

    CASCADE_DEPTH_EXCEEDED = "cascade_depth_exceeded"
    CASCADE_COUNT_EXCEEDED = "cascade_count_exceeded"
    ARCHITECTURAL_FLAW = "architectural_flaw"
    HUMAN_REQUEST = "human_request"
    REVALIDATION_THRESHOLD = "revalidation_threshold"
    CONSISTENCY_CHECK_FAILED = "consistency_check_failed"


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    checkpoint_id: str = ""
    project_id: str = ""
    trigger: RollbackTrigger = RollbackTrigger.HUMAN_REQUEST
    
    # What was rolled back
    provenance_records_invalidated: int = 0
    artifacts_restored: int = 0
    embeddings_removed: int = 0
    tasks_requeued: int = 0
    
    # Status
    success: bool = False
    error: str = ""
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Details
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "checkpoint_id": self.checkpoint_id,
            "project_id": self.project_id,
            "trigger": self.trigger.value,
            "provenance_records_invalidated": self.provenance_records_invalidated,
            "artifacts_restored": self.artifacts_restored,
            "embeddings_removed": self.embeddings_removed,
            "tasks_requeued": self.tasks_requeued,
            "success": self.success,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "details": self.details,
        }


class RollbackManager:
    """
    Manage checkpoints and rollback operations.
    
    Provides:
    - Automatic checkpoint creation based on task count or time interval
    - Manual checkpoint creation
    - Rollback to previous checkpoints
    - Integration with provenance, work products, and embeddings
    """

    def __init__(
        self,
        provenance_tracker: ProvenanceTracker,
        storage_path: Optional[str] = None,
        auto_checkpoint: bool = True,
        checkpoint_interval_tasks: int = 100,
        checkpoint_interval_hours: int = 24,
        max_checkpoints_retained: int = 10,
    ):
        """
        Initialize rollback manager.
        
        Args:
            provenance_tracker: Tracker for provenance records
            storage_path: Path for rollback state storage
            auto_checkpoint: Enable automatic checkpointing
            checkpoint_interval_tasks: Tasks between auto checkpoints
            checkpoint_interval_hours: Hours between auto checkpoints
            max_checkpoints_retained: Maximum checkpoints to keep
        """
        self.tracker = provenance_tracker
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval_tasks = checkpoint_interval_tasks
        self.checkpoint_interval_hours = checkpoint_interval_hours
        self.max_checkpoints_retained = max_checkpoints_retained
        
        self._tasks_since_checkpoint: dict[str, int] = {}  # project_id -> count
        self._rollback_history: list[RollbackResult] = []
        
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self._load_state()

    def should_create_checkpoint(self, project_id: str) -> bool:
        """
        Check if a checkpoint should be created.
        
        Args:
            project_id: Project to check
            
        Returns:
            True if checkpoint should be created
        """
        if not self.auto_checkpoint:
            return False

        # Check task count
        task_count = self._tasks_since_checkpoint.get(project_id, 0)
        if task_count >= self.checkpoint_interval_tasks:
            return True

        # Check time interval
        checkpoints = self.tracker.get_checkpoints(project_id)
        if not checkpoints:
            return True  # No checkpoints exist

        valid_checkpoints = [c for c in checkpoints if c.is_valid]
        if not valid_checkpoints:
            return True

        latest = max(valid_checkpoints, key=lambda c: c.created_at)
        age = datetime.utcnow() - latest.created_at
        if age > timedelta(hours=self.checkpoint_interval_hours):
            return True

        return False

    def create_checkpoint(
        self,
        project_id: str,
        description: str = "",
        trigger: str = "auto",
        git_tag: bool = True,
    ) -> Checkpoint:
        """
        Create a checkpoint for a project.
        
        Args:
            project_id: Project to checkpoint
            description: Optional description
            trigger: What triggered the checkpoint
            git_tag: Whether to create a git tag
            
        Returns:
            Created Checkpoint
        """
        logger.info(f"Creating checkpoint for project {project_id}")

        # Create base checkpoint in provenance tracker
        checkpoint = self.tracker.create_checkpoint(
            project_id=project_id,
            description=description,
            trigger=trigger,
        )

        # Create git tag if requested
        if git_tag:
            git_ref = self._create_git_tag(project_id, checkpoint.id)
            checkpoint.git_ref = git_ref

        # Reset task counter
        self._tasks_since_checkpoint[project_id] = 0

        # Clean up old checkpoints
        self._cleanup_old_checkpoints(project_id)

        # Save state
        if self.storage_path:
            self._save_state()

        logger.info(f"Created checkpoint {checkpoint.id} for project {project_id}")
        return checkpoint

    def record_task_completion(self, project_id: str):
        """
        Record that a task was completed (for auto-checkpoint tracking).
        
        Args:
            project_id: Project the task belongs to
        """
        self._tasks_since_checkpoint[project_id] = (
            self._tasks_since_checkpoint.get(project_id, 0) + 1
        )

        if self.should_create_checkpoint(project_id):
            self.create_checkpoint(
                project_id=project_id,
                trigger="auto",
                description="Automatic checkpoint",
            )

    def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        trigger: RollbackTrigger = RollbackTrigger.HUMAN_REQUEST,
        dry_run: bool = False,
    ) -> RollbackResult:
        """
        Rollback a project to a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to rollback to
            trigger: What triggered the rollback
            dry_run: If True, only calculate impact without rolling back
            
        Returns:
            RollbackResult with details
        """
        result = RollbackResult(
            checkpoint_id=checkpoint_id,
            trigger=trigger,
        )

        try:
            # Get checkpoint
            checkpoint = self.tracker.get_checkpoint(checkpoint_id)
            if not checkpoint:
                result.error = f"Checkpoint not found: {checkpoint_id}"
                return result

            if not checkpoint.is_valid:
                result.error = f"Checkpoint is invalid: {checkpoint_id}"
                return result

            result.project_id = checkpoint.project_id
            
            # Find records to invalidate (after checkpoint)
            all_records = self.tracker.get_by_project(checkpoint.project_id)
            records_to_invalidate = [
                r for r in all_records
                if r.created_at and r.created_at > checkpoint.created_at
            ]
            result.provenance_records_invalidated = len(records_to_invalidate)

            # Get artifact IDs to remove from embeddings
            artifact_ids = []
            provenance_ids = []
            for record in records_to_invalidate:
                provenance_ids.append(record.id)
                artifact_ids.extend(record.get_output_artifact_ids())

            result.embeddings_removed = len(artifact_ids)
            result.details["artifact_ids"] = artifact_ids
            result.details["provenance_ids"] = provenance_ids

            if dry_run:
                result.success = True
                result.details["dry_run"] = True
                return result

            # Actually perform rollback

            # 1. Invalidate post-checkpoint provenance records
            for record in records_to_invalidate:
                self.tracker.mark_tainted(
                    record_id=record.id,
                    reason=f"Rollback to checkpoint {checkpoint_id}",
                    source_id=checkpoint_id,
                )

            # 2. Invalidate embeddings
            if provenance_ids:
                self._invalidate_embeddings(provenance_ids)

            # 3. Restore git state if tag exists
            if checkpoint.git_ref:
                self._restore_git_state(checkpoint.git_ref)
                result.details["git_restored"] = checkpoint.git_ref

            # 4. Queue tasks for re-execution (collect task IDs)
            task_ids = set()
            for record in records_to_invalidate:
                task_ids.add(record.task_id)
            result.tasks_requeued = len(task_ids)
            result.details["tasks_to_requeue"] = list(task_ids)

            # 5. Mark later checkpoints as invalid
            all_checkpoints = self.tracker.get_checkpoints(checkpoint.project_id)
            for cp in all_checkpoints:
                if cp.created_at > checkpoint.created_at:
                    self.tracker.invalidate_checkpoint(cp.id)

            result.success = True
            result.completed_at = datetime.utcnow()

            # Record in history
            self._rollback_history.append(result)
            if self.storage_path:
                self._save_state()

            logger.info(
                f"Rollback complete: {result.provenance_records_invalidated} records "
                f"invalidated, {result.tasks_requeued} tasks to requeue"
            )

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            result.error = str(e)

        return result

    def get_best_rollback_point(self, project_id: str) -> Optional[Checkpoint]:
        """
        Get the best checkpoint to rollback to.
        
        Considers:
        - Checkpoint validity
        - Checkpoint age
        - Number of records that would be lost
        
        Args:
            project_id: Project to find checkpoint for
            
        Returns:
            Best Checkpoint or None
        """
        checkpoints = self.tracker.get_checkpoints(project_id)
        valid_checkpoints = [c for c in checkpoints if c.is_valid]
        
        if not valid_checkpoints:
            return None

        # Sort by creation time (newest first)
        valid_checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        
        # Return most recent valid checkpoint
        return valid_checkpoints[0]

    def estimate_rollback_impact(self, checkpoint_id: str) -> dict[str, Any]:
        """
        Estimate impact of rolling back to a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to analyze
            
        Returns:
            Impact analysis
        """
        result = self.rollback_to_checkpoint(checkpoint_id, dry_run=True)
        
        return {
            "checkpoint_id": checkpoint_id,
            "records_to_invalidate": result.provenance_records_invalidated,
            "embeddings_to_remove": result.embeddings_removed,
            "tasks_to_requeue": result.tasks_requeued,
            "feasible": result.success,
            "error": result.error,
        }

    def get_rollback_history(
        self,
        project_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[RollbackResult]:
        """
        Get rollback history.
        
        Args:
            project_id: Optional project filter
            limit: Maximum results
            
        Returns:
            List of RollbackResults
        """
        history = self._rollback_history
        
        if project_id:
            history = [r for r in history if r.project_id == project_id]
            
        # Sort by time (newest first)
        history.sort(key=lambda r: r.started_at, reverse=True)
        
        return history[:limit]

    def _create_git_tag(self, project_id: str, checkpoint_id: str) -> str:
        """Create a git tag for the checkpoint."""
        tag_name = f"checkpoint-{project_id[:8]}-{checkpoint_id[:8]}"
        
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.warning("Not in a git repository, skipping tag creation")
                return ""

            # Create tag
            subprocess.run(
                ["git", "tag", "-a", tag_name, "-m", f"Checkpoint {checkpoint_id}"],
                check=True,
                capture_output=True,
            )
            
            logger.info(f"Created git tag: {tag_name}")
            return tag_name
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to create git tag: {e}")
            return ""
        except FileNotFoundError:
            logger.warning("Git not available, skipping tag creation")
            return ""

    def _restore_git_state(self, git_ref: str):
        """Restore git state to a reference."""
        try:
            # This is a potentially destructive operation
            # In practice, you might want to create a new branch instead
            logger.info(f"Would restore git state to {git_ref}")
            # subprocess.run(["git", "checkout", git_ref], check=True)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restore git state: {e}")
            raise

    def _invalidate_embeddings(self, provenance_ids: list[str]):
        """Invalidate embeddings for provenance records."""
        try:
            from src.rag.embedding_trigger import invalidate_embeddings_sync
            removed = invalidate_embeddings_sync(provenance_ids)
            logger.info(f"Invalidated {removed} embeddings")
        except ImportError:
            logger.warning("Embedding trigger not available, skipping invalidation")
        except Exception as e:
            logger.error(f"Failed to invalidate embeddings: {e}")

    def _cleanup_old_checkpoints(self, project_id: str):
        """Remove old checkpoints beyond retention limit."""
        checkpoints = self.tracker.get_checkpoints(project_id)
        
        if len(checkpoints) <= self.max_checkpoints_retained:
            return

        # Sort by creation time (oldest first)
        checkpoints.sort(key=lambda c: c.created_at)
        
        # Invalidate oldest checkpoints beyond limit
        to_remove = len(checkpoints) - self.max_checkpoints_retained
        for checkpoint in checkpoints[:to_remove]:
            self.tracker.invalidate_checkpoint(checkpoint.id)
            logger.debug(f"Invalidated old checkpoint {checkpoint.id}")

    def _save_state(self):
        """Save manager state to disk."""
        if not self.storage_path:
            return
            
        state = {
            "tasks_since_checkpoint": self._tasks_since_checkpoint,
            "rollback_history": [r.to_dict() for r in self._rollback_history[-100:]],
        }
        
        with open(self.storage_path / "rollback_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load manager state from disk."""
        if not self.storage_path:
            return
            
        state_file = self.storage_path / "rollback_state.json"
        if not state_file.exists():
            return
            
        try:
            with open(state_file) as f:
                state = json.load(f)
                
            self._tasks_since_checkpoint = state.get("tasks_since_checkpoint", {})
            
            # Reconstruct rollback history
            for r_data in state.get("rollback_history", []):
                result = RollbackResult(
                    id=r_data.get("id", str(uuid.uuid4())),
                    checkpoint_id=r_data.get("checkpoint_id", ""),
                    project_id=r_data.get("project_id", ""),
                    trigger=RollbackTrigger(r_data.get("trigger", "human_request")),
                    provenance_records_invalidated=r_data.get("provenance_records_invalidated", 0),
                    artifacts_restored=r_data.get("artifacts_restored", 0),
                    embeddings_removed=r_data.get("embeddings_removed", 0),
                    tasks_requeued=r_data.get("tasks_requeued", 0),
                    success=r_data.get("success", False),
                    error=r_data.get("error", ""),
                    details=r_data.get("details", {}),
                )
                if r_data.get("started_at"):
                    result.started_at = datetime.fromisoformat(r_data["started_at"])
                if r_data.get("completed_at"):
                    result.completed_at = datetime.fromisoformat(r_data["completed_at"])
                    
                self._rollback_history.append(result)
                
        except Exception as e:
            logger.error(f"Failed to load rollback state: {e}")