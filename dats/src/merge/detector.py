"""
Conflict detection for merge coordination.

Provides mechanisms to detect conflicts between parallel task outputs
at file, dependency, and semantic levels.
"""

import difflib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.merge.config import MergeConfig, get_merge_config
from src.merge.models import (
    AffectedArtifact,
    ConflictAudit,
    ConflictRecord,
    ConflictRegion,
    InvolvedTask,
)
from src.storage.provenance import ProvenanceTracker
from src.storage.work_product import WorkProductStore, Artifact

logger = logging.getLogger(__name__)


@dataclass
class TaskOutput:
    """Represents a completed task output for conflict detection."""

    task_id: str
    project_id: str
    completed_at: datetime
    artifacts: list[Artifact] = field(default_factory=list)
    provenance_id: Optional[str] = None
    description: Optional[str] = None
    parent_task_id: Optional[str] = None


@dataclass
class InFlightTask:
    """Represents a task currently being executed."""

    task_id: str
    project_id: str
    started_at: datetime
    target_artifacts: list[str] = field(default_factory=list)  # Paths being modified
    description: Optional[str] = None


class ConflictDetector:
    """
    Detects conflicts between parallel task outputs.

    Supports three detection methods:
    1. File-level: Same file modified by multiple tasks
    2. Dependency-level: Task DAG analysis for divergent outputs
    3. Semantic-level: Conceptual overlap via RAG embeddings
    """

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        work_product_store: Optional[WorkProductStore] = None,
        embedding_client: Optional[Any] = None,  # EmbeddingClient
        vector_store: Optional[Any] = None,  # FileVectorStore
    ):
        """
        Initialize conflict detector.

        Args:
            config: Merge configuration
            provenance_tracker: Provenance tracking for lineage analysis
            work_product_store: Artifact storage
            embedding_client: For semantic similarity (optional)
            vector_store: For semantic search (optional)
        """
        self.config = config or get_merge_config()
        self.provenance = provenance_tracker
        self.work_products = work_product_store
        self.embedding_client = embedding_client
        self.vector_store = vector_store

        # Track recent completions for batching
        self._recent_completions: list[TaskOutput] = []

        # Track in-flight tasks
        self._in_flight: dict[str, InFlightTask] = {}

    def register_in_flight(self, task: InFlightTask):
        """
        Register a task as in-flight for overlap detection.

        Args:
            task: Task that is starting execution
        """
        self._in_flight[task.task_id] = task
        logger.debug(f"Registered in-flight task: {task.task_id}")

    def unregister_in_flight(self, task_id: str):
        """
        Remove a task from in-flight tracking.

        Args:
            task_id: Task that has completed
        """
        if task_id in self._in_flight:
            del self._in_flight[task_id]
            logger.debug(f"Unregistered in-flight task: {task_id}")

    def check_on_task_complete(
        self,
        completed_task: TaskOutput,
        recent_completions: Optional[list[TaskOutput]] = None,
    ) -> list[ConflictRecord]:
        """
        Check for conflicts when a task completes.

        This is the main entry point for conflict detection.

        Args:
            completed_task: The task that just completed
            recent_completions: Other recently completed tasks to check against

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Use provided completions or internal tracking
        check_against = recent_completions or self._recent_completions

        # Filter to same project
        same_project = [
            t for t in check_against
            if t.project_id == completed_task.project_id
            and t.task_id != completed_task.task_id
        ]

        # 1. File-level detection
        file_conflicts = self._detect_file_conflicts(completed_task, same_project)
        conflicts.extend(file_conflicts)

        # 2. Dependency-level detection (if provenance available)
        if self.provenance:
            dep_conflicts = self._detect_dependency_conflicts(
                completed_task, same_project
            )
            conflicts.extend(dep_conflicts)

        # 3. Semantic-level detection (if RAG available)
        if self.embedding_client and self.vector_store:
            sem_conflicts = self._detect_semantic_conflicts(
                completed_task, same_project
            )
            conflicts.extend(sem_conflicts)

        # 4. Check in-flight overlap
        if self.config.detection.check_in_flight_overlap:
            in_flight_conflicts = self._check_in_flight_overlap(completed_task)
            # These are warnings, not conflicts yet
            for warning in in_flight_conflicts:
                logger.warning(
                    f"Potential conflict: Task {completed_task.task_id} "
                    f"overlaps with in-flight task {warning}"
                )

        # Track this completion
        self._recent_completions.append(completed_task)
        self._cleanup_old_completions()

        # Unregister from in-flight
        self.unregister_in_flight(completed_task.task_id)

        return conflicts

    def _detect_file_conflicts(
        self,
        completed_task: TaskOutput,
        other_tasks: list[TaskOutput],
    ) -> list[ConflictRecord]:
        """
        Detect file-level conflicts (same file modified by multiple tasks).

        Args:
            completed_task: Task that just completed
            other_tasks: Other tasks to compare against

        Returns:
            List of file-level conflicts
        """
        conflicts = []

        # Get paths from completed task
        completed_paths = {
            a.path: a for a in completed_task.artifacts if a.path
        }

        for other_task in other_tasks:
            other_paths = {
                a.path: a for a in other_task.artifacts if a.path
            }

            # Find overlapping paths
            overlapping = set(completed_paths.keys()) & set(other_paths.keys())

            if not overlapping:
                continue

            # Create conflict record for each overlapping file
            affected_artifacts = []
            for path in overlapping:
                our_artifact = completed_paths[path]
                their_artifact = other_paths[path]

                # Compute conflict regions
                regions = self._compute_conflict_regions(
                    our_artifact.content,
                    their_artifact.content,
                )

                affected_artifacts.append(
                    AffectedArtifact(
                        artifact_id=our_artifact.id,
                        location=path,
                        artifact_type=our_artifact.type,
                        conflict_regions=regions,
                        ours_checksum=our_artifact.checksum,
                        theirs_checksum=their_artifact.checksum,
                    )
                )

            if affected_artifacts:
                conflict = ConflictRecord(
                    project_id=completed_task.project_id,
                    involved_tasks=[
                        InvolvedTask(
                            task_id=completed_task.task_id,
                            output_artifact_id=completed_task.artifacts[0].id
                            if completed_task.artifacts
                            else "",
                            completed_at=completed_task.completed_at,
                            description=completed_task.description,
                        ),
                        InvolvedTask(
                            task_id=other_task.task_id,
                            output_artifact_id=other_task.artifacts[0].id
                            if other_task.artifacts
                            else "",
                            completed_at=other_task.completed_at,
                            description=other_task.description,
                        ),
                    ],
                    affected_artifacts=affected_artifacts,
                    audit=ConflictAudit(detection_method="file_overlap"),
                )
                conflicts.append(conflict)
                logger.info(
                    f"Detected file conflict between tasks "
                    f"{completed_task.task_id} and {other_task.task_id}: "
                    f"{len(overlapping)} files"
                )

        return conflicts

    def _compute_conflict_regions(
        self,
        ours: str,
        theirs: str,
        base: str = "",
    ) -> list[ConflictRegion]:
        """
        Compute specific conflict regions between two file versions.

        Uses difflib to find overlapping changes.

        Args:
            ours: Our version content
            theirs: Their version content
            base: Optional base version

        Returns:
            List of ConflictRegion objects
        """
        regions = []

        ours_lines = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, ours_lines, theirs_lines)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                # Both sides changed this region - definite conflict
                regions.append(
                    ConflictRegion(
                        start_line=i1 + 1,
                        end_line=i2,
                        ours_content="".join(ours_lines[i1:i2]),
                        theirs_content="".join(theirs_lines[j1:j2]),
                        description="Both sides modified this region",
                    )
                )
            elif tag == "delete" or tag == "insert":
                # One side modified - might conflict if same logical region
                # For now, we note it but don't mark as conflict
                pass

        return regions

    def _detect_dependency_conflicts(
        self,
        completed_task: TaskOutput,
        other_tasks: list[TaskOutput],
    ) -> list[ConflictRecord]:
        """
        Detect dependency-level conflicts via task DAG analysis.

        Looks for sibling tasks that consume same inputs but produce
        divergent outputs.

        Args:
            completed_task: Task that just completed
            other_tasks: Other tasks to compare

        Returns:
            List of dependency conflicts
        """
        conflicts = []

        if not self.provenance:
            return conflicts

        # Get provenance for completed task
        completed_provenance = self.provenance.get_by_task(completed_task.task_id)
        if not completed_provenance:
            return conflicts

        # Get inputs consumed by completed task
        completed_inputs = set()
        for record in completed_provenance:
            completed_inputs.update(record.inputs_consumed)

        for other_task in other_tasks:
            other_provenance = self.provenance.get_by_task(other_task.task_id)
            if not other_provenance:
                continue

            # Get inputs consumed by other task
            other_inputs = set()
            for record in other_provenance:
                other_inputs.update(record.inputs_consumed)

            # Check for shared inputs
            shared_inputs = completed_inputs & other_inputs
            if not shared_inputs:
                continue

            # Check if both tasks have same parent
            same_parent = (
                completed_task.parent_task_id
                and completed_task.parent_task_id == other_task.parent_task_id
            )

            if same_parent or shared_inputs:
                # Potential sibling conflict - need semantic analysis
                # to determine if outputs are actually divergent
                # For now, just log it
                logger.debug(
                    f"Tasks {completed_task.task_id} and {other_task.task_id} "
                    f"share inputs: {shared_inputs}"
                )

        return conflicts

    def _detect_semantic_conflicts(
        self,
        completed_task: TaskOutput,
        other_tasks: list[TaskOutput],
    ) -> list[ConflictRecord]:
        """
        Detect semantic-level conflicts via RAG similarity.

        Uses embeddings to find conceptual overlap between outputs.

        Args:
            completed_task: Task that just completed
            other_tasks: Other tasks to compare

        Returns:
            List of semantic conflicts
        """
        conflicts = []

        if not self.embedding_client or not self.vector_store:
            return conflicts

        # This would be async in production - placeholder for sync version
        # Will be implemented properly in the coordinator which handles async

        return conflicts

    async def detect_semantic_conflicts_async(
        self,
        completed_task: TaskOutput,
        other_tasks: list[TaskOutput],
    ) -> list[ConflictRecord]:
        """
        Async version of semantic conflict detection.

        Args:
            completed_task: Task that just completed
            other_tasks: Other tasks to compare

        Returns:
            List of semantic conflicts
        """
        conflicts = []

        if not self.embedding_client or not self.vector_store:
            return conflicts

        threshold = self.config.detection.semantic_similarity_threshold

        for artifact in completed_task.artifacts:
            if not artifact.content:
                continue

            # Generate embedding for this artifact
            try:
                embedding = await self.embedding_client.embed(artifact.content)
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
                continue

            # Search for similar content
            results = self.vector_store.search(
                query_embedding=embedding,
                k=5,
                project_id=completed_task.project_id,
            )

            for result in results:
                if result.score < threshold:
                    continue

                # Check if this is from a different task
                if result.document.task_id == completed_task.task_id:
                    continue

                # Found semantic overlap
                logger.info(
                    f"Semantic overlap detected: {artifact.path} "
                    f"similar to {result.document.id} (score: {result.score:.3f})"
                )

                # Create conflict record if not already tracking
                conflict = ConflictRecord(
                    project_id=completed_task.project_id,
                    involved_tasks=[
                        InvolvedTask(
                            task_id=completed_task.task_id,
                            output_artifact_id=artifact.id,
                            completed_at=completed_task.completed_at,
                        ),
                        InvolvedTask(
                            task_id=result.document.task_id or "",
                            output_artifact_id=result.document.id,
                        ),
                    ],
                    audit=ConflictAudit(
                        detection_method=f"semantic_similarity:{result.score:.3f}"
                    ),
                )
                conflicts.append(conflict)

        return conflicts

    def _check_in_flight_overlap(
        self,
        completed_task: TaskOutput,
    ) -> list[str]:
        """
        Check if completed task overlaps with in-flight tasks.

        Returns list of in-flight task IDs that might conflict.

        Args:
            completed_task: Task that just completed

        Returns:
            List of potentially conflicting in-flight task IDs
        """
        warnings = []

        completed_paths = {a.path for a in completed_task.artifacts if a.path}

        for task_id, in_flight in self._in_flight.items():
            if in_flight.project_id != completed_task.project_id:
                continue

            # Check for path overlap
            in_flight_paths = set(in_flight.target_artifacts)
            overlap = completed_paths & in_flight_paths

            if overlap:
                warnings.append(task_id)

        return warnings

    def _cleanup_old_completions(self):
        """Remove old completions outside the batch window."""
        if not self._recent_completions:
            return

        now = datetime.utcnow()
        cutoff_seconds = self.config.batching.default_batch_window_seconds * 2

        self._recent_completions = [
            t for t in self._recent_completions
            if (now - t.completed_at).total_seconds() < cutoff_seconds
        ]

    def find_common_parent(
        self,
        task_ids: list[str],
    ) -> Optional[str]:
        """
        Find common parent task for a set of tasks.

        Used to provide context for conflict resolution.

        Args:
            task_ids: Task IDs to find common parent for

        Returns:
            Common parent task ID if found
        """
        if not self.provenance or len(task_ids) < 2:
            return None

        # Get parent chains for each task
        parent_chains: list[set[str]] = []
        for task_id in task_ids:
            records = self.provenance.get_by_task(task_id)
            if not records:
                continue

            # Follow inputs to find parents
            parents = set()
            for record in records:
                for input_id in record.inputs_consumed:
                    # Get the task that produced this input
                    for potential in self.provenance._records.values():
                        for output in potential.outputs:
                            if output.get("artifact_id") == input_id:
                                parents.add(potential.task_id)

            parent_chains.append(parents)

        if len(parent_chains) < 2:
            return None

        # Find intersection of all parent chains
        common = parent_chains[0]
        for chain in parent_chains[1:]:
            common = common & chain

        return next(iter(common)) if common else None