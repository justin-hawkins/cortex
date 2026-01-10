"""
Merge Coordinator for DATS.

Main orchestration component that coordinates conflict detection,
classification, strategy selection, and resolution execution.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.merge.classifier import ConflictClassifier
from src.merge.config import MergeConfig, get_merge_config
from src.merge.detector import ConflictDetector, TaskOutput, InFlightTask
from src.merge.models import (
    ConflictAudit,
    ConflictBatch,
    ConflictRecord,
    ConflictResolution,
    ResolutionStatus,
    ResolutionStrategy,
)
from src.merge.resolvers.base import BaseResolver, ResolverResult
from src.merge.resolvers.textual import TextualResolver
from src.merge.resolvers.semantic import SemanticResolver
from src.merge.resolvers.architectural import ArchitecturalResolver
from src.merge.strategies import ResolutionStrategySelector, StrategyRecommendation
from src.storage.provenance import ProvenanceTracker
from src.storage.work_product import WorkProductStore

logger = logging.getLogger(__name__)


@dataclass
class ResolutionOutcome:
    """Outcome of a conflict resolution attempt."""

    conflict_id: str
    success: bool
    resolution: Optional[ConflictResolution] = None
    error: Optional[str] = None
    escalated: bool = False
    escalation_target: Optional[str] = None
    attempts: int = 1
    total_tokens: int = 0
    total_duration_ms: int = 0


@dataclass
class BatchResolutionOutcome:
    """Outcome of batch conflict resolution."""

    batch_id: str
    outcomes: list[ResolutionOutcome] = field(default_factory=list)
    total_conflicts: int = 0
    resolved_count: int = 0
    escalated_count: int = 0
    failed_count: int = 0


class MergeCoordinator:
    """
    Main orchestration component for merge coordination.

    Coordinates:
    - Conflict detection when tasks complete
    - Classification of detected conflicts
    - Strategy selection based on classification
    - Execution of appropriate resolver
    - Escalation when needed
    - Recording of resolution outcomes

    Supports batching of conflicts that occur within a time window.
    """

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        detector: Optional[ConflictDetector] = None,
        classifier: Optional[ConflictClassifier] = None,
        strategy_selector: Optional[ResolutionStrategySelector] = None,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        work_product_store: Optional[WorkProductStore] = None,
    ):
        """
        Initialize merge coordinator.

        Args:
            config: Merge configuration
            detector: Conflict detector
            classifier: Conflict classifier
            strategy_selector: Strategy selector
            provenance_tracker: For lineage tracking
            work_product_store: For artifact storage
        """
        self.config = config or get_merge_config()

        # Core components
        self.detector = detector or ConflictDetector(
            config=self.config,
            provenance_tracker=provenance_tracker,
            work_product_store=work_product_store,
        )
        self.classifier = classifier or ConflictClassifier(config=self.config)
        self.strategy_selector = strategy_selector or ResolutionStrategySelector(
            config=self.config
        )

        # Storage
        self.provenance = provenance_tracker
        self.work_products = work_product_store

        # Resolvers
        self._resolvers: dict[str, BaseResolver] = {}
        self._init_resolvers()

        # Batching
        self._pending_conflicts: list[ConflictRecord] = []
        self._batch_timer: Optional[asyncio.Task] = None
        self._batch_lock = asyncio.Lock()

        # Conflict storage (in-memory for now)
        self._conflicts: dict[str, ConflictRecord] = {}

    def _init_resolvers(self):
        """Initialize available resolvers."""
        self._resolvers = {
            "textual": TextualResolver(config=self.config),
            "semantic": SemanticResolver(config=self.config),
            "architectural": ArchitecturalResolver(config=self.config),
        }

    def get_resolver(self, strategy: ResolutionStrategy) -> BaseResolver:
        """
        Get resolver for a strategy.

        Args:
            strategy: Resolution strategy

        Returns:
            Appropriate resolver
        """
        if strategy == ResolutionStrategy.AUTO_MERGE:
            return self._resolvers["textual"]
        elif strategy == ResolutionStrategy.SEMANTIC_MERGE:
            return self._resolvers["semantic"]
        elif strategy == ResolutionStrategy.REDESIGN:
            return self._resolvers["architectural"]
        elif strategy == ResolutionStrategy.HUMAN_DECISION:
            return self._resolvers["architectural"]
        else:
            return self._resolvers["semantic"]  # Default

    async def on_task_complete(
        self,
        task_output: TaskOutput,
    ) -> list[ConflictRecord]:
        """
        Handle task completion - main entry point.

        Detects conflicts with other recently completed tasks.

        Args:
            task_output: Output from completed task

        Returns:
            List of detected conflicts
        """
        if not self.config.detection.check_on_task_complete:
            return []

        # Detect conflicts
        conflicts = self.detector.check_on_task_complete(task_output)

        if not conflicts:
            return []

        logger.info(
            f"Detected {len(conflicts)} conflict(s) for task {task_output.task_id}"
        )

        # Add to pending batch
        async with self._batch_lock:
            for conflict in conflicts:
                # Store conflict
                self._conflicts[conflict.id] = conflict
                self._pending_conflicts.append(conflict)

            # Start batch timer if not running
            if self._batch_timer is None or self._batch_timer.done():
                batch_window = self.config.batching.get_window_for_project(
                    task_output.project_id
                )
                self._batch_timer = asyncio.create_task(
                    self._batch_timer_task(batch_window)
                )

        return conflicts

    async def _batch_timer_task(self, window_seconds: float):
        """
        Timer task for batch processing.

        Waits for the batch window then processes pending conflicts.

        Args:
            window_seconds: How long to wait
        """
        await asyncio.sleep(window_seconds)

        async with self._batch_lock:
            if self._pending_conflicts:
                batch = self._pending_conflicts.copy()
                self._pending_conflicts = []
            else:
                batch = []

        if batch:
            await self.process_batch(batch)

    async def process_batch(
        self,
        conflicts: list[ConflictRecord],
    ) -> BatchResolutionOutcome:
        """
        Process a batch of conflicts.

        Args:
            conflicts: Conflicts to process

        Returns:
            BatchResolutionOutcome with results
        """
        if not conflicts:
            return BatchResolutionOutcome(
                batch_id="empty",
                total_conflicts=0,
            )

        # Group by project
        project_id = conflicts[0].project_id if conflicts else ""
        batch = ConflictBatch(
            conflicts=conflicts,
            project_id=project_id,
            window_seconds=self.config.batching.get_window_for_project(project_id),
        )

        logger.info(f"Processing batch {batch.id} with {len(conflicts)} conflict(s)")

        outcomes = []
        for conflict in conflicts:
            outcome = await self.resolve_conflict(conflict)
            outcomes.append(outcome)

        # Summarize
        resolved = sum(1 for o in outcomes if o.success and not o.escalated)
        escalated = sum(1 for o in outcomes if o.escalated)
        failed = sum(1 for o in outcomes if not o.success and not o.escalated)

        return BatchResolutionOutcome(
            batch_id=batch.id,
            outcomes=outcomes,
            total_conflicts=len(conflicts),
            resolved_count=resolved,
            escalated_count=escalated,
            failed_count=failed,
        )

    async def resolve_conflict(
        self,
        conflict: ConflictRecord,
        attempt: int = 1,
    ) -> ResolutionOutcome:
        """
        Resolve a single conflict.

        Full resolution flow:
        1. Classify conflict
        2. Select strategy
        3. Execute resolver
        4. Handle escalation if needed

        Args:
            conflict: Conflict to resolve
            attempt: Which attempt this is

        Returns:
            ResolutionOutcome with result
        """
        started_at = datetime.utcnow()
        total_tokens = 0

        try:
            # Step 1: Classify
            if not conflict.classification or attempt > 1:
                classification_result = await self.classifier.classify(conflict)
                conflict.classification = classification_result.classification
                total_tokens += classification_result.tokens_input + classification_result.tokens_output

                # Update audit
                if not conflict.audit:
                    conflict.audit = ConflictAudit(
                        detection_method="batch_check" if attempt == 1 else "reclassification"
                    )
                conflict.audit.tokens_consumed += total_tokens

            # Step 2: Select strategy
            recommendation = self.strategy_selector.select_strategy(
                conflict=conflict,
                classification=conflict.classification,
                attempt_number=attempt,
            )

            logger.info(
                f"Conflict {conflict.id}: strategy={recommendation.strategy.value}, "
                f"confidence={recommendation.confidence:.2f}"
            )

            # Step 3: Execute resolver
            resolver = self.get_resolver(recommendation.strategy)
            result = await resolver.resolve(conflict)

            total_tokens += result.tokens_consumed
            if conflict.audit:
                conflict.audit.tokens_consumed = total_tokens
                conflict.audit.resolution_attempts = attempt

            # Step 4: Handle result
            if result.success:
                conflict.resolution = result.resolution
                self._conflicts[conflict.id] = conflict

                duration = datetime.utcnow() - started_at
                return ResolutionOutcome(
                    conflict_id=conflict.id,
                    success=True,
                    resolution=result.resolution,
                    attempts=attempt,
                    total_tokens=total_tokens,
                    total_duration_ms=int(duration.total_seconds() * 1000),
                    escalated=result.resolution.status == ResolutionStatus.ESCALATED
                    if result.resolution
                    else False,
                    escalation_target=recommendation.escalation_target
                    if result.needs_escalation
                    else None,
                )

            # Handle failure / escalation
            if result.needs_escalation:
                return await self._handle_escalation(
                    conflict=conflict,
                    recommendation=recommendation,
                    result=result,
                    attempt=attempt,
                    started_at=started_at,
                    tokens_so_far=total_tokens,
                )

            # Failed without escalation
            duration = datetime.utcnow() - started_at
            return ResolutionOutcome(
                conflict_id=conflict.id,
                success=False,
                error=result.error,
                attempts=attempt,
                total_tokens=total_tokens,
                total_duration_ms=int(duration.total_seconds() * 1000),
            )

        except Exception as e:
            logger.error(f"Resolution failed for conflict {conflict.id}: {e}")
            duration = datetime.utcnow() - started_at
            return ResolutionOutcome(
                conflict_id=conflict.id,
                success=False,
                error=str(e),
                attempts=attempt,
                total_tokens=total_tokens,
                total_duration_ms=int(duration.total_seconds() * 1000),
            )

    async def _handle_escalation(
        self,
        conflict: ConflictRecord,
        recommendation: StrategyRecommendation,
        result: ResolverResult,
        attempt: int,
        started_at: datetime,
        tokens_so_far: int,
    ) -> ResolutionOutcome:
        """
        Handle escalation when resolution needs it.

        Args:
            conflict: Conflict being resolved
            recommendation: Strategy recommendation
            result: Resolver result
            attempt: Current attempt number
            started_at: When resolution started
            tokens_so_far: Tokens consumed so far

        Returns:
            ResolutionOutcome after escalation handling
        """
        max_attempts = self.config.escalation.max_resolution_attempts

        if recommendation.escalation_target == "larger_model" and attempt < max_attempts:
            # Retry with larger model
            logger.info(
                f"Escalating conflict {conflict.id} to larger model (attempt {attempt + 1})"
            )

            # Reclassify with escalation
            if conflict.classification:
                new_classification = await self.classifier.reclassify_with_escalation(
                    conflict, conflict.classification
                )
                conflict.classification = new_classification.classification

            return await self.resolve_conflict(conflict, attempt + 1)

        elif recommendation.escalation_target == "human":
            # Escalate to human
            logger.info(f"Escalating conflict {conflict.id} to human decision")

            # Use architectural resolver to generate human decision request
            arch_resolver = self._resolvers["architectural"]
            human_result = await arch_resolver.resolve(conflict)

            conflict.resolution = human_result.resolution
            self._conflicts[conflict.id] = conflict

            duration = datetime.utcnow() - started_at
            return ResolutionOutcome(
                conflict_id=conflict.id,
                success=True,  # Successfully escalated
                resolution=human_result.resolution,
                escalated=True,
                escalation_target="human",
                attempts=attempt,
                total_tokens=tokens_so_far + human_result.tokens_consumed,
                total_duration_ms=int(duration.total_seconds() * 1000),
            )

        else:
            # Max attempts exceeded or unknown escalation
            duration = datetime.utcnow() - started_at
            return ResolutionOutcome(
                conflict_id=conflict.id,
                success=False,
                error=result.error or "Max resolution attempts exceeded",
                attempts=attempt,
                total_tokens=tokens_so_far,
                total_duration_ms=int(duration.total_seconds() * 1000),
            )

    def register_in_flight(self, task: InFlightTask):
        """
        Register a task as in-flight for overlap detection.

        Args:
            task: Task starting execution
        """
        self.detector.register_in_flight(task)

    def unregister_in_flight(self, task_id: str):
        """
        Unregister a task from in-flight tracking.

        Args:
            task_id: Task that has completed
        """
        self.detector.unregister_in_flight(task_id)

    def get_conflict(self, conflict_id: str) -> Optional[ConflictRecord]:
        """
        Get a conflict record by ID.

        Args:
            conflict_id: Conflict ID

        Returns:
            ConflictRecord if found
        """
        return self._conflicts.get(conflict_id)

    def get_pending_conflicts(self, project_id: Optional[str] = None) -> list[ConflictRecord]:
        """
        Get pending (unresolved) conflicts.

        Args:
            project_id: Optional filter by project

        Returns:
            List of pending conflicts
        """
        pending = [
            c for c in self._conflicts.values()
            if not c.is_resolved() and not c.is_escalated()
        ]

        if project_id:
            pending = [c for c in pending if c.project_id == project_id]

        return pending

    def get_escalated_conflicts(self, project_id: Optional[str] = None) -> list[ConflictRecord]:
        """
        Get escalated conflicts awaiting human decision.

        Args:
            project_id: Optional filter by project

        Returns:
            List of escalated conflicts
        """
        escalated = [c for c in self._conflicts.values() if c.is_escalated()]

        if project_id:
            escalated = [c for c in escalated if c.project_id == project_id]

        return escalated

    async def apply_human_decision(
        self,
        conflict_id: str,
        decision: str,
        decision_maker: str,
    ) -> ResolutionOutcome:
        """
        Apply a human decision to an escalated conflict.

        Args:
            conflict_id: Conflict that was escalated
            decision: Human's decision
            decision_maker: Who made the decision

        Returns:
            ResolutionOutcome after applying decision
        """
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return ResolutionOutcome(
                conflict_id=conflict_id,
                success=False,
                error=f"Conflict {conflict_id} not found",
            )

        if not conflict.is_escalated():
            return ResolutionOutcome(
                conflict_id=conflict_id,
                success=False,
                error=f"Conflict {conflict_id} is not escalated",
            )

        # Update resolution with human decision
        if conflict.resolution:
            conflict.resolution.status = ResolutionStatus.RESOLVED
            conflict.resolution.resolved_by = decision_maker
            conflict.resolution.completed_at = datetime.utcnow()

            if conflict.resolution.human_decision_request:
                conflict.resolution.human_decision_request.recommendation = decision

        self._conflicts[conflict_id] = conflict

        return ResolutionOutcome(
            conflict_id=conflict_id,
            success=True,
            resolution=conflict.resolution,
        )

    async def close(self):
        """Clean up resources."""
        # Cancel batch timer
        if self._batch_timer and not self._batch_timer.done():
            self._batch_timer.cancel()
            try:
                await self._batch_timer
            except asyncio.CancelledError:
                pass

        # Close resolvers
        for resolver in self._resolvers.values():
            if hasattr(resolver, "close"):
                await resolver.close()

        # Close classifier
        if hasattr(self.classifier, "close"):
            await self.classifier.close()