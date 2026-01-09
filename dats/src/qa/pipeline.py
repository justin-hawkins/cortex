"""
QA Pipeline orchestration for DATS.

Main entry point for running QA validation on worker outputs.
Coordinates validators, aggregates results, and determines verdicts.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from src.qa.results import (
    QAResult,
    QAVerdict,
    QAMetadata,
    ProfileResult,
    ProfileVerdict,
    RevisionGuidance,
    QAIssue,
    IssueSeverity,
    IssueCategory,
)
from src.qa.profiles import (
    QAProfile,
    QAProfileType,
    QAProfileSet,
    ConsensusProfile,
    AdversarialProfile,
    SecurityProfile,
    TestingProfile,
    DocumentationProfile,
    HumanProfile,
    get_profile,
)
from src.qa.validators.base import WorkerOutput, ValidatorContext
from src.qa.validators.consensus import ConsensusValidator
from src.qa.validators.adversarial import AdversarialValidator
from src.qa.validators.security import SecurityValidator
from src.qa.validators.testing import TestingValidator
from src.qa.validators.documentation import DocumentationValidator
from src.qa.human_review import (
    HumanReviewQueue,
    HumanReviewPriority,
    HumanReviewStatus,
)
from src.storage.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


@dataclass
class QAPipelineConfig:
    """Configuration for the QA pipeline."""

    # Severity thresholds
    block_on_critical: bool = True
    block_on_major_count: int = 3

    # Retry policy
    max_revision_attempts: int = 3
    escalate_after_failures: int = 2

    # Human review
    human_review_timeout_hours: int = 24
    human_review_reminder_hours: int = 4
    escalate_on_human_timeout: bool = True

    # General settings
    run_additional_checks_in_parallel: bool = True
    continue_on_validator_error: bool = True


class QAPipeline:
    """
    Main QA Pipeline for validating worker outputs.

    Coordinates multiple validators, handles consensus and human review,
    and produces aggregated QA results.
    """

    def __init__(
        self,
        config: Optional[QAPipelineConfig] = None,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        human_review_queue: Optional[HumanReviewQueue] = None,
    ):
        """
        Initialize the QA pipeline.

        Args:
            config: Pipeline configuration
            provenance_tracker: Optional provenance tracker for recording results
            human_review_queue: Optional human review queue
        """
        self.config = config or QAPipelineConfig()
        self.provenance_tracker = provenance_tracker
        self.human_review_queue = human_review_queue or HumanReviewQueue()

        # Validator factory
        self._validators = {
            QAProfileType.CONSENSUS: ConsensusValidator,
            QAProfileType.ADVERSARIAL: AdversarialValidator,
            QAProfileType.SECURITY: SecurityValidator,
            QAProfileType.TESTING: TestingValidator,
            QAProfileType.DOCUMENTATION: DocumentationValidator,
        }

    async def validate(
        self,
        task_id: str,
        output: WorkerOutput,
        profile: QAProfile | QAProfileType | str,
        additional_checks: list[str] = None,
        context: Optional[ValidatorContext] = None,
    ) -> QAResult:
        """
        Run QA validation on worker output.

        Args:
            task_id: Task ID for the output being validated
            output: Worker output to validate
            profile: Primary QA profile to use
            additional_checks: Additional check types (security, testing, etc.)
            context: Optional validation context

        Returns:
            QAResult with verdict and details
        """
        start_time = datetime.utcnow()

        # Normalize profile
        if isinstance(profile, str):
            profile = get_profile(profile)
        elif isinstance(profile, QAProfileType):
            profile = get_profile(profile)

        # Build context if not provided
        if context is None:
            context = ValidatorContext(
                task_id=task_id,
                project_id=output.metadata.get("project_id", "unknown"),
            )

        # Build profile set
        profile_set = self._build_profile_set(profile, additional_checks or [])

        # Track metadata
        metadata = QAMetadata(started_at=start_time)
        all_profile_results: list[ProfileResult] = []
        all_reviewer_ids: set[str] = set()
        total_tokens = 0

        try:
            # Run validations based on profile type
            if profile.profile_type == QAProfileType.HUMAN:
                # Handle human review specially
                return await self._handle_human_review(
                    output, profile, context, metadata
                )

            # Get ordered profiles to run
            profiles_to_run = profile_set.get_all_profiles()

            # Run profile validations
            if self.config.run_additional_checks_in_parallel:
                results = await self._run_validations_parallel(
                    output, profiles_to_run, context
                )
            else:
                results = await self._run_validations_sequential(
                    output, profiles_to_run, context
                )

            for result in results:
                all_profile_results.append(result)
                all_reviewer_ids.update(result.reviewer_ids)

            # Aggregate results
            verdict = self._determine_verdict(all_profile_results)
            revision_guidance = self._build_revision_guidance(all_profile_results)

            # Build final metadata
            end_time = datetime.utcnow()
            metadata.completed_at = end_time
            metadata.duration_ms = int(
                (end_time - start_time).total_seconds() * 1000
            )
            metadata.reviewer_ids = list(all_reviewer_ids)
            metadata.tokens_consumed = total_tokens

            # Create result based on verdict
            if verdict == QAVerdict.PASS:
                result = QAResult.create_pass(
                    task_id=task_id,
                    profile_results=all_profile_results,
                    metadata=metadata,
                )
            elif verdict == QAVerdict.FAIL:
                result = QAResult.create_fail(
                    task_id=task_id,
                    profile_results=all_profile_results,
                    revision_guidance=revision_guidance,
                    metadata=metadata,
                )
            elif verdict == QAVerdict.NEEDS_REVISION:
                result = QAResult.create_needs_revision(
                    task_id=task_id,
                    profile_results=all_profile_results,
                    revision_guidance=revision_guidance,
                    metadata=metadata,
                )
            elif verdict == QAVerdict.NEEDS_HUMAN:
                result = await self._trigger_human_review(
                    output, all_profile_results, context, metadata
                )
            else:  # ESCALATE
                result = QAResult.create_escalate(
                    task_id=task_id,
                    profile_results=all_profile_results,
                    reason="Persistent validation issues require escalation",
                    metadata=metadata,
                )

            # Update provenance if tracker available
            if self.provenance_tracker:
                self._update_provenance(task_id, result)

            logger.info(
                f"QA validation completed for task {task_id}: {result.verdict.value}"
            )
            return result

        except Exception as e:
            logger.error(f"QA pipeline error for task {task_id}: {e}")

            # Return failure result on error
            metadata.completed_at = datetime.utcnow()
            metadata.duration_ms = int(
                (metadata.completed_at - start_time).total_seconds() * 1000
            )

            error_result = ProfileResult(
                profile="error",
                verdict=ProfileVerdict.FAIL,
                confidence=0.0,
                issues=[
                    QAIssue(
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.CORRECTNESS,
                        description=f"QA pipeline error: {e}",
                    )
                ],
                details={"error": str(e)},
            )

            return QAResult.create_fail(
                task_id=task_id,
                profile_results=[error_result],
                metadata=metadata,
            )

    def _build_profile_set(
        self,
        primary: QAProfile,
        additional_checks: list[str],
    ) -> QAProfileSet:
        """Build profile set from primary profile and additional checks."""
        additional = []
        for check in additional_checks:
            try:
                additional.append(get_profile(check))
            except ValueError:
                logger.warning(f"Unknown additional check type: {check}")

        return QAProfileSet(primary_profile=primary, additional_checks=additional)

    async def _run_validations_parallel(
        self,
        output: WorkerOutput,
        profiles: list[QAProfile],
        context: ValidatorContext,
    ) -> list[ProfileResult]:
        """Run validations in parallel."""
        tasks = []
        for profile in profiles:
            if profile.profile_type == QAProfileType.HUMAN:
                continue  # Human reviews handled separately
            tasks.append(self._run_single_validation(output, profile, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        profile_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validator error: {result}")
                if not self.config.continue_on_validator_error:
                    raise result
                # Create error result
                profile_results.append(
                    ProfileResult(
                        profile=profiles[i].profile_type.value,
                        verdict=ProfileVerdict.FAIL,
                        confidence=0.0,
                        issues=[
                            QAIssue(
                                severity=IssueSeverity.MAJOR,
                                category=IssueCategory.CORRECTNESS,
                                description=f"Validator error: {result}",
                            )
                        ],
                        details={"error": str(result)},
                    )
                )
            else:
                profile_results.append(result)

        return profile_results

    async def _run_validations_sequential(
        self,
        output: WorkerOutput,
        profiles: list[QAProfile],
        context: ValidatorContext,
    ) -> list[ProfileResult]:
        """Run validations sequentially."""
        results = []
        for profile in profiles:
            if profile.profile_type == QAProfileType.HUMAN:
                continue

            try:
                result = await self._run_single_validation(output, profile, context)
                results.append(result)

                # Stop early if critical failure
                if result.has_critical_issues():
                    logger.info("Critical issue found, stopping sequential validation")
                    break

            except Exception as e:
                logger.error(f"Validator error: {e}")
                if not self.config.continue_on_validator_error:
                    raise
                results.append(
                    ProfileResult(
                        profile=profile.profile_type.value,
                        verdict=ProfileVerdict.FAIL,
                        confidence=0.0,
                        issues=[
                            QAIssue(
                                severity=IssueSeverity.MAJOR,
                                category=IssueCategory.CORRECTNESS,
                                description=f"Validator error: {e}",
                            )
                        ],
                        details={"error": str(e)},
                    )
                )

        return results

    async def _run_single_validation(
        self,
        output: WorkerOutput,
        profile: QAProfile,
        context: ValidatorContext,
    ) -> ProfileResult:
        """Run a single validation with appropriate validator."""
        validator_class = self._validators.get(profile.profile_type)
        if not validator_class:
            raise ValueError(f"No validator for profile type: {profile.profile_type}")

        validator = validator_class(profile)
        return await validator.validate(output, context)

    def _determine_verdict(
        self,
        profile_results: list[ProfileResult],
    ) -> QAVerdict:
        """Determine overall verdict from profile results."""
        if not profile_results:
            return QAVerdict.FAIL

        # Collect all issues
        all_issues: list[QAIssue] = []
        for result in profile_results:
            all_issues.extend(result.issues)

        # Count by severity
        critical_count = sum(
            1 for i in all_issues if i.severity == IssueSeverity.CRITICAL
        )
        major_count = sum(
            1 for i in all_issues if i.severity == IssueSeverity.MAJOR
        )

        # Check blocking conditions
        if critical_count > 0 and self.config.block_on_critical:
            return QAVerdict.FAIL

        if major_count >= self.config.block_on_major_count:
            return QAVerdict.NEEDS_REVISION

        # Check profile verdicts
        all_pass = all(r.verdict == ProfileVerdict.PASS for r in profile_results)
        any_fail = any(r.verdict == ProfileVerdict.FAIL for r in profile_results)
        any_partial = any(r.verdict == ProfileVerdict.PARTIAL for r in profile_results)

        if any_fail:
            return QAVerdict.FAIL
        if any_partial:
            # Check if this is a consensus disagreement requiring escalation
            consensus_results = [
                r for r in profile_results if r.profile == "consensus"
            ]
            for cr in consensus_results:
                if not cr.details.get("agreement_reached", True):
                    return QAVerdict.ESCALATE
            return QAVerdict.NEEDS_REVISION
        if all_pass:
            return QAVerdict.PASS

        return QAVerdict.NEEDS_REVISION

    def _build_revision_guidance(
        self,
        profile_results: list[ProfileResult],
    ) -> RevisionGuidance:
        """Build revision guidance from issues."""
        required_changes = []
        suggested_improvements = []
        focus_areas = set()

        for result in profile_results:
            for issue in result.issues:
                if issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.MAJOR):
                    change = issue.description
                    if issue.recommendation:
                        change += f" - {issue.recommendation}"
                    required_changes.append(change)
                    focus_areas.add(issue.category.value)
                elif issue.severity == IssueSeverity.MINOR:
                    suggested_improvements.append(issue.description)
                # Suggestions go to suggested_improvements
                elif issue.severity == IssueSeverity.SUGGESTION:
                    suggested_improvements.append(issue.description)

        return RevisionGuidance(
            required_changes=required_changes,
            suggested_improvements=suggested_improvements,
            focus_areas=list(focus_areas),
        )

    async def _handle_human_review(
        self,
        output: WorkerOutput,
        profile: HumanProfile,
        context: ValidatorContext,
        metadata: QAMetadata,
    ) -> QAResult:
        """Handle outputs that go directly to human review."""
        config = profile.get_validator_config()

        # Create human review request
        priority_map = {
            "low": HumanReviewPriority.LOW,
            "normal": HumanReviewPriority.NORMAL,
            "high": HumanReviewPriority.HIGH,
            "critical": HumanReviewPriority.CRITICAL,
        }
        priority = priority_map.get(
            config.get("priority_level", "normal"),
            HumanReviewPriority.NORMAL,
        )

        request = self.human_review_queue.create_request(
            task_id=context.task_id,
            project_id=context.project_id,
            task_description=context.task_description,
            work_output=output.content,
            acceptance_criteria=context.acceptance_criteria,
            output_type=output.content_type,
            decision_points=config.get("decision_points", []),
            priority=priority,
            timeout_hours=config.get("timeout_hours", 24),
            assigned_reviewers=config.get("specific_reviewers", []),
        )

        metadata.completed_at = datetime.utcnow()
        metadata.duration_ms = int(
            (metadata.completed_at - metadata.started_at).total_seconds() * 1000
        )

        return QAResult.create_needs_human(
            task_id=context.task_id,
            profile_results=[
                ProfileResult(
                    profile="human",
                    verdict=ProfileVerdict.PARTIAL,
                    confidence=0.0,
                    details={"human_review_request_id": request.id},
                )
            ],
            reason="Direct human review requested",
            metadata=metadata,
        )

    async def _trigger_human_review(
        self,
        output: WorkerOutput,
        profile_results: list[ProfileResult],
        context: ValidatorContext,
        metadata: QAMetadata,
    ) -> QAResult:
        """Trigger human review after automated validation."""
        # Prepare automated results summary
        automated_results = {
            "profile_count": len(profile_results),
            "profiles": [
                {
                    "profile": r.profile,
                    "verdict": r.verdict.value,
                    "issue_count": len(r.issues),
                }
                for r in profile_results
            ],
        }

        request = self.human_review_queue.create_request(
            task_id=context.task_id,
            project_id=context.project_id,
            task_description=context.task_description,
            work_output=output.content,
            acceptance_criteria=context.acceptance_criteria,
            output_type=output.content_type,
            automated_qa_results=automated_results,
            priority=HumanReviewPriority.NORMAL,
            timeout_hours=self.config.human_review_timeout_hours,
        )

        metadata.completed_at = datetime.utcnow()
        metadata.duration_ms = int(
            (metadata.completed_at - metadata.started_at).total_seconds() * 1000
        )

        # Add human review to profile results
        profile_results.append(
            ProfileResult(
                profile="human",
                verdict=ProfileVerdict.PARTIAL,
                confidence=0.0,
                details={"human_review_request_id": request.id},
            )
        )

        return QAResult.create_needs_human(
            task_id=context.task_id,
            profile_results=profile_results,
            reason="Automated validation inconclusive, human review required",
            metadata=metadata,
        )

    def _update_provenance(self, task_id: str, result: QAResult):
        """Update provenance with QA results."""
        if not self.provenance_tracker:
            return

        try:
            records = self.provenance_tracker.get_by_task(task_id)
            for record in records:
                status_map = {
                    QAVerdict.PASS: "verified",
                    QAVerdict.FAIL: "failed",
                    QAVerdict.NEEDS_REVISION: "needs_revision",
                    QAVerdict.NEEDS_HUMAN: "pending_human_review",
                    QAVerdict.ESCALATE: "escalated",
                }
                self.provenance_tracker.update_verification(
                    record_id=record.id,
                    status=status_map.get(result.verdict, "unknown"),
                    details=result.to_dict(),
                )
        except Exception as e:
            logger.warning(f"Failed to update provenance: {e}")

    async def check_human_review_status(
        self,
        request_id: str,
    ) -> tuple[bool, Optional[QAResult]]:
        """
        Check status of a human review request.

        Args:
            request_id: Human review request ID

        Returns:
            Tuple of (is_complete, result if complete)
        """
        request = self.human_review_queue.get_request(request_id)
        if not request:
            return False, None

        if request.status == HumanReviewStatus.PENDING:
            # Check for expiration
            if request.is_expired():
                if self.config.escalate_on_human_timeout:
                    request.status = HumanReviewStatus.ESCALATED
                    return True, QAResult.create_escalate(
                        task_id=request.task_id,
                        profile_results=[],
                        reason="Human review timeout",
                    )
                else:
                    return False, None
            return False, None

        # Review is complete
        metadata = QAMetadata(
            started_at=request.created_at,
            completed_at=request.updated_at,
        )

        if request.status == HumanReviewStatus.APPROVED:
            return True, QAResult.create_pass(
                task_id=request.task_id,
                profile_results=[
                    ProfileResult(
                        profile="human",
                        verdict=ProfileVerdict.PASS,
                        confidence=1.0,
                        details={"human_approved": True},
                    )
                ],
                metadata=metadata,
            )
        elif request.status == HumanReviewStatus.REJECTED:
            responses = self.human_review_queue.get_responses(request_id)
            guidance = RevisionGuidance()
            for response in responses:
                guidance.required_changes.extend(response.required_changes)
                guidance.suggested_improvements.extend(response.suggestions)

            return True, QAResult.create_fail(
                task_id=request.task_id,
                profile_results=[
                    ProfileResult(
                        profile="human",
                        verdict=ProfileVerdict.FAIL,
                        confidence=1.0,
                        details={"human_rejected": True},
                    )
                ],
                revision_guidance=guidance,
                metadata=metadata,
            )
        elif request.status == HumanReviewStatus.CHANGES_REQUESTED:
            responses = self.human_review_queue.get_responses(request_id)
            guidance = RevisionGuidance()
            for response in responses:
                guidance.required_changes.extend(response.required_changes)
                guidance.suggested_improvements.extend(response.suggestions)

            return True, QAResult.create_needs_revision(
                task_id=request.task_id,
                profile_results=[
                    ProfileResult(
                        profile="human",
                        verdict=ProfileVerdict.PARTIAL,
                        confidence=1.0,
                        details={"human_changes_requested": True},
                    )
                ],
                revision_guidance=guidance,
                metadata=metadata,
            )
        elif request.status == HumanReviewStatus.ESCALATED:
            return True, QAResult.create_escalate(
                task_id=request.task_id,
                profile_results=[
                    ProfileResult(
                        profile="human",
                        verdict=ProfileVerdict.PARTIAL,
                        confidence=0.5,
                        details={"escalated": True},
                    )
                ],
                reason="Human reviewer escalated",
                metadata=metadata,
            )

        return False, None