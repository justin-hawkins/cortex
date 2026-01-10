"""
Consensus validator for DATS QA.

Implements multi-model consensus validation where multiple reviewers
must agree on the verdict for an output to pass.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from src.qa.validators.base import (
    BaseValidator,
    ValidatorContext,
    WorkerOutput,
)
from src.qa.results import (
    ProfileResult,
    ProfileVerdict,
    QAIssue,
    IssueSeverity,
    IssueCategory,
)
from src.qa.profiles import ConsensusProfile

logger = logging.getLogger(__name__)


class ConsensusValidator(BaseValidator):
    """
    Consensus-based validator.

    Requires multiple reviewers (preferably different models) to agree
    on the validation verdict. Handles disagreements through additional
    reviewers or escalation.
    """

    validator_name = "consensus"

    def __init__(
        self,
        profile: ConsensusProfile,
        model_client=None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize consensus validator.

        Args:
            profile: Consensus profile configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override
        """
        super().__init__(profile, model_client, model_tier)
        self.consensus_profile = profile

    async def validate(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
    ) -> ProfileResult:
        """
        Validate output using consensus from multiple reviewers.

        Args:
            output: Worker output to validate
            context: Validation context

        Returns:
            ProfileResult with consensus verdict
        """
        start_time = datetime.utcnow()
        config = self.consensus_profile.get_validator_config()

        min_reviewers = config.get("min_reviewers", 2)
        max_reviewers = config.get("max_reviewers", 3)
        agreement_threshold = config.get("agreement_threshold", 1.0)

        # Get available models for diverse reviews
        available_models = self._get_available_models()
        if len(available_models) < min_reviewers:
            # If not enough diverse models, we'll use the same model multiple times
            logger.warning(
                f"Only {len(available_models)} models available, "
                f"need {min_reviewers} for consensus"
            )

        # Run initial reviews
        reviews = await self._run_reviews(
            output,
            context,
            available_models[:min_reviewers],
        )

        # Check for agreement
        verdict, confidence, needs_escalation = self._calculate_consensus(
            reviews, agreement_threshold
        )

        # Handle disagreement
        if needs_escalation and config.get("add_reviewer_on_disagreement", True):
            if len(reviews) < max_reviewers and len(available_models) > len(reviews):
                # Add another reviewer
                logger.info("Disagreement detected, adding additional reviewer")
                additional_reviews = await self._run_reviews(
                    output,
                    context,
                    available_models[len(reviews) : len(reviews) + 1],
                )
                reviews.extend(additional_reviews)

                # Recalculate consensus
                verdict, confidence, needs_escalation = self._calculate_consensus(
                    reviews, agreement_threshold
                )

        # Aggregate issues from all reviews
        all_issues = []
        reviewer_ids = []
        for review in reviews:
            all_issues.extend(review["issues"])
            reviewer_ids.append(review["reviewer_id"])

        # Deduplicate similar issues
        unique_issues = self._deduplicate_issues(all_issues)

        # Determine final verdict
        final_verdict = verdict
        if needs_escalation and config.get(
            "escalate_on_persistent_disagreement", True
        ):
            # Mark as partial to trigger escalation
            final_verdict = ProfileVerdict.PARTIAL

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return self._create_result(
            verdict=final_verdict,
            issues=unique_issues,
            confidence=confidence,
            details={
                "reviewer_count": len(reviews),
                "agreement_reached": not needs_escalation,
                "individual_verdicts": [r["verdict"].value for r in reviews],
                "individual_confidences": [r["confidence"] for r in reviews],
            },
            reviewer_ids=reviewer_ids,
            duration_ms=duration_ms,
        )

    def _get_available_models(self) -> list[str]:
        """
        Get list of available models for diverse reviews.

        Returns:
            List of model IDs
        """
        try:
            # Get all models from routing config
            models = []
            for tier in self._routing_config.tiers.values():
                for model in tier.models:
                    if model.name not in models:
                        models.append(model.name)
            return models if models else ["default"]
        except Exception:
            return ["default"]

    async def _run_reviews(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
        model_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        Run reviews with multiple models in parallel.

        Args:
            output: Output to review
            context: Validation context
            model_ids: List of model IDs to use

        Returns:
            List of review results
        """
        tasks = []
        for model_id in model_ids:
            tasks.append(self._run_single_review(output, context, model_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        reviews = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Review with model {model_ids[i]} failed: {result}")
                # Create a failed review result
                reviews.append(
                    {
                        "verdict": ProfileVerdict.FAIL,
                        "confidence": 0.0,
                        "issues": [
                            QAIssue(
                                severity=IssueSeverity.MAJOR,
                                category=IssueCategory.CORRECTNESS,
                                description=f"Review failed: {result}",
                                reviewer_id=model_ids[i],
                            )
                        ],
                        "reviewer_id": model_ids[i],
                        "details": {"error": str(result)},
                    }
                )
            else:
                reviews.append(result)

        return reviews

    async def _run_single_review(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
        model_id: str,
    ) -> dict[str, Any]:
        """
        Run a single review with a specific model.

        Args:
            output: Output to review
            context: Validation context
            model_id: Model to use for review

        Returns:
            Review result dictionary
        """
        additional_instructions = """
Please provide a thorough and objective review. Focus on:
- Whether the output correctly addresses the task requirements
- Code quality and maintainability
- Potential bugs or edge cases
- Adherence to best practices

Be balanced in your assessment - note both strengths and weaknesses.
"""

        prompt = self._build_review_prompt(output, context, additional_instructions)

        response, actual_model = await self._call_model(
            prompt=prompt,
            model_id=model_id if model_id != "default" else None,
        )

        verdict, issues, confidence, details = self._parse_review_response(
            response, actual_model
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "issues": issues,
            "reviewer_id": actual_model,
            "details": details,
        }

    def _calculate_consensus(
        self,
        reviews: list[dict[str, Any]],
        threshold: float,
    ) -> tuple[ProfileVerdict, float, bool]:
        """
        Calculate consensus verdict from multiple reviews.

        Args:
            reviews: List of review results
            threshold: Agreement threshold (1.0 = unanimous)

        Returns:
            Tuple of (verdict, confidence, needs_escalation)
        """
        if not reviews:
            return ProfileVerdict.FAIL, 0.0, True

        # Count verdicts
        verdict_counts = {
            ProfileVerdict.PASS: 0,
            ProfileVerdict.FAIL: 0,
            ProfileVerdict.PARTIAL: 0,
        }
        for review in reviews:
            verdict_counts[review["verdict"]] += 1

        total = len(reviews)
        pass_ratio = verdict_counts[ProfileVerdict.PASS] / total
        fail_ratio = verdict_counts[ProfileVerdict.FAIL] / total

        # Calculate average confidence
        avg_confidence = sum(r["confidence"] for r in reviews) / total

        # Determine consensus
        if pass_ratio >= threshold:
            return ProfileVerdict.PASS, avg_confidence, False
        elif fail_ratio >= threshold:
            return ProfileVerdict.FAIL, avg_confidence, False
        else:
            # No consensus - determine majority
            if pass_ratio > fail_ratio:
                return ProfileVerdict.PASS, avg_confidence * 0.7, True
            elif fail_ratio > pass_ratio:
                return ProfileVerdict.FAIL, avg_confidence * 0.7, True
            else:
                return ProfileVerdict.PARTIAL, avg_confidence * 0.5, True

    def _deduplicate_issues(self, issues: list[QAIssue]) -> list[QAIssue]:
        """
        Deduplicate similar issues from multiple reviewers.

        Issues are considered duplicates if they have the same severity,
        category, and similar descriptions.

        Args:
            issues: List of issues to deduplicate

        Returns:
            Deduplicated list of issues
        """
        if not issues:
            return []

        unique = []
        seen_signatures = set()

        for issue in issues:
            # Create a signature for the issue
            # Use first 50 chars of description for fuzzy matching
            desc_prefix = issue.description[:50].lower().strip()
            signature = (
                issue.severity.value,
                issue.category.value,
                desc_prefix,
                issue.location,
            )

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique.append(issue)

        return unique