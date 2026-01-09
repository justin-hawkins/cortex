"""
Adversarial validator for DATS QA.

Implements aggressive flaw-finding review where the reviewer
actively tries to find issues, edge cases, and potential failures.
"""

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
)
from src.qa.profiles import AdversarialProfile

logger = logging.getLogger(__name__)


class AdversarialValidator(BaseValidator):
    """
    Adversarial validator.

    Uses a single reviewer in adversarial mode, prompted to actively
    find flaws, edge cases, and potential failures in the output.
    """

    validator_name = "adversarial"

    def __init__(
        self,
        profile: AdversarialProfile,
        model_client=None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize adversarial validator.

        Args:
            profile: Adversarial profile configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override
        """
        super().__init__(profile, model_client, model_tier)
        self.adversarial_profile = profile

    def _get_system_prompt(self) -> str:
        """Get adversarial system prompt."""
        config = self.adversarial_profile.get_validator_config()
        intensity = config.get("intensity", "high")

        intensity_instructions = {
            "low": "Look for obvious issues and common mistakes.",
            "medium": "Actively search for bugs, edge cases, and potential issues.",
            "high": "Be extremely thorough and critical. Challenge every assumption. "
            "Actively try to break the code and find subtle bugs.",
        }

        focus_areas = config.get("focus_areas", [])
        focus_str = ""
        if focus_areas:
            focus_str = f"\n\nFocus especially on: {', '.join(focus_areas)}"

        return f"""You are an ADVERSARIAL code reviewer in a distributed agentic task system.
Your role is to actively find flaws, bugs, and potential issues in the code.

{intensity_instructions.get(intensity, intensity_instructions["high"])}

When reviewing, aggressively check for:
1. Logic errors and off-by-one bugs
2. Edge cases and boundary conditions
3. Error handling gaps
4. Race conditions and concurrency issues
5. Security vulnerabilities
6. Performance problems
7. Memory leaks or resource management issues
8. Incorrect assumptions
9. Missing validation
10. Potential crashes or exceptions
{focus_str}

You are NOT here to praise the code. You are here to find problems.
Even if the code looks good at first glance, dig deeper.
Think about how it could fail in production.

Provide your review in valid JSON format."""

    def _get_temperature(self) -> float:
        """Use slightly higher temperature for creative flaw-finding."""
        return 0.5

    async def validate(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
    ) -> ProfileResult:
        """
        Validate output using adversarial review.

        Args:
            output: Worker output to validate
            context: Validation context

        Returns:
            ProfileResult with adversarial verdict
        """
        start_time = datetime.utcnow()
        config = self.adversarial_profile.get_validator_config()

        max_issues = config.get("max_issues_to_report", 10)
        block_on_critical = config.get("block_on_critical", True)
        block_on_major = config.get("block_on_major", False)

        # Build adversarial review prompt
        additional_instructions = self._build_adversarial_instructions()
        prompt = self._build_review_prompt(output, context, additional_instructions)

        # Run the adversarial review
        response, reviewer_id = await self._call_model(prompt)

        # Parse the response
        verdict, issues, confidence, details = self._parse_review_response(
            response, reviewer_id
        )

        # Limit issues to max configured
        if len(issues) > max_issues:
            issues = self._prioritize_issues(issues, max_issues)

        # Override verdict based on blocking rules
        final_verdict = self._determine_verdict(
            verdict, issues, block_on_critical, block_on_major
        )

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return self._create_result(
            verdict=final_verdict,
            issues=issues,
            confidence=confidence,
            details={
                "intensity": config.get("intensity", "high"),
                "focus_areas": config.get("focus_areas", []),
                "original_verdict": verdict.value,
                "blocking_rules_applied": final_verdict != verdict,
                **details,
            },
            reviewer_ids=[reviewer_id],
            duration_ms=duration_ms,
        )

    def _build_adversarial_instructions(self) -> str:
        """Build adversarial-specific instructions."""
        return """
## ADVERSARIAL REVIEW MODE

You are in ADVERSARIAL mode. Your job is to find problems, not to approve code.

Think like an attacker or a malicious user:
- What inputs could cause this to fail?
- What assumptions is this code making that could be wrong?
- What happens under extreme load or with malformed data?
- Are there any security implications?
- Could this cause data corruption or loss?

Be specific about issues you find. Include:
- Exact location in the code
- Concrete example of how it could fail
- Severity based on real-world impact

Even minor issues should be reported. We want a comprehensive list of everything
that could potentially go wrong.

Important: Focus on REAL issues, not style preferences or theoretical problems
that would never occur in practice. Each issue should have a concrete failure scenario.
"""

    def _prioritize_issues(
        self, issues: list[QAIssue], max_count: int
    ) -> list[QAIssue]:
        """
        Prioritize and limit issues to max count.

        Keeps critical and major issues first, then minor, then suggestions.

        Args:
            issues: All issues found
            max_count: Maximum issues to return

        Returns:
            Prioritized list of issues
        """
        # Sort by severity
        severity_order = {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.MAJOR: 1,
            IssueSeverity.MINOR: 2,
            IssueSeverity.SUGGESTION: 3,
        }
        sorted_issues = sorted(
            issues, key=lambda i: severity_order.get(i.severity, 4)
        )
        return sorted_issues[:max_count]

    def _determine_verdict(
        self,
        original_verdict: ProfileVerdict,
        issues: list[QAIssue],
        block_on_critical: bool,
        block_on_major: bool,
    ) -> ProfileVerdict:
        """
        Determine final verdict based on issues and blocking rules.

        Args:
            original_verdict: Verdict from model
            issues: Issues found
            block_on_critical: Whether critical issues block
            block_on_major: Whether major issues block

        Returns:
            Final verdict
        """
        has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
        has_major = any(i.severity == IssueSeverity.MAJOR for i in issues)

        # Critical issues always fail if configured
        if has_critical and block_on_critical:
            return ProfileVerdict.FAIL

        # Major issues fail if configured
        if has_major and block_on_major:
            return ProfileVerdict.FAIL

        # If we have critical issues but aren't blocking, mark as partial
        if has_critical:
            return ProfileVerdict.PARTIAL

        # If we have major issues but aren't blocking, also partial
        if has_major:
            return ProfileVerdict.PARTIAL

        # If only minor/suggestions, let the original verdict stand
        # but be conservative - adversarial reviews tend to pass
        # only if the reviewer explicitly says it's good
        if original_verdict == ProfileVerdict.PASS:
            return ProfileVerdict.PASS

        return original_verdict