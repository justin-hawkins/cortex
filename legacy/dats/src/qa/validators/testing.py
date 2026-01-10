"""
Testing validator for DATS QA.

Validates test coverage, assertion quality, and edge case testing
for code outputs that include tests.
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
    IssueCategory,
)
from src.qa.profiles import TestingProfile

logger = logging.getLogger(__name__)


class TestingValidator(BaseValidator):
    """
    Testing-focused validator.

    Validates test coverage, assertion quality, edge case testing,
    and overall test suite completeness.
    """

    validator_name = "testing"

    def __init__(
        self,
        profile: TestingProfile,
        model_client=None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize testing validator.

        Args:
            profile: Testing profile configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override
        """
        super().__init__(profile, model_client, model_tier)
        self.testing_profile = profile

    def _get_system_prompt(self) -> str:
        """Get testing-focused system prompt."""
        return """You are a TEST QUALITY REVIEWER in a distributed agentic task system.
Your role is to evaluate the quality and completeness of test code.

When reviewing tests, assess:

1. Coverage - Are all important code paths tested?
   - Happy paths
   - Error conditions
   - Edge cases
   - Boundary conditions

2. Assertion Quality - Are assertions meaningful?
   - Testing the right things
   - Specific enough to catch bugs
   - Not too broad or vague
   - Proper error messages

3. Test Structure - Are tests well-organized?
   - Clear naming
   - Proper setup/teardown
   - Isolated and independent
   - Readable and maintainable

4. Edge Cases - Are unusual scenarios covered?
   - Empty inputs
   - Null/None values
   - Maximum/minimum values
   - Invalid inputs
   - Concurrent access (if applicable)

5. Mocking - Is mocking used appropriately?
   - External dependencies mocked
   - Not over-mocking
   - Realistic mock behavior

Provide your review in valid JSON format."""

    async def validate(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
    ) -> ProfileResult:
        """
        Validate test quality.

        Args:
            output: Worker output to validate
            context: Validation context

        Returns:
            ProfileResult with testing verdict
        """
        start_time = datetime.utcnow()
        config = self.testing_profile.get_validator_config()

        min_coverage = config.get("min_coverage_threshold", 0.8)
        can_request_tests = config.get("can_request_additional_tests", True)
        fail_on_missing = config.get("fail_on_missing_tests", False)

        # Build testing-specific prompt
        additional_instructions = self._build_testing_instructions(config)
        prompt = self._build_review_prompt(output, context, additional_instructions)

        # Run the testing review
        response, reviewer_id = await self._call_model(prompt)

        # Parse the response
        verdict, issues, confidence, details = self._parse_review_response(
            response, reviewer_id
        )

        # Tag all issues as testing category
        for issue in issues:
            if issue.category not in (IssueCategory.TESTING, IssueCategory.COMPLETENESS):
                issue.category = IssueCategory.TESTING

        # Extract coverage estimate if provided
        coverage_estimate = details.get("coverage_estimate", 0.0)
        if isinstance(coverage_estimate, str):
            try:
                coverage_estimate = float(coverage_estimate.rstrip("%")) / 100
            except ValueError:
                coverage_estimate = 0.0

        # Apply testing-specific verdict rules
        final_verdict, additional_issues = self._apply_testing_rules(
            verdict,
            issues,
            coverage_estimate,
            min_coverage,
            fail_on_missing,
            can_request_tests,
            reviewer_id,
        )

        issues.extend(additional_issues)

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return self._create_result(
            verdict=final_verdict,
            issues=issues,
            confidence=confidence,
            details={
                "coverage_estimate": coverage_estimate,
                "min_coverage_threshold": min_coverage,
                "checks_performed": {
                    "coverage": config.get("check_coverage", True),
                    "assertion_quality": config.get("check_assertion_quality", True),
                    "edge_cases": config.get("check_edge_cases", True),
                },
                "can_request_additional_tests": can_request_tests,
                **details,
            },
            reviewer_ids=[reviewer_id],
            duration_ms=duration_ms,
        )

    def _build_testing_instructions(self, config: dict[str, Any]) -> str:
        """Build testing-specific instructions based on config."""
        checks = []

        if config.get("check_coverage", True):
            min_coverage = config.get("min_coverage_threshold", 0.8)
            checks.append(f"""
## Coverage Analysis
Estimate the test coverage percentage (target: {min_coverage * 100}%)
- Identify which functions/methods are tested
- Note any untested code paths
- List critical functionality without tests""")

        if config.get("check_assertion_quality", True):
            checks.append("""
## Assertion Quality
Evaluate the quality of test assertions:
- Are assertions specific and meaningful?
- Do they test actual behavior, not implementation?
- Are error messages helpful?
- Are there too many assertions per test (should be focused)?""")

        if config.get("check_edge_cases", True):
            checks.append("""
## Edge Case Coverage
Check if edge cases are properly tested:
- Empty/null inputs
- Boundary values (0, -1, MAX_INT, etc.)
- Error conditions
- Timeout scenarios
- Concurrent access (if applicable)""")

        instructions = "\n".join(checks)

        return f"""
## TEST QUALITY REVIEW

Evaluate the quality and completeness of the test code.

{instructions}

Provide your response with:
1. coverage_estimate: Your estimated test coverage (0.0 to 1.0)
2. missing_tests: List of tests that should be added
3. weak_assertions: Tests with weak or missing assertions
4. edge_cases_needed: Edge cases that need testing

Rate issues based on impact:
- Critical: Core functionality untested
- Major: Important edge cases missing
- Minor: Nice-to-have additional tests
- Suggestion: Style or organizational improvements
"""

    def _apply_testing_rules(
        self,
        original_verdict: ProfileVerdict,
        issues: list[QAIssue],
        coverage_estimate: float,
        min_coverage: float,
        fail_on_missing: bool,
        can_request_tests: bool,
        reviewer_id: str,
    ) -> tuple[ProfileVerdict, list[QAIssue]]:
        """
        Apply testing-specific verdict rules.

        Args:
            original_verdict: Verdict from model
            issues: Issues found
            coverage_estimate: Estimated test coverage
            min_coverage: Minimum required coverage
            fail_on_missing: Whether to fail on missing tests
            can_request_tests: Whether we can request additional tests
            reviewer_id: Reviewer ID for any new issues

        Returns:
            Tuple of (final verdict, additional issues)
        """
        additional_issues = []

        # Check coverage threshold
        if coverage_estimate > 0 and coverage_estimate < min_coverage:
            severity = IssueSeverity.MAJOR if fail_on_missing else IssueSeverity.MINOR
            additional_issues.append(
                QAIssue(
                    severity=severity,
                    category=IssueCategory.TESTING,
                    description=(
                        f"Test coverage estimate ({coverage_estimate:.0%}) "
                        f"is below threshold ({min_coverage:.0%})"
                    ),
                    recommendation="Add tests to improve coverage",
                    reviewer_id=reviewer_id,
                )
            )

        # Count testing issues
        has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
        has_major = any(i.severity == IssueSeverity.MAJOR for i in issues)

        # Determine verdict
        if has_critical:
            return ProfileVerdict.FAIL, additional_issues

        if fail_on_missing and has_major:
            return ProfileVerdict.FAIL, additional_issues

        if has_major and can_request_tests:
            # Partial - we'll request additional tests
            return ProfileVerdict.PARTIAL, additional_issues

        if coverage_estimate > 0 and coverage_estimate < min_coverage:
            if fail_on_missing:
                return ProfileVerdict.FAIL, additional_issues
            elif can_request_tests:
                return ProfileVerdict.PARTIAL, additional_issues

        return original_verdict, additional_issues