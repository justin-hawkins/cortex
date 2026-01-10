"""
Documentation validator for DATS QA.

Validates documentation accuracy by cross-referencing against source
materials and using Q&A validation techniques.
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
from src.qa.profiles import DocumentationProfile

logger = logging.getLogger(__name__)


class DocumentationValidator(BaseValidator):
    """
    Documentation accuracy validator.

    Cross-references documentation against source materials,
    uses Q&A validation, and checks for completeness.
    """

    validator_name = "documentation"

    def __init__(
        self,
        profile: DocumentationProfile,
        model_client=None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize documentation validator.

        Args:
            profile: Documentation profile configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override
        """
        super().__init__(profile, model_client, model_tier)
        self.documentation_profile = profile

    def _get_system_prompt(self) -> str:
        """Get documentation-focused system prompt."""
        return """You are a DOCUMENTATION REVIEWER in a distributed agentic task system.
Your role is to verify documentation accuracy, completeness, and clarity.

When reviewing documentation, assess:

1. Accuracy - Is the information correct?
   - Technical details match the code
   - No outdated information
   - Correct terminology
   - Accurate examples

2. Completeness - Is everything covered?
   - All features documented
   - API endpoints described
   - Parameters explained
   - Return values documented
   - Error conditions covered

3. Clarity - Is it understandable?
   - Clear language
   - Good organization
   - Helpful examples
   - Appropriate level of detail

4. Consistency - Is it internally consistent?
   - Consistent terminology
   - Consistent formatting
   - No contradictions

5. Cross-references - Do links and references work?
   - Internal links valid
   - External references accurate
   - Code snippets match actual code

Flag any areas of uncertainty - places where you cannot verify
accuracy without additional context.

Provide your review in valid JSON format."""

    async def validate(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
    ) -> ProfileResult:
        """
        Validate documentation accuracy.

        Args:
            output: Worker output to validate
            context: Validation context

        Returns:
            ProfileResult with documentation verdict
        """
        start_time = datetime.utcnow()
        config = self.documentation_profile.get_validator_config()

        cross_reference = config.get("cross_reference_sources", True)
        use_qa = config.get("use_qa_validation", True)
        check_completeness = config.get("check_completeness", True)
        flag_uncertainty = config.get("flag_uncertainty", True)

        # Build documentation-specific prompt
        additional_instructions = self._build_documentation_instructions(config)
        prompt = self._build_review_prompt(output, context, additional_instructions)

        # Run the documentation review
        response, reviewer_id = await self._call_model(prompt)

        # Parse the response
        verdict, issues, confidence, details = self._parse_review_response(
            response, reviewer_id
        )

        # Tag all issues as documentation category
        for issue in issues:
            if issue.category != IssueCategory.DOCUMENTATION:
                issue.category = IssueCategory.DOCUMENTATION

        # Run Q&A validation if enabled
        qa_issues = []
        if use_qa and context.rag_context:
            qa_issues = await self._run_qa_validation(output, context, reviewer_id)
            issues.extend(qa_issues)

        # Check completeness against scope if defined
        scope_issues = []
        scope_definition = config.get("scope_definition")
        if check_completeness and scope_definition:
            scope_issues = await self._check_scope_completeness(
                output, scope_definition, reviewer_id
            )
            issues.extend(scope_issues)

        # Apply documentation-specific verdict rules
        final_verdict = self._apply_documentation_rules(
            verdict, issues, flag_uncertainty
        )

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return self._create_result(
            verdict=final_verdict,
            issues=issues,
            confidence=confidence,
            details={
                "checks_performed": {
                    "cross_reference": cross_reference,
                    "qa_validation": use_qa,
                    "completeness": check_completeness,
                    "uncertainty_flagging": flag_uncertainty,
                },
                "qa_validation_performed": len(qa_issues) > 0,
                "scope_check_performed": len(scope_issues) > 0,
                **details,
            },
            reviewer_ids=[reviewer_id],
            duration_ms=duration_ms,
        )

    def _build_documentation_instructions(self, config: dict[str, Any]) -> str:
        """Build documentation-specific instructions based on config."""
        checks = []

        if config.get("cross_reference_sources", True):
            source_materials = config.get("source_materials", [])
            if source_materials:
                sources_str = "\n".join(f"- {s}" for s in source_materials)
                checks.append(f"""
## Cross-Reference Verification
Verify documentation claims against these source materials:
{sources_str}

Check for:
- Incorrect statements
- Outdated information
- Missing important details from sources""")
            else:
                checks.append("""
## Accuracy Verification
Check documentation accuracy against general best practices:
- Technical accuracy
- Correct terminology
- Valid code examples""")

        if config.get("use_qa_validation", True):
            checks.append("""
## Q&A Validation
Generate questions that the documentation should answer:
- Can a reader understand how to use this?
- Are all parameters explained?
- Are error conditions documented?

Identify any questions that cannot be answered from the documentation.""")

        if config.get("check_completeness", True):
            scope = config.get("scope_definition", "")
            if scope:
                checks.append(f"""
## Completeness Check
The documentation should cover: {scope}

Identify any missing sections or topics.""")
            else:
                checks.append("""
## Completeness Check
Verify the documentation covers:
- All public APIs/functions
- All configuration options
- Error handling
- Examples and usage patterns""")

        if config.get("flag_uncertainty", True):
            checks.append("""
## Uncertainty Flagging
Flag any areas where you cannot verify accuracy:
- Complex technical claims
- Version-specific information
- External system interactions

Mark these as "uncertainty" issues with severity "minor".""")

        instructions = "\n".join(checks)

        return f"""
## DOCUMENTATION REVIEW

Evaluate the documentation for accuracy, completeness, and clarity.

{instructions}

Provide your response with:
1. accuracy_issues: Claims that appear incorrect or unverifiable
2. completeness_gaps: Topics that should be covered but aren't
3. clarity_issues: Sections that are confusing or unclear
4. uncertainty_areas: Areas where accuracy cannot be verified

Rate issues based on impact:
- Critical: Incorrect information that could cause failures
- Major: Missing essential information
- Minor: Unclear sections or missing nice-to-have content
- Suggestion: Style or organizational improvements
"""

    async def _run_qa_validation(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
        reviewer_id: str,
    ) -> list[QAIssue]:
        """
        Run Q&A validation on documentation.

        Generates questions from the documentation and verifies
        they can be answered correctly.

        Args:
            output: Documentation output
            context: Validation context with RAG context
            reviewer_id: Reviewer ID for issues

        Returns:
            List of issues found through Q&A validation
        """
        # Generate questions from documentation
        qa_prompt = f"""Based on this documentation, generate 5 key questions 
that a user should be able to answer after reading it:

Documentation:
```
{output.content[:2000]}  # Limit for prompt size
```

Then, for each question, check if the documentation actually answers it.

Respond with JSON:
{{
    "questions": [
        {{
            "question": "...",
            "answerable": true/false,
            "answer_quality": "complete/partial/missing",
            "notes": "..."
        }}
    ]
}}"""

        try:
            response, _ = await self._call_model(qa_prompt)
            
            import json
            # Try to parse the response
            try:
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                else:
                    json_str = response.strip()
                
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return []

            issues = []
            for q in data.get("questions", []):
                if not q.get("answerable", True):
                    issues.append(
                        QAIssue(
                            severity=IssueSeverity.MAJOR,
                            category=IssueCategory.DOCUMENTATION,
                            description=f"Documentation doesn't answer: {q.get('question', 'Unknown question')}",
                            recommendation=q.get("notes", "Add this information"),
                            reviewer_id=reviewer_id,
                        )
                    )
                elif q.get("answer_quality") == "partial":
                    issues.append(
                        QAIssue(
                            severity=IssueSeverity.MINOR,
                            category=IssueCategory.DOCUMENTATION,
                            description=f"Documentation partially answers: {q.get('question', 'Unknown question')}",
                            recommendation=q.get("notes", "Expand this section"),
                            reviewer_id=reviewer_id,
                        )
                    )

            return issues

        except Exception as e:
            logger.warning(f"Q&A validation failed: {e}")
            return []

    async def _check_scope_completeness(
        self,
        output: WorkerOutput,
        scope_definition: str,
        reviewer_id: str,
    ) -> list[QAIssue]:
        """
        Check documentation completeness against defined scope.

        Args:
            output: Documentation output
            scope_definition: What the docs should cover
            reviewer_id: Reviewer ID for issues

        Returns:
            List of completeness issues
        """
        completeness_prompt = f"""Check if this documentation covers all required topics.

Required scope:
{scope_definition}

Documentation:
```
{output.content[:3000]}  # Limit for prompt size
```

Respond with JSON:
{{
    "covered_topics": ["topic1", "topic2"],
    "missing_topics": ["topic3", "topic4"],
    "partial_topics": [
        {{"topic": "...", "missing": "what's missing"}}
    ]
}}"""

        try:
            response, _ = await self._call_model(completeness_prompt)
            
            import json
            try:
                if "```json" in response:
                    json_start = response.find("```json") + 7
                    json_end = response.find("```", json_start)
                    json_str = response[json_start:json_end].strip()
                else:
                    json_str = response.strip()
                
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return []

            issues = []
            
            for topic in data.get("missing_topics", []):
                issues.append(
                    QAIssue(
                        severity=IssueSeverity.MAJOR,
                        category=IssueCategory.COMPLETENESS,
                        description=f"Missing required topic: {topic}",
                        recommendation=f"Add documentation for {topic}",
                        reviewer_id=reviewer_id,
                    )
                )
            
            for partial in data.get("partial_topics", []):
                issues.append(
                    QAIssue(
                        severity=IssueSeverity.MINOR,
                        category=IssueCategory.COMPLETENESS,
                        description=f"Incomplete coverage of: {partial.get('topic', 'Unknown')}",
                        recommendation=partial.get("missing", "Expand this section"),
                        reviewer_id=reviewer_id,
                    )
                )

            return issues

        except Exception as e:
            logger.warning(f"Scope completeness check failed: {e}")
            return []

    def _apply_documentation_rules(
        self,
        original_verdict: ProfileVerdict,
        issues: list[QAIssue],
        flag_uncertainty: bool,
    ) -> ProfileVerdict:
        """
        Apply documentation-specific verdict rules.

        Documentation is more lenient than security, but
        incorrect information is still serious.

        Args:
            original_verdict: Verdict from model
            issues: Issues found
            flag_uncertainty: Whether uncertainty was flagged

        Returns:
            Final verdict
        """
        has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
        has_major = any(i.severity == IssueSeverity.MAJOR for i in issues)

        # Critical documentation issues (incorrect info) should fail
        if has_critical:
            return ProfileVerdict.FAIL

        # Major issues result in partial (needs revision)
        if has_major:
            return ProfileVerdict.PARTIAL

        return original_verdict