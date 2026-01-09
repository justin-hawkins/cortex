"""
Security validator for DATS QA.

Implements security-focused validation checking for common
vulnerabilities, input sanitization, and secure coding practices.
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
from src.qa.profiles import SecurityProfile

logger = logging.getLogger(__name__)


class SecurityValidator(BaseValidator):
    """
    Security-focused validator.

    Checks for common vulnerabilities, input sanitization issues,
    authentication/authorization flaws, and information leakage.
    """

    validator_name = "security"

    def __init__(
        self,
        profile: SecurityProfile,
        model_client=None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize security validator.

        Args:
            profile: Security profile configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override
        """
        super().__init__(profile, model_client, model_tier)
        self.security_profile = profile

    def _get_system_prompt(self) -> str:
        """Get security-focused system prompt."""
        config = self.security_profile.get_validator_config()
        categories = config.get("vulnerability_categories", [])

        category_descriptions = {
            "sql_injection": "SQL injection through unsanitized database queries",
            "xss": "Cross-site scripting (XSS) through unescaped output",
            "command_injection": "Command injection through shell execution",
            "path_traversal": "Path traversal allowing access to unauthorized files",
            "authentication": "Authentication bypasses or weak authentication",
            "authorization": "Authorization flaws allowing privilege escalation",
            "sensitive_data_exposure": "Exposure of sensitive data (credentials, PII, secrets)",
            "insecure_deserialization": "Insecure deserialization allowing code execution",
        }

        category_list = "\n".join(
            f"- {cat}: {category_descriptions.get(cat, cat)}"
            for cat in categories
        )

        return f"""You are a SECURITY REVIEWER in a distributed agentic task system.
Your role is to identify security vulnerabilities and unsafe coding practices.

You are checking for the following vulnerability categories:
{category_list}

For each potential vulnerability, assess:
1. Is this a real vulnerability or a false positive?
2. What is the attack vector?
3. What data or systems could be compromised?
4. What is the severity (critical/major/minor)?

Critical issues:
- Authentication bypass
- SQL injection with data exposure
- Remote code execution
- Credential exposure

Major issues:
- XSS vulnerabilities
- Missing input validation
- Information disclosure
- Weak cryptography

Minor issues:
- Missing security headers
- Verbose error messages
- Hardcoded non-sensitive configuration

Always provide specific remediation recommendations.
Provide your review in valid JSON format."""

    async def validate(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
    ) -> ProfileResult:
        """
        Validate output for security issues.

        Args:
            output: Worker output to validate
            context: Validation context

        Returns:
            ProfileResult with security verdict
        """
        start_time = datetime.utcnow()
        config = self.security_profile.get_validator_config()

        critical_always_blocks = config.get("critical_always_blocks", True)

        # Build security-specific prompt
        additional_instructions = self._build_security_instructions(config)
        prompt = self._build_review_prompt(output, context, additional_instructions)

        # Run the security review
        response, reviewer_id = await self._call_model(prompt)

        # Parse the response
        verdict, issues, confidence, details = self._parse_review_response(
            response, reviewer_id
        )

        # Tag all issues as security category
        for issue in issues:
            if issue.category != IssueCategory.SECURITY:
                issue.category = IssueCategory.SECURITY

        # Apply security-specific verdict rules
        final_verdict = self._apply_security_rules(
            verdict, issues, critical_always_blocks
        )

        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return self._create_result(
            verdict=final_verdict,
            issues=issues,
            confidence=confidence,
            details={
                "vulnerability_categories_checked": config.get(
                    "vulnerability_categories", []
                ),
                "checks_performed": {
                    "injection": config.get("check_injection", True),
                    "auth": config.get("check_auth", True),
                    "exposure": config.get("check_exposure", True),
                    "error_handling": config.get("check_error_handling", True),
                    "input_validation": config.get("check_input_validation", True),
                },
                **details,
            },
            reviewer_ids=[reviewer_id],
            duration_ms=duration_ms,
        )

    def _build_security_instructions(self, config: dict[str, Any]) -> str:
        """Build security-specific instructions based on config."""
        checks = []

        if config.get("check_injection", True):
            checks.append("""
## Injection Checks
- Look for unsanitized user input in SQL queries
- Check for command injection in shell/exec calls
- Verify template injection isn't possible
- Check for LDAP, XPath, or other injection vectors""")

        if config.get("check_auth", True):
            checks.append("""
## Authentication/Authorization Checks
- Verify authentication is required where needed
- Check for proper session management
- Look for authorization bypasses
- Verify password handling is secure
- Check for insecure direct object references""")

        if config.get("check_exposure", True):
            checks.append("""
## Data Exposure Checks
- Look for hardcoded credentials or API keys
- Check for sensitive data in logs
- Verify PII is properly protected
- Look for exposed internal paths or structure
- Check for unnecessary data in responses""")

        if config.get("check_error_handling", True):
            checks.append("""
## Error Handling Checks
- Verify errors don't expose stack traces
- Check that database errors are sanitized
- Look for information leakage in error messages
- Verify exception handling doesn't reveal internals""")

        if config.get("check_input_validation", True):
            checks.append("""
## Input Validation Checks
- Verify all user input is validated
- Check for proper type validation
- Look for missing boundary checks
- Verify file upload restrictions
- Check for proper encoding/escaping""")

        instructions = "\n".join(checks)

        return f"""
## SECURITY REVIEW

Perform a comprehensive security review of the code.

{instructions}

For EVERY potential security issue:
1. Identify the vulnerable code pattern
2. Explain the attack vector
3. Rate severity (critical/major/minor)
4. Provide specific remediation steps

Remember: Security issues are ALWAYS important. Even minor issues should be reported.
Critical and major issues will block approval.
"""

    def _apply_security_rules(
        self,
        original_verdict: ProfileVerdict,
        issues: list[QAIssue],
        critical_blocks: bool,
    ) -> ProfileVerdict:
        """
        Apply security-specific verdict rules.

        Security is strict - any critical issue blocks, and major
        issues result in partial verdict.

        Args:
            original_verdict: Verdict from model
            issues: Issues found
            critical_blocks: Whether critical issues block

        Returns:
            Final verdict
        """
        has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
        has_major = any(i.severity == IssueSeverity.MAJOR for i in issues)

        # Critical security issues ALWAYS block
        if has_critical and critical_blocks:
            return ProfileVerdict.FAIL

        # Major security issues cause partial
        if has_major:
            return ProfileVerdict.PARTIAL

        # If we have critical but not blocking (rare), still partial
        if has_critical:
            return ProfileVerdict.PARTIAL

        # No blocking issues - use original verdict
        return original_verdict