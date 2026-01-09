"""
Base validator class for DATS QA.

Provides the abstract interface and common functionality
for all QA validators.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.config.routing import get_routing_config
from src.config.settings import get_settings
from src.models.base import BaseModelClient
from src.models.ollama_client import OllamaClient
from src.models.openai_client import OpenAICompatibleClient
from src.models.anthropic_client import AnthropicClient
from src.qa.results import (
    ProfileResult,
    ProfileVerdict,
    QAIssue,
    IssueSeverity,
    IssueCategory,
)
from src.qa.profiles import QAProfile

logger = logging.getLogger(__name__)


@dataclass
class WorkerOutput:
    """Represents the output from a worker to be validated."""

    task_id: str
    content: str
    content_type: str = "code"  # code, documentation, test, etc.
    language: str = ""
    file_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "content": self.content,
            "content_type": self.content_type,
            "language": self.language,
            "file_path": self.file_path,
            "metadata": self.metadata,
        }


@dataclass
class ValidatorContext:
    """Context for validator execution."""

    task_id: str
    project_id: str
    task_description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    domain: str = "code-general"
    rag_context: Optional[str] = None
    previous_reviews: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class BaseValidator(ABC):
    """
    Abstract base class for QA validators.

    Each validator implements validation logic for a specific
    QA profile type.
    """

    validator_name: str = "base"

    def __init__(
        self,
        profile: QAProfile,
        model_client: Optional[BaseModelClient] = None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize the validator.

        Args:
            profile: QA profile configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override for routing
        """
        self.profile = profile
        self._model_client = model_client
        self._model_tier = model_tier
        self._settings = get_settings()
        self._routing_config = get_routing_config()

    def get_model_client(self, model_id: Optional[str] = None) -> BaseModelClient:
        """
        Get or create model client based on routing configuration.

        Args:
            model_id: Optional specific model to use

        Returns:
            Configured model client
        """
        if self._model_client and not model_id:
            return self._model_client

        # Get routing for QA reviewer agent
        routing = self._routing_config.get_agent_routing("qa_reviewer")
        if not routing:
            raise ValueError("No routing configured for qa_reviewer agent")

        # Use tier override if provided
        tier_name = self._model_tier or routing.preferred_tier
        tier = self._routing_config.get_tier(tier_name)
        if not tier:
            raise ValueError(f"Unknown tier: {tier_name}")

        # Get model config
        if model_id:
            model_config = tier.get_model_by_name(model_id)
        elif routing.preferred_model:
            model_config = tier.get_model_by_name(routing.preferred_model)
        else:
            model_config = tier.get_primary_model()

        if not model_config:
            raise ValueError(f"No model found for tier: {tier_name}")

        return self._create_client(model_config)

    def _create_client(self, model_config) -> BaseModelClient:
        """Create a model client from configuration."""
        if model_config.type == "ollama":
            return OllamaClient(
                endpoint=model_config.endpoint,
                model_name=model_config.name,
            )
        elif model_config.type == "openai_compatible":
            return OpenAICompatibleClient(
                endpoint=model_config.endpoint,
                model_name=model_config.name,
            )
        elif model_config.type == "anthropic":
            return AnthropicClient(
                endpoint=model_config.endpoint,
                model_name=model_config.name,
                api_key=self._settings.anthropic_api_key,
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.type}")

    @abstractmethod
    async def validate(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
    ) -> ProfileResult:
        """
        Validate worker output.

        Args:
            output: Worker output to validate
            context: Validation context

        Returns:
            ProfileResult with verdict and issues
        """
        pass

    def _get_system_prompt(self) -> str:
        """Get base system prompt for the validator."""
        return """You are a QA validator in a distributed agentic task system.
Your role is to carefully review work outputs and identify any issues.

When reviewing, consider:
1. Correctness: Does the output meet the requirements?
2. Completeness: Are all acceptance criteria addressed?
3. Quality: Is the code/output well-structured and maintainable?
4. Edge cases: Are potential edge cases handled?

Provide your review in valid JSON format."""

    def _get_temperature(self) -> float:
        """Get sampling temperature for validation."""
        return 0.3  # Low temperature for consistent reviews

    def _get_max_tokens(self) -> int:
        """Get maximum output tokens."""
        return 4096

    async def _call_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Call the model with a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            model_id: Optional specific model to use

        Returns:
            Tuple of (response_content, model_id)
        """
        client = self.get_model_client(model_id)
        response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt or self._get_system_prompt(),
            temperature=self._get_temperature(),
            max_tokens=self._get_max_tokens(),
        )
        return response.content, response.model

    def _parse_review_response(
        self,
        response: str,
        reviewer_id: str = "",
    ) -> tuple[ProfileVerdict, list[QAIssue], float, dict[str, Any]]:
        """
        Parse a model review response into structured data.

        Args:
            response: Model response content
            reviewer_id: ID of the reviewer model

        Returns:
            Tuple of (verdict, issues, confidence, details)
        """
        try:
            # Try to extract JSON from response
            json_str = response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif response.strip().startswith("{"):
                json_str = response.strip()

            data = json.loads(json_str)

            # Extract verdict
            verdict_str = data.get("verdict", data.get("approved", "fail"))
            if isinstance(verdict_str, bool):
                verdict = ProfileVerdict.PASS if verdict_str else ProfileVerdict.FAIL
            elif verdict_str.lower() in ("pass", "approved", "true"):
                verdict = ProfileVerdict.PASS
            elif verdict_str.lower() in ("partial", "needs_revision"):
                verdict = ProfileVerdict.PARTIAL
            else:
                verdict = ProfileVerdict.FAIL

            # Extract confidence
            confidence = float(data.get("confidence", 0.8))

            # Extract issues
            issues = []
            for issue_data in data.get("issues", []):
                try:
                    severity = IssueSeverity(
                        issue_data.get("severity", "minor").lower()
                    )
                except ValueError:
                    severity = IssueSeverity.MINOR

                try:
                    category = IssueCategory(
                        issue_data.get("category", "correctness").lower()
                    )
                except ValueError:
                    category = IssueCategory.CORRECTNESS

                issues.append(
                    QAIssue(
                        severity=severity,
                        category=category,
                        description=issue_data.get("description", ""),
                        location=issue_data.get("location", ""),
                        recommendation=issue_data.get("recommendation", ""),
                        reviewer_id=reviewer_id,
                    )
                )

            # Extract any additional details
            details = {
                k: v
                for k, v in data.items()
                if k not in ("verdict", "approved", "confidence", "issues")
            }

            return verdict, issues, confidence, details

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse review response: {e}")
            # Return a conservative result on parse failure
            return (
                ProfileVerdict.FAIL,
                [
                    QAIssue(
                        severity=IssueSeverity.MAJOR,
                        category=IssueCategory.CORRECTNESS,
                        description=f"Failed to parse review response: {e}",
                        reviewer_id=reviewer_id,
                    )
                ],
                0.0,
                {"raw_response": response, "parse_error": str(e)},
            )

    def _build_review_prompt(
        self,
        output: WorkerOutput,
        context: ValidatorContext,
        additional_instructions: str = "",
    ) -> str:
        """
        Build a review prompt for the model.

        Args:
            output: Worker output to review
            context: Validation context
            additional_instructions: Extra instructions for this validator

        Returns:
            Formatted prompt string
        """
        acceptance_str = "\n".join(
            f"- {criterion}" for criterion in context.acceptance_criteria
        )

        prompt = f"""Please review the following work output:

## Task Description
{context.task_description}

## Acceptance Criteria
{acceptance_str if acceptance_str else "No specific criteria provided."}

## Work Output
```{output.language or output.content_type}
{output.content}
```

{additional_instructions}

Respond with a JSON object containing:
{{
    "verdict": "pass" | "fail" | "partial",
    "confidence": <float 0.0-1.0>,
    "issues": [
        {{
            "severity": "critical" | "major" | "minor" | "suggestion",
            "category": "correctness" | "security" | "performance" | "style" | "testing" | "documentation" | "completeness",
            "description": "<description of the issue>",
            "location": "<where the issue is>",
            "recommendation": "<how to fix>"
        }}
    ],
    "summary": "<brief summary of the review>"
}}"""

        return prompt

    def _create_result(
        self,
        verdict: ProfileVerdict,
        issues: list[QAIssue],
        confidence: float,
        details: dict[str, Any],
        reviewer_ids: list[str],
        duration_ms: int,
    ) -> ProfileResult:
        """
        Create a ProfileResult from validation data.

        Args:
            verdict: Validation verdict
            issues: List of issues found
            confidence: Confidence score
            details: Additional details
            reviewer_ids: List of reviewer model IDs
            duration_ms: Duration in milliseconds

        Returns:
            ProfileResult instance
        """
        return ProfileResult(
            profile=self.profile.profile_type.value,
            verdict=verdict,
            confidence=confidence,
            issues=issues,
            details=details,
            reviewer_ids=reviewer_ids,
            duration_ms=duration_ms,
        )