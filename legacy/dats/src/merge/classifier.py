"""
Conflict classifier for merge coordination.

Provides LLM-powered classification of conflicts to determine
type, severity, and appropriate resolution strategy.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from src.config.routing import get_routing_config
from src.config.settings import get_settings
from src.merge.config import MergeConfig, get_merge_config
from src.merge.models import (
    ConflictClassification,
    ConflictRecord,
    ConflictSeverity,
    ConflictType,
)
from src.models.base import BaseModelClient
from src.models.ollama_client import OllamaClient
from src.models.openai_client import OpenAICompatibleClient
from src.models.anthropic_client import AnthropicClient
from src.prompts.renderer import PromptRenderer

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result from conflict classification."""

    classification: ConflictClassification
    raw_response: str
    model_used: str
    tokens_input: int = 0
    tokens_output: int = 0
    duration_ms: int = 0


class ConflictClassifier:
    """
    Classifies conflicts using LLM analysis.

    Determines:
    - Type: textual, semantic, or architectural
    - Severity: trivial, moderate, significant, or fundamental
    - Auto-resolvability: Whether the conflict can be resolved automatically
    """

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        model_client: Optional[BaseModelClient] = None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize classifier.

        Args:
            config: Merge configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override for routing
        """
        self.config = config or get_merge_config()
        self._model_client = model_client
        self._model_tier = model_tier
        self._renderer = PromptRenderer()
        self._settings = get_settings()
        self._routing_config = get_routing_config()

    def get_model_client(self) -> BaseModelClient:
        """
        Get or create model client for classification.

        Returns:
            Configured model client
        """
        if self._model_client:
            return self._model_client

        # Get routing for merge coordinator (or use general agent routing)
        routing = self._routing_config.get_agent_routing("merge_coordinator")
        if not routing:
            routing = self._routing_config.get_agent_routing("coordinator")
        if not routing:
            raise ValueError("No routing configured for merge_coordinator or coordinator")

        # Use tier override if provided
        tier_name = self._model_tier or routing.preferred_tier
        tier = self._routing_config.get_tier(tier_name)
        if not tier:
            raise ValueError(f"Unknown tier: {tier_name}")

        # Get model config
        if routing.preferred_model:
            model_config = tier.get_model_by_name(routing.preferred_model)
        else:
            model_config = tier.get_primary_model()

        if not model_config:
            raise ValueError(f"No model found for tier: {tier_name}")

        # Create appropriate client
        self._model_client = self._create_client(model_config)
        return self._model_client

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

    async def classify(
        self,
        conflict: ConflictRecord,
    ) -> ClassificationResult:
        """
        Classify a conflict using LLM analysis.

        Args:
            conflict: Conflict record to classify

        Returns:
            ClassificationResult with classification details
        """
        started_at = datetime.utcnow()

        # Build classification prompt
        prompt = self._build_classification_prompt(conflict)

        # Get model and classify
        client = self.get_model_client()
        response = await client.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.3,  # Low temperature for consistent classification
            max_tokens=1024,
        )

        # Parse response
        classification = self._parse_classification_response(response.content)

        duration = datetime.utcnow() - started_at

        return ClassificationResult(
            classification=classification,
            raw_response=response.content,
            model_used=response.model,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            duration_ms=int(duration.total_seconds() * 1000),
        )

    def classify_heuristic(
        self,
        conflict: ConflictRecord,
    ) -> ConflictClassification:
        """
        Classify a conflict using heuristics (no LLM).

        Useful for quick pre-classification or when LLM is unavailable.

        Args:
            conflict: Conflict record to classify

        Returns:
            ConflictClassification based on heuristics
        """
        # Default classification
        conflict_type = ConflictType.TEXTUAL
        severity = ConflictSeverity.MODERATE
        auto_resolvable = False
        confidence = 0.5
        reasoning = ""

        # Analyze affected artifacts
        total_regions = 0
        for artifact in conflict.affected_artifacts:
            total_regions += len(artifact.conflict_regions)

        # Check for architectural indicators
        architectural_indicators = [
            "interface",
            "api",
            "schema",
            "architecture",
            "design",
            "pattern",
        ]

        has_architectural = False
        for task in conflict.involved_tasks:
            desc = (task.description or "").lower()
            if any(ind in desc for ind in architectural_indicators):
                has_architectural = True
                break

        if has_architectural:
            conflict_type = ConflictType.ARCHITECTURAL
            severity = ConflictSeverity.SIGNIFICANT
            reasoning = "Task descriptions suggest architectural changes"
        elif total_regions == 0:
            # No specific conflict regions - might be semantic
            conflict_type = ConflictType.SEMANTIC
            severity = ConflictSeverity.MODERATE
            reasoning = "No specific line conflicts, possible semantic overlap"
        elif total_regions <= 2:
            # Small number of conflicts
            conflict_type = ConflictType.TEXTUAL
            severity = ConflictSeverity.TRIVIAL
            auto_resolvable = True
            confidence = 0.7
            reasoning = f"Small conflict ({total_regions} regions), likely auto-mergeable"
        else:
            # Multiple conflicts
            conflict_type = ConflictType.TEXTUAL
            severity = ConflictSeverity.MODERATE
            reasoning = f"Multiple conflicts ({total_regions} regions)"

        return ConflictClassification(
            type=conflict_type,
            severity=severity,
            auto_resolvable=auto_resolvable,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _build_classification_prompt(self, conflict: ConflictRecord) -> str:
        """
        Build the classification prompt for the LLM.

        Args:
            conflict: Conflict to classify

        Returns:
            Formatted prompt string
        """
        # Format involved tasks
        tasks_info = []
        for task in conflict.involved_tasks:
            tasks_info.append(
                f"- Task {task.task_id}: {task.description or 'No description'}"
            )
        tasks_str = "\n".join(tasks_info)

        # Format affected artifacts
        artifacts_info = []
        for artifact in conflict.affected_artifacts:
            region_count = len(artifact.conflict_regions)
            artifacts_info.append(
                f"- {artifact.location} ({artifact.artifact_type}): "
                f"{region_count} conflict region(s)"
            )

            # Include sample conflict regions
            for region in artifact.conflict_regions[:2]:  # Max 2 samples
                artifacts_info.append(f"  Region lines {region.start_line}-{region.end_line}:")
                if region.ours_content:
                    ours_preview = region.ours_content[:200]
                    artifacts_info.append(f"    Ours: {ours_preview}...")
                if region.theirs_content:
                    theirs_preview = region.theirs_content[:200]
                    artifacts_info.append(f"    Theirs: {theirs_preview}...")

        artifacts_str = "\n".join(artifacts_info)

        prompt = f"""Classify this conflict between parallel task outputs.

## Involved Tasks
{tasks_str}

## Affected Artifacts
{artifacts_str}

## Detection Method
{conflict.audit.detection_method if conflict.audit else "Unknown"}

## Classification Request
Analyze this conflict and provide classification:

1. **Type**: Is this conflict:
   - `textual`: Same lines/files modified, can potentially be merged at text level
   - `semantic`: Different approaches to same problem, requires understanding intent
   - `architectural`: Incompatible design assumptions, may require redesign

2. **Severity**:
   - `trivial`: Non-overlapping changes, easy auto-merge
   - `moderate`: Overlapping but reconcilable with some analysis
   - `significant`: Substantial differences requiring careful consideration
   - `fundamental`: Deep incompatibility, may need re-decomposition

3. **Auto-resolvable**: Can this be resolved without human intervention?

4. **Confidence**: Your confidence in this classification (0.0-1.0)

Respond with JSON:
```json
{{
    "type": "textual|semantic|architectural",
    "severity": "trivial|moderate|significant|fundamental",
    "auto_resolvable": true|false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification"
}}
```"""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for classification."""
        return """You are a conflict classifier in a distributed agentic task system.
Your role is to analyze conflicts between parallel task outputs and classify them
to help determine the appropriate resolution strategy.

Be precise and conservative in your classifications:
- Only mark as `trivial` if truly non-overlapping
- Mark as `architectural` if design patterns or API contracts differ
- Be cautious with `auto_resolvable` - when in doubt, say false

Provide clear reasoning for your classification."""

    def _parse_classification_response(
        self,
        response: str,
    ) -> ConflictClassification:
        """
        Parse LLM response into ConflictClassification.

        Args:
            response: Raw LLM response

        Returns:
            Parsed ConflictClassification
        """
        try:
            # Extract JSON from response
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

            # Parse fields
            type_str = data.get("type", "textual").lower()
            severity_str = data.get("severity", "moderate").lower()

            try:
                conflict_type = ConflictType(type_str)
            except ValueError:
                conflict_type = ConflictType.TEXTUAL

            try:
                severity = ConflictSeverity(severity_str)
            except ValueError:
                severity = ConflictSeverity.MODERATE

            return ConflictClassification(
                type=conflict_type,
                severity=severity,
                auto_resolvable=data.get("auto_resolvable", False),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse classification response: {e}")
            # Return conservative default
            return ConflictClassification(
                type=ConflictType.SEMANTIC,
                severity=ConflictSeverity.MODERATE,
                auto_resolvable=False,
                confidence=0.3,
                reasoning=f"Parse error: {e}. Defaulting to conservative classification.",
            )

    async def reclassify_with_escalation(
        self,
        conflict: ConflictRecord,
        previous_classification: ConflictClassification,
    ) -> ClassificationResult:
        """
        Reclassify a conflict using a larger/frontier model.

        Used when initial classification confidence is low or
        resolution failed.

        Args:
            conflict: Conflict to reclassify
            previous_classification: Previous classification attempt

        Returns:
            New ClassificationResult
        """
        # Get a larger tier for escalation
        if self._model_tier == "large":
            escalated_tier = "frontier"
        elif self._model_tier == "medium":
            escalated_tier = "large"
        else:
            escalated_tier = "large"

        # Create classifier with escalated tier
        escalated_classifier = ConflictClassifier(
            config=self.config,
            model_tier=escalated_tier,
        )

        # Add context about previous attempt
        conflict_with_context = conflict
        if conflict.audit:
            conflict.audit.escalation_history.append({
                "previous_type": previous_classification.type.value,
                "previous_severity": previous_classification.severity.value,
                "previous_confidence": previous_classification.confidence,
                "escalation_reason": "low_confidence_or_resolution_failure",
            })

        return await escalated_classifier.classify(conflict_with_context)

    async def close(self):
        """Clean up resources."""
        if self._model_client and hasattr(self._model_client, "close"):
            await self._model_client.close()