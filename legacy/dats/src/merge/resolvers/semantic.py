"""
Semantic resolver for merge coordination.

Handles semantic conflicts using LLM-powered analysis to understand
intent and produce merged outputs.
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from src.config.routing import get_routing_config
from src.config.settings import get_settings
from src.merge.config import MergeConfig, get_merge_config
from src.merge.models import (
    ConflictRecord,
    ConflictType,
    MergedOutput,
    ResolutionStatus,
    ResolutionStrategy,
)
from src.merge.resolvers.base import BaseResolver, ResolverResult
from src.models.base import BaseModelClient
from src.models.ollama_client import OllamaClient
from src.models.openai_client import OpenAICompatibleClient
from src.models.anthropic_client import AnthropicClient
from src.prompts.renderer import PromptRenderer

logger = logging.getLogger(__name__)


class SemanticResolver(BaseResolver):
    """
    Resolves conflicts using LLM-powered semantic analysis.

    Used for:
    - Overlapping changes that require understanding intent
    - Different implementations of the same requirement
    - Merging code that can't be auto-merged
    """

    resolver_name = "semantic"

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        model_client: Optional[BaseModelClient] = None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize semantic resolver.

        Args:
            config: Merge configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override for routing
        """
        super().__init__(config)
        self._model_client = model_client
        self._model_tier = model_tier
        self._renderer = PromptRenderer()
        self._settings = get_settings()
        self._routing_config = get_routing_config()

    def get_model_client(self) -> BaseModelClient:
        """
        Get or create model client for semantic resolution.

        Returns:
            Configured model client
        """
        if self._model_client:
            return self._model_client

        # Get routing for merge coordinator
        routing = self._routing_config.get_agent_routing("merge_coordinator")
        if not routing:
            routing = self._routing_config.get_agent_routing("coordinator")
        if not routing:
            raise ValueError("No routing configured for merge_coordinator")

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

    def can_resolve(
        self,
        conflict: ConflictRecord,
    ) -> bool:
        """
        Check if this resolver can handle the conflict.

        Semantic resolver handles:
        - Semantic conflicts
        - Textual conflicts that couldn't be auto-merged
        - Moderate severity conflicts

        Args:
            conflict: Conflict to check

        Returns:
            True if can handle
        """
        if not conflict.classification:
            return True  # Default handler

        classification = conflict.classification

        # Can handle semantic conflicts
        if classification.type == ConflictType.SEMANTIC:
            return True

        # Can handle textual conflicts as fallback
        if classification.type == ConflictType.TEXTUAL:
            return True

        return False

    async def resolve(
        self,
        conflict: ConflictRecord,
    ) -> ResolverResult:
        """
        Resolve a conflict using LLM semantic analysis.

        Args:
            conflict: Conflict to resolve

        Returns:
            ResolverResult with resolution details
        """
        started_at = datetime.utcnow()
        tokens_consumed = 0

        try:
            # Build the merge prompt
            prompt = self._build_merge_prompt(conflict)

            # Get model and generate
            client = self.get_model_client()
            response = await client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.4,  # Moderate temperature for creativity with consistency
                max_tokens=4096,
            )

            tokens_consumed = response.tokens_input + response.tokens_output

            # Parse response
            merge_result = self._parse_merge_response(response.content)

            if not merge_result.get("success", False):
                duration = datetime.utcnow() - started_at
                return ResolverResult(
                    success=False,
                    error=merge_result.get("error", "Merge failed"),
                    needs_escalation=True,
                    escalation_reason=merge_result.get("reason", "LLM could not merge"),
                    tokens_consumed=tokens_consumed,
                    duration_ms=int(duration.total_seconds() * 1000),
                )

            # Create merged output
            merged_output = MergedOutput(
                artifact_type=merge_result.get("artifact_type", "code"),
                content=merge_result.get("merged_content", ""),
                merge_notes=merge_result.get("merge_notes", ""),
            )

            # Create resolution
            resolution = self._create_resolution(
                strategy=ResolutionStrategy.SEMANTIC_MERGE,
                status=ResolutionStatus.RESOLVED,
                merged_output=merged_output,
                resolved_by=response.model,
            )

            # Verify resolution
            is_valid, error = self.verify_resolution(conflict, resolution)
            if not is_valid:
                duration = datetime.utcnow() - started_at
                return ResolverResult(
                    success=False,
                    error=f"Verification failed: {error}",
                    needs_escalation=True,
                    escalation_reason="Semantic merge produced invalid output",
                    tokens_consumed=tokens_consumed,
                    duration_ms=int(duration.total_seconds() * 1000),
                )

            duration = datetime.utcnow() - started_at
            return ResolverResult(
                success=True,
                resolution=resolution,
                tokens_consumed=tokens_consumed,
                duration_ms=int(duration.total_seconds() * 1000),
            )

        except Exception as e:
            logger.error(f"Semantic resolution failed: {e}")
            duration = datetime.utcnow() - started_at
            return ResolverResult(
                success=False,
                error=str(e),
                needs_escalation=True,
                escalation_reason=f"Semantic resolver exception: {e}",
                tokens_consumed=tokens_consumed,
                duration_ms=int(duration.total_seconds() * 1000),
            )

    def _build_merge_prompt(self, conflict: ConflictRecord) -> str:
        """
        Build the merge prompt for the LLM.

        Uses the merge_coordinator prompt template variables.

        Args:
            conflict: Conflict to resolve

        Returns:
            Formatted prompt string
        """
        # Format conflicting outputs
        outputs_info = []
        for i, task in enumerate(conflict.involved_tasks):
            outputs_info.append(f"### Output {i + 1} (Task {task.task_id})")
            outputs_info.append(f"Description: {task.description or 'No description'}")
            if task.output_summary:
                outputs_info.append(f"Summary: {task.output_summary}")
        
        # Add artifact content
        for artifact in conflict.affected_artifacts:
            outputs_info.append(f"\n### File: {artifact.location}")
            for region in artifact.conflict_regions:
                outputs_info.append("#### Version A (Ours):")
                outputs_info.append(f"```\n{region.ours_content}\n```")
                outputs_info.append("#### Version B (Theirs):")
                outputs_info.append(f"```\n{region.theirs_content}\n```")
                if region.base_content:
                    outputs_info.append("#### Base Version:")
                    outputs_info.append(f"```\n{region.base_content}\n```")

        outputs_str = "\n".join(outputs_info)

        # Format task descriptions
        task_descriptions = []
        for task in conflict.involved_tasks:
            task_descriptions.append(
                f"- Task {task.task_id}: {task.description or 'No description'}"
            )
        tasks_str = "\n".join(task_descriptions)

        # Build prompt
        prompt = f"""Merge these conflicting outputs from parallel tasks.

## Conflicting Outputs
{outputs_str}

## Original Task Descriptions
{tasks_str}

## Common Parent Context
{conflict.common_parent_task_id or "Unknown"}

## Project Context
{conflict.lightrag_context or "No additional context available."}

## Classification
- Type: {conflict.classification.type.value if conflict.classification else "Unknown"}
- Severity: {conflict.classification.severity.value if conflict.classification else "Unknown"}

## Instructions
1. Analyze both versions to understand the intent of each
2. Determine if one approach is clearly better, or if they can be combined
3. Produce a merged version that preserves the valuable contributions from both
4. If merging is not possible, explain why

Respond with JSON:
```json
{{
    "success": true|false,
    "artifact_type": "code",
    "merged_content": "<the merged output>",
    "merge_notes": "<explanation of what was combined and how>",
    "error": "<if success is false, explain why>",
    "reason": "<if success is false, why merging wasn't possible>"
}}
```"""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for semantic merging."""
        return """You are a Merge Coordinator in a distributed agentic task system.
Your role is to merge conflicting outputs from parallel tasks.

Guidelines:
1. Understand the intent behind each version before merging
2. Preserve valuable functionality from both sides
3. Ensure the merged output is syntactically correct
4. Maintain code style consistency
5. Document what you combined and why

When you cannot merge:
- Clearly explain why the outputs are incompatible
- Suggest which version might be preferred and why
- Note if the conflict suggests an architectural issue"""

    def _parse_merge_response(
        self,
        response: str,
    ) -> dict[str, Any]:
        """
        Parse LLM merge response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed response dictionary
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

            return json.loads(json_str)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse merge response: {e}")
            return {
                "success": False,
                "error": f"Parse error: {e}",
                "reason": "Could not parse LLM response as JSON",
            }

    async def close(self):
        """Clean up resources."""
        if self._model_client and hasattr(self._model_client, "close"):
            await self._model_client.close()