"""
Architectural resolver for merge coordination.

Handles architectural conflicts by producing redesign recommendations
and managing task invalidation cascades.
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
    ConflictResolution,
    ConflictType,
    HumanDecisionRequest,
    RedesignRecommendation,
    ResolutionStatus,
    ResolutionStrategy,
)
from src.merge.resolvers.base import BaseResolver, ResolverResult
from src.models.base import BaseModelClient
from src.models.ollama_client import OllamaClient
from src.models.openai_client import OpenAICompatibleClient
from src.models.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)


class ArchitecturalResolver(BaseResolver):
    """
    Handles architectural conflicts that cannot be merged.

    Produces redesign recommendations or escalates to human decision.
    Used for:
    - Incompatible design assumptions
    - Contradictory patterns or structures
    - Fundamental conflicts requiring re-decomposition
    """

    resolver_name = "architectural"

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
        model_client: Optional[BaseModelClient] = None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize architectural resolver.

        Args:
            config: Merge configuration
            model_client: Optional pre-configured model client
            model_tier: Optional tier override for routing
        """
        super().__init__(config)
        self._model_client = model_client
        self._model_tier = model_tier
        self._settings = get_settings()
        self._routing_config = get_routing_config()

    def get_model_client(self) -> BaseModelClient:
        """
        Get or create model client for architectural analysis.

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

        # Prefer larger models for architectural decisions
        tier_name = self._model_tier or "large"
        tier = self._routing_config.get_tier(tier_name)
        if not tier:
            tier_name = routing.preferred_tier
            tier = self._routing_config.get_tier(tier_name)
        if not tier:
            raise ValueError(f"Unknown tier: {tier_name}")

        # Get model config
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

        Architectural resolver handles:
        - Architectural conflicts
        - Fundamental severity conflicts
        - Conflicts requiring redesign

        Args:
            conflict: Conflict to check

        Returns:
            True if can handle
        """
        if not conflict.classification:
            return False

        classification = conflict.classification

        # Specifically handles architectural conflicts
        if classification.type == ConflictType.ARCHITECTURAL:
            return True

        return False

    async def resolve(
        self,
        conflict: ConflictRecord,
    ) -> ResolverResult:
        """
        Analyze architectural conflict and produce recommendation.

        This resolver doesn't "resolve" in the traditional sense.
        Instead, it produces a redesign recommendation or escalates
        to human decision.

        Args:
            conflict: Conflict to analyze

        Returns:
            ResolverResult with redesign recommendation or human decision
        """
        started_at = datetime.utcnow()
        tokens_consumed = 0

        try:
            # Decide approach based on configuration
            if self.config.resolution.prefer_human_for_architectural:
                # Generate analysis for human decision
                result = await self._generate_human_decision(conflict)
            else:
                # Generate redesign recommendation
                result = await self._generate_redesign_recommendation(conflict)

            tokens_consumed = result.get("tokens_consumed", 0)
            duration = datetime.utcnow() - started_at

            if result.get("needs_human", False):
                # Create human decision request
                resolution = ConflictResolution(
                    strategy=ResolutionStrategy.HUMAN_DECISION,
                    status=ResolutionStatus.ESCALATED,
                    human_decision_request=HumanDecisionRequest(
                        question=result.get("question", "How should this conflict be resolved?"),
                        options=result.get("options", []),
                        recommendation=result.get("recommendation", ""),
                        context=result.get("context", ""),
                    ),
                    started_at=started_at,
                )

                return ResolverResult(
                    success=True,  # Successfully produced decision request
                    resolution=resolution,
                    needs_escalation=True,
                    escalation_reason="Architectural conflict requires human decision",
                    tokens_consumed=tokens_consumed,
                    duration_ms=int(duration.total_seconds() * 1000),
                )
            else:
                # Create redesign recommendation
                resolution = ConflictResolution(
                    strategy=ResolutionStrategy.REDESIGN,
                    status=ResolutionStatus.RESOLVED,
                    redesign_recommendation=RedesignRecommendation(
                        problem=result.get("problem", ""),
                        suggested_approach=result.get("suggested_approach", ""),
                        tasks_to_invalidate=result.get("tasks_to_invalidate", []),
                        context_for_redecomposition=result.get("context", ""),
                    ),
                    resolved_by=result.get("model_used", "architectural_resolver"),
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )

                return ResolverResult(
                    success=True,
                    resolution=resolution,
                    tokens_consumed=tokens_consumed,
                    duration_ms=int(duration.total_seconds() * 1000),
                )

        except Exception as e:
            logger.error(f"Architectural resolution failed: {e}")
            duration = datetime.utcnow() - started_at
            return ResolverResult(
                success=False,
                error=str(e),
                needs_escalation=True,
                escalation_reason=f"Architectural resolver exception: {e}",
                tokens_consumed=tokens_consumed,
                duration_ms=int(duration.total_seconds() * 1000),
            )

    async def _generate_redesign_recommendation(
        self,
        conflict: ConflictRecord,
    ) -> dict[str, Any]:
        """
        Generate a redesign recommendation using LLM.

        Args:
            conflict: Conflict to analyze

        Returns:
            Dictionary with redesign details
        """
        prompt = self._build_redesign_prompt(conflict)

        client = self.get_model_client()
        response = await client.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.5,
            max_tokens=2048,
        )

        tokens = response.tokens_input + response.tokens_output

        # Parse response
        result = self._parse_redesign_response(response.content)
        result["tokens_consumed"] = tokens
        result["model_used"] = response.model
        result["needs_human"] = False

        return result

    async def _generate_human_decision(
        self,
        conflict: ConflictRecord,
    ) -> dict[str, Any]:
        """
        Generate analysis for human decision.

        Args:
            conflict: Conflict to analyze

        Returns:
            Dictionary with decision request details
        """
        prompt = self._build_decision_prompt(conflict)

        client = self.get_model_client()
        response = await client.generate(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            temperature=0.4,
            max_tokens=2048,
        )

        tokens = response.tokens_input + response.tokens_output

        # Parse response
        result = self._parse_decision_response(response.content)
        result["tokens_consumed"] = tokens
        result["needs_human"] = True

        return result

    def _build_redesign_prompt(self, conflict: ConflictRecord) -> str:
        """Build prompt for redesign recommendation."""
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
            artifacts_info.append(f"- {artifact.location} ({artifact.artifact_type})")
        artifacts_str = "\n".join(artifacts_info)

        prompt = f"""Analyze this architectural conflict and recommend a redesign approach.

## Conflicting Tasks
{tasks_str}

## Affected Artifacts
{artifacts_str}

## Classification
- Type: {conflict.classification.type.value if conflict.classification else "Unknown"}
- Severity: {conflict.classification.severity.value if conflict.classification else "Unknown"}
- Reasoning: {conflict.classification.reasoning if conflict.classification else "Unknown"}

## Conflict Details
{self._format_conflict_details(conflict)}

## Instructions
Analyze why these tasks produced incompatible outputs and recommend how to:
1. Identify the root cause of the architectural conflict
2. Suggest how to re-decompose the parent task to avoid this conflict
3. List which task outputs should be invalidated
4. Provide context for the re-decomposition

Respond with JSON:
```json
{{
    "problem": "<describe the architectural incompatibility>",
    "root_cause": "<what caused the conflict at decomposition time>",
    "suggested_approach": "<how to re-decompose to avoid conflict>",
    "tasks_to_invalidate": ["<task_id1>", "<task_id2>"],
    "context": "<context to include when re-decomposing>",
    "confidence": 0.0-1.0
}}
```"""

        return prompt

    def _build_decision_prompt(self, conflict: ConflictRecord) -> str:
        """Build prompt for human decision request."""
        # Format involved tasks
        tasks_info = []
        for task in conflict.involved_tasks:
            tasks_info.append(
                f"- Task {task.task_id}: {task.description or 'No description'}"
            )
        tasks_str = "\n".join(tasks_info)

        prompt = f"""Analyze this architectural conflict and prepare options for human decision.

## Conflicting Tasks
{tasks_str}

## Conflict Details
{self._format_conflict_details(conflict)}

## Classification
- Type: {conflict.classification.type.value if conflict.classification else "Unknown"}
- Severity: {conflict.classification.severity.value if conflict.classification else "Unknown"}

## Instructions
Prepare a clear decision request for a human reviewer:
1. Frame the key question they need to answer
2. Present 2-4 options with their implications
3. Provide your recommendation with reasoning

Respond with JSON:
```json
{{
    "question": "<clear question for the human>",
    "options": [
        {{
            "option": "<description of choice A>",
            "implications": "<what choosing this means>"
        }},
        {{
            "option": "<description of choice B>",
            "implications": "<what choosing this means>"
        }}
    ],
    "recommendation": "<your recommended choice and why>",
    "context": "<additional context for the decision>"
}}
```"""

        return prompt

    def _format_conflict_details(self, conflict: ConflictRecord) -> str:
        """Format conflict details for prompts."""
        details = []
        for artifact in conflict.affected_artifacts:
            details.append(f"### {artifact.location}")
            for region in artifact.conflict_regions[:3]:  # Limit regions
                details.append(f"Lines {region.start_line}-{region.end_line}:")
                if region.ours_content:
                    preview = region.ours_content[:300]
                    details.append(f"Version A:\n```\n{preview}\n```")
                if region.theirs_content:
                    preview = region.theirs_content[:300]
                    details.append(f"Version B:\n```\n{preview}\n```")
        return "\n".join(details)

    def _get_system_prompt(self) -> str:
        """Get system prompt for architectural analysis."""
        return """You are an Architectural Analyst in a distributed agentic task system.
Your role is to analyze architectural conflicts and provide clear recommendations.

When analyzing conflicts:
1. Identify the fundamental incompatibility (not just symptoms)
2. Consider the original decomposition and why it failed
3. Propose concrete ways to avoid the conflict in re-decomposition
4. Be specific about which outputs are affected

When preparing human decisions:
1. Frame the question clearly and neutrally
2. Present options with honest trade-offs
3. Provide a clear recommendation with reasoning
4. Include enough context for an informed decision"""

    def _parse_redesign_response(
        self,
        response: str,
    ) -> dict[str, Any]:
        """Parse redesign recommendation response."""
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
            logger.warning(f"Failed to parse redesign response: {e}")
            return {
                "problem": "Failed to analyze conflict",
                "suggested_approach": "Manual review required",
                "tasks_to_invalidate": [],
                "context": f"Parse error: {e}",
            }

    def _parse_decision_response(
        self,
        response: str,
    ) -> dict[str, Any]:
        """Parse human decision response."""
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
            logger.warning(f"Failed to parse decision response: {e}")
            return {
                "question": "How should this architectural conflict be resolved?",
                "options": [
                    {"option": "Review manually", "implications": "Requires human analysis"},
                ],
                "recommendation": "Manual review recommended due to parse error",
                "context": f"Parse error: {e}",
            }

    def get_invalidation_cascade(
        self,
        conflict: ConflictRecord,
        redesign: RedesignRecommendation,
    ) -> list[str]:
        """
        Compute full invalidation cascade from redesign recommendation.

        Traces through provenance DAG to find all dependent outputs.

        Args:
            conflict: Original conflict
            redesign: Redesign recommendation with tasks to invalidate

        Returns:
            List of all artifact IDs that should be marked as tainted
        """
        # Start with directly invalidated tasks
        invalidated = set(redesign.tasks_to_invalidate)

        # Add artifacts from the conflict
        for artifact in conflict.affected_artifacts:
            invalidated.add(artifact.artifact_id)

        # In production, would trace through provenance DAG
        # to find all dependent outputs
        # For now, return direct invalidations

        return list(invalidated)

    async def close(self):
        """Clean up resources."""
        if self._model_client and hasattr(self._model_client, "close"):
            await self._model_client.close()