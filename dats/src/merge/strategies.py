"""
Resolution strategy selection for merge coordination.

Maps conflict classifications to appropriate resolution strategies
and handles escalation logic.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Type

from src.merge.config import MergeConfig, get_merge_config
from src.merge.models import (
    ConflictClassification,
    ConflictRecord,
    ConflictSeverity,
    ConflictType,
    ResolutionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """Recommendation for resolution strategy."""

    strategy: ResolutionStrategy
    confidence: float
    reasoning: str
    should_escalate: bool = False
    escalation_target: Optional[str] = None  # "larger_model" or "human"


class ResolutionStrategySelector:
    """
    Selects appropriate resolution strategy based on conflict classification.

    Implements the resolution flow:
    - Trivial textual → Auto-merge
    - Moderate textual/semantic → Semantic merge with LLM
    - Significant/Fundamental → Redesign or human decision
    - Uncertain → Escalate to larger model or human
    """

    def __init__(
        self,
        config: Optional[MergeConfig] = None,
    ):
        """
        Initialize strategy selector.

        Args:
            config: Merge configuration
        """
        self.config = config or get_merge_config()

    def select_strategy(
        self,
        conflict: ConflictRecord,
        classification: ConflictClassification,
        attempt_number: int = 1,
    ) -> StrategyRecommendation:
        """
        Select resolution strategy for a classified conflict.

        Args:
            conflict: Conflict record
            classification: Conflict classification
            attempt_number: Which attempt this is (for escalation logic)

        Returns:
            StrategyRecommendation with strategy and reasoning
        """
        # Check if we've exceeded max attempts
        max_attempts = self.config.escalation.max_resolution_attempts
        if attempt_number > max_attempts:
            return self._recommend_human_decision(
                classification,
                f"Exceeded max resolution attempts ({max_attempts})",
            )

        # Low confidence classifications need escalation
        if classification.confidence < 0.5:
            return self._recommend_escalation(
                classification,
                "Low classification confidence",
                attempt_number,
            )

        # Route based on conflict type and severity
        if classification.type == ConflictType.TEXTUAL:
            return self._handle_textual_conflict(classification, attempt_number)
        elif classification.type == ConflictType.SEMANTIC:
            return self._handle_semantic_conflict(classification, attempt_number)
        elif classification.type == ConflictType.ARCHITECTURAL:
            return self._handle_architectural_conflict(classification, attempt_number)
        else:
            # Unknown type - be conservative
            return self._recommend_human_decision(
                classification,
                "Unknown conflict type",
            )

    def _handle_textual_conflict(
        self,
        classification: ConflictClassification,
        attempt_number: int,
    ) -> StrategyRecommendation:
        """
        Handle textual conflict strategy selection.

        Args:
            classification: Conflict classification
            attempt_number: Which attempt this is

        Returns:
            Strategy recommendation
        """
        severity = classification.severity
        auto_max = self.config.resolution.auto_merge_max_severity
        semantic_max = self.config.resolution.semantic_merge_max_severity

        # Trivial conflicts can be auto-merged
        if severity == ConflictSeverity.TRIVIAL:
            if classification.auto_resolvable:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.AUTO_MERGE,
                    confidence=classification.confidence,
                    reasoning="Trivial textual conflict, auto-merge possible",
                )
            else:
                # Try semantic merge even for trivial
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    confidence=classification.confidence * 0.9,
                    reasoning="Trivial conflict but not auto-resolvable, using semantic merge",
                )

        # Moderate conflicts use semantic merge
        if severity == ConflictSeverity.MODERATE:
            if self._severity_within_limit(severity, semantic_max):
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    confidence=classification.confidence * 0.85,
                    reasoning="Moderate textual conflict, semantic merge appropriate",
                )
            else:
                return self._recommend_escalation(
                    classification,
                    "Moderate conflict exceeds semantic merge limit",
                    attempt_number,
                )

        # Significant or fundamental - may need redesign or human
        if severity in (ConflictSeverity.SIGNIFICANT, ConflictSeverity.FUNDAMENTAL):
            # First attempt: try semantic merge
            if attempt_number == 1:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    confidence=classification.confidence * 0.7,
                    reasoning=f"{severity.value} textual conflict, attempting semantic merge first",
                )
            else:
                return self._recommend_human_decision(
                    classification,
                    f"{severity.value} textual conflict after failed semantic merge",
                )

        # Default fallback
        return StrategyRecommendation(
            strategy=ResolutionStrategy.SEMANTIC_MERGE,
            confidence=0.5,
            reasoning="Default textual conflict handling",
        )

    def _handle_semantic_conflict(
        self,
        classification: ConflictClassification,
        attempt_number: int,
    ) -> StrategyRecommendation:
        """
        Handle semantic conflict strategy selection.

        Args:
            classification: Conflict classification
            attempt_number: Which attempt this is

        Returns:
            Strategy recommendation
        """
        severity = classification.severity
        semantic_max = self.config.resolution.semantic_merge_max_severity

        # Trivial semantic conflicts - still use semantic merge
        if severity == ConflictSeverity.TRIVIAL:
            return StrategyRecommendation(
                strategy=ResolutionStrategy.SEMANTIC_MERGE,
                confidence=classification.confidence,
                reasoning="Trivial semantic conflict, LLM can resolve",
            )

        # Moderate semantic conflicts
        if severity == ConflictSeverity.MODERATE:
            if self._severity_within_limit(severity, semantic_max):
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    confidence=classification.confidence * 0.8,
                    reasoning="Moderate semantic conflict, LLM analysis needed",
                )
            else:
                return self._recommend_escalation(
                    classification,
                    "Moderate semantic conflict exceeds limit",
                    attempt_number,
                )

        # Significant semantic conflicts
        if severity == ConflictSeverity.SIGNIFICANT:
            if attempt_number == 1 and self.config.escalation.escalate_to_frontier_before_human:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    confidence=classification.confidence * 0.6,
                    reasoning="Significant semantic conflict, attempting with frontier model",
                    should_escalate=True,
                    escalation_target="larger_model",
                )
            else:
                return self._recommend_human_decision(
                    classification,
                    "Significant semantic conflict requires human judgment",
                )

        # Fundamental semantic conflicts - likely need redesign
        if severity == ConflictSeverity.FUNDAMENTAL:
            return StrategyRecommendation(
                strategy=ResolutionStrategy.REDESIGN,
                confidence=classification.confidence * 0.7,
                reasoning="Fundamental semantic conflict suggests architectural mismatch",
            )

        # Default
        return StrategyRecommendation(
            strategy=ResolutionStrategy.SEMANTIC_MERGE,
            confidence=0.5,
            reasoning="Default semantic conflict handling",
        )

    def _handle_architectural_conflict(
        self,
        classification: ConflictClassification,
        attempt_number: int,
    ) -> StrategyRecommendation:
        """
        Handle architectural conflict strategy selection.

        Args:
            classification: Conflict classification
            attempt_number: Which attempt this is

        Returns:
            Strategy recommendation
        """
        severity = classification.severity

        # Even trivial architectural conflicts deserve attention
        if severity == ConflictSeverity.TRIVIAL:
            return StrategyRecommendation(
                strategy=ResolutionStrategy.SEMANTIC_MERGE,
                confidence=classification.confidence * 0.7,
                reasoning="Trivial architectural conflict, may be reconcilable",
            )

        # Moderate architectural - attempt redesign recommendation
        if severity == ConflictSeverity.MODERATE:
            if self.config.resolution.prefer_human_for_architectural:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.REDESIGN,
                    confidence=classification.confidence * 0.7,
                    reasoning="Moderate architectural conflict, producing redesign recommendation",
                )
            else:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.SEMANTIC_MERGE,
                    confidence=classification.confidence * 0.5,
                    reasoning="Moderate architectural conflict, attempting semantic resolution",
                )

        # Significant or fundamental - redesign or human
        if severity in (ConflictSeverity.SIGNIFICANT, ConflictSeverity.FUNDAMENTAL):
            if self.config.resolution.prefer_human_for_architectural:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.HUMAN_DECISION,
                    confidence=classification.confidence,
                    reasoning=f"{severity.value} architectural conflict needs human decision",
                    should_escalate=True,
                    escalation_target="human",
                )
            else:
                return StrategyRecommendation(
                    strategy=ResolutionStrategy.REDESIGN,
                    confidence=classification.confidence * 0.6,
                    reasoning=f"{severity.value} architectural conflict, recommending redesign",
                )

        # Default for architectural
        return StrategyRecommendation(
            strategy=ResolutionStrategy.REDESIGN,
            confidence=0.5,
            reasoning="Default architectural conflict handling",
        )

    def _recommend_escalation(
        self,
        classification: ConflictClassification,
        reason: str,
        attempt_number: int,
    ) -> StrategyRecommendation:
        """
        Recommend escalation to larger model or human.

        Args:
            classification: Conflict classification
            reason: Why escalation is needed
            attempt_number: Which attempt this is

        Returns:
            Strategy recommendation with escalation flag
        """
        if (
            self.config.escalation.escalate_to_frontier_before_human
            and attempt_number < self.config.escalation.max_resolution_attempts
        ):
            return StrategyRecommendation(
                strategy=ResolutionStrategy.SEMANTIC_MERGE,
                confidence=0.4,
                reasoning=f"{reason}, escalating to larger model",
                should_escalate=True,
                escalation_target="larger_model",
            )
        else:
            return self._recommend_human_decision(
                classification,
                f"{reason}, escalating to human",
            )

    def _recommend_human_decision(
        self,
        classification: ConflictClassification,
        reason: str,
    ) -> StrategyRecommendation:
        """
        Recommend human decision.

        Args:
            classification: Conflict classification
            reason: Why human decision is needed

        Returns:
            Strategy recommendation for human decision
        """
        return StrategyRecommendation(
            strategy=ResolutionStrategy.HUMAN_DECISION,
            confidence=classification.confidence,
            reasoning=reason,
            should_escalate=True,
            escalation_target="human",
        )

    def _severity_within_limit(
        self,
        severity: ConflictSeverity,
        limit: ConflictSeverity,
    ) -> bool:
        """
        Check if severity is within the allowed limit.

        Args:
            severity: Actual severity
            limit: Maximum allowed severity

        Returns:
            True if within limit
        """
        severity_order = {
            ConflictSeverity.TRIVIAL: 0,
            ConflictSeverity.MODERATE: 1,
            ConflictSeverity.SIGNIFICANT: 2,
            ConflictSeverity.FUNDAMENTAL: 3,
        }
        return severity_order[severity] <= severity_order[limit]

    def get_resolver_for_strategy(
        self,
        strategy: ResolutionStrategy,
    ) -> str:
        """
        Get the resolver class name for a strategy.

        Args:
            strategy: Resolution strategy

        Returns:
            Resolver class name
        """
        resolver_map = {
            ResolutionStrategy.AUTO_MERGE: "TextualResolver",
            ResolutionStrategy.SEMANTIC_MERGE: "SemanticResolver",
            ResolutionStrategy.REDESIGN: "ArchitecturalResolver",
            ResolutionStrategy.HUMAN_DECISION: "HumanDecisionHandler",
        }
        return resolver_map.get(strategy, "SemanticResolver")