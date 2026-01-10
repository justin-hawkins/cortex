# File: services/model-gateway/src/services/failover.py
"""Failover service for handling provider failures."""

import logging
from typing import Any

from src.config import get_gateway_config
from src.providers.base import BaseProvider, ModelInfo

logger = logging.getLogger(__name__)


class FailoverService:
    """
    Service for handling failover between providers and models.
    
    Strategies:
    - same_tier: Try another model in the same tier
    - tier_up: Try a model in a higher tier
    - provider_fallback: Try a different provider entirely
    """

    def __init__(self) -> None:
        """Initialize the failover service."""
        self._tier_order: list[str] = []
        self._strategies: list[str] = []
        self._initialized = False

    def initialize(self) -> None:
        """Load failover configuration."""
        if self._initialized:
            return

        config = get_gateway_config()
        failover_config = config.get("failover", {})
        
        self._strategies = failover_config.get("strategies", [
            "same_tier",
            "tier_up",
            "provider_fallback",
        ])
        
        self._tier_order = failover_config.get("tier_order", [
            "tiny",
            "small",
            "medium",
            "large",
            "frontier",
        ])
        
        self._initialized = True
        logger.info(f"Failover service initialized with strategies: {self._strategies}")

    def get_tier_index(self, tier: str) -> int:
        """Get the index of a tier in the tier order."""
        try:
            return self._tier_order.index(tier)
        except ValueError:
            return -1

    def find_alternatives(
        self,
        failed_model: str,
        failed_provider: str,
        all_models: list[ModelInfo],
        failed_tier: str | None = None,
    ) -> list[ModelInfo]:
        """
        Find alternative models when one fails.
        
        Args:
            failed_model: The model that failed
            failed_provider: The provider that failed
            all_models: List of all available models
            failed_tier: Optional tier of the failed model
            
        Returns:
            List of alternative models to try, in priority order
        """
        self.initialize()
        
        alternatives: list[ModelInfo] = []
        
        # Filter out the failed model
        available = [m for m in all_models if m.name != failed_model and m.status == "available"]
        
        if not available:
            return []

        # Determine the tier of the failed model if not provided
        if failed_tier is None:
            for m in all_models:
                if m.name == failed_model:
                    failed_tier = m.tier
                    break

        failed_tier_idx = self.get_tier_index(failed_tier) if failed_tier else -1

        for strategy in self._strategies:
            if strategy == "same_tier" and failed_tier:
                # Find models in the same tier (prefer different provider)
                same_tier = [
                    m for m in available
                    if m.tier == failed_tier and m.name not in [a.name for a in alternatives]
                ]
                # Sort to prefer different providers first
                same_tier.sort(key=lambda m: 0 if m.provider != failed_provider else 1)
                alternatives.extend(same_tier)

            elif strategy == "tier_up" and failed_tier_idx >= 0:
                # Find models in higher tiers
                higher_tiers = [
                    m for m in available
                    if self.get_tier_index(m.tier) > failed_tier_idx
                    and m.name not in [a.name for a in alternatives]
                ]
                # Sort by tier (closest first) and prefer different providers
                higher_tiers.sort(key=lambda m: (
                    self.get_tier_index(m.tier),
                    0 if m.provider != failed_provider else 1,
                ))
                alternatives.extend(higher_tiers)

            elif strategy == "provider_fallback":
                # Find any model from a different provider
                different_provider = [
                    m for m in available
                    if m.provider != failed_provider
                    and m.name not in [a.name for a in alternatives]
                ]
                # Sort by tier similarity
                if failed_tier_idx >= 0:
                    different_provider.sort(
                        key=lambda m: abs(self.get_tier_index(m.tier) - failed_tier_idx)
                    )
                alternatives.extend(different_provider)

        return alternatives

    def should_retry(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int = 3,
    ) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            error: The exception that occurred
            attempt: Current attempt number (1-indexed)
            max_attempts: Maximum retry attempts
            
        Returns:
            True if should retry, False otherwise
        """
        if attempt >= max_attempts:
            return False

        # Retry on connection errors, timeouts
        error_str = str(type(error).__name__).lower()
        retryable_errors = [
            "timeout",
            "connection",
            "network",
            "unavailable",
            "servererror",
        ]
        
        return any(err in error_str for err in retryable_errors)


# Global failover service instance
_failover_service: FailoverService | None = None


def get_failover_service() -> FailoverService:
    """Get the global failover service instance."""
    global _failover_service
    if _failover_service is None:
        _failover_service = FailoverService()
    return _failover_service