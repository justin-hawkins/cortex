"""
Routing configuration loader for DATS.

Parses routing_config.yaml into strongly-typed dataclasses.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    endpoint: str
    type: str  # ollama, openai_compatible, anthropic
    priority: int = 1


@dataclass
class ModelTier:
    """Configuration for a model tier (tiny, small, large, frontier)."""

    context_window: int
    safe_working_limit: int
    models: list[ModelConfig] = field(default_factory=list)

    def get_primary_model(self) -> Optional[ModelConfig]:
        """Get the highest priority (lowest number) model."""
        if not self.models:
            return None
        return min(self.models, key=lambda m: m.priority)

    def get_model_by_name(self, name: str) -> Optional[ModelConfig]:
        """Get a model by name."""
        for model in self.models:
            if model.name == name:
                return model
        return None


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""

    model: str
    endpoint: str
    type: str


@dataclass
class AgentRouting:
    """Routing configuration for an agent role."""

    preferred_tier: str
    preferred_model: Optional[str] = None
    fallback_tier: Optional[str] = None
    min_tier_vs_worker: Optional[str] = None


@dataclass
class RoutingConfig:
    """Complete routing configuration."""

    model_tiers: dict[str, ModelTier]
    embedding: EmbeddingConfig
    agent_routing: dict[str, AgentRouting]

    def get_tier(self, tier_name: str) -> Optional[ModelTier]:
        """Get a model tier by name."""
        return self.model_tiers.get(tier_name)

    def get_agent_routing(self, agent_name: str) -> Optional[AgentRouting]:
        """Get routing config for an agent."""
        return self.agent_routing.get(agent_name)

    def get_model_for_agent(self, agent_name: str) -> Optional[ModelConfig]:
        """Get the recommended model for an agent."""
        routing = self.get_agent_routing(agent_name)
        if not routing:
            return None

        tier = self.get_tier(routing.preferred_tier)
        if not tier:
            return None

        # Try preferred model first
        if routing.preferred_model:
            model = tier.get_model_by_name(routing.preferred_model)
            if model:
                return model

        # Fall back to primary model in tier
        return tier.get_primary_model()


def load_routing_config(config_path: str | Path) -> RoutingConfig:
    """
    Load routing configuration from YAML file.

    Args:
        config_path: Path to routing_config.yaml

    Returns:
        Parsed RoutingConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Routing config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse model tiers
    model_tiers = {}
    for tier_name, tier_data in data.get("model_tiers", {}).items():
        models = [
            ModelConfig(
                name=m["name"],
                endpoint=m["endpoint"],
                type=m["type"],
                priority=m.get("priority", 1),
            )
            for m in tier_data.get("models", [])
        ]
        model_tiers[tier_name] = ModelTier(
            context_window=tier_data["context_window"],
            safe_working_limit=tier_data["safe_working_limit"],
            models=models,
        )

    # Parse embedding config
    embedding_data = data.get("embedding", {})
    embedding = EmbeddingConfig(
        model=embedding_data.get("model", ""),
        endpoint=embedding_data.get("endpoint", ""),
        type=embedding_data.get("type", "ollama"),
    )

    # Parse agent routing
    agent_routing = {}
    for agent_name, routing_data in data.get("agent_routing", {}).items():
        agent_routing[agent_name] = AgentRouting(
            preferred_tier=routing_data["preferred_tier"],
            preferred_model=routing_data.get("preferred_model"),
            fallback_tier=routing_data.get("fallback_tier"),
            min_tier_vs_worker=routing_data.get("min_tier_vs_worker"),
        )

    return RoutingConfig(
        model_tiers=model_tiers,
        embedding=embedding,
        agent_routing=agent_routing,
    )


# Cached singleton instance
_routing_config: Optional[RoutingConfig] = None


def get_routing_config(config_path: Optional[str] = None) -> RoutingConfig:
    """
    Get cached routing configuration.

    Args:
        config_path: Optional path to config file. If not provided,
                    uses default from settings.

    Returns:
        Cached RoutingConfig instance
    """
    global _routing_config

    if _routing_config is None:
        if config_path is None:
            from src.config.settings import get_settings

            config_path = get_settings().routing_config_path

        _routing_config = load_routing_config(config_path)

    return _routing_config