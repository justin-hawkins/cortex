"""
Base agent class for DATS.

Provides common functionality for all orchestration agents.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from src.config.routing import ModelConfig, get_routing_config
from src.models.base import BaseModelClient, ModelResponse
from src.models.ollama_client import OllamaClient
from src.models.openai_client import OpenAICompatibleClient
from src.models.anthropic_client import AnthropicClient
from src.prompts.renderer import PromptRenderer
from src.config.settings import get_settings
from src.telemetry.llm_tracer import llm_subsystem_context

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent execution."""

    task_id: str
    project_id: str
    model_tier: str
    model_name: str
    context_window: int
    safe_working_limit: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Result from agent execution."""

    success: bool
    content: Any
    model_response: Optional[ModelResponse] = None
    prompt_version: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def execution_time_ms(self) -> Optional[int]:
        """Calculate execution time in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None


class BaseAgent(ABC):
    """
    Abstract base class for DATS agents.

    Agents are responsible for orchestration tasks like decomposition,
    coordination, quality review, and result merging.
    """

    # Agent name used for routing and prompt loading
    agent_name: str = "base"

    def __init__(
        self,
        model_client: Optional[BaseModelClient] = None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            model_client: Optional pre-configured model client
            model_tier: Optional tier override for routing
        """
        self._model_client = model_client
        self._model_tier = model_tier
        self._renderer = PromptRenderer()
        self._settings = get_settings()
        self._routing_config = get_routing_config()

    def get_model_client(self) -> BaseModelClient:
        """
        Get or create model client based on routing configuration.

        Returns:
            Configured model client for this agent
        """
        if self._model_client:
            return self._model_client

        # Get routing for this agent
        routing = self._routing_config.get_agent_routing(self.agent_name)
        if not routing:
            raise ValueError(f"No routing configured for agent: {self.agent_name}")

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

    def _create_client(self, model_config: ModelConfig) -> BaseModelClient:
        """
        Create a model client from configuration.

        Args:
            model_config: Model configuration

        Returns:
            Configured model client
        """
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

    def get_context(self, task_id: str, project_id: str) -> AgentContext:
        """
        Build execution context for this agent.

        Args:
            task_id: Current task ID
            project_id: Current project ID

        Returns:
            AgentContext with model and tier information
        """
        routing = self._routing_config.get_agent_routing(self.agent_name)
        tier_name = self._model_tier or (routing.preferred_tier if routing else "large")
        tier = self._routing_config.get_tier(tier_name)

        model_config = self._routing_config.get_model_for_agent(self.agent_name)

        return AgentContext(
            task_id=task_id,
            project_id=project_id,
            model_tier=tier_name,
            model_name=model_config.name if model_config else "unknown",
            context_window=tier.context_window if tier else 32000,
            safe_working_limit=tier.safe_working_limit if tier else 22000,
        )

    async def execute(
        self,
        task_data: dict[str, Any],
        context: Optional[AgentContext] = None,
    ) -> AgentResult:
        """
        Execute the agent's primary function.

        Args:
            task_data: Task configuration and inputs
            context: Optional execution context

        Returns:
            AgentResult with output and metadata
        """
        started_at = datetime.utcnow()

        try:
            # Build context if not provided
            if context is None:
                context = self.get_context(
                    task_id=task_data.get("id", "unknown"),
                    project_id=task_data.get("project_id", "unknown"),
                )

            # Render prompt
            prompt, prompt_version = self._render_prompt(task_data, context)

            # Set LLM subsystem context for tracing
            # This ensures all LLM calls are tagged with the agent name
            with llm_subsystem_context(
                subsystem=f"agent_{self.agent_name}",
                task_id=context.task_id,
            ):
                # Get model client and generate
                client = self.get_model_client()
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(context),
                    temperature=self._get_temperature(),
                    max_tokens=self._get_max_tokens(context),
                    prompt_template=prompt_version,
                )

            # Process response
            content = self._process_response(response, task_data)

            return AgentResult(
                success=True,
                content=content,
                model_response=response,
                prompt_version=prompt_version,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Agent {self.agent_name} execution failed: {e}")
            return AgentResult(
                success=False,
                content=None,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    @abstractmethod
    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: AgentContext,
    ) -> tuple[str, str]:
        """
        Render the prompt for this agent.

        Args:
            task_data: Task configuration
            context: Execution context

        Returns:
            Tuple of (rendered_prompt, version_hash)
        """
        pass

    def _get_system_prompt(self, context: AgentContext) -> Optional[str]:
        """
        Get system prompt for this agent.

        Override in subclasses for custom system prompts.

        Args:
            context: Execution context

        Returns:
            Optional system prompt string
        """
        return None

    def _get_temperature(self) -> float:
        """
        Get sampling temperature for this agent.

        Override in subclasses for different temperatures.

        Returns:
            Temperature value (0.0-2.0)
        """
        return 0.7

    def _get_max_tokens(self, context: AgentContext) -> int:
        """
        Get maximum output tokens for this agent.

        Args:
            context: Execution context

        Returns:
            Maximum tokens to generate
        """
        # Use quarter of safe working limit for output
        return min(4096, context.safe_working_limit // 4)

    def _process_response(
        self,
        response: ModelResponse,
        task_data: dict[str, Any],
    ) -> Any:
        """
        Process and validate model response.

        Override in subclasses for custom processing.

        Args:
            response: Raw model response
            task_data: Original task data

        Returns:
            Processed content
        """
        return response.content

    async def close(self):
        """Clean up resources."""
        if self._model_client and hasattr(self._model_client, "close"):
            await self._model_client.close()