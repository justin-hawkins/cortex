"""
Base worker class for DATS.

Provides common functionality for all task execution workers.
"""

import json
import logging
from abc import abstractmethod
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
class WorkerContext:
    """Context for worker execution."""

    task_id: str
    project_id: str
    domain: str
    model_tier: str
    model_name: str
    context_window: int
    safe_working_limit: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerResult:
    """Result from worker execution."""

    success: bool
    content: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
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


class BaseWorker:
    """
    Base class for DATS workers.

    Workers execute specific task types within domains like
    code generation, documentation, UI design, etc.
    """

    # Worker name used for prompt loading
    worker_name: str = "base"
    # Domain this worker handles
    domain: str = "code-general"

    def __init__(
        self,
        model_client: Optional[BaseModelClient] = None,
        model_tier: Optional[str] = None,
    ):
        """
        Initialize the worker.

        Args:
            model_client: Optional pre-configured model client
            model_tier: Optional tier override for routing
        """
        self._model_client = model_client
        self._model_tier = model_tier
        self._renderer = PromptRenderer()
        self._settings = get_settings()
        self._routing_config = get_routing_config()

    def get_model_client(self, tier: Optional[str] = None) -> BaseModelClient:
        """
        Get or create model client for specified tier.

        Args:
            tier: Optional tier override

        Returns:
            Configured model client
        """
        if self._model_client and not tier:
            return self._model_client

        # Use specified tier or default
        tier_name = tier or self._model_tier or "small"
        tier_config = self._routing_config.get_tier(tier_name)
        
        if not tier_config:
            raise ValueError(f"Unknown tier: {tier_name}")

        model_config = tier_config.get_primary_model()
        if not model_config:
            raise ValueError(f"No model found for tier: {tier_name}")

        return self._create_client(model_config)

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

    def get_context(
        self,
        task_id: str,
        project_id: str,
        tier: Optional[str] = None,
    ) -> WorkerContext:
        """
        Build execution context for this worker.

        Args:
            task_id: Current task ID
            project_id: Current project ID
            tier: Optional tier override

        Returns:
            WorkerContext with model and tier information
        """
        tier_name = tier or self._model_tier or "small"
        tier_config = self._routing_config.get_tier(tier_name)
        model_config = tier_config.get_primary_model() if tier_config else None

        return WorkerContext(
            task_id=task_id,
            project_id=project_id,
            domain=self.domain,
            model_tier=tier_name,
            model_name=model_config.name if model_config else "unknown",
            context_window=tier_config.context_window if tier_config else 32000,
            safe_working_limit=tier_config.safe_working_limit if tier_config else 22000,
        )

    async def execute(
        self,
        task_data: dict[str, Any],
        context: Optional[WorkerContext] = None,
    ) -> WorkerResult:
        """
        Execute the worker's primary function.

        Args:
            task_data: Task configuration and inputs
            context: Optional execution context

        Returns:
            WorkerResult with output and metadata
        """
        started_at = datetime.utcnow()

        try:
            # Build context if not provided
            if context is None:
                tier = task_data.get("routing", {}).get("tier", self._model_tier)
                context = self.get_context(
                    task_id=task_data.get("id", "unknown"),
                    project_id=task_data.get("project_id", "unknown"),
                    tier=tier,
                )

            # Render prompt
            prompt, prompt_version = self._render_prompt(task_data, context)

            # Count RAG context items if present
            context_items = 0
            if task_data.get("lightrag_context"):
                # Estimate context items by counting context sections
                context_items = task_data.get("lightrag_context", "").count("---") + 1

            # Set LLM subsystem context for tracing
            # This ensures all LLM calls are tagged with the worker domain
            with llm_subsystem_context(
                subsystem=f"worker_{self.domain}",
                task_id=context.task_id,
            ):
                # Get model client and generate
                client = self.get_model_client(context.model_tier)
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(context),
                    temperature=self._get_temperature(),
                    max_tokens=self._get_max_tokens(context),
                    prompt_template=prompt_version,
                    context_items=context_items,
                )

            # Process response
            content, artifacts = self._process_response(response, task_data)

            return WorkerResult(
                success=True,
                content=content,
                artifacts=artifacts,
                model_response=response,
                prompt_version=prompt_version,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Worker {self.worker_name} execution failed: {e}")
            return WorkerResult(
                success=False,
                content="",
                error=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )

    @abstractmethod
    def _render_prompt(
        self,
        task_data: dict[str, Any],
        context: WorkerContext,
    ) -> tuple[str, str]:
        """
        Render the prompt for this worker.

        Args:
            task_data: Task configuration
            context: Execution context

        Returns:
            Tuple of (rendered_prompt, version_hash)
        """
        pass

    def _get_system_prompt(self, context: WorkerContext) -> Optional[str]:
        """
        Get system prompt for this worker.

        Override in subclasses for custom system prompts.

        Args:
            context: Execution context

        Returns:
            Optional system prompt string
        """
        return None

    def _get_temperature(self) -> float:
        """
        Get sampling temperature for this worker.

        Override in subclasses for different temperatures.

        Returns:
            Temperature value (0.0-2.0)
        """
        return 0.7

    def _get_max_tokens(self, context: WorkerContext) -> int:
        """
        Get maximum output tokens for this worker.

        Args:
            context: Execution context

        Returns:
            Maximum tokens to generate
        """
        # Use half of safe working limit for output
        return min(8192, context.safe_working_limit // 2)

    def _process_response(
        self,
        response: ModelResponse,
        task_data: dict[str, Any],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Process and extract artifacts from model response.

        Override in subclasses for custom processing.

        Args:
            response: Raw model response
            task_data: Original task data

        Returns:
            Tuple of (content, artifacts)
        """
        content = response.content
        artifacts = []

        # Try to extract code blocks as artifacts
        if "```" in content:
            artifacts = self._extract_code_blocks(content)

        return content, artifacts

    def _extract_code_blocks(self, content: str) -> list[dict[str, Any]]:
        """
        Extract code blocks from markdown content.

        Args:
            content: Content with code blocks

        Returns:
            List of artifact dictionaries
        """
        artifacts = []
        lines = content.split("\n")
        in_code_block = False
        current_block = []
        current_language = ""
        block_count = 0

        for line in lines:
            if line.startswith("```"):
                if in_code_block:
                    # End of code block
                    artifacts.append({
                        "type": "code",
                        "language": current_language,
                        "content": "\n".join(current_block),
                        "index": block_count,
                    })
                    block_count += 1
                    current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    current_language = line[3:].strip() or "text"
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)

        return artifacts

    async def close(self):
        """Clean up resources."""
        if self._model_client and hasattr(self._model_client, "close"):
            await self._model_client.close()