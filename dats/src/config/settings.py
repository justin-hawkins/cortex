"""
Environment and configuration settings for DATS.

Uses pydantic-settings for environment variable management with validation.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # RabbitMQ Configuration (existing broker)
    rabbitmq_host: str = Field(default="192.168.1.49", description="RabbitMQ host")
    rabbitmq_port: int = Field(default=5672, description="RabbitMQ port")
    rabbitmq_user: str = Field(default="guest", description="RabbitMQ username")
    rabbitmq_password: str = Field(default="guest", description="RabbitMQ password")
    rabbitmq_vhost: str = Field(default="/", description="RabbitMQ virtual host")

    @property
    def broker_url(self) -> str:
        """Construct RabbitMQ broker URL."""
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/{self.rabbitmq_vhost}"
        )

    # Result Backend (Redis for task results)
    redis_host: str = Field(default="192.168.1.44", description="Redis host for results")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")

    @property
    def result_backend_url(self) -> str:
        """Construct Redis result backend URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Model Endpoints (from routing_config.yaml)
    ollama_endpoint_primary: str = Field(
        default="http://192.168.1.79:11434",
        description="Primary Ollama endpoint (tiny/small models)",
    )
    ollama_endpoint_secondary: str = Field(
        default="http://192.168.1.11:11434",
        description="Secondary Ollama endpoint (embedding/large models)",
    )
    vllm_endpoint: str = Field(
        default="http://192.168.1.11:8000/v1",
        description="vLLM OpenAI-compatible endpoint",
    )
    anthropic_endpoint: str = Field(
        default="https://api.anthropic.com/v1",
        description="Anthropic API endpoint",
    )

    # API Keys
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for frontier tier",
    )
    github_token: Optional[str] = Field(
        default=None,
        description="GitHub token for work product storage",
    )

    # LightRAG Configuration
    lightrag_endpoint: Optional[str] = Field(
        default=None,
        description="LightRAG endpoint for knowledge retrieval",
    )

    # RAG Configuration
    rag_embedding_endpoint: str = Field(
        default="http://192.168.1.12:11434",
        description="Ollama endpoint for embeddings",
    )
    rag_embedding_model: str = Field(
        default="mxbai-embed-large:335m",
        description="Embedding model name",
    )
    rag_batch_interval: int = Field(
        default=300,
        description="Batch embedding interval in seconds (5 minutes)",
    )
    rag_batch_size: int = Field(
        default=10,
        description="Batch size threshold to trigger immediate embedding",
    )
    rag_storage_path: str = Field(
        default="data/rag",
        description="Directory for RAG vector storage",
    )

    # Prompts Configuration
    prompts_dir: str = Field(
        default="prompts",
        description="Directory containing prompt templates",
    )

    # Routing Configuration
    routing_config_path: str = Field(
        default="prompts/schemas/routing_config.yaml",
        description="Path to routing configuration YAML",
    )

    # Worker Configuration
    worker_concurrency: int = Field(
        default=1,
        description="Number of concurrent tasks per worker",
    )
    task_time_limit: int = Field(
        default=3600,
        description="Maximum task execution time in seconds",
    )
    task_soft_time_limit: int = Field(
        default=3300,
        description="Soft time limit for graceful shutdown",
    )

    # Storage Configuration
    provenance_path: str = Field(
        default="data/provenance",
        description="Directory for provenance records",
    )
    work_product_path: str = Field(
        default="data/work_products",
        description="Directory for work product artifacts",
    )

    # Testing Configuration
    use_mock_models: bool = Field(
        default=False,
        description="Use mock model clients for testing",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()