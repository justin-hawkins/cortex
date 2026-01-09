"""
Environment and configuration settings for DATS.

Uses pydantic-settings for environment variable management with validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
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

    # Model Endpoints (from servers.yaml)
    ollama_endpoint_primary: str = Field(
        default="http://192.168.1.12:11434",
        description="Primary Ollama endpoint - GPU server (tiny/small models)",
    )
    ollama_endpoint_secondary: str = Field(
        default="http://192.168.1.11:11434",
        description="Secondary Ollama endpoint - CPU server (large/high-precision models)",
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

    # Server Configuration
    servers_config_path: str = Field(
        default="dats/src/config/servers.yaml",
        description="Path to centralized server configuration YAML",
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

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication",
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key",
    )
    api_cors_origins: list[str] = Field(
        default=["*"],
        description="CORS allowed origins",
    )

    # Rate Limiting
    rate_limit_submit_per_minute: int = Field(
        default=10,
        description="Rate limit for submit endpoint",
    )
    rate_limit_status_per_minute: int = Field(
        default=60,
        description="Rate limit for status endpoint",
    )
    rate_limit_review_per_minute: int = Field(
        default=30,
        description="Rate limit for review endpoints",
    )

    # CLI Configuration
    cli_config_path: str = Field(
        default="~/.dats/config.yaml",
        description="Path to CLI configuration file",
    )
    default_project: str = Field(
        default="default",
        description="Default project name",
    )
    default_mode: str = Field(
        default="autonomous",
        description="Default task mode (autonomous/collaborative)",
    )
    default_output_format: str = Field(
        default="human",
        description="Default output format (human/json)",
    )

    # Project Storage
    projects_path: str = Field(
        default="data/projects",
        description="Directory for project configurations",
    )

    @property
    def api_url(self) -> str:
        """Construct API URL."""
        return f"http://{self.api_host}:{self.api_port}"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


def load_servers_config(config_path: Optional[str] = None) -> dict:
    """
    Load server configuration from YAML file.
    
    Args:
        config_path: Path to servers.yaml. If not provided,
                    uses default from settings.
    
    Returns:
        Dictionary containing server configuration.
    """
    if config_path is None:
        config_path = get_settings().servers_config_path
    
    path = Path(config_path)
    if not path.exists():
        # Try relative to current working directory
        path = Path.cwd() / config_path
        if not path.exists():
            raise FileNotFoundError(f"Server config not found: {config_path}")
    
    with open(path) as f:
        return yaml.safe_load(f)


_servers_config: Optional[dict] = None


def get_servers_config(config_path: Optional[str] = None) -> dict:
    """
    Get cached server configuration.
    
    Args:
        config_path: Optional path to config file.
    
    Returns:
        Cached server configuration dictionary.
    """
    global _servers_config
    
    if _servers_config is None:
        _servers_config = load_servers_config(config_path)
    
    return _servers_config


def get_endpoint(endpoint_name: str) -> str:
    """
    Get endpoint URL by name from servers.yaml.
    
    Args:
        endpoint_name: Name of endpoint (e.g., 'ollama_cpu_large', 'vllm_gpu')
    
    Returns:
        Endpoint URL string.
    
    Raises:
        KeyError: If endpoint not found.
    """
    config = get_servers_config()
    endpoints = config.get("endpoints", {})
    
    if endpoint_name not in endpoints:
        raise KeyError(f"Endpoint '{endpoint_name}' not found in servers.yaml")
    
    return endpoints[endpoint_name]["url"]


def get_server_host(server_name: str) -> str:
    """
    Get server host IP by server name.
    
    Args:
        server_name: Name of server (e.g., 'epyc_server', 'rtx4060_server')
    
    Returns:
        Server host IP string.
    """
    config = get_servers_config()
    servers = config.get("servers", {})
    
    if server_name not in servers:
        raise KeyError(f"Server '{server_name}' not found in servers.yaml")
    
    return servers[server_name]["host"]


def get_infrastructure(service_name: str) -> dict:
    """
    Get infrastructure service configuration.
    
    Args:
        service_name: Name of service (e.g., 'rabbitmq', 'redis')
    
    Returns:
        Service configuration dictionary.
    """
    config = get_servers_config()
    infrastructure = config.get("infrastructure", {})
    
    if service_name not in infrastructure:
        raise KeyError(f"Infrastructure service '{service_name}' not found")
    
    return infrastructure[service_name]
