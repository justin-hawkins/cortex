# File: services/model-gateway/src/config.py
"""Configuration management for Model Gateway service."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service configuration
    service_name: str = Field(default="model-gateway")
    service_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    
    # Provider endpoints
    ollama_cpu_endpoint: str = Field(default="http://192.168.1.11:11434")
    ollama_gpu_endpoint: str = Field(default="http://192.168.1.12:11434")
    vllm_endpoint: str = Field(default="http://192.168.1.11:8000/v1")
    
    # API keys
    anthropic_api_key: str = Field(default="")
    
    # OpenTelemetry
    otel_service_name: str = Field(default="model-gateway")
    otel_exporter_otlp_endpoint: str = Field(default="")
    
    # Timeouts
    default_timeout: float = Field(default=300.0)
    
    # Config file path
    config_path: str = Field(default="config/model-gateway.yaml")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def _substitute_env_vars(value: Any, settings: Settings) -> Any:
    """Recursively substitute environment variables in config values."""
    if isinstance(value, str):
        # Match ${VAR:-default} or ${VAR}
        pattern = r'\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}'
        
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2) or ""
            
            # Check settings first, then environment
            attr_name = var_name.lower()
            if hasattr(settings, attr_name):
                return str(getattr(settings, attr_name))
            return os.environ.get(var_name, default)
        
        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v, settings) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item, settings) for item in value]
    return value


def load_gateway_config(settings: Settings) -> dict[str, Any]:
    """Load and parse the model-gateway.yaml configuration."""
    config_path = Path(settings.config_path)
    
    if not config_path.exists():
        # Try relative to this file's directory
        config_path = Path(__file__).parent.parent / "config" / "model-gateway.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    # Substitute environment variables
    return _substitute_env_vars(raw_config, settings)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_gateway_config() -> dict[str, Any]:
    """Get the processed gateway configuration."""
    return load_gateway_config(get_settings())