# File: services/model-gateway/src/routers/health.py
"""Health check endpoints."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src import __version__
from src.services.model_registry import ModelRegistry, get_registry

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Basic health check response."""
    
    status: str
    version: str
    timestamp: datetime


class DetailedHealthResponse(BaseModel):
    """Detailed health check response with provider status."""
    
    status: str
    version: str
    timestamp: datetime
    dependencies: dict[str, str]


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns service status, version, and timestamp.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow(),
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    registry: ModelRegistry = Depends(get_registry),
) -> DetailedHealthResponse:
    """
    Detailed health check with provider status.
    
    Returns service status, version, timestamp, and the status
    of all configured LLM providers.
    """
    # Check all provider health
    dependencies = await registry.health_check()
    
    # Determine overall status
    all_healthy = all(status == "connected" for status in dependencies.values())
    any_healthy = any(status == "connected" for status in dependencies.values())
    
    if all_healthy:
        status = "healthy"
    elif any_healthy:
        status = "degraded"
    else:
        status = "unhealthy"
    
    return DetailedHealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.utcnow(),
        dependencies=dependencies,
    )