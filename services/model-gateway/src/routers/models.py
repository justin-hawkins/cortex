# File: services/model-gateway/src/routers/models.py
"""Models listing and info endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.services.model_registry import ModelRegistry, get_registry

router = APIRouter(prefix="/models", tags=["models"])


class ModelResponse(BaseModel):
    """Model information response."""
    
    name: str
    provider: str
    endpoint_name: str
    context_window: int
    tier: str
    model_type: str
    description: str
    status: str
    metadata: dict[str, Any]


class ModelsListResponse(BaseModel):
    """Response for listing all models."""
    
    models: list[ModelResponse]
    total: int
    aliases: dict[str, str]


@router.get("", response_model=ModelsListResponse)
async def list_models(
    registry: ModelRegistry = Depends(get_registry),
) -> ModelsListResponse:
    """
    List all available models.
    
    Returns a list of all configured models from all providers,
    along with their status and model aliases.
    """
    models = await registry.list_models()
    
    model_responses = [
        ModelResponse(
            name=m.name,
            provider=m.provider,
            endpoint_name=m.endpoint_name,
            context_window=m.context_window,
            tier=m.tier,
            model_type=m.model_type,
            description=m.description,
            status=m.status,
            metadata=m.metadata,
        )
        for m in models
    ]
    
    return ModelsListResponse(
        models=model_responses,
        total=len(model_responses),
        aliases=registry.aliases,
    )


@router.get("/{name}", response_model=ModelResponse)
async def get_model(
    name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> ModelResponse:
    """
    Get information about a specific model.
    
    Args:
        name: Model name or alias
        
    Returns:
        Model information including status and capabilities.
        
    Raises:
        404: If model is not found
    """
    model = await registry.get_model(name)
    
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {name}",
        )
    
    return ModelResponse(
        name=model.name,
        provider=model.provider,
        endpoint_name=model.endpoint_name,
        context_window=model.context_window,
        tier=model.tier,
        model_type=model.model_type,
        description=model.description,
        status=model.status,
        metadata=model.metadata,
    )