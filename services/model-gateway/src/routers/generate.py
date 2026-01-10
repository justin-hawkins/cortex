# File: services/model-gateway/src/routers/generate.py
"""Text generation endpoints with streaming support."""

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.providers.base import GenerateRequest, GenerateResponse
from src.services.model_registry import ModelRegistry, get_registry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["generate"])


class GenerateRequestModel(BaseModel):
    """Request model for text generation."""
    
    model: str = Field(..., description="Model name or alias")
    prompt: str = Field(..., description="The prompt to generate from")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2000, ge=1, le=100000, description="Maximum tokens to generate")
    stop_sequences: list[str] | None = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")
    metadata: dict[str, Any] | None = Field(None, description="Custom metadata")


class GenerateResponseModel(BaseModel):
    """Response model for text generation."""
    
    id: str
    model: str
    provider: str
    content: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    finish_reason: str
    created_at: datetime
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str
    detail: str | None = None


@router.post(
    "/generate",
    response_model=GenerateResponseModel,
    responses={
        200: {"description": "Successful generation"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Generation error"},
    },
)
async def generate(
    request: GenerateRequestModel,
    registry: ModelRegistry = Depends(get_registry),
) -> GenerateResponseModel | StreamingResponse:
    """
    Generate text completion from an LLM.
    
    If `stream=true`, returns a Server-Sent Events stream with chunks.
    Otherwise returns the complete response.
    
    ## Example Request
    ```json
    {
        "model": "gemma3:12b",
        "prompt": "Write a Python function to calculate fibonacci",
        "temperature": 0.7,
        "max_tokens": 2000
    }
    ```
    
    ## Streaming
    When `stream=true`, the response is an SSE stream with events:
    ```
    data: {"id": "gen-xxx", "content": "def ", "done": false}
    data: {"id": "gen-xxx", "content": "fibonacci", "done": false}
    ...
    data: {"id": "gen-xxx", "content": "", "done": true, "finish_reason": "stop"}
    ```
    """
    # Handle streaming request
    if request.stream:
        return await _generate_stream(request, registry)
    
    # Convert to internal request model
    internal_request = GenerateRequest(
        model=request.model,
        prompt=request.prompt,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop_sequences=request.stop_sequences,
        stream=False,
        metadata=request.metadata,
    )
    
    try:
        response = await registry.generate(internal_request)
        
        return GenerateResponseModel(
            id=response.id,
            model=response.model,
            provider=response.provider,
            content=response.content,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            created_at=response.created_at,
            metadata=response.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


async def _generate_stream(
    request: GenerateRequestModel,
    registry: ModelRegistry,
) -> EventSourceResponse:
    """Generate a streaming SSE response."""
    
    async def event_generator():
        """Generate SSE events from stream chunks."""
        internal_request = GenerateRequest(
            model=request.model,
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            stream=True,
            metadata=request.metadata,
        )
        
        try:
            async for chunk in registry.generate_stream(internal_request):
                data = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "done": chunk.done,
                }
                if chunk.finish_reason:
                    data["finish_reason"] = chunk.finish_reason
                if chunk.tokens_output:
                    data["tokens_output"] = chunk.tokens_output
                
                yield {
                    "event": "message",
                    "data": json.dumps(data),
                }
        except ValueError as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": "model_not_found", "detail": str(e)}),
            }
        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": "generation_failed", "detail": str(e)}),
            }
    
    return EventSourceResponse(event_generator())