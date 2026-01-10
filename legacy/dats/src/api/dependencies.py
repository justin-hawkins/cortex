"""
Shared dependencies for DATS API.

Provides singleton instances and authentication for dependency injection.
"""

import logging
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from src.config.settings import Settings, get_settings
from src.pipeline.orchestrator import AgentPipeline
from src.qa.human_review import HumanReviewQueue
from src.storage.provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


# ===== Singleton Instances =====


_pipeline: Optional[AgentPipeline] = None
_provenance_tracker: Optional[ProvenanceTracker] = None
_review_queue: Optional[HumanReviewQueue] = None


def get_pipeline() -> AgentPipeline:
    """
    Get or create AgentPipeline singleton.
    
    Returns:
        AgentPipeline instance
    """
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = AgentPipeline(
            provenance_path=settings.provenance_path,
            use_celery=True,
        )
    return _pipeline


def get_provenance_tracker() -> ProvenanceTracker:
    """
    Get or create ProvenanceTracker singleton.
    
    Returns:
        ProvenanceTracker instance
    """
    global _provenance_tracker
    if _provenance_tracker is None:
        settings = get_settings()
        _provenance_tracker = ProvenanceTracker(
            storage_path=settings.provenance_path
        )
    return _provenance_tracker


def get_review_queue() -> HumanReviewQueue:
    """
    Get or create HumanReviewQueue singleton.
    
    Returns:
        HumanReviewQueue instance
    """
    global _review_queue
    if _review_queue is None:
        settings = get_settings()
        _review_queue = HumanReviewQueue(
            storage_path=f"{settings.provenance_path}/reviews"
        )
    return _review_queue


# ===== Authentication =====


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> Optional[str]:
    """
    Verify API key from header.
    
    Args:
        x_api_key: API key from X-API-Key header
        settings: Application settings
        
    Returns:
        Verified API key or None if auth is disabled
        
    Raises:
        HTTPException: If authentication fails
    """
    # If no API key is configured, auth is disabled
    if not settings.api_key:
        return None
    
    # If API key is configured but not provided
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_API_KEY",
                "message": "API key required. Provide via X-API-Key header.",
            },
        )
    
    # Verify the API key
    if x_api_key != settings.api_key:
        logger.warning("Invalid API key attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_API_KEY",
                "message": "Invalid API key.",
            },
        )
    
    return x_api_key


# ===== Project Context =====


async def get_current_project(
    project_id: Optional[str] = None,
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Get the current project ID.
    
    Uses provided project_id or falls back to default.
    
    Args:
        project_id: Explicitly provided project ID
        settings: Application settings
        
    Returns:
        Project ID to use
    """
    return project_id or settings.default_project


# ===== Cleanup =====


async def cleanup_dependencies():
    """Clean up singleton instances on shutdown."""
    global _pipeline, _provenance_tracker, _review_queue
    
    if _pipeline is not None:
        await _pipeline.close()
        _pipeline = None
    
    # ProvenanceTracker and HumanReviewQueue don't need async cleanup
    _provenance_tracker = None
    _review_queue = None
    
    logger.info("API dependencies cleaned up")


# ===== Rate Limiting Helpers =====


def get_rate_limit_key(request) -> str:
    """
    Generate rate limit key from request.
    
    Uses API key if available, otherwise IP address.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Rate limit key string
    """
    # Check for API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key[:8]}"
    
    # Fall back to IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    
    return f"ip:{ip}"