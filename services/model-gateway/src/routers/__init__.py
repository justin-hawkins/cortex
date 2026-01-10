# File: services/model-gateway/src/routers/__init__.py
"""API routers for Model Gateway."""

from src.routers.health import router as health_router
from src.routers.models import router as models_router
from src.routers.generate import router as generate_router

__all__ = [
    "health_router",
    "models_router",
    "generate_router",
]