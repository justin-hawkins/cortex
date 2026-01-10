# File: services/model-gateway/src/main.py
"""FastAPI application entry point for Model Gateway service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.config import get_settings
from src.routers import health_router, models_router, generate_router
from src.services.model_registry import get_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    logger.info("Starting Model Gateway service...")
    
    settings = get_settings()
    logger.info(f"Log level: {settings.log_level}")
    
    # Initialize model registry
    registry = get_registry()
    await registry.initialize()
    
    logger.info("Model Gateway service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Model Gateway service...")
    await registry.close()
    logger.info("Model Gateway service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="DATS Model Gateway",
    description=(
        "Unified LLM interface abstracting Ollama, vLLM, and Anthropic behind a consistent API. "
        "Provides text generation, model discovery, and streaming support."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with /api/v1 prefix
app.include_router(health_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(generate_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "model-gateway",
        "version": __version__,
        "docs": "/api/v1/docs",
    }


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=True,
    )