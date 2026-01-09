"""
FastAPI application for DATS.

Main API entry point with middleware, exception handlers, and route registration.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.api.dependencies import cleanup_dependencies, get_rate_limit_key
from src.config.settings import get_settings
from src.telemetry.config import configure_telemetry, shutdown_telemetry

logger = logging.getLogger(__name__)

# API Version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Manages startup and shutdown events.
    """
    # Startup
    logger.info(f"DATS API v{API_VERSION} starting up")
    settings = get_settings()
    
    # Configure OpenTelemetry
    configure_telemetry(
        service_name="dats-api",
        service_version=API_VERSION,
        additional_attributes={
            "dats.component": "api",
            "dats.api.host": settings.api_host,
            "dats.api.port": settings.api_port,
        },
    )
    
    logger.info(f"API running on {settings.api_host}:{settings.api_port}")
    
    yield
    
    # Shutdown
    logger.info("DATS API shutting down")
    shutdown_telemetry()
    await cleanup_dependencies()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    app = FastAPI(
        title="DATS API",
        description="Distributed Agentic Task System - HTTP API",
        version=API_VERSION,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    
    # Configure rate limiting
    limiter = Limiter(key_func=get_rate_limit_key)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register routes
    register_routes(app)
    
    # Instrument FastAPI with OpenTelemetry
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,/,/api/docs,/api/redoc,/api/openapi.json",
    )
    
    return app


def register_exception_handlers(app: FastAPI):
    """Register custom exception handlers."""
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded errors."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please slow down.",
                    "details": {"retry_after": exc.detail},
                }
            },
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors as bad requests."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "code": "BAD_REQUEST",
                    "message": str(exc),
                }
            },
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.exception("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal error occurred.",
                    "details": {"type": type(exc).__name__},
                }
            },
        )


def register_routes(app: FastAPI):
    """Register all API routes."""
    from src.api.routes import monitoring, projects, provenance, review, tasks
    
    # Include route modules
    app.include_router(
        tasks.router,
        prefix="/api/v1",
        tags=["tasks"],
    )
    app.include_router(
        review.router,
        prefix="/api/v1",
        tags=["review"],
    )
    app.include_router(
        projects.router,
        prefix="/api/v1",
        tags=["projects"],
    )
    app.include_router(
        monitoring.router,
        prefix="/api/v1",
        tags=["monitoring"],
    )
    app.include_router(
        provenance.router,
        prefix="/api/v1",
        tags=["provenance"],
    )


# Create the application instance
app = create_app()


# Health check at root
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs."""
    return {
        "name": "DATS API",
        "version": API_VERSION,
        "docs": "/api/docs",
    }