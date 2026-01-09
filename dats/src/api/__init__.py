"""
DATS HTTP API.

FastAPI application for programmatic access to the distributed agentic task system.
"""

from src.api.app import create_app

__all__ = [
    "create_app",
]