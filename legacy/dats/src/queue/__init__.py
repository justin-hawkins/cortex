"""
Queue module for DATS.

Provides Celery configuration and task definitions.
"""

from src.queue.celery_app import app as celery_app
from src.queue.tasks import (
    execute_task,
    execute_tiny,
    execute_small,
    execute_large,
    execute_frontier,
)

__all__ = [
    "celery_app",
    "execute_task",
    "execute_tiny",
    "execute_small",
    "execute_large",
    "execute_frontier",
]