"""
Celery application configuration for DATS.

Connects to RabbitMQ broker and configures queues for different model tiers.
"""

from celery import Celery

from src.config.settings import get_settings

# Get settings
settings = get_settings()

# Create Celery app
app = Celery(
    "dats",
    broker=settings.broker_url,
    backend=settings.result_backend_url,
    include=["src.queue.tasks"],
)

# Celery configuration
app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task tracking
    task_track_started=True,
    task_acks_late=True,  # Acknowledge after task completes
    # Concurrency
    worker_prefetch_multiplier=1,  # One task at a time per worker
    worker_concurrency=settings.worker_concurrency,
    # Time limits
    task_time_limit=settings.task_time_limit,
    task_soft_time_limit=settings.task_soft_time_limit,
    # Result backend settings
    result_expires=86400,  # Results expire after 24 hours
    result_extended=True,  # Store additional metadata
    # Task execution settings
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
)

# Define queues for different model tiers
app.conf.task_queues = {
    "tiny": {
        "exchange": "dats",
        "routing_key": "dats.tiny",
    },
    "small": {
        "exchange": "dats",
        "routing_key": "dats.small",
    },
    "large": {
        "exchange": "dats",
        "routing_key": "dats.large",
    },
    "frontier": {
        "exchange": "dats",
        "routing_key": "dats.frontier",
    },
    "default": {
        "exchange": "dats",
        "routing_key": "dats.default",
    },
}

# Task routing configuration
app.conf.task_routes = {
    "src.queue.tasks.execute_tiny": {"queue": "tiny"},
    "src.queue.tasks.execute_small": {"queue": "small"},
    "src.queue.tasks.execute_large": {"queue": "large"},
    "src.queue.tasks.execute_frontier": {"queue": "frontier"},
    "src.queue.tasks.execute_task": {"queue": "default"},
}

# Default queue
app.conf.task_default_queue = "default"
app.conf.task_default_exchange = "dats"
app.conf.task_default_routing_key = "dats.default"


if __name__ == "__main__":
    app.start()