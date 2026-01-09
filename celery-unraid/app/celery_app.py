"""
Celery application configuration.
Connects to RabbitMQ broker on the Unraid server.

Server configuration is centralized in dats/src/config/servers.yaml
Infrastructure defaults:
  - RabbitMQ: 192.168.1.49:5672
  - Redis: 192.168.1.44:6379
"""
import os
from celery import Celery

# RabbitMQ broker configuration - use environment variable or default
# Default matches infrastructure.rabbitmq in servers.yaml
BROKER_URL = os.environ.get(
    'CELERY_BROKER_URL',
    'amqp://guest:guest@192.168.1.49:5672//'
)
RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'rpc://')

# Create Celery app
app = Celery(
    'celery_worker',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=['app.tasks']
)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='America/Chicago',
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

if __name__ == '__main__':
    app.start()