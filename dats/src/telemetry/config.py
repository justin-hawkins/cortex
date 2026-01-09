"""
OpenTelemetry configuration for DATS.

Configures the tracer provider, meter provider, and exporters for sending
telemetry data to the OTLP collector.
"""

import logging
import os
from functools import lru_cache
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SERVICE_NAME = "dats"
DEFAULT_SERVICE_VERSION = "1.0.0"
DEFAULT_OTLP_ENDPOINT = "http://192.168.1.201:4317"

# Global state
_telemetry_configured = False
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None


def get_otlp_endpoint() -> str:
    """Get the OTLP endpoint from environment or use default."""
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT)


def get_service_name() -> str:
    """Get the service name from environment or use default."""
    return os.environ.get("OTEL_SERVICE_NAME", DEFAULT_SERVICE_NAME)


def get_service_version() -> str:
    """Get the service version from environment or use default."""
    return os.environ.get("OTEL_SERVICE_VERSION", DEFAULT_SERVICE_VERSION)


def configure_telemetry(
    service_name: Optional[str] = None,
    service_version: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    enable_logging: bool = True,
    additional_attributes: Optional[dict] = None,
) -> None:
    """
    Configure OpenTelemetry for the application.
    
    This function should be called once at application startup.
    
    Args:
        service_name: Name of the service (e.g., "dats-api", "dats-worker")
        service_version: Version of the service
        otlp_endpoint: OTLP collector endpoint (e.g., "http://192.168.1.201:4317")
        enable_logging: Whether to enable log correlation with traces
        additional_attributes: Additional resource attributes to include
    """
    global _telemetry_configured, _tracer_provider, _meter_provider
    
    if _telemetry_configured:
        logger.warning("Telemetry already configured, skipping reconfiguration")
        return
    
    # Resolve configuration
    service_name = service_name or get_service_name()
    service_version = service_version or get_service_version()
    otlp_endpoint = otlp_endpoint or get_otlp_endpoint()
    
    logger.info(
        f"Configuring OpenTelemetry: service={service_name}, "
        f"version={service_version}, endpoint={otlp_endpoint}"
    )
    
    # Build resource attributes
    resource_attributes = {
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
    }
    
    if additional_attributes:
        resource_attributes.update(additional_attributes)
    
    resource = Resource.create(resource_attributes)
    
    # Configure trace provider
    _tracer_provider = TracerProvider(resource=resource)
    
    # Configure OTLP trace exporter
    trace_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True,  # Use insecure for local development
    )
    
    span_processor = BatchSpanProcessor(
        trace_exporter,
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    _tracer_provider.add_span_processor(span_processor)
    
    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    
    # Configure meter provider for metrics
    metric_exporter = OTLPMetricExporter(
        endpoint=otlp_endpoint,
        insecure=True,
    )
    
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=60000,  # Export every minute
    )
    
    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader],
    )
    
    # Set as global meter provider
    metrics.set_meter_provider(_meter_provider)
    
    # Enable log correlation if requested
    if enable_logging:
        LoggingInstrumentor().instrument(set_logging_format=True)
    
    _telemetry_configured = True
    logger.info("OpenTelemetry configured successfully")


def shutdown_telemetry() -> None:
    """
    Shutdown telemetry providers gracefully.
    
    This should be called during application shutdown to ensure all
    pending telemetry data is exported.
    """
    global _telemetry_configured, _tracer_provider, _meter_provider
    
    if not _telemetry_configured:
        return
    
    logger.info("Shutting down OpenTelemetry...")
    
    if _tracer_provider:
        _tracer_provider.force_flush()
        _tracer_provider.shutdown()
        _tracer_provider = None
    
    if _meter_provider:
        _meter_provider.shutdown()
        _meter_provider = None
    
    _telemetry_configured = False
    logger.info("OpenTelemetry shutdown complete")


@lru_cache(maxsize=32)
def get_tracer(name: str = "dats") -> trace.Tracer:
    """
    Get a tracer instance for creating spans.
    
    Args:
        name: Name of the tracer (typically the module name)
    
    Returns:
        Tracer instance for creating spans
    """
    if not _telemetry_configured:
        # Return a no-op tracer if not configured
        return trace.get_tracer(name)
    
    return trace.get_tracer(
        name,
        schema_url="https://opentelemetry.io/schemas/1.21.0",
    )


@lru_cache(maxsize=32)
def get_meter(name: str = "dats") -> metrics.Meter:
    """
    Get a meter instance for creating metrics.
    
    Args:
        name: Name of the meter (typically the module name)
    
    Returns:
        Meter instance for creating metrics
    """
    if not _telemetry_configured:
        # Return a no-op meter if not configured
        return metrics.get_meter(name)
    
    return metrics.get_meter(
        name,
        schema_url="https://opentelemetry.io/schemas/1.21.0",
    )


def is_telemetry_configured() -> bool:
    """Check if telemetry has been configured."""
    return _telemetry_configured