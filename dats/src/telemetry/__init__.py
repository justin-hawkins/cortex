"""
OpenTelemetry telemetry module for DATS.

Provides tracing, metrics, and logging instrumentation for the distributed
agentic task system.
"""

from src.telemetry.config import (
    configure_telemetry,
    get_tracer,
    get_meter,
    shutdown_telemetry,
)
from src.telemetry.decorators import (
    trace_async,
    trace_sync,
    trace_agent,
    trace_worker,
    trace_llm_call,
)
from src.telemetry.context import (
    get_current_trace_id,
    get_current_span_id,
    inject_context,
    extract_context,
)

__all__ = [
    # Configuration
    "configure_telemetry",
    "get_tracer",
    "get_meter",
    "shutdown_telemetry",
    # Decorators
    "trace_async",
    "trace_sync",
    "trace_agent",
    "trace_worker",
    "trace_llm_call",
    # Context propagation
    "get_current_trace_id",
    "get_current_span_id",
    "inject_context",
    "extract_context",
]