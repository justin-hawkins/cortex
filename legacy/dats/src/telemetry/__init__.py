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
from src.telemetry.llm_tracer import (
    LLMCallTracer,
    LLMCallRecorder,
    LLMCallInput,
    LLMCallOutput,
    llm_subsystem_context,
    set_llm_context,
    reset_llm_context,
    get_current_subsystem,
    get_current_task_id,
    trace_llm_call_sync,
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
    # LLM Call Tracing
    "LLMCallTracer",
    "LLMCallRecorder",
    "LLMCallInput",
    "LLMCallOutput",
    "llm_subsystem_context",
    "set_llm_context",
    "reset_llm_context",
    "get_current_subsystem",
    "get_current_task_id",
    "trace_llm_call_sync",
]
