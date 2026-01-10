"""
Context propagation utilities for distributed tracing.

Provides functions for propagating trace context across service boundaries,
particularly for Celery tasks and HTTP requests.
"""

from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.context import Context, get_current
from opentelemetry.propagate import inject, extract
from opentelemetry.trace import SpanContext, format_trace_id, format_span_id


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a hex string.
    
    Returns:
        Trace ID hex string or None if no active span
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format_trace_id(span.get_span_context().trace_id)
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID as a hex string.
    
    Returns:
        Span ID hex string or None if no active span
    """
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format_span_id(span.get_span_context().span_id)
    return None


def get_current_span_context() -> Optional[SpanContext]:
    """
    Get the current span context.
    
    Returns:
        Current SpanContext or None if no active span
    """
    span = trace.get_current_span()
    if span:
        ctx = span.get_span_context()
        if ctx.is_valid:
            return ctx
    return None


def inject_context(carrier: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Inject the current trace context into a carrier dict.
    
    Use this to propagate context to Celery tasks or HTTP requests.
    
    Args:
        carrier: Optional dict to inject into (creates new if None)
    
    Returns:
        Carrier dict with trace context headers
    
    Example:
        # For Celery tasks
        task_data = {"id": "123", "description": "Do something"}
        task_data["_trace_context"] = inject_context()
        my_task.delay(task_data)
        
        # For HTTP requests
        headers = inject_context()
        requests.get(url, headers=headers)
    """
    if carrier is None:
        carrier = {}
    
    inject(carrier)
    return carrier


def extract_context(carrier: Dict[str, Any]) -> Context:
    """
    Extract trace context from a carrier dict.
    
    Use this to restore context in Celery tasks or HTTP handlers.
    
    Args:
        carrier: Dict containing trace context headers
    
    Returns:
        Context object that can be used to create child spans
    
    Example:
        # In a Celery task
        @app.task
        def my_task(task_data):
            trace_context = task_data.get("_trace_context", {})
            context = extract_context(trace_context)
            
            with tracer.start_as_current_span("my_task", context=context):
                # Task work here
                pass
    """
    return extract(carrier)


def get_trace_headers() -> Dict[str, str]:
    """
    Get trace context as HTTP headers.
    
    Convenient method for getting headers to pass to HTTP clients.
    
    Returns:
        Dict of header name to value
    """
    headers: Dict[str, str] = {}
    inject(headers)
    return headers


class TraceContextCarrier:
    """
    Helper class for carrying trace context through task boundaries.
    
    Automatically handles injection and extraction of trace context.
    
    Example:
        # Creating a task with context
        carrier = TraceContextCarrier.from_current()
        task_data = {
            "id": "123",
            "_trace": carrier.to_dict(),
        }
        
        # Restoring context in task
        carrier = TraceContextCarrier.from_dict(task_data.get("_trace", {}))
        with carrier.restore_context():
            # Work happens here within the parent trace
            pass
    """
    
    def __init__(self, carrier: Optional[Dict[str, str]] = None):
        self._carrier = carrier or {}
    
    @classmethod
    def from_current(cls) -> "TraceContextCarrier":
        """Create a carrier from the current trace context."""
        carrier: Dict[str, str] = {}
        inject(carrier)
        return cls(carrier)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContextCarrier":
        """Create a carrier from a dictionary."""
        return cls(dict(data))
    
    def to_dict(self) -> Dict[str, str]:
        """Get the carrier as a dictionary."""
        return self._carrier.copy()
    
    def extract(self) -> Context:
        """Extract the context from this carrier."""
        return extract(self._carrier)
    
    def restore_context(self):
        """
        Context manager to restore the trace context.
        
        Example:
            carrier = TraceContextCarrier.from_dict(data)
            with carrier.restore_context():
                with tracer.start_as_current_span("child_span"):
                    # This span will be a child of the original trace
                    pass
        """
        from contextlib import contextmanager
        from opentelemetry.context import attach, detach
        
        @contextmanager
        def _restore():
            token = attach(self.extract())
            try:
                yield
            finally:
                detach(token)
        
        return _restore()


def add_task_context(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add trace context to task data for Celery tasks.
    
    Args:
        task_data: Task data dictionary
    
    Returns:
        Task data with trace context added
    
    Example:
        task_data = {"id": "123", "description": "Task"}
        task_data = add_task_context(task_data)
        execute_task.delay(task_data, tier)
    """
    task_data = task_data.copy()
    task_data["_trace_context"] = inject_context()
    
    # Also add trace/span IDs as regular attributes for logging
    trace_id = get_current_trace_id()
    span_id = get_current_span_id()
    
    if trace_id:
        task_data["_trace_id"] = trace_id
    if span_id:
        task_data["_parent_span_id"] = span_id
    
    return task_data


def extract_task_context(task_data: Dict[str, Any]) -> Context:
    """
    Extract trace context from task data.
    
    Args:
        task_data: Task data dictionary with trace context
    
    Returns:
        Extracted context
    
    Example:
        @app.task
        def execute_task(task_data, tier):
            context = extract_task_context(task_data)
            with tracer.start_as_current_span("execute_task", context=context):
                # Task execution
                pass
    """
    trace_context = task_data.get("_trace_context", {})
    return extract_context(trace_context)