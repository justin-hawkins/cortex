"""
Tracing decorators for DATS components.

Provides convenient decorators for adding tracing to functions, agents,
workers, and LLM calls.
"""

import functools
import asyncio
import time
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

from src.telemetry.config import get_tracer

P = ParamSpec("P")
T = TypeVar("T")


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[dict] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing async functions.
    
    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to the span
        kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)
    
    Example:
        @trace_async("process_request")
        async def process_request(data):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        tracer = get_tracer(func.__module__)
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(
                span_name,
                kind=kind,
                attributes=attributes,
            ) as span:
                try:
                    # Add function arguments as span attributes (if simple types)
                    _add_arg_attributes(span, func, args, kwargs)
                    
                    result = await func(*args, **kwargs)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_sync(
    name: Optional[str] = None,
    attributes: Optional[dict] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing synchronous functions.
    
    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to the span
        kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)
    
    Example:
        @trace_sync("calculate_complexity")
        def calculate_complexity(task):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or func.__name__
        tracer = get_tracer(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(
                span_name,
                kind=kind,
                attributes=attributes,
            ) as span:
                try:
                    _add_arg_attributes(span, func, args, kwargs)
                    
                    result = func(*args, **kwargs)
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_agent(
    agent_type: str,
    operation: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing agent operations.
    
    Adds agent-specific attributes to spans for better filtering in Jaeger.
    
    Args:
        agent_type: Type of agent (coordinator, decomposer, complexity_estimator, etc.)
        operation: Operation name (defaults to function name)
    
    Example:
        @trace_agent("coordinator", "analyze_task")
        async def analyze_task(self, task_data):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = f"agent.{agent_type}.{operation or func.__name__}"
        tracer = get_tracer("dats.agents")
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "dats.agent.type": agent_type,
                    "dats.agent.operation": operation or func.__name__,
                },
            ) as span:
                try:
                    # Try to extract task_id from arguments
                    task_id = _extract_task_id(args, kwargs)
                    if task_id:
                        span.set_attribute("dats.task.id", task_id)
                    
                    start_time = time.perf_counter()
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    span.set_attribute("dats.duration_ms", duration_ms)
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_worker(
    domain: str,
    operation: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing worker operations.
    
    Adds worker-specific attributes to spans.
    
    Args:
        domain: Worker domain (code-general, code-vision, documentation, etc.)
        operation: Operation name (defaults to function name)
    
    Example:
        @trace_worker("code-general", "execute")
        async def execute(self, task_data):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = f"worker.{domain}.{operation or func.__name__}"
        tracer = get_tracer("dats.workers")
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.INTERNAL,
                attributes={
                    "dats.worker.domain": domain,
                    "dats.worker.operation": operation or func.__name__,
                },
            ) as span:
                try:
                    task_id = _extract_task_id(args, kwargs)
                    if task_id:
                        span.set_attribute("dats.task.id", task_id)
                    
                    start_time = time.perf_counter()
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    span.set_attribute("dats.duration_ms", duration_ms)
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def trace_llm_call(
    provider: str,
    model: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tracing LLM API calls.
    
    Adds LLM-specific attributes including model, tokens, and latency.
    
    Args:
        provider: LLM provider (ollama, anthropic, openai, vllm)
        model: Model name (can be overridden at runtime)
    
    Example:
        @trace_llm_call("anthropic", "claude-3-opus")
        async def generate(self, prompt, **kwargs):
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = f"llm.{provider}.generate"
        tracer = get_tracer("dats.models")
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Try to get model from kwargs or use default
            actual_model = kwargs.get("model", model) or "unknown"
            
            with tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CLIENT,
                attributes={
                    "dats.llm.provider": provider,
                    "dats.llm.model": actual_model,
                    "gen_ai.system": provider,
                    "gen_ai.request.model": actual_model,
                },
            ) as span:
                try:
                    start_time = time.perf_counter()
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    
                    span.set_attribute("dats.llm.latency_ms", duration_ms)
                    
                    # Try to extract token counts from result
                    if isinstance(result, dict):
                        if "tokens_input" in result:
                            span.set_attribute("gen_ai.usage.input_tokens", result["tokens_input"])
                        if "tokens_output" in result:
                            span.set_attribute("gen_ai.usage.output_tokens", result["tokens_output"])
                        if "usage" in result:
                            usage = result["usage"]
                            if isinstance(usage, dict):
                                if "input_tokens" in usage:
                                    span.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
                                if "output_tokens" in usage:
                                    span.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
                    
                    span.set_status(Status(StatusCode.OK))
                    return result
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


def _add_arg_attributes(span: trace.Span, func: Callable, args: tuple, kwargs: dict) -> None:
    """Add function arguments as span attributes (only simple types)."""
    # Get function parameter names
    import inspect
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    
    # Add positional args
    for i, (param_name, value) in enumerate(zip(param_names, args)):
        if param_name == "self":
            continue
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"arg.{param_name}", value)
        elif isinstance(value, dict) and "id" in value:
            span.set_attribute(f"arg.{param_name}.id", value["id"])
    
    # Add keyword args
    for key, value in kwargs.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(f"arg.{key}", value)


def _extract_task_id(args: tuple, kwargs: dict) -> Optional[str]:
    """Extract task_id from function arguments."""
    # Check kwargs first
    if "task_id" in kwargs:
        return str(kwargs["task_id"])
    
    if "task_data" in kwargs:
        task_data = kwargs["task_data"]
        if isinstance(task_data, dict) and "id" in task_data:
            return str(task_data["id"])
    
    # Check positional args (skip self)
    for arg in args:
        if isinstance(arg, dict):
            if "id" in arg:
                return str(arg["id"])
            if "task_id" in arg:
                return str(arg["task_id"])
    
    return None