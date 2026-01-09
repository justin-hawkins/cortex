"""
LLM Call Tracer for comprehensive prompt/response telemetry.

Captures full input/output data for every LLM call including:
- Prompt text and system prompt
- Response content
- Token counts
- Latency metrics
- Calling subsystem context
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind

from src.telemetry.config import get_tracer

# Context variable to track which subsystem is making the LLM call
_current_subsystem: ContextVar[str] = ContextVar("llm_subsystem", default="unknown")
_current_task_id: ContextVar[str] = ContextVar("llm_task_id", default="")


@dataclass
class LLMCallInput:
    """Input data for an LLM call."""
    
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stop_sequences: Optional[list[str]] = None
    model: str = ""
    provider: str = ""
    
    # Context info
    prompt_template: Optional[str] = None
    context_items: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for span events."""
        return {
            "prompt": self.prompt,
            "system_prompt": self.system_prompt or "",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": ",".join(self.stop_sequences) if self.stop_sequences else "",
            "model": self.model,
            "provider": self.provider,
            "prompt_template": self.prompt_template or "",
            "context_items": self.context_items,
            "prompt_length": len(self.prompt),
            "system_prompt_length": len(self.system_prompt) if self.system_prompt else 0,
        }


@dataclass
class LLMCallOutput:
    """Output data from an LLM call."""
    
    content: str
    tokens_input: int = 0
    tokens_output: int = 0
    model: str = ""
    finish_reason: str = ""
    latency_ms: float = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for span events."""
        return {
            "content": self.content,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_input + self.tokens_output,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "content_length": len(self.content),
        }


def set_llm_context(subsystem: str, task_id: str = "") -> tuple:
    """
    Set the current LLM call context.
    
    Args:
        subsystem: Name of the subsystem (e.g., "coordinator", "code_general")
        task_id: Optional task ID for correlation
        
    Returns:
        Tuple of context tokens for resetting
    """
    subsystem_token = _current_subsystem.set(subsystem)
    task_token = _current_task_id.set(task_id)
    return subsystem_token, task_token


def reset_llm_context(tokens: tuple) -> None:
    """Reset the LLM context to previous values."""
    subsystem_token, task_token = tokens
    _current_subsystem.reset(subsystem_token)
    _current_task_id.reset(task_token)


def get_current_subsystem() -> str:
    """Get the current subsystem name."""
    return _current_subsystem.get()


def get_current_task_id() -> str:
    """Get the current task ID."""
    return _current_task_id.get()


@contextmanager
def llm_subsystem_context(subsystem: str, task_id: str = ""):
    """
    Context manager for setting LLM subsystem context.
    
    Usage:
        with llm_subsystem_context("coordinator", task_id="abc123"):
            response = await client.generate(prompt)
    """
    tokens = set_llm_context(subsystem, task_id)
    try:
        yield
    finally:
        reset_llm_context(tokens)


class LLMCallTracer:
    """
    Tracer for LLM calls with full prompt/response capture.
    
    This class wraps LLM calls to capture comprehensive telemetry data
    including the full prompt input, response output, and all metadata.
    
    Usage:
        tracer = LLMCallTracer(provider="ollama", model="gemma3:12b")
        with tracer.trace_call(prompt, system_prompt, temperature, max_tokens) as call:
            response = await actual_llm_call(...)
            call.record_response(response)
    """
    
    def __init__(self, provider: str, model: str):
        """
        Initialize the tracer.
        
        Args:
            provider: LLM provider name (ollama, anthropic, openai)
            model: Model name
        """
        self.provider = provider
        self.model = model
        self._tracer = get_tracer("dats.llm")
    
    @contextmanager
    def trace_call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: Optional[list[str]] = None,
        prompt_template: Optional[str] = None,
        context_items: int = 0,
    ):
        """
        Context manager for tracing an LLM call.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences
            prompt_template: Name/version of prompt template used
            context_items: Number of RAG context items included
            
        Yields:
            LLMCallRecorder to record the response
        """
        subsystem = get_current_subsystem()
        task_id = get_current_task_id()
        
        # Create input record
        call_input = LLMCallInput(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            model=self.model,
            provider=self.provider,
            prompt_template=prompt_template,
            context_items=context_items,
        )
        
        # Create span with comprehensive attributes
        span_name = f"llm.{self.provider}.generate"
        
        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CLIENT,
            attributes={
                # Standard GenAI attributes
                "gen_ai.system": self.provider,
                "gen_ai.request.model": self.model,
                "gen_ai.request.temperature": temperature,
                "gen_ai.request.max_tokens": max_tokens,
                
                # DATS-specific attributes
                "dats.llm.provider": self.provider,
                "dats.llm.model": self.model,
                "dats.llm.subsystem": subsystem,
                "dats.llm.prompt_length": len(prompt),
                "dats.llm.system_prompt_length": len(system_prompt) if system_prompt else 0,
                "dats.llm.prompt_template": prompt_template or "",
                "dats.llm.context_items": context_items,
            },
        ) as span:
            # Add task ID if available
            if task_id:
                span.set_attribute("dats.task.id", task_id)
            
            # Record prompt input as span event
            span.add_event(
                "llm.prompt_input",
                attributes=call_input.to_dict(),
            )
            
            # Create recorder for response
            recorder = LLMCallRecorder(span, call_input)
            
            start_time = time.perf_counter()
            
            try:
                yield recorder
                
                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                recorder.set_latency(latency_ms)
                
                # Finalize recording
                recorder.finalize()
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise


class LLMCallRecorder:
    """
    Records LLM response data within a traced call.
    
    Used within the LLMCallTracer context manager.
    """
    
    def __init__(self, span: trace.Span, call_input: LLMCallInput):
        """
        Initialize the recorder.
        
        Args:
            span: Active span to record to
            call_input: Input data for this call
        """
        self._span = span
        self._call_input = call_input
        self._output: Optional[LLMCallOutput] = None
        self._latency_ms: float = 0
    
    def set_latency(self, latency_ms: float) -> None:
        """Set the call latency."""
        self._latency_ms = latency_ms
    
    def record_response(
        self,
        content: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        model: str = "",
        finish_reason: str = "",
    ) -> None:
        """
        Record the LLM response.
        
        Args:
            content: Response content text
            tokens_input: Input token count
            tokens_output: Output token count
            model: Model that generated response
            finish_reason: Why generation stopped
        """
        self._output = LLMCallOutput(
            content=content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model or self._call_input.model,
            finish_reason=finish_reason,
            latency_ms=self._latency_ms,
        )
    
    def record_from_model_response(self, response) -> None:
        """
        Record from a ModelResponse object.
        
        Args:
            response: ModelResponse instance
        """
        self.record_response(
            content=response.content,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            model=response.model,
            finish_reason=response.finish_reason,
        )
    
    def finalize(self) -> None:
        """Finalize the recording and add response event."""
        if self._output is None:
            return
        
        # Update latency
        self._output.latency_ms = self._latency_ms
        
        # Add span attributes for tokens and latency
        self._span.set_attribute("gen_ai.usage.input_tokens", self._output.tokens_input)
        self._span.set_attribute("gen_ai.usage.output_tokens", self._output.tokens_output)
        self._span.set_attribute("gen_ai.usage.total_tokens", self._output.tokens_input + self._output.tokens_output)
        self._span.set_attribute("dats.llm.latency_ms", self._latency_ms)
        self._span.set_attribute("dats.llm.response_length", len(self._output.content))
        self._span.set_attribute("dats.llm.finish_reason", self._output.finish_reason)
        
        # Record response as span event
        self._span.add_event(
            "llm.response_output",
            attributes=self._output.to_dict(),
        )
        
        # Set status to OK
        self._span.set_status(Status(StatusCode.OK))


# Convenience function for one-shot tracing
def trace_llm_call_sync(
    provider: str,
    model: str,
    prompt: str,
    response_content: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tokens_input: int = 0,
    tokens_output: int = 0,
    finish_reason: str = "",
    latency_ms: float = 0,
    prompt_template: Optional[str] = None,
    context_items: int = 0,
) -> None:
    """
    Record an LLM call synchronously (after the fact).
    
    Use this when you can't use the context manager approach.
    
    Args:
        provider: LLM provider
        model: Model name
        prompt: Input prompt
        response_content: Response text
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        max_tokens: Max tokens
        tokens_input: Input token count
        tokens_output: Output token count
        finish_reason: Finish reason
        latency_ms: Call latency
        prompt_template: Template name/version
        context_items: Number of context items
    """
    tracer = get_tracer("dats.llm")
    subsystem = get_current_subsystem()
    task_id = get_current_task_id()
    
    call_input = LLMCallInput(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        provider=provider,
        prompt_template=prompt_template,
        context_items=context_items,
    )
    
    call_output = LLMCallOutput(
        content=response_content,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        model=model,
        finish_reason=finish_reason,
        latency_ms=latency_ms,
    )
    
    with tracer.start_as_current_span(
        f"llm.{provider}.generate",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
            "gen_ai.usage.input_tokens": tokens_input,
            "gen_ai.usage.output_tokens": tokens_output,
            "gen_ai.usage.total_tokens": tokens_input + tokens_output,
            "dats.llm.provider": provider,
            "dats.llm.model": model,
            "dats.llm.subsystem": subsystem,
            "dats.llm.prompt_length": len(prompt),
            "dats.llm.response_length": len(response_content),
            "dats.llm.latency_ms": latency_ms,
            "dats.llm.finish_reason": finish_reason,
            "dats.llm.prompt_template": prompt_template or "",
            "dats.llm.context_items": context_items,
        },
    ) as span:
        if task_id:
            span.set_attribute("dats.task.id", task_id)
        
        # Record input event
        span.add_event("llm.prompt_input", attributes=call_input.to_dict())
        
        # Record output event
        span.add_event("llm.response_output", attributes=call_output.to_dict())
        
        span.set_status(Status(StatusCode.OK))