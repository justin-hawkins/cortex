#!/usr/bin/env python3
"""
Test script to verify OpenTelemetry integration is working end-to-end.

This script:
1. Configures OpenTelemetry
2. Creates test spans simulating pipeline flow
3. Sends traces to the OTLP collector
4. Provides a link to view the trace in Jaeger

Usage:
    cd dats && source venv/bin/activate
    python scripts/test_otel_integration.py
"""

import asyncio
import time
import uuid
from datetime import datetime

# Configure telemetry before any other imports that might use it
import sys
sys.path.insert(0, ".")

from src.telemetry.config import configure_telemetry, shutdown_telemetry, get_tracer
from src.telemetry.context import get_current_trace_id, inject_context, extract_context


def simulate_pipeline_trace():
    """
    Simulate a full pipeline trace with nested spans.
    
    This mimics the actual DATS pipeline flow:
    - API Request (parent)
      - pipeline.process_request
        - pipeline.coordinate
        - pipeline.estimate_and_queue
          - celery.execute_task (simulated)
            - rag.get_context
            - worker.code-general.execute
              - llm.ollama.generate
    """
    tracer = get_tracer("dats.test")
    task_id = str(uuid.uuid4())
    
    print(f"\n📊 Creating test trace with task_id: {task_id}")
    
    # Simulate API request span
    with tracer.start_as_current_span(
        "POST /api/v1/tasks/submit",
        attributes={
            "http.method": "POST",
            "http.url": "/api/v1/tasks/submit",
            "dats.task.id": task_id,
        },
    ) as api_span:
        trace_id = get_current_trace_id()
        print(f"   Trace ID: {trace_id}")
        
        # Simulate pipeline.process_request
        with tracer.start_as_current_span(
            "pipeline.process_request",
            attributes={
                "dats.task.id": task_id,
                "dats.project.id": "test-project",
                "dats.request.length": 50,
            },
        ):
            time.sleep(0.05)  # Simulate some work
            
            # Simulate coordination
            with tracer.start_as_current_span(
                "pipeline.coordinate",
                attributes={"dats.task.id": task_id},
            ) as coord_span:
                time.sleep(0.1)  # Simulate LLM call
                coord_span.set_attribute("dats.coordination.mode", "new_project")
                coord_span.set_attribute("dats.coordination.needs_decomposition", False)
            
            # Simulate estimate_and_queue
            with tracer.start_as_current_span(
                "pipeline.estimate_and_queue",
                attributes={"dats.task.id": task_id},
            ) as queue_span:
                time.sleep(0.05)
                queue_span.set_attribute("dats.task.tier", "small")
                
                # Capture context for "Celery task"
                carrier = inject_context()
        
        # Simulate Celery task execution (in a real scenario, this would be in a worker)
        parent_context = extract_context(carrier)
        
        with tracer.start_as_current_span(
            "execute_task.small",
            context=parent_context,
            attributes={
                "dats.task.id": task_id,
                "dats.task.tier": "small",
                "dats.task.domain": "code-general",
            },
        ) as task_span:
            # Simulate RAG context retrieval
            with tracer.start_as_current_span(
                "rag.get_context",
                attributes={"dats.task.id": task_id},
            ) as rag_span:
                time.sleep(0.08)
                rag_span.set_attribute("dats.rag.context_length", 1500)
            
            # Simulate worker execution
            with tracer.start_as_current_span(
                "worker.code-general.execute",
                attributes={
                    "dats.task.id": task_id,
                    "dats.worker.domain": "code-general",
                    "dats.model.name": "gemma3:12b",
                },
            ) as worker_span:
                # Simulate LLM call
                with tracer.start_as_current_span(
                    "llm.ollama.generate",
                    attributes={
                        "dats.llm.provider": "ollama",
                        "dats.llm.model": "gemma3:12b",
                        "gen_ai.system": "ollama",
                        "gen_ai.request.model": "gemma3:12b",
                    },
                ) as llm_span:
                    time.sleep(0.3)  # Simulate LLM latency
                    llm_span.set_attribute("gen_ai.usage.input_tokens", 500)
                    llm_span.set_attribute("gen_ai.usage.output_tokens", 250)
                    llm_span.set_attribute("dats.llm.latency_ms", 300)
                
                worker_span.set_attribute("dats.worker.execution_time_ms", 350)
            
            task_span.set_attribute("dats.tokens.input", 500)
            task_span.set_attribute("dats.tokens.output", 250)
            task_span.set_attribute("dats.execution_time_ms", 500)
    
    return trace_id


def main():
    """Run the integration test."""
    print("=" * 60)
    print("  OpenTelemetry Integration Test")
    print("=" * 60)
    print(f"  Timestamp: {datetime.now().isoformat()}")
    
    # Configure telemetry
    print("\n🔧 Configuring OpenTelemetry...")
    configure_telemetry(
        service_name="dats-integration-test",
        service_version="1.0.0",
        additional_attributes={
            "dats.component": "integration-test",
            "dats.test.type": "manual",
        },
    )
    print("   ✅ Telemetry configured")
    
    # Create test traces
    print("\n🚀 Creating test traces...")
    
    trace_ids = []
    for i in range(3):
        trace_id = simulate_pipeline_trace()
        trace_ids.append(trace_id)
        print(f"   ✅ Trace {i+1}/3 created")
        time.sleep(0.1)
    
    # Shutdown telemetry (flushes all pending spans)
    print("\n📤 Flushing traces to collector...")
    shutdown_telemetry()
    print("   ✅ Traces flushed")
    
    # Wait a moment for the collector to process
    print("\n⏳ Waiting for traces to be processed...")
    time.sleep(2)
    
    # Provide links to view traces
    print("\n" + "=" * 60)
    print("  View Your Traces")
    print("=" * 60)
    print("\n  🔍 Jaeger UI: http://192.168.1.201:16686")
    print("\n  Search for traces:")
    print("    - Service: dats-integration-test")
    print("    - Operation: POST /api/v1/tasks/submit")
    print("\n  Direct trace links:")
    for i, trace_id in enumerate(trace_ids, 1):
        if trace_id:
            print(f"    {i}. http://192.168.1.201:16686/trace/{trace_id}")
    
    print("\n  📊 Grafana: http://192.168.1.201:3000")
    print("  📈 Prometheus: http://192.168.1.201:9090")
    
    print("\n" + "=" * 60)
    print("  ✅ Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()