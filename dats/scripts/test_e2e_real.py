#!/usr/bin/env python3
"""
Real End-to-End Test for DATS.

This test actually uses the DATS pipeline orchestrator:
1. Initializes telemetry (sends to Jaeger/Grafana)
2. Runs the AgentPipeline orchestrator
3. Executes through real agents (Coordinator, Decomposer, ComplexityEstimator)
4. Saves work products to disk
5. Records provenance

This is NOT a mock test - it uses real LLM calls and real infrastructure.
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Ensure we're not using mocks
os.environ["USE_MOCK_MODELS"] = "False"


async def run_e2e_test():
    """Run a real E2E test through the DATS pipeline."""
    
    print("=" * 70)
    print(" DATS Real End-to-End Test")
    print(" Using actual pipeline orchestrator with telemetry")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    # Step 1: Initialize telemetry
    print("\n[1] Initializing OpenTelemetry...")
    try:
        from src.telemetry.config import configure_telemetry, shutdown_telemetry
        
        configure_telemetry(
            service_name="dats-e2e-test",
            otlp_endpoint="http://192.168.1.201:4317",
        )
        print("    ✓ Telemetry initialized - traces will be sent to Jaeger")
        print("    View at: http://192.168.1.201:16686 (search for service: dats-e2e-test)")
    except Exception as e:
        print(f"    ✗ Telemetry initialization failed: {e}")
        print("    Continuing without telemetry...")
    
    # Step 2: Create work products directory
    work_products_dir = Path(__file__).parent.parent / "work_products" / "e2e_test"
    work_products_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[2] Work products directory: {work_products_dir}")
    
    # Step 3: Initialize the REAL pipeline
    print("\n[3] Initializing AgentPipeline...")
    try:
        from src.pipeline.orchestrator import AgentPipeline, TaskStatus
        
        pipeline = AgentPipeline(
            provenance_path=str(work_products_dir / "provenance"),
            use_celery=False,  # Run synchronously for testing
        )
        print("    ✓ Pipeline initialized (sync mode - no Celery)")
    except Exception as e:
        print(f"    ✗ Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Run the calculator test
    print("\n[4] Submitting task to pipeline...")
    print("    Request: 'Create a Python function that calculates fibonacci numbers'")
    print("    This will go through: Coordinator → ComplexityEstimator → Worker")
    print()
    
    try:
        result = await pipeline.process_request(
            user_request="Create a Python function that calculates fibonacci numbers. Include type hints, docstrings, and a simple test.",
            project_id="e2e-test-project",
            context={"test_type": "e2e", "timestamp": datetime.now().isoformat()},
        )
        
        print(f"    Task ID: {result.task_id}")
        print(f"    Project ID: {result.project_id}")
        print(f"    Status: {result.status.value}")
        print(f"    Mode: {result.mode.value}")
        print(f"    Tier: {result.tier}")
        
        if result.execution_time_ms:
            print(f"    Execution time: {result.execution_time_ms}ms")
        
        if result.error:
            print(f"    Error: {result.error}")
            success = False
        else:
            success = result.status in [TaskStatus.COMPLETED, TaskStatus.QUEUED]
        
        # Save work products
        if result.output:
            output_file = work_products_dir / f"fibonacci_{result.task_id[:8]}.py"
            output_file.write_text(result.output)
            print(f"\n    ✓ Work product saved: {output_file}")
        
        if result.artifacts:
            print(f"    Artifacts: {len(result.artifacts)}")
            for i, artifact in enumerate(result.artifacts):
                print(f"      [{i+1}] {artifact.get('type', 'unknown')}: {artifact.get('path', 'N/A')}")
        
        if result.provenance_ids:
            print(f"    Provenance IDs: {result.provenance_ids}")
        
        # Print the generated code
        if result.output:
            print("\n" + "=" * 70)
            print(" Generated Code")
            print("=" * 70)
            print(result.output[:2000] + "..." if len(result.output) > 2000 else result.output)
        
    except Exception as e:
        print(f"    ✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Step 5: Clean up
    print("\n[5] Cleaning up...")
    try:
        await pipeline.close()
        print("    ✓ Pipeline closed")
    except Exception as e:
        print(f"    ✗ Pipeline close failed: {e}")
    
    try:
        shutdown_telemetry()
        print("    ✓ Telemetry flushed and shutdown")
    except Exception as e:
        print(f"    ✗ Telemetry shutdown failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    
    if success:
        print("  ✓ E2E Test PASSED")
        print(f"\n  Work products saved to: {work_products_dir}")
        print(f"  View traces at: http://192.168.1.201:16686")
        print(f"  Search for service: dats-e2e-test")
    else:
        print("  ✗ E2E Test FAILED")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(run_e2e_test())
    sys.exit(0 if success else 1)