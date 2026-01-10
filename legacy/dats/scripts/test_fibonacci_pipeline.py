#!/usr/bin/env python3
"""
Integration test script for the DATS pipeline.

Tests the full pipeline flow with a fibonacci program request.
This script can be run with or without actual model services.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup mock routing config before importing agents
from src.config.routing import RoutingConfig, ModelTier, ModelConfig, EmbeddingConfig, AgentRouting
import src.config.routing as routing_module

def create_mock_routing_config():
    """Create a mock routing configuration for testing."""
    return RoutingConfig(
        model_tiers={
            "tiny": ModelTier(
                context_window=32000,
                safe_working_limit=22000,
                models=[
                    ModelConfig(
                        name="mock-gemma-4b",
                        endpoint="mock://localhost",
                        type="ollama",
                        priority=1,
                    )
                ],
            ),
            "small": ModelTier(
                context_window=32000,
                safe_working_limit=22000,
                models=[
                    ModelConfig(
                        name="mock-gemma-12b",
                        endpoint="mock://localhost",
                        type="ollama",
                        priority=1,
                    )
                ],
            ),
            "large": ModelTier(
                context_window=64000,
                safe_working_limit=45000,
                models=[
                    ModelConfig(
                        name="mock-qwen-coder",
                        endpoint="mock://localhost",
                        type="openai_compatible",
                        priority=1,
                    )
                ],
            ),
            "frontier": ModelTier(
                context_window=200000,
                safe_working_limit=150000,
                models=[
                    ModelConfig(
                        name="mock-claude-sonnet",
                        endpoint="mock://localhost",
                        type="anthropic",
                        priority=1,
                    )
                ],
            ),
        },
        embedding=EmbeddingConfig(
            model="mock-embed",
            endpoint="mock://localhost",
            type="ollama",
        ),
        agent_routing={
            "coordinator": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
            "decomposer": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
            "complexity_estimator": AgentRouting(
                preferred_tier="small",
                preferred_model=None,
                fallback_tier="tiny",
            ),
            "qa_reviewer": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
            "merge_coordinator": AgentRouting(
                preferred_tier="large",
                preferred_model=None,
                fallback_tier="small",
            ),
        },
    )

# Patch routing config before importing agents
routing_module._routing_config = create_mock_routing_config()
routing_module.get_routing_config = lambda config_path=None: create_mock_routing_config()

from src.pipeline.orchestrator import AgentPipeline, TaskMode, TaskStatus
from src.agents.coordinator import Coordinator
from src.agents.decomposer import Decomposer
from src.agents.complexity_estimator import ComplexityEstimator
from src.storage.provenance import ProvenanceTracker
from src.models.mock_client import MockModelClient


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_step(step: int, description: str):
    """Print a step indicator."""
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


async def test_fibonacci_pipeline():
    """
    Test the full pipeline with a fibonacci request.
    
    Flow:
    1. Coordinator analyzes request
    2. Decomposer checks if atomic
    3. Complexity Estimator routes to tier
    4. Task executes (with mock model)
    5. Provenance is recorded
    """
    print_header("DATS Pipeline Integration Test - Fibonacci")
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Test request
    user_request = "Create a Python function that calculates fibonacci numbers"
    project_id = f"fibonacci-test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    print(f"\nRequest: '{user_request}'")
    print(f"Project ID: {project_id}")
    
    # Step 1: Coordinator Analysis
    print_step(1, "Coordinator Analysis (Heuristics)")
    
    coordinator = Coordinator()
    
    # Quick heuristic analysis (no LLM needed)
    quick_analysis = coordinator._quick_analyze(user_request)
    
    print(f"  Mode: {quick_analysis['mode']}")
    print(f"  Domain: {quick_analysis['domain']}")
    print(f"  Needs Decomposition: {quick_analysis['needs_decomposition']}")
    print(f"  Complexity: {quick_analysis['complexity']}")
    
    # Step 2: Check Atomicity
    print_step(2, "Decomposer Atomicity Check")
    
    decomposer = Decomposer()
    
    task_data = {
        "id": project_id,
        "project_id": project_id,
        "description": user_request,
        "domain": quick_analysis["domain"],
        "complexity": quick_analysis["complexity"],
    }
    
    is_atomic = decomposer.is_atomic(task_data)
    atomicity_score = decomposer.estimate_atomicity_score(task_data)
    
    print(f"  Is Atomic: {is_atomic}")
    print(f"  Atomicity Score: {atomicity_score:.2f}")
    
    # Step 3: Complexity Estimation
    print_step(3, "Complexity Estimation & Routing")
    
    complexity_estimator = ComplexityEstimator()
    
    # Quick estimate (no LLM needed)
    estimated_complexity = quick_analysis["complexity"]
    
    # Map complexity to tier
    tier_map = {
        "tiny": "tiny",
        "small": "small",
        "medium": "large",
        "large": "large",
        "frontier": "frontier",
    }
    recommended_tier = tier_map.get(estimated_complexity, "small")
    
    # Determine QA profile
    qa_profile = coordinator._determine_qa_profile(
        estimated_complexity,
        quick_analysis["domain"]
    )
    
    print(f"  Estimated Complexity: {estimated_complexity}")
    print(f"  Recommended Tier: {recommended_tier}")
    print(f"  QA Profile: {qa_profile}")
    
    # Step 4: Mock Execution
    print_step(4, "Task Execution (Mock Model)")
    
    mock_client = MockModelClient()
    
    # Generate the fibonacci code
    response = await mock_client.generate(
        prompt=user_request,
        system_prompt="You are a code generation assistant.",
    )
    
    print(f"  Model: {response.model}")
    print(f"  Tokens In: {response.tokens_input}")
    print(f"  Tokens Out: {response.tokens_output}")
    print(f"  Finish Reason: {response.finish_reason}")
    print(f"\n  Generated Code:")
    print("  " + "-" * 36)
    for line in response.content.split("\n"):
        print(f"  {line}")
    print("  " + "-" * 36)
    
    # Step 5: Provenance Recording
    print_step(5, "Provenance Recording")
    
    # Create temp directory for provenance
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(storage_path=tmpdir)
        
        record = tracker.create_record(
            task_id=task_data["id"],
            project_id=project_id,
            model_used=response.model,
            worker_id=quick_analysis["domain"],
        )
        
        completed_record = tracker.complete_record(
            record_id=record.id,
            outputs=[{
                "type": "code",
                "language": "python",
                "content": response.content,
            }],
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            confidence=0.85,
        )
        
        print(f"  Provenance ID: {completed_record.id}")
        print(f"  Task ID: {completed_record.task_id}")
        print(f"  Model Used: {completed_record.model_used}")
        print(f"  Tokens Total: {completed_record.tokens_input + completed_record.tokens_output}")
        print(f"  Execution Time: {completed_record.execution_time_ms}ms")
        print(f"  Confidence: {completed_record.confidence}")
        
        # Verify provenance file was created
        provenance_file = Path(tmpdir) / f"{completed_record.id}.json"
        print(f"  Saved to: {provenance_file.name}")
        print(f"  File exists: {provenance_file.exists()}")
    
    # Step 6: Full Pipeline Test (with mock)
    print_step(6, "Full Pipeline Integration")
    
    pipeline = AgentPipeline(use_celery=False)
    
    try:
        result = await pipeline.process_request(
            user_request=user_request,
            project_id=project_id,
        )
        
        print(f"  Pipeline Result:")
        print(f"    Task ID: {result.task_id}")
        print(f"    Project ID: {result.project_id}")
        print(f"    Status: {result.status.value}")
        print(f"    Mode: {result.mode.value}")
        print(f"    Tier: {result.tier}")
        print(f"    Execution Time: {result.execution_time_ms}ms" if result.execution_time_ms else "    Execution Time: N/A")
        
        if result.error:
            print(f"    Error: {result.error}")
        
        if result.subtasks:
            print(f"    Subtasks: {len(result.subtasks)}")
            for subtask in result.subtasks:
                print(f"      - {subtask.subtask_id}: {subtask.status.value}")
        
    finally:
        await pipeline.close()
    
    # Summary
    print_header("Test Summary")
    
    success_criteria = [
        ("Coordinator analyzed request", quick_analysis["mode"] == "new_project"),
        ("Request is atomic", is_atomic is True),
        ("Routed to correct tier", recommended_tier in ["tiny", "small"]),
        ("Mock model generated code", "fibonacci" in response.content.lower() or "fib" in response.content.lower()),
        ("Provenance recorded", completed_record.id is not None),
        ("Pipeline completed", result.task_id is not None),
    ]
    
    all_passed = True
    for criteria, passed in success_criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criteria}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print(f"Completed at: {datetime.now().isoformat()}")
    
    return all_passed


async def test_complex_pipeline():
    """Test pipeline with a complex request that requires decomposition."""
    print_header("DATS Pipeline Integration Test - Complex Request")
    
    user_request = """
    Build a complete system with multiple components including:
    - User authentication service with JWT tokens
    - Database integration with several tables
    - REST API with several endpoints
    - Also include unit tests as well as documentation
    The system should integrate with external notification services.
    """
    
    coordinator = Coordinator()
    quick_analysis = coordinator._quick_analyze(user_request)
    
    print(f"Request (complex): '{user_request[:80]}...'")
    print(f"\nAnalysis:")
    print(f"  Mode: {quick_analysis['mode']}")
    print(f"  Domain: {quick_analysis['domain']}")
    print(f"  Needs Decomposition: {quick_analysis['needs_decomposition']}")
    print(f"  Complexity: {quick_analysis['complexity']}")
    
    decomposer = Decomposer()
    task_data = {
        "description": user_request,
        "complexity": quick_analysis["complexity"],
        "needs_decomposition": quick_analysis["needs_decomposition"],
    }
    
    is_atomic = decomposer.is_atomic(task_data)
    atomicity_score = decomposer.estimate_atomicity_score(task_data)
    
    print(f"\nAtomicity:")
    print(f"  Is Atomic: {is_atomic}")
    print(f"  Atomicity Score: {atomicity_score:.2f}")
    
    # This complex request should NOT be atomic
    passed = quick_analysis["needs_decomposition"] is True or atomicity_score < 0.5
    
    print(f"\nResult: {'PASS' if passed else 'FAIL'} - Complex request correctly identified")
    
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" DATS - Distributed Agentic Task System")
    print(" Pipeline Integration Tests")
    print("=" * 60)
    
    # Run fibonacci test
    result1 = asyncio.run(test_fibonacci_pipeline())
    
    # Run complex request test
    result2 = asyncio.run(test_complex_pipeline())
    
    print("\n" + "=" * 60)
    print(" Final Results")
    print("=" * 60)
    print(f"  Fibonacci Test: {'PASS' if result1 else 'FAIL'}")
    print(f"  Complex Request Test: {'PASS' if result2 else 'FAIL'}")
    print(f"  Overall: {'PASS' if result1 and result2 else 'FAIL'}")
    
    sys.exit(0 if (result1 and result2) else 1)