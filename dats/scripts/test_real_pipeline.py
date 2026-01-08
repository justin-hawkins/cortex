#!/usr/bin/env python3
"""
Real Integration test for the DATS pipeline.

Tests the full pipeline with actual model services.
Shows detailed input/output at each stage.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if we can reach the Ollama servers first
async def check_services():
    """Check if required services are available."""
    services = {
        "Ollama Primary (192.168.1.79:11434)": "http://192.168.1.79:11434/api/tags",
        "Ollama Secondary (192.168.1.11:11434)": "http://192.168.1.11:11434/api/tags",
    }
    
    print("\n" + "=" * 60)
    print(" Service Connectivity Check")
    print("=" * 60)
    
    all_ok = True
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "unknown") for m in data.get("models", [])]
                    print(f"  ✓ {name}")
                    print(f"    Models: {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
                else:
                    print(f"  ✗ {name} - Status: {response.status_code}")
                    all_ok = False
            except Exception as e:
                print(f"  ✗ {name} - Error: {e}")
                all_ok = False
    
    return all_ok


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_json(label: str, data: dict):
    """Print formatted JSON."""
    print(f"\n{label}:")
    print("-" * 40)
    print(json.dumps(data, indent=2, default=str))
    print("-" * 40)


async def test_ollama_directly():
    """Test direct Ollama connection with fibonacci request."""
    print_section("Step 1: Direct Ollama Model Test")
    
    # Request to send
    prompt = "Create a Python function that calculates fibonacci numbers. Return only the code."
    
    print(f"\nPrompt being sent to model:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # Use secondary Ollama (192.168.1.11) which is available
    endpoint = "http://192.168.1.11:11434/api/generate"
    model = "qwen3-coder:30b-a3b-q8_0"  # available model on secondary
    
    request_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
        }
    }
    
    print(f"\nEndpoint: {endpoint}")
    print(f"Model: {model}")
    print_json("Request Payload", request_payload)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("\nSending request to Ollama...")
            start_time = datetime.now()
            
            response = await client.post(endpoint, json=request_payload)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"Response received in {duration:.2f}s")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print_json("Response Metadata", {
                    "model": result.get("model"),
                    "total_duration": result.get("total_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_count": result.get("prompt_eval_count"),
                    "eval_count": result.get("eval_count"),
                    "done": result.get("done"),
                })
                
                print("\nGenerated Response:")
                print("=" * 40)
                print(result.get("response", "No response"))
                print("=" * 40)
                
                return True, result.get("response", "")
            else:
                print(f"Error: {response.text}")
                return False, None
                
    except Exception as e:
        print(f"Error: {e}")
        return False, None


async def test_coordinator_with_real_model():
    """Test coordinator agent with real model."""
    print_section("Step 2: Coordinator Agent Test")
    
    # Import after checking services
    from src.models.ollama_client import OllamaClient
    
    # Use secondary Ollama which is available
    client = OllamaClient(
        endpoint="http://192.168.1.11:11434",
        model_name="qwen3-coder:30b-a3b-q8_0"
    )
    
    # Coordinator analysis prompt
    user_request = "Create a Python function that calculates fibonacci numbers"
    
    system_prompt = """You are a Coordinator agent in a distributed agentic task system.
Your role is to analyze incoming requests and provide structured analysis.

Respond in JSON format with these fields:
- mode: One of "new_project", "modify", "fix_bug", "refactor", "documentation", "testing"
- domain: One of "code-general", "code-vision", "code-embedded", "documentation", "ui-design"
- needs_decomposition: true if task needs to be broken into subtasks, false if atomic
- complexity: One of "tiny", "small", "medium", "large", "frontier"
- acceptance_criteria: What defines success for this task"""

    prompt = f"""Analyze this request and provide structured output:

Request: {user_request}

Provide your analysis as JSON."""

    print(f"\nTask: Analyze user request")
    print(f"User Request: '{user_request}'")
    print(f"\nSystem Prompt:")
    print("-" * 40)
    print(system_prompt)
    print("-" * 40)
    print(f"\nUser Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    try:
        print("\nCalling Coordinator (Ollama gemma3:12b)...")
        start_time = datetime.now()
        
        response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=500,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"Response received in {duration:.2f}s")
        print(f"Tokens In: {response.tokens_input}")
        print(f"Tokens Out: {response.tokens_output}")
        
        print("\nCoordinator Response:")
        print("=" * 40)
        print(response.content)
        print("=" * 40)
        
        return True, response.content
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None


async def test_complexity_estimator_with_real_model():
    """Test complexity estimator with real model."""
    print_section("Step 3: Complexity Estimator Test")
    
    from src.models.ollama_client import OllamaClient
    
    # Use secondary Ollama which is available
    client = OllamaClient(
        endpoint="http://192.168.1.11:11434",
        model_name="qwen3-coder:30b-a3b-q8_0"
    )
    
    task_description = "Create a Python function that calculates fibonacci numbers"
    
    system_prompt = """You are a Complexity Estimator agent.
Analyze tasks and estimate their complexity for routing to appropriate model tiers.

Respond in JSON format with:
- recommended_tier: One of "tiny", "small", "large", "frontier"
- estimated_tokens: Approximate output tokens needed
- confidence: 0.0 to 1.0
- reasoning: Brief explanation"""

    prompt = f"""Estimate complexity for this task:

Task: {task_description}
Domain: code-general

Consider:
- Code complexity
- Context needed
- Output length expected

Provide your estimation as JSON."""

    print(f"\nTask: Estimate complexity")
    print(f"Description: '{task_description}'")
    print(f"\nSystem Prompt:")
    print("-" * 40)
    print(system_prompt)
    print("-" * 40)
    print(f"\nUser Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    try:
        print("\nCalling Complexity Estimator (Ollama gemma3:4b)...")
        start_time = datetime.now()
        
        response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=300,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"Response received in {duration:.2f}s")
        print(f"Tokens In: {response.tokens_input}")
        print(f"Tokens Out: {response.tokens_output}")
        
        print("\nComplexity Estimator Response:")
        print("=" * 40)
        print(response.content)
        print("=" * 40)
        
        return True, response.content
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None


async def test_code_worker_with_real_model():
    """Test code generation worker with real model."""
    print_section("Step 4: Code Worker Execution")
    
    from src.models.ollama_client import OllamaClient
    
    # Use secondary Ollama which is available
    client = OllamaClient(
        endpoint="http://192.168.1.11:11434",
        model_name="qwen3-coder:30b-a3b-q8_0"
    )
    
    system_prompt = """You are an expert Python developer.
Generate clean, well-documented, production-ready code.
Include:
- Type hints
- Docstrings with examples
- Error handling
- Unit tests if appropriate"""

    prompt = """Create a Python function that calculates fibonacci numbers.

Requirements:
- Function should take an integer n and return the nth fibonacci number
- Handle edge cases (negative numbers, 0, 1)
- Use efficient iterative approach
- Include docstring with examples
- Include a simple test

Provide the complete Python code."""

    print(f"\nTask: Generate fibonacci code")
    print(f"\nSystem Prompt:")
    print("-" * 40)
    print(system_prompt)
    print("-" * 40)
    print(f"\nUser Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    try:
        print("\nCalling Code Worker (Ollama gemma3:12b)...")
        start_time = datetime.now()
        
        response = await client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"Response received in {duration:.2f}s")
        print(f"Tokens In: {response.tokens_input}")
        print(f"Tokens Out: {response.tokens_output}")
        
        print("\nCode Worker Response:")
        print("=" * 40)
        print(response.content)
        print("=" * 40)
        
        return True, response.content
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None


async def test_provenance_recording(code_output: str):
    """Test provenance recording."""
    print_section("Step 5: Provenance Recording")
    
    from src.storage.provenance import ProvenanceTracker
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ProvenanceTracker(storage_path=tmpdir)
        
        task_id = f"fibonacci-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        project_id = "test-project"
        
        print(f"\nCreating provenance record:")
        print(f"  Task ID: {task_id}")
        print(f"  Project ID: {project_id}")
        
        record = tracker.create_record(
            task_id=task_id,
            project_id=project_id,
            model_used="gemma3:12b",
            worker_id="code-general",
        )
        
        print(f"  Provenance ID: {record.id}")
        print(f"  Started at: {record.started_at}")
        
        completed = tracker.complete_record(
            record_id=record.id,
            outputs=[{
                "type": "code",
                "language": "python",
                "content": code_output[:500] if code_output else "No output",
            }],
            tokens_input=150,
            tokens_output=500,
            confidence=0.85,
        )
        
        print(f"\nCompleted record:")
        print(f"  Completed at: {completed.completed_at}")
        print(f"  Execution time: {completed.execution_time_ms}ms")
        print(f"  Tokens total: {completed.tokens_input + completed.tokens_output}")
        print(f"  Confidence: {completed.confidence}")
        
        # Load and display saved record
        saved_file = Path(tmpdir) / f"{record.id}.json"
        if saved_file.exists():
            print(f"\nSaved provenance file: {saved_file.name}")
            with open(saved_file) as f:
                saved_data = json.load(f)
            print_json("Provenance Record", saved_data)
            return True
        else:
            print("  ✗ Provenance file not saved!")
            return False


async def main():
    """Run the full integration test."""
    print("\n" + "=" * 60)
    print(" DATS - Real Integration Test")
    print(" Testing actual system connectivity and model responses")
    print("=" * 60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    # Check services first
    services_ok = await check_services()
    
    if not services_ok:
        print("\n⚠️  Some services are not available. Tests may fail.")
        print("Make sure Ollama is running on the configured hosts.")
    
    results = {}
    
    # Test 1: Direct Ollama
    success, output = await test_ollama_directly()
    results["Ollama Direct"] = success
    
    if not success:
        print("\n⚠️  Cannot proceed without Ollama connection.")
        return
    
    # Test 2: Coordinator
    success, coord_output = await test_coordinator_with_real_model()
    results["Coordinator"] = success
    
    # Test 3: Complexity Estimator
    success, complexity_output = await test_complexity_estimator_with_real_model()
    results["Complexity Estimator"] = success
    
    # Test 4: Code Worker
    success, code_output = await test_code_worker_with_real_model()
    results["Code Worker"] = success
    
    # Test 5: Provenance
    success = await test_provenance_recording(code_output)
    results["Provenance"] = success
    
    # Summary
    print_section("Test Summary")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print(f"Completed at: {datetime.now().isoformat()}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
