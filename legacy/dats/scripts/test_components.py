#!/usr/bin/env python3
"""
Component-Level Tests for DATS.

Tests individual components in isolation:
1. Model clients (Ollama, vLLM, Anthropic)
2. Prompt system (loading and rendering)
3. Provenance system
4. Embedding system
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


async def test_model_clients():
    """Test 2.1: Model Client Tests"""
    print("\n" + "="*60)
    print(" Phase 2.1: Model Client Tests")
    print("="*60)
    
    results = {}
    
    # Test Ollama Client
    print("\n[2.1.1] Testing OllamaClient...")
    try:
        from src.models.ollama_client import OllamaClient
        
        client = OllamaClient(
            endpoint="http://192.168.1.12:11434",
            model="gemma3:4b"
        )
        
        response = await client.generate(
            prompt="What is 2+2? Answer with just the number.",
            max_tokens=10
        )
        
        if response and "4" in response.content:
            print(f"    Response: {response.content.strip()}")
            print(f"    Tokens: {response.tokens_input}/{response.tokens_output}")
            print("    ✓ OllamaClient PASS")
            results["ollama"] = True
        else:
            print(f"    Response: {response.content if response else 'None'}")
            print("    ✗ OllamaClient FAIL - unexpected response")
            results["ollama"] = False
    except Exception as e:
        print(f"    ✗ OllamaClient FAIL: {e}")
        results["ollama"] = False
    
    # Test OpenAI-compatible client (vLLM)
    print("\n[2.1.2] Testing OpenAIClient (vLLM)...")
    try:
        from src.models.openai_client import OpenAIClient
        
        client = OpenAIClient(
            endpoint="http://192.168.1.11:8000/v1",
            model="openai/gpt-oss-20b",
            api_key="not-needed"  # vLLM doesn't require API key
        )
        
        response = await client.generate(
            prompt="What is 3+3? Answer with just the number.",
            max_tokens=10
        )
        
        if response and response.content:
            print(f"    Response: {response.content.strip()}")
            print("    ✓ OpenAIClient (vLLM) PASS")
            results["vllm"] = True
        else:
            print("    ✗ OpenAIClient (vLLM) FAIL - no response")
            results["vllm"] = False
    except Exception as e:
        print(f"    ✗ OpenAIClient (vLLM) FAIL: {e}")
        results["vllm"] = False
    
    # Test Anthropic Client
    print("\n[2.1.3] Testing AnthropicClient...")
    try:
        from src.models.anthropic_client import AnthropicClient
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your-anthropic-key-here":
            print("    ⚠ Skipping - ANTHROPIC_API_KEY not configured")
            results["anthropic"] = None
        else:
            client = AnthropicClient(
                api_key=api_key,
                model="claude-sonnet-4-20250514"
            )
            
            response = await client.generate(
                prompt="What is 4+4? Answer with just the number.",
                max_tokens=10
            )
            
            if response and "8" in response.content:
                print(f"    Response: {response.content.strip()}")
                print(f"    Tokens: {response.tokens_input}/{response.tokens_output}")
                print("    ✓ AnthropicClient PASS")
                results["anthropic"] = True
            else:
                print(f"    Response: {response.content if response else 'None'}")
                print("    ✗ AnthropicClient FAIL - unexpected response")
                results["anthropic"] = False
    except Exception as e:
        print(f"    ✗ AnthropicClient FAIL: {e}")
        results["anthropic"] = False
    
    return results


async def test_prompt_system():
    """Test 2.2: Prompt System Tests"""
    print("\n" + "="*60)
    print(" Phase 2.2: Prompt System Tests")
    print("="*60)
    
    results = {}
    
    # Test prompt loading
    print("\n[2.2.1] Testing Prompt Loader...")
    try:
        from src.prompts.loader import PromptLoader
        
        loader = PromptLoader(prompts_dir=Path(__file__).parent.parent.parent / "prompts")
        
        # Test loading agent prompt
        coordinator_prompt = loader.load("agents/coordinator")
        if coordinator_prompt and len(coordinator_prompt) > 100:
            print(f"    Loaded coordinator prompt: {len(coordinator_prompt)} chars")
            print("    ✓ Prompt Loader PASS")
            results["loader"] = True
        else:
            print("    ✗ Prompt Loader FAIL - empty or short prompt")
            results["loader"] = False
    except Exception as e:
        print(f"    ✗ Prompt Loader FAIL: {e}")
        results["loader"] = False
    
    # Test prompt rendering
    print("\n[2.2.2] Testing Prompt Renderer...")
    try:
        from src.prompts.renderer import PromptRenderer
        
        renderer = PromptRenderer()
        
        template = "Hello {{ name }}, your task is: {{ task }}"
        rendered = renderer.render(template, name="DATS", task="build a calculator")
        
        if "Hello DATS" in rendered and "build a calculator" in rendered:
            print(f"    Rendered: {rendered}")
            print("    ✓ Prompt Renderer PASS")
            results["renderer"] = True
        else:
            print(f"    Rendered: {rendered}")
            print("    ✗ Prompt Renderer FAIL - variable substitution failed")
            results["renderer"] = False
    except Exception as e:
        print(f"    ✗ Prompt Renderer FAIL: {e}")
        results["renderer"] = False
    
    return results


async def test_provenance_system():
    """Test 2.3: Provenance System Tests"""
    print("\n" + "="*60)
    print(" Phase 2.3: Provenance System Tests")
    print("="*60)
    
    results = {}
    
    # Test provenance record creation
    print("\n[2.3.1] Testing Provenance Record Creation...")
    try:
        from src.storage.provenance import ProvenanceStore, ProvenanceRecord
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProvenanceStore(base_path=tmpdir)
            
            # Create a test record
            record = ProvenanceRecord(
                task_id="test-task-001",
                project_id="test-project",
                inputs=["input1.py", "input2.py"],
                outputs=["output1.py"],
                model_used="gemma3:4b",
                tier="small",
                status="completed"
            )
            
            # Save it
            store.save(record)
            
            # Retrieve it
            retrieved = store.get("test-task-001")
            
            if retrieved and retrieved.task_id == "test-task-001":
                print(f"    Created and retrieved record: {retrieved.task_id}")
                print("    ✓ Provenance Creation PASS")
                results["creation"] = True
            else:
                print("    ✗ Provenance Creation FAIL - record not retrieved correctly")
                results["creation"] = False
    except Exception as e:
        print(f"    ✗ Provenance Creation FAIL: {e}")
        results["creation"] = False
    
    # Test provenance traversal
    print("\n[2.3.2] Testing Provenance Traversal...")
    try:
        from src.storage.provenance import ProvenanceStore, ProvenanceRecord
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ProvenanceStore(base_path=tmpdir)
            
            # Create chain: A -> B -> C
            record_a = ProvenanceRecord(
                task_id="task-A",
                project_id="test-project",
                inputs=[],
                outputs=["file_a.py"],
                model_used="gemma3:4b",
                tier="small",
                status="completed"
            )
            
            record_b = ProvenanceRecord(
                task_id="task-B",
                project_id="test-project",
                parent_id="task-A",
                inputs=["file_a.py"],
                outputs=["file_b.py"],
                model_used="gemma3:4b",
                tier="small",
                status="completed"
            )
            
            record_c = ProvenanceRecord(
                task_id="task-C",
                project_id="test-project",
                parent_id="task-B",
                inputs=["file_b.py"],
                outputs=["file_c.py"],
                model_used="gemma3:4b",
                tier="small",
                status="completed"
            )
            
            store.save(record_a)
            store.save(record_b)
            store.save(record_c)
            
            # Get descendants of A
            descendants = store.get_descendants("task-A")
            
            if len(descendants) >= 2:
                print(f"    Found {len(descendants)} descendants of task-A")
                print("    ✓ Provenance Traversal PASS")
                results["traversal"] = True
            else:
                print(f"    Found only {len(descendants)} descendants (expected 2+)")
                print("    ✗ Provenance Traversal FAIL")
                results["traversal"] = False
    except Exception as e:
        print(f"    ✗ Provenance Traversal FAIL: {e}")
        results["traversal"] = False
    
    return results


async def test_embedding_system():
    """Test 2.4: Embedding System Tests"""
    print("\n" + "="*60)
    print(" Phase 2.4: Embedding System Tests")
    print("="*60)
    
    results = {}
    
    # Test embedding generation
    print("\n[2.4.1] Testing Embedding Generation...")
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://192.168.1.12:11434/api/embeddings",
                json={
                    "model": "mxbai-embed-large:335m",
                    "prompt": "This is a test sentence for embedding."
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                if len(embedding) > 100:  # Should be a long vector
                    print(f"    Embedding dimension: {len(embedding)}")
                    print("    ✓ Embedding Generation PASS")
                    results["generation"] = True
                else:
                    print(f"    Embedding too short: {len(embedding)}")
                    print("    ✗ Embedding Generation FAIL")
                    results["generation"] = False
            else:
                print(f"    HTTP Error: {response.status_code}")
                print("    ✗ Embedding Generation FAIL")
                results["generation"] = False
    except Exception as e:
        print(f"    ✗ Embedding Generation FAIL: {e}")
        results["generation"] = False
    
    # Test RAG client
    print("\n[2.4.2] Testing RAG Client...")
    try:
        from src.rag.client import RAGClient
        
        client = RAGClient(
            embedding_endpoint="http://192.168.1.12:11434",
            embedding_model="mxbai-embed-large:335m"
        )
        
        # Test that client initializes
        print(f"    RAG Client initialized with model: {client.embedding_model}")
        print("    ✓ RAG Client PASS")
        results["rag_client"] = True
    except Exception as e:
        print(f"    ✗ RAG Client FAIL: {e}")
        results["rag_client"] = False
    
    return results


async def main():
    """Run all component tests."""
    print("\n" + "="*60)
    print(" DATS Component-Level Tests")
    print(" Testing individual components in isolation")
    print("="*60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    all_results = {}
    
    # Run all test categories
    all_results["model_clients"] = await test_model_clients()
    all_results["prompt_system"] = await test_prompt_system()
    all_results["provenance"] = await test_provenance_system()
    all_results["embedding"] = await test_embedding_system()
    
    # Summary
    print("\n" + "="*60)
    print(" Component Test Summary")
    print("="*60)
    
    total_pass = 0
    total_fail = 0
    total_skip = 0
    
    for category, results in all_results.items():
        print(f"\n  {category}:")
        for test_name, passed in results.items():
            if passed is None:
                status = "⚠ SKIP"
                total_skip += 1
            elif passed:
                status = "✓ PASS"
                total_pass += 1
            else:
                status = "✗ FAIL"
                total_fail += 1
            print(f"    {status} - {test_name}")
    
    print(f"\n  Total: {total_pass} passed, {total_fail} failed, {total_skip} skipped")
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    return total_fail == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)