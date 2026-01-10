#!/usr/bin/env python3
"""
Quick connectivity test for all model servers.

Tests:
- Ollama GPU (192.168.1.12:11434) - gemma3:4b
- Ollama CPU (192.168.1.11:11434) - qwen3-coder:30b-a3b-q8_0-64k  
- vLLM (192.168.1.11:8000/v1) - gpt-oss:20b
"""

import asyncio
import httpx
from datetime import datetime


async def test_ollama(endpoint: str, model: str, name: str) -> bool:
    """Test Ollama endpoint with a simple prompt."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print("="*60)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # First check if server is reachable
            print("  Checking server availability...")
            tags_response = await client.get(f"{endpoint}/api/tags")
            if tags_response.status_code == 200:
                models = tags_response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models]
                print(f"  ✓ Server reachable - {len(models)} models available")
                print(f"    Models: {', '.join(model_names[:5])}{'...' if len(model_names) > 5 else ''}")
            else:
                print(f"  ✗ Server returned status {tags_response.status_code}")
                return False
            
            # Now test generation
            print(f"  Sending test prompt to {model}...")
            start = datetime.now()
            
            response = await client.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": "Say 'Hello from DATS!' and nothing else.",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50,
                    }
                }
            )
            
            duration = (datetime.now() - start).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                output = result.get("response", "").strip()[:200]
                print(f"  ✓ Response received in {duration:.2f}s")
                print(f"  Output: {output}")
                return True
            else:
                print(f"  ✗ Generation failed: {response.status_code}")
                print(f"    {response.text[:200]}")
                return False
                
    except httpx.ConnectError as e:
        print(f"  ✗ Connection failed: Cannot reach {endpoint}")
        return False
    except httpx.TimeoutException:
        print(f"  ✗ Timeout: Request took too long")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def test_vllm(endpoint: str, model: str, name: str) -> bool:
    """Test vLLM endpoint with OpenAI-compatible API."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print("="*60)
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # First check available models
            print("  Checking server availability...")
            try:
                models_response = await client.get(f"{endpoint}/models")
                if models_response.status_code == 200:
                    models = models_response.json().get("data", [])
                    model_names = [m.get("id", "unknown") for m in models]
                    print(f"  ✓ Server reachable - {len(models)} models available")
                    print(f"    Models: {', '.join(model_names)}")
                else:
                    print(f"  ! Models endpoint returned {models_response.status_code}, continuing...")
            except Exception as e:
                print(f"  ! Could not list models: {e}, continuing...")
            
            # Test chat completion
            print(f"  Sending test prompt to {model}...")
            start = datetime.now()
            
            response = await client.post(
                f"{endpoint}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": "Say 'Hello from DATS!' and nothing else."}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1,
                }
            )
            
            duration = (datetime.now() - start).total_seconds()
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices:
                    output = choices[0].get("message", {}).get("content", "").strip()[:200]
                    print(f"  ✓ Response received in {duration:.2f}s")
                    print(f"  Output: {output}")
                    return True
                else:
                    print(f"  ✗ No choices in response")
                    return False
            else:
                print(f"  ✗ Request failed: {response.status_code}")
                print(f"    {response.text[:200]}")
                return False
                
    except httpx.ConnectError as e:
        print(f"  ✗ Connection failed: Cannot reach {endpoint}")
        return False
    except httpx.TimeoutException:
        print(f"  ✗ Timeout: Request took too long")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


async def main():
    """Run all connectivity tests."""
    print("\n" + "="*60)
    print(" DATS Server Connectivity Test")
    print(" Testing all model endpoints from servers.yaml")
    print("="*60)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    
    results = {}
    
    # Test 1: Ollama GPU server (192.168.1.12)
    results["Ollama GPU (192.168.1.12)"] = await test_ollama(
        endpoint="http://192.168.1.12:11434",
        model="gemma3:4b",
        name="Ollama GPU Server (rtx4060_server)"
    )
    
    # Test 2: Ollama CPU server (192.168.1.11)
    results["Ollama CPU (192.168.1.11)"] = await test_ollama(
        endpoint="http://192.168.1.11:11434",
        model="qwen3-coder:30b-a3b-q8_0-64k",
        name="Ollama CPU Server (epyc_server)"
    )
    
    # Test 3: vLLM (192.168.1.11:8000)
    # Note: vLLM uses OpenAI-style model names (openai/gpt-oss-20b)
    results["vLLM (192.168.1.11:8000)"] = await test_vllm(
        endpoint="http://192.168.1.11:8000/v1",
        model="openai/gpt-oss-20b",
        name="vLLM Server (epyc_server GPU)"
    )
    
    # Summary
    print("\n" + "="*60)
    print(" Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False
    
    print(f"\nOverall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print(f"Completed at: {datetime.now().isoformat()}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)