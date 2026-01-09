#!/usr/bin/env python3
"""Test Anthropic API connectivity."""

import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print('='*60)
print(' Testing Anthropic API Connection')
print('='*60)

api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key or api_key == 'your-anthropic-key-here':
    print('  ✗ ANTHROPIC_API_KEY not set or is placeholder')
    sys.exit(1)

print(f'  API Key: {api_key[:20]}...{api_key[-4:]}')

try:
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    print('  Sending test prompt to claude-sonnet-4-20250514...')
    
    message = client.messages.create(
        model='claude-sonnet-4-20250514',
        max_tokens=50,
        messages=[
            {'role': 'user', 'content': 'Say "Hello from DATS!" and nothing else.'}
        ]
    )
    
    response_text = message.content[0].text
    print(f'  Response: {response_text}')
    print(f'  Input tokens: {message.usage.input_tokens}')
    print(f'  Output tokens: {message.usage.output_tokens}')
    print('  ✓ PASS - Anthropic API Connection')
    
except Exception as e:
    print(f'  ✗ Anthropic API failed: {e}')
    sys.exit(1)