#!/usr/bin/env python3
"""
Simple test for HyDE enhancer without full system dependencies.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_hyde_import():
    """Test that HyDE enhancer can be imported."""
    print("Testing HyDE import...")
    
    try:
        from src.rag.hyde_enhancer import HyDEEnhancer, HyDERetrievalWrapper
        print("✅ HyDE enhancer imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import HyDE enhancer: {e}")
        return False


async def test_hyde_creation():
    """Test that HyDE enhancer can be created."""
    print("Testing HyDE creation...")
    
    try:
        from src.rag.hyde_enhancer import HyDEEnhancer
        
        enhancer = HyDEEnhancer(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_hypothetical_docs=2,
        )
        
        print("✅ HyDE enhancer created successfully")
        print(f"   Model: {enhancer.model}")
        print(f"   Temperature: {enhancer.temperature}")
        print(f"   Max docs: {enhancer.max_hypothetical_docs}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create HyDE enhancer: {e}")
        return False


async def test_hyde_with_openai():
    """Test HyDE with actual OpenAI call (if API key available)."""
    print("Testing HyDE with OpenAI...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No OpenAI API key found, skipping live test")
        return True  # Not a failure, just skipped
    
    try:
        from openai import AsyncOpenAI
        from src.rag.hyde_enhancer import HyDEEnhancer
        
        client = AsyncOpenAI(api_key=api_key)
        enhancer = HyDEEnhancer(max_hypothetical_docs=1)  # Just 1 for speed
        
        result = await enhancer.enhance_query(
            query="What is machine learning?",
            client=client,
            doc_type="technical",
        )
        
        print("✅ HyDE generation completed")
        print(f"   Original query: {result['original_query']}")
        print(f"   Hypothetical docs: {len(result['hypothetical_docs'])}")
        print(f"   Enhanced queries: {len(result['enhanced_queries'])}")
        
        if result['hypothetical_docs']:
            print(f"   Sample doc: {result['hypothetical_docs'][0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ HyDE with OpenAI failed: {e}")
        return False


async def main():
    """Run simple HyDE tests."""
    print("🔬 Running simple HyDE tests...\n")
    
    tests = [
        test_hyde_import,
        test_hyde_creation,
        test_hyde_with_openai,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            success = await test()
            if success:
                passed += 1
            print()  # Add spacing
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)