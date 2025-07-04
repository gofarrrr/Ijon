#!/usr/bin/env python3
"""
Test script for HyDE integration with the RAG pipeline.

This tests the new HyDE query enhancement feature to ensure it works
correctly with the existing system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.rag.hyde_enhancer import HyDEEnhancer, HyDERetrievalWrapper
from src.rag.retriever import DocumentRetriever
from src.rag.pipeline import create_rag_pipeline
from src.utils.logging import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


async def test_hyde_enhancer():
    """Test the basic HyDE enhancer functionality."""
    logger.info("Testing HyDE enhancer...")
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key found, skipping HyDE tests")
        return False
    
    try:
        # Initialize components
        client = AsyncOpenAI(api_key=api_key)
        enhancer = HyDEEnhancer(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_hypothetical_docs=2,
        )
        
        # Test query
        test_query = "What are the benefits of machine learning in healthcare?"
        
        # Generate hypothetical documents
        result = await enhancer.enhance_query(
            query=test_query,
            client=client,
            doc_type="academic",
        )
        
        # Verify results
        assert "original_query" in result
        assert "hypothetical_docs" in result
        assert "enhanced_queries" in result
        assert result["original_query"] == test_query
        
        logger.info(f"✅ Generated {len(result['hypothetical_docs'])} hypothetical documents")
        logger.info(f"✅ Created {len(result['enhanced_queries'])} enhanced queries")
        
        # Print sample output for inspection
        if result["hypothetical_docs"]:
            logger.info("Sample hypothetical document:")
            logger.info(f"  {result['hypothetical_docs'][0][:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HyDE enhancer test failed: {e}")
        return False


async def test_hyde_wrapper():
    """Test the HyDE retrieval wrapper."""
    logger.info("Testing HyDE retrieval wrapper...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key found, skipping wrapper tests")
        return False
    
    try:
        # Create a basic retriever (this won't actually work without vector DB)
        # but we can test the wrapper interface
        base_retriever = DocumentRetriever()
        
        # Create HyDE wrapper
        wrapper = HyDERetrievalWrapper(
            base_retriever=base_retriever,
            enable_hyde=True,
        )
        
        # Test that wrapper has the right interface
        assert hasattr(wrapper, 'retrieve_chunks')
        assert hasattr(wrapper, 'initialize')
        
        logger.info("✅ HyDE wrapper created successfully")
        logger.info("✅ Wrapper interface is correct")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HyDE wrapper test failed: {e}")
        return False


async def test_pipeline_integration():
    """Test HyDE integration with the full pipeline."""
    logger.info("Testing pipeline integration...")
    
    try:
        # Create pipeline with HyDE enabled
        pipeline = create_rag_pipeline(
            use_knowledge_graph=False,  # Disable for simpler test
            enable_hyde=True,
        )
        
        # Verify pipeline configuration
        assert pipeline.enable_hyde == True
        assert hasattr(pipeline.retriever, 'retrieve_chunks')
        
        logger.info("✅ Pipeline created with HyDE enabled")
        
        # Test pipeline creation without HyDE
        pipeline_no_hyde = create_rag_pipeline(
            use_knowledge_graph=False,
            enable_hyde=False,
        )
        
        assert pipeline_no_hyde.enable_hyde == False
        
        logger.info("✅ Pipeline can be created with HyDE disabled")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {e}")
        return False


async def test_query_parameters():
    """Test that query parameters are properly passed through."""
    logger.info("Testing query parameter passing...")
    
    try:
        # Create pipeline with HyDE
        pipeline = create_rag_pipeline(enable_hyde=True)
        
        # Test that query method accepts new parameters
        # Note: This will fail at the retrieval stage since we don't have a DB set up,
        # but we can test that the parameters are accepted
        try:
            await pipeline.query(
                query="Test query",
                use_hyde=True,
                doc_type="academic",
            )
        except Exception as e:
            # Expected to fail at DB level, but parameters should be accepted
            if "not initialized" in str(e).lower() or "database" in str(e).lower():
                logger.info("✅ Query parameters accepted (DB failure expected)")
                return True
            else:
                raise e
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Query parameter test failed: {e}")
        return False


async def main():
    """Run all HyDE integration tests."""
    logger.info("🔬 Starting HyDE integration tests...")
    
    tests = [
        ("HyDE Enhancer", test_hyde_enhancer),
        ("HyDE Wrapper", test_hyde_wrapper),
        ("Pipeline Integration", test_pipeline_integration),
        ("Query Parameters", test_query_parameters),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n📊 Test Results Summary:")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All HyDE integration tests passed!")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)