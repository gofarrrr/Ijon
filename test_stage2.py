"""
Test script for Stage 2: Document-Aware Extraction.

Runs A/B tests comparing baseline vs document-aware strategies.
"""

import asyncio
import os
from dotenv import load_dotenv

from extraction.testing.ab_test_framework import run_stage2_validation
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Run Stage 2 tests."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return
    
    logger.info("=" * 60)
    logger.info("Stage 2: Document-Aware Extraction Testing")
    logger.info("=" * 60)
    
    try:
        # Run validation tests
        results, passed = await run_stage2_validation(api_key)
        
        if passed:
            logger.info("\n✅ Stage 2 PASSED - Document-aware extraction shows significant improvements")
            logger.info("Ready to proceed to Stage 3: Quality Scoring & Feedback")
        else:
            logger.info("\n❌ Stage 2 FAILED - Improvements not significant enough")
            logger.info("Need to refine strategies before proceeding")
        
    except Exception as e:
        logger.error(f"Stage 2 testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())