"""
Quick test for Stage 2 document-aware extraction on a single PDF.
"""

import asyncio
import os
from dotenv import load_dotenv

from extraction.document_aware.extractor import DocumentAwareExtractor
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Test single PDF extraction."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return
    
    # Test with academic paper (should have better structure)
    pdf_path = "test_documents/academic_paper.pdf"
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    logger.info("Testing document-aware extraction on single PDF...")
    
    try:
        # Create extractor
        extractor = DocumentAwareExtractor(api_key)
        
        # Run comparison
        comparison = await extractor.compare_with_baseline(pdf_path)
        
        # Print results
        logger.info("\nComparison Results:")
        logger.info(f"Document Type: {comparison['document_aware']['document_type']}")
        logger.info(f"Strategy Used: {comparison['document_aware']['strategy']}")
        logger.info(f"Baseline Facts: {comparison['baseline']['facts']}")
        logger.info(f"Document-Aware Facts: {comparison['document_aware']['facts']}")
        logger.info(f"Confidence Improvement: {comparison['improvements']['confidence_delta']:+.3f}")
        
        # Calculate simple quality score
        baseline = comparison["baseline"]
        aware = comparison["document_aware"]
        quality_score = sum([
            aware["confidence"] > baseline["confidence"],
            aware["facts"] >= baseline["facts"],
            aware["avg_fact_confidence"] > baseline["avg_fact_confidence"],
            comparison["document_aware"]["strategy"] != "unknown"
        ]) / 4.0
        
        logger.info(f"Quality Score: {quality_score:.2f}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())