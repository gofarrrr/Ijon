"""
Debug test for Stage 2 document-aware extraction.
"""

import asyncio
import os
import json
from dotenv import load_dotenv

from extraction.document_aware.extractor import DocumentAwareExtractor
from extraction.strategies.document_profiler import DocumentProfiler
from extraction.pdf_processor import PDFProcessor
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
import logging
setup_logging()
logging.getLogger().setLevel(logging.DEBUG)
logger = get_logger(__name__)


async def main():
    """Debug test for extraction."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return
    
    pdf_path = "test_documents/academic_paper.pdf"
    
    logger.info("Starting debug test...")
    
    try:
        # Test document profiling
        profiler = DocumentProfiler()
        profile = await profiler.profile_document(pdf_path)
        
        logger.info(f"\nDocument Profile:")
        logger.info(f"  Type: {profile.document_type} (confidence: {profile.type_confidence:.2f})")
        logger.info(f"  Structure Score: {profile.structure_score:.2f}")
        logger.info(f"  OCR Quality: {profile.ocr_quality:.2f}")
        logger.info(f"  Strategy: {profile.recommended_strategy}")
        logger.info(f"  Special Elements: {profile.special_elements}")
        
        # Test PDF extraction
        processor = PDFProcessor()
        chunks = await processor.process_pdf(pdf_path)
        content = "\n".join([chunk.content for chunk in chunks])
        
        logger.info(f"\nExtracted Content Preview:")
        logger.info(f"  Chunks: {len(chunks)}")
        logger.info(f"  Total Length: {len(content)} chars")
        logger.info(f"  First 500 chars: {content[:500]}...")
        
        # Test document-aware extraction
        extractor = DocumentAwareExtractor(api_key)
        result = await extractor.extract(pdf_path)
        
        logger.info(f"\nExtraction Results:")
        logger.info(f"  Topics: {len(result.topics)}")
        logger.info(f"  Facts: {len(result.facts)}")
        logger.info(f"  Relationships: {len(result.relationships)}")
        logger.info(f"  Questions: {len(result.questions)}")
        logger.info(f"  Confidence: {result.overall_confidence:.2f}")
        
        if result.topics:
            logger.info(f"\nSample Topics:")
            for topic in result.topics[:2]:
                logger.info(f"  - {topic.name}: {topic.description[:100]}...")
        
        if result.facts:
            logger.info(f"\nSample Facts:")
            for fact in result.facts[:2]:
                logger.info(f"  - {fact.claim[:100]}...")
        
    except Exception as e:
        logger.error(f"Debug test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())