"""
Test the new 12-factor extraction system.

Shows the difference between old complex approach and new simple approach.
"""

import asyncio
import os
from dotenv import load_dotenv

from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_old_way():
    """The old complex way - for comparison."""
    logger.info("\n" + "="*60)
    logger.info("OLD WAY: Complex Document-Aware Extraction")
    logger.info("="*60)
    
    from extraction.document_aware.extractor import DocumentAwareExtractor
    
    api_key = os.getenv("OPENAI_API_KEY")
    extractor = DocumentAwareExtractor(api_key)
    
    # Lots of magic happening here - hard to control
    result = await extractor.extract("test_documents/academic_paper.pdf")
    
    logger.info(f"Extracted: {len(result.topics)} topics, {len(result.facts)} facts")
    logger.info("But how did it work? What model? What prompts? 🤷")


async def test_new_way():
    """The new 12-factor way - explicit and simple."""
    logger.info("\n" + "="*60)
    logger.info("NEW WAY: 12-Factor Extraction")
    logger.info("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Explicit configuration - we control everything
    config = ExtractionConfig(
        api_key=api_key,
        max_enhancer_loops=2,           # Clear limits
        quality_threshold=0.7,          # Explicit threshold
        enable_human_validation=False,  # We decide
        enhancers_enabled=["citation", "question", "summary"]  # Pick what we need
    )
    
    # Create pipeline - we own the control flow
    pipeline = ExtractionPipeline(config)
    
    # Extract with clear requirements
    result = await pipeline.extract(
        pdf_path="test_documents/academic_paper.pdf",
        requirements={
            "quality": True,    # We want quality
            "budget": False     # Cost is not primary concern
        }
    )
    
    # We get back everything - no hidden state
    logger.info("\nWhat we know:")
    logger.info(f"- Model used: {result['metadata']['model_used']}")
    logger.info(f"- Extractor: {result['metadata']['extractor_used']}")
    logger.info(f"- Processing time: {result['metadata']['processing_time']:.2f}s")
    logger.info(f"- Quality score: {result['quality_report']['overall_score']:.3f}")
    logger.info(f"- Enhancements applied: {result['metadata']['enhancements_applied']}")
    
    extraction = result['extraction']
    logger.info(f"\nExtracted: {len(extraction.topics)} topics, {len(extraction.facts)} facts")
    
    # Show quality breakdown
    logger.info("\nQuality dimensions:")
    for dim, score in result['quality_report']['dimension_scores'].items():
        logger.info(f"- {dim}: {score:.3f}")


async def show_enhancer_composition():
    """Show how enhancers can be mixed and matched."""
    logger.info("\n" + "="*60)
    logger.info("ENHANCER COMPOSITION: Pick What You Need")
    logger.info("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Example 1: Fast extraction, no enhancements
    fast_config = ExtractionConfig(
        api_key=api_key,
        max_enhancer_loops=0,  # No enhancement
        enhancers_enabled=[]   # No enhancers
    )
    
    # Example 2: Academic focus - citations and questions
    academic_config = ExtractionConfig(
        api_key=api_key,
        max_enhancer_loops=3,
        enhancers_enabled=["citation", "question"]  # Just what we need
    )
    
    # Example 3: Business focus - relationships and summary  
    business_config = ExtractionConfig(
        api_key=api_key,
        max_enhancer_loops=2,
        enhancers_enabled=["relationship", "summary"]
    )
    
    logger.info("Configurations created:")
    logger.info("1. Fast: No enhancements, quick results")
    logger.info("2. Academic: Citations + Questions") 
    logger.info("3. Business: Relationships + Summary")
    logger.info("\nPick the right tool for the job!")


async def show_explicit_control():
    """Show how we own the control flow."""
    logger.info("\n" + "="*60)
    logger.info("EXPLICIT CONTROL: See Every Step")
    logger.info("="*60)
    
    from openai import AsyncOpenAI
    from extraction.v2.extractors import BaselineExtractor, select_model_for_document
    from extraction.v2.enhancers import CitationEnhancer
    from extraction.quality.scorer import QualityScorer
    from extraction.pdf_processor import PDFProcessor
    
    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)
    
    # Step by step - we control everything
    pdf_path = "test_documents/technical_manual.pdf"
    
    # Step 1: Extract content
    logger.info("Step 1: Extract PDF content")
    processor = PDFProcessor()
    chunks = await processor.process_pdf(pdf_path)
    content = "\n".join([c.content for c in chunks])[:5000]
    logger.info(f"Extracted {len(content)} characters")
    
    # Step 2: Choose model
    logger.info("\nStep 2: Select model")
    model_config = select_model_for_document(
        doc_type="technical",
        doc_length=len(content),
        quality_required=True
    )
    logger.info(f"Selected: {model_config['model']} because {model_config['reason']}")
    
    # Step 3: Extract
    logger.info("\nStep 3: Extract knowledge")
    extraction = await BaselineExtractor.extract(content, client, model_config['model'])
    logger.info(f"Extracted: {len(extraction.facts)} facts")
    
    # Step 4: Enhance
    logger.info("\nStep 4: Enhance with citations")
    extraction = CitationEnhancer.enhance(extraction, content)
    
    # Step 5: Score
    logger.info("\nStep 5: Score quality")
    scorer = QualityScorer()
    quality = scorer.score_extraction(extraction)
    logger.info(f"Quality score: {quality['overall_score']:.3f}")
    
    logger.info("\nEvery step is explicit - no magic!")


async def main():
    """Run all examples."""
    # Show the evolution
    await test_old_way()
    await test_new_way()
    await show_enhancer_composition()
    await show_explicit_control()
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY: 12-Factor Extraction")
    logger.info("="*60)
    logger.info("✅ Stateless functions - easy to test")
    logger.info("✅ Explicit control - easy to debug")
    logger.info("✅ Composable enhancers - use what you need")
    logger.info("✅ Clear configuration - no surprises")
    logger.info("✅ Quality scoring - know when it's good enough")


if __name__ == "__main__":
    asyncio.run(main())