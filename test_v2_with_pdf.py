"""
Test v2 extraction with an actual PDF file.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our v2 modules
from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
from extraction.v2.state import StateStore
from extraction.v2.human_contact import HumanValidationService, ConsoleChannel
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_pdf_extraction():
    """Test extraction with a real PDF."""
    logger.info("\n" + "="*80)
    logger.info("TESTING V2 EXTRACTION WITH PDF")
    logger.info("="*80)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return
    
    # Configure pipeline
    config = ExtractionConfig(
        api_key=api_key,
        max_enhancer_loops=2,
        quality_threshold=0.7,
        enable_human_validation=False,  # Disable for automated test
        enhancers_enabled=["citation", "question", "relationship", "summary"]
    )
    
    # Create pipeline
    pipeline = ExtractionPipeline(config)
    
    # Test PDF path - use a simple test PDF
    pdf_path = "test_documents/sample.pdf"
    
    # Check if test PDF exists
    if not os.path.exists(pdf_path):
        logger.warning(f"Test PDF not found at {pdf_path}")
        logger.info("Creating a simple test PDF...")
        
        # Create test directory
        os.makedirs("test_documents", exist_ok=True)
        
        # Create a simple test PDF using reportlab (if available)
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(pdf_path, pagesize=letter)
            
            # Add content
            c.setFont("Helvetica", 16)
            c.drawString(100, 750, "Test Document: AI in Healthcare")
            
            c.setFont("Helvetica", 12)
            y = 700
            content = [
                "Artificial Intelligence is revolutionizing healthcare delivery.",
                "",
                "Key Applications:",
                "1. Predictive Analytics: AI algorithms analyze patient data to predict",
                "   disease onset and progression, enabling early intervention.",
                "",
                "2. Medical Imaging: Deep learning models can detect abnormalities in",
                "   X-rays, MRIs, and CT scans with high accuracy.",
                "",
                "3. Drug Discovery: AI accelerates the identification of new drug",
                "   candidates by analyzing molecular structures and interactions.",
                "",
                "Challenges:",
                "- Data privacy and security concerns",
                "- Need for interpretable AI models",
                "- Integration with existing healthcare systems",
                "",
                "The future of healthcare will be shaped by the responsible",
                "implementation of AI technologies."
            ]
            
            for line in content:
                c.drawString(100, y, line)
                y -= 20
            
            c.save()
            logger.info(f"Created test PDF at {pdf_path}")
            
        except ImportError:
            logger.error("reportlab not installed. Cannot create test PDF.")
            logger.info("Install with: pip install reportlab")
            return
    
    # Run extraction
    logger.info(f"\nExtracting from: {pdf_path}")
    start_time = datetime.now()
    
    try:
        result = await pipeline.extract(pdf_path)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        if result["extraction"]:
            extraction = result["extraction"]
            quality = result["quality_report"]
            metadata = result["metadata"]
            
            logger.info("\n" + "-"*60)
            logger.info("EXTRACTION RESULTS")
            logger.info("-"*60)
            
            # Summary
            logger.info(f"\n📊 Metrics:")
            logger.info(f"  - Processing time: {duration:.2f}s")
            logger.info(f"  - Model used: {metadata['model_used']}")
            logger.info(f"  - Quality score: {quality['overall_score']:.2%}")
            logger.info(f"  - Topics: {len(extraction.topics)}")
            logger.info(f"  - Facts: {len(extraction.facts)}")
            logger.info(f"  - Questions: {len(extraction.questions)}")
            logger.info(f"  - Relationships: {len(extraction.relationships)}")
            
            # Quality breakdown
            logger.info(f"\n📈 Quality Dimensions:")
            for dim, score in quality['dimension_scores'].items():
                logger.info(f"  - {dim}: {score:.2%}")
            
            # Sample content
            logger.info(f"\n📚 Top Topics:")
            for i, topic in enumerate(extraction.topics[:3], 1):
                logger.info(f"  {i}. {topic.name} (confidence: {topic.confidence:.2%})")
            
            logger.info(f"\n💡 Sample Facts:")
            for i, fact in enumerate(extraction.facts[:3], 1):
                logger.info(f"  {i}. {fact.claim}")
                if fact.evidence:
                    logger.info(f"     Evidence: {fact.evidence[:100]}...")
            
            logger.info(f"\n❓ Sample Questions:")
            for i, question in enumerate(extraction.questions[:3], 1):
                logger.info(f"  {i}. {question.question_text}")
            
            if extraction.summary:
                logger.info(f"\n📝 Summary:")
                logger.info(f"  {extraction.summary}")
            
            # Enhancements applied
            logger.info(f"\n🔧 Enhancements Applied: {', '.join(metadata['enhancements_applied'])}")
            
        else:
            logger.error(f"❌ Extraction failed: {result['metadata']['error']}")
            if 'error_details' in result['metadata']:
                logger.error(f"Details: {result['metadata']['error_details']}")
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


async def test_with_state_management():
    """Test extraction with pause/resume capability."""
    logger.info("\n" + "="*80)
    logger.info("TESTING STATE MANAGEMENT")
    logger.info("="*80)
    
    # This is a demonstration of how state management would work
    # In production, you'd use the stateful pipeline wrapper
    
    state_store = StateStore()
    
    # Simulate starting an extraction
    from extraction.v2.state import ExtractionState
    
    state = ExtractionState.create_new(
        pdf_path="test_documents/sample.pdf",
        state_id="demo-extraction-001"
    )
    
    logger.info(f"Created extraction: {state.id}")
    
    # Save initial state
    await state_store.save(state)
    
    # Simulate progress
    state.current_step = "extraction"
    state.metadata["progress"] = "25%"
    await state_store.save(state)
    logger.info("Progress: Extraction phase")
    
    # Simulate pause
    state.status = "paused"
    state.metadata["pause_reason"] = "User requested"
    await state_store.save(state)
    logger.info("Extraction paused")
    
    # Simulate resume
    loaded_state = await state_store.load("demo-extraction-001")
    logger.info(f"Resumed from: {loaded_state.current_step}")
    loaded_state.status = "running"
    
    # Continue processing
    loaded_state.current_step = "enhancement"
    loaded_state.metadata["progress"] = "75%"
    await state_store.save(loaded_state)
    logger.info("Progress: Enhancement phase")
    
    # Complete
    loaded_state.status = "completed"
    loaded_state.current_step = "done"
    loaded_state.metadata["progress"] = "100%"
    await state_store.save(loaded_state)
    logger.info("Extraction completed!")


async def main():
    """Run all tests."""
    logger.info("🚀 TESTING V2 EXTRACTION SYSTEM WITH PDF")
    logger.info("=" * 80)
    
    # Test with PDF
    await test_pdf_extraction()
    
    # Test state management
    await test_with_state_management()
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL TESTS COMPLETED!")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())