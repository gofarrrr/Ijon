"""
Simple test of v2 extraction system without heavy dependencies.
"""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mock the missing dependencies for testing
class MockPDFProcessor:
    async def process_pdf(self, path):
        return [type('Chunk', (), {'content': 'Machine learning is revolutionizing healthcare through predictive analytics and early disease detection.'})]

class MockQualityScorer:
    def score_extraction(self, extraction, content=None):
        return {
            'overall_score': 0.75,
            'dimension_scores': {
                'consistency': 0.8,
                'grounding': 0.7,
                'coherence': 0.75,
                'completeness': 0.75
            },
            'weaknesses': [],
            'suggestions': []
        }

# Patch the imports
import sys
sys.modules['extraction.pdf_processor'] = type(sys)('extraction.pdf_processor')
sys.modules['extraction.pdf_processor'].PDFProcessor = MockPDFProcessor
sys.modules['extraction.quality.scorer'] = type(sys)('extraction.quality.scorer')
sys.modules['extraction.quality.scorer'].QualityScorer = MockQualityScorer

# Now import our v2 modules
from extraction.v2.extractors import BaselineExtractor, select_model_for_document
from extraction.v2.enhancers import CitationEnhancer, RelationshipEnhancer
from extraction.v2.error_handling import ErrorCompactor
from extraction.v2.state import ExtractionState, StateStore
from openai import AsyncOpenAI


def setup_logging():
    """Simple logging setup."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)


logger = setup_logging()


async def test_stateless_extraction():
    """Test basic stateless extraction (Factors 1, 2, 3, 4, 12)."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Stateless Extraction")
    logger.info("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        return
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Test content
    content = """
    Artificial Intelligence is transforming healthcare through several key applications:
    1. Predictive analytics for early disease detection
    2. Personalized treatment recommendations
    3. Drug discovery acceleration
    """
    
    # Extract with stateless function
    logger.info("Extracting knowledge...")
    extraction = await BaselineExtractor.extract(content, client)
    
    logger.info(f"✅ Extracted {len(extraction.topics)} topics and {len(extraction.facts)} facts")
    logger.info(f"✅ Confidence: {extraction.overall_confidence:.2%}")
    
    # Show some facts
    for i, fact in enumerate(extraction.facts[:3], 1):
        logger.info(f"  Fact {i}: {fact.claim}")


async def test_enhancers():
    """Test micro-agent enhancers (Factor 10)."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Micro-Agent Enhancers")
    logger.info("="*60)
    
    from extraction.models import ExtractedKnowledge, Topic, Fact
    
    # Create sample extraction
    extraction = ExtractedKnowledge(
        topics=[
            Topic(name="AI", description="Artificial Intelligence", confidence=0.9),
            Topic(name="Healthcare", description="Healthcare applications", confidence=0.85)
        ],
        facts=[
            Fact(claim="AI improves disease detection", confidence=0.8),
            Fact(claim="Machine learning accelerates drug discovery", confidence=0.75)
        ],
        overall_confidence=0.8
    )
    
    # Test citation enhancer
    logger.info("Testing CitationEnhancer...")
    source_text = "AI improves disease detection according to recent studies."
    enhanced = CitationEnhancer.enhance(extraction, source_text)
    logger.info(f"✅ Citation enhancement complete")
    
    # Test relationship enhancer
    logger.info("Testing RelationshipEnhancer...")
    enhanced = RelationshipEnhancer.enhance(enhanced)
    logger.info(f"✅ Found {len(enhanced.relationships)} relationships")


async def test_error_handling():
    """Test error compaction (Factor 9)."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Error Compaction")
    logger.info("="*60)
    
    # Test various error types
    errors = [
        Exception("Rate limit exceeded: You have made too many requests to the OpenAI API. Please wait 60 seconds before trying again."),
        Exception("Connection timeout: Failed to connect to api.openai.com after 30000ms"),
        Exception("JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
        Exception("This is a very long error message that contains lots of unnecessary detail about the internal state of the system and various stack traces that don't really help with debugging but make the error message extremely long and hard to read")
    ]
    
    for error in errors:
        compacted = ErrorCompactor.compact_error(error, max_length=80)
        logger.info(f"Original ({len(str(error))} chars): {str(error)[:60]}...")
        logger.info(f"Compacted ({len(compacted)} chars): {compacted}")
        logger.info("")


async def test_state_management():
    """Test state management (Factors 5 & 6)."""
    logger.info("\n" + "="*60)
    logger.info("TEST: State Management")
    logger.info("="*60)
    
    # Create state store
    state_store = StateStore()
    
    # Create new state
    state = ExtractionState.create_new(
        pdf_path="test_document.pdf",
        state_id="test-extraction-001"
    )
    
    logger.info(f"Created state: {state.id}")
    logger.info(f"Status: {state.status}, Step: {state.current_step}")
    
    # Save state
    await state_store.save(state)
    logger.info("✅ State saved")
    
    # Simulate progress
    state.current_step = "enhancement"
    state.metadata["progress"] = "50%"
    await state_store.save(state)
    
    # Load state
    loaded = await state_store.load("test-extraction-001")
    logger.info(f"✅ State loaded: Step = {loaded.current_step}, Progress = {loaded.metadata.get('progress')}")
    
    # Simulate pause
    state.status = "paused"
    await state_store.save(state)
    logger.info("✅ State paused")
    
    # List active
    active = await state_store.list_active()
    logger.info(f"✅ Active extractions: {len(active)}")


async def test_model_selection():
    """Test model selection logic."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Model Selection")
    logger.info("="*60)
    
    # Test different scenarios
    scenarios = [
        ("academic", 5000, True, False),
        ("technical", 20000, False, True),
        ("narrative", 3000, False, False),
        ("unknown", 100000, True, True)
    ]
    
    for doc_type, length, quality_req, budget in scenarios:
        config = select_model_for_document(
            doc_type=doc_type,
            doc_length=length,
            quality_required=quality_req,
            budget_conscious=budget
        )
        logger.info(f"Document: {doc_type}, {length} chars, quality={quality_req}, budget={budget}")
        logger.info(f"  → Model: {config['model']}")
        logger.info(f"  → Reason: {config['reason']}")
        logger.info("")


async def main():
    """Run all tests."""
    logger.info("🚀 TESTING V2 EXTRACTION SYSTEM")
    logger.info("=" * 80)
    
    # Run tests
    await test_stateless_extraction()
    await test_enhancers()
    await test_error_handling()
    await test_state_management()
    await test_model_selection()
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL TESTS COMPLETED!")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())