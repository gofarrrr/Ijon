"""
Test the full pipeline with validation UI integration.
"""

import asyncio
import os
from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def test_pipeline_with_validation():
    """Test pipeline triggering validation UI for low quality extraction."""
    
    # Configure pipeline with human validation enabled
    config = ExtractionConfig(
        quality_threshold=0.7,  # Set high threshold to trigger validation
        enable_human_validation=True,
        enhancers_enabled=["citation", "relationship"],
        max_enhancer_loops=2,
        api_key=os.getenv("OPENAI_API_KEY", "test-key")
    )
    
    # Create pipeline
    pipeline = ExtractionPipeline(config)
    
    # Test with a PDF that will likely get low quality score
    # Using a simple test PDF with minimal content
    test_pdf = "test_data/manual_test_minimal.pdf"
    
    if not os.path.exists(test_pdf):
        # Create a simple test case
        logger.warning(f"Test PDF not found at {test_pdf}")
        logger.info("Creating mock test with low quality content...")
        
        # Mock a low-quality extraction scenario
        from extraction.models import ExtractedKnowledge, Topic, Fact
        
        # Create intentionally poor extraction (missing evidence, low confidence)
        mock_extraction = ExtractedKnowledge(
            topics=[
                Topic(name="Unknown Topic", description="Unclear description", confidence=0.5)
            ],
            facts=[
                Fact(claim="Some claim without evidence", confidence=0.4),
                Fact(claim="Another unsupported claim", confidence=0.3)
            ],
            overall_confidence=0.4
        )
        
        # Manually create quality report
        mock_quality = {
            "overall_score": 0.35,  # Very low score
            "dimensions": {
                "consistency": 0.4,
                "grounding": 0.2,  # Very poor grounding
                "coherence": 0.5,
                "completeness": 0.3
            },
            "weaknesses": [
                {"dimension": "grounding", "severity": "high", "details": "No evidence provided"},
                {"dimension": "completeness", "severity": "high", "details": "Missing key information"},
                {"dimension": "consistency", "severity": "medium", "details": "Facts don't align well"}
            ]
        }
        
        # Simulate pipeline result
        result = {
            "extraction": mock_extraction,
            "quality_report": mock_quality,
            "metadata": {
                "pdf_path": "mock_test.pdf",
                "document_profile": {"type": "unknown", "length": 1000},
                "model_used": "gpt-3.5-turbo",
                "extractor_used": "BaselineExtractor",
                "processing_time": 2.5,
                "enhancements_applied": ["citation", "relationship"],
                "human_validated": False
            },
            "human_feedback": None
        }
        
        # Manually trigger validation
        logger.info("Triggering validation for mock extraction...")
        human_feedback = await pipeline._request_human_validation(
            mock_extraction,
            mock_quality,
            "mock_test.pdf"
        )
        
        result["human_feedback"] = human_feedback
        
    else:
        # Run actual pipeline
        logger.info(f"Processing {test_pdf}...")
        result = await pipeline.extract(test_pdf)
    
    # Display results
    print("\n" + "="*60)
    print("PIPELINE RESULTS")
    print("="*60)
    
    if result.get("extraction"):
        extraction = result["extraction"]
        print(f"\n📊 Extraction Summary:")
        print(f"   Topics: {len(extraction.topics)}")
        print(f"   Facts: {len(extraction.facts)}")
        print(f"   Questions: {len(extraction.questions)}")
        print(f"   Relationships: {len(extraction.relationships)}")
        print(f"   Overall Confidence: {extraction.overall_confidence:.2f}")
    
    if result.get("quality_report"):
        quality = result["quality_report"]
        print(f"\n📈 Quality Report:")
        print(f"   Overall Score: {quality['overall_score']:.2f}")
        print(f"   Dimensions:")
        for dim, score in quality.get("dimensions", {}).items():
            print(f"     - {dim}: {score:.2f}")
    
    if result.get("human_feedback"):
        feedback = result["human_feedback"]
        print(f"\n👤 Human Validation:")
        print(f"   Status: {feedback.get('status')}")
        print(f"   Validation ID: {feedback.get('validation_id')}")
        print(f"   URL: {feedback.get('validation_url')}")
        print(f"   Message: {feedback.get('message')}")
    
    print("\n" + "="*60)
    
    if result.get("human_feedback") and result["human_feedback"].get("status") == "pending_validation":
        print("\n⚠️  VALIDATION REQUIRED")
        print("   The extraction quality is below threshold and requires human review.")
        print(f"   Visit: {result['human_feedback']['validation_url']}")
        print("\n   To start the validation UI server, run:")
        print("   python extraction/v2/validation_ui.py")
    
    return result

if __name__ == "__main__":
    print("🚀 Testing Pipeline with Validation UI Integration")
    print("="*60)
    
    result = asyncio.run(test_pipeline_with_validation())
    
    print("\n✅ Test complete!")
    print("\nNext steps:")
    print("1. Start the validation UI: python extraction/v2/validation_ui.py")
    print("2. Open the dashboard: http://localhost:8001/validator/dashboard")
    print("3. Review and validate extractions with low quality scores")