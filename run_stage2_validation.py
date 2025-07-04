"""
Run full Stage 2 validation with smaller test set.
"""

import asyncio
import os
from dotenv import load_dotenv

from extraction.testing.ab_test_framework import ABTestFramework
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def main():
    """Run Stage 2 validation."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return
    
    logger.info("=" * 60)
    logger.info("Stage 2: Document-Aware Extraction Validation")
    logger.info("=" * 60)
    
    # Use a subset of test PDFs to run faster
    test_pdfs = [
        "test_documents/academic_paper.pdf",
        "test_documents/technical_manual.pdf",
        "test_documents/business_book.pdf",
    ]
    
    try:
        # Create framework
        framework = ABTestFramework(api_key)
        
        # Run A/B test
        results = await framework.run_ab_test(
            test_pdfs,
            test_name="stage2_quick_validation"
        )
        
        # Check success
        agg = results["aggregate_metrics"]
        success = {
            "improvement_rate": agg.get("improvement_rate", 0) > 0.5,  # 50% improvement
            "confidence_gain": agg.get("average_confidence_improvement", 0) > 0,
            "quality_score": agg.get("average_quality_score", 0) > 0.4
        }
        
        logger.info("\nStage 2 Validation Results:")
        logger.info(f"- Improvement Rate: {agg.get('improvement_rate', 0):.1%}")
        logger.info(f"- Avg Confidence Gain: {agg.get('average_confidence_improvement', 0):.3f}")
        logger.info(f"- Avg Quality Score: {agg.get('average_quality_score', 0):.2f}")
        logger.info(f"- Strategy Distribution: {agg.get('strategy_distribution', {})}")
        logger.info(f"- PASS: {all(success.values())}")
        
        # Show individual results
        logger.info("\nIndividual Results:")
        for result in results["individual_results"]:
            if "error" not in result:
                logger.info(f"- {result['pdf_path']}: "
                           f"Strategy={result['document_aware']['strategy']}, "
                           f"QualityScore={result['quality_metrics']['overall_improvement_score']:.2f}")
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())