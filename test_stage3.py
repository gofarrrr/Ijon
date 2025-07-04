"""
Test script for Stage 3: Quality Scoring & Feedback.

Tests quality scorer and feedback extraction improvements.
"""

import asyncio
import os
from dotenv import load_dotenv

from extraction.quality.scorer import QualityScorer
from extraction.quality.feedback_extractor import FeedbackExtractor
from extraction.document_aware.extractor import DocumentAwareExtractor
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def test_quality_scorer(api_key: str, pdf_path: str):
    """Test quality scoring on an extraction."""
    logger.info("\n" + "="*60)
    logger.info("Testing Quality Scorer")
    logger.info("="*60)
    
    # Extract knowledge
    extractor = DocumentAwareExtractor(api_key)
    extraction = await extractor.extract(pdf_path)
    
    # Get source content
    from extraction.pdf_processor import PDFProcessor
    processor = PDFProcessor()
    chunks = await processor.process_pdf(pdf_path)
    source_content = "\n".join([chunk.content for chunk in chunks])
    
    # Score quality
    scorer = QualityScorer()
    quality_scores = scorer.score_extraction(extraction, source_content)
    
    # Display results
    logger.info(f"\nQuality Scores for {pdf_path}:")
    logger.info(f"Overall Score: {quality_scores['overall_score']:.3f}")
    logger.info("\nDimension Scores:")
    for dim, score in quality_scores['dimension_scores'].items():
        logger.info(f"  - {dim.title()}: {score:.3f}")
    
    logger.info(f"\nNeeds Re-extraction: {quality_scores['needs_reextraction']}")
    
    if quality_scores['weaknesses']:
        logger.info("\nWeaknesses:")
        for weakness in quality_scores['weaknesses']:
            logger.info(f"  - {weakness['dimension']}: {weakness['score']:.2f} ({weakness['severity']})")
    
    if quality_scores['suggestions']:
        logger.info("\nSuggestions:")
        for suggestion in quality_scores['suggestions']:
            logger.info(f"  - [{suggestion['priority']}] {suggestion['suggestion']}")
    
    return quality_scores


async def test_feedback_extraction(api_key: str, pdf_path: str):
    """Test feedback extraction improvement."""
    logger.info("\n" + "="*60)
    logger.info("Testing Feedback Extraction")
    logger.info("="*60)
    
    # Run extraction with feedback
    feedback_extractor = FeedbackExtractor(api_key)
    result = await feedback_extractor.extract_with_feedback(pdf_path)
    
    # Display results
    logger.info(f"\nFeedback Extraction Results:")
    logger.info(f"Iterations: {result['iterations']}")
    logger.info(f"Overall Improvement: {result['improvement']:+.3f}")
    logger.info(f"Final Score: {result['quality_scores']['overall_score']:.3f}")
    
    logger.info("\nIteration History:")
    for iteration in result['iteration_history']:
        logger.info(f"  Iteration {iteration['iteration']}: {iteration['overall_score']:.3f}")
        scores = iteration['dimension_scores']
        logger.info(f"    - Consistency: {scores['consistency']:.2f}, "
                   f"Grounding: {scores['grounding']:.2f}, "
                   f"Coherence: {scores['coherence']:.2f}, "
                   f"Completeness: {scores['completeness']:.2f}")
    
    # Show extraction stats
    extraction = result['extraction']
    logger.info(f"\nFinal Extraction:")
    logger.info(f"  - Topics: {len(extraction.topics)}")
    logger.info(f"  - Facts: {len(extraction.facts)}")
    logger.info(f"  - Relationships: {len(extraction.relationships)}")
    logger.info(f"  - Questions: {len(extraction.questions)}")
    
    return result


async def test_feedback_comparison(api_key: str, pdf_path: str):
    """Compare baseline vs feedback extraction."""
    logger.info("\n" + "="*60)
    logger.info("Comparing Baseline vs Feedback Extraction")
    logger.info("="*60)
    
    feedback_extractor = FeedbackExtractor(api_key)
    comparison = await feedback_extractor.compare_with_baseline(pdf_path)
    
    logger.info(f"\nComparison Results:")
    logger.info(f"Baseline Score: {comparison['baseline']['overall_score']:.3f}")
    logger.info(f"Feedback Score: {comparison['feedback']['overall_score']:.3f}")
    logger.info(f"Score Improvement: {comparison['improvements']['overall_score_gain']:+.3f}")
    logger.info(f"Dimensions Improved: {comparison['improvements']['dimensions_improved']}/4")
    logger.info(f"Iterations Used: {comparison['feedback']['iterations']}")
    
    # Show dimension improvements
    logger.info("\nDimension Changes:")
    baseline_dims = comparison['baseline']['dimension_scores']
    feedback_dims = comparison['feedback']['dimension_scores']
    for dim in baseline_dims:
        change = feedback_dims[dim] - baseline_dims[dim]
        logger.info(f"  - {dim.title()}: {baseline_dims[dim]:.3f} → "
                   f"{feedback_dims[dim]:.3f} ({change:+.3f})")
    
    return comparison


async def main():
    """Run Stage 3 tests."""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        return
    
    # Test PDFs
    test_pdfs = [
        ("Academic Paper", "test_documents/academic_paper.pdf"),
        ("Technical Manual", "test_documents/technical_manual.pdf"),
        ("Business Book", "test_documents/business_book.pdf")
    ]
    
    logger.info("=" * 60)
    logger.info("Stage 3: Quality Scoring & Feedback Testing")
    logger.info("=" * 60)
    
    results = []
    
    for name, pdf_path in test_pdfs[:1]:  # Test with first PDF initially
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not found: {pdf_path}")
            continue
        
        logger.info(f"\n\nTesting: {name}")
        logger.info("-" * 40)
        
        try:
            # Test quality scorer
            quality_scores = await test_quality_scorer(api_key, pdf_path)
            
            # Only test feedback if quality needs improvement
            if quality_scores['overall_score'] < 0.8:
                # Test feedback extraction
                feedback_result = await test_feedback_extraction(api_key, pdf_path)
                
                # Compare baseline vs feedback
                comparison = await test_feedback_comparison(api_key, pdf_path)
                
                results.append({
                    "document": name,
                    "initial_score": quality_scores['overall_score'],
                    "final_score": feedback_result['quality_scores']['overall_score'],
                    "improvement": comparison['improvements']['overall_score_gain'],
                    "iterations": feedback_result['iterations']
                })
            else:
                logger.info(f"\nSkipping feedback test - quality already high: "
                           f"{quality_scores['overall_score']:.3f}")
                results.append({
                    "document": name,
                    "initial_score": quality_scores['overall_score'],
                    "final_score": quality_scores['overall_score'],
                    "improvement": 0.0,
                    "iterations": 0
                })
                
        except Exception as e:
            logger.error(f"Test failed for {name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Stage 3 Test Summary")
    logger.info("="*60)
    
    if results:
        avg_improvement = sum(r['improvement'] for r in results) / len(results)
        logger.info(f"Average Improvement: {avg_improvement:+.3f}")
        logger.info(f"Documents Improved: {sum(1 for r in results if r['improvement'] > 0)}/{len(results)}")
        
        for result in results:
            logger.info(f"\n{result['document']}:")
            logger.info(f"  Initial: {result['initial_score']:.3f} → "
                       f"Final: {result['final_score']:.3f} "
                       f"({result['improvement']:+.3f} in {result['iterations']} iterations)")


if __name__ == "__main__":
    asyncio.run(main())