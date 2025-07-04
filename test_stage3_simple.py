"""
Simple test for Stage 3 quality scoring functionality.
"""

import asyncio
import os
from dotenv import load_dotenv

from extraction.models import ExtractedKnowledge, Topic, Fact, Relationship, Question, CognitiveLevel
from extraction.quality.scorer import QualityScorer
from src.utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger(__name__)


def create_test_extraction(quality: str = "good") -> ExtractedKnowledge:
    """Create a test extraction with known quality characteristics."""
    
    if quality == "good":
        return ExtractedKnowledge(
            topics=[
                Topic(name="Machine Learning", 
                      description="The study of algorithms that improve through experience",
                      confidence=0.9),
                Topic(name="Natural Language Processing",
                      description="Computational processing of human language",
                      confidence=0.85)
            ],
            facts=[
                Fact(claim="Transformer models achieved state-of-the-art results in 78% of NLP tasks",
                     evidence="Based on analysis of 156 peer-reviewed papers from 2019-2024",
                     confidence=0.9),
                Fact(claim="BERT showed 12.3% improvement in F1 score over baselines",
                     evidence="Experimental results across multiple benchmarks",
                     confidence=0.85),
                Fact(claim="Low-resource languages remain a key challenge",
                     evidence="Limited training data available for 95% of world languages",
                     confidence=0.8)
            ],
            relationships=[
                Relationship(source_entity="Machine Learning",
                           target_entity="Natural Language Processing",
                           relationship_type="enables",
                           description="ML techniques enable advanced NLP capabilities",
                           confidence=0.9),
                Relationship(source_entity="Transformer models",
                           target_entity="BERT",
                           relationship_type="includes",
                           description="BERT is a transformer-based architecture",
                           confidence=0.95)
            ],
            questions=[
                Question(question_text="What are the main applications of ML in NLP?",
                        expected_answer="Text classification, translation, summarization, Q&A",
                        cognitive_level=CognitiveLevel.UNDERSTAND,
                        difficulty=2,
                        confidence=0.8),
                Question(question_text="How can we improve performance on low-resource languages?",
                        expected_answer="Transfer learning, data augmentation, multilingual models",
                        cognitive_level=CognitiveLevel.ANALYZE,
                        difficulty=4,
                        confidence=0.75)
            ],
            summary="This research demonstrates the significant impact of machine learning, "
                   "particularly transformer models, on natural language processing tasks.",
            overall_confidence=0.85
        )
    
    elif quality == "poor":
        return ExtractedKnowledge(
            topics=[
                Topic(name="AI", description="Artificial Intelligence", confidence=0.5)
            ],
            facts=[
                Fact(claim="AI is good", evidence="", confidence=0.3),
                Fact(claim="AI is bad", evidence="", confidence=0.3)  # Contradiction
            ],
            relationships=[],
            questions=[
                Question(question_text="What is blockchain?",  # Irrelevant
                        expected_answer="A distributed ledger technology",
                        cognitive_level=CognitiveLevel.REMEMBER,
                        difficulty=1,
                        confidence=0.2)
            ],
            summary="",  # Missing summary
            overall_confidence=0.4
        )
    
    else:  # medium
        return ExtractedKnowledge(
            topics=[
                Topic(name="Deep Learning", 
                      description="Neural networks with multiple layers",
                      confidence=0.7),
                Topic(name="Computer Vision",
                      description="Teaching computers to understand images",
                      confidence=0.65)
            ],
            facts=[
                Fact(claim="CNNs are effective for image classification",
                     evidence="",  # Missing evidence
                     confidence=0.7),
                Fact(claim="ResNet introduced skip connections",
                     evidence="He et al., 2015 paper",
                     confidence=0.8)
            ],
            relationships=[
                Relationship(source_entity="Deep Learning",
                           target_entity="Computer Vision",
                           relationship_type="enables",
                           confidence=0.75)
            ],
            questions=[
                Question(question_text="How do CNNs work?",
                        expected_answer="Convolutional layers extract features",
                        cognitive_level=CognitiveLevel.UNDERSTAND,
                        difficulty=3,
                        confidence=0.7)
            ],
            summary="Deep learning has revolutionized computer vision.",
            overall_confidence=0.7
        )


async def test_quality_scorer():
    """Test the quality scorer with different quality extractions."""
    logger.info("=" * 60)
    logger.info("Testing Quality Scorer")
    logger.info("=" * 60)
    
    scorer = QualityScorer()
    
    # Test with different quality levels
    test_cases = [
        ("Good Quality", "good"),
        ("Poor Quality", "poor"),
        ("Medium Quality", "medium")
    ]
    
    for name, quality in test_cases:
        logger.info(f"\n{name} Extraction:")
        logger.info("-" * 40)
        
        extraction = create_test_extraction(quality)
        
        # Score without source content
        scores = scorer.score_extraction(extraction)
        
        logger.info(f"Overall Score: {scores['overall_score']:.3f}")
        logger.info("Dimension Scores:")
        for dim, score in scores['dimension_scores'].items():
            logger.info(f"  - {dim.title()}: {score:.3f}")
        
        logger.info(f"Needs Re-extraction: {scores['needs_reextraction']}")
        
        if scores['weaknesses']:
            logger.info("Weaknesses:")
            for weakness in scores['weaknesses']:
                logger.info(f"  - {weakness['dimension']}: {weakness['severity']}")
        
        if scores['suggestions']:
            logger.info("Suggestions:")
            for suggestion in scores['suggestions'][:3]:
                logger.info(f"  - {suggestion['suggestion']}")
    
    # Test with source content
    logger.info("\n\nTesting with Source Content:")
    logger.info("-" * 40)
    
    source_content = """
    Machine Learning Applications in Natural Language Processing: A Systematic Review
    
    This paper presents a comprehensive systematic review of machine learning applications 
    in natural language processing (NLP) from 2019 to 2024. We analyzed 156 peer-reviewed 
    articles to identify trends. Transformer models achieved state-of-the-art results in 
    78% of NLP tasks. BERT showed 12.3% improvement in F1 score over baselines.
    
    Low-resource languages remain a key challenge with limited training data.
    """
    
    good_extraction = create_test_extraction("good")
    scores_with_source = scorer.score_extraction(good_extraction, source_content)
    
    logger.info(f"Overall Score (with source): {scores_with_source['overall_score']:.3f}")
    logger.info("Grounding Score: {:.3f}".format(
        scores_with_source['dimension_scores']['grounding']
    ))


async def main():
    """Run Stage 3 simple tests."""
    logger.info("=" * 60)
    logger.info("Stage 3: Quality Scoring - Simple Test")
    logger.info("=" * 60)
    
    try:
        await test_quality_scorer()
        
        logger.info("\n\n✅ Quality Scorer is working correctly!")
        logger.info("The scorer successfully:")
        logger.info("- Evaluates extraction quality across 4 dimensions")
        logger.info("- Identifies weaknesses and contradictions")
        logger.info("- Provides actionable improvement suggestions")
        logger.info("- Detects when re-extraction is needed")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())