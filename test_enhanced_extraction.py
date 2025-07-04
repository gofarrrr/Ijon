"""
Test script to compare original vs enhanced extraction systems.

Tests The Great Mental Models PDF with various enhanced extractors.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any

# Add project root to path
import sys
sys.path.append('/Users/marcin/Desktop/aplikacje/Ijon')

from extraction.baseline.extractor import BaselineExtractor
from extraction.baseline.extractor_enhanced import EnhancedBaselineExtractor
from extraction.document_aware.extractor import DocumentAwareExtractor
from extraction.document_aware.extractor_enhanced import EnhancedDocumentAwareExtractor
from extraction.quality.feedback_extractor import FeedbackExtractor
from extraction.quality.feedback_extractor_enhanced import EnhancedFeedbackExtractor
from src.rag.generator import AnswerGenerator
from src.rag.generator_enhanced import EnhancedAnswerGenerator
from extraction.pdf_processor import PDFProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)

# PDF path
PDF_PATH = "/Users/marcin/Desktop/aplikacje/The Great Mental Models, Volume 1_ General Thinking Concepts -- Shane Parrish & Rhiannon Beaubien -- 2024 -- Penguin Publishing Group -- 9780593719978 -- 05e6df82c51f9b5aeaaf563cea13db10 -- Anna's Archive.pdf"

# OpenAI API key (you'll need to set this)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}\n")

def print_extraction_summary(extraction: Any, label: str):
    """Print a summary of extraction results."""
    print(f"\n{label}:")
    print(f"  Topics: {len(extraction.topics)}")
    print(f"  Facts: {len(extraction.facts)}")
    print(f"  - With evidence: {sum(1 for f in extraction.facts if f.evidence)}")
    print(f"  Relationships: {len(extraction.relationships)}")
    print(f"  Questions: {len(extraction.questions)}")
    
    # Cognitive level distribution
    if extraction.questions:
        levels = {}
        for q in extraction.questions:
            level = q.cognitive_level.value
            levels[level] = levels.get(level, 0) + 1
        print(f"  - Question levels: {dict(levels)}")
    
    print(f"  Confidence: {extraction.overall_confidence:.2f}")
    
    # Sample content
    if extraction.topics:
        print(f"\n  Sample topic: {extraction.topics[0].name}")
        print(f"  Description: {extraction.topics[0].description[:150]}...")
    
    if extraction.summary:
        print(f"\n  Summary preview: {extraction.summary[:200]}...")

async def test_baseline_extraction():
    """Test original vs enhanced baseline extraction."""
    print_section("BASELINE EXTRACTOR COMPARISON")
    
    # Process PDF
    processor = PDFProcessor()
    chunks = await processor.process_pdf(PDF_PATH)
    
    if not chunks:
        print("Failed to extract chunks from PDF")
        return
    
    # Use first chunk for baseline test
    chunk = chunks[0]
    print(f"Testing with chunk from pages: {chunk.page_numbers}")
    print(f"Content preview: {chunk.content[:200]}...\n")
    
    # Original baseline
    print("Running original baseline extractor...")
    original = BaselineExtractor(OPENAI_API_KEY)
    original_result = await original.extract(
        chunk_id="test_original",
        content=chunk.content
    )
    print_extraction_summary(original_result, "Original Baseline")
    
    # Enhanced baseline
    print("\nRunning enhanced baseline extractor...")
    enhanced = EnhancedBaselineExtractor(OPENAI_API_KEY)
    enhanced_result = await enhanced.extract(
        chunk_id="test_enhanced",
        content=chunk.content
    )
    print_extraction_summary(enhanced_result, "Enhanced Baseline")
    
    # Compare quality
    print("\nQuality Improvements:")
    print(f"  Topics: +{len(enhanced_result.topics) - len(original_result.topics)}")
    print(f"  Facts with evidence: +{sum(1 for f in enhanced_result.facts if f.evidence) - sum(1 for f in original_result.facts if f.evidence)}")
    print(f"  Confidence: {enhanced_result.overall_confidence - original_result.overall_confidence:+.2f}")

async def test_document_aware_extraction():
    """Test original vs enhanced document-aware extraction."""
    print_section("DOCUMENT-AWARE EXTRACTOR COMPARISON")
    
    # Original document-aware
    print("Running original document-aware extractor...")
    original = DocumentAwareExtractor(OPENAI_API_KEY)
    original_result = await original.extract(PDF_PATH)
    print_extraction_summary(original_result, "Original Document-Aware")
    
    # Enhanced document-aware
    print("\nRunning enhanced document-aware extractor...")
    enhanced = EnhancedDocumentAwareExtractor(OPENAI_API_KEY)
    enhanced_result = await enhanced.extract_with_enhanced_awareness(PDF_PATH)
    print_extraction_summary(enhanced_result, "Enhanced Document-Aware")
    
    # Show document profiling
    if "document_type" in enhanced_result.extraction_metadata:
        print(f"\nDocument Profile:")
        print(f"  Type: {enhanced_result.extraction_metadata['document_type']}")
        print(f"  Complexity: {enhanced_result.extraction_metadata.get('complexity_score', 'N/A')}")
        print(f"  Strategy: {enhanced_result.extraction_metadata.get('strategy_used', 'N/A')}")

async def test_feedback_extraction():
    """Test feedback extraction with quality improvements."""
    print_section("FEEDBACK EXTRACTOR TEST")
    
    # Only test enhanced version due to time
    print("Running enhanced feedback extractor (iterative improvement)...")
    enhanced = EnhancedFeedbackExtractor(OPENAI_API_KEY)
    result = await enhanced.extract_with_enhanced_feedback(PDF_PATH)
    
    print(f"\nIterations completed: {result['iterations']}")
    print(f"Total improvement: {result['total_improvement']:.3f}")
    
    # Show improvement trajectory
    print("\nQuality score progression:")
    for i, score in enumerate(result['improvement_trajectory'], 1):
        print(f"  Iteration {i}: {score:.3f}")
    
    # Final extraction summary
    print_extraction_summary(result['extraction'], "Final Enhanced Extraction")
    
    # Key improvements
    if result['key_improvements']:
        print("\nKey improvements made:")
        for improvement in result['key_improvements'][:3]:
            print(f"  - {improvement}")

async def test_rag_generation():
    """Test RAG answer generation."""
    print_section("RAG GENERATOR COMPARISON")
    
    # Get some chunks for RAG
    processor = PDFProcessor()
    chunks = await processor.process_pdf(PDF_PATH)
    
    if len(chunks) < 3:
        print("Not enough chunks for RAG test")
        return
    
    # Prepare chunks with scores (simulate retrieval)
    scored_chunks = [(chunk, 0.85 - i*0.1) for i, chunk in enumerate(chunks[:3])]
    
    query = "What are the main mental models discussed in this book and how do they help with decision making?"
    
    print(f"Query: {query}\n")
    
    # Original generator
    print("Running original RAG generator...")
    original = AnswerGenerator(OPENAI_API_KEY)
    original_answer = await original.generate_answer(query, scored_chunks)
    
    print(f"Original Answer ({len(original_answer.answer)} chars):")
    print(f"{original_answer.answer[:300]}...")
    print(f"Citations: {len(original_answer.citations)}")
    print(f"Confidence: {original_answer.confidence_score:.2f}")
    
    # Enhanced generator
    print("\n\nRunning enhanced RAG generator...")
    enhanced = EnhancedAnswerGenerator(OPENAI_API_KEY)
    enhanced_answer = await enhanced.generate_answer(query, scored_chunks)
    
    print(f"Enhanced Answer ({len(enhanced_answer.answer)} chars):")
    print(f"{enhanced_answer.answer[:500]}...")
    print(f"Citations: {len(enhanced_answer.citations)}")
    print(f"Confidence: {enhanced_answer.confidence_score:.2f}")
    
    # Quality check
    quality = await enhanced.check_enhanced_answer_quality(enhanced_answer)
    print(f"\nAnswer Quality Analysis:")
    print(f"  Academic prose score: {quality['academic_prose_score']:.2f}")
    print(f"  Query relevance: {quality['query_relevance']:.2f}")
    print(f"  Overall quality: {quality['overall_quality']:.2f}")
    print(f"  Quality grade: {quality['quality_grade']}")

async def main():
    """Run all tests."""
    if not OPENAI_API_KEY:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    print(f"Testing enhanced extraction on: {os.path.basename(PDF_PATH)}")
    print(f"Start time: {datetime.now()}")
    
    try:
        # Test different extraction methods
        await test_baseline_extraction()
        await test_document_aware_extraction()
        
        # These take longer, so make them optional
        if input("\nRun feedback extraction test? (y/n): ").lower() == 'y':
            await test_feedback_extraction()
        
        if input("\nRun RAG generation test? (y/n): ").lower() == 'y':
            await test_rag_generation()
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\nTests completed at: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())