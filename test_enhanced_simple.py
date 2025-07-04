"""
Simple test script to demonstrate enhanced extraction improvements.
"""

import asyncio
import os
import sys
sys.path.append('/Users/marcin/Desktop/aplikacje/Ijon')

from extraction.baseline.extractor import BaselineExtractor
from extraction.baseline.extractor_enhanced import EnhancedBaselineExtractor
from extraction.pdf_processor import PDFProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PDF path - using a test PDF for now
PDF_PATH = "/Users/marcin/Desktop/aplikacje/Ijon/test_pdfs/ai_research_paper.pdf"

async def main():
    """Run a simple comparison test."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in .env file")
        return
    
    print("Processing PDF...")
    processor = PDFProcessor()
    chunks = await processor.process_pdf(PDF_PATH)
    
    if not chunks:
        print("Failed to process PDF")
        return
    
    # Use first chunk
    chunk = chunks[0]
    print(f"\nTesting with chunk from page: {chunk.page_num}, index: {chunk.chunk_index}")
    print(f"Content preview: {chunk.content[:200]}...\n")
    
    # Test original
    print("="*60)
    print("ORIGINAL BASELINE EXTRACTOR")
    print("="*60)
    
    original = BaselineExtractor(api_key, model="gpt-3.5-turbo")
    original_result = await original.extract("test1", chunk.content)
    
    print(f"Topics: {len(original_result.topics)}")
    for i, topic in enumerate(original_result.topics[:3], 1):
        print(f"  {i}. {topic.name}")
    
    print(f"\nFacts: {len(original_result.facts)}")
    print(f"  With evidence: {sum(1 for f in original_result.facts if f.evidence)}")
    
    print(f"\nQuestions: {len(original_result.questions)}")
    levels = {}
    for q in original_result.questions:
        levels[q.cognitive_level.value] = levels.get(q.cognitive_level.value, 0) + 1
    print(f"  Levels: {levels}")
    
    print(f"\nConfidence: {original_result.overall_confidence:.2f}")
    
    # Test enhanced
    print("\n" + "="*60)
    print("ENHANCED BASELINE EXTRACTOR")
    print("="*60)
    
    enhanced = EnhancedBaselineExtractor(api_key)
    enhanced_result = await enhanced.extract("test2", chunk.content)
    
    print(f"Topics: {len(enhanced_result.topics)}")
    for i, topic in enumerate(enhanced_result.topics[:3], 1):
        print(f"  {i}. {topic.name}")
        if topic.description:
            print(f"     {topic.description[:100]}...")
    
    print(f"\nFacts: {len(enhanced_result.facts)}")
    print(f"  With evidence: {sum(1 for f in enhanced_result.facts if f.evidence)}")
    if enhanced_result.facts:
        print(f"\nSample fact with evidence:")
        for fact in enhanced_result.facts:
            if fact.evidence:
                print(f"  Claim: {fact.claim[:80]}...")
                print(f"  Evidence: {fact.evidence[:80]}...")
                break
    
    print(f"\nQuestions: {len(enhanced_result.questions)}")
    levels = {}
    for q in enhanced_result.questions:
        levels[q.cognitive_level.value] = levels.get(q.cognitive_level.value, 0) + 1
    print(f"  Levels: {levels}")
    
    if enhanced_result.questions:
        print(f"\nSample question:")
        q = enhanced_result.questions[0]
        print(f"  Q: {q.question_text}")
        if q.expected_answer:
            print(f"  A: {q.expected_answer[:100]}...")
        print(f"  Level: {q.cognitive_level.value}, Difficulty: {q.difficulty}")
    
    print(f"\nSummary:")
    if enhanced_result.summary:
        print(f"  {enhanced_result.summary[:200]}...")
    
    print(f"\nConfidence: {enhanced_result.overall_confidence:.2f}")
    
    # Comparison
    print("\n" + "="*60)
    print("IMPROVEMENTS")
    print("="*60)
    
    print(f"Topics:    {len(original_result.topics)} → {len(enhanced_result.topics)} (+{len(enhanced_result.topics) - len(original_result.topics)})")
    print(f"Facts:     {len(original_result.facts)} → {len(enhanced_result.facts)} (+{len(enhanced_result.facts) - len(original_result.facts)})")
    
    orig_evidence = sum(1 for f in original_result.facts if f.evidence)
    enh_evidence = sum(1 for f in enhanced_result.facts if f.evidence)
    print(f"Evidence:  {orig_evidence} → {enh_evidence} (+{enh_evidence - orig_evidence})")
    
    print(f"Questions: {len(original_result.questions)} → {len(enhanced_result.questions)} (+{len(enhanced_result.questions) - len(original_result.questions)})")
    
    orig_levels = len(set(q.cognitive_level for q in original_result.questions))
    enh_levels = len(set(q.cognitive_level for q in enhanced_result.questions))
    print(f"Q Levels:  {orig_levels} → {enh_levels} (+{enh_levels - orig_levels})")
    
    print(f"Confidence: {original_result.overall_confidence:.2f} → {enhanced_result.overall_confidence:.2f} ({enhanced_result.overall_confidence - original_result.overall_confidence:+.2f})")

if __name__ == "__main__":
    asyncio.run(main())