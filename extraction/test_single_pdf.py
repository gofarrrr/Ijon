"""
Test extraction on a single PDF to verify the pipeline.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from extraction.baseline import BaselineExtractor
from extraction.pdf_processor import PDFProcessor
from extraction.evaluator import ExtractionEvaluator

async def test_single_pdf():
    """Test extraction on one PDF."""
    
    # Use the technical manual as test
    pdf_path = Path("test_documents/technical_manual.pdf")
    
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return
    
    print(f"📄 Testing extraction on: {pdf_path.name}")
    
    # Initialize
    extractor = BaselineExtractor(os.getenv("OPENAI_API_KEY"))
    pdf_processor = PDFProcessor()
    evaluator = ExtractionEvaluator()
    
    try:
        # Extract first chunk
        print("📖 Extracting text from PDF...")
        chunks = await pdf_processor.process_pdf(str(pdf_path))
        print(f"✅ Found {len(chunks)} chunks")
        
        if not chunks:
            print("❌ No text extracted")
            return
        
        # Use first chunk
        chunk = chunks[0]
        print(f"\n📝 First chunk preview ({len(chunk.content)} chars):")
        print(chunk.content[:200] + "...")
        
        # Extract knowledge
        print("\n🧠 Extracting knowledge with OpenAI...")
        extraction = await extractor.extract(
            chunk_id="test_chunk",
            content=chunk.content
        )
        
        print("\n✅ Extraction Results:")
        print(f"  • Topics: {len(extraction.topics)}")
        for topic in extraction.topics:
            print(f"    - {topic.name}")
        
        print(f"  • Facts: {len(extraction.facts)}")
        for i, fact in enumerate(extraction.facts[:3]):  # First 3 facts
            print(f"    {i+1}. {fact.claim[:80]}...")
        
        print(f"  • Questions: {len(extraction.questions)}")
        for i, q in enumerate(extraction.questions[:3]):  # First 3 questions
            print(f"    {i+1}. {q.question_text}")
        
        print(f"  • Overall Confidence: {extraction.overall_confidence:.2f}")
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(extraction)
        print(f"\n📊 Metrics:")
        print(f"  • Average Confidence: {metrics.avg_confidence:.2f}")
        print(f"  • Processing Time: {metrics.processing_time_ms:.0f}ms")
        
        # Save results
        output_dir = Path("extraction/evaluation/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "single_pdf_test.json"
        with open(output_file, "w") as f:
            json.dump({
                "pdf": pdf_path.name,
                "chunk_size": len(chunk.content),
                "extraction_summary": {
                    "topics": len(extraction.topics),
                    "facts": len(extraction.facts),
                    "questions": len(extraction.questions),
                    "confidence": extraction.overall_confidence
                },
                "metrics": {
                    "avg_confidence": metrics.avg_confidence,
                    "processing_time_ms": metrics.processing_time_ms
                }
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_pdf())