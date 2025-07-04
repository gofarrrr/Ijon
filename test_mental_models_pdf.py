"""
Test enhanced extraction with The Great Mental Models PDF.
"""

import asyncio
import os
import sys
sys.path.append('/Users/marcin/Desktop/aplikacje/Ijon')

from extraction.document_aware.extractor import DocumentAwareExtractor
from extraction.document_aware.extractor_enhanced import EnhancedDocumentAwareExtractor
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Try to find the PDF with a simpler approach
def find_mental_models_pdf():
    """Find the Great Mental Models PDF."""
    base_path = "/Users/marcin/Desktop/aplikacje"
    
    # List all files in the directory
    for filename in os.listdir(base_path):
        if "Great Mental Models" in filename and filename.endswith(".pdf"):
            return os.path.join(base_path, filename)
    
    # If not found, try the exact name
    exact_name = "The Great Mental Models, Volume 1_ General Thinking Concepts -- Shane Parrish & Rhiannon Beaubien -- 2024 -- Penguin Publishing Group -- 9780593719978 -- 05e6df82c51f9b5aeaaf563cea13db10 -- Anna's Archive.pdf"
    exact_path = os.path.join(base_path, exact_name)
    
    if os.path.exists(exact_path):
        return exact_path
    
    return None

async def test_document_aware():
    """Test document-aware extraction."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in .env file")
        return
    
    # Find the PDF
    pdf_path = find_mental_models_pdf()
    if not pdf_path:
        print("Could not find The Great Mental Models PDF")
        print("Let's use a test PDF instead...")
        pdf_path = "/Users/marcin/Desktop/aplikacje/Ijon/test_documents/business_book.pdf"
        if not os.path.exists(pdf_path):
            print("No suitable PDF found for testing")
            return
    
    print(f"Testing with: {os.path.basename(pdf_path)}")
    print("="*80)
    
    # Test enhanced document-aware extraction
    print("\nENHANCED DOCUMENT-AWARE EXTRACTION")
    print("-"*60)
    
    try:
        enhanced = EnhancedDocumentAwareExtractor(api_key, model="gpt-3.5-turbo")
        result = await enhanced.extract_with_enhanced_awareness(pdf_path)
        
        print(f"\nDocument Profile:")
        print(f"  Type: {result.extraction_metadata.get('document_type', 'Unknown')}")
        print(f"  Complexity: {result.extraction_metadata.get('complexity_score', 'N/A')}")
        print(f"  Strategy: {result.extraction_metadata.get('strategy_used', 'N/A')}")
        
        print(f"\nExtraction Results:")
        print(f"  Topics: {len(result.topics)}")
        for i, topic in enumerate(result.topics[:3], 1):
            print(f"    {i}. {topic.name}")
            if topic.description:
                print(f"       {topic.description[:120]}...")
        
        print(f"\n  Facts: {len(result.facts)}")
        print(f"    With evidence: {sum(1 for f in result.facts if f.evidence)}")
        
        # Show sample facts
        evidence_facts = [f for f in result.facts if f.evidence]
        if evidence_facts:
            print(f"\n  Sample facts with evidence:")
            for fact in evidence_facts[:2]:
                print(f"    • {fact.claim}")
                print(f"      Evidence: {fact.evidence[:100]}...")
        
        print(f"\n  Relationships: {len(result.relationships)}")
        if result.relationships:
            print(f"  Sample relationships:")
            for rel in result.relationships[:2]:
                print(f"    • {rel.source_entity} {rel.relationship_type} {rel.target_entity}")
                if rel.description:
                    print(f"      {rel.description}")
        
        print(f"\n  Questions: {len(result.questions)}")
        # Cognitive level distribution
        levels = {}
        for q in result.questions:
            level = q.cognitive_level.value
            levels[level] = levels.get(level, 0) + 1
        print(f"    Distribution: {levels}")
        
        # Show diverse questions
        print(f"\n  Sample questions by level:")
        shown_levels = set()
        for q in result.questions:
            if q.cognitive_level.value not in shown_levels:
                print(f"    [{q.cognitive_level.value}] {q.question_text}")
                shown_levels.add(q.cognitive_level.value)
                if len(shown_levels) >= 3:
                    break
        
        print(f"\n  Summary:")
        if result.summary:
            # Wrap summary text
            import textwrap
            wrapped = textwrap.fill(result.summary, width=70, initial_indent="    ", subsequent_indent="    ")
            print(wrapped)
        
        print(f"\n  Overall Confidence: {result.overall_confidence:.2f}")
        
        # Quality metrics
        if "quality_metrics" in result.extraction_metadata:
            metrics = result.extraction_metadata["quality_metrics"]
            print(f"\n  Quality Metrics:")
            print(f"    Overall quality: {metrics.get('overall_quality', 0):.2f}")
            print(f"    Facts with evidence: {metrics.get('facts_with_evidence_ratio', 0):.1%}")
            print(f"    Question diversity: {metrics.get('question_diversity', 0):.1%}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run tests."""
    await test_document_aware()

if __name__ == "__main__":
    asyncio.run(main())