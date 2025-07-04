"""
Quick demonstration of enhanced extraction improvements.
"""

import asyncio
import os
import sys
sys.path.append('/Users/marcin/Desktop/aplikacje/Ijon')

from extraction.baseline.extractor import BaselineExtractor
from extraction.baseline.extractor_enhanced import EnhancedBaselineExtractor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample content about mental models (from The Great Mental Models book concept)
SAMPLE_CONTENT = """
The Map is Not the Territory

The map of reality is not reality itself. Even the best maps are imperfect because they are reductions 
of what they represent. If a map were to represent the territory with perfect fidelity, it would no 
longer be a reduction and thus would no longer be useful to us. A map can also be a snapshot of a 
point in time, representing something that no longer exists. This is important to keep in mind as we 
think through problems and make better decisions.

We can't navigate reality without some sort of mental map. The key is remembering that our maps are 
not reality itself—they are tools we use to navigate reality. When we forget this, we start to think 
that the map is the territory, and we can get into trouble. Alfred Korzybski, who coined the phrase 
"the map is not the territory," put it this way: "A map is not the territory it represents, but, if 
correct, it has a similar structure to the territory, which accounts for its usefulness."

The map is a reduction of the territory. It is a simplified representation that is useful precisely 
because it doesn't show everything. A map that showed every tree, every blade of grass, every pothole, 
would be so detailed as to be unusable. We create maps—mental models—precisely because we need to 
reduce complexity to something we can work with.
"""

async def main():
    """Run a demonstration."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in .env file")
        return
    
    print("ENHANCED EXTRACTION DEMONSTRATION")
    print("="*80)
    print(f"\nContent preview: {SAMPLE_CONTENT[:150]}...\n")
    
    # Test original
    print("ORIGINAL BASELINE EXTRACTOR")
    print("-"*60)
    
    original = BaselineExtractor(api_key, model="gpt-3.5-turbo")
    original_result = await original.extract("demo1", SAMPLE_CONTENT)
    
    print(f"Topics: {len(original_result.topics)}")
    for topic in original_result.topics:
        print(f"  • {topic.name}")
    
    print(f"\nFacts: {len(original_result.facts)}")
    print(f"  With evidence: {sum(1 for f in original_result.facts if f.evidence)}")
    if original_result.facts:
        print(f"  Example: {original_result.facts[0].claim[:80]}...")
    
    print(f"\nQuestions: {len(original_result.questions)}")
    print(f"Confidence: {original_result.overall_confidence:.2f}")
    
    # Test enhanced
    print("\n\nENHANCED BASELINE EXTRACTOR")
    print("-"*60)
    
    enhanced = EnhancedBaselineExtractor(api_key)
    enhanced.model = "gpt-3.5-turbo"  # Use same model for fair comparison
    enhanced_result = await enhanced.extract("demo2", SAMPLE_CONTENT)
    
    print(f"Topics: {len(enhanced_result.topics)}")
    for topic in enhanced_result.topics:
        print(f"  • {topic.name}")
        if topic.description:
            print(f"    Description: {topic.description[:100]}...")
    
    print(f"\nFacts: {len(enhanced_result.facts)}")
    print(f"  With evidence: {sum(1 for f in enhanced_result.facts if f.evidence)}")
    if enhanced_result.facts:
        for fact in enhanced_result.facts[:2]:
            print(f"\n  Fact: {fact.claim}")
            if fact.evidence:
                print(f"  Evidence: {fact.evidence[:100]}...")
    
    print(f"\nQuestions: {len(enhanced_result.questions)}")
    # Show cognitive distribution
    levels = {}
    for q in enhanced_result.questions:
        levels[q.cognitive_level.value] = levels.get(q.cognitive_level.value, 0) + 1
    print(f"  Cognitive levels: {levels}")
    
    # Show a sample question
    if enhanced_result.questions:
        q = enhanced_result.questions[0]
        print(f"\n  Sample question:")
        print(f"    Q: {q.question_text}")
        if q.expected_answer:
            print(f"    A: {q.expected_answer[:150]}...")
        print(f"    Level: {q.cognitive_level.value}, Difficulty: {q.difficulty}")
    
    print(f"\nSummary:")
    if enhanced_result.summary:
        print(f"  {enhanced_result.summary}")
    
    print(f"\nConfidence: {enhanced_result.overall_confidence:.2f}")
    
    # Show improvements
    print("\n\nIMPROVEMENTS SUMMARY")
    print("-"*60)
    
    print(f"Topics:         {len(original_result.topics)} → {len(enhanced_result.topics)}")
    print(f"Facts:          {len(original_result.facts)} → {len(enhanced_result.facts)}")
    print(f"With evidence:  {sum(1 for f in original_result.facts if f.evidence)} → {sum(1 for f in enhanced_result.facts if f.evidence)}")
    print(f"Questions:      {len(original_result.questions)} → {len(enhanced_result.questions)}")
    print(f"Question types: {len(set(q.cognitive_level for q in original_result.questions))} → {len(set(q.cognitive_level for q in enhanced_result.questions))}")
    print(f"Confidence:     {original_result.overall_confidence:.2f} → {enhanced_result.overall_confidence:.2f}")
    
    # Quality assessment
    print("\nQuality Assessment:")
    print(f"  ✓ Academic prose in descriptions: {'Yes' if any(len(t.description) > 100 for t in enhanced_result.topics) else 'No'}")
    print(f"  ✓ Evidence-based facts: {'Yes' if sum(1 for f in enhanced_result.facts if f.evidence) > len(enhanced_result.facts) * 0.7 else 'Partial'}")
    print(f"  ✓ Diverse question levels: {'Yes' if len(levels) >= 4 else 'Partial'}")
    print(f"  ✓ Comprehensive summary: {'Yes' if enhanced_result.summary and len(enhanced_result.summary) > 150 else 'No'}")

if __name__ == "__main__":
    asyncio.run(main())