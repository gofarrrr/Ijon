"""
Test script for baseline extractor.

This script tests the baseline extractor with sample text to verify it works correctly.
"""

import asyncio
import os
import json
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from extraction.baseline import BaselineExtractor


async def test_baseline_extractor():
    """Test the baseline extractor with sample text."""
    
    # Sample text for testing
    sample_text = """
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being explicitly 
    programmed. Machine learning focuses on the development of computer programs that can 
    access data and use it to learn for themselves.
    
    The process of learning begins with observations or data, such as examples, direct 
    experience, or instruction, in order to look for patterns in data and make better 
    decisions in the future based on the examples that we provide. The primary aim is to 
    allow the computers to learn automatically without human intervention or assistance 
    and adjust actions accordingly.
    
    Some machine learning methods include:
    - Supervised learning: The computer is presented with example inputs and their desired 
      outputs, and the goal is to learn a general rule that maps inputs to outputs.
    - Unsupervised learning: No labels are given to the learning algorithm, leaving it 
      on its own to find structure in its input.
    - Reinforcement learning: A computer program interacts with a dynamic environment in 
      which it must perform a certain goal.
    """
    
    # Initialize extractor
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    extractor = BaselineExtractor(openai_api_key=openai_key)
    
    print("🔍 Testing baseline extractor...")
    print("=" * 70)
    
    # Extract knowledge
    try:
        knowledge = await extractor.extract(
            chunk_id="test_chunk_001",
            content=sample_text,
            chunk_metadata={"source": "test", "page": 1}
        )
        
        # Display results
        print(f"\n✅ Extraction completed!")
        print(f"Overall confidence: {knowledge.overall_confidence:.2f}")
        
        print(f"\n📚 Topics ({len(knowledge.topics)}):")
        for topic in knowledge.topics:
            print(f"  - {topic.name}: {topic.description[:50]}... (confidence: {topic.confidence:.2f})")
        
        print(f"\n📋 Facts ({len(knowledge.facts)}):")
        for fact in knowledge.facts:
            print(f"  - {fact.claim[:80]}... (confidence: {fact.confidence:.2f})")
        
        print(f"\n🔗 Relationships ({len(knowledge.relationships)}):")
        for rel in knowledge.relationships:
            print(f"  - {rel.source_entity} {rel.relationship_type} {rel.target_entity} (confidence: {rel.confidence:.2f})")
        
        print(f"\n❓ Questions ({len(knowledge.questions)}):")
        for q in knowledge.questions:
            print(f"  - [{q.cognitive_level.value}] {q.question_text} (difficulty: {q.difficulty}/5)")
        
        print(f"\n📝 Summary:")
        print(f"  {knowledge.summary}")
        
        # Save results
        output_dir = Path("extraction/baseline/test_outputs")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "baseline_test_result.json"
        with open(output_file, "w") as f:
            json.dump(knowledge.model_dump(), f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_baseline_extractor())