"""
Debug enhancers to see what's happening.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction.models import ExtractedKnowledge, Topic, Fact
from extraction.v2.enhancers import CitationEnhancer, RelationshipEnhancer

def debug_enhancers():
    """Debug enhancer behavior."""
    
    # Create sample data
    extraction = ExtractedKnowledge(
        topics=[
            Topic(name="Machine Learning", description="AI subfield", confidence=0.9),
            Topic(name="Healthcare", description="Medical services", confidence=0.85)
        ],
        facts=[
            Fact(claim="Machine learning algorithms can predict disease onset with 85% accuracy", confidence=0.8),
            Fact(claim="Early detection through AI reduces treatment costs by 40%", confidence=0.75),
            Fact(claim="Machine learning is transforming healthcare through predictive analytics", confidence=0.85),
        ],
        overall_confidence=0.8
    )
    
    source_text = """
    Recent studies have shown that machine learning algorithms can predict disease onset 
    with 85% accuracy when analyzing patient data from electronic health records. 
    
    Early detection through AI reduces treatment costs by 40% according to a 2023 
    report from the Healthcare Analytics Institute.
    """
    
    print("ENHANCER DEBUG")
    print("=" * 60)
    
    print(f"\nOriginal extraction:")
    print(f"  Facts: {len(extraction.facts)}")
    for i, fact in enumerate(extraction.facts):
        print(f"    {i+1}. {fact.claim[:50]}...")
        print(f"       Evidence: {fact.evidence}")
    
    # Test citation enhancer
    print(f"\n\nApplying CitationEnhancer...")
    enhanced = CitationEnhancer.enhance(extraction, source_text)
    
    print(f"\nAfter citation enhancement:")
    print(f"  Facts: {len(enhanced.facts)}")
    facts_with_evidence = 0
    for i, fact in enumerate(enhanced.facts):
        print(f"    {i+1}. {fact.claim[:50]}...")
        print(f"       Evidence: {fact.evidence}")
        if fact.evidence:
            facts_with_evidence += 1
    print(f"\n  Facts with evidence: {facts_with_evidence}")
    
    # Test relationship enhancer
    print(f"\n\nApplying RelationshipEnhancer...")
    enhanced2 = RelationshipEnhancer.enhance(enhanced)
    
    print(f"\nAfter relationship enhancement:")
    print(f"  Relationships: {len(enhanced2.relationships)}")
    for i, rel in enumerate(enhanced2.relationships):
        print(f"    {i+1}. {rel.source_entity} -> {rel.target_entity}")
        print(f"       Type: {rel.relationship_type}")
        print(f"       Description: {rel.description}")

if __name__ == "__main__":
    debug_enhancers()