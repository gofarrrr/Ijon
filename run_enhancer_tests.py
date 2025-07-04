"""
Direct runner for enhancer composition tests (non-API tests only).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_enhancer_composition import TestEnhancerComposition
from extraction.models import ExtractedKnowledge, Topic, Fact, Question, Relationship

def run_tests():
    """Run enhancer tests that don't require API."""
    print("🧪 RUNNING ENHANCER COMPOSITION TESTS")
    print("=" * 60)
    
    test_suite = TestEnhancerComposition()
    
    # Create fixtures
    sample_extraction = ExtractedKnowledge(
        topics=[
            Topic(name="Machine Learning", description="AI subfield focusing on algorithms that improve through experience", confidence=0.9),
            Topic(name="Healthcare", description="Medical and health services", confidence=0.85),
            Topic(name="Predictive Analytics", description="Using data to predict future outcomes", confidence=0.8)
        ],
        facts=[
            Fact(claim="Machine learning algorithms can predict disease onset with 85% accuracy", confidence=0.8),
            Fact(claim="Early detection through AI reduces treatment costs by 40%", confidence=0.75),
            Fact(claim="Deep learning models analyze medical images faster than human radiologists", confidence=0.85),
            Fact(claim="AI-driven drug discovery reduces development time from 10 years to 5 years", confidence=0.7),
            Fact(claim="Machine Learning is revolutionizing Healthcare through Predictive Analytics applications", confidence=0.9)
        ],
        overall_confidence=0.8
    )
    
    sample_source_text = """
    Recent studies have shown that machine learning algorithms can predict disease onset 
    with 85% accuracy when analyzing patient data from electronic health records. 
    
    The implementation of AI in healthcare has led to significant cost reductions. 
    Early detection through AI reduces treatment costs by 40% according to a 2023 
    report from the Healthcare Analytics Institute.
    
    In radiology, deep learning models analyze medical images faster than human 
    radiologists, processing scans in seconds rather than minutes.
    
    Pharmaceutical companies are leveraging AI-driven drug discovery to reduce 
    development time from the traditional 10 years to just 5 years.
    """
    
    tests = [
        ("test_citation_enhancer", lambda: test_suite.test_citation_enhancer(sample_extraction, sample_source_text)),
        ("test_relationship_enhancer", lambda: test_suite.test_relationship_enhancer(sample_extraction)),
        ("test_enhancer_idempotency", lambda: test_suite.test_enhancer_idempotency(sample_extraction, sample_source_text)),
        ("test_enhancer_performance", lambda: test_suite.test_enhancer_performance(sample_extraction, sample_source_text)),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n✓ Running {test_name}...")
            test_func()
            print(f"  ✅ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    print("\nNote: Async tests requiring API keys were skipped.")
    print("To run full tests, set OPENAI_API_KEY environment variable.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)