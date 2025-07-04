"""
Direct runner for model routing tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.test_model_routing import TestModelRouting, test_routing_performance

def run_tests():
    """Run all model routing tests."""
    print("🧪 RUNNING MODEL ROUTING TESTS")
    print("=" * 60)
    
    test_suite = TestModelRouting()
    tests = [
        ("test_academic_document_routing", test_suite.test_academic_document_routing),
        ("test_technical_document_routing", test_suite.test_technical_document_routing),
        ("test_long_document_routing", test_suite.test_long_document_routing),
        ("test_default_routing", test_suite.test_default_routing),
        ("test_extractor_selection", test_suite.test_extractor_selection),
        ("test_edge_cases", test_suite.test_edge_cases),
        ("test_routing_consistency", test_suite.test_routing_consistency),
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
            failed += 1
    
    # Run performance test
    print(f"\n✓ Running performance test...")
    try:
        test_routing_performance()
        print(f"  ✅ PASSED")
        passed += 1
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)