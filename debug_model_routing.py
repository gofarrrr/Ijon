"""
Debug model routing to see actual outputs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extraction.v2.extractors import select_model_for_document

def debug_routing():
    """Debug routing decisions."""
    test_cases = [
        ("Academic + Budget", {
            "doc_type": "academic",
            "doc_length": 5000,
            "quality_required": False,
            "budget_conscious": True
        }),
        ("Technical + Quality", {
            "doc_type": "technical",
            "doc_length": 10000,
            "quality_required": True,
            "budget_conscious": False
        }),
        ("Unknown + Quality", {
            "doc_type": "unknown",
            "doc_length": 3000,
            "quality_required": True,
            "budget_conscious": False
        }),
        ("Extremely long", {
            "doc_type": "technical",
            "doc_length": 500000,
            "quality_required": True,
            "budget_conscious": False
        }),
    ]
    
    print("MODEL ROUTING DEBUG")
    print("=" * 80)
    
    for name, params in test_cases:
        config = select_model_for_document(**params)
        print(f"\n{name}:")
        print(f"  Input: {params}")
        print(f"  Model: {config['model']}")
        print(f"  Reason: {config['reason']}")
        print(f"  Extractor: {config['extractor'].__name__}")

if __name__ == "__main__":
    debug_routing()