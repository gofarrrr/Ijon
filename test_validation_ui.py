"""
Test the validation UI with sample extraction data.
"""

import asyncio
import json
from datetime import datetime
from extraction.models import ExtractedKnowledge, Topic, Fact, Question, Relationship
from extraction.v2.validation_ui import create_validation_request, validation_store
from extraction.v2.state import ExtractionState
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def test_validation_ui():
    """Test validation UI with mock extraction."""
    
    # Create sample extraction
    extraction = ExtractedKnowledge(
        topics=[
            Topic(
                name="Machine Learning",
                description="AI subfield focusing on algorithms that improve through experience",
                confidence=0.9
            ),
            Topic(
                name="Healthcare",
                description="Medical and health services industry",
                confidence=0.85
            ),
            Topic(
                name="Predictive Analytics",
                description="Using data patterns to predict future outcomes",
                confidence=0.8
            )
        ],
        facts=[
            Fact(
                claim="Machine learning algorithms can predict disease onset with 85% accuracy",
                confidence=0.8,
                evidence="Source: Recent studies have shown that machine learning algorithms can predict disease onset with 85% accuracy when analyzing patient data."
            ),
            Fact(
                claim="Early detection through AI reduces treatment costs by 40%",
                confidence=0.75,
                evidence="Source: Early detection through AI reduces treatment costs by 40% according to a 2023 report."
            ),
            Fact(
                claim="Deep learning models analyze medical images faster than human radiologists",
                confidence=0.85
            ),
            Fact(
                claim="AI-driven drug discovery reduces development time from 10 years to 5 years",
                confidence=0.7
            )
        ],
        questions=[
            Question(
                question_text="How do machine learning algorithms achieve high accuracy in disease prediction?",
                cognitive_level="analyze",
                expected_answer="By analyzing patterns in large datasets of patient records",
                difficulty=3,
                confidence=0.85
            ),
            Question(
                question_text="What are the ethical implications of AI in healthcare?",
                cognitive_level="evaluate",
                expected_answer="Privacy concerns, bias in algorithms, and accountability for decisions",
                difficulty=4,
                confidence=0.8
            )
        ],
        relationships=[
            Relationship(
                source_entity="Machine Learning",
                target_entity="Healthcare",
                relationship_type="transforms",
                description="Machine Learning transforms Healthcare through predictive analytics",
                confidence=0.9
            ),
            Relationship(
                source_entity="Predictive Analytics",
                target_entity="Healthcare",
                relationship_type="improves",
                description="Predictive Analytics improves Healthcare outcomes",
                confidence=0.85
            )
        ],
        summary="Machine learning is revolutionizing healthcare through predictive analytics, enabling early disease detection and reducing treatment costs.",
        overall_confidence=0.82
    )
    
    # Create quality report with some issues
    quality_report = {
        "overall_score": 0.68,  # Below threshold to trigger validation
        "dimensions": {
            "consistency": 0.85,
            "grounding": 0.55,  # Low grounding score
            "coherence": 0.9,
            "completeness": 0.7
        },
        "weaknesses": [
            {
                "dimension": "grounding",
                "severity": "medium",
                "details": "Several facts lack proper evidence or citations"
            },
            {
                "dimension": "completeness",
                "severity": "low", 
                "details": "Could include more details about implementation challenges"
            }
        ],
        "strengths": [
            "Good topic coverage",
            "Clear relationships identified",
            "High coherence between facts"
        ]
    }
    
    # Create validation request
    logger.info("Creating validation request...")
    val_state = create_validation_request(
        state_id="test_extraction_001",
        extraction=extraction,
        quality_report=quality_report,
        pdf_path="/path/to/test_document.pdf"
    )
    
    # Save to store
    await validation_store.save(val_state)
    logger.info(f"✅ Validation request created with ID: {val_state.id}")
    
    # Create another one for dashboard testing
    extraction2 = ExtractedKnowledge(
        topics=[
            Topic(name="Quantum Computing", description="Computing using quantum phenomena", confidence=0.95),
            Topic(name="Cryptography", description="Science of secure communication", confidence=0.9)
        ],
        facts=[
            Fact(claim="Quantum computers can break RSA encryption", confidence=0.8),
            Fact(claim="Post-quantum cryptography is being developed", confidence=0.85)
        ],
        overall_confidence=0.85
    )
    
    quality_report2 = {
        "overall_score": 0.45,  # Very low score
        "dimensions": {
            "consistency": 0.6,
            "grounding": 0.3,
            "coherence": 0.5,
            "completeness": 0.4
        },
        "weaknesses": [
            {
                "dimension": "grounding",
                "severity": "high",
                "details": "Most facts lack evidence"
            },
            {
                "dimension": "completeness",
                "severity": "high",
                "details": "Missing critical context about quantum computing limitations"
            }
        ]
    }
    
    val_state2 = create_validation_request(
        state_id="test_extraction_002",
        extraction=extraction2,
        quality_report=quality_report2,
        pdf_path="/path/to/quantum_computing.pdf"
    )
    
    await validation_store.save(val_state2)
    logger.info(f"✅ Second validation request created with ID: {val_state2.id}")
    
    # Test loading
    loaded = await validation_store.load(val_state.id)
    assert loaded is not None
    assert loaded.id == val_state.id
    logger.info("✅ Successfully loaded validation request from store")
    
    # List active validations
    active = await validation_store.list_active()
    pending_validations = [
        s for s in active 
        if s.metadata.get("needs_validation") and s.status == "pending_validation"
    ]
    logger.info(f"✅ Found {len(pending_validations)} pending validations")
    
    print("\n" + "="*60)
    print("VALIDATION UI TEST COMPLETE")
    print("="*60)
    print(f"\nCreated {len(pending_validations)} test validations:")
    for val in pending_validations:
        print(f"\n- ID: {val.id}")
        print(f"  PDF: {val.pdf_path}")
        print(f"  Quality Score: {val.quality_report.get('overall_score', 0)*100:.0f}%")
        print(f"  Status: {val.status}")
    
    print("\n✅ To view the validation UI:")
    print("   1. Run: python extraction/v2/validation_ui.py")
    print("   2. Open: http://localhost:8001/validator/dashboard")
    print(f"   3. Click on a validation to review")
    
    return val_state.id

if __name__ == "__main__":
    extraction_id = asyncio.run(test_validation_ui())
    print(f"\n📋 Direct link to first validation:")
    print(f"   http://localhost:8001/validator/{extraction_id}")