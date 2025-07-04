"""
Demo script showing the complete validation workflow.
"""

import asyncio
import webbrowser
from extraction.models import ExtractedKnowledge, Topic, Fact, Question, Relationship
from extraction.v2.validation_ui import create_validation_request, validation_store, get_validation_stats
from src.utils.logging import get_logger

logger = get_logger(__name__)

async def demo_validation_workflow():
    """Demonstrate the complete validation workflow."""
    
    print("\n" + "="*60)
    print("VALIDATION UI WORKFLOW DEMO")
    print("="*60)
    
    # Step 1: Create a sample extraction with quality issues
    print("\n1️⃣ Creating sample extraction with quality issues...")
    
    extraction = ExtractedKnowledge(
        topics=[
            Topic(
                name="Artificial Intelligence",
                description="Computer systems that can perform tasks requiring human intelligence",
                confidence=0.7
            ),
            Topic(
                name="Climate Change",
                description="Long-term shifts in global temperatures and weather patterns",
                confidence=0.65
            )
        ],
        facts=[
            Fact(
                claim="AI can solve climate change",
                confidence=0.4  # Low confidence, vague claim
            ),
            Fact(
                claim="Global temperatures have risen",
                confidence=0.5,  # Missing evidence
            ),
            Fact(
                claim="Machine learning requires data",
                confidence=0.6
            )
        ],
        questions=[
            Question(
                question_text="How can AI help with climate change?",
                cognitive_level="analyze",
                expected_answer="Through optimization of energy systems and climate modeling",
                difficulty=3,
                confidence=0.7
            )
        ],
        relationships=[],  # Missing relationships
        summary="AI and climate change are related topics.",  # Weak summary
        overall_confidence=0.55
    )
    
    # Step 2: Generate quality report showing issues
    print("\n2️⃣ Quality assessment reveals issues...")
    
    quality_report = {
        "overall_score": 0.48,  # Below threshold
        "dimensions": {
            "consistency": 0.5,
            "grounding": 0.3,   # Very low - no evidence
            "coherence": 0.6,
            "completeness": 0.5
        },
        "weaknesses": [
            {
                "dimension": "grounding",
                "severity": "high",
                "details": "Most facts lack supporting evidence or citations"
            },
            {
                "dimension": "coherence",
                "severity": "medium",
                "details": "Topics don't connect well, missing relationships"
            },
            {
                "dimension": "completeness",
                "severity": "medium", 
                "details": "Summary is too brief and doesn't capture key insights"
            }
        ],
        "strengths": [
            "Topics are relevant",
            "Questions are well-formed"
        ]
    }
    
    # Step 3: Create validation request
    print("\n3️⃣ Creating validation request due to low quality score...")
    
    val_state = create_validation_request(
        state_id="demo_extraction_001",
        extraction=extraction,
        quality_report=quality_report,
        pdf_path="/demo/climate_ai_paper.pdf"
    )
    
    await validation_store.save(val_state)
    
    # Step 4: Show validation stats
    print("\n4️⃣ Current validation statistics:")
    stats = await get_validation_stats()
    print(f"   • Pending validations: {stats.pending_validations}")
    print(f"   • Total validations: {stats.total_validations}")
    print(f"   • Approved: {stats.approved}")
    print(f"   • Rejected: {stats.rejected}")
    print(f"   • Needs revision: {stats.needs_revision}")
    
    # Step 5: Provide instructions
    print("\n5️⃣ Human validation required!")
    print(f"\n📋 Validation URL: http://localhost:8001/validator/{val_state.id}")
    print("\n✅ What you can do in the validation UI:")
    print("   • Review the extraction quality report")
    print("   • See highlighted weaknesses (low grounding, missing relationships)")
    print("   • Click on individual items to mark for review")
    print("   • Adjust confidence scores")
    print("   • Provide structured feedback")
    print("   • Approve, reject, or request revision")
    
    print("\n🚀 Next steps:")
    print("   1. Visit the validation dashboard: http://localhost:8001/validator/dashboard")
    print("   2. Click on the pending validation to review")
    print("   3. Provide feedback and submit your assessment")
    print("   4. The system will update the extraction based on your feedback")
    
    # Optionally open browser
    try:
        print("\n🌐 Opening validation UI in browser...")
        webbrowser.open(f"http://localhost:8001/validator/{val_state.id}")
    except:
        print("   (Could not open browser automatically)")
    
    return val_state.id

async def check_validation_result(validation_id: str):
    """Check if validation has been completed."""
    print(f"\n📊 Checking validation status for {validation_id}...")
    
    state = await validation_store.load(validation_id)
    if not state:
        print("❌ Validation not found")
        return
    
    print(f"   Status: {state.status}")
    
    if state.metadata.get("validation_result"):
        result = state.metadata["validation_result"]
        print(f"\n✅ Validation completed!")
        print(f"   Assessment: {result['overall_assessment']}")
        print(f"   Confidence adjustment: {result['confidence_adjustment']}x")
        if result.get('comments'):
            print(f"   Comments: {result['comments']}")
        if result.get('suggested_improvements'):
            print(f"   Improvements: {', '.join(result['suggested_improvements'])}")
    else:
        print("   ⏳ Validation still pending...")

if __name__ == "__main__":
    print("🚀 Starting Validation UI Workflow Demo")
    print("\n⚠️  Make sure the validation UI server is running:")
    print("   python -m extraction.v2.validation_ui")
    
    validation_id = asyncio.run(demo_validation_workflow())
    
    print("\n" + "="*60)
    print("Demo complete! The validation UI is ready for human review.")
    print("="*60)
    
    # Optionally check result later
    # await asyncio.sleep(30)
    # await check_validation_result(validation_id)