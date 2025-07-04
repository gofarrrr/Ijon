"""
Test the knowledge extraction service.
"""

import asyncio
import json
from extraction.v2.knowledge_service import (
    extract_knowledge,
    analyze_extraction_quality,
    get_extraction_examples,
    get_extraction_insights,
    check_extraction_status,
    get_extraction_guidelines
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def test_knowledge_service():
    """Test all knowledge service endpoints."""
    
    print("\n" + "="*60)
    print("KNOWLEDGE EXTRACTION SERVICE TEST")
    print("="*60)
    
    # Test 1: Extract knowledge
    print("\n1️⃣ Testing extract_knowledge...")
    test_content = """
    Machine learning is revolutionizing healthcare through predictive analytics.
    Deep learning models can analyze medical images with 95% accuracy, 
    surpassing human radiologists in detecting certain conditions like pneumonia.
    Early detection through AI reduces treatment costs by up to 40% according 
    to recent studies. Natural language processing enables automated analysis
    of clinical notes, improving diagnostic accuracy.
    """
    
    result = await extract_knowledge(
        content=test_content,
        doc_type="technical",
        apply_citation_enhancer=True,
        apply_relationship_enhancer=True
    )
    
    print(f"   Status: {result['status']}")
    if result['status'] == 'success':
        print(f"   Quality Score: {result['quality_score']:.2f}")
        print(f"   Model Used: {result['model_used']}")
        extraction = result['extraction']
        print(f"   Topics Found: {len(extraction['topics'])}")
        for topic in extraction['topics']:
            print(f"     - {topic['name']} (confidence: {topic['confidence']:.2f})")
        print(f"   Facts Found: {len(extraction['facts'])}")
        for fact in extraction['facts'][:2]:
            print(f"     - {fact['claim'][:60]}...")
        print(f"   Enhancers Applied: {result['enhancers_applied']}")
    
    # Test 2: Get extraction insights
    print("\n2️⃣ Testing get_extraction_insights...")
    
    for topic in ["machine learning", "healthcare", "blockchain"]:
        insights = get_extraction_insights(topic)
        print(f"\n   Insights for '{topic}':")
        print(f"   Matched Domain: {insights.get('matched_domain', 'general')}")
        print(f"   Recommended Doc Type: {insights['recommended_doc_type']}")
        print("   Top Tips:")
        for tip in insights['tips'][:3]:
            print(f"     - {tip}")
        print(f"   Recommended Enhancers: {', '.join(insights['recommended_enhancers'])}")
    
    # Test 3: Get extraction examples
    print("\n3️⃣ Testing get_extraction_examples...")
    examples = get_extraction_examples("all")
    
    print(f"   Found {len(examples)} examples")
    for example in examples:
        print(f"\n   📚 {example['doc_type'].upper()} Example:")
        print(f"   Quality Score: {example['example']['quality_score']}")
        print(f"   Explanation: {example['explanation']}")
        ex_data = example['example']
        print(f"   Structure: {len(ex_data['topics'])} topics, {len(ex_data['facts'])} facts")
        if 'questions' in ex_data:
            print(f"   Questions: {len(ex_data.get('questions', []))}")
        if 'relationships' in ex_data:
            print(f"   Relationships: {len(ex_data.get('relationships', []))}")
    
    # Test 4: Analyze extraction quality
    print("\n4️⃣ Testing analyze_extraction_quality...")
    
    # Create a sample extraction with some quality issues
    test_extraction = {
        "topics": [
            {"name": "AI Healthcare", "description": "AI applications in medicine", "confidence": 0.8},
            {"name": "Cost Reduction", "description": "Economic benefits", "confidence": 0.7}
        ],
        "facts": [
            {"claim": "AI reduces costs", "confidence": 0.5},  # Vague, no evidence
            {"claim": "Deep learning achieves 95% accuracy", "evidence": "Studies show high accuracy", "confidence": 0.85},
            {"claim": "NLP improves diagnostics", "confidence": 0.6}  # No specific evidence
        ],
        "questions": [],
        "relationships": [],  # Missing relationships between topics
        "overall_confidence": 0.65
    }
    
    quality_result = await analyze_extraction_quality(
        extraction_data=test_extraction,
        source_content=test_content
    )
    
    if quality_result['status'] == 'success':
        print(f"   Overall Score: {quality_result['overall_score']:.2f}")
        print("   Dimension Scores:")
        for dim, score in quality_result['dimensions'].items():
            print(f"     - {dim}: {score:.2f}")
        print("   Weaknesses Found:")
        for weakness in quality_result['weaknesses']:
            details = weakness.get('details', f"Score: {weakness.get('score', 'N/A')}")
            print(f"     - {weakness['dimension']}: {details}")
        print("   Recommendations:")
        for i, rec in enumerate(quality_result['recommendations'], 1):
            print(f"     {i}. {rec}")
    
    # Test 5: Check extraction status
    print("\n5️⃣ Testing check_extraction_status...")
    
    # First create a mock extraction state
    from extraction.v2.state import ExtractionState
    from extraction.v2.knowledge_service import state_store
    mock_state = ExtractionState.create_new("test.pdf", "test-123")
    mock_state.status = "pending_validation"
    mock_state.quality_report = {"overall_score": 0.55}
    mock_state.metadata["needs_validation"] = True
    await state_store.save(mock_state)
    
    status_result = await check_extraction_status("test-123")
    print(f"   Status: {status_result['status']}")
    if status_result['status'] == 'success':
        print(f"   State: {status_result['state']}")
        print(f"   Quality Score: {status_result['quality_score']}")
        print(f"   Needs Validation: {status_result['needs_validation']}")
        if status_result.get('validation_url'):
            print(f"   Validation URL: {status_result['validation_url']}")
    
    # Test 6: Get guidelines
    print("\n6️⃣ Testing get_extraction_guidelines...")
    guidelines = get_extraction_guidelines()
    print(f"   Guidelines Length: {len(guidelines)} characters")
    print(f"   Sections: {guidelines.count('##')} main sections")
    print("   First 200 chars:")
    print(f"   {guidelines[:200]}...")
    
    print("\n✅ All service tests completed!")


async def demonstrate_agent_usage():
    """Show how other agents would use this service."""
    
    print("\n" + "="*60)
    print("HOW AGENTS USE THE KNOWLEDGE SERVICE")
    print("="*60)
    
    print("\n📋 Example Agent Workflows:")
    
    print("\n1. Research Agent extracting from a paper:")
    print("""
    # Agent reads a research paper
    paper_content = read_pdf("research_paper.pdf")
    
    # Extract knowledge with high quality requirements
    result = await extract_knowledge(
        content=paper_content,
        doc_type="academic",
        apply_citation_enhancer=True,
        apply_relationship_enhancer=True
    )
    
    # Check quality
    if result['quality_score'] < 0.8:
        # Get recommendations
        quality = await analyze_extraction_quality(
            result['extraction'], 
            paper_content
        )
        # Apply improvements based on recommendations
    """)
    
    print("\n2. Learning Agent improving extraction skills:")
    print("""
    # Get insights for a domain
    insights = get_extraction_insights("quantum computing")
    
    # Study high-quality examples
    examples = get_extraction_examples("academic")
    
    # Learn patterns and best practices
    for example in examples:
        study_extraction_patterns(example)
    """)
    
    print("\n3. Quality Assurance Agent:")
    print("""
    # Monitor extraction quality
    extraction_id = "job-456"
    status = await check_extraction_status(extraction_id)
    
    if status['needs_validation']:
        # Route to human validator
        notify_validator(status['validation_url'])
    """)
    
    print("\n4. Documentation Agent:")
    print("""
    # Get comprehensive guidelines
    guidelines = get_extraction_guidelines()
    
    # Create training materials
    create_agent_training_doc(guidelines)
    
    # Share best practices with other agents
    broadcast_to_agent_network(guidelines)
    """)


if __name__ == "__main__":
    print("🚀 Testing Knowledge Extraction Service")
    
    asyncio.run(test_knowledge_service())
    asyncio.run(demonstrate_agent_usage())
    
    print("\n" + "="*60)
    print("SERVICE READY FOR AGENT EDUCATION")
    print("="*60)
    print("\n✨ The Knowledge Extraction Service provides:")
    print("   • Stateless extraction functions")
    print("   • Quality analysis and recommendations")
    print("   • Domain-specific insights")
    print("   • High-quality examples to learn from")
    print("   • Comprehensive guidelines")
    print("\n🤖 Other agents can now:")
    print("   • Extract knowledge from any text")
    print("   • Analyze and improve extraction quality")
    print("   • Learn best practices for different domains")
    print("   • Access extraction status and validation needs")