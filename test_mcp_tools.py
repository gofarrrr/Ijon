"""
Test MCP server tools directly without full MCP client.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from extraction.v2.mcp_server import (
    extract_knowledge, 
    analyze_extraction_quality,
    get_extraction_examples,
    get_extraction_insights,
    check_extraction_status
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def test_mcp_tools():
    """Test MCP tools directly."""
    
    print("\n" + "="*60)
    print("MCP TOOLS DIRECT TEST")
    print("="*60)
    
    # Test 1: Extract knowledge
    print("\n1️⃣ Testing extract_knowledge...")
    test_content = """
    Machine learning is transforming healthcare through predictive analytics.
    Deep learning models can analyze medical images with 95% accuracy, 
    surpassing human radiologists in detecting certain conditions.
    Early detection through AI reduces treatment costs by up to 40%.
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
        print(f"   Topics Found: {len(result['extraction']['topics'])}")
        print(f"   Facts Found: {len(result['extraction']['facts'])}")
        print(f"   Enhancers Applied: {result['enhancers_applied']}")
    
    # Test 2: Get extraction insights
    print("\n2️⃣ Testing get_extraction_insights...")
    insights = await get_extraction_insights("machine learning")
    
    print(f"   Topic: {insights['topic']}")
    print(f"   Recommended Doc Type: {insights['recommended_doc_type']}")
    print("   Tips:")
    for tip in insights['insights']['tips'][:3]:
        print(f"     - {tip}")
    print("   Common Patterns:")
    for pattern in insights['insights']['common_patterns'][:3]:
        print(f"     - {pattern}")
    
    # Test 3: Get extraction examples
    print("\n3️⃣ Testing get_extraction_examples...")
    examples = await get_extraction_examples("academic")
    
    print(f"   Found {len(examples)} examples")
    for example in examples:
        print(f"\n   Document Type: {example['doc_type']}")
        print(f"   Quality Score: {example['example']['quality_score']}")
        print(f"   Explanation: {example['explanation']}")
        print(f"   Topics: {len(example['example']['topics'])}")
        print(f"   Facts: {len(example['example']['facts'])}")
    
    # Test 4: Analyze extraction quality
    print("\n4️⃣ Testing analyze_extraction_quality...")
    
    # Create a sample extraction with some quality issues
    extraction_data = {
        "topics": [
            {"name": "Healthcare AI", "description": "AI applications in medicine", "confidence": 0.8},
            {"name": "Cost Reduction", "description": "Economic benefits", "confidence": 0.7}
        ],
        "facts": [
            {"claim": "AI reduces costs", "confidence": 0.5},  # Vague, no evidence
            {"claim": "Deep learning achieves 95% accuracy", "evidence": "Study shows...", "confidence": 0.85}
        ],
        "questions": [],
        "relationships": [],
        "overall_confidence": 0.65
    }
    
    quality_result = await analyze_extraction_quality(
        extraction_data=extraction_data,
        source_content=test_content
    )
    
    if quality_result['status'] == 'success':
        print(f"   Overall Score: {quality_result['overall_score']:.2f}")
        print("   Dimension Scores:")
        for dim, score in quality_result['dimensions'].items():
            print(f"     - {dim}: {score:.2f}")
        print("   Weaknesses:")
        for weakness in quality_result['weaknesses'][:3]:
            print(f"     - {weakness['dimension']}: {weakness['details']}")
        print("   Recommendations:")
        for rec in quality_result['recommendations'][:3]:
            print(f"     - {rec}")
    
    # Test 5: Check extraction status (with mock ID)
    print("\n5️⃣ Testing check_extraction_status...")
    status_result = await check_extraction_status("test-extraction-123")
    
    print(f"   Status: {status_result['status']}")
    if status_result['status'] == 'not_found':
        print(f"   Message: {status_result['error']}")
    
    print("\n✅ All MCP tool tests completed!")


async def demonstrate_mcp_usage():
    """Demonstrate how other agents would use the MCP server."""
    
    print("\n" + "="*60)
    print("MCP USAGE DEMONSTRATION")
    print("="*60)
    
    print("\n📋 How other agents can use Ijon Extraction MCP:")
    
    print("\n1. Knowledge Extraction Agent:")
    print("""
    # Extract knowledge from a research paper
    result = await mcp_client.call_tool(
        "ijon-extraction.extract_knowledge",
        content=paper_text,
        doc_type="academic",
        apply_citation_enhancer=True
    )
    """)
    
    print("\n2. Quality Assurance Agent:")
    print("""
    # Check extraction quality
    quality = await mcp_client.call_tool(
        "ijon-extraction.analyze_extraction_quality",
        extraction_data=my_extraction,
        source_content=original_text
    )
    
    if quality['overall_score'] < 0.7:
        # Apply recommendations
        for rec in quality['recommendations']:
            apply_improvement(rec)
    """)
    
    print("\n3. Learning Agent:")
    print("""
    # Get insights for better extraction
    insights = await mcp_client.call_tool(
        "ijon-extraction.get_extraction_insights",
        topic="quantum computing"
    )
    
    # Learn from examples
    examples = await mcp_client.call_tool(
        "ijon-extraction.get_extraction_examples",
        doc_type="technical"
    )
    """)
    
    print("\n4. Monitoring Agent:")
    print("""
    # Check extraction status
    status = await mcp_client.call_tool(
        "ijon-extraction.check_extraction_status",
        extraction_id=job_id
    )
    
    # Read system metrics
    metrics = await mcp_client.read_resource(
        "extraction://metrics"
    )
    """)


if __name__ == "__main__":
    print("🚀 Testing MCP Server Tools Directly")
    
    asyncio.run(test_mcp_tools())
    asyncio.run(demonstrate_mcp_usage())
    
    print("\n" + "="*60)
    print("MCP SERVER READY FOR AGENT EDUCATION")
    print("="*60)
    print("\nThe MCP server provides:")
    print("✅ 5 tools for extraction and analysis")
    print("✅ 2 resources with guidelines and metrics")
    print("✅ Stateless, focused endpoints")
    print("✅ Clear examples and insights")
    print("\nOther agents can now learn from and use these extraction capabilities!")