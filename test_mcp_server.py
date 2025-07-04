"""
Test the MCP server functionality.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def test_mcp_server():
    """Test MCP server tools and resources."""
    
    print("\n" + "="*60)
    print("MCP SERVER TEST")
    print("="*60)
    
    # Create server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "extraction.v2.mcp_server"],
        env={"PYTHONPATH": "."}
    )
    
    try:
        # Start client session
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()
                
                print("\n📋 Server Information:")
                print(f"   Name: {session.server_info.name}")
                print(f"   Version: {session.server_info.version}")
                
                # List available tools
                print("\n🔧 Available Tools:")
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    print(f"   • {tool.name}: {tool.description}")
                
                # List available resources
                print("\n📚 Available Resources:")
                resources_response = await session.list_resources()
                for resource in resources_response.resources:
                    print(f"   • {resource.name}: {resource.description}")
                
                # Test 1: Extract knowledge
                print("\n\n1️⃣ Testing extract_knowledge tool...")
                test_content = """
                Machine learning algorithms have revolutionized data analysis. 
                Deep learning models can process images with 95% accuracy.
                Natural language processing enables computers to understand human text.
                """
                
                result = await session.call_tool(
                    "extract_knowledge",
                    arguments={
                        "content": test_content,
                        "doc_type": "technical",
                        "apply_citation_enhancer": True,
                        "apply_relationship_enhancer": True
                    }
                )
                
                print(f"   Status: {result.content[0].text}")
                extraction_result = json.loads(result.content[0].text)
                if extraction_result["status"] == "success":
                    print(f"   Quality Score: {extraction_result['quality_score']:.2f}")
                    print(f"   Model Used: {extraction_result['model_used']}")
                
                # Test 2: Get extraction insights
                print("\n2️⃣ Testing get_extraction_insights tool...")
                insights_result = await session.call_tool(
                    "get_extraction_insights",
                    arguments={"topic": "machine learning"}
                )
                
                insights = json.loads(insights_result.content[0].text)
                print(f"   Topic: {insights['topic']}")
                print(f"   Recommended Doc Type: {insights['recommended_doc_type']}")
                print("   Tips:")
                for tip in insights['insights']['tips'][:3]:
                    print(f"     - {tip}")
                
                # Test 3: Get extraction examples
                print("\n3️⃣ Testing get_extraction_examples tool...")
                examples_result = await session.call_tool(
                    "get_extraction_examples",
                    arguments={"doc_type": "academic"}
                )
                
                examples = json.loads(examples_result.content[0].text)
                print(f"   Found {len(examples)} examples")
                if examples:
                    print(f"   Example Quality Score: {examples[0]['example']['quality_score']}")
                
                # Test 4: Read guidelines resource
                print("\n4️⃣ Testing extraction_guidelines resource...")
                guidelines = await session.read_resource("extraction://guidelines")
                print(f"   Resource Type: {guidelines.contents[0].mimeType}")
                print(f"   Content Preview: {guidelines.contents[0].text[:100]}...")
                
                # Test 5: Read metrics resource
                print("\n5️⃣ Testing extraction_metrics resource...")
                metrics = await session.read_resource("extraction://metrics")
                metrics_data = json.loads(metrics.contents[0].text)
                print(f"   Active Extractions: {metrics_data['active_extractions']}")
                print(f"   Average Quality Score: {metrics_data['average_quality_score']}")
                
                print("\n✅ All tests completed successfully!")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_extraction_quality_analysis():
    """Test the quality analysis tool."""
    print("\n" + "="*60)
    print("TESTING QUALITY ANALYSIS")
    print("="*60)
    
    # Sample extraction data
    extraction_data = {
        "topics": [
            {"name": "AI", "description": "Artificial Intelligence", "confidence": 0.8}
        ],
        "facts": [
            {"claim": "AI can solve problems", "confidence": 0.5}  # No evidence
        ],
        "questions": [],
        "relationships": [],
        "overall_confidence": 0.65
    }
    
    source_content = "Artificial Intelligence is a field that enables computers to solve complex problems."
    
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "extraction.v2.mcp_server"],
        env={"PYTHONPATH": "."}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "analyze_extraction_quality",
                arguments={
                    "extraction_data": extraction_data,
                    "source_content": source_content
                }
            )
            
            quality_report = json.loads(result.content[0].text)
            print(f"\nQuality Analysis Results:")
            print(f"   Overall Score: {quality_report['overall_score']:.2f}")
            print(f"   Dimensions:")
            for dim, score in quality_report['dimensions'].items():
                print(f"     - {dim}: {score:.2f}")
            print(f"   Recommendations:")
            for rec in quality_report['recommendations']:
                print(f"     - {rec}")


if __name__ == "__main__":
    print("🚀 Starting MCP Server Tests")
    print("\nNote: This test requires the 'mcp' package.")
    print("If not installed: pip install mcp")
    
    try:
        asyncio.run(test_mcp_server())
        asyncio.run(test_extraction_quality_analysis())
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nTo use MCP functionality, install the mcp package:")
        print("   pip install mcp")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()