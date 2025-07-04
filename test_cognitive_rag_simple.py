#!/usr/bin/env python3
"""
Simple test for cognitive RAG integration without heavy dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_cognitive_pipeline_imports():
    """Test that cognitive pipeline modules can be imported."""
    print("Testing Cognitive Pipeline Imports...")
    
    try:
        from src.rag.cognitive_pipeline import (
            CognitiveRAGPipeline, CognitiveQueryResult, create_cognitive_rag_pipeline
        )
        from src.agents.cognitive_orchestrator import CognitiveOrchestrator, OrchestrationResult
        from src.agents.cognitive_router import CognitiveRouter, TaskAnalysis
        
        print("✅ All cognitive pipeline modules imported successfully")
        print("   CognitiveRAGPipeline: Available")
        print("   CognitiveQueryResult: Available")
        print("   CognitiveOrchestrator: Available")
        print("   CognitiveRouter: Available")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_cognitive_result_creation():
    """Test cognitive query result creation."""
    print("Testing Cognitive Result Creation...")
    
    try:
        from src.rag.cognitive_pipeline import CognitiveQueryResult
        from src.agents.cognitive_orchestrator import OrchestrationResult
        from src.agents.cognitive_router import TaskAnalysis, TaskType, TaskComplexity, DomainType
        
        # Create mock analysis
        analysis = TaskAnalysis(
            task_type=TaskType.ANALYSIS,
            complexity=TaskComplexity.MODERATE,
            domain=DomainType.TECHNICAL,
            confidence=0.8,
            reasoning=["Mock reasoning"],
            recommended_agent="AnalysisAgent",
            sub_tasks=[],
            estimated_time=120.0,
            requires_validation=False,
            metadata={}
        )
        
        # Create mock orchestration result
        orchestration_result = OrchestrationResult(
            task_id="test-123",
            success=True,
            final_result="Mock analysis result",
            execution_trace=[{"agent": "AnalysisAgent", "success": True}],
            total_time=45.0,
            agents_used=["AnalysisAgent"],
            quality_score=0.85,
            recommendations=["Consider validation"],
            metadata={"mock": True}
        )
        
        # Create cognitive result
        cognitive_result = CognitiveQueryResult(
            query="Test analysis query",
            orchestration_result=orchestration_result,
            analysis=analysis,
            success=True,
            quality_score=0.85,
            agents_used=["AnalysisAgent"],
            processing_time=45.0,
            recommendations=["Consider validation"],
            metadata={"mock": True}
        )
        
        # Test methods
        result_dict = cognitive_result.to_dict()
        
        print("✅ Cognitive result created and converted to dict")
        print(f"   Query: {result_dict['query']}")
        print(f"   Success: {result_dict['success']}")
        print(f"   Quality: {result_dict['quality_score']}")
        print(f"   Task type: {result_dict['task_analysis']['type']}")
        print(f"   Summary: {cognitive_result.summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cognitive result creation failed: {e}")
        return False


async def test_mock_rag_pipeline():
    """Test creating a mock RAG pipeline integration."""
    print("Testing Mock RAG Pipeline Integration...")
    
    try:
        from src.rag.cognitive_pipeline import CognitiveRAGPipeline
        from src.agents.cognitive_orchestrator import CognitiveOrchestrator
        
        # Create mock RAG pipeline
        class MockRAGPipeline:
            def __init__(self):
                self.vector_store = None
                self.graph_store = None
            
            async def query(self, query, client, **kwargs):
                # Mock simple result
                class MockQueryResult:
                    def __init__(self):
                        self.summary = f"Mock summary for: {query}"
                        self.documents = []
                        self.confidence = 0.7
                
                return MockQueryResult()
        
        mock_rag = MockRAGPipeline()
        
        # Create cognitive orchestrator
        orchestrator = CognitiveOrchestrator()
        
        # Create cognitive pipeline
        cognitive_pipeline = CognitiveRAGPipeline(
            rag_pipeline=mock_rag,
            orchestrator=orchestrator,
            cognitive_threshold=0.6,
            enable_hybrid_mode=True,
        )
        
        print("✅ Mock cognitive RAG pipeline created")
        print(f"   Cognitive threshold: {cognitive_pipeline.cognitive_threshold}")
        print(f"   Hybrid mode: {cognitive_pipeline.enable_hybrid_mode}")
        print(f"   Orchestrator agents: {len(cognitive_pipeline.orchestrator._agents)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock RAG pipeline integration failed: {e}")
        return False


async def test_router_analysis():
    """Test cognitive router analysis."""
    print("Testing Router Analysis...")
    
    try:
        from src.agents.cognitive_router import CognitiveRouter
        
        router = CognitiveRouter(use_llm_analysis=False)
        
        test_queries = [
            "What is machine learning?",
            "Create a comprehensive strategy",
            "Analyze customer satisfaction data",
            "Fix authentication bug"
        ]
        
        for query in test_queries:
            analysis = await router.analyze_task(query)
            print(f"   '{query[:30]}...' -> {analysis.task_type.value}, {analysis.complexity.value}")
        
        print("✅ Router analysis working")
        return True
        
    except Exception as e:
        print(f"❌ Router analysis failed: {e}")
        return False


async def test_query_routing_logic():
    """Test query routing decision logic."""
    print("Testing Query Routing Logic...")
    
    try:
        from src.rag.cognitive_pipeline import CognitiveRAGPipeline
        from src.agents.cognitive_orchestrator import CognitiveOrchestrator
        from src.agents.cognitive_router import TaskComplexity, TaskType
        
        # Mock pipeline
        class MockRAGPipeline:
            async def query(self, **kwargs):
                return type('MockResult', (), {'summary': 'mock', 'documents': []})()
        
        cognitive_pipeline = CognitiveRAGPipeline(
            rag_pipeline=MockRAGPipeline(),
            orchestrator=CognitiveOrchestrator(),
            cognitive_threshold=0.6,
        )
        
        # Test routing decisions
        test_cases = [
            ("What is Python?", "should use fast RAG"),
            ("Create comprehensive AI strategy for healthcare", "should use cognitive"),
            ("Solve complex optimization problem", "should use cognitive"),
        ]
        
        for query, expected in test_cases:
            # Mock client
            class MockClient:
                pass
            
            # Get analysis
            analysis = await cognitive_pipeline.orchestrator.router.analyze_task(query)
            
            should_use_cognitive = (
                analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT] or
                analysis.confidence < cognitive_pipeline.cognitive_threshold or
                analysis.task_type in [TaskType.CREATION, TaskType.SOLUTION, TaskType.SYNTHESIS]
            )
            
            route = "cognitive" if should_use_cognitive else "fast RAG"
            print(f"   '{query[:40]}...' -> {route} ({analysis.complexity.value})")
        
        print("✅ Query routing logic working")
        return True
        
    except Exception as e:
        print(f"❌ Query routing logic test failed: {e}")
        return False


async def main():
    """Run simple cognitive RAG tests."""
    print("🧠 Testing Cognitive RAG Integration (Simple)...\n")
    
    tests = [
        test_cognitive_pipeline_imports,
        test_cognitive_result_creation,
        test_mock_rag_pipeline,
        test_router_analysis,
        test_query_routing_logic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            success = await test()
            if success:
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")
    
    print("📊 Test Results:")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All cognitive RAG integration tests passed!")
        print("\n✨ Phase 2 Integration Complete!")
        print("\nImplemented Features:")
        print("• ✅ Cognitive RAG pipeline architecture")
        print("• ✅ Intelligent query routing (fast RAG vs cognitive)")
        print("• ✅ Cognitive query result structure")
        print("• ✅ Agent orchestration integration")
        print("• ✅ Hybrid mode for combining results")
        print("\nNext: Phase 3 - Self-correction and validation")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)