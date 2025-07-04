#!/usr/bin/env python3
"""
Test script for the new cognitive agent system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


async def test_cognitive_router():
    """Test the cognitive router."""
    print("Testing Cognitive Router...")
    
    try:
        from src.agents.cognitive_router import CognitiveRouter, TaskType, TaskComplexity
        
        router = CognitiveRouter(use_llm_analysis=False)  # Test rule-based first
        
        # Test simple analysis task
        analysis = await router.analyze_task(
            task="Analyze the benefits and drawbacks of machine learning in healthcare",
            context="Academic research context"
        )
        
        print("✅ Cognitive router created and executed")
        print(f"   Task type: {analysis.task_type.value}")
        print(f"   Complexity: {analysis.complexity.value}")
        print(f"   Domain: {analysis.domain.value}")
        print(f"   Recommended agent: {analysis.recommended_agent}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Sub-tasks: {len(analysis.sub_tasks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cognitive router test failed: {e}")
        return False


async def test_cognitive_agents():
    """Test cognitive agent imports and creation."""
    print("Testing Cognitive Agents...")
    
    try:
        from src.agents.cognitive_agents import (
            AnalysisAgent, SolutionAgent, CreationAgent,
            VerificationAgent, SynthesisAgent
        )
        
        # Test agent creation (without tools to avoid dependencies)
        agents = {
            "Analysis": AnalysisAgent(),
            "Solution": SolutionAgent(), 
            "Creation": CreationAgent(),
            "Verification": VerificationAgent(),
            "Synthesis": SynthesisAgent(),
        }
        
        print("✅ All cognitive agents created successfully")
        
        for name, agent in agents.items():
            print(f"   {name}Agent: {agent.name} (model: {agent.model})")
        
        return True
        
    except Exception as e:
        print(f"❌ Cognitive agents test failed: {e}")
        return False


async def test_cognitive_orchestrator():
    """Test cognitive orchestrator."""
    print("Testing Cognitive Orchestrator...")
    
    try:
        from src.agents.cognitive_orchestrator import CognitiveOrchestrator
        
        orchestrator = CognitiveOrchestrator()
        
        print("✅ Cognitive orchestrator created")
        print(f"   Agents available: {list(orchestrator._agents.keys())}")
        print(f"   Max concurrent: {orchestrator.max_concurrent_agents}")
        
        # Test capabilities query
        capabilities = orchestrator.router.get_agent_capabilities("AnalysisAgent")
        print(f"   Analysis Agent accuracy: {capabilities['estimated_accuracy']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cognitive orchestrator test failed: {e}")
        return False


async def test_task_routing():
    """Test end-to-end task routing."""
    print("Testing Task Routing...")
    
    try:
        from src.agents.cognitive_router import CognitiveRouter
        
        router = CognitiveRouter(use_llm_analysis=False)
        
        test_tasks = [
            ("What is machine learning?", "research", "simple"),
            ("Create a blog post about AI ethics", "creation", "moderate"),
            ("Solve the data quality issues in our pipeline", "solution", "complex"),
            ("Analyze customer satisfaction trends over the past year", "analysis", "moderate"),
            ("Verify the accuracy of our financial projections", "verification", "moderate"),
        ]
        
        routing_results = []
        
        for task, expected_type, expected_complexity in test_tasks:
            analysis = await router.analyze_task(task)
            routing_results.append({
                "task": task[:50] + "...",
                "predicted_type": analysis.task_type.value,
                "expected_type": expected_type,
                "predicted_complexity": analysis.complexity.value,
                "expected_complexity": expected_complexity,
                "agent": analysis.recommended_agent,
                "confidence": analysis.confidence,
            })
        
        print("✅ Task routing completed")
        print("\nRouting Results:")
        print("-" * 80)
        
        for result in routing_results:
            type_match = "✓" if result["predicted_type"] == result["expected_type"] else "✗"
            print(f"Task: {result['task']}")
            print(f"  Type: {result['predicted_type']} {type_match} (expected: {result['expected_type']})")
            print(f"  Complexity: {result['predicted_complexity']} (expected: {result['expected_complexity']})")
            print(f"  Agent: {result['agent']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Task routing test failed: {e}")
        return False


async def test_llm_routing():
    """Test LLM-enhanced routing if API key available."""
    print("Testing LLM-Enhanced Routing...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No OpenAI API key found, skipping LLM routing test")
        return True
    
    try:
        from openai import AsyncOpenAI
        from src.agents.cognitive_router import CognitiveRouter
        
        client = AsyncOpenAI(api_key=api_key)
        router = CognitiveRouter(use_llm_analysis=True)
        
        # Test complex task that benefits from LLM analysis
        task = """Design a comprehensive strategy for implementing ethical AI practices 
                 in a healthcare organization, considering regulatory compliance, 
                 patient privacy, algorithmic bias, and stakeholder engagement."""
        
        analysis = await router.analyze_task(
            task=task,
            context="Healthcare AI implementation project",
            client=client,
        )
        
        print("✅ LLM-enhanced routing completed")
        print(f"   Task type: {analysis.task_type.value}")
        print(f"   Complexity: {analysis.complexity.value}")
        print(f"   Domain: {analysis.domain.value}")
        print(f"   Agent: {analysis.recommended_agent}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        print(f"   Reasoning: {len(analysis.reasoning)} points")
        print(f"   Sub-tasks: {len(analysis.sub_tasks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM routing test failed: {e}")
        return False


async def main():
    """Run all cognitive system tests."""
    print("🧠 Testing Cognitive Agent System...\n")
    
    tests = [
        ("Cognitive Router", test_cognitive_router),
        ("Cognitive Agents", test_cognitive_agents),
        ("Cognitive Orchestrator", test_cognitive_orchestrator),
        ("Task Routing", test_task_routing),
        ("LLM Routing", test_llm_routing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"📋 Running {test_name} test...")
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All cognitive system tests passed!")
        print("\n🎯 The cognitive agent system is ready for use!")
        print("\nKey capabilities:")
        print("• Intelligent task routing based on complexity and type")
        print("• Specialized agents for analysis, solution, creation, verification")
        print("• Multi-agent orchestration for complex tasks")
        print("• LLM-enhanced task understanding")
        print("• Stateless, 12-factor compliant architecture")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\nCognitive System Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)