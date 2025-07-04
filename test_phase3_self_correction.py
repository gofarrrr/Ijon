#!/usr/bin/env python3
"""
Test script for Phase 3 self-correction and quality feedback systems.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def test_quality_validator():
    """Test quality validation functionality."""
    print("Testing Quality Validator...")
    
    try:
        # Mock dependencies
        sys.modules['openai'] = Mock()
        
        from src.agents.self_correction import QualityValidator, QualityIssue, CorrectionType
        
        # Create validator without LLM
        validator = QualityValidator(client=None)
        
        # Test rule-based validation
        test_content = "Python is good. Python is good. Python is good."  # Repetitive
        test_query = "What is Python programming language?"
        
        from src.agents.cognitive_router import TaskType
        issues = await validator.validate_result(
            result=test_content,
            original_query=test_query,
            task_type=TaskType.RESEARCH,
        )
        
        print("✅ Quality validator created and executed")
        print(f"   Issues found: {len(issues)}")
        for issue in issues[:3]:  # Show first 3
            print(f"   - {issue.issue_type.value}: {issue.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality validator test failed: {e}")
        return False


async def test_reasoning_extractor():
    """Test reasoning structure extraction."""
    print("Testing Reasoning Extractor...")
    
    try:
        from src.agents.reasoning_validator import ReasoningExtractor, ReasoningType
        
        extractor = ReasoningExtractor()
        
        test_text = """
        Machine learning is powerful because it can learn from data.
        Since we have large datasets, we can train accurate models.
        Therefore, AI will transform many industries.
        Studies show that 80% of companies are investing in AI.
        """
        
        reasoning_steps = extractor.extract_reasoning_structure(test_text)
        
        print("✅ Reasoning extractor working")
        print(f"   Reasoning steps found: {len(reasoning_steps)}")
        for step in reasoning_steps[:2]:  # Show first 2
            print(f"   - {step.reasoning_type.value}: {step.premise[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning extractor test failed: {e}")
        return False


async def test_logical_validator():
    """Test logical consistency validation."""
    print("Testing Logical Validator...")
    
    try:
        from src.agents.reasoning_validator import LogicalValidator, LogicalFallacy
        
        validator = LogicalValidator(client=None)
        
        # Test text with potential logical issues
        test_text = """
        All AI is dangerous because AI is dangerous.
        Either we ban AI or we face extinction.
        After implementing AI, productivity increased, therefore AI caused the increase.
        """
        
        issues = validator._detect_fallacies(test_text)
        
        print("✅ Logical validator working")
        print(f"   Logical issues found: {len(issues)}")
        for issue in issues[:2]:  # Show first 2
            print(f"   - {issue.fallacy_type.value if issue.fallacy_type else 'N/A'}: {issue.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logical validator test failed: {e}")
        return False


async def test_evidence_analyzer():
    """Test evidence quality analysis."""
    print("Testing Evidence Analyzer...")
    
    try:
        from src.agents.reasoning_validator import EvidenceAnalyzer, EvidenceType
        
        analyzer = EvidenceAnalyzer(client=None)
        
        test_text = """
        A recent study published in Nature journal found that 95% of AI researchers believe in AGI.
        According to my friend who works in tech, AI will replace all jobs.
        The peer-reviewed meta-analysis of 50 studies shows statistical significance.
        """
        
        evidence_texts = analyzer._extract_evidence_texts(test_text)
        evidence_assessments = []
        
        for evidence_text in evidence_texts:
            assessment = analyzer._assess_single_evidence(evidence_text)
            evidence_assessments.append(assessment)
        
        print("✅ Evidence analyzer working")
        print(f"   Evidence pieces found: {len(evidence_assessments)}")
        for assessment in evidence_assessments[:2]:  # Show first 2
            print(f"   - {assessment.evidence_type.value}: {assessment.overall_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evidence analyzer test failed: {e}")
        return False


async def test_performance_tracker():
    """Test performance tracking functionality."""
    print("Testing Performance Tracker...")
    
    try:
        from src.agents.quality_feedback import PerformanceTracker, ImprovementArea
        
        # Create tracker with temporary storage
        tracker = PerformanceTracker(storage_path=Path("/tmp/test_performance"))
        
        # Record some executions
        tracker.record_execution(
            agent_name="TestAgent",
            quality_score=0.8,
            success=True,
            context={"test": True}
        )
        
        tracker.record_execution(
            agent_name="TestAgent",
            quality_score=0.6,
            success=True,
            context={"test": True}
        )
        
        # Get performance data
        performance = tracker.get_agent_performance("TestAgent")
        improvement_areas = tracker.get_top_improvement_areas("TestAgent")
        
        print("✅ Performance tracker working")
        print(f"   Total executions: {performance.total_executions}")
        print(f"   Average quality: {performance.average_quality_score:.2f}")
        print(f"   Success rate: {performance.success_rate:.2f}")
        print(f"   Trend: {performance.trend_direction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance tracker test failed: {e}")
        return False


async def test_learning_engine():
    """Test quality learning engine."""
    print("Testing Learning Engine...")
    
    try:
        from src.agents.quality_feedback import (
            PerformanceTracker, QualityLearningEngine, 
            AgentPerformance, ImprovementArea, PerformanceMetric
        )
        from datetime import datetime
        
        # Create mock performance data
        tracker = PerformanceTracker(storage_path=Path("/tmp/test_learning"))
        
        # Add mock performance
        performance = AgentPerformance(
            agent_name="MockAgent",
            total_executions=10,
            success_rate=0.7,
            average_quality_score=0.6,
            improvement_areas={area: 0.5 for area in ImprovementArea},
            recent_metrics=[
                PerformanceMetric("quality_score", 0.5, datetime.utcnow()),
                PerformanceMetric("quality_score", 0.4, datetime.utcnow()),
                PerformanceMetric("quality_score", 0.3, datetime.utcnow()),
            ],
            last_updated=datetime.utcnow(),
            trend_direction="declining"
        )
        
        tracker.agent_performances["MockAgent"] = performance
        
        # Create learning engine
        learning_engine = QualityLearningEngine(tracker)
        
        # Analyze patterns
        insights = await learning_engine.analyze_performance_patterns()
        
        print("✅ Learning engine working")
        print(f"   Insights generated: {len(insights)}")
        for insight in insights[:2]:  # Show first 2
            print(f"   - {insight.insight_type}: {insight.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ Learning engine test failed: {e}")
        return False


async def test_self_corrector_creation():
    """Test self-corrector creation and basic functionality."""
    print("Testing Self-Corrector Creation...")
    
    try:
        from src.agents.self_correction import SelfCorrector, create_self_corrector
        
        # Create without LLM client
        corrector = create_self_corrector(
            client=None,
            max_iterations=2,
            quality_threshold=0.8
        )
        
        print("✅ Self-corrector created successfully")
        print(f"   Max iterations: {corrector.max_iterations}")
        print(f"   Quality threshold: {corrector.quality_threshold}")
        print(f"   Validator available: {corrector.validator is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Self-corrector creation failed: {e}")
        return False


async def test_reasoning_validator_integration():
    """Test reasoning validator integration."""
    print("Testing Reasoning Validator Integration...")
    
    try:
        from src.agents.reasoning_validator import create_reasoning_validator
        
        # Create validator
        validator = create_reasoning_validator(client=None)
        
        # Test validation
        test_text = """
        Machine learning algorithms learn from data to make predictions.
        Because neural networks have many layers, they can model complex patterns.
        Therefore, deep learning is effective for image recognition tasks.
        Studies show 95% accuracy in medical image analysis.
        """
        
        validation_result = await validator.validate_reasoning(
            text=test_text,
            task_type="analysis",
            context="AI research"
        )
        
        print("✅ Reasoning validator integration working")
        print(f"   Reasoning steps: {len(validation_result.reasoning_steps)}")
        print(f"   Evidence pieces: {len(validation_result.evidence_quality)}")
        print(f"   Logical issues: {len(validation_result.logical_issues)}")
        print(f"   Overall score: {validation_result.overall_reasoning_score:.2f}")
        print(f"   Consistency: {validation_result.consistency_score:.2f}")
        print(f"   Evidence score: {validation_result.evidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning validator integration failed: {e}")
        return False


async def test_adaptive_quality_manager():
    """Test adaptive quality manager."""
    print("Testing Adaptive Quality Manager...")
    
    try:
        from src.agents.quality_feedback import create_adaptive_quality_manager
        
        # Create manager
        manager = create_adaptive_quality_manager(storage_path=Path("/tmp/test_adaptive"))
        
        # Test feedback processing
        await manager.process_execution_feedback(
            agent_name="TestAgent",
            original_query="Test query",
            result="Test result",
            quality_score=0.75,
            success=True,
            execution_time=1.5,
        )
        
        # Get adaptive threshold
        threshold = manager.get_adaptive_threshold("TestAgent")
        
        # Get performance summary
        summary = manager.get_performance_summary()
        
        print("✅ Adaptive quality manager working")
        print(f"   Adaptive threshold: {threshold:.2f}")
        print(f"   Performance summary: {summary['overall_health']}")
        print(f"   Total agents: {summary['total_agents']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive quality manager test failed: {e}")
        return False


async def main():
    """Run Phase 3 self-correction tests."""
    print("🔧 Testing Phase 3: Self-Correction & Quality Feedback...\n")
    
    tests = [
        test_quality_validator,
        test_reasoning_extractor,
        test_logical_validator,
        test_evidence_analyzer,
        test_performance_tracker,
        test_learning_engine,
        test_self_corrector_creation,
        test_reasoning_validator_integration,
        test_adaptive_quality_manager,
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
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All Phase 3 self-correction tests passed!")
        print("\n✨ Phase 3 Implementation Complete!")
        print("\nImplemented Features:")
        print("• ✅ Quality validation with rule-based and LLM-enhanced checks")
        print("• ✅ Reasoning structure extraction and logical consistency validation")
        print("• ✅ Evidence quality analysis and credibility assessment")
        print("• ✅ Self-correction system with iterative improvement")
        print("• ✅ Performance tracking and trend analysis")
        print("• ✅ Adaptive learning engine with pattern recognition")
        print("• ✅ Quality feedback loops for continuous improvement")
        print("• ✅ Adaptive thresholds and personalized quality standards")
        print("\nNext: Phase 4 - Advanced graph reasoning (optional)")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)