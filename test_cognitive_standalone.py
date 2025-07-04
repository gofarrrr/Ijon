#!/usr/bin/env python3
"""
Standalone test for cognitive routing logic.
"""

import asyncio
import re
import sys
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass


class TaskType(Enum):
    """Types of cognitive tasks."""
    ANALYSIS = "analysis"
    SOLUTION = "solution"
    CREATION = "creation"
    VERIFICATION = "verification"
    RESEARCH = "research"
    EXTRACTION = "extraction"
    SYNTHESIS = "synthesis"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class DomainType(Enum):
    """Domain categories."""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    LEGAL = "legal"
    BUSINESS = "business"
    CREATIVE = "creative"
    GENERAL = "general"


@dataclass
class TaskAnalysis:
    """Analysis result from cognitive router."""
    task_type: TaskType
    complexity: TaskComplexity
    domain: DomainType
    confidence: float
    reasoning: List[str]
    recommended_agent: str
    sub_tasks: List[str]
    estimated_time: float
    requires_validation: bool
    metadata: Dict[str, Any]


class StandaloneCognitiveRouter:
    """Standalone cognitive router for testing."""
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for task analysis."""
        self.patterns = {
            'question_words': re.compile(r'\b(what|how|why|when|where|who|which)\b', re.IGNORECASE),
            'analysis_words': re.compile(r'\b(analyze|examine|compare|evaluate|assess|study|investigate)\b', re.IGNORECASE),
            'creation_words': re.compile(r'\b(create|generate|design|build|develop|write|compose)\b', re.IGNORECASE),
            'solution_words': re.compile(r'\b(solve|fix|resolve|troubleshoot|debug|optimize|implement)\b', re.IGNORECASE),
            'verification_words': re.compile(r'\b(verify|validate|check|test|confirm|review|audit)\b', re.IGNORECASE),
            'research_words': re.compile(r'\b(research|investigate|explore|survey|discover|find out)\b', re.IGNORECASE),
            'synthesis_words': re.compile(r'\b(synthesize|combine|integrate|merge|consolidate|summarize)\b', re.IGNORECASE),
            'complexity_indicators': re.compile(r'\b(complex|complicated|sophisticated|advanced|detailed|comprehensive)\b', re.IGNORECASE),
            'simple_indicators': re.compile(r'\b(simple|basic|quick|easy|straightforward|brief)\b', re.IGNORECASE),
            'multi_step': re.compile(r'\b(first|then|next|finally|step by step|process|procedure)\b', re.IGNORECASE),
            'domain_academic': re.compile(r'\b(research|study|paper|academic|scientific|theory|hypothesis|experiment)\b', re.IGNORECASE),
            'domain_technical': re.compile(r'\b(code|software|system|algorithm|implementation|technical|programming|api)\b', re.IGNORECASE),
            'domain_medical': re.compile(r'\b(medical|health|clinical|patient|diagnosis|treatment|therapeutic|disease)\b', re.IGNORECASE),
            'domain_legal': re.compile(r'\b(legal|law|regulation|compliance|contract|rights|liability|court)\b', re.IGNORECASE),
            'domain_business': re.compile(r'\b(business|market|strategy|revenue|profit|customer|sales|management)\b', re.IGNORECASE),
        }
    
    async def analyze_task(self, task: str, context: Optional[str] = None) -> TaskAnalysis:
        """Analyze a task to determine optimal routing."""
        task_lower = task.lower()
        full_text = f"{task} {context or ''}".lower()
        
        reasoning = []
        
        # Determine task type
        task_type_scores = {
            TaskType.ANALYSIS: len(self.patterns['analysis_words'].findall(full_text)),
            TaskType.CREATION: len(self.patterns['creation_words'].findall(full_text)),
            TaskType.SOLUTION: len(self.patterns['solution_words'].findall(full_text)),
            TaskType.VERIFICATION: len(self.patterns['verification_words'].findall(full_text)),
            TaskType.RESEARCH: len(self.patterns['research_words'].findall(full_text)),
            TaskType.SYNTHESIS: len(self.patterns['synthesis_words'].findall(full_text)),
        }
        
        # Default to research if query-like
        if self.patterns['question_words'].search(full_text):
            task_type_scores[TaskType.RESEARCH] += 2
        
        task_type = max(task_type_scores, key=task_type_scores.get)
        if task_type_scores[task_type] == 0:
            task_type = TaskType.RESEARCH
        
        reasoning.append(f"Task type '{task_type.value}' based on keyword analysis")
        
        # Determine complexity
        complexity_score = 0
        
        if len(task.split()) > 20:
            complexity_score += 1
            reasoning.append("Complex: long task description")
        
        if self.patterns['multi_step'].search(full_text):
            complexity_score += 2
            reasoning.append("Complex: multi-step process indicated")
        
        if self.patterns['complexity_indicators'].search(full_text):
            complexity_score += 2
            reasoning.append("Complex: complexity indicators found")
        
        if self.patterns['simple_indicators'].search(full_text):
            complexity_score -= 1
            reasoning.append("Simple: simplicity indicators found")
        
        # Multiple domains
        domain_matches = sum([
            bool(self.patterns['domain_academic'].search(full_text)),
            bool(self.patterns['domain_technical'].search(full_text)),
            bool(self.patterns['domain_medical'].search(full_text)),
            bool(self.patterns['domain_legal'].search(full_text)),
            bool(self.patterns['domain_business'].search(full_text)),
        ])
        
        if domain_matches > 1:
            complexity_score += 1
            reasoning.append("Complex: multiple domains involved")
        
        if complexity_score >= 4:
            complexity = TaskComplexity.EXPERT
        elif complexity_score >= 2:
            complexity = TaskComplexity.COMPLEX
        elif complexity_score >= 1:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.SIMPLE
        
        # Determine domain
        domain_scores = {
            DomainType.ACADEMIC: len(self.patterns['domain_academic'].findall(full_text)),
            DomainType.TECHNICAL: len(self.patterns['domain_technical'].findall(full_text)),
            DomainType.MEDICAL: len(self.patterns['domain_medical'].findall(full_text)),
            DomainType.LEGAL: len(self.patterns['domain_legal'].findall(full_text)),
            DomainType.BUSINESS: len(self.patterns['domain_business'].findall(full_text)),
        }
        
        domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[domain] == 0:
            domain = DomainType.GENERAL
        
        reasoning.append(f"Domain '{domain.value}' based on terminology")
        
        # Select agent
        agent_map = {
            (TaskType.ANALYSIS, TaskComplexity.SIMPLE): "AnalysisAgent",
            (TaskType.ANALYSIS, TaskComplexity.MODERATE): "AnalysisAgent", 
            (TaskType.ANALYSIS, TaskComplexity.COMPLEX): "ExpertAnalysisAgent",
            (TaskType.ANALYSIS, TaskComplexity.EXPERT): "ExpertAnalysisAgent",
            (TaskType.SOLUTION, TaskComplexity.SIMPLE): "SolutionAgent",
            (TaskType.SOLUTION, TaskComplexity.MODERATE): "SolutionAgent",
            (TaskType.SOLUTION, TaskComplexity.COMPLEX): "ExpertSolutionAgent",
            (TaskType.SOLUTION, TaskComplexity.EXPERT): "ExpertSolutionAgent",
            (TaskType.CREATION, TaskComplexity.SIMPLE): "CreationAgent",
            (TaskType.CREATION, TaskComplexity.MODERATE): "CreationAgent",
            (TaskType.CREATION, TaskComplexity.COMPLEX): "ExpertCreationAgent",
            (TaskType.CREATION, TaskComplexity.EXPERT): "ExpertCreationAgent",
            (TaskType.VERIFICATION, TaskComplexity.SIMPLE): "VerificationAgent",
            (TaskType.VERIFICATION, TaskComplexity.MODERATE): "VerificationAgent",
            (TaskType.VERIFICATION, TaskComplexity.COMPLEX): "ExpertVerificationAgent",
            (TaskType.VERIFICATION, TaskComplexity.EXPERT): "ExpertVerificationAgent",
            (TaskType.RESEARCH, TaskComplexity.SIMPLE): "QueryAgent",
            (TaskType.RESEARCH, TaskComplexity.MODERATE): "ResearchAgent",
            (TaskType.RESEARCH, TaskComplexity.COMPLEX): "ResearchAgent",
            (TaskType.RESEARCH, TaskComplexity.EXPERT): "ExpertResearchAgent",
            (TaskType.SYNTHESIS, TaskComplexity.SIMPLE): "SynthesisAgent",
            (TaskType.SYNTHESIS, TaskComplexity.MODERATE): "SynthesisAgent",
            (TaskType.SYNTHESIS, TaskComplexity.COMPLEX): "ExpertSynthesisAgent",
            (TaskType.SYNTHESIS, TaskComplexity.EXPERT): "ExpertSynthesisAgent",
        }
        
        recommended_agent = agent_map.get((task_type, complexity), "QueryAgent")
        
        # Estimate time
        base_times = {
            TaskComplexity.SIMPLE: 30,
            TaskComplexity.MODERATE: 120,
            TaskComplexity.COMPLEX: 300,
            TaskComplexity.EXPERT: 600,
        }
        estimated_time = base_times[complexity]
        
        # Check validation requirement
        requires_validation = (
            complexity == TaskComplexity.EXPERT or
            domain in [DomainType.MEDICAL, DomainType.LEGAL] or
            task_type in [TaskType.CREATION, TaskType.SOLUTION]
        )
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            domain=domain,
            confidence=0.7,
            reasoning=reasoning,
            recommended_agent=recommended_agent,
            sub_tasks=[],
            estimated_time=estimated_time,
            requires_validation=requires_validation,
            metadata={
                "task_type_scores": {k.value: v for k, v in task_type_scores.items()},
                "domain_scores": {k.value: v for k, v in domain_scores.items()},
                "complexity_score": complexity_score,
            }
        )


async def test_basic_routing():
    """Test basic routing functionality."""
    print("Testing Basic Routing...")
    
    router = StandaloneCognitiveRouter()
    
    test_cases = [
        ("Analyze the benefits of machine learning in healthcare", TaskType.ANALYSIS, DomainType.ACADEMIC),
        ("Create a blog post about AI ethics", TaskType.CREATION, DomainType.TECHNICAL),
        ("Solve the data quality issues in our pipeline", TaskType.SOLUTION, DomainType.TECHNICAL),
        ("Verify the accuracy of financial projections", TaskType.VERIFICATION, DomainType.BUSINESS),
        ("What is artificial intelligence?", TaskType.RESEARCH, DomainType.GENERAL),
        ("Research the latest developments in quantum computing", TaskType.RESEARCH, DomainType.ACADEMIC),
        ("Synthesize information from multiple research papers", TaskType.SYNTHESIS, DomainType.ACADEMIC),
    ]
    
    print("\nRouting Results:")
    print("-" * 80)
    
    correct_predictions = 0
    
    for task, expected_type, expected_domain in test_cases:
        analysis = await router.analyze_task(task)
        
        type_correct = analysis.task_type == expected_type
        domain_correct = analysis.domain == expected_domain
        
        if type_correct:
            correct_predictions += 1
        
        print(f"Task: {task}")
        print(f"  Type: {analysis.task_type.value} {'✓' if type_correct else '✗'} (expected: {expected_type.value})")
        print(f"  Domain: {analysis.domain.value} {'✓' if domain_correct else '✗'} (expected: {expected_domain.value})")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Agent: {analysis.recommended_agent}")
        print(f"  Time: {analysis.estimated_time}s")
        print(f"  Validation: {analysis.requires_validation}")
        print()
    
    accuracy = correct_predictions / len(test_cases)
    print(f"Task Type Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_cases)})")
    
    return accuracy >= 0.6  # 60% accuracy threshold


async def test_complexity_detection():
    """Test complexity detection."""
    print("Testing Complexity Detection...")
    
    router = StandaloneCognitiveRouter()
    
    complexity_tests = [
        ("Fix bug", TaskComplexity.SIMPLE),
        ("Analyze customer satisfaction data", TaskComplexity.MODERATE),
        ("Design a comprehensive AI strategy for healthcare implementation", TaskComplexity.COMPLEX),
        ("Create a complex multi-step process for advanced neural network optimization", TaskComplexity.EXPERT),
    ]
    
    correct = 0
    for task, expected_complexity in complexity_tests:
        analysis = await router.analyze_task(task)
        is_correct = analysis.complexity == expected_complexity
        if is_correct:
            correct += 1
        
        print(f"'{task}' -> {analysis.complexity.value} {'✓' if is_correct else '✗'}")
    
    print(f"Complexity Accuracy: {correct}/{len(complexity_tests)}")
    return correct >= len(complexity_tests) // 2


async def test_domain_detection():
    """Test domain detection."""
    print("Testing Domain Detection...")
    
    router = StandaloneCognitiveRouter()
    
    domain_tests = [
        ("Medical diagnosis using machine learning", DomainType.MEDICAL),
        ("Software bug in the API implementation", DomainType.TECHNICAL),
        ("Legal compliance for data privacy", DomainType.LEGAL),
        ("Market analysis for Q4 revenue", DomainType.BUSINESS),
        ("Academic research on neural networks", DomainType.ACADEMIC),
        ("General question about weather", DomainType.GENERAL),
    ]
    
    correct = 0
    for task, expected_domain in domain_tests:
        analysis = await router.analyze_task(task)
        is_correct = analysis.domain == expected_domain
        if is_correct:
            correct += 1
        
        print(f"'{task}' -> {analysis.domain.value} {'✓' if is_correct else '✗'}")
    
    print(f"Domain Accuracy: {correct}/{len(domain_tests)}")
    return correct >= len(domain_tests) // 2


async def test_agent_selection():
    """Test agent selection logic."""
    print("Testing Agent Selection...")
    
    router = StandaloneCognitiveRouter()
    
    selection_tests = [
        ("Analyze data trends", "AnalysisAgent"),
        ("Create a marketing document", "CreationAgent"),
        ("Fix the authentication system", "SolutionAgent"),
        ("Verify calculation accuracy", "VerificationAgent"),
        ("What is machine learning?", "QueryAgent"),
        ("Research the latest AI developments", "ResearchAgent"),
    ]
    
    correct = 0
    for task, expected_agent_type in selection_tests:
        analysis = await router.analyze_task(task)
        # Check if the selected agent contains the expected type
        is_correct = expected_agent_type.replace("Agent", "") in analysis.recommended_agent
        if is_correct:
            correct += 1
        
        print(f"'{task}' -> {analysis.recommended_agent} {'✓' if is_correct else '✗'}")
    
    print(f"Agent Selection Accuracy: {correct}/{len(selection_tests)}")
    return correct >= len(selection_tests) // 2


async def main():
    """Run all tests."""
    print("🧠 Testing Standalone Cognitive Router...\n")
    
    tests = [
        ("Basic Routing", test_basic_routing),
        ("Complexity Detection", test_complexity_detection),
        ("Domain Detection", test_domain_detection),
        ("Agent Selection", test_agent_selection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}:")
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}\n")
    
    print("=" * 60)
    print(f"Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All cognitive routing tests passed!")
        print("\n✨ Phase 2 Core Logic Complete!")
        print("\nImplemented Features:")
        print("• ✅ Task type classification (analysis, solution, creation, etc.)")
        print("• ✅ Complexity assessment (simple → expert)")
        print("• ✅ Domain detection (academic, technical, medical, etc.)")
        print("• ✅ Intelligent agent selection")
        print("• ✅ Time estimation")
        print("• ✅ Validation requirement detection")
        print("\n🎯 Ready for integration with RAG pipeline!")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)