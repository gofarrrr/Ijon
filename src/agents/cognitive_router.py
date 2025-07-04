"""
Cognitive Router for intelligent task analysis and agent selection.

This module implements a router that analyzes task complexity and requirements
to select the most appropriate cognitive agent for the task.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass
from openai import AsyncOpenAI

from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


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
    SIMPLE = "simple"      # Single-step, straightforward
    MODERATE = "moderate"  # Multi-step, some reasoning
    COMPLEX = "complex"    # Multi-domain, deep reasoning
    EXPERT = "expert"      # Highly specialized, advanced


class DomainType(Enum):
    """Domain categories for specialized handling."""
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
    estimated_time: float  # in seconds
    requires_validation: bool
    metadata: Dict[str, Any]


class CognitiveRouter:
    """
    Stateless cognitive router following 12-factor principles.
    
    Analyzes tasks to determine optimal cognitive agent assignment
    and task decomposition strategies.
    """
    
    def __init__(
        self,
        use_llm_analysis: bool = True,
        model: str = "gpt-3.5-turbo",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize cognitive router.
        
        Args:
            use_llm_analysis: Whether to use LLM for complex analysis
            model: LLM model for analysis
            confidence_threshold: Minimum confidence for routing decisions
        """
        self.use_llm_analysis = use_llm_analysis
        self.model = model
        self.confidence_threshold = confidence_threshold
        
        # Compile regex patterns for efficiency
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
    
    @log_performance
    async def analyze_task(
        self,
        task: str,
        context: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> TaskAnalysis:
        """
        Analyze a task to determine optimal routing - pure function.
        
        Args:
            task: Task description or query
            context: Additional context about the task
            client: OpenAI client for LLM analysis (optional)
            
        Returns:
            TaskAnalysis with routing recommendations
        """
        logger.info(f"Analyzing task: {task[:100]}...")
        
        # Step 1: Rule-based analysis (fast)
        rule_based_analysis = self._rule_based_analysis(task, context)
        
        # Step 2: LLM-enhanced analysis (if enabled and client available)
        if self.use_llm_analysis and client:
            try:
                llm_analysis = await self._llm_enhanced_analysis(task, context, client)
                # Combine rule-based and LLM analysis
                final_analysis = self._combine_analyses(rule_based_analysis, llm_analysis)
            except Exception as e:
                logger.warning(f"LLM analysis failed, using rule-based: {e}")
                final_analysis = rule_based_analysis
        else:
            final_analysis = rule_based_analysis
        
        # Step 3: Determine recommended agent
        final_analysis.recommended_agent = self._select_agent(final_analysis)
        
        # Step 4: Estimate requirements
        final_analysis.estimated_time = self._estimate_time(final_analysis)
        final_analysis.requires_validation = self._requires_validation(final_analysis)
        
        logger.info(
            f"Task routed to {final_analysis.recommended_agent} "
            f"({final_analysis.task_type.value}, {final_analysis.complexity.value})"
        )
        
        return final_analysis
    
    def _rule_based_analysis(self, task: str, context: Optional[str] = None) -> TaskAnalysis:
        """Perform rule-based task analysis using patterns."""
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
            task_type = TaskType.RESEARCH  # Default
        
        reasoning.append(f"Task type '{task_type.value}' based on keyword analysis")
        
        # Determine complexity
        complexity_score = 0
        
        # Length indicator
        if len(task.split()) > 20:
            complexity_score += 1
            reasoning.append("Complex: long task description")
        
        # Multi-step indicators
        if self.patterns['multi_step'].search(full_text):
            complexity_score += 2
            reasoning.append("Complex: multi-step process indicated")
        
        # Explicit complexity words
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
        
        # Map complexity score to enum
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
        
        # Generate sub-tasks (simple heuristic)
        sub_tasks = self._generate_sub_tasks(task, task_type, complexity)
        
        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            domain=domain,
            confidence=0.6,  # Rule-based confidence
            reasoning=reasoning,
            recommended_agent="",  # Will be set later
            sub_tasks=sub_tasks,
            estimated_time=0.0,  # Will be set later
            requires_validation=False,  # Will be set later
            metadata={
                "analysis_method": "rule_based",
                "task_type_scores": {k.value: v for k, v in task_type_scores.items()},
                "domain_scores": {k.value: v for k, v in domain_scores.items()},
                "complexity_score": complexity_score,
            }
        )
    
    async def _llm_enhanced_analysis(
        self,
        task: str,
        context: Optional[str],
        client: AsyncOpenAI,
    ) -> TaskAnalysis:
        """Use LLM for enhanced task analysis."""
        
        prompt = f"""Analyze this task to determine the optimal cognitive approach:

Task: {task}
{f"Context: {context}" if context else ""}

Classify the task along these dimensions:

1. TASK TYPE (pick the primary one):
   - analysis: Deep examination, comparison, evaluation
   - solution: Problem-solving, troubleshooting, implementation
   - creation: Content generation, design, development
   - verification: Validation, testing, quality assurance
   - research: Information gathering, investigation
   - synthesis: Combining information, summarization

2. COMPLEXITY:
   - simple: Single-step, straightforward
   - moderate: Multi-step, some reasoning required
   - complex: Multi-domain, deep reasoning
   - expert: Highly specialized, advanced knowledge

3. DOMAIN:
   - academic: Research, scientific, theoretical
   - technical: Code, systems, implementation
   - medical: Health, clinical, therapeutic
   - legal: Law, regulations, compliance
   - business: Strategy, operations, commercial
   - creative: Design, artistic, innovative
   - general: Cross-domain or general purpose

4. SUB-TASKS: Break down into 2-4 specific steps if complex

Respond with JSON:
{{
  "task_type": "analysis|solution|creation|verification|research|synthesis",
  "complexity": "simple|moderate|complex|expert",
  "domain": "academic|technical|medical|legal|business|creative|general",
  "confidence": 0.8,
  "reasoning": ["reason1", "reason2"],
  "sub_tasks": ["step1", "step2"]
}}"""
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing cognitive tasks. Respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return TaskAnalysis(
                task_type=TaskType(result["task_type"]),
                complexity=TaskComplexity(result["complexity"]),
                domain=DomainType(result["domain"]),
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", []),
                recommended_agent="",  # Will be set later
                sub_tasks=result.get("sub_tasks", []),
                estimated_time=0.0,  # Will be set later
                requires_validation=False,  # Will be set later
                metadata={
                    "analysis_method": "llm_enhanced",
                    "model_used": self.model,
                }
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            raise
    
    def _combine_analyses(
        self,
        rule_based: TaskAnalysis,
        llm_based: TaskAnalysis,
    ) -> TaskAnalysis:
        """Combine rule-based and LLM analyses."""
        
        # Take LLM analysis as primary, but boost confidence if they agree
        confidence = llm_based.confidence
        if (rule_based.task_type == llm_based.task_type and 
            rule_based.complexity == llm_based.complexity):
            confidence = min(1.0, confidence + 0.2)
        
        # Combine reasoning
        combined_reasoning = rule_based.reasoning + llm_based.reasoning
        combined_reasoning.append("Combined rule-based and LLM analysis")
        
        # Prefer LLM sub-tasks if available, else rule-based
        sub_tasks = llm_based.sub_tasks if llm_based.sub_tasks else rule_based.sub_tasks
        
        return TaskAnalysis(
            task_type=llm_based.task_type,
            complexity=llm_based.complexity,
            domain=llm_based.domain,
            confidence=confidence,
            reasoning=combined_reasoning,
            recommended_agent="",  # Will be set later
            sub_tasks=sub_tasks,
            estimated_time=0.0,  # Will be set later
            requires_validation=False,  # Will be set later
            metadata={
                "analysis_method": "combined",
                "rule_based_metadata": rule_based.metadata,
                "llm_metadata": llm_based.metadata,
            }
        )
    
    def _select_agent(self, analysis: TaskAnalysis) -> str:
        """Select the best agent based on analysis."""
        
        # Agent selection matrix
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
            
            (TaskType.RESEARCH, TaskComplexity.SIMPLE): "QueryAgent",  # Use existing
            (TaskType.RESEARCH, TaskComplexity.MODERATE): "ResearchAgent",  # Use existing
            (TaskType.RESEARCH, TaskComplexity.COMPLEX): "ResearchAgent",
            (TaskType.RESEARCH, TaskComplexity.EXPERT): "ExpertResearchAgent",
            
            (TaskType.SYNTHESIS, TaskComplexity.SIMPLE): "SynthesisAgent",
            (TaskType.SYNTHESIS, TaskComplexity.MODERATE): "SynthesisAgent",
            (TaskType.SYNTHESIS, TaskComplexity.COMPLEX): "ExpertSynthesisAgent",
            (TaskType.SYNTHESIS, TaskComplexity.EXPERT): "ExpertSynthesisAgent",
        }
        
        return agent_map.get(
            (analysis.task_type, analysis.complexity),
            "QueryAgent"  # Default fallback
        )
    
    def _generate_sub_tasks(
        self,
        task: str,
        task_type: TaskType,
        complexity: TaskComplexity,
    ) -> List[str]:
        """Generate sub-tasks based on task type and complexity."""
        
        if complexity == TaskComplexity.SIMPLE:
            return [task]  # No decomposition needed
        
        # Basic decomposition patterns
        base_patterns = {
            TaskType.ANALYSIS: [
                "Gather relevant information",
                "Analyze key components",
                "Identify patterns and relationships",
                "Draw conclusions and insights"
            ],
            TaskType.SOLUTION: [
                "Understand the problem",
                "Identify potential solutions",
                "Evaluate solution options",
                "Implement chosen solution"
            ],
            TaskType.CREATION: [
                "Define requirements and scope",
                "Research and gather inspiration",
                "Design and develop content",
                "Review and refine output"
            ],
            TaskType.VERIFICATION: [
                "Define verification criteria",
                "Gather evidence and test cases", 
                "Execute validation tests",
                "Report findings and recommendations"
            ],
            TaskType.RESEARCH: [
                "Define research scope",
                "Collect relevant information",
                "Analyze and synthesize findings",
                "Present comprehensive results"
            ],
            TaskType.SYNTHESIS: [
                "Identify source materials",
                "Extract key information",
                "Combine and integrate findings",
                "Create coherent synthesis"
            ]
        }
        
        pattern = base_patterns.get(task_type, base_patterns[TaskType.RESEARCH])
        
        # Adjust based on complexity
        if complexity == TaskComplexity.EXPERT:
            return pattern  # Full decomposition
        elif complexity == TaskComplexity.COMPLEX:
            return pattern[:3]  # Skip last step
        else:  # MODERATE
            return pattern[:2]  # First two steps
    
    def _estimate_time(self, analysis: TaskAnalysis) -> float:
        """Estimate time requirements in seconds."""
        
        base_times = {
            TaskComplexity.SIMPLE: 30,      # 30 seconds
            TaskComplexity.MODERATE: 120,   # 2 minutes
            TaskComplexity.COMPLEX: 300,    # 5 minutes
            TaskComplexity.EXPERT: 600,     # 10 minutes
        }
        
        base_time = base_times[analysis.complexity]
        
        # Adjust for task type
        type_multipliers = {
            TaskType.ANALYSIS: 1.2,
            TaskType.SOLUTION: 1.5,
            TaskType.CREATION: 2.0,
            TaskType.VERIFICATION: 1.1,
            TaskType.RESEARCH: 1.3,
            TaskType.SYNTHESIS: 1.4,
        }
        
        multiplier = type_multipliers.get(analysis.task_type, 1.0)
        
        # Adjust for domain complexity
        if analysis.domain in [DomainType.MEDICAL, DomainType.LEGAL]:
            multiplier *= 1.3
        elif analysis.domain == DomainType.TECHNICAL:
            multiplier *= 1.2
        
        return base_time * multiplier
    
    def _requires_validation(self, analysis: TaskAnalysis) -> bool:
        """Determine if task requires human validation."""
        
        # Always validate expert-level tasks
        if analysis.complexity == TaskComplexity.EXPERT:
            return True
        
        # Validate high-stakes domains
        if analysis.domain in [DomainType.MEDICAL, DomainType.LEGAL]:
            return True
        
        # Validate creation and solution tasks
        if analysis.task_type in [TaskType.CREATION, TaskType.SOLUTION]:
            if analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
                return True
        
        # Validate low-confidence analyses
        if analysis.confidence < self.confidence_threshold:
            return True
        
        return False
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get capabilities information for an agent."""
        
        capabilities = {
            "AnalysisAgent": {
                "strengths": ["Pattern recognition", "Comparative analysis", "Data interpretation"],
                "best_for": ["Understanding relationships", "Finding insights", "Evaluating information"],
                "limitations": ["Complex multi-domain tasks", "Creative solutions"],
                "estimated_accuracy": 0.85,
            },
            "SolutionAgent": {
                "strengths": ["Problem decomposition", "Solution design", "Implementation planning"],
                "best_for": ["Fixing issues", "Optimizing processes", "Implementing changes"],
                "limitations": ["Highly creative tasks", "Domain expertise requirements"],
                "estimated_accuracy": 0.80,
            },
            "CreationAgent": {
                "strengths": ["Content generation", "Creative thinking", "Structured output"],
                "best_for": ["Writing", "Design", "Content development"],
                "limitations": ["Technical implementation", "Complex analysis"],
                "estimated_accuracy": 0.75,
            },
            "VerificationAgent": {
                "strengths": ["Quality assessment", "Fact checking", "Logical validation"],
                "best_for": ["Testing", "Review", "Quality assurance"],
                "limitations": ["Creative evaluation", "Subjective judgments"],
                "estimated_accuracy": 0.90,
            },
            "QueryAgent": {
                "strengths": ["Information retrieval", "Question answering", "Context synthesis"],
                "best_for": ["Research", "Information lookup", "Basic queries"],
                "limitations": ["Complex reasoning", "Multi-step tasks"],
                "estimated_accuracy": 0.85,
            },
            "ResearchAgent": {
                "strengths": ["Information gathering", "Source analysis", "Comprehensive research"],
                "best_for": ["In-depth investigation", "Literature review", "Evidence gathering"],
                "limitations": ["Real-time information", "Highly specialized domains"],
                "estimated_accuracy": 0.85,
            },
        }
        
        return capabilities.get(agent_name, {
            "strengths": ["General purpose"],
            "best_for": ["Various tasks"],
            "limitations": ["Specialized requirements"],
            "estimated_accuracy": 0.70,
        })


def create_cognitive_router(use_llm: bool = True) -> CognitiveRouter:
    """
    Factory function to create a cognitive router.
    
    Args:
        use_llm: Whether to enable LLM-enhanced analysis
        
    Returns:
        Configured cognitive router
    """
    return CognitiveRouter(
        use_llm_analysis=use_llm,
        model="gpt-3.5-turbo",
        confidence_threshold=0.7,
    )