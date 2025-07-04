"""
Specialized cognitive agents for different types of thinking tasks.

These agents provide focused cognitive capabilities for analysis, solution,
creation, verification, and synthesis tasks.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent, AgentContext, ToolRegistry
from src.agents.prompts import (
    COGNITIVE_ANALYSIS_AGENT_PROMPT,
    COGNITIVE_SOLUTION_AGENT_PROMPT,
    COGNITIVE_CREATION_AGENT_PROMPT,
    COGNITIVE_VERIFICATION_AGENT_PROMPT,
    COGNITIVE_SYNTHESIS_AGENT_PROMPT,
)
from src.agents.tools import AgentTools
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Response models for structured outputs

class AnalysisResult(BaseModel):
    """Structured result from analysis agent."""
    key_findings: List[str] = Field(..., description="Main insights discovered")
    patterns: List[str] = Field(default_factory=list, description="Patterns identified")
    relationships: List[Dict[str, str]] = Field(default_factory=list, description="Key relationships")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    limitations: List[str] = Field(default_factory=list, description="Analysis limitations")
    sources_used: List[str] = Field(default_factory=list, description="Sources referenced")


class SolutionResult(BaseModel):
    """Structured result from solution agent."""
    problem_summary: str = Field(..., description="Clear problem definition")
    solutions: List[Dict[str, Any]] = Field(..., description="Proposed solutions")
    recommended_solution: Dict[str, Any] = Field(..., description="Best solution with details")
    implementation_plan: List[str] = Field(..., description="Step-by-step implementation")
    risks: List[Dict[str, str]] = Field(default_factory=list, description="Identified risks and mitigations")
    success_metrics: List[str] = Field(default_factory=list, description="How to measure success")
    resources_needed: List[str] = Field(default_factory=list, description="Required resources")


class CreationResult(BaseModel):
    """Structured result from creation agent."""
    created_content: str = Field(..., description="The generated content")
    content_type: str = Field(..., description="Type of content created")
    design_rationale: List[str] = Field(default_factory=list, description="Reasoning behind design choices")
    alternatives: List[str] = Field(default_factory=list, description="Alternative versions or approaches")
    quality_assessment: Dict[str, Any] = Field(default_factory=dict, description="Self-assessment of quality")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Ways to improve further")
    inspiration_sources: List[str] = Field(default_factory=list, description="Sources that inspired the creation")


class VerificationResult(BaseModel):
    """Structured result from verification agent."""
    overall_assessment: str = Field(..., description="Overall quality assessment")
    passed_checks: List[str] = Field(default_factory=list, description="Tests that passed")
    failed_checks: List[str] = Field(default_factory=list, description="Tests that failed")
    issues_found: List[Dict[str, str]] = Field(default_factory=list, description="Specific issues identified")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    validation_method: str = Field(..., description="How validation was performed")


class SynthesisResult(BaseModel):
    """Structured result from synthesis agent."""
    unified_understanding: str = Field(..., description="Integrated synthesis")
    key_themes: List[str] = Field(default_factory=list, description="Main themes identified")
    source_perspectives: List[Dict[str, str]] = Field(default_factory=list, description="Different viewpoints")
    agreements: List[str] = Field(default_factory=list, description="Where sources agree")
    conflicts: List[str] = Field(default_factory=list, description="Where sources disagree")
    gaps: List[str] = Field(default_factory=list, description="Information gaps identified")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in synthesis")


# Specialized Cognitive Agents

class AnalysisAgent(BaseAgent[AnalysisResult]):
    """Agent specialized in deep analysis and pattern recognition."""
    
    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        self.tools = tools or AgentTools()
        
        super().__init__(
            name="AnalysisAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=COGNITIVE_ANALYSIS_AGENT_PROMPT,
            response_model=AnalysisResult,
            temperature=temperature,
        )
    
    def _get_default_system_prompt(self) -> str:
        return COGNITIVE_ANALYSIS_AGENT_PROMPT
    
    def _register_tools(self) -> None:
        """Register analysis-specific tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("search_patterns", "Search for patterns and relationships in data")
        async def search_patterns(ctx, query: str, pattern_type: str = "general"):
            """Search for patterns in the knowledge base."""
            return await self.tools.search_documents(ctx, f"patterns relationships {query}", top_k=10)
        
        @registry.register("compare_sources", "Compare information across multiple sources")
        async def compare_sources(ctx, topics: List[str]):
            """Compare how different sources handle the same topics."""
            comparisons = []
            for topic in topics:
                results = await self.tools.search_documents(ctx, topic, top_k=5)
                comparisons.append({"topic": topic, "sources": results})
            return comparisons
        
        @registry.register("analyze_trends", "Analyze trends and temporal patterns")
        async def analyze_trends(ctx, topic: str, time_focus: str = "recent"):
            """Analyze trends over time for a given topic."""
            query = f"{topic} trends {time_focus} changes evolution"
            return await self.tools.search_documents(ctx, query, top_k=8)
    
    async def analyze_topic(
        self,
        topic: str,
        context: Optional[AgentContext] = None,
        depth: str = "comprehensive",
    ) -> AnalysisResult:
        """
        Perform comprehensive analysis of a topic.
        
        Args:
            topic: Topic to analyze
            context: Agent execution context
            depth: Analysis depth (quick, standard, comprehensive)
            
        Returns:
            Structured analysis result
        """
        analysis_prompt = f"""Perform a {depth} analysis of: {topic}

Focus on:
1. Key patterns and relationships
2. Multiple perspectives and interpretations  
3. Evidence quality and source reliability
4. Practical implications and insights
5. Areas of uncertainty or conflicting information

Use your search tools to gather comprehensive information before analyzing."""
        
        response = await self.run(analysis_prompt, context)
        return response.result


class SolutionAgent(BaseAgent[SolutionResult]):
    """Agent specialized in problem-solving and solution design."""
    
    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,
    ):
        self.tools = tools or AgentTools()
        
        super().__init__(
            name="SolutionAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=COGNITIVE_SOLUTION_AGENT_PROMPT,
            response_model=SolutionResult,
            temperature=temperature,
        )
    
    def _get_default_system_prompt(self) -> str:
        return COGNITIVE_SOLUTION_AGENT_PROMPT
    
    def _register_tools(self) -> None:
        """Register solution-specific tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("find_precedents", "Find similar problems and their solutions")
        async def find_precedents(ctx, problem_description: str):
            """Find precedents for similar problems."""
            query = f"solutions approaches methods {problem_description} similar problems"
            return await self.tools.search_documents(ctx, query, top_k=8)
        
        @registry.register("evaluate_constraints", "Analyze constraints and requirements")
        async def evaluate_constraints(ctx, problem: str, constraints: List[str]):
            """Evaluate constraints and their impact on solutions."""
            constraint_str = " ".join(constraints)
            query = f"{problem} constraints limitations requirements {constraint_str}"
            return await self.tools.search_documents(ctx, query, top_k=6)
        
        @registry.register("assess_feasibility", "Assess solution feasibility")
        async def assess_feasibility(ctx, solution: str, context_info: str):
            """Assess the feasibility of a proposed solution."""
            query = f"feasibility implementation {solution} {context_info} practical considerations"
            return await self.tools.search_documents(ctx, query, top_k=5)
    
    async def solve_problem(
        self,
        problem: str,
        constraints: Optional[List[str]] = None,
        context: Optional[AgentContext] = None,
    ) -> SolutionResult:
        """
        Generate comprehensive solution for a problem.
        
        Args:
            problem: Problem description
            constraints: Known constraints or limitations
            context: Agent execution context
            
        Returns:
            Structured solution result
        """
        constraints_str = f"Constraints: {', '.join(constraints)}" if constraints else ""
        
        solution_prompt = f"""Solve this problem: {problem}

{constraints_str}

Provide:
1. Multiple solution alternatives with pros/cons
2. Detailed implementation plan for the best solution
3. Risk assessment and mitigation strategies
4. Resource requirements and timeline
5. Success metrics and validation methods

Use your tools to research similar problems and proven solutions."""
        
        response = await self.run(solution_prompt, context)
        return response.result


class CreationAgent(BaseAgent[CreationResult]):
    """Agent specialized in content creation and innovative thinking."""
    
    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.tools = tools or AgentTools()
        
        super().__init__(
            name="CreationAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=COGNITIVE_CREATION_AGENT_PROMPT,
            response_model=CreationResult,
            temperature=temperature,
        )
    
    def _get_default_system_prompt(self) -> str:
        return COGNITIVE_CREATION_AGENT_PROMPT
    
    def _register_tools(self) -> None:
        """Register creation-specific tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("gather_inspiration", "Gather inspiration from diverse sources")
        async def gather_inspiration(ctx, topic: str, style: str = "diverse"):
            """Gather inspiration for creative work."""
            query = f"examples inspiration {topic} {style} creative approaches"
            return await self.tools.search_documents(ctx, query, top_k=10)
        
        @registry.register("find_patterns", "Find creative patterns and frameworks")
        async def find_patterns(ctx, content_type: str):
            """Find patterns and frameworks for content creation."""
            query = f"{content_type} structure framework template patterns best practices"
            return await self.tools.search_documents(ctx, query, top_k=6)
        
        @registry.register("validate_originality", "Check for similar existing content")
        async def validate_originality(ctx, concept: str):
            """Check if similar content already exists."""
            query = f"similar existing {concept} precedents examples"
            return await self.tools.search_documents(ctx, query, top_k=5)
    
    async def create_content(
        self,
        requirements: str,
        content_type: str,
        context: Optional[AgentContext] = None,
        creativity_level: str = "balanced",
    ) -> CreationResult:
        """
        Create original content based on requirements.
        
        Args:
            requirements: What needs to be created
            content_type: Type of content (article, design, code, etc.)
            context: Agent execution context
            creativity_level: balanced, conservative, innovative
            
        Returns:
            Structured creation result
        """
        creation_prompt = f"""Create {content_type} that meets these requirements: {requirements}

Creativity level: {creativity_level}

Process:
1. Gather inspiration and understand requirements
2. Generate original content that meets all criteria
3. Explain your design rationale and creative choices
4. Provide alternatives or variations
5. Assess quality and suggest improvements

Use your tools to research relevant examples and frameworks."""
        
        response = await self.run(creation_prompt, context)
        return response.result


class VerificationAgent(BaseAgent[VerificationResult]):
    """Agent specialized in quality assurance and validation."""
    
    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.tools = tools or AgentTools()
        
        super().__init__(
            name="VerificationAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=COGNITIVE_VERIFICATION_AGENT_PROMPT,
            response_model=VerificationResult,
            temperature=temperature,
        )
    
    def _get_default_system_prompt(self) -> str:
        return COGNITIVE_VERIFICATION_AGENT_PROMPT
    
    def _register_tools(self) -> None:
        """Register verification-specific tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("cross_reference", "Cross-reference information with sources")
        async def cross_reference(ctx, claim: str):
            """Cross-reference a claim with authoritative sources."""
            query = f"verify confirm {claim} authoritative sources evidence"
            return await self.tools.search_documents(ctx, query, top_k=8)
        
        @registry.register("check_consistency", "Check for logical consistency")
        async def check_consistency(ctx, content: str, topic: str):
            """Check content for internal consistency."""
            query = f"{topic} logical consistency standards best practices"
            return await self.tools.search_documents(ctx, query, top_k=5)
        
        @registry.register("validate_standards", "Validate against standards and best practices")
        async def validate_standards(ctx, content_type: str, domain: str):
            """Validate against domain standards."""
            query = f"{content_type} {domain} standards requirements quality criteria"
            return await self.tools.search_documents(ctx, query, top_k=6)
    
    async def verify_content(
        self,
        content: str,
        criteria: List[str],
        context: Optional[AgentContext] = None,
        verification_type: str = "comprehensive",
    ) -> VerificationResult:
        """
        Verify content against specified criteria.
        
        Args:
            content: Content to verify
            criteria: Verification criteria
            context: Agent execution context
            verification_type: Type of verification (quick, standard, comprehensive)
            
        Returns:
            Structured verification result
        """
        criteria_str = "\n".join(f"- {criterion}" for criterion in criteria)
        
        verification_prompt = f"""Verify this content against the specified criteria:

CONTENT:
{content}

VERIFICATION CRITERIA:
{criteria_str}

VERIFICATION TYPE: {verification_type}

For each criterion:
1. Check if the content meets the requirement
2. Provide specific evidence for your assessment
3. Identify any issues or areas for improvement
4. Give an overall quality score

Use your tools to cross-reference information and validate against standards."""
        
        response = await self.run(verification_prompt, context)
        return response.result


class SynthesisAgent(BaseAgent[SynthesisResult]):
    """Agent specialized in synthesizing information from multiple sources."""
    
    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,
    ):
        self.tools = tools or AgentTools()
        
        super().__init__(
            name="SynthesisAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=COGNITIVE_SYNTHESIS_AGENT_PROMPT,
            response_model=SynthesisResult,
            temperature=temperature,
        )
    
    def _get_default_system_prompt(self) -> str:
        return COGNITIVE_SYNTHESIS_AGENT_PROMPT
    
    def _register_tools(self) -> None:
        """Register synthesis-specific tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("gather_perspectives", "Gather multiple perspectives on a topic")
        async def gather_perspectives(ctx, topic: str):
            """Gather diverse perspectives on a topic."""
            perspectives = []
            for angle in ["overview", "benefits", "challenges", "applications", "future"]:
                query = f"{topic} {angle} perspective viewpoint"
                results = await self.tools.search_documents(ctx, query, top_k=3)
                perspectives.append({"angle": angle, "sources": results})
            return perspectives
        
        @registry.register("identify_themes", "Identify common themes across sources")
        async def identify_themes(ctx, topic: str, source_count: int = 10):
            """Identify common themes in multiple sources."""
            query = f"{topic} themes patterns common elements"
            return await self.tools.search_documents(ctx, query, top_k=source_count)
        
        @registry.register("resolve_conflicts", "Find information to resolve conflicting viewpoints")
        async def resolve_conflicts(ctx, conflict_description: str):
            """Find authoritative information to resolve conflicts."""
            query = f"authoritative definitive {conflict_description} resolution evidence"
            return await self.tools.search_documents(ctx, query, top_k=6)
    
    async def synthesize_information(
        self,
        topic: str,
        sources: Optional[List[str]] = None,
        context: Optional[AgentContext] = None,
        synthesis_focus: str = "comprehensive",
    ) -> SynthesisResult:
        """
        Synthesize information from multiple sources on a topic.
        
        Args:
            topic: Topic to synthesize information about
            sources: Specific sources to focus on (optional)
            context: Agent execution context
            synthesis_focus: Focus area (comprehensive, conflicts, themes, timeline)
            
        Returns:
            Structured synthesis result
        """
        sources_str = f"Focus on these sources: {', '.join(sources)}" if sources else ""
        
        synthesis_prompt = f"""Synthesize information about: {topic}

{sources_str}

Focus: {synthesis_focus}

Process:
1. Gather information from multiple sources using your tools
2. Identify key themes and common elements
3. Note where sources agree and disagree
4. Resolve conflicts using authoritative evidence
5. Create a unified understanding that preserves important nuances
6. Identify gaps where more information is needed

Provide a comprehensive synthesis that integrates all perspectives."""
        
        response = await self.run(synthesis_prompt, context)
        return response.result


# Agent factory functions

def create_analysis_agent(tools: Optional[AgentTools] = None) -> AnalysisAgent:
    """Create an analysis agent with default configuration."""
    return AnalysisAgent(tools=tools)


def create_solution_agent(tools: Optional[AgentTools] = None) -> SolutionAgent:
    """Create a solution agent with default configuration."""
    return SolutionAgent(tools=tools)


def create_creation_agent(tools: Optional[AgentTools] = None) -> CreationAgent:
    """Create a creation agent with default configuration."""
    return CreationAgent(tools=tools)


def create_verification_agent(tools: Optional[AgentTools] = None) -> VerificationAgent:
    """Create a verification agent with default configuration."""
    return VerificationAgent(tools=tools)


def create_synthesis_agent(tools: Optional[AgentTools] = None) -> SynthesisAgent:
    """Create a synthesis agent with default configuration."""
    return SynthesisAgent(tools=tools)