"""
Deep research agent for comprehensive academic-style analysis.

This module implements a research agent that produces exhaustive, detailed
reports with minimum depth requirements and academic prose style.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio

from src.agents.base_agent import BaseAgent, AgentContext, ToolRegistry
from src.agents.event_stream import EventStream, EventType
from src.agents.tools import AgentTools
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ResearchSection(BaseModel):
    """Represents a section of the research report."""
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content in academic prose")
    word_count: int = Field(..., description="Word count for this section")
    subsections: List["ResearchSection"] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list, description="Citations used")
    confidence: float = Field(0.8, ge=0.0, le=1.0)


class DeepResearchResult(BaseModel):
    """Structured result from deep research agent."""
    title: str = Field(..., description="Research title")
    abstract: str = Field(..., description="Executive summary of findings")
    sections: List[ResearchSection] = Field(..., description="Research sections")
    total_word_count: int = Field(..., description="Total word count")
    sources_consulted: int = Field(..., description="Number of sources consulted")
    research_phases: List[str] = Field(..., description="Phases completed")
    confidence_assessment: Dict[str, float] = Field(..., description="Confidence by section")
    key_findings: List[str] = Field(..., description="Major discoveries")
    knowledge_gaps: List[str] = Field(..., description="Identified gaps")


DEEP_RESEARCH_AGENT_PROMPT = """You are a specialized Deep Research Agent capable of producing comprehensive academic reports.

## Agent Loop Architecture
You operate in iterative research cycles:
1. **Analyze Query**: Decompose the research question into major themes
2. **Plan Research**: Design comprehensive investigation strategy
3. **Execute Research**: Gather information systematically
4. **Synthesize Findings**: Create academic narrative
5. **Validate Depth**: Ensure minimum depth requirements are met

## Research Methodology

### Planning Phase (Always Verbalize)
When beginning research, explicitly state your plan:
- Break the topic into 5-7 major themes or sections
- Identify required source diversity (aim for 20+ sources)
- Plan investigation sequence with dependencies
- Estimate depth needed per section (minimum 1500 words each)
- Design synthesis approach for academic narrative

### Investigation Phase
Systematic information gathering:
- Conduct broad initial search to map the topic landscape
- Deep dive into each identified theme sequentially
- Cross-reference information across multiple sources
- Follow citation trails to authoritative sources
- Identify and fill knowledge gaps iteratively
- Track source reliability and potential biases

### Synthesis Phase
Creating academic narrative:
- Write in continuous paragraphs of 4-6 sentences
- Build ideas progressively with smooth transitions
- Avoid lists or bullet points - convert to flowing prose
- Use varied sentence structures for engagement
- Include inline citations in [1][2] format
- Maintain scholarly tone throughout

## Deep Research Mode Requirements

### Minimum Depth Standards
- Total report: 10,000+ words minimum
- Major sections: 1500-2500 words each
- Subsections: 500-1000 words each
- Introduction: 500-750 words setting comprehensive context
- Conclusion: 750-1000 words synthesizing all findings

### Academic Writing Standards
1. **Prose Quality**
   - Topic sentences that guide readers
   - Evidence-based arguments throughout
   - Smooth transitions between ideas
   - No bullet points or lists in main text
   - Tables only for comparative data

2. **Citation Density**
   - Minimum 2-3 citations per paragraph
   - Mix of direct quotes and paraphrases
   - Diverse source types and perspectives
   - Recent and historical sources balanced

3. **Analytical Depth**
   - Multiple perspectives explored
   - Contradictions addressed explicitly
   - Implications thoroughly examined
   - Future directions considered
   - Limitations acknowledged

## Research Process Tracking

### Thinking Blocks
After each research phase, reflect:
```
What patterns are emerging?
Which perspectives need more exploration?
What contradictions require reconciliation?
How does this fit the larger narrative?
What depth is still needed?
```

### Progress Monitoring
Track and report:
- Words written per section
- Sources consulted per theme
- Knowledge gaps identified
- Confidence levels by topic
- Time spent per phase

## Output Structure

### Document Organization
1. **Title and Abstract** (250-300 words)
   - Comprehensive overview
   - Key findings preview
   - Methodology summary

2. **Introduction** (500-750 words)
   - Context and background
   - Research questions
   - Scope and approach
   - Document structure

3. **Main Body** (8000+ words)
   - 5-7 major themed sections
   - Progressive idea development
   - Rich evidence integration
   - Multiple perspective analysis

4. **Conclusion** (750-1000 words)
   - Synthesis of findings
   - Implications analysis
   - Future research directions
   - Final thoughts

5. **References**
   - All sources cited
   - Diverse source types
   - Proper formatting

Remember: Never mention tools or technical processes. Focus on creating comprehensive, scholarly narratives that provide exhaustive treatment of the topic."""


class DeepResearchAgent(BaseAgent[DeepResearchResult]):
    """Agent specialized in deep, comprehensive research with academic output."""
    
    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.5,
        min_word_count: int = 10000,
        min_sources: int = 20,
    ):
        self.tools = tools or AgentTools()
        self.min_word_count = min_word_count
        self.min_sources = min_sources
        self.research_phases: List[str] = []
        self.source_count = 0
        self.current_word_count = 0
        
        super().__init__(
            name="DeepResearchAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=DEEP_RESEARCH_AGENT_PROMPT,
            response_model=DeepResearchResult,
            temperature=temperature,
        )
    
    def _get_default_system_prompt(self) -> str:
        return DEEP_RESEARCH_AGENT_PROMPT
    
    def _register_tools(self) -> None:
        """Register research-specific tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("comprehensive_search", "Search extensively across multiple angles")
        async def comprehensive_search(ctx, topic: str, perspectives: List[str]):
            """Search topic from multiple perspectives."""
            all_results = []
            for perspective in perspectives:
                query = f"{topic} {perspective}"
                results = await self.tools.search_documents(ctx, query, top_k=10)
                all_results.extend(results)
                self.source_count += len(results)
            return all_results
        
        @registry.register("deep_dive_search", "Deep exploration of specific theme")
        async def deep_dive_search(ctx, theme: str, depth_level: str = "comprehensive"):
            """Deep dive into a specific research theme."""
            # Multiple searches with different angles
            angles = [
                "overview fundamentals",
                "recent developments",
                "controversies debates", 
                "practical applications",
                "theoretical frameworks",
                "future directions",
                "limitations challenges",
                "case studies examples"
            ]
            
            theme_results = []
            for angle in angles:
                query = f"{theme} {angle}"
                results = await self.tools.search_documents(ctx, query, top_k=5)
                theme_results.extend(results)
            
            self.source_count += len(theme_results)
            return theme_results
        
        @registry.register("citation_trace", "Follow citation trails to authoritative sources")
        async def citation_trace(ctx, initial_source: str, depth: int = 3):
            """Trace citations to find authoritative sources."""
            traced_sources = []
            current_sources = [initial_source]
            
            for level in range(depth):
                next_sources = []
                for source in current_sources:
                    # Search for citations and references
                    query = f"{source} citations references cited by"
                    results = await self.tools.search_documents(ctx, query, top_k=5)
                    next_sources.extend(results)
                    traced_sources.extend(results)
                
                current_sources = next_sources[:10]  # Limit breadth
                if not current_sources:
                    break
            
            self.source_count += len(traced_sources)
            return traced_sources
        
        @registry.register("track_word_count", "Track current word count progress")
        async def track_word_count(ctx, section_content: str) -> Dict[str, int]:
            """Track word count for depth monitoring."""
            words = len(section_content.split())
            self.current_word_count += words
            
            return {
                "section_words": words,
                "total_words": self.current_word_count,
                "remaining_words": max(0, self.min_word_count - self.current_word_count),
                "progress_percentage": min(100, (self.current_word_count / self.min_word_count) * 100)
            }
    
    async def conduct_deep_research(
        self,
        topic: str,
        research_questions: Optional[List[str]] = None,
        context: Optional[AgentContext] = None,
        target_depth: str = "comprehensive",  # comprehensive, exhaustive, focused
    ) -> DeepResearchResult:
        """
        Conduct deep research on a topic.
        
        Args:
            topic: Main research topic
            research_questions: Specific questions to address
            context: Agent execution context
            target_depth: Depth level for research
            
        Returns:
            Comprehensive research result
        """
        # Initialize tracking
        self.research_phases = []
        self.source_count = 0
        self.current_word_count = 0
        
        questions_str = ""
        if research_questions:
            questions_str = "Research Questions:\n" + "\n".join(f"- {q}" for q in research_questions)
        
        research_prompt = f"""Conduct comprehensive deep research on: {topic}

{questions_str}

Target Depth: {target_depth}
Minimum Word Count: {self.min_word_count}
Minimum Sources: {self.min_sources}

Follow the Deep Research Methodology:

1. Planning Phase (Verbalize your complete plan):
   - Decompose into 5-7 major themes
   - Identify investigation sequence
   - Estimate depth per section
   - Plan synthesis approach

2. Investigation Phase:
   - Use comprehensive_search for broad coverage
   - Use deep_dive_search for each theme
   - Use citation_trace for authoritative sources
   - Track progress with track_word_count

3. Synthesis Phase:
   - Write in academic prose
   - Build narrative progressively
   - Ensure minimum depth is met
   - Include extensive citations

4. Validation Phase:
   - Verify word count requirements
   - Check source diversity
   - Ensure comprehensive coverage
   - Identify any remaining gaps

Remember: This is DEEP research. Be exhaustive, thorough, and comprehensive. 
Every section should provide rich detail and analysis."""
        
        # Add event stream tracking if available
        if context and context.event_stream:
            context.event_stream.add_plan({
                "research_topic": topic,
                "target_depth": target_depth,
                "min_requirements": {
                    "words": self.min_word_count,
                    "sources": self.min_sources
                }
            }, source="DeepResearchAgent")
        
        # Execute research
        response = await self.run(research_prompt, context)
        
        # Track completion
        if response.success and isinstance(response.result, DeepResearchResult):
            self.research_phases = response.result.research_phases
            
            # Log research metrics
            logger.info(
                f"Deep research completed: {response.result.total_word_count} words, "
                f"{response.result.sources_consulted} sources consulted"
            )
        
        return response.result
    
    async def research_with_outline(
        self,
        topic: str,
        outline: Dict[str, List[str]],
        context: Optional[AgentContext] = None,
    ) -> DeepResearchResult:
        """
        Conduct research following a specific outline.
        
        Args:
            topic: Research topic
            outline: Section titles mapped to subtopics
            context: Agent execution context
            
        Returns:
            Research result following outline
        """
        outline_str = "Research Outline:\n"
        for section, subtopics in outline.items():
            outline_str += f"\n{section}:\n"
            for subtopic in subtopics:
                outline_str += f"  - {subtopic}\n"
        
        research_prompt = f"""Conduct deep research on: {topic}

Follow this specific outline:
{outline_str}

Requirements:
- Minimum {self.min_word_count} total words
- Each major section: 1500-2500 words
- Each subtopic: 500-1000 words
- Academic prose throughout
- Extensive citations from {self.min_sources}+ sources

Use your research tools to gather comprehensive information for each section and subtopic."""
        
        response = await self.run(research_prompt, context)
        return response.result
    
    async def iterative_deepening(
        self,
        topic: str,
        initial_result: DeepResearchResult,
        focus_areas: List[str],
        context: Optional[AgentContext] = None,
    ) -> DeepResearchResult:
        """
        Deepen existing research in specific areas.
        
        Args:
            topic: Original topic
            initial_result: Previous research result
            focus_areas: Areas to explore more deeply
            context: Agent execution context
            
        Returns:
            Enhanced research result
        """
        current_state = f"""
Current Research State:
- Total Words: {initial_result.total_word_count}
- Sources: {initial_result.sources_consulted}
- Sections: {len(initial_result.sections)}
- Gaps Identified: {len(initial_result.knowledge_gaps)}
"""
        
        deepening_prompt = f"""Deepen the existing research on: {topic}

{current_state}

Focus on these areas for deeper exploration:
{chr(10).join(f'- {area}' for area in focus_areas)}

Previous gaps identified:
{chr(10).join(f'- {gap}' for gap in initial_result.knowledge_gaps[:5])}

Requirements:
- Add at least 3000 more words
- Consult 10+ additional sources
- Maintain academic prose quality
- Address identified gaps
- Deepen analysis in focus areas

Build upon the existing research to create an even more comprehensive analysis."""
        
        response = await self.run(deepening_prompt, context)
        return response.result


def create_deep_research_agent(
    tools: Optional[AgentTools] = None,
    min_word_count: int = 10000,
    min_sources: int = 20,
) -> DeepResearchAgent:
    """Create a deep research agent with specified requirements."""
    return DeepResearchAgent(
        tools=tools,
        min_word_count=min_word_count,
        min_sources=min_sources,
    )