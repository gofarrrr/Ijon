"""
Research agent for in-depth analysis and investigation.

This agent conducts comprehensive research by iteratively exploring
topics, following connections, and synthesizing findings.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.agents.base_agent import BaseAgent, AgentContext, ToolRegistry
from src.agents.prompts import RESEARCH_AGENT_PROMPT
from src.agents.tools import AgentTools
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ResearchTopic(BaseModel):
    """A research topic to explore."""
    
    topic: str = Field(..., description="The topic to research")
    questions: List[str] = Field(..., description="Specific questions to answer")
    entities: List[str] = Field(default_factory=list, description="Key entities involved")
    explored: bool = Field(False, description="Whether this topic has been explored")


class ResearchFindings(BaseModel):
    """Findings from research investigation."""
    
    summary: str = Field(..., description="Executive summary of findings")
    key_insights: List[str] = Field(..., description="Key insights discovered")
    evidence: List[Dict[str, Any]] = Field(..., description="Supporting evidence with sources")
    entities_analyzed: List[Dict[str, Any]] = Field(..., description="Entities examined")
    relationships_found: List[Dict[str, Any]] = Field(..., description="Relationships discovered")
    gaps_identified: List[str] = Field(default_factory=list, description="Information gaps")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for further research")


class ResearchAgent(BaseAgent[ResearchFindings]):
    """
    Agent specialized in conducting deep research investigations.
    
    This agent can:
    - Develop research plans with multiple angles
    - Iteratively explore topics and follow leads
    - Analyze entity relationships and patterns
    - Synthesize findings from multiple sources
    - Identify information gaps and contradictions
    """

    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.4,
        max_research_depth: int = 5,
    ):
        """Initialize research agent."""
        self.tools = tools or AgentTools()
        self.max_research_depth = max_research_depth
        
        super().__init__(
            name="ResearchAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=RESEARCH_AGENT_PROMPT,
            response_model=ResearchFindings,
            temperature=temperature,
        )
        
        # Research state
        self._research_topics: List[ResearchTopic] = []
        self._explored_entities: Set[str] = set()
        self._findings_cache: Dict[str, Any] = {}

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return RESEARCH_AGENT_PROMPT

    def _register_tools(self) -> None:
        """Register available tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("search_documents", "Search for relevant documents")
        async def search_documents(
            ctx: RunContext[Any],
            query: str,
            top_k: int = 10,
        ) -> Any:
            """Search for documents."""
            return await self.tools.search_documents(ctx, query, top_k, use_graph=True)
        
        @registry.register("explore_entity", "Explore an entity in depth")
        async def explore_entity(
            ctx: RunContext[Any],
            entity_name: str,
        ) -> Any:
            """Explore entity relationships and mentions."""
            # Search for entity
            search_result = await self.tools.search_documents(ctx, entity_name, top_k=5)
            
            # Get graph data
            graph_result = await self.tools.query_knowledge_graph(
                ctx, entity_name, entity_names=[entity_name]
            )
            
            return {
                "entity": entity_name,
                "documents": search_result.chunks,
                "graph_data": graph_result.entities,
                "relationships": graph_result.relationships,
            }
        
        @registry.register("trace_connections", "Trace connections between entities")
        async def trace_connections(
            ctx: RunContext[Any],
            entities: List[str],
        ) -> Any:
            """Find connections between multiple entities."""
            connections = []
            
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    connection = await self.tools.explain_connection(
                        ctx, entity1, entity2
                    )
                    connections.append({
                        "from": entity1,
                        "to": entity2,
                        "connection": connection,
                    })
            
            return connections
        
        @registry.register("analyze_topic_coverage", "Analyze how well a topic is covered")
        async def analyze_topic_coverage(
            ctx: RunContext[Any],
            topic: str,
            subtopics: List[str],
        ) -> Any:
            """Analyze coverage of a topic and its subtopics."""
            coverage = {"topic": topic, "subtopics": {}}
            
            for subtopic in subtopics:
                search_result = await self.tools.search_documents(
                    ctx, f"{topic} {subtopic}", top_k=3
                )
                coverage["subtopics"][subtopic] = {
                    "found": len(search_result.chunks) > 0,
                    "confidence": search_result.confidence,
                    "sources": len(search_result.chunks),
                }
            
            return coverage

    async def _initialize_components(self) -> None:
        """Initialize agent components."""
        await self.tools.initialize()

    async def develop_research_plan(
        self,
        main_topic: str,
        context: Optional[AgentContext] = None,
    ) -> List[ResearchTopic]:
        """
        Develop a comprehensive research plan.
        
        Args:
            main_topic: The main research topic
            context: Execution context
            
        Returns:
            List of research topics to explore
        """
        planning_prompt = f"""Develop a comprehensive research plan for investigating:

Topic: {main_topic}

Create a structured plan that includes:
1. Main research questions
2. Subtopics to explore
3. Key entities to investigate
4. Potential connections to examine
5. Different perspectives to consider

Think systematically about how to thoroughly investigate this topic."""

        response = await self.run(planning_prompt, context)
        
        # Extract topics from response
        if response.success and hasattr(response.result, 'recommendations'):
            # Parse recommendations into research topics
            topics = []
            
            # Add main topic
            topics.append(ResearchTopic(
                topic=main_topic,
                questions=[
                    f"What is the overview of {main_topic}?",
                    f"What are the key aspects of {main_topic}?",
                    f"How does {main_topic} relate to other topics?",
                ],
            ))
            
            # Add subtopics from recommendations
            for rec in response.result.recommendations[:5]:
                topics.append(ResearchTopic(
                    topic=rec,
                    questions=[f"How does {rec} relate to {main_topic}?"],
                ))
            
            self._research_topics = topics
            return topics
        
        # Fallback plan
        return [ResearchTopic(
            topic=main_topic,
            questions=["What information is available about this topic?"],
        )]

    async def conduct_research(
        self,
        topic: str,
        context: Optional[AgentContext] = None,
        depth: int = 3,
    ) -> ResearchFindings:
        """
        Conduct in-depth research on a topic.
        
        Args:
            topic: The topic to research
            context: Execution context
            depth: Maximum research depth
            
        Returns:
            Research findings
        """
        context = context or AgentContext()
        
        # Develop research plan
        research_plan = await self.develop_research_plan(topic, context)
        
        # Initialize findings collection
        all_evidence = []
        all_entities = []
        all_relationships = []
        key_insights = []
        
        # Execute research plan iteratively
        for iteration in range(min(depth, len(research_plan))):
            current_topic = research_plan[iteration]
            
            if current_topic.explored:
                continue
            
            logger.info(f"Researching: {current_topic.topic}")
            
            # Search for information
            for question in current_topic.questions:
                search_result = await self.tools.search_documents(
                    RunContext(deps={"tools": self.tools}),
                    question,
                    top_k=5,
                )
                
                # Collect evidence
                for chunk in search_result.chunks:
                    all_evidence.append({
                        "content": chunk["content"],
                        "source": chunk["metadata"],
                        "relevance": chunk["score"],
                        "question": question,
                    })
                
                # Collect entities
                all_entities.extend(search_result.entities)
            
            # Mark as explored
            current_topic.explored = True
            
            # Explore interesting entities
            for entity in current_topic.entities[:3]:
                if entity not in self._explored_entities:
                    self._explored_entities.add(entity)
                    
                    entity_data = await self._explore_entity_in_depth(
                        entity, context
                    )
                    
                    # Add to findings
                    all_entities.extend(entity_data.get("related_entities", []))
                    all_relationships.extend(entity_data.get("relationships", []))
            
            # Extract insights from current findings
            insights = await self._extract_insights(
                current_topic.topic,
                all_evidence[-10:],  # Recent evidence
                context,
            )
            key_insights.extend(insights)
        
        # Synthesize final findings
        findings = await self._synthesize_findings(
            topic,
            all_evidence,
            all_entities,
            all_relationships,
            key_insights,
            context,
        )
        
        return findings

    async def investigate_question(
        self,
        question: str,
        context: Optional[AgentContext] = None,
        follow_leads: bool = True,
    ) -> ResearchFindings:
        """
        Investigate a specific research question.
        
        Args:
            question: The question to investigate
            context: Execution context
            follow_leads: Whether to follow up on interesting findings
            
        Returns:
            Research findings
        """
        context = context or AgentContext()
        
        # Initial investigation
        initial_search = await self.tools.search_documents(
            RunContext(deps={"tools": self.tools}),
            question,
            top_k=10,
        )
        
        # Analyze initial findings
        evidence = []
        entities_to_explore = []
        
        for chunk in initial_search.chunks:
            evidence.append({
                "content": chunk["content"],
                "source": chunk["metadata"],
                "relevance": chunk["score"],
            })
        
        for entity in initial_search.entities:
            entities_to_explore.append(entity["name"])
        
        # Follow leads if enabled
        additional_evidence = []
        relationships = []
        
        if follow_leads and entities_to_explore:
            # Explore top entities
            for entity_name in entities_to_explore[:3]:
                entity_data = await self._explore_entity_in_depth(
                    entity_name, context
                )
                
                # Look for additional relevant information
                for doc in entity_data.get("documents", []):
                    if self._is_relevant_to_question(doc["content"], question):
                        additional_evidence.append({
                            "content": doc["content"],
                            "source": doc["metadata"],
                            "via_entity": entity_name,
                        })
                
                relationships.extend(entity_data.get("relationships", []))
        
        # Combine all evidence
        all_evidence = evidence + additional_evidence
        
        # Generate findings
        prompt = f"""Based on the investigation of this question:

Question: {question}

Evidence found:
{self._format_evidence(all_evidence[:10])}

Entities involved:
{', '.join(entities_to_explore[:10])}

Relationships discovered:
{self._format_relationships(relationships[:5])}

Provide comprehensive research findings including:
1. Direct answer to the question
2. Key insights discovered
3. Supporting evidence
4. Any gaps or contradictions found
5. Recommendations for further investigation"""

        response = await self.run(prompt, context)
        
        if response.success:
            return response.result
        
        # Fallback findings
        return ResearchFindings(
            summary=f"Investigation of: {question}",
            key_insights=["Limited information found"],
            evidence=all_evidence[:5],
            entities_analyzed=initial_search.entities,
            relationships_found=relationships,
            gaps_identified=["Insufficient data for comprehensive analysis"],
        )

    async def _explore_entity_in_depth(
        self,
        entity_name: str,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """Explore an entity in depth."""
        # Check cache
        if entity_name in self._findings_cache:
            return self._findings_cache[entity_name]
        
        # Search for entity information
        search_result = await self.tools.search_documents(
            RunContext(deps={"tools": self.tools}),
            entity_name,
            top_k=5,
        )
        
        # Get graph information
        graph_result = await self.tools.query_knowledge_graph(
            RunContext(deps={"tools": self.tools}),
            entity_name,
            entity_names=[entity_name],
        )
        
        # Compile findings
        findings = {
            "entity": entity_name,
            "documents": search_result.chunks,
            "related_entities": search_result.entities,
            "relationships": graph_result.relationships,
            "graph_properties": graph_result.entities[0] if graph_result.entities else {},
        }
        
        # Cache findings
        self._findings_cache[entity_name] = findings
        
        return findings

    async def _extract_insights(
        self,
        topic: str,
        evidence: List[Dict[str, Any]],
        context: AgentContext,
    ) -> List[str]:
        """Extract insights from evidence."""
        if not evidence:
            return []
        
        prompt = f"""Extract key insights from this evidence about {topic}:

{self._format_evidence(evidence)}

Identify:
1. Important patterns or trends
2. Surprising findings
3. Connections between different pieces of information
4. Implications of the findings

List the top insights discovered."""

        response = await self.run(prompt, context)
        
        if response.success and hasattr(response.result, 'key_insights'):
            return response.result.key_insights
        
        return []

    async def _synthesize_findings(
        self,
        topic: str,
        evidence: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        insights: List[str],
        context: AgentContext,
    ) -> ResearchFindings:
        """Synthesize all findings into a coherent report."""
        synthesis_prompt = f"""Synthesize comprehensive research findings for:

Topic: {topic}

Key Insights Discovered:
{self._format_list(insights[:10])}

Evidence Collected ({len(evidence)} pieces):
{self._format_evidence(evidence[:5])}

Entities Analyzed ({len(entities)} total):
{self._format_entities_brief(entities[:10])}

Relationships Found ({len(relationships)} total):
{self._format_relationships(relationships[:5])}

Create a comprehensive research report with:
1. Executive summary
2. Key findings and insights
3. Supporting evidence
4. Identified gaps
5. Recommendations for further research"""

        response = await self.run(synthesis_prompt, context)
        
        if response.success:
            return response.result
        
        # Fallback synthesis
        return ResearchFindings(
            summary=f"Research on {topic} examined {len(evidence)} sources",
            key_insights=insights[:5] if insights else ["Limited insights available"],
            evidence=evidence[:10],
            entities_analyzed=entities[:10],
            relationships_found=relationships[:10],
            gaps_identified=["Further investigation needed"],
            recommendations=["Expand search scope", "Examine related topics"],
        )

    def _is_relevant_to_question(self, content: str, question: str) -> bool:
        """Check if content is relevant to a question."""
        # Simple keyword overlap check
        question_terms = set(question.lower().split())
        content_terms = set(content.lower().split())
        
        overlap = len(question_terms & content_terms)
        return overlap >= min(3, len(question_terms) // 2)

    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Format evidence for prompts."""
        formatted = []
        
        for i, item in enumerate(evidence):
            formatted.append(f"""
Evidence {i+1}:
Content: {item.get('content', '')[:200]}...
Source: {item.get('source', {}).get('pdf_id', 'Unknown')}
Relevance: {item.get('relevance', 0):.2f}
""")
        
        return "\n".join(formatted)

    def _format_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """Format relationships for prompts."""
        if not relationships:
            return "No relationships found"
        
        formatted = []
        for rel in relationships:
            formatted.append(
                f"- {rel.get('source', 'Unknown')} "
                f"{rel.get('type', 'relates to')} "
                f"{rel.get('target', 'Unknown')}"
            )
        
        return "\n".join(formatted)

    def _format_entities_brief(self, entities: List[Dict[str, Any]]) -> str:
        """Format entities briefly."""
        if not entities:
            return "No entities identified"
        
        entity_list = []
        for entity in entities:
            entity_list.append(f"- {entity.get('name', 'Unknown')} ({entity.get('type', 'unknown')})")
        
        return "\n".join(entity_list)

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items."""
        return "\n".join(f"- {item}" for item in items)