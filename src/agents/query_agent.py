"""
Query agent for intelligent information retrieval.

This agent handles complex queries by combining vector search,
graph traversal, and multi-step reasoning.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.agents.base_agent import BaseAgent, AgentContext, ToolRegistry
from src.agents.prompts import QUERY_AGENT_PROMPT
from src.agents.tools import AgentTools, SearchResult, GraphQueryResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryPlan(BaseModel):
    """Plan for executing a complex query."""
    
    main_query: str = Field(..., description="The main query to answer")
    sub_queries: List[str] = Field(..., description="Sub-queries to explore")
    entities_to_find: List[str] = Field(default_factory=list, description="Key entities to search for")
    use_graph: bool = Field(True, description="Whether to use graph search")
    reasoning_steps: List[str] = Field(default_factory=list, description="Reasoning steps taken")


class QueryResult(BaseModel):
    """Structured result from query agent."""
    
    answer: str = Field(..., description="The main answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[Dict[str, Any]] = Field(..., description="Source citations")
    entities_found: List[Dict[str, Any]] = Field(default_factory=list, description="Entities discovered")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")


class QueryAgent(BaseAgent[QueryResult]):
    """
    Agent specialized in answering queries using hybrid retrieval.
    
    This agent can:
    - Decompose complex queries into sub-questions
    - Search both vector index and knowledge graph
    - Perform multi-hop reasoning
    - Provide detailed source citations
    """

    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """Initialize query agent."""
        self.tools = tools or AgentTools()
        
        super().__init__(
            name="QueryAgent",
            model=model or "gpt-4-turbo-preview",
            system_prompt=QUERY_AGENT_PROMPT,
            response_model=QueryResult,
            temperature=temperature,
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return QUERY_AGENT_PROMPT

    def _register_tools(self) -> None:
        """Register available tools."""
        registry = ToolRegistry(self._agent)
        
        @registry.register("search_documents", "Search for relevant documents using vector and graph search")
        async def search_documents(
            ctx: RunContext[Any],
            query: str,
            top_k: int = 5,
            use_graph: bool = True,
        ) -> SearchResult:
            """Search for documents relevant to the query."""
            return await self.tools.search_documents(ctx, query, top_k, use_graph)
        
        @registry.register("query_graph", "Query the knowledge graph for entities and relationships")
        async def query_graph(
            ctx: RunContext[Any],
            query: str,
            entity_names: Optional[List[str]] = None,
        ) -> GraphQueryResult:
            """Query the knowledge graph."""
            return await self.tools.query_knowledge_graph(ctx, query, entity_names)
        
        @registry.register("analyze_section", "Analyze a specific document section")
        async def analyze_section(
            ctx: RunContext[Any],
            pdf_id: str,
            pages: List[int],
            analysis_type: str = "summary",
        ) -> str:
            """Analyze a document section."""
            return await self.tools.analyze_document_section(ctx, pdf_id, pages, analysis_type)
        
        @registry.register("find_connections", "Find connections between entities")
        async def find_connections(
            ctx: RunContext[Any],
            entity1: str,
            entity2: str,
        ) -> str:
            """Explain connection between two entities."""
            return await self.tools.explain_connection(ctx, entity1, entity2)

    async def _initialize_components(self) -> None:
        """Initialize agent components."""
        await self.tools.initialize()

    async def plan_query(
        self,
        query: str,
        context: Optional[AgentContext] = None,
    ) -> QueryPlan:
        """
        Create a plan for answering a complex query.
        
        Args:
            query: The query to plan for
            context: Execution context
            
        Returns:
            Query execution plan
        """
        planning_prompt = f"""Analyze this query and create an execution plan:

Query: {query}

Break it down into:
1. Main question to answer
2. Sub-questions that need to be explored
3. Key entities to search for
4. Whether graph search would be helpful

Think step by step."""

        response = await self.run(planning_prompt, context)
        
        if response.success and isinstance(response.result, QueryResult):
            # Extract plan from response
            plan = QueryPlan(
                main_query=query,
                sub_queries=response.result.follow_up_questions[:3],
                entities_to_find=[],  # Would be extracted from response
                use_graph=True,
                reasoning_steps=response.reasoning,
            )
            return plan
        
        # Fallback plan
        return QueryPlan(
            main_query=query,
            sub_queries=[query],
            use_graph=True,
        )

    async def answer_with_reasoning(
        self,
        query: str,
        context: Optional[AgentContext] = None,
        max_iterations: int = 3,
    ) -> QueryResult:
        """
        Answer a query with explicit reasoning steps.
        
        Args:
            query: The query to answer
            context: Execution context
            max_iterations: Maximum reasoning iterations
            
        Returns:
            Query result with reasoning trace
        """
        context = context or AgentContext()
        
        # Create query plan
        plan = await self.plan_query(query, context)
        
        # Execute plan iteratively
        all_sources = []
        all_entities = []
        reasoning_trace = [f"Query: {query}"]
        
        for i, sub_query in enumerate(plan.sub_queries[:max_iterations]):
            reasoning_trace.append(f"Step {i+1}: Exploring '{sub_query}'")
            
            # Search for relevant information
            search_result = await self.tools.search_documents(
                RunContext(deps={"tools": self.tools}),
                sub_query,
                top_k=3,
                use_graph=plan.use_graph,
            )
            
            # Collect sources and entities
            all_sources.extend(search_result.chunks)
            all_entities.extend(search_result.entities)
            
            # Check if we have enough information
            if search_result.confidence > 0.8:
                reasoning_trace.append("Found high-confidence information")
                break
        
        # Generate final answer with all collected information
        final_prompt = f"""Based on the following information, answer the query:

Query: {query}

Information found:
{self._format_search_results(all_sources)}

Entities involved:
{self._format_entities(all_entities)}

Provide a comprehensive answer with confidence score and source citations."""

        response = await self.run(final_prompt, context)
        
        if response.success and isinstance(response.result, QueryResult):
            # Add reasoning trace
            response.result.sources = self._deduplicate_sources(all_sources)
            response.result.entities_found = self._deduplicate_entities(all_entities)
            return response.result
        
        # Fallback response
        return QueryResult(
            answer="Unable to find sufficient information to answer the query.",
            confidence=0.0,
            sources=[],
        )

    async def answer_comparative(
        self,
        query: str,
        entities: List[str],
        context: Optional[AgentContext] = None,
    ) -> QueryResult:
        """
        Answer a comparative query about multiple entities.
        
        Args:
            query: The comparative query
            entities: Entities to compare
            context: Execution context
            
        Returns:
            Comparative analysis result
        """
        context = context or AgentContext()
        
        # Gather information about each entity
        entity_info = {}
        
        for entity in entities[:5]:  # Limit to 5 entities
            # Search for entity information
            search_result = await self.tools.search_documents(
                RunContext(deps={"tools": self.tools}),
                entity,
                top_k=3,
                use_graph=True,
            )
            
            # Query graph for relationships
            graph_result = await self.tools.query_knowledge_graph(
                RunContext(deps={"tools": self.tools}),
                entity,
                entity_names=[entity],
            )
            
            entity_info[entity] = {
                "documents": search_result.chunks,
                "graph_data": graph_result.entities,
                "relationships": graph_result.relationships,
            }
        
        # Generate comparative analysis
        comparative_prompt = f"""Compare and contrast the following entities based on the query:

Query: {query}
Entities: {', '.join(entities)}

Information gathered:
{self._format_entity_comparison(entity_info)}

Provide a structured comparison addressing the query."""

        response = await self.run(comparative_prompt, context)
        
        return response.result if response.success else QueryResult(
            answer="Unable to perform comparison.",
            confidence=0.0,
            sources=[],
        )

    def _format_search_results(self, chunks: List[Dict[str, Any]]) -> str:
        """Format search results for prompt."""
        formatted = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5
            formatted.append(f"""
Source {i+1} (Score: {chunk.get('score', 0):.2f}):
Content: {chunk['content'][:500]}...
PDF: {chunk['metadata'].get('pdf_id', 'Unknown')}
Pages: {chunk['metadata'].get('pages', [])}
""")
        return "\n".join(formatted)

    def _format_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Format entities for prompt."""
        if not entities:
            return "No specific entities identified."
        
        formatted = []
        for entity in entities[:10]:  # Limit to 10 entities
            formatted.append(
                f"- {entity['name']} ({entity['type']}): "
                f"{entity.get('properties', {}).get('description', 'No description')}"
            )
        
        return "\n".join(formatted)

    def _format_entity_comparison(self, entity_info: Dict[str, Any]) -> str:
        """Format entity comparison data."""
        formatted = []
        
        for entity, info in entity_info.items():
            formatted.append(f"\n{entity}:")
            
            # Add document info
            if info["documents"]:
                formatted.append("  Documents mentioning this entity:")
                for doc in info["documents"][:2]:
                    formatted.append(f"    - {doc['content'][:100]}...")
            
            # Add graph info
            if info["graph_data"]:
                formatted.append("  Graph properties:")
                for prop in info["graph_data"][:3]:
                    formatted.append(f"    - {prop}")
            
            # Add relationships
            if info["relationships"]:
                formatted.append("  Relationships:")
                for rel in info["relationships"][:3]:
                    formatted.append(f"    - {rel['type']}: {rel.get('target', 'Unknown')}")
        
        return "\n".join(formatted)

    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate source citations."""
        seen = set()
        unique = []
        
        for source in sources:
            key = source.get("id", str(source))
            if key not in seen:
                seen.add(key)
                unique.append(source)
        
        return unique

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities."""
        seen = set()
        unique = []
        
        for entity in entities:
            key = (entity.get("name", ""), entity.get("type", ""))
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        
        return unique