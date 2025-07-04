"""
Shared tools for agents.

This module provides reusable tools that agents can use to interact
with the PDF RAG system, including search, retrieval, and analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.models import PDFChunk, GraphEntity
from src.rag.hybrid_retriever import HybridGraphRetriever
from src.rag.pipeline import RAGPipeline
from src.graph_db.base import GraphDatabase
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SearchResult(BaseModel):
    """Result from a search operation."""
    
    chunks: List[Dict[str, Any]] = Field(..., description="Retrieved chunks")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Related entities")
    confidence: float = Field(..., description="Search confidence score")


class GraphQueryResult(BaseModel):
    """Result from a graph query."""
    
    entities: List[Dict[str, Any]] = Field(..., description="Graph entities")
    relationships: List[Dict[str, Any]] = Field(..., description="Relationships")
    paths: List[List[str]] = Field(default_factory=list, description="Paths found")


class AgentTools:
    """
    Collection of tools available to agents.
    
    These tools provide access to:
    - Vector and hybrid search
    - Knowledge graph queries
    - Document analysis
    - Entity extraction
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        retriever: Optional[HybridGraphRetriever] = None,
        graph_db: Optional[GraphDatabase] = None,
    ):
        """Initialize agent tools."""
        self.rag_pipeline = rag_pipeline
        self.retriever = retriever
        self.graph_db = graph_db
        self._initialized = False

    async def initialize(self):
        """Initialize tools with necessary components."""
        if self._initialized:
            return
        
        # Initialize components if not provided
        if not self.rag_pipeline:
            from src.rag.pipeline import create_rag_pipeline
            self.rag_pipeline = create_rag_pipeline()
            await self.rag_pipeline.initialize()
        
        if not self.retriever:
            from src.rag.hybrid_retriever import HybridGraphRetriever
            self.retriever = HybridGraphRetriever()
            await self.retriever.initialize()
        
        if not self.graph_db:
            from src.config import get_settings
            settings = get_settings()
            if settings.enable_knowledge_graph:
                from src.graph_db.base import GraphDatabaseFactory
                self.graph_db = GraphDatabaseFactory.create("neo4j")
                await self.graph_db.initialize()
        
        self._initialized = True
        logger.info("Agent tools initialized")

    async def search_documents(
        self,
        ctx: RunContext[Any],
        query: str,
        top_k: int = 5,
        use_graph: bool = True,
    ) -> SearchResult:
        """
        Search for relevant documents.
        
        Args:
            ctx: Run context
            query: Search query
            top_k: Number of results
            use_graph: Whether to use graph enhancement
            
        Returns:
            Search results
        """
        await self.initialize()
        
        try:
            # Perform search
            chunks_with_scores = await self.retriever.retrieve_chunks(
                query=query,
                top_k=top_k,
                use_graph=use_graph,
            )
            
            # Format chunks
            chunks = []
            for chunk, score in chunks_with_scores:
                chunks.append({
                    "id": chunk.id,
                    "content": chunk.content,
                    "score": score,
                    "metadata": {
                        "pdf_id": chunk.pdf_id,
                        "pages": chunk.page_numbers,
                        "section": chunk.section_title,
                    },
                })
            
            # Get related entities if graph is available
            entities = []
            if use_graph and self.graph_db:
                entity_objs = await self._get_entities_from_chunks(
                    [chunk for chunk, _ in chunks_with_scores[:3]]
                )
                for entity in entity_objs:
                    entities.append({
                        "name": entity.name,
                        "type": entity.type.value,
                        "properties": entity.properties,
                    })
            
            # Calculate confidence
            confidence = sum(score for _, score in chunks_with_scores[:3]) / min(3, len(chunks_with_scores))
            
            return SearchResult(
                chunks=chunks,
                entities=entities,
                confidence=confidence,
            )
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return SearchResult(chunks=[], confidence=0.0)

    async def query_knowledge_graph(
        self,
        ctx: RunContext[Any],
        query: str,
        entity_names: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> GraphQueryResult:
        """
        Query the knowledge graph.
        
        Args:
            ctx: Run context
            query: Natural language query
            entity_names: Specific entities to query
            max_hops: Maximum graph traversal depth
            
        Returns:
            Graph query results
        """
        await self.initialize()
        
        if not self.graph_db:
            return GraphQueryResult(entities=[], relationships=[])
        
        try:
            entities = []
            relationships = []
            paths = []
            
            # Search for entities if names provided
            if entity_names:
                for name in entity_names[:5]:  # Limit
                    entity_list = await self.graph_db.search_entities(
                        query=name,
                        limit=1,
                    )
                    if entity_list:
                        entity = entity_list[0]
                        entities.append({
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.type.value,
                            "properties": entity.properties,
                        })
                        
                        # Get relationships
                        rels = await self.graph_db.get_relationships(
                            entity_id=entity.id,
                            direction="both",
                        )
                        for rel in rels[:5]:
                            relationships.append({
                                "source": rel.source_id,
                                "target": rel.target_id,
                                "type": rel.type.value,
                                "properties": rel.properties,
                            })
            else:
                # General entity search
                entity_list = await self.graph_db.search_entities(
                    query=query,
                    limit=10,
                )
                for entity in entity_list:
                    entities.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type.value,
                        "properties": entity.properties,
                    })
            
            # Find paths between entities if multiple found
            if len(entities) >= 2:
                path = await self.graph_db.find_path(
                    entities[0]["id"],
                    entities[1]["id"],
                    max_hops=max_hops,
                )
                if path:
                    path_names = []
                    for i, element in enumerate(path):
                        if i % 2 == 0:  # Entity
                            path_names.append(element.name)
                    paths.append(path_names)
            
            return GraphQueryResult(
                entities=entities,
                relationships=relationships,
                paths=paths,
            )
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return GraphQueryResult(entities=[], relationships=[])

    async def analyze_document_section(
        self,
        ctx: RunContext[Any],
        pdf_id: str,
        page_numbers: List[int],
        analysis_type: str = "summary",
    ) -> str:
        """
        Analyze a specific section of a document.
        
        Args:
            ctx: Run context
            pdf_id: PDF identifier
            page_numbers: Pages to analyze
            analysis_type: Type of analysis (summary, entities, questions)
            
        Returns:
            Analysis result
        """
        await self.initialize()
        
        try:
            # Retrieve chunks for specified pages
            chunks_with_scores = await self.retriever.retrieve_chunks(
                query=f"pdf_id:{pdf_id} pages:{page_numbers}",
                top_k=20,
                filters={
                    "pdf_id": pdf_id,
                    "page_numbers": {"$in": page_numbers},
                },
                use_graph=False,
            )
            
            if not chunks_with_scores:
                return "No content found for specified pages."
            
            # Combine chunk content
            combined_text = "\n\n".join(
                chunk.content for chunk, _ in chunks_with_scores
            )
            
            # Perform analysis based on type
            if analysis_type == "summary":
                # Use RAG pipeline to generate summary
                from src.rag.generator import AnswerGenerator
                generator = AnswerGenerator()
                summary = await generator.generate_summary(
                    [chunk for chunk, _ in chunks_with_scores],
                    max_length=500,
                )
                return summary
                
            elif analysis_type == "entities":
                # Extract entities
                if self.graph_db:
                    entities = await self._extract_entities_from_text(combined_text)
                    entity_names = [e.name for e in entities[:10]]
                    return f"Found entities: {', '.join(entity_names)}"
                return "Entity extraction not available without graph database."
                
            elif analysis_type == "questions":
                # Generate questions about the content
                return f"Questions about pages {page_numbers}: What are the main topics? What conclusions are drawn?"
                
            else:
                return f"Unknown analysis type: {analysis_type}"
                
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return f"Analysis failed: {str(e)}"

    async def find_related_documents(
        self,
        ctx: RunContext[Any],
        reference_chunk_id: str,
        similarity_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find documents related to a reference chunk.
        
        Args:
            ctx: Run context
            reference_chunk_id: ID of reference chunk
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of related documents
        """
        await self.initialize()
        
        try:
            # Get reference chunk
            reference_results = await self.retriever.vector_db.search(
                query_embedding=[0.0] * self.retriever.embedding_generator.dimension,
                top_k=1,
                filters={"id": reference_chunk_id},
            )
            
            if not reference_results:
                return []
            
            reference_doc = reference_results[0].document
            
            # Find similar chunks
            if reference_doc.embedding:
                similar_results = await self.retriever.vector_db.search(
                    query_embedding=reference_doc.embedding,
                    top_k=10,
                    filters={"id": {"$ne": reference_chunk_id}},
                )
                
                related = []
                for result in similar_results:
                    if result.score >= similarity_threshold:
                        related.append({
                            "chunk_id": result.document.id,
                            "content_preview": result.document.content[:200] + "...",
                            "similarity": result.score,
                            "pdf_id": result.document.metadata.get("pdf_id"),
                            "pages": result.document.metadata.get("page_numbers", []),
                        })
                
                return related
            
            return []
            
        except Exception as e:
            logger.error(f"Finding related documents failed: {e}")
            return []

    async def explain_connection(
        self,
        ctx: RunContext[Any],
        entity1: str,
        entity2: str,
    ) -> str:
        """
        Explain the connection between two entities.
        
        Args:
            ctx: Run context
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            Explanation of connection
        """
        await self.initialize()
        
        if not self.graph_db:
            return "Graph database not available for connection analysis."
        
        try:
            # Find entities
            entities1 = await self.graph_db.search_entities(entity1, limit=1)
            entities2 = await self.graph_db.search_entities(entity2, limit=1)
            
            if not entities1 or not entities2:
                return f"Could not find both entities: {entity1} and {entity2}"
            
            e1, e2 = entities1[0], entities2[0]
            
            # Find path
            path = await self.graph_db.find_path(
                e1.id,
                e2.id,
                max_hops=4,
            )
            
            if not path:
                return f"No direct connection found between {entity1} and {entity2}"
            
            # Build explanation
            explanation = f"Connection between {entity1} and {entity2}:\n"
            
            for i in range(0, len(path) - 1, 2):
                entity = path[i]
                if i + 1 < len(path):
                    relationship = path[i + 1]
                    next_entity = path[i + 2] if i + 2 < len(path) else None
                    
                    explanation += f"- {entity.name} "
                    explanation += f"{relationship.type.value.replace('_', ' ')} "
                    if next_entity:
                        explanation += f"{next_entity.name}\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Connection explanation failed: {e}")
            return f"Failed to explain connection: {str(e)}"

    async def _get_entities_from_chunks(
        self,
        chunks: List[PDFChunk],
    ) -> List[GraphEntity]:
        """Get entities associated with chunks."""
        if not self.graph_db:
            return []
        
        entities = []
        for chunk in chunks:
            # Simple approach - in production, use more sophisticated query
            chunk_entities = await self.graph_db.execute_cypher(
                """
                MATCH (e:Entity)
                WHERE $chunk_id IN e.source_chunk_ids
                RETURN e
                LIMIT 5
                """,
                {"chunk_id": chunk.id},
            )
            
            # Convert to GraphEntity objects
            # This is simplified - implement proper conversion
            entities.extend(chunk_entities)
        
        return entities

    async def _extract_entities_from_text(
        self,
        text: str,
    ) -> List[GraphEntity]:
        """Extract entities from text using NLP."""
        # Simplified version - in production, use proper NER
        from src.knowledge_graph.extractor import KnowledgeExtractor
        
        extractor = KnowledgeExtractor()
        # Create a dummy chunk for extraction
        chunk = PDFChunk(
            id="temp",
            pdf_id="temp",
            content=text,
            page_numbers=[1],
            chunk_index=0,
            word_count=len(text.split()),
            char_count=len(text),
        )
        
        entities, _ = await extractor.extract_from_chunk(chunk)
        return entities


# Tool function decorators for use with Pydantic AI agents
def search_tool(description: str = "Search for relevant documents"):
    """Decorator to create a search tool."""
    def decorator(func):
        async def tool_wrapper(
            ctx: RunContext[Any],
            query: str,
            top_k: int = 5,
        ) -> SearchResult:
            tools: AgentTools = ctx.deps.get("tools")
            if not tools:
                raise ValueError("AgentTools not found in context")
            return await tools.search_documents(ctx, query, top_k)
        
        tool_wrapper.__name__ = func.__name__
        tool_wrapper.__doc__ = description
        return tool_wrapper
    
    return decorator


def graph_tool(description: str = "Query the knowledge graph"):
    """Decorator to create a graph query tool."""
    def decorator(func):
        async def tool_wrapper(
            ctx: RunContext[Any],
            query: str,
            entities: Optional[List[str]] = None,
        ) -> GraphQueryResult:
            tools: AgentTools = ctx.deps.get("tools")
            if not tools:
                raise ValueError("AgentTools not found in context")
            return await tools.query_knowledge_graph(ctx, query, entities)
        
        tool_wrapper.__name__ = func.__name__
        tool_wrapper.__doc__ = description
        return tool_wrapper
    
    return decorator