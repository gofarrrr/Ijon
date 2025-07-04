"""
Hybrid retriever combining vector search with knowledge graph.

This module implements a sophisticated retrieval system that leverages
both vector similarity and graph relationships for enhanced context retrieval.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import get_settings
from src.graph_db.base import GraphDatabase, GraphDatabaseFactory
from src.models import (
    Document,
    GraphEntity,
    GraphRelationship,
    PDFChunk,
    SearchResult,
)
from src.rag.embedder import EmbeddingGenerator
from src.rag.retriever import DocumentRetriever
from src.utils.errors import RetrievalError
from src.utils.logging import get_logger, log_performance
from src.vector_db.base import VectorDatabase

logger = get_logger(__name__)


class HybridGraphRetriever(DocumentRetriever):
    """
    Advanced retriever that combines vector search with graph traversal.
    
    This retriever enhances traditional RAG by:
    - Using graph relationships to find related context
    - Expanding search through entity connections
    - Providing richer context through graph structure
    - Supporting multi-hop reasoning
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        graph_db: Optional[GraphDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        max_graph_hops: int = 2,
        enable_entity_expansion: bool = True,
    ) -> None:
        """
        Initialize hybrid retriever.
        
        Args:
            vector_db: Vector database instance
            graph_db: Graph database instance
            embedding_generator: Embedding generator
            vector_weight: Weight for vector search scores
            graph_weight: Weight for graph-based scores
            max_graph_hops: Maximum hops for graph traversal
            enable_entity_expansion: Whether to expand search through entities
        """
        super().__init__(vector_db, embedding_generator, rerank=True)
        
        self.settings = get_settings()
        self.graph_db = graph_db
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.max_graph_hops = max_graph_hops
        self.enable_entity_expansion = enable_entity_expansion
        
        # Cache for entity embeddings
        self._entity_embedding_cache = {}

    async def initialize(self) -> None:
        """Initialize both vector and graph databases."""
        await super().initialize()
        
        # Initialize graph database if not provided
        if not self.graph_db and self.settings.enable_knowledge_graph:
            self.graph_db = GraphDatabaseFactory.create("neo4j")
            await self.graph_db.initialize()
            logger.info("Graph database initialized for hybrid retrieval")

    @log_performance
    async def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        use_graph: bool = True,
    ) -> List[Tuple[PDFChunk, float]]:
        """
        Retrieve chunks using both vector search and graph traversal.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            min_score: Minimum similarity score
            filters: Metadata filters
            use_graph: Whether to use graph enhancement
            
        Returns:
            List of (chunk, score) tuples
        """
        # Start with vector search
        vector_results = await super().retrieve_chunks(
            query=query,
            top_k=top_k * 2,  # Get more for merging
            min_score=min_score,
            filters=filters,
        )
        
        if not use_graph or not self.graph_db:
            return vector_results[:top_k]
        
        # Enhance with graph search
        enhanced_results = await self._enhance_with_graph(
            query=query,
            vector_results=vector_results,
            top_k=top_k,
        )
        
        return enhanced_results

    async def _enhance_with_graph(
        self,
        query: str,
        vector_results: List[Tuple[PDFChunk, float]],
        top_k: int,
    ) -> List[Tuple[PDFChunk, float]]:
        """Enhance vector results with graph-based context."""
        try:
            # Extract entities from query
            query_entities = await self._extract_query_entities(query)
            
            # Get entities from vector results
            result_entities = await self._get_entities_from_chunks(
                [chunk for chunk, _ in vector_results[:5]]  # Top 5 chunks
            )
            
            # Expand through graph
            expanded_entities = await self._expand_entities(
                query_entities + result_entities
            )
            
            # Find chunks related to expanded entities
            graph_chunks = await self._get_chunks_from_entities(expanded_entities)
            
            # Merge and rerank results
            merged_results = self._merge_results(
                vector_results,
                graph_chunks,
                query_entities,
                top_k,
            )
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Graph enhancement failed: {e}")
            # Fall back to vector results
            return vector_results[:top_k]

    async def _extract_query_entities(self, query: str) -> List[GraphEntity]:
        """Extract entities mentioned in the query."""
        # Use graph database's search capabilities
        entities = await self.graph_db.search_entities(
            query=query,
            limit=5,
        )
        
        logger.debug(f"Extracted {len(entities)} entities from query")
        return entities

    async def _get_entities_from_chunks(
        self,
        chunks: List[PDFChunk],
    ) -> List[GraphEntity]:
        """Get entities associated with chunks."""
        all_entities = []
        
        for chunk in chunks:
            # Query graph for entities linked to this chunk
            query_result = await self.graph_db.execute_cypher(
                """
                MATCH (e:Entity)
                WHERE $chunk_id IN e.source_chunk_ids
                RETURN e
                LIMIT 10
                """,
                {"chunk_id": chunk.id},
            )
            
            for record in query_result:
                node = record["e"]
                entity = self._node_to_entity(node)
                all_entities.append(entity)
        
        # Deduplicate
        unique_entities = {(e.name, e.type): e for e in all_entities}
        return list(unique_entities.values())

    async def _expand_entities(
        self,
        seed_entities: List[GraphEntity],
    ) -> List[GraphEntity]:
        """Expand entities through graph relationships."""
        if not seed_entities:
            return []
        
        expanded = []
        seen_ids = set()
        
        # Get neighbors for each seed entity
        for entity in seed_entities[:10]:  # Limit to prevent explosion
            neighbors = await self.graph_db.get_neighbors(
                entity_id=entity.id,
                hops=self.max_graph_hops,
                relationship_types=["RELATED_TO", "PART_OF", "DESCRIBES"],
            )
            
            for neighbor in neighbors:
                if neighbor.id not in seen_ids:
                    seen_ids.add(neighbor.id)
                    expanded.append(neighbor)
        
        logger.debug(f"Expanded to {len(expanded)} entities through graph")
        return expanded

    async def _get_chunks_from_entities(
        self,
        entities: List[GraphEntity],
    ) -> List[Tuple[PDFChunk, float]]:
        """Get chunks associated with entities."""
        if not entities:
            return []
        
        # Collect all chunk IDs from entities
        chunk_ids = set()
        for entity in entities:
            chunk_ids.update(entity.source_chunk_ids)
        
        if not chunk_ids:
            return []
        
        # Retrieve chunks from vector database
        chunks_with_scores = []
        
        # Query vector DB for these specific chunks
        for chunk_id in list(chunk_ids)[:50]:  # Limit
            results = await self.vector_db.search(
                query_embedding=[0.0] * self.embedding_generator.dimension,  # Dummy
                top_k=1,
                filters={"id": chunk_id},
            )
            
            if results:
                doc = results[0].document
                chunk = self._document_to_chunk(doc)
                # Give graph-based chunks a base score
                chunks_with_scores.append((chunk, 0.7))
        
        return chunks_with_scores

    def _merge_results(
        self,
        vector_results: List[Tuple[PDFChunk, float]],
        graph_chunks: List[Tuple[PDFChunk, float]],
        query_entities: List[GraphEntity],
        top_k: int,
    ) -> List[Tuple[PDFChunk, float]]:
        """Merge and rerank vector and graph results."""
        # Create a map to track best scores
        chunk_scores = {}
        
        # Add vector results with vector weight
        for chunk, score in vector_results:
            chunk_scores[chunk.id] = {
                "chunk": chunk,
                "vector_score": score * self.vector_weight,
                "graph_score": 0.0,
                "entity_overlap": 0.0,
            }
        
        # Add/update with graph results
        for chunk, score in graph_chunks:
            if chunk.id in chunk_scores:
                chunk_scores[chunk.id]["graph_score"] = score * self.graph_weight
            else:
                chunk_scores[chunk.id] = {
                    "chunk": chunk,
                    "vector_score": 0.0,
                    "graph_score": score * self.graph_weight,
                    "entity_overlap": 0.0,
                }
        
        # Calculate entity overlap bonus
        query_entity_names = {e.name.lower() for e in query_entities}
        for chunk_id, data in chunk_scores.items():
            chunk_text = data["chunk"].content.lower()
            overlap_count = sum(
                1 for name in query_entity_names
                if name in chunk_text
            )
            data["entity_overlap"] = min(overlap_count * 0.1, 0.3)
        
        # Calculate final scores
        final_results = []
        for chunk_id, data in chunk_scores.items():
            final_score = (
                data["vector_score"] +
                data["graph_score"] +
                data["entity_overlap"]
            )
            final_results.append((data["chunk"], final_score))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and limit
        seen_ids = set()
        unique_results = []
        for chunk, score in final_results:
            if chunk.id not in seen_ids and score >= 0.3:  # Min threshold
                seen_ids.add(chunk.id)
                unique_results.append((chunk, score))
                if len(unique_results) >= top_k:
                    break
        
        logger.info(
            f"Merged {len(vector_results)} vector and {len(graph_chunks)} "
            f"graph results into {len(unique_results)} final results"
        )
        
        return unique_results

    @log_performance
    async def retrieve_with_reasoning(
        self,
        query: str,
        top_k: int = 5,
        reasoning_steps: int = 3,
    ) -> Tuple[List[Tuple[PDFChunk, float]], List[str]]:
        """
        Retrieve chunks with multi-step reasoning through the graph.
        
        Args:
            query: Initial query
            top_k: Number of final chunks
            reasoning_steps: Number of reasoning iterations
            
        Returns:
            Tuple of (chunks with scores, reasoning trace)
        """
        reasoning_trace = [f"Initial query: {query}"]
        
        # Start with initial retrieval
        current_results = await self.retrieve_chunks(
            query=query,
            top_k=top_k * 2,
            use_graph=True,
        )
        
        # Iterative reasoning
        for step in range(reasoning_steps):
            # Get entities from current results
            current_entities = await self._get_entities_from_chunks(
                [chunk for chunk, _ in current_results[:3]]
            )
            
            if not current_entities:
                break
            
            # Find related questions/gaps
            expansion_query = await self._generate_expansion_query(
                query,
                current_entities,
                current_results[:3],
            )
            
            if not expansion_query:
                break
            
            reasoning_trace.append(f"Step {step + 1}: Exploring '{expansion_query}'")
            
            # Search with expanded query
            new_results = await self.retrieve_chunks(
                query=expansion_query,
                top_k=top_k,
                use_graph=True,
            )
            
            # Merge with previous results
            current_results = self._merge_reasoning_results(
                current_results,
                new_results,
                decay_factor=0.8 ** (step + 1),
            )
        
        # Final ranking
        final_results = current_results[:top_k]
        
        reasoning_trace.append(
            f"Final: Retrieved {len(final_results)} chunks after {len(reasoning_trace) - 1} steps"
        )
        
        return final_results, reasoning_trace

    async def _generate_expansion_query(
        self,
        original_query: str,
        entities: List[GraphEntity],
        current_chunks: List[Tuple[PDFChunk, float]],
    ) -> Optional[str]:
        """Generate an expansion query based on current context."""
        # This is a simplified version - in production, use LLM
        # to generate smarter expansion queries
        
        if not entities:
            return None
        
        # Find entities not well covered in current results
        entity_names = [e.name for e in entities[:3]]
        
        # Simple expansion strategy
        expansion = f"{original_query} specifically about {' and '.join(entity_names)}"
        
        return expansion

    def _merge_reasoning_results(
        self,
        current: List[Tuple[PDFChunk, float]],
        new: List[Tuple[PDFChunk, float]],
        decay_factor: float,
    ) -> List[Tuple[PDFChunk, float]]:
        """Merge results from reasoning steps with decay."""
        merged = {}
        
        # Add current results
        for chunk, score in current:
            merged[chunk.id] = (chunk, score * decay_factor)
        
        # Add new results
        for chunk, score in new:
            if chunk.id in merged:
                # Take maximum score
                old_chunk, old_score = merged[chunk.id]
                merged[chunk.id] = (chunk, max(old_score, score))
            else:
                merged[chunk.id] = (chunk, score)
        
        # Sort by score
        results = list(merged.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    def _document_to_chunk(self, doc: Document) -> PDFChunk:
        """Convert Document to PDFChunk."""
        return PDFChunk(
            id=doc.id,
            pdf_id=doc.metadata.get("pdf_id", ""),
            content=doc.content,
            page_numbers=doc.metadata.get("page_numbers", [1]),
            chunk_index=doc.metadata.get("chunk_index", 0),
            metadata=doc.metadata,
            word_count=doc.metadata.get("word_count", 0),
            char_count=doc.metadata.get("char_count", len(doc.content)),
            section_title=doc.metadata.get("section_title"),
            chapter_title=doc.metadata.get("chapter_title"),
        )

    def _node_to_entity(self, node: Dict[str, Any]) -> GraphEntity:
        """Convert graph node to GraphEntity."""
        from datetime import datetime
        from src.models import EntityType
        
        return GraphEntity(
            id=node.get("id", ""),
            name=node.get("name", ""),
            type=EntityType(node.get("type", "other")),
            properties={k: v for k, v in node.items() 
                       if k not in ["id", "name", "type", "source_pdf_ids", "source_chunk_ids"]},
            source_pdf_ids=node.get("source_pdf_ids", []),
            source_chunk_ids=node.get("source_chunk_ids", []),
            confidence_score=node.get("confidence_score", 1.0),
            created_at=datetime.utcnow(),
        )

    async def explain_retrieval(
        self,
        query: str,
        chunk: PDFChunk,
    ) -> Dict[str, Any]:
        """
        Explain why a specific chunk was retrieved for a query.
        
        Args:
            query: Original query
            chunk: Retrieved chunk
            
        Returns:
            Explanation with vector and graph contributions
        """
        explanation = {
            "chunk_id": chunk.id,
            "chunk_preview": chunk.content[:200] + "...",
            "vector_similarity": 0.0,
            "graph_connections": [],
            "entity_matches": [],
            "total_score": 0.0,
        }
        
        # Get vector similarity
        query_embedding = await self.embedding_generator.generate_embedding(query)
        chunk_results = await self.vector_db.search(
            query_embedding=query_embedding,
            top_k=1,
            filters={"id": chunk.id},
        )
        
        if chunk_results:
            explanation["vector_similarity"] = chunk_results[0].score
        
        # Get graph connections
        if self.graph_db:
            # Find entities in chunk
            chunk_entities = await self._get_entities_from_chunks([chunk])
            
            # Find paths to query entities
            query_entities = await self._extract_query_entities(query)
            
            for chunk_entity in chunk_entities[:3]:
                for query_entity in query_entities[:3]:
                    path = await self.graph_db.find_path(
                        chunk_entity.id,
                        query_entity.id,
                        max_hops=3,
                    )
                    if path:
                        explanation["graph_connections"].append({
                            "from": chunk_entity.name,
                            "to": query_entity.name,
                            "path_length": len(path) // 2,  # Nodes only
                        })
            
            # Check entity matches
            chunk_text_lower = chunk.content.lower()
            for entity in query_entities:
                if entity.name.lower() in chunk_text_lower:
                    explanation["entity_matches"].append(entity.name)
        
        # Calculate total score
        explanation["total_score"] = (
            explanation["vector_similarity"] * self.vector_weight +
            len(explanation["graph_connections"]) * 0.1 * self.graph_weight +
            len(explanation["entity_matches"]) * 0.1
        )
        
        return explanation