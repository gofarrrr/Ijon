"""
Graph-enhanced retriever combining vector search with knowledge graph.

This module implements retrieval that leverages both vector similarity
and entity relationships for improved context discovery.
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import get_settings
from src.models import SearchResult, PDFChunk
from src.rag.retriever import DocumentRetriever
from src.rag.graph_store import PostgresGraphStore, Entity
from src.rag.entity_extractor import EntityExtractor
from src.utils.errors import RetrievalError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class GraphEnhancedRetriever(DocumentRetriever):
    """
    Retriever that combines vector search with graph-based context expansion.
    
    Enhances traditional RAG by:
    1. Extracting entities from queries
    2. Finding related entities through graph traversal
    3. Including chunks from related entities
    4. Scoring based on both similarity and graph relationships
    """
    
    def __init__(
        self,
        graph_store: Optional[PostgresGraphStore] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        graph_weight: float = 0.3,
        max_graph_depth: int = 2,
        **kwargs
    ):
        """
        Initialize graph-enhanced retriever.
        
        Args:
            graph_store: PostgreSQL graph store instance
            entity_extractor: Entity extractor instance
            graph_weight: Weight for graph-based scores (0-1)
            max_graph_depth: Maximum depth for graph traversal
            **kwargs: Arguments for parent DocumentRetriever
        """
        super().__init__(**kwargs)
        
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.graph_weight = graph_weight
        self.vector_weight = 1.0 - graph_weight
        self.max_graph_depth = max_graph_depth
        
        logger.info(
            f"Initialized graph-enhanced retriever "
            f"(graph_weight={graph_weight}, max_depth={max_graph_depth})"
        )
    
    async def initialize(self) -> None:
        """Initialize retriever and graph components."""
        await super().initialize()
        
        if self.graph_store:
            await self.graph_store.initialize()
            logger.info("Graph store initialized for enhanced retrieval")
    
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
        Retrieve chunks using both vector search and graph expansion.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            min_score: Minimum similarity score
            filters: Metadata filters
            use_graph: Whether to use graph enhancement
            
        Returns:
            List of (chunk, score) tuples
        """
        # Start with standard vector retrieval
        vector_results = await super().retrieve_chunks(
            query=query,
            top_k=top_k * 2 if use_graph else top_k,  # Get more for merging
            min_score=min_score,
            filters=filters
        )
        
        if not use_graph or not self.graph_store:
            return vector_results[:top_k]
        
        try:
            # Enhance with graph-based retrieval
            enhanced_results = await self._enhance_with_graph(
                query=query,
                vector_results=vector_results,
                top_k=top_k,
            )
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Graph enhancement failed: {e}")
            # Fall back to vector results
            return vector_results[:top_k]
    
    async def _enhance_with_graph(
        self,
        query: str,
        vector_results: List[Tuple[PDFChunk, float]],
        top_k: int,
    ) -> List[Tuple[PDFChunk, float]]:
        """Enhance vector results with graph-based context."""
        # Extract entities from query
        query_entities = await self._extract_query_entities(query)
        
        if not query_entities:
            logger.debug("No entities found in query, using vector results only")
            return vector_results[:top_k]
        
        # Get entities from top vector results
        result_entities = await self._get_entities_from_chunks(
            [chunk for chunk, _ in vector_results[:5]]  # Top 5 chunks
        )
        
        # Expand entities through graph
        expanded_entities = await self._expand_entities(
            seed_entities=query_entities + result_entities,
            max_depth=self.max_graph_depth
        )
        
        # Get chunks related to expanded entities
        graph_chunk_ids = await self.graph_store.get_related_chunks(
            entity_ids=[e.id for e in expanded_entities if e.id],
            limit=50  # Reasonable limit
        )
        
        # Merge and rerank results
        merged_results = await self._merge_graph_results(
            vector_results=vector_results,
            graph_chunk_ids=graph_chunk_ids,
            query_entities=query_entities,
            expanded_entities=expanded_entities,
            top_k=top_k
        )
        
        return merged_results
    
    async def _extract_query_entities(self, query: str) -> List[Entity]:
        """Extract entities from the query."""
        try:
            # Create a temporary chunk for the query
            query_chunk = PDFChunk(
                id="query",
                pdf_id="query",
                content=query,
                page_numbers=[0],
                chunk_index=0,
                metadata={},
                word_count=len(query.split()),
                char_count=len(query)
            )
            
            # Extract entities
            extraction_result = await self.entity_extractor.extract_from_chunk(
                query_chunk,
                context="User search query"
            )
            
            # Find matching entities in database
            matched_entities = []
            for entity in extraction_result.entities:
                db_entities = await self.graph_store.find_entities(
                    name_query=entity.name,
                    entity_type=entity.type,
                    limit=3
                )
                matched_entities.extend(db_entities)
            
            logger.debug(f"Found {len(matched_entities)} entities in query")
            return matched_entities
            
        except Exception as e:
            logger.error(f"Failed to extract query entities: {e}")
            return []
    
    async def _get_entities_from_chunks(
        self,
        chunks: List[PDFChunk]
    ) -> List[Entity]:
        """Get entities associated with chunks."""
        all_entities = []
        
        for chunk in chunks:
            entities = await self.graph_store.find_entities(
                chunk_id=chunk.id,
                limit=10
            )
            all_entities.extend(entities)
        
        # Deduplicate by entity ID
        seen_ids = set()
        unique_entities = []
        for entity in all_entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def _expand_entities(
        self,
        seed_entities: List[Entity],
        max_depth: int
    ) -> List[Entity]:
        """Expand entities through graph relationships."""
        if not seed_entities:
            return []
        
        expanded_entities = []
        seen_ids = set()
        
        # Get graph neighborhood for each seed entity
        for entity in seed_entities[:10]:  # Limit to prevent explosion
            if not entity.id:
                continue
                
            try:
                graph_data = await self.graph_store.get_entity_graph(
                    entity_id=entity.id,
                    max_depth=max_depth
                )
                
                # Extract unique entities from graph
                for node in graph_data['nodes']:
                    if node['id'] not in seen_ids:
                        seen_ids.add(node['id'])
                        
                        # Fetch full entity data
                        db_entities = await self.graph_store.find_entities(
                            name_query=node['name'],
                            entity_type=node['type'],
                            limit=1
                        )
                        
                        if db_entities:
                            expanded_entities.append(db_entities[0])
                            
            except Exception as e:
                logger.warning(f"Failed to expand entity {entity.id}: {e}")
                continue
        
        logger.debug(f"Expanded to {len(expanded_entities)} entities through graph")
        return expanded_entities
    
    async def _merge_graph_results(
        self,
        vector_results: List[Tuple[PDFChunk, float]],
        graph_chunk_ids: List[str],
        query_entities: List[Entity],
        expanded_entities: List[Entity],
        top_k: int
    ) -> List[Tuple[PDFChunk, float]]:
        """Merge vector and graph results with scoring."""
        # Create chunk score map
        chunk_scores = {}
        
        # Add vector results with vector weight
        for chunk, score in vector_results:
            chunk_scores[chunk.id] = {
                'chunk': chunk,
                'vector_score': score * self.vector_weight,
                'graph_score': 0.0,
                'entity_bonus': 0.0
            }
        
        # Add graph-based scoring
        graph_chunk_set = set(graph_chunk_ids)
        
        # Calculate graph scores based on entity relationships
        for chunk_id in graph_chunk_ids:
            if chunk_id not in chunk_scores:
                # Need to fetch chunk data
                # For now, we'll skip chunks not in vector results
                continue
            
            # Base graph score
            chunk_scores[chunk_id]['graph_score'] = 0.5 * self.graph_weight
        
        # Add entity overlap bonus
        query_entity_names = {e.name.lower() for e in query_entities}
        
        for chunk_id, data in chunk_scores.items():
            chunk_text = data['chunk'].content.lower()
            
            # Count entity mentions
            entity_mentions = sum(
                1 for name in query_entity_names
                if name in chunk_text
            )
            
            # Bonus for entity mentions (up to 0.2)
            data['entity_bonus'] = min(entity_mentions * 0.05, 0.2)
        
        # Calculate final scores
        final_results = []
        
        for chunk_id, data in chunk_scores.items():
            final_score = (
                data['vector_score'] +
                data['graph_score'] +
                data['entity_bonus']
            )
            
            final_results.append((data['chunk'], final_score))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Log statistics
        graph_enhanced = sum(1 for _, d in chunk_scores.items() if d['graph_score'] > 0)
        logger.info(
            f"Graph enhancement: {len(vector_results)} vector results, "
            f"{len(graph_chunk_ids)} graph chunks, {graph_enhanced} enhanced, "
            f"returning top {top_k}"
        )
        
        return final_results[:top_k]
    
    async def explain_retrieval(
        self,
        query: str,
        chunk: PDFChunk
    ) -> Dict[str, Any]:
        """
        Explain why a chunk was retrieved.
        
        Provides transparency into vector similarity, graph connections,
        and entity matches that contributed to retrieval.
        """
        explanation = {
            'chunk_id': chunk.id,
            'chunk_preview': chunk.content[:200] + '...',
            'vector_score': 0.0,
            'graph_connections': [],
            'entity_matches': [],
            'total_score': 0.0
        }
        
        try:
            # Get vector similarity score
            query_embedding = await self.embedding_generator.generate_embedding(query)
            chunk_embedding = await self.embedding_generator.generate_embedding(chunk.content)
            
            vector_similarity = await self.embedding_generator.compute_similarity(
                query_embedding, chunk_embedding, metric="cosine"
            )
            explanation['vector_score'] = float(vector_similarity)
            
            # Get entities from query and chunk
            query_entities = await self._extract_query_entities(query)
            chunk_entities = await self._get_entities_from_chunks([chunk])
            
            # Find entity matches
            query_names = {e.name.lower() for e in query_entities}
            chunk_names = {e.name.lower() for e in chunk_entities}
            
            explanation['entity_matches'] = list(query_names & chunk_names)
            
            # Find graph connections
            if self.graph_store and query_entities and chunk_entities:
                for q_entity in query_entities[:3]:  # Limit for performance
                    for c_entity in chunk_entities[:3]:
                        if q_entity.id and c_entity.id:
                            path = await self.graph_store.find_path(
                                q_entity.id, c_entity.id, max_depth=3
                            )
                            
                            if path:
                                explanation['graph_connections'].append({
                                    'from': q_entity.name,
                                    'to': c_entity.name,
                                    'path_length': len(path),
                                    'path': [p['name'] for p in path]
                                })
            
            # Calculate total score
            explanation['total_score'] = (
                explanation['vector_score'] * self.vector_weight +
                len(explanation['graph_connections']) * 0.1 * self.graph_weight +
                len(explanation['entity_matches']) * 0.1
            )
            
        except Exception as e:
            logger.error(f"Failed to explain retrieval: {e}")
            explanation['error'] = str(e)
        
        return explanation


def create_graph_retriever(**kwargs) -> GraphEnhancedRetriever:
    """Create a graph-enhanced retriever instance."""
    return GraphEnhancedRetriever(**kwargs)