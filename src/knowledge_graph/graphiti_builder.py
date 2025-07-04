"""
Graphiti integration for advanced knowledge graph construction.

This module provides integration with Graphiti for more sophisticated
knowledge graph building with temporal awareness and advanced features.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EntityNode, EpisodicNode
    from graphiti_core.edges import EntityRelation
    from graphiti_core.embeddings import EmbeddingModel
    from graphiti_core.llm import LLMClient
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    logger.warning("Graphiti not installed. Install with: pip install graphiti-core")

from src.config import get_settings
from src.graph_db.base import GraphDatabase
from src.models import (
    GraphEntity,
    GraphRelationship,
    PDFChunk,
    PDFMetadata,
)
from src.rag.embedder import EmbeddingGenerator
from src.utils.errors import GraphDatabaseError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class GraphitiBuilder:
    """
    Build and manage knowledge graphs using Graphiti.
    
    Graphiti provides advanced features like:
    - Temporal graph construction
    - Episodic memory
    - Entity deduplication
    - Relationship inference
    - Graph embeddings
    """

    def __init__(
        self,
        graph_db: GraphDatabase,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        llm_model: Optional[str] = None,
        enable_episodic_memory: bool = True,
    ) -> None:
        """
        Initialize Graphiti builder.
        
        Args:
            graph_db: Graph database instance
            embedding_generator: Embedding generator
            llm_model: LLM model name
            enable_episodic_memory: Whether to use episodic memory
        """
        if not GRAPHITI_AVAILABLE:
            raise ImportError(
                "Graphiti is not installed. Install it with: pip install graphiti-core"
            )
        
        self.settings = get_settings()
        self.graph_db = graph_db
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.llm_model = llm_model or self.settings.question_gen_model
        self.enable_episodic_memory = enable_episodic_memory
        
        # Graphiti instance will be initialized lazily
        self._graphiti = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Graphiti with Neo4j backend."""
        if self._initialized:
            return
        
        try:
            # Create custom embedding model wrapper
            class CustomEmbeddingModel(EmbeddingModel):
                def __init__(self, generator):
                    self.generator = generator
                
                async def embed(self, texts: List[str]) -> List[List[float]]:
                    return await self.generator.generate_embeddings(texts)
                
                @property
                def dimension(self) -> int:
                    return self.generator.dimension
            
            # Create custom LLM client wrapper
            class CustomLLMClient(LLMClient):
                def __init__(self, model_name, api_key):
                    self.model_name = model_name
                    self.api_key = api_key
                    self._client = None
                
                async def generate(self, prompt: str, **kwargs) -> str:
                    if not self._client:
                        import openai
                        self._client = openai.AsyncOpenAI(api_key=self.api_key)
                    
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    return response.choices[0].message.content
            
            # Initialize Graphiti
            self._graphiti = Graphiti(
                neo4j_uri=self.settings.neo4j_uri,
                neo4j_user=self.settings.neo4j_username,
                neo4j_password=self.settings.neo4j_password,
                neo4j_database=self.settings.neo4j_database,
                embedding_model=CustomEmbeddingModel(self.embedding_generator),
                llm_client=CustomLLMClient(
                    self.llm_model,
                    self.settings.openai_api_key,
                ),
                enable_episodic_memory=self.enable_episodic_memory,
            )
            
            await self._graphiti.initialize()
            self._initialized = True
            logger.info("Graphiti builder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Graphiti: {e}")
            raise GraphDatabaseError(f"Graphiti initialization failed: {str(e)}")

    async def close(self) -> None:
        """Close Graphiti connection."""
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
            self._initialized = False

    @log_performance
    async def build_from_chunks(
        self,
        chunks: List[PDFChunk],
        pdf_metadata: PDFMetadata,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from extracted entities and relationships.
        
        Args:
            chunks: PDF chunks
            pdf_metadata: PDF metadata
            entities: Extracted entities
            relationships: Extracted relationships
            
        Returns:
            Build statistics
        """
        await self.initialize()
        
        try:
            stats = {
                "entities_created": 0,
                "relationships_created": 0,
                "episodes_created": 0,
                "errors": [],
            }
            
            # Create document node
            doc_entity = await self._create_document_node(pdf_metadata)
            stats["entities_created"] += 1
            
            # Create entity nodes
            entity_map = {}
            for entity in entities:
                try:
                    node = await self._create_entity_node(entity)
                    entity_map[entity.id] = node
                    stats["entities_created"] += 1
                    
                    # Link to document
                    await self._create_document_link(doc_entity, node, entity)
                    stats["relationships_created"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to create entity {entity.name}: {e}")
                    stats["errors"].append(f"Entity {entity.name}: {str(e)}")
            
            # Create relationships
            for rel in relationships:
                try:
                    if rel.source_id in entity_map and rel.target_id in entity_map:
                        await self._create_relationship(
                            entity_map[rel.source_id],
                            entity_map[rel.target_id],
                            rel,
                        )
                        stats["relationships_created"] += 1
                except Exception as e:
                    logger.error(f"Failed to create relationship: {e}")
                    stats["errors"].append(f"Relationship: {str(e)}")
            
            # Create episodic memories if enabled
            if self.enable_episodic_memory:
                for chunk in chunks:
                    try:
                        episode = await self._create_episode(chunk, entity_map)
                        if episode:
                            stats["episodes_created"] += 1
                    except Exception as e:
                        logger.error(f"Failed to create episode: {e}")
                        stats["errors"].append(f"Episode: {str(e)}")
            
            # Run Graphiti's consolidation
            await self._graphiti.consolidate()
            
            logger.info(
                f"Built graph with {stats['entities_created']} entities, "
                f"{stats['relationships_created']} relationships, "
                f"{stats['episodes_created']} episodes"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise GraphDatabaseError(f"Graph building failed: {str(e)}")

    async def _create_document_node(
        self,
        pdf_metadata: PDFMetadata,
    ) -> EntityNode:
        """Create a document node in the graph."""
        node = EntityNode(
            name=pdf_metadata.filename,
            entity_type="document",
            properties={
                "file_id": pdf_metadata.file_id,
                "drive_path": pdf_metadata.drive_path,
                "total_pages": pdf_metadata.total_pages,
                "file_size_bytes": pdf_metadata.file_size_bytes,
                "created_at": pdf_metadata.created_at.isoformat(),
            },
            summary=f"PDF document: {pdf_metadata.filename}",
        )
        
        return await self._graphiti.add_entity(node)

    async def _create_entity_node(
        self,
        entity: GraphEntity,
    ) -> EntityNode:
        """Create an entity node in Graphiti."""
        # Prepare properties
        properties = {
            "original_id": entity.id,
            "confidence_score": entity.confidence_score,
            "source_pdf_ids": entity.source_pdf_ids,
            "source_chunk_ids": entity.source_chunk_ids,
            **entity.properties,
        }
        
        # Create node
        node = EntityNode(
            name=entity.name,
            entity_type=entity.type.value,
            properties=properties,
            summary=entity.properties.get("description", ""),
        )
        
        # Add to Graphiti
        return await self._graphiti.add_entity(node)

    async def _create_document_link(
        self,
        doc_node: EntityNode,
        entity_node: EntityNode,
        entity: GraphEntity,
    ) -> None:
        """Create a link between document and entity."""
        relation = EntityRelation(
            source=doc_node,
            target=entity_node,
            relation_type="mentions",
            properties={
                "confidence": entity.confidence_score,
                "first_chunk": entity.source_chunk_ids[0] if entity.source_chunk_ids else None,
            },
        )
        
        await self._graphiti.add_relation(relation)

    async def _create_relationship(
        self,
        source_node: EntityNode,
        target_node: EntityNode,
        relationship: GraphRelationship,
    ) -> None:
        """Create a relationship in Graphiti."""
        relation = EntityRelation(
            source=source_node,
            target=target_node,
            relation_type=relationship.type.value,
            properties={
                "original_id": relationship.id,
                "confidence_score": relationship.confidence_score,
                "source_pdf_ids": relationship.source_pdf_ids,
                "source_chunk_ids": relationship.source_chunk_ids,
                **relationship.properties,
            },
        )
        
        await self._graphiti.add_relation(relation)

    async def _create_episode(
        self,
        chunk: PDFChunk,
        entity_map: Dict[str, EntityNode],
    ) -> Optional[EpisodicNode]:
        """Create an episodic memory node for a chunk."""
        # Find entities mentioned in this chunk
        mentioned_entities = []
        for entity_id, node in entity_map.items():
            if chunk.id in node.properties.get("source_chunk_ids", []):
                mentioned_entities.append(node)
        
        if not mentioned_entities:
            return None
        
        # Create episode
        episode = EpisodicNode(
            content=chunk.content,
            timestamp=datetime.utcnow(),
            entities=mentioned_entities,
            properties={
                "chunk_id": chunk.id,
                "pdf_id": chunk.pdf_id,
                "page_numbers": chunk.page_numbers,
                "section_title": chunk.section_title,
                "chapter_title": chunk.chapter_title,
            },
        )
        
        return await self._graphiti.add_episode(episode)

    @log_performance
    async def query(
        self,
        query: str,
        include_episodes: bool = True,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the Graphiti knowledge graph.
        
        Args:
            query: Natural language query
            include_episodes: Whether to include episodic memories
            max_results: Maximum results to return
            
        Returns:
            Query results with entities, relationships, and episodes
        """
        await self.initialize()
        
        try:
            # Use Graphiti's built-in search
            results = await self._graphiti.search(
                query=query,
                include_episodes=include_episodes,
                limit=max_results,
            )
            
            # Format results
            formatted_results = {
                "entities": [],
                "relationships": [],
                "episodes": [],
                "query": query,
            }
            
            # Process entities
            for entity in results.get("entities", []):
                formatted_results["entities"].append({
                    "name": entity.name,
                    "type": entity.entity_type,
                    "summary": entity.summary,
                    "properties": entity.properties,
                    "relevance_score": entity.relevance_score,
                })
            
            # Process relationships
            for rel in results.get("relations", []):
                formatted_results["relationships"].append({
                    "source": rel.source.name,
                    "target": rel.target.name,
                    "type": rel.relation_type,
                    "properties": rel.properties,
                })
            
            # Process episodes
            if include_episodes:
                for episode in results.get("episodes", []):
                    formatted_results["episodes"].append({
                        "content": episode.content,
                        "timestamp": episode.timestamp.isoformat(),
                        "entities": [e.name for e in episode.entities],
                        "properties": episode.properties,
                        "relevance_score": episode.relevance_score,
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Graphiti query failed: {e}")
            raise GraphDatabaseError(f"Query failed: {str(e)}")

    async def get_entity_timeline(
        self,
        entity_name: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of episodes for an entity.
        
        Args:
            entity_name: Entity name
            limit: Maximum episodes to return
            
        Returns:
            List of episodes in chronological order
        """
        await self.initialize()
        
        try:
            timeline = await self._graphiti.get_entity_timeline(
                entity_name=entity_name,
                limit=limit,
            )
            
            return [
                {
                    "timestamp": episode.timestamp.isoformat(),
                    "content": episode.content,
                    "properties": episode.properties,
                }
                for episode in timeline
            ]
            
        except Exception as e:
            logger.error(f"Failed to get entity timeline: {e}")
            return []

    async def consolidate_graph(self) -> Dict[str, Any]:
        """
        Run Graphiti's graph consolidation.
        
        This merges duplicate entities, infers new relationships,
        and optimizes the graph structure.
        
        Returns:
            Consolidation statistics
        """
        await self.initialize()
        
        try:
            stats = await self._graphiti.consolidate()
            logger.info(f"Graph consolidation completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Graph consolidation failed: {e}")
            raise GraphDatabaseError(f"Consolidation failed: {str(e)}")