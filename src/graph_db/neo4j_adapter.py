"""
Neo4j adapter for graph database operations.

This module provides a concrete implementation of the GraphDatabase interface
using Neo4j as the backend graph database.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

from neo4j import AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError

from src.config import get_settings
from src.graph_db.base import GraphDatabase, GraphDatabaseFactory
from src.models import (
    GraphEntity,
    GraphQuery,
    GraphRelationship,
    GraphResult,
)
from src.utils.errors import (
    GraphDatabaseConnectionError,
    GraphDatabaseError,
    GraphDatabaseNotInitializedError,
    GraphQueryError,
    GraphSchemaError,
)
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class Neo4jAdapter(GraphDatabase):
    """
    Neo4j implementation of the graph database interface.
    
    This adapter provides full graph database functionality using Neo4j,
    including entity management, relationship handling, and complex
    graph traversals.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        """
        Initialize Neo4j adapter.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
        """
        self.settings = get_settings()
        self.uri = uri or self.settings.neo4j_uri
        self.username = username or self.settings.neo4j_username
        self.password = password or self.settings.neo4j_password
        self.database = database or self.settings.neo4j_database
        
        self._driver = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Neo4j connection and create indexes."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Connecting to Neo4j at {self.uri}")
            
            # Create driver
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                database=self.database,
            )
            
            # Verify connection
            async with self._driver.session() as session:
                result = await session.run("RETURN 1")
                await result.single()
            
            # Create indexes
            await self._create_indexes()
            
            self._initialized = True
            logger.info("Neo4j adapter initialized successfully")
            
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise GraphDatabaseConnectionError(
                f"Failed to connect to Neo4j: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}")
            raise GraphDatabaseError(f"Initialization failed: {str(e)}")

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info("Neo4j connection closed")

    def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
        if not self._initialized or not self._driver:
            raise GraphDatabaseNotInitializedError(
                "Neo4j adapter not initialized. Call initialize() first."
            )

    async def _create_indexes(self) -> None:
        """Create necessary indexes for performance."""
        indexes = [
            # Entity indexes
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
            
            # Document tracking indexes
            "CREATE INDEX entity_pdf IF NOT EXISTS FOR (e:Entity) ON (e.source_pdf_id)",
            "CREATE INDEX entity_chunk IF NOT EXISTS FOR (e:Entity) ON (e.source_chunk_id)",
            
            # Relationship indexes
            "CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.type)",
        ]
        
        async with self._driver.session() as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                except Neo4jError as e:
                    logger.warning(f"Index creation warning: {e}")

    # =========================================================================
    # Entity Operations
    # =========================================================================

    @log_performance
    async def create_entity(
        self,
        entity: GraphEntity,
        update_if_exists: bool = True,
    ) -> str:
        """Create or update an entity in Neo4j."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Prepare properties
                props = {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "confidence_score": entity.confidence_score,
                    "created_at": entity.created_at.isoformat(),
                    **entity.properties,
                }
                
                # Add source tracking
                if entity.source_pdf_ids:
                    props["source_pdf_ids"] = entity.source_pdf_ids
                if entity.source_chunk_ids:
                    props["source_chunk_ids"] = entity.source_chunk_ids
                
                # Create or update query
                if update_if_exists:
                    query = """
                    MERGE (e:Entity {id: $id})
                    SET e += $props
                    SET e.updated_at = datetime()
                    RETURN e.id as id
                    """
                else:
                    query = """
                    CREATE (e:Entity $props)
                    RETURN e.id as id
                    """
                
                result = await session.run(
                    query,
                    id=entity.id,
                    props=props,
                )
                record = await result.single()
                
                logger.debug(f"Created/updated entity: {entity.name} ({entity.type})")
                return record["id"]
                
        except Neo4jError as e:
            logger.error(f"Failed to create entity: {e}")
            raise GraphDatabaseError(f"Failed to create entity: {str(e)}")

    async def get_entity(
        self,
        entity_id: str,
        include_relationships: bool = False,
    ) -> Optional[GraphEntity]:
        """Get an entity by ID from Neo4j."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                if include_relationships:
                    query = """
                    MATCH (e:Entity {id: $id})
                    OPTIONAL MATCH (e)-[r]-(connected)
                    RETURN e, collect(distinct {
                        relationship: r,
                        connected: connected
                    }) as relationships
                    """
                else:
                    query = "MATCH (e:Entity {id: $id}) RETURN e"
                
                result = await session.run(query, id=entity_id)
                record = await result.single()
                
                if not record:
                    return None
                
                # Convert to GraphEntity
                node = record["e"]
                entity = self._node_to_entity(node)
                
                return entity
                
        except Neo4jError as e:
            logger.error(f"Failed to get entity: {e}")
            raise GraphDatabaseError(f"Failed to get entity: {str(e)}")

    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any],
    ) -> bool:
        """Update entity properties in Neo4j."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                query = """
                MATCH (e:Entity {id: $id})
                SET e += $props
                SET e.updated_at = datetime()
                RETURN e.id as id
                """
                
                result = await session.run(
                    query,
                    id=entity_id,
                    props=properties,
                )
                record = await result.single()
                
                return record is not None
                
        except Neo4jError as e:
            logger.error(f"Failed to update entity: {e}")
            raise GraphDatabaseError(f"Failed to update entity: {str(e)}")

    async def delete_entity(
        self,
        entity_id: str,
        cascade: bool = False,
    ) -> bool:
        """Delete an entity from Neo4j."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                if cascade:
                    # Delete entity and all its relationships
                    query = """
                    MATCH (e:Entity {id: $id})
                    DETACH DELETE e
                    RETURN count(e) > 0 as deleted
                    """
                else:
                    # Only delete if no relationships exist
                    query = """
                    MATCH (e:Entity {id: $id})
                    WHERE NOT EXISTS((e)-[]-())
                    DELETE e
                    RETURN count(e) > 0 as deleted
                    """
                
                result = await session.run(query, id=entity_id)
                record = await result.single()
                
                return record["deleted"]
                
        except Neo4jError as e:
            logger.error(f"Failed to delete entity: {e}")
            raise GraphDatabaseError(f"Failed to delete entity: {str(e)}")

    @log_performance
    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[GraphEntity]:
        """Search for entities using fulltext search."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Build search query
                cypher = """
                CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
                YIELD node, score
                WHERE score > 0.5
                """
                
                # Add type filter
                if entity_type:
                    cypher += " AND node.type = $type"
                
                # Add custom filters
                if filters:
                    for key in filters:
                        cypher += f" AND node.{key} = ${key}"
                
                cypher += """
                RETURN node
                ORDER BY score DESC
                LIMIT $limit
                """
                
                # Prepare parameters
                params = {"query": query, "limit": limit}
                if entity_type:
                    params["type"] = entity_type
                if filters:
                    params.update(filters)
                
                result = await session.run(cypher, **params)
                
                entities = []
                async for record in result:
                    entity = self._node_to_entity(record["node"])
                    entities.append(entity)
                
                return entities
                
        except Neo4jError as e:
            logger.error(f"Failed to search entities: {e}")
            raise GraphDatabaseError(f"Failed to search entities: {str(e)}")

    # =========================================================================
    # Relationship Operations
    # =========================================================================

    @log_performance
    async def create_relationship(
        self,
        relationship: GraphRelationship,
        update_if_exists: bool = True,
    ) -> str:
        """Create or update a relationship in Neo4j."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Prepare properties
                props = {
                    "id": relationship.id,
                    "type": relationship.type.value,
                    "confidence_score": relationship.confidence_score,
                    "created_at": relationship.created_at.isoformat(),
                    **relationship.properties,
                }
                
                # Add source tracking
                if relationship.source_pdf_ids:
                    props["source_pdf_ids"] = relationship.source_pdf_ids
                if relationship.source_chunk_ids:
                    props["source_chunk_ids"] = relationship.source_chunk_ids
                
                # Create relationship
                query = """
                MATCH (source:Entity {id: $source_id})
                MATCH (target:Entity {id: $target_id})
                MERGE (source)-[r:RELATED {id: $id}]->(target)
                SET r += $props
                RETURN r.id as id
                """
                
                result = await session.run(
                    query,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    id=relationship.id,
                    props=props,
                )
                record = await result.single()
                
                if not record:
                    raise GraphDatabaseError(
                        f"Failed to create relationship: source or target entity not found"
                    )
                
                return record["id"]
                
        except Neo4jError as e:
            logger.error(f"Failed to create relationship: {e}")
            raise GraphDatabaseError(f"Failed to create relationship: {str(e)}")

    async def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[GraphRelationship]:
        """Get relationships for an entity."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Build query based on direction
                if direction == "outgoing":
                    match_pattern = "(e:Entity {id: $id})-[r]->(target)"
                elif direction == "incoming":
                    match_pattern = "(source)-[r]->(e:Entity {id: $id})"
                else:  # both
                    match_pattern = "(e:Entity {id: $id})-[r]-(other)"
                
                query = f"MATCH {match_pattern}"
                
                # Add type filter
                if relationship_type:
                    query += " WHERE r.type = $type"
                
                query += " RETURN r, startNode(r) as source, endNode(r) as target"
                
                # Prepare parameters
                params = {"id": entity_id}
                if relationship_type:
                    params["type"] = relationship_type
                
                result = await session.run(query, **params)
                
                relationships = []
                async for record in result:
                    rel = self._edge_to_relationship(
                        record["r"],
                        record["source"],
                        record["target"],
                    )
                    relationships.append(rel)
                
                return relationships
                
        except Neo4jError as e:
            logger.error(f"Failed to get relationships: {e}")
            raise GraphDatabaseError(f"Failed to get relationships: {str(e)}")

    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship from Neo4j."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                query = """
                MATCH ()-[r:RELATED {id: $id}]-()
                DELETE r
                RETURN count(r) > 0 as deleted
                """
                
                result = await session.run(query, id=relationship_id)
                record = await result.single()
                
                return record["deleted"]
                
        except Neo4jError as e:
            logger.error(f"Failed to delete relationship: {e}")
            raise GraphDatabaseError(f"Failed to delete relationship: {str(e)}")

    # =========================================================================
    # Graph Traversal
    # =========================================================================

    @log_performance
    async def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        relationship_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
    ) -> List[GraphEntity]:
        """Get neighboring entities within N hops."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Build relationship pattern
                if relationship_types:
                    rel_pattern = f":RELATED{{type: {relationship_types}}}"
                else:
                    rel_pattern = ""
                
                # Build query with variable length path
                query = f"""
                MATCH (start:Entity {{id: $id}})
                MATCH (start)-[{rel_pattern}*1..{hops}]-(neighbor:Entity)
                WHERE neighbor.id <> $id
                """
                
                # Add entity type filter
                if entity_types:
                    query += " AND neighbor.type IN $entity_types"
                
                query += """
                RETURN DISTINCT neighbor
                ORDER BY neighbor.name
                """
                
                # Prepare parameters
                params = {"id": entity_id}
                if entity_types:
                    params["entity_types"] = entity_types
                
                result = await session.run(query, **params)
                
                entities = []
                async for record in result:
                    entity = self._node_to_entity(record["neighbor"])
                    entities.append(entity)
                
                return entities
                
        except Neo4jError as e:
            logger.error(f"Failed to get neighbors: {e}")
            raise GraphDatabaseError(f"Failed to get neighbors: {str(e)}")

    async def find_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_hops: int = 5,
        relationship_types: Optional[List[str]] = None,
    ) -> Optional[List[Union[GraphEntity, GraphRelationship]]]:
        """Find shortest path between two entities."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Build query for shortest path
                query = """
                MATCH (start:Entity {id: $start_id})
                MATCH (end:Entity {id: $end_id})
                MATCH path = shortestPath((start)-[*..%d]-(end))
                RETURN path
                """ % max_hops
                
                result = await session.run(
                    query,
                    start_id=start_entity_id,
                    end_id=end_entity_id,
                )
                record = await result.single()
                
                if not record:
                    return None
                
                # Extract path elements
                path = record["path"]
                elements = []
                
                # Alternate between nodes and relationships
                for i, node in enumerate(path.nodes):
                    elements.append(self._node_to_entity(node))
                    if i < len(path.relationships):
                        rel = path.relationships[i]
                        elements.append(
                            self._edge_to_relationship(
                                rel,
                                path.nodes[i],
                                path.nodes[i + 1],
                            )
                        )
                
                return elements
                
        except Neo4jError as e:
            logger.error(f"Failed to find path: {e}")
            raise GraphDatabaseError(f"Failed to find path: {str(e)}")

    async def get_subgraph(
        self,
        entity_ids: List[str],
        include_relationships: bool = True,
        expand_hops: int = 0,
    ) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """Extract a subgraph containing specified entities."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Start with specified entities
                entity_query = """
                MATCH (e:Entity)
                WHERE e.id IN $entity_ids
                """
                
                # Expand if requested
                if expand_hops > 0:
                    entity_query += f"""
                    OPTIONAL MATCH (e)-[*1..{expand_hops}]-(expanded:Entity)
                    WITH collect(DISTINCT e) + collect(DISTINCT expanded) as all_entities
                    UNWIND all_entities as entity
                    """
                else:
                    entity_query += " WITH collect(DISTINCT e) as all_entities\nUNWIND all_entities as entity"
                
                entity_query += " RETURN DISTINCT entity"
                
                # Get entities
                result = await session.run(entity_query, entity_ids=entity_ids)
                entities = []
                entity_ids_in_subgraph = []
                
                async for record in result:
                    entity = self._node_to_entity(record["entity"])
                    entities.append(entity)
                    entity_ids_in_subgraph.append(entity.id)
                
                # Get relationships if requested
                relationships = []
                if include_relationships and len(entity_ids_in_subgraph) > 1:
                    rel_query = """
                    MATCH (source:Entity)-[r]-(target:Entity)
                    WHERE source.id IN $entity_ids 
                    AND target.id IN $entity_ids
                    AND id(source) < id(target)
                    RETURN DISTINCT r, source, target
                    """
                    
                    rel_result = await session.run(
                        rel_query,
                        entity_ids=entity_ids_in_subgraph,
                    )
                    
                    async for record in rel_result:
                        rel = self._edge_to_relationship(
                            record["r"],
                            record["source"],
                            record["target"],
                        )
                        relationships.append(rel)
                
                return entities, relationships
                
        except Neo4jError as e:
            logger.error(f"Failed to get subgraph: {e}")
            raise GraphDatabaseError(f"Failed to get subgraph: {str(e)}")

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def execute_query(self, query: GraphQuery) -> GraphResult:
        """Execute a structured graph query."""
        self._ensure_initialized()
        
        try:
            # Route to appropriate query handler
            if query.query_type == "neighbors":
                entities = await self.get_neighbors(
                    query.parameters["entity_id"],
                    query.parameters.get("hops", 1),
                    query.filters.get("relationship_types"),
                    query.filters.get("entity_types"),
                )
                return GraphResult(
                    entities=entities[:query.limit],
                    relationships=[],
                    metadata={"query_type": "neighbors"},
                    query_time=0.0,  # TODO: Add timing
                )
            
            elif query.query_type == "search":
                entities = await self.search_entities(
                    query.parameters["query"],
                    query.filters.get("entity_type"),
                    query.filters,
                    query.limit,
                )
                return GraphResult(
                    entities=entities,
                    relationships=[],
                    metadata={"query_type": "search"},
                    query_time=0.0,
                )
            
            elif query.query_type == "subgraph":
                entities, relationships = await self.get_subgraph(
                    query.parameters["entity_ids"],
                    query.parameters.get("include_relationships", True),
                    query.parameters.get("expand_hops", 0),
                )
                return GraphResult(
                    entities=entities[:query.limit],
                    relationships=relationships,
                    metadata={"query_type": "subgraph"},
                    query_time=0.0,
                )
            
            else:
                raise GraphQueryError(f"Unknown query type: {query.query_type}")
                
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise GraphQueryError(f"Query execution failed: {str(e)}")

    async def execute_cypher(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute raw Cypher query."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                result = await session.run(cypher, **(parameters or {}))
                
                records = []
                async for record in result:
                    records.append(dict(record))
                
                return records
                
        except Neo4jError as e:
            logger.error(f"Failed to execute Cypher: {e}")
            raise GraphQueryError(f"Cypher execution failed: {str(e)}")

    # =========================================================================
    # Schema Operations
    # =========================================================================

    async def create_index(
        self,
        entity_type: str,
        property_name: str,
        index_type: str = "exact",
    ) -> bool:
        """Create an index on entity property."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                if index_type == "fulltext":
                    query = f"""
                    CREATE FULLTEXT INDEX {entity_type}_{property_name}_fulltext IF NOT EXISTS
                    FOR (e:Entity) ON EACH [e.{property_name}]
                    WHERE e.type = '{entity_type}'
                    """
                else:
                    query = f"""
                    CREATE INDEX {entity_type}_{property_name}_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.{property_name})
                    """
                
                await session.run(query)
                return True
                
        except Neo4jError as e:
            logger.error(f"Failed to create index: {e}")
            raise GraphSchemaError(f"Failed to create index: {str(e)}")

    async def get_schema(self) -> Dict[str, Any]:
        """Get graph schema information."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Get entity types
                entity_result = await session.run(
                    "MATCH (e:Entity) RETURN DISTINCT e.type as type"
                )
                entity_types = [
                    record["type"] async for record in entity_result
                ]
                
                # Get relationship types
                rel_result = await session.run(
                    "MATCH ()-[r:RELATED]-() RETURN DISTINCT r.type as type"
                )
                relationship_types = [
                    record["type"] async for record in rel_result
                ]
                
                # Get indexes
                index_result = await session.run("SHOW INDEXES")
                indexes = []
                async for record in index_result:
                    indexes.append({
                        "name": record.get("name"),
                        "type": record.get("type"),
                        "properties": record.get("properties"),
                    })
                
                return {
                    "entity_types": entity_types,
                    "relationship_types": relationship_types,
                    "indexes": indexes,
                    "node_count": await self._get_node_count(),
                    "relationship_count": await self._get_relationship_count(),
                }
                
        except Neo4jError as e:
            logger.error(f"Failed to get schema: {e}")
            raise GraphSchemaError(f"Failed to get schema: {str(e)}")

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    @log_performance
    async def bulk_create_entities(
        self,
        entities: List[GraphEntity],
        batch_size: int = 1000,
    ) -> List[str]:
        """Bulk create entities in Neo4j."""
        self._ensure_initialized()
        
        created_ids = []
        
        try:
            async with self._driver.session() as session:
                for i in range(0, len(entities), batch_size):
                    batch = entities[i:i + batch_size]
                    
                    # Prepare batch data
                    batch_data = []
                    for entity in batch:
                        props = {
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.type.value,
                            "confidence_score": entity.confidence_score,
                            "created_at": entity.created_at.isoformat(),
                            **entity.properties,
                        }
                        
                        if entity.source_pdf_ids:
                            props["source_pdf_ids"] = entity.source_pdf_ids
                        if entity.source_chunk_ids:
                            props["source_chunk_ids"] = entity.source_chunk_ids
                        
                        batch_data.append(props)
                    
                    # Bulk create
                    query = """
                    UNWIND $batch as props
                    MERGE (e:Entity {id: props.id})
                    SET e += props
                    RETURN e.id as id
                    """
                    
                    result = await session.run(query, batch=batch_data)
                    
                    async for record in result:
                        created_ids.append(record["id"])
                
                logger.info(f"Bulk created {len(created_ids)} entities")
                return created_ids
                
        except Neo4jError as e:
            logger.error(f"Failed to bulk create entities: {e}")
            raise GraphDatabaseError(f"Failed to bulk create entities: {str(e)}")

    async def bulk_create_relationships(
        self,
        relationships: List[GraphRelationship],
        batch_size: int = 1000,
    ) -> List[str]:
        """Bulk create relationships in Neo4j."""
        self._ensure_initialized()
        
        created_ids = []
        
        try:
            async with self._driver.session() as session:
                for i in range(0, len(relationships), batch_size):
                    batch = relationships[i:i + batch_size]
                    
                    # Prepare batch data
                    batch_data = []
                    for rel in batch:
                        props = {
                            "id": rel.id,
                            "source_id": rel.source_id,
                            "target_id": rel.target_id,
                            "type": rel.type.value,
                            "confidence_score": rel.confidence_score,
                            "created_at": rel.created_at.isoformat(),
                            **rel.properties,
                        }
                        
                        if rel.source_pdf_ids:
                            props["source_pdf_ids"] = rel.source_pdf_ids
                        if rel.source_chunk_ids:
                            props["source_chunk_ids"] = rel.source_chunk_ids
                        
                        batch_data.append(props)
                    
                    # Bulk create
                    query = """
                    UNWIND $batch as props
                    MATCH (source:Entity {id: props.source_id})
                    MATCH (target:Entity {id: props.target_id})
                    MERGE (source)-[r:RELATED {id: props.id}]->(target)
                    SET r += props
                    RETURN r.id as id
                    """
                    
                    result = await session.run(query, batch=batch_data)
                    
                    async for record in result:
                        created_ids.append(record["id"])
                
                logger.info(f"Bulk created {len(created_ids)} relationships")
                return created_ids
                
        except Neo4jError as e:
            logger.error(f"Failed to bulk create relationships: {e}")
            raise GraphDatabaseError(f"Failed to bulk create relationships: {str(e)}")

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def clear_graph(self) -> bool:
        """Clear all data from the graph."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Delete all nodes and relationships
                await session.run("MATCH (n) DETACH DELETE n")
                logger.warning("Cleared all data from graph database")
                return True
                
        except Neo4jError as e:
            logger.error(f"Failed to clear graph: {e}")
            raise GraphDatabaseError(f"Failed to clear graph: {str(e)}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        self._ensure_initialized()
        
        try:
            async with self._driver.session() as session:
                # Get various statistics
                stats = {
                    "total_entities": await self._get_node_count(),
                    "total_relationships": await self._get_relationship_count(),
                    "entity_types": {},
                    "relationship_types": {},
                }
                
                # Count by entity type
                type_result = await session.run("""
                    MATCH (e:Entity)
                    RETURN e.type as type, count(e) as count
                    ORDER BY count DESC
                """)
                
                async for record in type_result:
                    stats["entity_types"][record["type"]] = record["count"]
                
                # Count by relationship type
                rel_result = await session.run("""
                    MATCH ()-[r:RELATED]-()
                    RETURN r.type as type, count(r) as count
                    ORDER BY count DESC
                """)
                
                async for record in rel_result:
                    stats["relationship_types"][record["type"]] = record["count"]
                
                return stats
                
        except Neo4jError as e:
            logger.error(f"Failed to get statistics: {e}")
            raise GraphDatabaseError(f"Failed to get statistics: {str(e)}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _node_to_entity(self, node) -> GraphEntity:
        """Convert Neo4j node to GraphEntity."""
        from datetime import datetime
        from src.models import EntityType
        
        # Extract properties
        props = dict(node)
        
        # Parse dates
        created_at = props.pop("created_at", None)
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.utcnow()
        
        updated_at = props.pop("updated_at", None)
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        # Create entity
        return GraphEntity(
            id=props.pop("id"),
            name=props.pop("name"),
            type=EntityType(props.pop("type", "other")),
            properties={k: v for k, v in props.items() 
                       if k not in ["source_pdf_ids", "source_chunk_ids", "confidence_score"]},
            source_pdf_ids=props.get("source_pdf_ids", []),
            source_chunk_ids=props.get("source_chunk_ids", []),
            confidence_score=props.get("confidence_score", 1.0),
            created_at=created_at,
            updated_at=updated_at,
        )

    def _edge_to_relationship(
        self,
        edge,
        source_node,
        target_node,
    ) -> GraphRelationship:
        """Convert Neo4j edge to GraphRelationship."""
        from datetime import datetime
        from src.models import RelationshipType
        
        # Extract properties
        props = dict(edge)
        
        # Parse date
        created_at = props.pop("created_at", None)
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.utcnow()
        
        # Create relationship
        return GraphRelationship(
            id=props.pop("id"),
            source_id=source_node["id"],
            target_id=target_node["id"],
            type=RelationshipType(props.pop("type", "related_to")),
            properties={k: v for k, v in props.items() 
                       if k not in ["source_pdf_ids", "source_chunk_ids", "confidence_score"]},
            source_pdf_ids=props.get("source_pdf_ids", []),
            source_chunk_ids=props.get("source_chunk_ids", []),
            confidence_score=props.get("confidence_score", 1.0),
            created_at=created_at,
        )

    async def _get_node_count(self) -> int:
        """Get total node count."""
        async with self._driver.session() as session:
            result = await session.run("MATCH (n:Entity) RETURN count(n) as count")
            record = await result.single()
            return record["count"]

    async def _get_relationship_count(self) -> int:
        """Get total relationship count."""
        async with self._driver.session() as session:
            result = await session.run("MATCH ()-[r:RELATED]-() RETURN count(r) as count")
            record = await result.single()
            return record["count"]


# Register the adapter
GraphDatabaseFactory.register("neo4j", Neo4jAdapter)