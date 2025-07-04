"""
Base interface for graph database operations.

This module defines the abstract interface that all graph database
implementations must follow, ensuring consistent behavior across
different graph database backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from src.models import GraphEntity, GraphRelationship, GraphQuery, GraphResult
from src.utils.errors import GraphDatabaseError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GraphDatabase(ABC):
    """
    Abstract base class for graph database operations.
    
    This interface supports:
    - Entity and relationship management
    - Graph traversal and querying
    - Subgraph extraction
    - Schema management
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the graph database connection.
        
        Raises:
            GraphDatabaseError: If initialization fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the graph database connection."""
        pass

    # =========================================================================
    # Entity Operations
    # =========================================================================
    
    @abstractmethod
    async def create_entity(
        self,
        entity: GraphEntity,
        update_if_exists: bool = True,
    ) -> str:
        """
        Create or update an entity in the graph.
        
        Args:
            entity: Entity to create
            update_if_exists: Whether to update if entity exists
            
        Returns:
            Entity ID
            
        Raises:
            GraphDatabaseError: If operation fails
        """
        pass

    @abstractmethod
    async def get_entity(
        self,
        entity_id: str,
        include_relationships: bool = False,
    ) -> Optional[GraphEntity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: Entity ID
            include_relationships: Whether to include relationships
            
        Returns:
            Entity or None if not found
        """
        pass

    @abstractmethod
    async def update_entity(
        self,
        entity_id: str,
        properties: Dict[str, Any],
    ) -> bool:
        """
        Update entity properties.
        
        Args:
            entity_id: Entity ID
            properties: Properties to update
            
        Returns:
            True if updated, False if not found
        """
        pass

    @abstractmethod
    async def delete_entity(
        self,
        entity_id: str,
        cascade: bool = False,
    ) -> bool:
        """
        Delete an entity.
        
        Args:
            entity_id: Entity ID
            cascade: Whether to delete related relationships
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[GraphEntity]:
        """
        Search for entities using text query.
        
        Args:
            query: Search query
            entity_type: Filter by entity type
            filters: Additional property filters
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        pass

    # =========================================================================
    # Relationship Operations
    # =========================================================================
    
    @abstractmethod
    async def create_relationship(
        self,
        relationship: GraphRelationship,
        update_if_exists: bool = True,
    ) -> str:
        """
        Create or update a relationship.
        
        Args:
            relationship: Relationship to create
            update_if_exists: Whether to update if exists
            
        Returns:
            Relationship ID
        """
        pass

    @abstractmethod
    async def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[GraphRelationship]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: Entity ID
            relationship_type: Filter by relationship type
            direction: 'incoming', 'outgoing', or 'both'
            
        Returns:
            List of relationships
        """
        pass

    @abstractmethod
    async def delete_relationship(
        self,
        relationship_id: str,
    ) -> bool:
        """
        Delete a relationship.
        
        Args:
            relationship_id: Relationship ID
            
        Returns:
            True if deleted, False if not found
        """
        pass

    # =========================================================================
    # Graph Traversal
    # =========================================================================
    
    @abstractmethod
    async def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        relationship_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
    ) -> List[GraphEntity]:
        """
        Get neighboring entities within N hops.
        
        Args:
            entity_id: Starting entity ID
            hops: Number of hops (depth)
            relationship_types: Filter by relationship types
            entity_types: Filter by entity types
            
        Returns:
            List of neighboring entities
        """
        pass

    @abstractmethod
    async def find_path(
        self,
        start_entity_id: str,
        end_entity_id: str,
        max_hops: int = 5,
        relationship_types: Optional[List[str]] = None,
    ) -> Optional[List[Union[GraphEntity, GraphRelationship]]]:
        """
        Find shortest path between two entities.
        
        Args:
            start_entity_id: Starting entity
            end_entity_id: Target entity
            max_hops: Maximum path length
            relationship_types: Allowed relationship types
            
        Returns:
            Path as alternating list of entities and relationships
        """
        pass

    @abstractmethod
    async def get_subgraph(
        self,
        entity_ids: List[str],
        include_relationships: bool = True,
        expand_hops: int = 0,
    ) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """
        Extract a subgraph containing specified entities.
        
        Args:
            entity_ids: Entity IDs to include
            include_relationships: Include relationships between entities
            expand_hops: Expand subgraph by N hops
            
        Returns:
            Tuple of (entities, relationships)
        """
        pass

    # =========================================================================
    # Query Operations
    # =========================================================================
    
    @abstractmethod
    async def execute_query(
        self,
        query: GraphQuery,
    ) -> GraphResult:
        """
        Execute a graph query.
        
        Args:
            query: Graph query object
            
        Returns:
            Query results
        """
        pass

    @abstractmethod
    async def execute_cypher(
        self,
        cypher: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute raw Cypher query (for Neo4j-compatible DBs).
        
        Args:
            cypher: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        pass

    # =========================================================================
    # Schema Operations
    # =========================================================================
    
    @abstractmethod
    async def create_index(
        self,
        entity_type: str,
        property_name: str,
        index_type: str = "exact",
    ) -> bool:
        """
        Create an index on entity property.
        
        Args:
            entity_type: Entity type/label
            property_name: Property to index
            index_type: Index type ('exact', 'fulltext', 'vector')
            
        Returns:
            True if created successfully
        """
        pass

    @abstractmethod
    async def get_schema(self) -> Dict[str, Any]:
        """
        Get graph schema information.
        
        Returns:
            Schema information including entity types,
            relationship types, and indexes
        """
        pass

    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    @abstractmethod
    async def bulk_create_entities(
        self,
        entities: List[GraphEntity],
        batch_size: int = 1000,
    ) -> List[str]:
        """
        Bulk create entities.
        
        Args:
            entities: Entities to create
            batch_size: Batch size for operations
            
        Returns:
            List of created entity IDs
        """
        pass

    @abstractmethod
    async def bulk_create_relationships(
        self,
        relationships: List[GraphRelationship],
        batch_size: int = 1000,
    ) -> List[str]:
        """
        Bulk create relationships.
        
        Args:
            relationships: Relationships to create
            batch_size: Batch size for operations
            
        Returns:
            List of created relationship IDs
        """
        pass

    # =========================================================================
    # Maintenance Operations
    # =========================================================================
    
    @abstractmethod
    async def clear_graph(self) -> bool:
        """
        Clear all data from the graph.
        
        WARNING: This deletes all entities and relationships!
        
        Returns:
            True if cleared successfully
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Statistics including entity counts, relationship counts, etc.
        """
        pass


class GraphDatabaseFactory:
    """Factory for creating graph database instances."""
    
    _databases: Dict[str, type[GraphDatabase]] = {}
    
    @classmethod
    def register(
        cls,
        db_type: str,
        database_class: type[GraphDatabase],
    ) -> None:
        """
        Register a graph database implementation.
        
        Args:
            db_type: Database type identifier
            database_class: Database class
        """
        cls._databases[db_type] = database_class
        logger.info(f"Registered graph database type: {db_type}")
    
    @classmethod
    def create(
        cls,
        db_type: str,
        **kwargs,
    ) -> GraphDatabase:
        """
        Create a graph database instance.
        
        Args:
            db_type: Database type
            **kwargs: Database-specific configuration
            
        Returns:
            Graph database instance
            
        Raises:
            ValueError: If database type not registered
        """
        if db_type not in cls._databases:
            raise ValueError(
                f"Unknown graph database type: {db_type}. "
                f"Available types: {list(cls._databases.keys())}"
            )
        
        database_class = cls._databases[db_type]
        return database_class(**kwargs)