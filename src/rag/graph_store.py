"""
PostgreSQL-based graph storage for entity relationships.

This module implements a lean graph RAG system using PostgreSQL
instead of requiring a separate graph database like Neo4j.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
import asyncpg
from dataclasses import dataclass, asdict

from src.config import get_settings
from src.utils.errors import VectorDatabaseError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class Entity:
    """Entity extracted from document chunks."""
    id: Optional[int] = None
    name: str = ""
    type: str = ""
    description: Optional[str] = None
    properties: Dict[str, Any] = None
    chunk_ids: List[str] = None
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.chunk_ids is None:
            self.chunk_ids = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Relationship:
    """Relationship between entities."""
    id: Optional[int] = None
    source_entity_id: int = 0
    target_entity_id: int = 0
    type: str = ""
    properties: Dict[str, Any] = None
    chunk_ids: List[str] = None
    confidence: float = 1.0
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        if self.chunk_ids is None:
            self.chunk_ids = []
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class PostgresGraphStore:
    """
    PostgreSQL-based graph storage for entities and relationships.
    
    Uses existing Neon PostgreSQL database to store graph data
    without requiring additional infrastructure.
    """
    
    def __init__(self, connection_string: str = None):
        """Initialize PostgreSQL graph store."""
        self.settings = get_settings()
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.pool = None
        
    async def initialize(self):
        """Initialize connection pool and create tables."""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create schema
            await self._create_schema()
            
            logger.info("PostgreSQL graph store initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize graph store: {e}")
            raise VectorDatabaseError(f"Graph store initialization failed: {str(e)}")
    
    async def _create_schema(self):
        """Create tables for entities and relationships."""
        async with self.pool.acquire() as conn:
            # Enable pg_trgm extension for trigram similarity
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
            
            # Create entities table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    description TEXT,
                    properties JSONB DEFAULT '{}',
                    chunk_ids TEXT[] DEFAULT '{}',
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Indexes for efficient querying
                    UNIQUE(name, type)
                );
                
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
                CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin(name gin_trgm_ops);
                CREATE INDEX IF NOT EXISTS idx_entities_chunk_ids ON entities USING gin(chunk_ids);
            """)
            
            # Create relationships table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id SERIAL PRIMARY KEY,
                    source_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                    target_entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    properties JSONB DEFAULT '{}',
                    chunk_ids TEXT[] DEFAULT '{}',
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Prevent duplicate relationships
                    UNIQUE(source_entity_id, target_entity_id, type)
                );
                
                CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(type);
            """)
            
            # Skip creating functions for now - they seem to have syntax issues
            # We'll implement the logic directly in Python
            
            logger.info("Graph schema created successfully")
    
    @log_performance
    async def add_entity(self, entity: Entity) -> Entity:
        """Add or update an entity."""
        async with self.pool.acquire() as conn:
            # Use upsert to handle duplicates
            result = await conn.fetchrow("""
                INSERT INTO entities (name, type, description, properties, chunk_ids, confidence)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (name, type) 
                DO UPDATE SET
                    description = COALESCE(EXCLUDED.description, entities.description),
                    properties = entities.properties || EXCLUDED.properties,
                    chunk_ids = array_cat(entities.chunk_ids, EXCLUDED.chunk_ids),
                    confidence = GREATEST(entities.confidence, EXCLUDED.confidence)
                RETURNING *
            """, entity.name, entity.type, entity.description, 
                json.dumps(entity.properties), entity.chunk_ids, entity.confidence)
            
            entity.id = result['id']
            entity.created_at = result['created_at']
            
            return entity
    
    @log_performance
    async def add_relationship(self, relationship: Relationship) -> Relationship:
        """Add or update a relationship."""
        async with self.pool.acquire() as conn:
            # Use upsert to handle duplicates
            result = await conn.fetchrow("""
                INSERT INTO relationships 
                (source_entity_id, target_entity_id, type, properties, chunk_ids, confidence)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_entity_id, target_entity_id, type)
                DO UPDATE SET
                    properties = relationships.properties || EXCLUDED.properties,
                    chunk_ids = array_cat(relationships.chunk_ids, EXCLUDED.chunk_ids),
                    confidence = GREATEST(relationships.confidence, EXCLUDED.confidence)
                RETURNING *
            """, relationship.source_entity_id, relationship.target_entity_id,
                relationship.type, json.dumps(relationship.properties),
                relationship.chunk_ids, relationship.confidence)
            
            relationship.id = result['id']
            relationship.created_at = result['created_at']
            
            return relationship
    
    async def find_entities(
        self,
        name_query: str = None,
        entity_type: str = None,
        chunk_id: str = None,
        limit: int = 10
    ) -> List[Entity]:
        """Find entities by name, type, or chunk."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM entities WHERE 1=1"
            params = []
            
            if name_query:
                params.append(f"%{name_query}%")
                query += f" AND LOWER(name) LIKE LOWER(${len(params)})"  # Simple LIKE for now
                
            if entity_type:
                params.append(entity_type)
                query += f" AND type = ${len(params)}"
                
            if chunk_id:
                params.append(chunk_id)
                query += f" AND ${len(params)} = ANY(chunk_ids)"
            
            query += f" ORDER BY confidence DESC LIMIT {limit}"
            
            rows = await conn.fetch(query, *params)
            
            return [
                Entity(
                    id=row['id'],
                    name=row['name'],
                    type=row['type'],
                    description=row['description'],
                    properties=row['properties'],
                    chunk_ids=row['chunk_ids'],
                    confidence=row['confidence'],
                    created_at=row['created_at']
                )
                for row in rows
            ]
    
    async def get_entity_graph(
        self,
        entity_id: int,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get entity's graph neighborhood."""
        async with self.pool.acquire() as conn:
            # Simple implementation without recursive CTE
            nodes = {}
            edges = []
            
            # Get starting entity
            entity = await conn.fetchrow(
                "SELECT * FROM entities WHERE id = $1", entity_id
            )
            
            if entity:
                nodes[entity_id] = {
                    'id': entity_id,
                    'name': entity['name'],
                    'type': entity['type'],
                    'depth': 0
                }
            
            # Get immediate relationships (depth 1)
            if max_depth >= 1:
                relationships = await conn.fetch("""
                    SELECT r.*, e1.name as source_name, e1.type as source_type,
                           e2.name as target_name, e2.type as target_type
                    FROM relationships r
                    JOIN entities e1 ON r.source_entity_id = e1.id
                    JOIN entities e2 ON r.target_entity_id = e2.id
                    WHERE r.source_entity_id = $1 OR r.target_entity_id = $1
                """, entity_id)
                
                for rel in relationships:
                    # Add related entities
                    if rel['source_entity_id'] != entity_id:
                        nodes[rel['source_entity_id']] = {
                            'id': rel['source_entity_id'],
                            'name': rel['source_name'],
                            'type': rel['source_type'],
                            'depth': 1
                        }
                    if rel['target_entity_id'] != entity_id:
                        nodes[rel['target_entity_id']] = {
                            'id': rel['target_entity_id'],
                            'name': rel['target_name'],
                            'type': rel['target_type'],
                            'depth': 1
                        }
                    
                    edges.append({
                        'source': rel['source_entity_id'],
                        'target': rel['target_entity_id'],
                        'type': rel['type'],
                        'depth': 1
                    })
            
            return {
                'center_entity_id': entity_id,
                'nodes': list(nodes.values()),
                'edges': edges,
                'max_depth': max_depth
            }
    
    async def find_path(
        self,
        source_entity_id: int,
        target_entity_id: int,
        max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities (simplified BFS)."""
        if source_entity_id == target_entity_id:
            return []
            
        async with self.pool.acquire() as conn:
            # Simple BFS implementation
            queue = [(source_entity_id, [source_entity_id])]
            visited = {source_entity_id}
            
            while queue and len(visited) < 1000:  # Safety limit
                current_id, path = queue.pop(0)
                
                if len(path) > max_depth:
                    continue
                
                # Get neighbors
                neighbors = await conn.fetch("""
                    SELECT DISTINCT 
                        CASE 
                            WHEN source_entity_id = $1 THEN target_entity_id
                            ELSE source_entity_id
                        END as neighbor_id
                    FROM relationships
                    WHERE source_entity_id = $1 OR target_entity_id = $1
                """, current_id)
                
                for row in neighbors:
                    neighbor_id = row['neighbor_id']
                    
                    if neighbor_id == target_entity_id:
                        # Found path!
                        full_path = path + [neighbor_id]
                        
                        # Build path with entity details
                        result = []
                        for i, entity_id in enumerate(full_path):
                            entity = await conn.fetchrow(
                                "SELECT * FROM entities WHERE id = $1", entity_id
                            )
                            if entity:
                                result.append({
                                    'entity_id': entity_id,
                                    'name': entity['name'],
                                    'type': entity['type'],
                                    'position': i
                                })
                        
                        return result
                    
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))
            
            return None
    
    async def get_related_chunks(
        self,
        entity_ids: List[int],
        limit: int = 20
    ) -> List[str]:
        """Get all chunk IDs related to given entities."""
        async with self.pool.acquire() as conn:
            # Get chunks from entities
            entity_chunks = await conn.fetch("""
                SELECT DISTINCT unnest(chunk_ids) as chunk_id
                FROM entities
                WHERE id = ANY($1::INT[])
            """, entity_ids)
            
            # Get chunks from relationships
            rel_chunks = await conn.fetch("""
                SELECT DISTINCT unnest(chunk_ids) as chunk_id
                FROM relationships
                WHERE source_entity_id = ANY($1::INT[]) 
                   OR target_entity_id = ANY($1::INT[])
            """, entity_ids)
            
            # Combine and deduplicate
            all_chunks = set()
            for row in entity_chunks + rel_chunks:
                if row['chunk_id']:
                    all_chunks.add(row['chunk_id'])
            
            return list(all_chunks)[:limit]
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL graph store closed")


def create_graph_store() -> PostgresGraphStore:
    """Create a PostgreSQL graph store instance."""
    return PostgresGraphStore()


# Enable trigram extension for similarity search
import os