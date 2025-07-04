"""
Graph database components for knowledge graph support.

This package provides interfaces and implementations for graph database
operations used in building and querying knowledge graphs from PDF content.
"""

from src.graph_db.base import GraphDatabase, GraphDatabaseFactory
from src.graph_db.neo4j_adapter import Neo4jAdapter

__all__ = ["GraphDatabase", "GraphDatabaseFactory", "Neo4jAdapter"]