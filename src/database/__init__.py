"""
Database management package for the Ijon RAG system.

This package provides centralized database connection management,
health monitoring, and resource optimization.
"""

from .connection_manager import (
    DatabaseConnectionManager,
    ConnectionHealth,
    get_connection_manager,
    initialize_database,
    close_database
)

__all__ = [
    'DatabaseConnectionManager',
    'ConnectionHealth', 
    'get_connection_manager',
    'initialize_database',
    'close_database'
]