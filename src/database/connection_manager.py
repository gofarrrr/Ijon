"""
Centralized database connection manager for the Ijon RAG system.

This module provides a single, shared connection pool that all components
can use, eliminating connection pool proliferation and improving resource efficiency.
"""

import asyncio
import os
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import asyncpg
from dataclasses import dataclass

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectionHealth:
    """Connection pool health status."""
    status: str
    total_connections: int
    idle_connections: int
    active_connections: int
    last_check: datetime
    error_count: int = 0
    last_error: Optional[str] = None


class DatabaseConnectionManager:
    """
    Centralized database connection manager with health monitoring.
    
    Provides a single, shared connection pool that all components can use,
    eliminating the need for each component to manage its own pool.
    
    Features:
    - Single shared connection pool
    - Health monitoring and auto-recovery
    - Connection retry logic
    - Resource cleanup guarantees
    - Performance monitoring
    """
    
    def __init__(self, connection_string: str = None):
        """Initialize connection manager."""
        self.settings = get_settings()
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.pool: Optional[asyncpg.Pool] = None
        self._health = ConnectionHealth(
            status="uninitialized",
            total_connections=0,
            idle_connections=0,
            active_connections=0,
            last_check=datetime.utcnow()
        )
        self._initialization_lock = asyncio.Lock()
        self._is_closing = False
        
        # Configuration
        self.min_connections = 3
        self.max_connections = 15
        self.connection_timeout = 30
        self.command_timeout = 60
        self.retry_attempts = 3
        self.retry_delay = 2.0
    
    async def initialize(self) -> None:
        """Initialize the shared connection pool."""
        async with self._initialization_lock:
            if self.pool is not None:
                logger.debug("Database connection manager already initialized")
                return
                
            if self._is_closing:
                raise RuntimeError("Connection manager is being closed")
            
            try:
                logger.info("Initializing shared database connection pool...")
                
                self.pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=self.min_connections,
                    max_size=self.max_connections,
                    timeout=self.connection_timeout,
                    command_timeout=self.command_timeout,
                    server_settings={
                        'application_name': 'ijon_rag_system',
                        'tcp_keepalives_idle': '600',
                        'tcp_keepalives_interval': '30',
                        'tcp_keepalives_count': '3'
                    }
                )
                
                # Test the connection
                await self._health_check()
                
                logger.info(f"Database connection pool initialized: {self.min_connections}-{self.max_connections} connections")
                self._health.status = "healthy"
                
            except Exception as e:
                logger.error(f"Failed to initialize database connection pool: {e}")
                self._health.status = "failed"
                self._health.error_count += 1
                self._health.last_error = str(e)
                
                # Clean up on failure
                if self.pool:
                    await self.pool.close()
                    self.pool = None
                raise
    
    async def _health_check(self) -> None:
        """Perform health check on the connection pool."""
        if not self.pool:
            self._health.status = "uninitialized"
            return
            
        try:
            # Quick connectivity test
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            # Update health metrics
            self._health.total_connections = self.pool.get_size()
            self._health.idle_connections = self.pool.get_idle_size()
            self._health.active_connections = self._health.total_connections - self._health.idle_connections
            self._health.last_check = datetime.utcnow()
            self._health.status = "healthy"
            
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            self._health.status = "unhealthy"
            self._health.error_count += 1
            self._health.last_error = str(e)
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database connection from the pool.
        
        Usage:
            async with connection_manager.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
        """
        if not self.pool:
            await self.initialize()
        
        if self._is_closing:
            raise RuntimeError("Connection manager is being closed")
        
        retry_count = 0
        while retry_count < self.retry_attempts:
            try:
                async with self.pool.acquire() as conn:
                    yield conn
                return
                
            except Exception as e:
                retry_count += 1
                self._health.error_count += 1
                self._health.last_error = str(e)
                
                if retry_count >= self.retry_attempts:
                    logger.error(f"Failed to acquire database connection after {self.retry_attempts} attempts: {e}")
                    self._health.status = "failed"
                    raise
                
                logger.warning(f"Database connection attempt {retry_count} failed: {e}, retrying...")
                await asyncio.sleep(self.retry_delay * retry_count)
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query with connection retry logic."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def execute_many(self, query: str, args_list: list) -> None:
        """Execute a query multiple times with different parameters."""
        async with self.get_connection() as conn:
            await conn.executemany(query, args_list)
    
    async def fetch_all(self, query: str, *args) -> list:
        """Fetch all results from a query."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetch_one(self, query: str, *args) -> Optional[Any]:
        """Fetch one result from a query."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    def get_health(self) -> ConnectionHealth:
        """Get current connection pool health status."""
        return self._health
    
    async def refresh_health(self) -> ConnectionHealth:
        """Refresh and return health status."""
        await self._health_check()
        return self._health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed connection pool statistics."""
        if not self.pool:
            return {
                "status": "uninitialized",
                "total_connections": 0,
                "configuration": {
                    "min_connections": self.min_connections,
                    "max_connections": self.max_connections,
                    "connection_timeout": self.connection_timeout,
                    "command_timeout": self.command_timeout
                }
            }
        
        return {
            "status": self._health.status,
            "total_connections": self.pool.get_size(),
            "idle_connections": self.pool.get_idle_size(),
            "active_connections": self.pool.get_size() - self.pool.get_idle_size(),
            "error_count": self._health.error_count,
            "last_error": self._health.last_error,
            "last_check": self._health.last_check.isoformat(),
            "configuration": {
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "connection_timeout": self.connection_timeout,
                "command_timeout": self.command_timeout,
                "retry_attempts": self.retry_attempts,
                "retry_delay": self.retry_delay
            }
        }
    
    async def close(self) -> None:
        """Close the connection pool and clean up resources."""
        async with self._initialization_lock:
            if self.pool is None:
                logger.debug("Database connection manager already closed")
                return
            
            self._is_closing = True
            
            try:
                logger.info("Closing shared database connection pool...")
                await self.pool.close()
                self.pool = None
                self._health.status = "closed"
                logger.info("Database connection pool closed successfully")
                
            except Exception as e:
                logger.error(f"Error closing database connection pool: {e}")
                self._health.status = "error"
                raise
            finally:
                self._is_closing = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global connection manager instance
_connection_manager: Optional[DatabaseConnectionManager] = None


def get_connection_manager() -> DatabaseConnectionManager:
    """Get the global database connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = DatabaseConnectionManager()
    return _connection_manager


async def initialize_database():
    """Initialize the global database connection manager."""
    manager = get_connection_manager()
    await manager.initialize()
    return manager


async def close_database():
    """Close the global database connection manager."""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.close()
        _connection_manager = None