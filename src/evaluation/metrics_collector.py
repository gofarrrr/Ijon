"""
Unified metrics collection system for the Ijon RAG pipeline.

This module provides centralized metrics collection that integrates with
existing logging and monitoring systems.
"""

import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
import json
from pathlib import Path

import asyncpg
from dataclasses import dataclass, asdict

import os

from src.config import get_settings
from src.utils.logging import get_logger
from src.database import get_connection_manager

logger = get_logger(__name__)


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    component: str
    operation: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MetricsCollector:
    """
    Centralized metrics collection for all RAG components.
    
    Integrates with existing logging and stores metrics in PostgreSQL.
    """
    
    def __init__(self, use_shared_pool: bool = True):
        """
        Initialize metrics collector.
        
        Args:
            use_shared_pool: Whether to use the shared connection manager (recommended)
        """
        self.settings = get_settings()
        self.use_shared_pool = use_shared_pool
        self.connection_manager = get_connection_manager() if use_shared_pool else None
        
        # Legacy pool for backward compatibility
        self.pool = None
        
        self._metrics_buffer = []
        self._buffer_size = 100
        self._flush_interval = 60  # seconds
        self._last_flush = time.time()
        
        # In-memory aggregations for quick access
        self._aggregations = defaultdict(lambda: {
            'count': 0,
            'sum': 0,
            'min': float('inf'),
            'max': float('-inf'),
            'latest': 0
        })
    
    async def initialize(self):
        """Initialize database connection and create metrics table."""
        if self.use_shared_pool:
            # Use shared connection manager - no guard needed, manager handles it
            await self.connection_manager.initialize()
            await self._create_schema()
            logger.info("Metrics collector initialized (using shared pool)")
            return
        
        # Legacy pool initialization with guard
        if self.pool is not None:
            logger.debug("Metrics collector already initialized, skipping")
            return
            
        try:
            connection_string = os.getenv("DATABASE_URL")
            self.pool = await asyncpg.create_pool(
                connection_string,
                min_size=2,
                max_size=10,
                timeout=30,  # Add timeout to prevent hangs
                command_timeout=10  # Command timeout
            )
            
            await self._create_schema()
            logger.info("Metrics collector initialized (legacy pool)")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics collector: {e}")
            # Clean up on failure
            if self.pool:
                await self.pool.close()
                self.pool = None
            raise
    
    async def _create_schema(self):
        """Create metrics storage table."""
        if self.use_shared_pool:
            async with self.connection_manager.get_connection() as conn:
                await self._execute_schema_creation(conn)
        else:
            async with self.pool.acquire() as conn:
                await self._execute_schema_creation(conn)
    
    async def _execute_schema_creation(self, conn):
        """Execute schema creation commands."""
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    value FLOAT NOT NULL,
                    component TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    metadata JSONB DEFAULT '{}'
                );
                
                -- Create indexes separately
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component);
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
                
                -- Create partitions for time-series data
                CREATE TABLE IF NOT EXISTS metrics_hourly (
                    component TEXT,
                    operation TEXT,
                    metric_name TEXT,
                    hour TIMESTAMP,
                    count INTEGER,
                    sum FLOAT,
                    min FLOAT,
                    max FLOAT,
                    avg FLOAT,
                    p50 FLOAT,
                    p95 FLOAT,
                    p99 FLOAT,
                    PRIMARY KEY (component, operation, metric_name, hour)
                );
        """)
    
    async def record_metric(
        self,
        name: str,
        value: float,
        component: str,
        operation: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Record a single metric.
        
        Args:
            name: Metric name (e.g., 'latency_ms', 'token_count')
            value: Metric value
            component: Component name (e.g., 'reranker', 'entity_extractor')
            operation: Operation name (e.g., 'hybrid_rerank', 'extract_entities')
            metadata: Additional context
        """
        metric = Metric(
            name=name,
            value=value,
            component=component,
            operation=operation,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Add to buffer
        self._metrics_buffer.append(metric)
        
        # Update in-memory aggregations
        key = f"{component}.{operation}.{name}"
        agg = self._aggregations[key]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['latest'] = value
        
        # Flush if needed
        if len(self._metrics_buffer) >= self._buffer_size or \
           time.time() - self._last_flush > self._flush_interval:
            await self.flush_metrics()
    
    async def flush_metrics(self):
        """Flush buffered metrics to database."""
        if not self._metrics_buffer:
            return
            
        if self.use_shared_pool and not self.connection_manager:
            logger.warning("Shared connection manager not available, skipping flush")
            return
            
        if not self.use_shared_pool and not self.pool:
            logger.warning("Legacy pool not available, skipping flush")
            return
        
        buffer_size = len(self._metrics_buffer)
        try:
            if self.use_shared_pool:
                async with self.connection_manager.get_connection() as conn:
                    await self._execute_flush(conn, buffer_size)
            else:
                async with self.pool.acquire() as conn:
                    await self._execute_flush(conn, buffer_size)
                
        except Exception as e:
            logger.error(f"Failed to flush {buffer_size} metrics: {e}")
            # Clear buffer anyway to prevent memory leaks, but keep some for retry
            if len(self._metrics_buffer) > self._buffer_size * 2:
                # If buffer is getting too large, discard oldest metrics
                self._metrics_buffer = self._metrics_buffer[-self._buffer_size:]
                logger.warning(f"Discarded old metrics to prevent memory leak")
    
    async def _execute_flush(self, conn, buffer_size):
        """Execute the actual flush operation."""
        # Batch insert metrics
        await conn.executemany(
            """
            INSERT INTO metrics (name, value, component, operation, timestamp, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            [
                (m.name, m.value, m.component, m.operation, m.timestamp, json.dumps(m.metadata))
                for m in self._metrics_buffer
            ]
        )
        
        logger.debug(f"Flushed {buffer_size} metrics to database")
        self._metrics_buffer.clear()
        self._last_flush = time.time()
    
    def get_aggregated_metrics(self, component: str = None, operation: str = None) -> Dict[str, Any]:
        """Get aggregated metrics from memory."""
        results = {}
        
        for key, agg in self._aggregations.items():
            parts = key.split('.')
            if component and not key.startswith(f"{component}."):
                continue
            if operation and not key.startswith(f"{component}.{operation}."):
                continue
            
            results[key] = {
                'count': agg['count'],
                'average': agg['sum'] / agg['count'] if agg['count'] > 0 else 0,
                'min': agg['min'] if agg['min'] != float('inf') else 0,
                'max': agg['max'] if agg['max'] != float('-inf') else 0,
                'latest': agg['latest']
            }
        
        return results
    
    async def get_metrics_summary(
        self,
        component: str = None,
        operation: str = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get metrics summary from database."""
        if not self.pool:
            return {}
        
        query = """
            SELECT 
                component,
                operation,
                name as metric_name,
                COUNT(*) as count,
                AVG(value) as avg,
                MIN(value) as min,
                MAX(value) as max,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as p50,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY value) as p99
            FROM metrics
            WHERE timestamp > NOW() - INTERVAL '%s hours'
        """
        
        conditions = []
        params = [hours]
        
        if component:
            conditions.append("component = $2")
            params.append(component)
        if operation:
            conditions.append("operation = $3")
            params.append(operation)
        
        if conditions:
            query += " AND " + " AND ".join(conditions)
        
        query += " GROUP BY component, operation, name"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
        return {
            'summary': [dict(row) for row in rows],
            'time_range_hours': hours
        }
    
    async def close(self):
        """Close database connections."""
        await self.flush_metrics()
        
        if self.use_shared_pool:
            # Don't close shared connection manager - other components may be using it
            logger.debug("Metrics collector closed (shared pool remains active)")
        else:
            # Close legacy pool
            if self.pool:
                await self.pool.close()
                self.pool = None
                logger.debug("Metrics collector closed (legacy pool closed)")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Decorator for automatic metrics collection
def collect_metrics(component: str, operation: str):
    """
    Decorator to automatically collect metrics for a function.
    
    Collects:
    - Execution time (latency_ms)
    - Success/failure (success_rate)
    - Any exceptions (error_count)
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                latency = (time.time() - start_time) * 1000
                await collector.record_metric(
                    'latency_ms', latency, component, operation
                )
                await collector.record_metric(
                    'success_rate', 1.0, component, operation
                )
                
                return result
                
            except Exception as e:
                # Record failure metrics
                latency = (time.time() - start_time) * 1000
                await collector.record_metric(
                    'latency_ms', latency, component, operation
                )
                await collector.record_metric(
                    'success_rate', 0.0, component, operation
                )
                await collector.record_metric(
                    'error_count', 1.0, component, operation,
                    metadata={'error': str(e)}
                )
                raise
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async metrics
            # Just execute the function
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator