"""
Abstract base interface for vector database implementations.

This module defines the interface that all vector database adapters must implement,
allowing for easy switching between different vector database providers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from src.models import Document, SearchQuery, SearchResult
from src.utils.errors import VectorDatabaseError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingFunction(Protocol):
    """Protocol for embedding generation functions."""

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        ...


class VectorDatabase(ABC):
    """
    Abstract base class for vector database implementations.
    
    This class defines the interface that all vector database adapters
    (Neon, Supabase, Pinecone) must implement.
    """

    def __init__(self, embedding_function: Optional[EmbeddingFunction] = None) -> None:
        """
        Initialize the vector database.

        Args:
            embedding_function: Optional function to generate embeddings
        """
        self.embedding_function = embedding_function
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the database connection and create necessary tables/indexes.
        
        This method should:
        - Establish connection to the database
        - Create tables/collections if they don't exist
        - Create indexes for efficient search
        - Validate configuration
        
        Raises:
            VectorDatabaseConnectionError: If connection fails
            VectorDatabaseError: For other initialization errors
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the database connection and clean up resources.
        
        This method should:
        - Close all active connections
        - Clean up any temporary resources
        - Log closure information
        """
        pass

    @abstractmethod
    async def upsert_documents(self, documents: List[Document]) -> None:
        """
        Insert or update documents in the vector database.
        
        If documents have embeddings, use them. Otherwise, generate embeddings
        using the configured embedding function.
        
        Args:
            documents: List of documents to upsert
            
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            EmbeddingGenerationError: If embedding generation fails
            VectorDatabaseError: For other database errors
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results ordered by similarity
            
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            VectorSearchError: If search operation fails
        """
        pass

    @abstractmethod
    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using text query.
        
        This method should generate embeddings for the query text
        and then perform vector search.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results ordered by similarity
            
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            EmbeddingGenerationError: If embedding generation fails
            VectorSearchError: If search operation fails
        """
        pass

    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
            
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            VectorDatabaseError: For database errors
        """
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a single document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            VectorDatabaseError: For database errors
        """
        pass

    @abstractmethod
    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the database.
        
        Args:
            filters: Optional metadata filters
            
        Returns:
            Number of documents matching filters
            
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            VectorDatabaseError: For database errors
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear all documents from the database.
        
        Use with caution! This will delete all data.
        
        Raises:
            VectorDatabaseNotInitializedError: If database not initialized
            VectorDatabaseError: For database errors
        """
        pass

    # =============================================================================
    # Helper Methods (implemented in base class)
    # =============================================================================

    def ensure_initialized(self) -> None:
        """
        Ensure the database is initialized before operations.
        
        Raises:
            VectorDatabaseNotInitializedError: If not initialized
        """
        if not self._initialized:
            from src.utils.errors import VectorDatabaseNotInitializedError
            raise VectorDatabaseNotInitializedError(
                "Vector database not initialized. Call initialize() first."
            )

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
            
        Raises:
            EmbeddingGenerationError: If embedding function not set or fails
        """
        if not self.embedding_function:
            from src.utils.errors import EmbeddingGenerationError
            raise EmbeddingGenerationError(
                "No embedding function configured. "
                "Either provide embeddings or set embedding_function."
            )
        
        try:
            embeddings = await self.embedding_function(texts)
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            from src.utils.errors import EmbeddingGenerationError
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {str(e)}")

    async def __aenter__(self) -> "VectorDatabase":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


class VectorDatabaseFactory:
    """Factory for creating vector database instances."""

    _databases: Dict[str, type[VectorDatabase]] = {}

    @classmethod
    def register(cls, name: str, database_class: type[VectorDatabase]) -> None:
        """
        Register a vector database implementation.
        
        Args:
            name: Name of the database (e.g., 'pinecone', 'neon')
            database_class: Database implementation class
        """
        cls._databases[name.lower()] = database_class
        logger.info(f"Registered vector database: {name}")

    @classmethod
    def create(
        cls,
        name: str,
        embedding_function: Optional[EmbeddingFunction] = None,
        **kwargs: Any,
    ) -> VectorDatabase:
        """
        Create a vector database instance.
        
        Args:
            name: Name of the database to create
            embedding_function: Optional embedding function
            **kwargs: Additional arguments for the database
            
        Returns:
            Vector database instance
            
        Raises:
            ValueError: If database name not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._databases:
            available = ", ".join(cls._databases.keys())
            raise ValueError(
                f"Unknown vector database: {name}. "
                f"Available: {available}"
            )
        
        database_class = cls._databases[name_lower]
        return database_class(embedding_function=embedding_function, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Get list of registered database names."""
        return list(cls._databases.keys())