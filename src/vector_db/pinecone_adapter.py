"""
Pinecone vector database adapter implementation.

This module provides the Pinecone implementation of the VectorDatabase interface,
enabling efficient vector storage and similarity search.
"""

import asyncio
from typing import Any, Dict, List, Optional

import pinecone
from pinecone import ServerlessSpec

from src.config import get_settings
from src.models import Document, SearchResult
from src.utils.errors import (
    EmbeddingGenerationError,
    IndexNotFoundError,
    VectorDatabaseConnectionError,
    VectorDatabaseError,
    VectorSearchError,
)
from src.utils.logging import get_logger, log_performance
from src.vector_db.base import VectorDatabase, VectorDatabaseFactory

logger = get_logger(__name__)


class PineconeVectorDB(VectorDatabase):
    """Pinecone implementation of the VectorDatabase interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        namespace: Optional[str] = None,
        embedding_function=None,
    ) -> None:
        """
        Initialize Pinecone vector database.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
            dimension: Vector dimension (will be inferred if not provided)
            metric: Distance metric (cosine, euclidean, dotproduct)
            namespace: Namespace for data isolation
            embedding_function: Function to generate embeddings
        """
        super().__init__(embedding_function)
        
        self.settings = get_settings()
        self.api_key = api_key or self.settings.pinecone_api_key
        self.environment = environment or self.settings.pinecone_environment
        self.index_name = index_name or self.settings.pinecone_index_name
        self.dimension = dimension or self.settings.embedding_dimension
        self.metric = metric
        self.namespace = namespace or "default"
        
        self.index = None
        self._pc = None  # Pinecone client

    @log_performance
    async def initialize(self) -> None:
        """Initialize Pinecone connection and create index if needed."""
        if self._initialized:
            logger.debug("Pinecone already initialized")
            return
        
        try:
            logger.info(f"Initializing Pinecone with index: {self.index_name}")
            
            # Initialize Pinecone client
            self._pc = pinecone.Pinecone(api_key=self.api_key)
            
            # Check if index exists
            existing_indexes = self._pc.list_indexes()
            index_exists = any(
                idx.name == self.index_name for idx in existing_indexes
            )
            
            if not index_exists:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with serverless spec for the free tier
                self._pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment or "us-east-1",
                    ),
                )
                
                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                await self._wait_for_index_ready()
            
            # Connect to index
            self.index = self._pc.Index(self.index_name)
            
            # Verify index stats
            stats = self.index.describe_index_stats()
            logger.info(
                f"Connected to Pinecone index: {self.index_name}",
                extra={
                    "total_vectors": stats.get("total_vector_count", 0),
                    "dimension": stats.get("dimension", self.dimension),
                    "namespaces": list(stats.get("namespaces", {}).keys()),
                },
            )
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise VectorDatabaseConnectionError(
                f"Failed to connect to Pinecone: {str(e)}"
            )

    async def _wait_for_index_ready(self, timeout: int = 60) -> None:
        """Wait for index to be ready."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                # Check if index is ready
                index_info = self._pc.describe_index(self.index_name)
                if index_info.status.ready:
                    return
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        raise VectorDatabaseError(f"Index {self.index_name} not ready after {timeout} seconds")

    @log_performance
    async def close(self) -> None:
        """Close Pinecone connection."""
        logger.info("Closing Pinecone connection")
        self.index = None
        self._pc = None
        self._initialized = False

    @log_performance
    async def upsert_documents(self, documents: List[Document]) -> None:
        """
        Insert or update documents in Pinecone.

        Args:
            documents: List of documents to upsert
        """
        self.ensure_initialized()
        
        if not documents:
            return
        
        try:
            # Prepare documents for upsert
            vectors_to_upsert = []
            
            for doc in documents:
                # Generate embedding if not provided
                if doc.embedding is None:
                    if not self.embedding_function:
                        raise EmbeddingGenerationError(
                            "No embedding provided and no embedding function configured"
                        )
                    
                    embeddings = await self.generate_embeddings([doc.content])
                    doc.embedding = embeddings[0]
                
                # Prepare vector for Pinecone
                vector = {
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": {
                        "content": doc.content[:1000],  # Pinecone metadata size limit
                        **doc.metadata,
                    },
                }
                vectors_to_upsert.append(vector)
            
            # Batch upsert (Pinecone recommends max 100 vectors per batch)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                
                # Run upsert in thread pool (Pinecone client is sync)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.index.upsert(vectors=batch, namespace=self.namespace)
                )
                
                logger.debug(f"Upserted batch of {len(batch)} vectors")
            
            logger.info(
                f"Successfully upserted {len(documents)} documents",
                extra={"namespace": self.namespace},
            )
            
        except Exception as e:
            logger.error(f"Failed to upsert documents: {str(e)}")
            raise VectorDatabaseError(f"Failed to upsert documents: {str(e)}")

    @log_performance
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Metadata filters

        Returns:
            List of search results
        """
        self.ensure_initialized()
        
        try:
            # Prepare query
            query_params = {
                "vector": query_embedding,
                "top_k": top_k,
                "namespace": self.namespace,
                "include_metadata": True,
                "include_values": True,
            }
            
            if filters:
                query_params["filter"] = filters
            
            # Run query in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.query(**query_params)
            )
            
            # Convert to SearchResult objects
            results = []
            for i, match in enumerate(response.matches):
                doc = Document(
                    id=match.id,
                    content=match.metadata.get("content", ""),
                    metadata={
                        k: v for k, v in match.metadata.items() 
                        if k != "content"
                    },
                    embedding=match.values if hasattr(match, "values") else None,
                )
                
                result = SearchResult(
                    document=doc,
                    score=match.score,
                    rank=i + 1,
                )
                results.append(result)
            
            logger.debug(
                f"Found {len(results)} results",
                extra={"top_k": top_k, "namespace": self.namespace},
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise VectorSearchError(f"Search failed: {str(e)}")

    @log_performance
    async def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search using text query."""
        # Generate embedding for query text
        embeddings = await self.generate_embeddings([query_text])
        query_embedding = embeddings[0]
        
        # Perform vector search
        return await self.search(query_embedding, top_k, filters)

    @log_performance
    async def delete_documents(self, ids: List[str]) -> int:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        self.ensure_initialized()
        
        if not ids:
            return 0
        
        try:
            # Delete vectors
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.delete(ids=ids, namespace=self.namespace)
            )
            
            logger.info(f"Deleted {len(ids)} documents")
            return len(ids)
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise VectorDatabaseError(f"Failed to delete documents: {str(e)}")

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a single document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        self.ensure_initialized()
        
        try:
            # Fetch vector by ID
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.fetch(ids=[doc_id], namespace=self.namespace)
            )
            
            if doc_id in response.vectors:
                vector = response.vectors[doc_id]
                return Document(
                    id=doc_id,
                    content=vector.metadata.get("content", ""),
                    metadata={
                        k: v for k, v in vector.metadata.items() 
                        if k != "content"
                    },
                    embedding=vector.values if hasattr(vector, "values") else None,
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document: {str(e)}")
            raise VectorDatabaseError(f"Failed to get document: {str(e)}")

    async def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the index.

        Args:
            filters: Optional metadata filters

        Returns:
            Number of documents
        """
        self.ensure_initialized()
        
        try:
            stats = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.describe_index_stats()
            )
            
            # Get count for our namespace
            namespace_stats = stats.namespaces.get(self.namespace, {})
            count = namespace_stats.get("vector_count", 0)
            
            # Note: Pinecone doesn't support counting with filters directly
            if filters:
                logger.warning("Pinecone doesn't support filtered counts directly")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to count documents: {str(e)}")
            raise VectorDatabaseError(f"Failed to count documents: {str(e)}")

    async def clear(self) -> None:
        """Clear all documents from the namespace."""
        self.ensure_initialized()
        
        try:
            # Delete all vectors in namespace
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.index.delete(delete_all=True, namespace=self.namespace)
            )
            
            logger.warning(f"Cleared all documents from namespace: {self.namespace}")
            
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            raise VectorDatabaseError(f"Failed to clear documents: {str(e)}")


# Register Pinecone adapter with factory
VectorDatabaseFactory.register("pinecone", PineconeVectorDB)