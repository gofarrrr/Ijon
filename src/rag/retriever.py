"""
Document retrieval module for RAG pipeline.

This module handles searching for relevant documents using vector similarity
and metadata filtering, with support for hybrid search and re-ranking.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.config import get_settings
from src.models import Document, PDFChunk, SearchQuery, SearchResult
from src.utils.errors import RetrievalError, VectorDatabaseError
from src.utils.logging import get_logger, log_performance
from src.vector_db.base import VectorDatabase, VectorDatabaseFactory
from src.rag.embedder import EmbeddingGenerator

logger = get_logger(__name__)


class DocumentRetriever:
    """
    Retrieve relevant documents using vector search and filtering.
    
    Supports:
    - Vector similarity search
    - Metadata filtering
    - Hybrid search (vector + keyword)
    - Result re-ranking
    - Query expansion
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        rerank: bool = True,
        expand_query: bool = False,
    ) -> None:
        """
        Initialize document retriever.

        Args:
            vector_db: Vector database instance
            embedding_generator: Embedding generator instance
            rerank: Whether to re-rank results
            expand_query: Whether to expand queries
        """
        self.settings = get_settings()
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.rerank = rerank
        self.expand_query = expand_query
        
        # Re-ranker will be loaded lazily if needed
        self._reranker = None

    async def initialize(self) -> None:
        """Initialize retriever components."""
        # Initialize vector database if not provided
        if not self.vector_db:
            self.vector_db = VectorDatabaseFactory.create(
                self.settings.vector_db_type,
                embedding_function=self.embedding_generator.generate_embeddings,
            )
            await self.vector_db.initialize()
        
        logger.info("Document retriever initialized")

    @log_performance
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            include_scores: Whether to include similarity scores

        Returns:
            List of search results

        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            # Expand query if configured
            if self.expand_query:
                queries = await self._expand_query(query)
                logger.debug(f"Expanded query to {len(queries)} variations")
            else:
                queries = [query]
            
            # Perform vector search for each query
            all_results = []
            for q in queries:
                # Generate query embedding
                query_embedding = await self.embedding_generator.generate_embedding(q)
                
                # Search vector database
                results = await self.vector_db.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2 if self.rerank else top_k,  # Get more if re-ranking
                    filters=filters,
                )
                
                all_results.extend(results)
            
            # Deduplicate results by document ID
            unique_results = self._deduplicate_results(all_results)
            
            # Re-rank if configured
            if self.rerank and len(unique_results) > 1:
                unique_results = await self._rerank_results(query, unique_results)
            
            # Return top K results
            final_results = unique_results[:top_k]
            
            logger.info(
                f"Retrieved {len(final_results)} documents for query",
                extra={"query_length": len(query), "top_k": top_k},
            )
            
            return final_results
            
        except VectorDatabaseError as e:
            logger.error(f"Vector database error during retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {e}")
            raise RetrievalError(f"Retrieval failed: {str(e)}")

    @log_performance
    async def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[PDFChunk, float]]:
        """
        Retrieve relevant PDF chunks with scores.

        Args:
            query: Search query
            top_k: Number of chunks to return
            min_score: Minimum similarity score
            filters: Metadata filters

        Returns:
            List of (chunk, score) tuples

        Raises:
            RetrievalError: If retrieval fails
        """
        # Retrieve documents
        results = await self.retrieve(query, top_k * 2, filters)
        
        # Convert to chunks with filtering
        chunks_with_scores = []
        
        for result in results:
            if result.score >= min_score:
                # Reconstruct PDFChunk from document
                chunk = PDFChunk(
                    id=result.document.id,
                    pdf_id=result.document.metadata.get("pdf_id", ""),
                    content=result.document.content,
                    page_numbers=result.document.metadata.get("page_numbers", [1]),
                    chunk_index=result.document.metadata.get("chunk_index", 0),
                    metadata=result.document.metadata,
                    word_count=result.document.metadata.get("word_count", 0),
                    char_count=result.document.metadata.get("char_count", len(result.document.content)),
                    section_title=result.document.metadata.get("section_title"),
                    chapter_title=result.document.metadata.get("chapter_title"),
                )
                chunks_with_scores.append((chunk, result.score))
        
        # Sort by score and limit
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        return chunks_with_scores[:top_k]

    async def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with variations for better recall.

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        variations = [query]
        
        # Simple query expansion techniques
        # TODO: Implement more sophisticated expansion with synonyms, etc.
        
        # Add question variations
        if not query.endswith("?"):
            variations.append(query + "?")
        
        # Add statement version
        if query.endswith("?"):
            variations.append(query.rstrip("?"))
        
        return variations

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Deduplicate search results by document ID.

        Args:
            results: List of search results

        Returns:
            Deduplicated results
        """
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.document.id not in seen_ids:
                seen_ids.add(result.document.id)
                unique_results.append(result)
        
        return unique_results

    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Re-rank search results for better relevance.

        Args:
            query: Original query
            results: Initial search results

        Returns:
            Re-ranked results
        """
        # For now, use a simple re-ranking based on multiple factors
        # TODO: Implement cross-encoder re-ranking for better accuracy
        
        reranked = []
        
        for result in results:
            # Calculate additional relevance signals
            doc = result.document
            
            # Length penalty - prefer chunks of moderate length
            ideal_length = 500
            length_score = 1.0 - abs(len(doc.content) - ideal_length) / ideal_length * 0.2
            length_score = max(0.0, length_score)
            
            # Query term overlap
            query_terms = set(query.lower().split())
            doc_terms = set(doc.content.lower().split())
            overlap_score = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
            
            # Combine scores (vector similarity + other signals)
            combined_score = (
                result.score * 0.7 +  # Vector similarity
                overlap_score * 0.2 +  # Term overlap
                length_score * 0.1    # Length preference
            )
            
            # Create new result with updated score
            reranked_result = SearchResult(
                document=doc,
                score=combined_score,
                rank=result.rank,
            )
            reranked.append(reranked_result)
        
        # Sort by combined score
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        return reranked

    async def find_similar_chunks(
        self,
        chunk: PDFChunk,
        top_k: int = 5,
        exclude_same_pdf: bool = True,
    ) -> List[Tuple[PDFChunk, float]]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk: Reference chunk
            top_k: Number of similar chunks to return
            exclude_same_pdf: Whether to exclude chunks from the same PDF

        Returns:
            List of (similar_chunk, score) tuples
        """
        # Use chunk content as query
        query = chunk.content
        
        # Add filter to exclude same PDF if requested
        filters = None
        if exclude_same_pdf:
            filters = {"pdf_id": {"$ne": chunk.pdf_id}}
        
        # Retrieve similar chunks
        similar_chunks = await self.retrieve_chunks(
            query=query,
            top_k=top_k + 1,  # Get extra in case we need to exclude the same chunk
            filters=filters,
        )
        
        # Exclude the same chunk
        similar_chunks = [
            (c, score) for c, score in similar_chunks
            if c.id != chunk.id
        ]
        
        return similar_chunks[:top_k]

    async def get_context_window(
        self,
        chunk: PDFChunk,
        window_size: int = 2,
    ) -> List[PDFChunk]:
        """
        Get surrounding chunks for context.

        Args:
            chunk: Central chunk
            window_size: Number of chunks before/after

        Returns:
            List of chunks in order
        """
        # Create filters for same PDF and nearby chunks
        filters = {
            "pdf_id": chunk.pdf_id,
            "chunk_index": {
                "$gte": chunk.chunk_index - window_size,
                "$lte": chunk.chunk_index + window_size,
            },
        }
        
        # Search for nearby chunks
        results = await self.vector_db.search(
            query_embedding=[0.0] * self.embedding_generator.dimension,  # Dummy embedding
            top_k=window_size * 2 + 1,
            filters=filters,
        )
        
        # Convert to chunks and sort by index
        chunks = []
        for result in results:
            chunk_obj = PDFChunk(
                id=result.document.id,
                pdf_id=result.document.metadata.get("pdf_id", ""),
                content=result.document.content,
                page_numbers=result.document.metadata.get("page_numbers", [1]),
                chunk_index=result.document.metadata.get("chunk_index", 0),
                metadata=result.document.metadata,
                word_count=result.document.metadata.get("word_count", 0),
                char_count=result.document.metadata.get("char_count", 0),
            )
            chunks.append(chunk_obj)
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.chunk_index)
        
        return chunks


class HybridRetriever(DocumentRetriever):
    """
    Enhanced retriever with hybrid search capabilities.
    
    Combines vector search with keyword search for better results.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        """
        Initialize hybrid retriever.

        Args:
            vector_db: Vector database instance
            embedding_generator: Embedding generator
            vector_weight: Weight for vector search scores
            keyword_weight: Weight for keyword search scores
        """
        super().__init__(vector_db, embedding_generator, rerank=True)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    @log_performance
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
    ) -> List[SearchResult]:
        """
        Perform hybrid retrieval combining vector and keyword search.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            include_scores: Whether to include scores

        Returns:
            List of search results
        """
        # Get vector search results
        vector_results = await super().retrieve(
            query=query,
            top_k=top_k * 3,  # Get more for merging
            filters=filters,
            include_scores=True,
        )
        
        # TODO: Implement keyword search when we have full-text search capability
        # For now, just use vector results with our re-ranking
        
        return vector_results[:top_k]


def create_retriever(hybrid: bool = False) -> DocumentRetriever:
    """
    Create a document retriever.

    Args:
        hybrid: Whether to use hybrid retrieval

    Returns:
        Document retriever instance
    """
    if hybrid:
        return HybridRetriever()
    return DocumentRetriever()