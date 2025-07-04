"""
Tests for Pinecone vector database adapter.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import Document, SearchResult
from src.utils.errors import (
    EmbeddingGenerationError,
    VectorDatabaseConnectionError,
    VectorDatabaseError,
    VectorDatabaseNotInitializedError,
)
from src.vector_db.pinecone_adapter import PineconeVectorDB


class TestPineconeVectorDB:
    """Test Pinecone vector database adapter."""

    @pytest.fixture
    def mock_pinecone_client(self):
        """Mock Pinecone client."""
        with patch("src.vector_db.pinecone_adapter.pinecone") as mock:
            # Mock client
            mock_client = MagicMock()
            mock.Pinecone.return_value = mock_client
            
            # Mock list_indexes
            mock_index_info = MagicMock()
            mock_index_info.name = "test-index"
            mock_client.list_indexes.return_value = [mock_index_info]
            
            # Mock describe_index
            mock_index_desc = MagicMock()
            mock_index_desc.status.ready = True
            mock_client.describe_index.return_value = mock_index_desc
            
            # Mock Index
            mock_index = MagicMock()
            mock_client.Index.return_value = mock_index
            
            # Mock index stats
            mock_index.describe_index_stats.return_value = {
                "total_vector_count": 100,
                "dimension": 384,
                "namespaces": {"default": {"vector_count": 100}},
            }
            
            yield mock, mock_client, mock_index

    @pytest.fixture
    async def pinecone_db(self, mock_embedding_function):
        """Create Pinecone DB instance."""
        db = PineconeVectorDB(
            api_key="test-api-key",
            environment="test-env",
            index_name="test-index",
            dimension=384,
            embedding_function=mock_embedding_function,
        )
        return db

    @pytest.mark.asyncio
    async def test_initialize_existing_index(self, pinecone_db, mock_pinecone_client):
        """Test initialization with existing index."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        
        await pinecone_db.initialize()
        
        assert pinecone_db._initialized
        assert pinecone_db.index is not None
        mock_client.create_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_create_index(self, pinecone_db, mock_pinecone_client):
        """Test initialization creating new index."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        
        # Mock no existing indexes
        mock_client.list_indexes.return_value = []
        
        await pinecone_db.initialize()
        
        assert pinecone_db._initialized
        mock_client.create_index.assert_called_once()
        create_call = mock_client.create_index.call_args
        assert create_call.kwargs["name"] == "test-index"
        assert create_call.kwargs["dimension"] == 384

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, pinecone_db, mock_pinecone_client):
        """Test initialization with connection error."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        
        # Mock connection error
        mock_pinecone.Pinecone.side_effect = Exception("Connection failed")
        
        with pytest.raises(VectorDatabaseConnectionError):
            await pinecone_db.initialize()

    @pytest.mark.asyncio
    async def test_upsert_documents(self, pinecone_db, mock_pinecone_client, sample_documents):
        """Test upserting documents."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Mock upsert response
        mock_index.upsert.return_value = {"upserted_count": len(sample_documents)}
        
        await pinecone_db.upsert_documents(sample_documents)
        
        # Verify upsert was called
        mock_index.upsert.assert_called()
        upsert_call = mock_index.upsert.call_args
        vectors = upsert_call.kwargs["vectors"]
        
        assert len(vectors) == len(sample_documents)
        assert vectors[0]["id"] == sample_documents[0].id
        assert vectors[0]["values"] == sample_documents[0].embedding

    @pytest.mark.asyncio
    async def test_upsert_documents_generate_embeddings(
        self, pinecone_db, mock_pinecone_client, mock_embedding_function
    ):
        """Test upserting documents with embedding generation."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Create documents without embeddings
        docs = [
            Document(id="doc-1", content="Test content 1", metadata={}),
            Document(id="doc-2", content="Test content 2", metadata={}),
        ]
        
        await pinecone_db.upsert_documents(docs)
        
        # Verify embeddings were generated
        mock_embedding_function.assert_called_once()
        call_args = mock_embedding_function.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == "Test content 1"

    @pytest.mark.asyncio
    async def test_upsert_documents_not_initialized(self, pinecone_db):
        """Test upserting documents when not initialized."""
        with pytest.raises(VectorDatabaseNotInitializedError):
            await pinecone_db.upsert_documents([])

    @pytest.mark.asyncio
    async def test_upsert_documents_no_embedding_function(self, pinecone_db, mock_pinecone_client):
        """Test upserting documents without embeddings or embedding function."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        pinecone_db.embedding_function = None
        
        # Create document without embedding
        doc = Document(id="doc-1", content="Test content", metadata={})
        
        with pytest.raises(EmbeddingGenerationError):
            await pinecone_db.upsert_documents([doc])

    @pytest.mark.asyncio
    async def test_search(self, pinecone_db, mock_pinecone_client):
        """Test vector search."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Mock query response
        mock_match = MagicMock()
        mock_match.id = "doc-1"
        mock_match.score = 0.95
        mock_match.metadata = {"content": "Test content", "page": 1}
        mock_match.values = [0.1] * 384
        
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response
        
        # Perform search
        query_embedding = [0.2] * 384
        results = await pinecone_db.search(query_embedding, top_k=5)
        
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].document.id == "doc-1"
        assert results[0].score == 0.95
        assert results[0].rank == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, pinecone_db, mock_pinecone_client):
        """Test vector search with metadata filters."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        mock_response = MagicMock()
        mock_response.matches = []
        mock_index.query.return_value = mock_response
        
        # Perform search with filters
        query_embedding = [0.2] * 384
        filters = {"page": {"$gte": 5}}
        results = await pinecone_db.search(query_embedding, top_k=10, filters=filters)
        
        # Verify query was called with filters
        query_call = mock_index.query.call_args
        assert query_call.kwargs["filter"] == filters

    @pytest.mark.asyncio
    async def test_search_by_text(self, pinecone_db, mock_pinecone_client, mock_embedding_function):
        """Test text-based search."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Mock query response
        mock_response = MagicMock()
        mock_response.matches = []
        mock_index.query.return_value = mock_response
        
        # Perform text search
        results = await pinecone_db.search_by_text("test query", top_k=3)
        
        # Verify embedding was generated for query
        mock_embedding_function.assert_called_once()
        call_args = mock_embedding_function.call_args[0][0]
        assert call_args == ["test query"]

    @pytest.mark.asyncio
    async def test_delete_documents(self, pinecone_db, mock_pinecone_client):
        """Test deleting documents."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Delete documents
        ids_to_delete = ["doc-1", "doc-2", "doc-3"]
        count = await pinecone_db.delete_documents(ids_to_delete)
        
        assert count == 3
        mock_index.delete.assert_called_once_with(
            ids=ids_to_delete,
            namespace="default"
        )

    @pytest.mark.asyncio
    async def test_get_document(self, pinecone_db, mock_pinecone_client):
        """Test retrieving a single document."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Mock fetch response
        mock_vector = MagicMock()
        mock_vector.metadata = {"content": "Test content", "page": 1}
        mock_vector.values = [0.1] * 384
        
        mock_response = MagicMock()
        mock_response.vectors = {"doc-1": mock_vector}
        mock_index.fetch.return_value = mock_response
        
        # Get document
        doc = await pinecone_db.get_document("doc-1")
        
        assert doc is not None
        assert doc.id == "doc-1"
        assert doc.content == "Test content"
        assert doc.metadata["page"] == 1

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, pinecone_db, mock_pinecone_client):
        """Test retrieving non-existent document."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Mock empty fetch response
        mock_response = MagicMock()
        mock_response.vectors = {}
        mock_index.fetch.return_value = mock_response
        
        # Get document
        doc = await pinecone_db.get_document("non-existent")
        
        assert doc is None

    @pytest.mark.asyncio
    async def test_count_documents(self, pinecone_db, mock_pinecone_client):
        """Test counting documents."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Already mocked in fixture to return 100
        count = await pinecone_db.count_documents()
        
        assert count == 100

    @pytest.mark.asyncio
    async def test_clear(self, pinecone_db, mock_pinecone_client):
        """Test clearing all documents."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        await pinecone_db.clear()
        
        mock_index.delete.assert_called_once_with(
            delete_all=True,
            namespace="default"
        )

    @pytest.mark.asyncio
    async def test_close(self, pinecone_db, mock_pinecone_client):
        """Test closing connection."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        await pinecone_db.close()
        
        assert pinecone_db.index is None
        assert pinecone_db._pc is None
        assert not pinecone_db._initialized

    @pytest.mark.asyncio
    async def test_batch_upsert(self, pinecone_db, mock_pinecone_client):
        """Test batch upserting large number of documents."""
        mock_pinecone, mock_client, mock_index = mock_pinecone_client
        pinecone_db._initialized = True
        pinecone_db.index = mock_index
        
        # Create 250 documents (should be split into 3 batches)
        docs = [
            Document(
                id=f"doc-{i}",
                content=f"Content {i}",
                metadata={"index": i},
                embedding=[0.1] * 384
            )
            for i in range(250)
        ]
        
        await pinecone_db.upsert_documents(docs)
        
        # Verify upsert was called 3 times (100, 100, 50)
        assert mock_index.upsert.call_count == 3