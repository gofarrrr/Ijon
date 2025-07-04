name: "PDF Extraction & RAG System with Google Drive Integration"
description: |

## Purpose
Build a comprehensive PDF data extraction and RAG system that processes PDFs from Google Drive, creates searchable vector embeddings, and provides both CLI and MCP server interfaces for automated question-answering and knowledge extraction.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a production-ready system that:
- Extracts and processes PDFs from Google Drive automatically
- Builds a searchable RAG knowledge base using vector embeddings
- Provides MCP server for terminal-based AI agent access
- Enables automated question generation and answering
- Supports multiple vector databases (Neon, Supabase, Pinecone)

## Why
- **Business Value**: Transform static PDF libraries into queryable knowledge bases
- **User Impact**: Enable researchers/students to extract insights from large document collections
- **Integration**: Allow AI agents to access document knowledge via MCP protocol
- **Automation**: Replace manual document analysis with automated Q&A generation

## What
### User-visible behavior:
- CLI tool for PDF processing and querying
- MCP server accessible from terminal agents
- Real-time processing progress updates
- Export capabilities for extracted Q&A pairs

### Technical requirements:
- Google Drive API integration for PDF storage
- Vector database abstraction layer
- Streaming PDF processing for large files
- WebSocket support for progress tracking
- Concurrent processing capabilities

### Success Criteria
- [ ] Process PDFs >100MB without memory issues
- [ ] Generate embeddings with <2s latency per chunk
- [ ] Support 3 vector databases (Neon, Supabase, Pinecone)
- [ ] MCP server handles concurrent requests
- [ ] 95%+ accuracy in Q&A generation
- [ ] Full test coverage for critical paths

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window

# Google Drive Integration
- url: https://developers.google.com/drive/api/v3/quickstart/python
  why: OAuth2 setup and basic Drive operations
  
- url: https://developers.google.com/drive/api/v3/manage-downloads
  why: Streaming large file downloads, resumable downloads

# PDF Processing
- url: https://pymupdf.readthedocs.io/en/latest/tutorial.html
  why: Text extraction, layout preservation, image handling
  critical: Use fitz.open() with stream=True for memory efficiency

# Vector Databases
- url: https://neon.tech/docs/extensions/pgvector
  why: Neon pgvector setup and operations
  critical: Use connection pooling, index type affects performance

- url: https://supabase.com/docs/guides/ai/vector-embeddings
  why: Supabase vector operations and best practices
  
- url: https://docs.pinecone.io/docs/python-client
  why: Pinecone SDK usage and batch operations
  critical: Batch upserts for efficiency, namespace for isolation

# MCP Server
- url: https://modelcontextprotocol.io/docs/python/tutorial
  why: MCP server implementation in Python
  critical: Use async handlers for concurrent operations

- url: https://github.com/modelcontextprotocol/servers/tree/main/src/postgres
  why: Example MCP server with database operations

# RAG Framework
- url: https://python.langchain.com/docs/tutorials/rag/
  why: RAG pipeline implementation patterns
  
- url: https://docs.llamaindex.ai/en/stable/examples/low_level/rag.html
  why: Alternative RAG implementation with LlamaIndex

# Embeddings
- url: https://www.sbert.net/docs/quickstart.html
  why: Local embedding generation with sentence-transformers
  critical: Model selection affects quality/speed tradeoff

# Testing
- file: examples/tests/test_patterns.py
  why: Project testing conventions and mocking strategies
```

### Current Codebase tree
```bash
/Users/marcin/Desktop/aplikacje/Ijon/
├── CLAUDE.md
├── INITIAL.md
├── PRPs/
│   ├── pdf_rag_system.md (this file)
│   └── templates/
├── README.md
└── examples/
```

### Desired Codebase tree with files to be added
```bash
/Users/marcin/Desktop/aplikacje/Ijon/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Main CLI entry point
│   ├── config.py                 # Configuration management
│   ├── google_drive/
│   │   ├── __init__.py
│   │   ├── auth.py              # OAuth2 authentication
│   │   ├── client.py            # Drive API wrapper
│   │   └── sync.py              # Folder sync logic
│   ├── pdf_processor/
│   │   ├── __init__.py
│   │   ├── extractor.py         # PDF text/image extraction
│   │   ├── chunker.py           # Semantic chunking
│   │   └── preprocessor.py      # Text cleaning
│   ├── vector_db/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract vector DB interface
│   │   ├── neon_adapter.py      # Neon implementation
│   │   ├── supabase_adapter.py  # Supabase implementation
│   │   └── pinecone_adapter.py  # Pinecone implementation
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py          # Embedding generation
│   │   ├── retriever.py         # Vector search
│   │   └── generator.py         # Answer generation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── question_agent.py    # Q&A generation agent
│   │   ├── tools.py             # Agent tools
│   │   └── prompts.py           # System prompts
│   ├── mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py            # MCP server setup
│   │   ├── tools.py             # MCP tool definitions
│   │   └── handlers.py          # Request handlers
│   └── utils/
│       ├── __init__.py
│       ├── progress.py          # Progress tracking
│       └── errors.py            # Custom exceptions
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── test_google_drive.py
│   ├── test_pdf_processor.py
│   ├── test_vector_db.py
│   ├── test_rag.py
│   ├── test_mcp_server.py
│   └── test_cli.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: PyMuPDF memory management
# Always use context managers and stream mode for large PDFs
with fitz.open(pdf_path, stream=pdf_bytes, filetype="pdf") as doc:
    # Process pages in chunks to avoid memory issues
    
# CRITICAL: Google Drive API quotas
# Implement exponential backoff for rate limits
# Use batch requests when possible

# CRITICAL: Vector DB connections
# Neon/Supabase: Use connection pooling with pgvector
# Pinecone: Batch operations, max 100 vectors per request

# CRITICAL: MCP server async
# All MCP handlers must be async
# Use asyncio.create_task for background operations

# CRITICAL: Embedding model size
# all-MiniLM-L6-v2: Fast but lower quality (384 dims)
# all-mpnet-base-v2: Balanced (768 dims)
# Consider GPU availability for local models
```

## Implementation Blueprint

### Data models and structure

```python
# src/config.py
from pydantic import BaseSettings, Field
from typing import Literal, Optional

class Settings(BaseSettings):
    # Google Drive
    drive_credentials_path: str = Field(..., env="DRIVE_CREDENTIALS_PATH")
    drive_folder_ids: list[str] = Field(..., env="DRIVE_FOLDER_IDS")
    
    # Vector Database
    vector_db_type: Literal["neon", "supabase", "pinecone"] = Field(..., env="VECTOR_DB_TYPE")
    neon_connection_string: Optional[str] = Field(None, env="NEON_CONNECTION_STRING")
    supabase_url: Optional[str] = Field(None, env="SUPABASE_URL")
    supabase_key: Optional[str] = Field(None, env="SUPABASE_KEY")
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    
    # Processing
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # MCP Server
    mcp_host: str = Field("localhost", env="MCP_HOST")
    mcp_port: int = Field(8080, env="MCP_PORT")
    mcp_auth_token: Optional[str] = Field(None, env="MCP_AUTH_TOKEN")
    
    class Config:
        env_file = ".env"

# src/vector_db/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    document: Document
    score: float

class VectorDatabase(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database connection and create tables/indexes"""
        
    @abstractmethod
    async def upsert_documents(self, documents: List[Document]) -> None:
        """Insert or update documents with embeddings"""
        
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents"""
        
    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by ID"""
        
    @abstractmethod
    async def close(self) -> None:
        """Close database connection"""

# src/pdf_processor/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PDFMetadata(BaseModel):
    file_id: str
    filename: str
    drive_path: str
    total_pages: int
    processed_at: datetime
    file_size_bytes: int
    
class PDFChunk(BaseModel):
    id: str = Field(..., description="Unique chunk ID")
    pdf_id: str = Field(..., description="Source PDF ID")
    content: str = Field(..., description="Chunk text content")
    page_numbers: List[int] = Field(..., description="Pages this chunk spans")
    chunk_index: int = Field(..., description="Position in document")
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### List of tasks to be completed

```yaml
Task 1: Project Setup and Configuration
CREATE src/__init__.py:
  - Empty file for package initialization
  
CREATE src/config.py:
  - MIRROR pattern from: examples/config_patterns.py
  - Add comprehensive settings with validation
  - Include all environment variables

CREATE .env.example:
  - Include all required environment variables with descriptions
  - Add sample values for testing

CREATE pyproject.toml:
  - Setup project metadata and dependencies
  - Configure ruff, mypy, pytest

Task 2: Google Drive Integration
CREATE src/google_drive/auth.py:
  - Implement OAuth2 flow with google-auth
  - Handle token refresh automatically
  - Support both user and service account auth

CREATE src/google_drive/client.py:
  - Wrapper around Google Drive API
  - List PDFs in folders
  - Download with progress tracking
  - Handle rate limits with exponential backoff

CREATE src/google_drive/sync.py:
  - Track processed/unprocessed PDFs
  - Sync folder contents
  - Store processing status in Drive metadata

Task 3: PDF Processing Pipeline
CREATE src/pdf_processor/extractor.py:
  - Use PyMuPDF for text extraction
  - Handle multi-column layouts
  - Extract images with context
  - Preserve document structure

CREATE src/pdf_processor/chunker.py:
  - Implement semantic chunking
  - Respect sentence boundaries
  - Handle overlap correctly
  - Include metadata in chunks

CREATE src/pdf_processor/preprocessor.py:
  - Clean extracted text
  - Handle special characters
  - Normalize whitespace
  - Remove headers/footers

Task 4: Vector Database Abstraction
CREATE src/vector_db/base.py:
  - Define abstract interface
  - Common data models
  - Error handling

CREATE src/vector_db/neon_adapter.py:
  - Implement pgvector operations
  - Connection pooling
  - Batch operations

CREATE src/vector_db/supabase_adapter.py:
  - Similar to Neon but with Supabase client
  - Handle Supabase-specific features

CREATE src/vector_db/pinecone_adapter.py:
  - Use Pinecone Python client
  - Handle namespaces
  - Batch upserts

Task 5: RAG Pipeline
CREATE src/rag/embedder.py:
  - Load sentence-transformers model
  - Batch embedding generation
  - Handle GPU/CPU selection

CREATE src/rag/retriever.py:
  - Query vector database
  - Rerank results
  - Handle metadata filtering

CREATE src/rag/generator.py:
  - Generate answers with citations
  - Handle context window limits
  - Format responses

Task 6: Question Generation Agent
CREATE src/agents/question_agent.py:
  - Use Pydantic AI for agent
  - Generate diverse questions
  - Quality scoring

CREATE src/agents/tools.py:
  - Tools for document access
  - Search functionality

CREATE src/agents/prompts.py:
  - System prompts for Q&A generation
  - Different question types

Task 7: MCP Server Implementation
CREATE src/mcp_server/server.py:
  - FastAPI-based MCP server
  - WebSocket support
  - Authentication middleware

CREATE src/mcp_server/tools.py:
  - Define MCP tools:
    - list_pdfs: Browse Drive folders
    - process_pdf: Extract and index PDF
    - search_knowledge: Query RAG system
    - generate_questions: Create Q&A pairs
    - get_status: Processing status

CREATE src/mcp_server/handlers.py:
  - Async request handlers
  - Progress streaming
  - Error handling

Task 8: CLI Implementation
CREATE src/cli.py:
  - Use Typer for CLI
  - Commands for all operations
  - Progress bars with Rich
  - Interactive mode support

Task 9: Testing Suite
CREATE tests/conftest.py:
  - Pytest fixtures
  - Mock services
  - Test data

CREATE tests/test_*.py:
  - Unit tests for each module
  - Integration tests
  - E2E tests for critical paths

Task 10: Documentation and Deployment
UPDATE README.md:
  - Installation instructions
  - Usage examples
  - API documentation

CREATE Dockerfile:
  - Multi-stage build
  - Optimize for size

CREATE docker-compose.yml:
  - Service definitions
  - Environment configuration
```

### Per task pseudocode

```python
# Task 2: Google Drive Integration
# src/google_drive/client.py
class DriveClient:
    async def download_pdf_stream(self, file_id: str, progress_callback=None):
        """Stream download large PDFs with progress tracking"""
        request = self.service.files().get_media(fileId=file_id)
        
        # CRITICAL: Use resumable download for large files
        file_handle = io.BytesIO()
        downloader = MediaIoBaseDownload(file_handle, request, chunksize=1024*1024)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if progress_callback:
                await progress_callback(status.progress())
                
        file_handle.seek(0)
        return file_handle

# Task 3: PDF Processing
# src/pdf_processor/chunker.py
class SemanticChunker:
    def chunk_text(self, text: str, metadata: dict) -> List[PDFChunk]:
        """Smart chunking that respects sentence boundaries"""
        # PATTERN: Use spacy or nltk for sentence segmentation
        sentences = self.nlp(text).sents
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in sentences:
            sent_size = len(sent.text)
            
            # CRITICAL: Don't break mid-sentence
            if current_size + sent_size > self.chunk_size and current_chunk:
                # Create chunk with overlap from previous
                chunk_text = ' '.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata))
                
                # Keep last N sentences for overlap
                overlap_sents = current_chunk[-self.overlap_sentences:]
                current_chunk = overlap_sents
                current_size = sum(len(s) for s in overlap_sents)
            
            current_chunk.append(sent.text)
            current_size += sent_size

# Task 4: Vector Database - Neon
# src/vector_db/neon_adapter.py
class NeonVectorDB(VectorDatabase):
    async def initialize(self):
        """Create tables and indexes"""
        async with self.pool.acquire() as conn:
            # CRITICAL: Create extension first
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Create documents table with vector column
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(384),  -- Dimension depends on model
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            
            # CRITICAL: Create index for fast similarity search
            # Use ivfflat for large datasets, hnsw for smaller
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            ''')

# Task 5: RAG Pipeline
# src/rag/retriever.py
class HybridRetriever:
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Hybrid search combining vector and keyword search"""
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)
        
        # PATTERN: Combine vector and FTS results
        vector_results = await self.vector_db.search(query_embedding, top_k * 2)
        
        # CRITICAL: Rerank using cross-encoder for better accuracy
        if self.reranker:
            reranked = self.reranker.rerank(query, vector_results)
            return reranked[:top_k]
        
        return vector_results[:top_k]

# Task 7: MCP Server
# src/mcp_server/tools.py
@tool
async def process_pdf(file_id: str, webhook_url: Optional[str] = None) -> Dict:
    """Process PDF from Google Drive"""
    # PATTERN: Return job ID for async processing
    job_id = str(uuid.uuid4())
    
    # CRITICAL: Use background task for long operations
    asyncio.create_task(
        _process_pdf_background(job_id, file_id, webhook_url)
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "PDF processing started"
    }

# Task 8: CLI Implementation
# src/cli.py
@app.command()
def serve(
    host: str = typer.Option("localhost", help="MCP server host"),
    port: int = typer.Option(8080, help="MCP server port"),
):
    """Start the MCP server"""
    # PATTERN: Use uvicorn for production
    import uvicorn
    
    # CRITICAL: Enable access logs for debugging
    uvicorn.run(
        "src.mcp_server.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        # Enable auto-reload in dev
        reload=bool(os.getenv("DEV_MODE"))
    )
```

### Integration Points
```yaml
DATABASE:
  - Neon/Supabase:
    - CREATE EXTENSION vector
    - Create documents table with vector column
    - Create appropriate indexes (ivfflat or hnsw)
  - Pinecone:
    - Create index with appropriate dimensions
    - Configure replicas and shards

CONFIG:
  - add to: src/config.py
  - pattern: Load from env with pydantic BaseSettings
  - validation: Ensure required vars based on vector_db_type

MCP_SERVER:
  - add to: src/mcp_server/server.py
  - pattern: Use FastAPI with MCP protocol handler
  - middleware: Add auth, CORS, rate limiting

GOOGLE_DRIVE:
  - oauth2: Store tokens securely
  - service_account: Option for automated access
  - scopes: https://www.googleapis.com/auth/drive.readonly
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/ --fix          # Auto-fix style issues
mypy src/ --python-version 3.11  # Type checking
black src/ --check             # Code formatting

# Expected: No errors. If errors, READ and fix.
```

### Level 2: Unit Tests
```python
# tests/test_pdf_processor.py
@pytest.mark.asyncio
async def test_extract_pdf_with_images():
    """Test PDF extraction preserves images"""
    extractor = PDFExtractor()
    result = await extractor.extract("tests/data/sample_with_images.pdf")
    
    assert len(result.pages) > 0
    assert any(page.images for page in result.pages)
    assert result.metadata.total_pages == 10

# tests/test_vector_db.py
@pytest.mark.parametrize("db_type", ["neon", "supabase", "pinecone"])
@pytest.mark.asyncio
async def test_vector_search(db_type, mock_vector_db):
    """Test vector search across all adapters"""
    db = get_vector_db(db_type)
    await db.initialize()
    
    # Insert test documents
    docs = [Document(id=f"doc_{i}", content=f"Test content {i}", 
                    embedding=[0.1] * 384) for i in range(5)]
    await db.upsert_documents(docs)
    
    # Search
    results = await db.search([0.1] * 384, top_k=3)
    assert len(results) == 3
    assert all(r.score > 0.9 for r in results)

# tests/test_mcp_server.py
@pytest.mark.asyncio
async def test_mcp_tool_list_pdfs():
    """Test MCP tool for listing PDFs"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/tools/list_pdfs", 
            json={"folder_id": "test_folder"},
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert isinstance(data["files"], list)
```

```bash
# Run unit tests
python -m pytest tests/ -v --cov=src --cov-report=html

# If failing: Check coverage report, fix uncovered code
```

### Level 3: Integration Tests
```bash
# Start services
docker-compose up -d postgres  # For Neon/Supabase testing

# Run integration tests
python -m pytest tests/integration/ -v

# Test CLI commands
ijon connect --credentials-path creds.json
ijon list-pdfs test_folder_id
ijon extract sample_file_id --dry-run

# Test MCP server
ijon serve --dev &
curl -X POST http://localhost:8080/tools/search \
  -H "Authorization: Bearer $MCP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 5}'

# Expected: All commands work, server responds correctly
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check src/`
- [ ] No type errors: `mypy src/`
- [ ] Google Drive auth works: `ijon connect`
- [ ] PDF processing works: `ijon extract <file_id>`
- [ ] Vector search returns results: `ijon query "test"`
- [ ] MCP server accessible: `curl localhost:8080/health`
- [ ] Memory usage stable for 100MB+ PDFs
- [ ] Concurrent requests handled properly
- [ ] All vector DBs tested (Neon, Supabase, Pinecone)
- [ ] Documentation complete and accurate

---

## Anti-Patterns to Avoid
- ❌ Don't load entire PDFs into memory - use streaming
- ❌ Don't skip sentence boundaries in chunking
- ❌ Don't ignore Google Drive rate limits
- ❌ Don't hardcode vector dimensions - get from model
- ❌ Don't use sync functions in async context
- ❌ Don't store credentials in code
- ❌ Don't skip progress callbacks for long operations
- ❌ Don't ignore connection pooling for databases
- ❌ Don't catch all exceptions - be specific
- ❌ Don't forget to close resources properly