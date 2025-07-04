## FEATURE:

Build a comprehensive PDF data extraction and RAG (Retrieval-Augmented Generation) system with the following capabilities:

### Core Components:

1. **Google Drive Integration & PDF Data Extraction Pipeline**
   - Connect to Google Drive API for PDF storage and retrieval
   - List, search, and download PDFs from specified Drive folders
   - Track processing status of PDFs in Drive (processed/unprocessed)
   - Extract text, images, tables, and metadata from PDF books
   - Handle complex document structures (chapters, sections, subsections)
   - Support for OCR when dealing with scanned PDFs
   - Preserve document hierarchy and formatting information
   - Extract and associate images with their context
   - Handle multi-column layouts and academic paper formats
   - Sync processing results back to Drive as metadata or companion files

2. **MCP (Model Context Protocol) Server**
   - Local MCP server implementation for terminal-based access
   - Expose tools for PDF processing, data extraction, and RAG queries
   - Support for concurrent operations and queue management
   - Authentication and rate limiting for API access
   - WebSocket support for real-time processing updates

3. **RAG Data Preparation**
   - Semantic chunking with configurable chunk sizes and overlap
   - Metadata preservation (page numbers, sections, document info)
   - Vector embeddings generation using state-of-the-art models
   - Support for multiple embedding strategies (sentence, paragraph, semantic)
   - Hierarchical indexing for better retrieval accuracy

4. **Automated Question Generation & Answering**
   - Generate diverse question types (factual, analytical, comparative)
   - Batch question processing with progress tracking
   - Context-aware answer generation with source citations
   - Quality scoring for generated Q&A pairs
   - Export capabilities for Q&A datasets

5. **CLI Interface**
   - Command: `ijon connect` - Authenticate with Google Drive
   - Command: `ijon list-pdfs <folder_id>` - List PDFs in Drive folder
   - Command: `ijon extract <file_id|folder_id>` - Extract and process PDF(s) from Drive
   - Command: `ijon sync` - Sync all unprocessed PDFs from configured Drive folders
   - Command: `ijon serve` - Start MCP server
   - Command: `ijon query <question>` - Query the RAG system
   - Command: `ijon generate-qa <num_questions>` - Generate Q&A pairs
   - Command: `ijon export <format>` - Export data (json, csv, parquet)
   - Progress bars and real-time status updates
   - Batch processing support for multiple PDFs from Drive folders

6. **Terminal Agent Integration**
   - MCP tools for external agents (like Claude Code) to:
     - Browse Google Drive folders and list PDFs
     - Submit PDFs from Drive for processing
     - Query the knowledge base
     - Generate questions programmatically
     - Retrieve processing status
     - Access extracted data and metadata
     - Trigger batch processing of Drive folders

## EXAMPLES:

In the `examples/` folder, create the following reference implementations:

- `examples/google_drive/` - Google Drive integration patterns
  - `auth.py` - OAuth2 authentication flow
  - `client.py` - Drive API client wrapper
  - `sync.py` - Folder synchronization logic

- `examples/mcp_server/` - MCP server implementation patterns
  - `server.py` - Basic MCP server structure
  - `tools.py` - MCP tool definitions including Drive operations
  - `handlers.py` - Request handling patterns

- `examples/pdf_processor/` - PDF processing patterns
  - `extractor.py` - Text and metadata extraction
  - `chunker.py` - Semantic chunking strategies
  - `preprocessor.py` - Data cleaning and normalization

- `examples/rag_pipeline/` - RAG implementation patterns
  - `embedder.py` - Embedding generation
  - `retriever.py` - Vector search implementation
  - `generator.py` - Answer generation with citations

- `examples/cli.py` - CLI structure following project conventions
- `examples/agent/` - Agent patterns for question generation
- `examples/tests/` - Testing patterns for each component

## DOCUMENTATION:

### Google Drive Integration:
- Google Drive API v3: https://developers.google.com/drive/api/v3/about-sdk
- Google Auth Python: https://google-auth.readthedocs.io/en/latest/
- PyDrive2: https://docs.iterative.ai/PyDrive2/ - Simplified Google Drive API wrapper
- Google API Python Client: https://github.com/googleapis/google-api-python-client

### PDF Processing Libraries:
- PyMuPDF (fitz): https://pymupdf.readthedocs.io/ - Fast and comprehensive PDF processing
- pdfplumber: https://github.com/jsvine/pdfplumber - Table extraction capabilities
- pdf2image: https://pypi.org/project/pdf2image/ - PDF to image conversion
- pytesseract: https://pypi.org/project/pytesseract/ - OCR for scanned PDFs

### MCP Server Resources:
- MCP Specification: https://modelcontextprotocol.io/docs/specification
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP Server Examples: https://github.com/modelcontextprotocol/servers

### Vector Databases & RAG:
- ChromaDB: https://docs.trychroma.com/ - Embedded vector database
- LangChain: https://python.langchain.com/docs/get_started/introduction - RAG framework
- LlamaIndex: https://docs.llamaindex.ai/en/stable/ - Alternative RAG framework
- Sentence Transformers: https://www.sbert.net/ - Embedding models

### Supporting Tools:
- Pydantic AI: https://ai.pydantic.dev/ - For agent implementation
- Rich: https://rich.readthedocs.io/ - Terminal UI components
- Typer: https://typer.tiangolo.com/ - CLI framework
- FastAPI: https://fastapi.tiangolo.com/ - For API endpoints if needed

## OTHER CONSIDERATIONS:

### Google Drive Specific:
- **Authentication**: OAuth2 flow for user consent, service account for automated access
- **Rate Limits**: Google Drive API has quotas (1000 queries/100 seconds/user)
- **File Size Limits**: Large PDFs may need chunked download/upload
- **Permissions**: Handle shared drives and different permission levels
- **Change Detection**: Use Drive API change tokens to track new/modified PDFs
- **Metadata Storage**: Store processing results as Drive file properties or companion JSON files

### Technical Challenges:
- **PDF Parsing Complexity**: Books may have complex layouts, footnotes, references, mathematical formulas
- **Memory Management**: Large PDFs can consume significant memory; implement streaming where possible
- **OCR Quality**: Scanned PDFs may have varying quality; implement confidence scoring
- **Table Extraction**: Complex tables may need special handling and structured output formats
- **Drive Sync**: Handle network interruptions during large file downloads

### Performance & Scalability:
- **Chunk Size Optimization**: Balance between context preservation and retrieval accuracy
- **Embedding Model Selection**: Trade-offs between model size, speed, and quality
- **Vector Database Choice**: Consider embedded vs. hosted, persistence needs, query performance
- **Concurrent Processing**: Handle multiple PDF extractions simultaneously
- **Caching Strategy**: Cache embeddings and processed data to avoid recomputation

### RAG-Specific Considerations:
- **Context Window Management**: Handle long documents that exceed LLM context limits
- **Source Attribution**: Maintain clear references to source documents and page numbers
- **Answer Quality**: Implement verification mechanisms for generated answers
- **Question Diversity**: Ensure generated questions cover different aspects and difficulty levels

### Integration Requirements:
- **MCP Authentication**: Implement secure token-based auth for MCP server
- **Rate Limiting**: Prevent abuse of compute-intensive operations
- **Progress Tracking**: WebSocket or SSE for real-time progress updates
- **Error Recovery**: Graceful handling of PDF corruption, network issues, API failures

### Data Privacy & Security:
- **Local Processing**: Ensure all processing can run locally without external API calls (except for LLMs)
- **Data Isolation**: Implement user/session-based data isolation in the MCP server
- **Secure Storage**: Encrypt sensitive extracted data at rest

### Development Setup:
- Include comprehensive .env.example with all configuration options
- Document Google Cloud project setup and API enablement steps
- Provide OAuth2 consent screen configuration guide
- Document GPU requirements for local embedding models
- Provide docker-compose.yml for easy deployment
- Setup instructions for Google Drive API credentials (client_secrets.json)
- Memory profiling tools integration for optimization
- Example Drive folder structure for organizing PDFs by category/topic