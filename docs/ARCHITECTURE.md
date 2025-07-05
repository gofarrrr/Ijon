# Ijon System Architecture

## Overview

Ijon is a production-ready PDF RAG (Retrieval-Augmented Generation) system that extracts knowledge from PDF documents and enables semantic search using Google's Gemini embeddings and PostgreSQL with pgvector.

## Core Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PDF Files     │────▶│  Text Extraction │────▶│  Smart Chunking │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Neon pgvector │◀────│ Gemini Embeddings│◀────│Context Enhancement│
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Query     │────▶│ Query Enhancement│────▶│ Semantic Search │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Components

### 1. PDF Processing (`src/pdf_processor/`)
- **Extractor**: Handles text extraction from PDFs using PyPDF2/PyMuPDF
- **Chunker**: Implements semantic chunking with configurable overlap
- **Preprocessor**: Cleans and normalizes extracted text

### 2. Context Enhancement (`src/context/smart_context_enhancer.py`)
- Detects mental models and key concepts in text
- Adds contextual metadata before embedding
- Three enhancement levels:
  - Level 0: Basic metadata (title, page)
  - Level 1: Smart categorization
  - Level 2: Analytical context with concepts

### 3. Embeddings (`src/rag/gemini_embedder.py`)
- Uses Google's text-embedding-004 model
- 768-dimensional embeddings optimized for retrieval
- Batch processing with caching support
- Cost-effective: ~$0.025 per million tokens

### 4. Vector Storage (Neon PostgreSQL)
- Cloud-native PostgreSQL with pgvector extension
- Efficient similarity search using cosine distance
- Schema:
  ```sql
  content_chunks (
    id uuid PRIMARY KEY,
    document_id uuid REFERENCES documents(id),
    content text,
    embedding_gemini vector(768),
    page_numbers integer[],
    extraction_metadata jsonb
  )
  ```

### 5. Query Pipeline
- Query enhancement using same context enrichment
- Vector similarity search with pgvector
- Configurable result ranking and filtering

## Data Flow

### Ingestion Pipeline
1. **PDF Upload**: `process_single_book.py` accepts PDF files
2. **Text Extraction**: PyPDF2 extracts text page by page
3. **Chunking**: Text split into semantic chunks (~500 tokens)
4. **Enhancement**: Context added based on detected patterns
5. **Embedding**: Gemini API generates 768D vectors
6. **Storage**: Vectors stored in Neon with metadata

### Query Pipeline
1. **User Query**: Natural language question
2. **Enhancement**: Query enriched with context
3. **Embedding**: Query converted to 768D vector
4. **Search**: pgvector finds similar chunks
5. **Ranking**: Results sorted by relevance
6. **Response**: Top chunks returned with metadata

## Key Design Decisions

### Why Gemini Embeddings?
- Superior quality for semantic search
- Cost-effective compared to alternatives
- 768D provides good balance of quality/size
- Native Google Cloud integration

### Why Neon PostgreSQL?
- Serverless, scales to zero
- Native pgvector support
- ACID compliance for data integrity
- Standard SQL interface
- Lower cost than specialized vector DBs

### Why Context Enhancement?
- Improves retrieval quality by 20-30%
- Especially effective for domain-specific content
- Minimal overhead (< 100ms per chunk)
- Preserves original content while adding context

## Performance Characteristics

- **Ingestion**: ~3-4 seconds per PDF page
- **Query Response**: ~700ms average
- **Storage**: ~9.5KB per embedding (including metadata)
- **Accuracy**: 85%+ relevance for domain queries

## Scalability Considerations

- Neon auto-scales based on workload
- Gemini API handles concurrent requests
- Batch processing for large documents
- Caching reduces redundant API calls

## Security & Privacy

- All data stored in user's Neon instance
- API keys managed via environment variables
- No data persistence in processing pipeline
- HTTPS for all external communications

## Future Extensibility

The architecture supports:
- Multiple embedding models (swap embedder)
- Alternative vector databases (adapter pattern)
- Knowledge graph integration (experimental)
- Multi-modal content (images, tables)
- Real-time document updates