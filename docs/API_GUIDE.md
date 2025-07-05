# Ijon API Guide

## Quick Start

```python
# Process a PDF
python process_single_book.py path/to/book.pdf

# Query the system
python query_mental_models.py "What is flow state?"
```

## Core APIs

### 1. PDF Processing

#### Command Line
```bash
# Process with default settings
python process_single_book.py book.pdf

# Limit pages for large PDFs
python process_single_book.py book.pdf --max-pages 100

# Specify context enhancement level
python process_single_book.py book.pdf --context-level 2
```

#### Python API
```python
from src.rag.gemini_embedder import GeminiEmbeddingGenerator
from src.context.smart_context_enhancer import SmartContextEnhancer
from src.pdf_processor.extractor import PDFExtractor

# Initialize components
embedder = GeminiEmbeddingGenerator()
enhancer = SmartContextEnhancer()
extractor = PDFExtractor()

# Process PDF
metadata, pages = await extractor.extract_from_file("book.pdf")
for page in pages:
    # Enhance content
    enhanced = enhancer.add_context(page.text, metadata, level=1)
    
    # Generate embedding
    embedding = await embedder.generate_embeddings([enhanced])
    
    # Store in database (see Database API)
```

### 2. Query System

#### Command Line
```bash
# Interactive mode
python query_mental_models.py

# Direct query
python query_mental_models.py "How to achieve flow state?"

# Demo mode
python query_mental_models.py demo
```

#### Python API
```python
async def search_knowledge(query: str, top_k: int = 5):
    """Search the knowledge base."""
    # Enhance query
    enhanced_query = enhancer.add_context(query, query_metadata, level=1)
    
    # Generate embedding
    query_embedding = await embedder.generate_embeddings([enhanced_query])
    
    # Search database
    results = await search_similar_chunks(query_embedding[0], top_k)
    
    return results
```

### 3. Database Operations

#### Connection
```python
import asyncpg
import os

# Get connection string from environment
database_url = os.getenv("DATABASE_URL") or os.getenv("NEON_CONNECTION_STRING")
conn = await asyncpg.connect(database_url)
```

#### Storing Embeddings
```python
async def store_chunk(conn, doc_id, content, embedding, metadata):
    """Store a chunk with its embedding."""
    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
    
    await conn.execute("""
        INSERT INTO content_chunks 
        (document_id, content, embedding_gemini, extraction_metadata)
        VALUES ($1, $2, $3::vector, $4)
    """, doc_id, content, embedding_str, json.dumps(metadata))
```

#### Searching Embeddings
```python
async def search_similar_chunks(conn, query_embedding, top_k=5):
    """Search for similar chunks using pgvector."""
    query_vector = '[' + ','.join(map(str, query_embedding)) + ']'
    
    results = await conn.fetch("""
        SELECT 
            content,
            embedding_gemini <-> $1::vector as distance,
            extraction_metadata
        FROM content_chunks
        WHERE embedding_gemini IS NOT NULL
        ORDER BY embedding_gemini <-> $1::vector
        LIMIT $2
    """, query_vector, top_k)
    
    return results
```

### 4. Embeddings API

#### Basic Usage
```python
from src.rag.gemini_embedder import GeminiEmbeddingGenerator

embedder = GeminiEmbeddingGenerator()

# Single text
embedding = await embedder.generate_embeddings(["Hello world"])

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = await embedder.generate_embeddings(texts)

# Check dimensions
print(f"Embedding size: {len(embeddings[0])}")  # 768
```

#### With Caching
```python
# Enable caching (default)
embedder = GeminiEmbeddingGenerator(use_cache=True)

# Clear cache if needed
embedder.clear_cache()
```

### 5. Context Enhancement API

#### Basic Enhancement
```python
from src.context.smart_context_enhancer import SmartContextEnhancer, ContextMetadata

enhancer = SmartContextEnhancer()

# Create metadata
metadata = ContextMetadata(
    title="Book Title",
    page_num=42,
    category="Psychology"
)

# Enhance text
enhanced = enhancer.add_context("Original text", metadata, level=1)
```

#### Enhancement Levels
- **Level 0**: Basic metadata (title, page)
- **Level 1**: Smart categorization and mental model detection
- **Level 2**: Full analytical context with concepts

### 6. Environment Configuration

Required environment variables:
```bash
# Gemini API (required)
GEMINI_API_KEY=your-gemini-api-key

# Database (required)
DATABASE_URL=postgresql://user:pass@host/db
# or
NEON_CONNECTION_STRING=postgresql://user:pass@host/db

# Optional
LOG_LEVEL=INFO
CACHE_DIR=.cache
```

## Advanced Usage

### Batch Processing
```python
# Process multiple PDFs
pdf_files = ["book1.pdf", "book2.pdf", "book3.pdf"]

for pdf_file in pdf_files:
    print(f"Processing {pdf_file}...")
    # Use process_single_book.py logic
```

### Custom Chunking
```python
from src.pdf_processor.chunker import SemanticChunker

chunker = SemanticChunker(
    chunk_size=1000,      # tokens
    chunk_overlap=200,    # tokens
    min_chunk_size=100    # tokens
)

chunks = await chunker.chunk_text(text, metadata)
```

### Error Handling
```python
from src.utils.errors import EmbeddingGenerationError, PDFProcessingError

try:
    embeddings = await embedder.generate_embeddings(texts)
except EmbeddingGenerationError as e:
    logger.error(f"Embedding failed: {e}")
    # Handle error
```

## Performance Tips

1. **Batch Operations**: Process multiple texts in one API call
2. **Enable Caching**: Reuse embeddings for repeated texts
3. **Limit Context**: Use appropriate enhancement level
4. **Page Limits**: Process large PDFs in chunks
5. **Connection Pooling**: Reuse database connections

## Common Patterns

### Full Pipeline Example
```python
async def process_and_search():
    # 1. Process PDF
    await process_pdf("book.pdf")
    
    # 2. Search knowledge
    results = await search_knowledge("What is flow state?")
    
    # 3. Format results
    for result in results:
        print(f"Score: {1 - result['distance']:.3f}")
        print(f"Content: {result['content'][:200]}...")
```

### Integration Example
```python
# Use in a web API
from fastapi import FastAPI

app = FastAPI()

@app.post("/search")
async def search_endpoint(query: str):
    results = await search_knowledge(query)
    return {"results": results}
```