# Ijon - Intelligent PDF RAG System with Gemini Embeddings

An advanced Retrieval-Augmented Generation (RAG) system for extracting and querying knowledge from PDF documents, featuring Google's Gemini embeddings and smart contextual enhancement.

## 🎯 Overview

Ijon processes PDF documents (particularly books on mental models, psychology, and decision-making) into a searchable knowledge base using:

- **Gemini text-embedding-004** for semantic embeddings (768 dimensions)
- **Neon PostgreSQL with pgvector** for vector storage and similarity search
- **Smart Context Enhancement** for improved retrieval quality
- **Mental Model Detection** for domain-specific understanding

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone [repository-url]
cd Ijon

# 2. Set up environment
python -m venv venv_ijon
source venv_ijon/bin/activate  # On Windows: venv_ijon\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env with your API keys:
# - GEMINI_API_KEY
# - NEON_CONNECTION_STRING or DATABASE_URL

# 4. Initialize the database
python setup_neon_database.py
python migrate_neon_for_gemini.py

# 5. Process a PDF
python process_single_book.py path/to/your.pdf

# 6. Query the system
python query_mental_models.py
# Or direct query:
python query_mental_models.py "What is flow state?"
```

## 📚 System Architecture

### Core Pipeline
```
PDF → Text Extraction → Smart Chunking → Context Enhancement → Gemini Embeddings → Neon pgvector
                                                ↓
Query → Context Enhancement → Gemini Embedding → Similarity Search → Ranked Results
```

### Key Components

1. **PDF Processing** (`src/pdf_processor/`)
   - Extracts text from PDFs using PyPDF2/PyMuPDF
   - Handles both text-based and scanned documents
   - Implements semantic chunking with overlap

2. **Smart Context Enhancement** (`src/context/smart_context_enhancer.py`)
   - Detects mental models and concepts
   - Adds contextual metadata before embedding
   - Three enhancement levels: basic, smart, analytical

3. **Gemini Embeddings** (`src/rag/gemini_embedder.py`)
   - Uses Google's text-embedding-004 model
   - 768-dimensional embeddings
   - ~$0.025 per million tokens
   - Optimized for retrieval tasks

4. **Vector Storage** (Neon PostgreSQL)
   - pgvector extension for similarity search
   - Indexed for fast retrieval
   - Supports filtering by metadata

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your-gemini-api-key
DATABASE_URL=postgresql://user:pass@host/db  # or NEON_CONNECTION_STRING

# Optional
LOG_LEVEL=INFO
CACHE_DIR=.cache
ENABLE_CACHE=true
```

### Processing Options
```python
# Limit pages for large PDFs
python process_single_book.py book.pdf --max-pages 100

# Process with custom context level
python process_single_book.py book.pdf --context-level 2
```

## 📊 Current Data

The system currently contains:
- **Flow: The Psychology of Optimal Experience** - Complete (317 pages)
- Additional mental models books can be added via `process_single_book.py`

## 🔍 Querying

### Interactive Mode
```bash
python query_mental_models.py
```

### Command Line Query
```bash
python query_mental_models.py "How do I achieve flow state?"
```

### Demo Queries
```bash
python query_mental_models.py demo
```

### Python API
```python
from src.rag.gemini_embedder import GeminiEmbeddingGenerator
from src.context.smart_context_enhancer import SmartContextEnhancer
import asyncio

async def query_knowledge(question):
    embedder = GeminiEmbeddingGenerator()
    enhancer = SmartContextEnhancer()
    
    # Enhance query for better retrieval
    enhanced_query = enhancer.add_context(question, metadata, level=1)
    
    # Generate embedding and search
    embedding = await embedder.generate_embeddings([enhanced_query])
    # ... perform vector search
```

## 🧪 Testing

```bash
# Test database connection
python test_neon_connection.py

# Test Gemini embeddings
python test_gemini_embeddings.py

# Test full pipeline
python test_system.py

# Run all tests
python -m pytest tests/
```

## 📁 Project Structure

```
Ijon/
├── src/
│   ├── rag/
│   │   ├── gemini_embedder.py      # Gemini embedding generation
│   │   └── pipeline.py             # Main RAG pipeline
│   ├── context/
│   │   └── smart_context_enhancer.py # Context enhancement
│   ├── pdf_processor/              # PDF processing modules
│   └── utils/                      # Utilities and helpers
├── extraction/
│   └── v2/                         # Experimental 12-factor extraction
├── query_mental_models.py          # Main query interface
├── process_single_book.py          # PDF processing script
├── setup_neon_database.py          # Database setup
└── requirements.txt                # Dependencies
```

## 🚧 Experimental Features

The following features are in development:
- **Cognitive RAG Pipeline** - Agent-based query routing for complex tasks
- **12-Factor Extraction** - Quality-driven extraction with human validation
- **Knowledge Graph Integration** - Entity and relationship extraction

## 📈 Performance

- **Embedding Generation**: ~1-2 seconds per page
- **Query Response**: 50-500ms for semantic search
- **Storage**: ~9.5KB per embedding (when transferred as text)
- **Quality**: 85%+ relevance for domain-specific queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow existing code patterns (see CLAUDE.md)
4. Add tests for new functionality
5. Submit a Pull Request

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- Google Gemini for embeddings API
- Neon for serverless PostgreSQL
- pgvector for vector similarity search