# Development Guide

## Setting Up Development Environment

### Prerequisites
- Python 3.11 or higher
- PostgreSQL client libraries
- Git

### 1. Clone Repository
```bash
git clone [repository-url]
cd Ijon
```

### 2. Create Virtual Environment
```bash
python -m venv venv_ijon
source venv_ijon/bin/activate  # On Windows: venv_ijon\Scripts\activate
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (testing, linting)
pip install -r requirements_test.txt
```

### 4. Environment Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
# Required:
# - GEMINI_API_KEY (get from Google AI Studio)
# - DATABASE_URL or NEON_CONNECTION_STRING (from Neon dashboard)
```

### 5. Database Setup
```bash
# Initialize Neon database
python setup_neon_database.py

# Run migrations for Gemini support
python migrate_neon_for_gemini.py
```

### 6. Verify Installation
```bash
# Test database connection
python test_neon_connection.py

# Test Gemini embeddings
python test_gemini_simple.py

# Test the full pipeline
python test_system.py
```

## Project Structure

```
Ijon/
├── src/                    # Core application code
│   ├── rag/               # RAG pipeline components
│   │   ├── embedder.py    # Main embedding interface
│   │   ├── gemini_embedder.py  # Gemini-specific implementation
│   │   └── pipeline.py    # Main RAG pipeline
│   ├── context/           # Context enhancement
│   │   └── smart_context_enhancer.py
│   ├── pdf_processor/     # PDF processing
│   │   ├── extractor.py   # Text extraction
│   │   ├── chunker.py     # Text chunking
│   │   └── preprocessor.py # Text cleaning
│   └── utils/             # Utilities
│       ├── errors.py      # Custom exceptions
│       └── logging.py     # Logging configuration
├── tests/                 # Test suite
│   ├── core/             # Essential tests
│   └── experimental/     # R&D tests
├── docs/                  # Documentation
└── extraction/           # Extraction experiments (R&D)
```

## Development Workflow

### 1. Making Changes
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Follow existing patterns in the codebase

# Test your changes
python -m pytest tests/
```

### 2. Code Style
```bash
# Format code with black
black src/ tests/

# Check with ruff
ruff check src/ tests/

# Type checking (optional)
mypy src/
```

### 3. Testing

#### Run Specific Tests
```bash
# Run core tests only
python -m pytest tests/core/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test file
python -m pytest tests/test_gemini_embeddings.py -v
```

#### Write New Tests
```python
# tests/test_your_feature.py
import pytest
from your_module import your_function

def test_your_function_success():
    """Test successful case."""
    result = your_function("input")
    assert result == "expected"

def test_your_function_error():
    """Test error handling."""
    with pytest.raises(YourError):
        your_function("bad_input")
```

### 4. Adding New Features

#### New Embedding Model
1. Create new embedder in `src/rag/`
2. Implement same interface as `EmbeddingGenerator`
3. Update config to support new model
4. Add tests

#### New Context Enhancement
1. Extend `SmartContextEnhancer`
2. Add new detection patterns
3. Test with sample content
4. Measure impact on retrieval quality

## Common Development Tasks

### Processing a New PDF Collection
```python
# Create a batch processor
import glob
from pathlib import Path

pdf_files = glob.glob("books/*.pdf")
for pdf_file in pdf_files:
    print(f"Processing {pdf_file}")
    # Run process_single_book.py logic
```

### Testing Embeddings Quality
```python
# Compare different context levels
texts = ["sample text 1", "sample text 2"]
for level in [0, 1, 2]:
    enhanced = enhancer.add_context(text, metadata, level)
    embedding = await embedder.generate_embeddings([enhanced])
    # Measure similarity, retrieval quality
```

### Database Migrations
```python
# Add new column
async def add_column(conn):
    await conn.execute("""
        ALTER TABLE content_chunks 
        ADD COLUMN new_field JSONB DEFAULT '{}'
    """)
```

## Debugging

### Enable Debug Logging
```python
# In your script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
LOG_LEVEL=DEBUG python your_script.py
```

### Common Issues

#### "No module named 'src'"
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Gemini API Errors
- Check API key is valid
- Verify quota/rate limits
- Use exponential backoff for retries

#### Database Connection Issues
- Verify connection string format
- Check network connectivity
- Ensure pgvector extension is enabled

## Performance Profiling

```python
from src.utils.logging import log_performance

@log_performance
async def slow_function():
    # Your code here
    pass

# Check logs for timing information
```

## Contributing Guidelines

1. **Code Quality**
   - Follow PEP 8 style guide
   - Add type hints to functions
   - Write docstrings (Google style)

2. **Testing**
   - Write tests for new features
   - Maintain >80% code coverage
   - Test edge cases

3. **Documentation**
   - Update README for user-facing changes
   - Update API_GUIDE for new APIs
   - Add inline comments for complex logic

4. **Pull Requests**
   - Clear description of changes
   - Reference any related issues
   - Ensure all tests pass
   - Request review from maintainers

## Resources

- [Gemini API Documentation](https://ai.google.dev/api/python/google/generativeai)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Neon Documentation](https://neon.tech/docs/introduction)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)

## Getting Help

- Check existing issues on GitHub
- Review test files for usage examples
- Consult API_GUIDE.md for common patterns
- Ask in project discussions