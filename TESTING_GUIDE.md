# Testing Guide for Ijon

## Overview

This guide covers testing practices for the Ijon PDF RAG system, which uses Gemini embeddings and Neon PostgreSQL with pgvector.

## Test Structure

```
tests/
├── core/                    # Essential production tests
├── experimental/            # R&D and experimental features
├── test_config_simple.py    # Configuration tests
├── test_pdf_processing.py   # PDF processing tests
└── test_runner.py          # Test execution utilities
```

## Running Tests

### Quick Test Commands

```bash
# Activate virtual environment
source venv_ijon/bin/activate

# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_gemini_embeddings.py -v

# Run only core tests
python -m pytest tests/core/ -v
```

### Essential System Tests

1. **Database Connection**
   ```bash
   python test_neon_connection.py
   ```
   Verifies Neon PostgreSQL connection and pgvector operations.

2. **Gemini Embeddings**
   ```bash
   python test_gemini_simple.py
   ```
   Tests Gemini text-embedding-004 API integration.

3. **Full Pipeline**
   ```bash
   python test_system.py
   ```
   End-to-end test of PDF processing and querying.

## Test Categories

### Core Tests (Always Run)
- `test_neon_connection.py` - Database operations
- `test_gemini_embeddings.py` - Embedding generation
- `test_gemini_simple.py` - Gemini API integration
- `tests/test_config_simple.py` - Configuration management
- `tests/test_pdf_processing.py` - PDF extraction reliability

### Integration Tests
- `test_system.py` - Full system integration
- `test_real_pipeline.py` - Real PDF processing pipeline
- `test_v2_with_pdf.py` - V2 extraction system

### Demo Scripts
- `demo_simple.py` - Basic functionality demo
- `demo_with_apis.py` - API integration demo

## Writing New Tests

### Test Template
```python
import pytest
import asyncio
from your_module import YourClass

class TestYourFeature:
    """Test suite for your feature."""
    
    @pytest.mark.asyncio
    async def test_success_case(self):
        """Test normal operation."""
        result = await your_async_function()
        assert result.status == "success"
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(YourError):
            problematic_function()
    
    def test_edge_case(self):
        """Test boundary conditions."""
        result = your_function("")
        assert result is None
```

### Testing Guidelines

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Isolation**: Each test should be independent
3. **Coverage**: Aim for >80% code coverage
4. **Edge Cases**: Test boundaries and error conditions
5. **Async Tests**: Use `pytest.mark.asyncio` for async functions

## Environment Setup for Testing

### Required Environment Variables
```bash
# .env.test (create for testing)
GEMINI_API_KEY=your-test-api-key
DATABASE_URL=postgresql://test-db-connection
LOG_LEVEL=DEBUG
```

### Test Database Setup
```bash
# Use a separate test database
python setup_neon_database.py --test
python migrate_neon_for_gemini.py --test
```

## Common Test Scenarios

### 1. Testing Embeddings
```python
async def test_embedding_generation():
    embedder = GeminiEmbeddingGenerator()
    embeddings = await embedder.generate_embeddings(["test text"])
    assert len(embeddings[0]) == 768  # Gemini dimension
```

### 2. Testing Context Enhancement
```python
def test_context_enhancement():
    enhancer = SmartContextEnhancer()
    enhanced = enhancer.add_context("text", metadata, level=1)
    assert "[" in enhanced  # Context added
```

### 3. Testing Database Operations
```python
async def test_vector_search():
    conn = await asyncpg.connect(DATABASE_URL)
    results = await search_similar_chunks(conn, query_embedding)
    assert len(results) > 0
```

## Performance Testing

### Measure Processing Time
```python
import time

start = time.time()
await process_pdf("test.pdf")
duration = time.time() - start
assert duration < 60  # Should process in under 1 minute
```

### Memory Usage
```python
import psutil
import os

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss

# Run your operation
await heavy_operation()

final_memory = process.memory_info().rss
memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
assert memory_increase < 500  # Should use less than 500MB
```

## Debugging Failed Tests

### Enable Debug Logging
```bash
LOG_LEVEL=DEBUG python -m pytest tests/failing_test.py -v -s
```

### Common Issues

1. **API Key Missing**
   ```
   Error: GEMINI_API_KEY not found
   Solution: Set environment variable or create .env file
   ```

2. **Database Connection Failed**
   ```
   Error: Connection refused
   Solution: Check DATABASE_URL and network connectivity
   ```

3. **Import Errors**
   ```
   Error: No module named 'src'
   Solution: Run from project root or set PYTHONPATH
   ```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements_test.txt
      - name: Run tests
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          python -m pytest --cov=src
```

## Test Maintenance

### Regular Tasks
1. Update tests when APIs change
2. Remove obsolete tests
3. Add tests for new features
4. Review and improve test coverage
5. Update this guide as needed

### Test Review Checklist
- [ ] Tests pass locally
- [ ] New features have tests
- [ ] Edge cases covered
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Cleanup after tests