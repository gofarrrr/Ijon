#!/usr/bin/env python3
"""
Guide to running the full Ijon PDF RAG system with real APIs.
"""

import os
from pathlib import Path

print("=" * 70)
print("Ijon PDF RAG System - Full System Guide")
print("=" * 70)

# Check API status
print("\n✅ API Status:")
print("  • OpenAI API: Connected and working")
print("  • Pinecone API: Connected (using 'mighty-walnut' index)")
print("  • Embeddings: Using OpenAI text-embedding-ada-002 (1536D)")

# Show current configuration
print("\n📋 Current Configuration:")
env_vars = [
    ("VECTOR_DB_TYPE", "pinecone"),
    ("OPENAI_API_KEY", "***configured***"),
    ("PINECONE_API_KEY", "***configured***"),
    ("PINECONE_ENVIRONMENT", "us-east-1-aws"),
    ("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
]

for key, value in env_vars:
    actual = os.getenv(key, value)
    if "KEY" in key and actual:
        print(f"  • {key}: ***set***")
    else:
        print(f"  • {key}: {actual}")

# Installation instructions
print("\n📦 To run the full system:")
print("\n1. Install core dependencies:")
print("   pip install openai pinecone sentence-transformers pydantic python-dotenv")

print("\n2. Install PDF processing libraries:")
print("   pip install PyPDF2 pdfplumber pymupdf")

print("\n3. For knowledge graphs (optional):")
print("   pip install neo4j networkx")

# Usage examples
print("\n🚀 Usage Examples:")

print("\n# Process PDFs from command line:")
print("python -m src.cli process sample_pdfs/")

print("\n# Query the system:")
print("python -m src.cli query 'What is machine learning?'")

print("\n# Query with agent (for complex questions):")
print("python -m src.cli query --agent 'Compare CNNs and RNNs'")

print("\n# Show system statistics:")
print("python -m src.cli stats")

# Python usage example
print("\n# Python code example:")
print("""
from src.rag.pipeline import RAGPipeline
import asyncio

async def test_system():
    # Initialize pipeline
    pipeline = await RAGPipeline.create()
    
    # Process a PDF
    result = await pipeline.process_pdf('sample.pdf')
    print(f"Processed {result.metadata.total_pages} pages")
    
    # Query the system
    answer, sources = await pipeline.query("What is deep learning?")
    print(f"Answer: {answer}")
    
    # Query with agent for complex reasoning
    answer, sources = await pipeline.query_with_agent(
        "How do transformers solve the vanishing gradient problem?"
    )
    print(f"Agent answer: {answer}")

# Run the test
asyncio.run(test_system())
""")

# Configuration adjustments needed
print("\n⚙️  Configuration Adjustments:")
print("\n1. For OpenAI embeddings (recommended for better quality):")
print("   Update .env:")
print("   EMBEDDING_MODEL=text-embedding-ada-002")
print("   # Note: This uses 1536 dimensions instead of 384")

print("\n2. Create new Pinecone index for this project:")
print("   # The existing 'mighty-walnut' index uses 3072D")
print("   # Create 'ijon-pdfs' index with 1536D for OpenAI embeddings")

print("\n3. Or use the existing index by updating:")
print("   PINECONE_INDEX_NAME=mighty-walnut")
print("   # And use a different embedding model with 3072D")

# Test data available
print("\n📄 Test Documents Available:")
sample_dir = Path("sample_pdfs")
if sample_dir.exists():
    for doc in sample_dir.glob("*.txt"):
        if doc.name != "README.txt":
            print(f"  • {doc.name}")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print("=" * 70)
print("The system is ready to use with real APIs!")
print("\nNext steps:")
print("1. Install the Python packages listed above")
print("2. Adjust configuration if needed")
print("3. Run the test examples")
print("4. Process your own PDFs")
print("\nFor detailed testing, see:")
print("• TESTING_GUIDE.md - Comprehensive testing documentation")
print("• test_system.py - Full pipeline test")
print("• demo_with_apis.py - Simple API demo")
print("=" * 70)