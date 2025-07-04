#!/usr/bin/env python3
"""
Test real API connections for the Ijon system.
This script tests actual API connections without full dependencies.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Simple color printing
def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_warning(msg):
    print(f"⚠️  {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

print("=" * 60)
print("Ijon PDF RAG System - Real Connection Test")
print("=" * 60)

# Load environment variables
env_path = Path(".env")
if env_path.exists():
    print_info("Loading environment variables from .env")
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
else:
    print_error("No .env file found!")
    sys.exit(1)

# Test results
results = {
    "timestamp": datetime.now().isoformat(),
    "tests": {}
}

# 1. Test OpenAI Connection
print("\n1. Testing OpenAI API Connection...")
try:
    import openai
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment")
        results["tests"]["openai"] = {"status": "failed", "error": "No API key"}
    else:
        # Simple API test
        client = openai.Client(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API connected' in 3 words"}],
            max_tokens=10
        )
        
        print_success(f"OpenAI API connected! Response: {response.choices[0].message.content}")
        results["tests"]["openai"] = {
            "status": "success",
            "model": "gpt-3.5-turbo",
            "response": response.choices[0].message.content
        }
        
except ImportError:
    print_error("OpenAI package not installed. Run: pip install openai")
    results["tests"]["openai"] = {"status": "failed", "error": "Package not installed"}
except Exception as e:
    print_error(f"OpenAI connection failed: {str(e)}")
    results["tests"]["openai"] = {"status": "failed", "error": str(e)}

# 2. Test Pinecone Connection
print("\n2. Testing Pinecone API Connection...")
try:
    # Try new package name first
    try:
        import pinecone
    except ImportError:
        # Try old package name
        import pinecone_client as pinecone
    
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    index_name = os.getenv("PINECONE_INDEX_NAME", "ijon-pdfs")
    
    if not api_key:
        print_error("PINECONE_API_KEY not found in environment")
        results["tests"]["pinecone"] = {"status": "failed", "error": "No API key"}
    else:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=api_key)
        
        # List indexes
        indexes = pc.list_indexes()
        print_info(f"Found {len(indexes)} Pinecone indexes")
        
        # Check if our index exists
        index_exists = any(idx.name == index_name for idx in indexes)
        
        if index_exists:
            print_success(f"Pinecone index '{index_name}' exists!")
            # Get index stats
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print_info(f"Index stats: {stats.get('total_vector_count', 0)} vectors")
            
            results["tests"]["pinecone"] = {
                "status": "success",
                "index_name": index_name,
                "vector_count": stats.get('total_vector_count', 0)
            }
        else:
            print_warning(f"Index '{index_name}' not found. Will be created on first use.")
            results["tests"]["pinecone"] = {
                "status": "success",
                "index_name": index_name,
                "note": "Index will be created on first use"
            }
            
except ImportError:
    print_error("Pinecone package not installed. Run: pip install pinecone-client")
    results["tests"]["pinecone"] = {"status": "failed", "error": "Package not installed"}
except Exception as e:
    print_error(f"Pinecone connection failed: {str(e)}")
    results["tests"]["pinecone"] = {"status": "failed", "error": str(e)}

# 3. Test Sentence Transformers
print("\n3. Testing Sentence Transformers...")
try:
    from sentence_transformers import SentenceTransformer
    
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print_info(f"Loading model: {model_name}")
    
    # This will download the model if not cached
    model = SentenceTransformer(model_name)
    
    # Test embedding
    test_text = "This is a test sentence."
    embedding = model.encode(test_text)
    
    print_success(f"Sentence transformer loaded! Embedding dimension: {len(embedding)}")
    results["tests"]["sentence_transformers"] = {
        "status": "success",
        "model": model_name,
        "embedding_dim": len(embedding)
    }
    
except ImportError:
    print_error("Sentence-transformers not installed. Run: pip install sentence-transformers")
    results["tests"]["sentence_transformers"] = {"status": "failed", "error": "Package not installed"}
except Exception as e:
    print_error(f"Sentence transformers failed: {str(e)}")
    results["tests"]["sentence_transformers"] = {"status": "failed", "error": str(e)}

# 4. Test PDF Processing Libraries
print("\n4. Testing PDF Processing Libraries...")
pdf_results = {}

# Test PyPDF2
try:
    import PyPDF2
    print_success("PyPDF2 is installed")
    pdf_results["PyPDF2"] = "installed"
except ImportError:
    print_warning("PyPDF2 not installed")
    pdf_results["PyPDF2"] = "not installed"

# Test pdfplumber
try:
    import pdfplumber
    print_success("pdfplumber is installed")
    pdf_results["pdfplumber"] = "installed"
except ImportError:
    print_warning("pdfplumber not installed")
    pdf_results["pdfplumber"] = "not installed"

# Test PyMuPDF
try:
    import fitz  # PyMuPDF
    print_success("PyMuPDF is installed")
    pdf_results["PyMuPDF"] = "installed"
except ImportError:
    print_warning("PyMuPDF not installed")
    pdf_results["PyMuPDF"] = "not installed"

results["tests"]["pdf_libraries"] = pdf_results

# 5. Test File System
print("\n5. Testing File System...")
required_dirs = ["sample_pdfs", "logs", "exports"]
fs_results = {}

for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print_success(f"Directory '{dir_name}' exists")
        fs_results[dir_name] = "exists"
    else:
        dir_path.mkdir(exist_ok=True)
        print_info(f"Created directory '{dir_name}'")
        fs_results[dir_name] = "created"

results["tests"]["file_system"] = fs_results

# Save results
results_dir = Path("test_results")
results_dir.mkdir(exist_ok=True)
result_file = results_dir / f"connection_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📄 Test results saved to: {result_file}")

# Summary
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

working_components = []
missing_components = []

for test_name, test_result in results["tests"].items():
    if isinstance(test_result, dict) and test_result.get("status") == "success":
        working_components.append(test_name)
    elif isinstance(test_result, dict) and test_result.get("status") == "failed":
        missing_components.append(test_name)

if working_components:
    print("\n✅ Working Components:")
    for comp in working_components:
        print(f"  • {comp}")

if missing_components:
    print("\n❌ Missing/Failed Components:")
    for comp in missing_components:
        print(f"  • {comp}")

# Quick test with available components
if "openai" in working_components and "sentence_transformers" in working_components:
    print("\n🚀 Core components are working! You can:")
    print("  1. Process text documents")
    print("  2. Generate embeddings")
    print("  3. Use OpenAI for Q&A")
    
    if "pinecone" in working_components:
        print("  4. Store and search vectors in Pinecone")

print("\n" + "=" * 60)