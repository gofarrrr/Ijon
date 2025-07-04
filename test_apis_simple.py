#!/usr/bin/env python3
"""
Test API connections using only built-in libraries.
"""

import os
import json
import urllib.request
import urllib.error
import ssl
from pathlib import Path

print("=" * 60)
print("Ijon PDF RAG System - Simple API Test")
print("=" * 60)

# Load .env file
env_path = Path(".env")
if env_path.exists():
    print("Loading .env file...")
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# 1. Test OpenAI API
print("\n1. Testing OpenAI API...")
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    print(f"✓ OpenAI API key found (length: {len(openai_key)})")
    
    # Test API with curl-like request
    try:
        url = "https://api.openai.com/v1/models"
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {openai_key}")
        
        context = ssl.create_default_context()
        with urllib.request.urlopen(req, context=context) as response:
            data = json.loads(response.read())
            print(f"✓ OpenAI API is accessible! Found {len(data.get('data', []))} models")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("✗ OpenAI API key is invalid")
        else:
            print(f"✗ OpenAI API error: {e.code} - {e.reason}")
    except Exception as e:
        print(f"✗ OpenAI API error: {str(e)}")
else:
    print("✗ No OpenAI API key found")

# 2. Test Pinecone API
print("\n2. Testing Pinecone API...")
pinecone_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
if pinecone_key:
    print(f"✓ Pinecone API key found (length: {len(pinecone_key)})")
    print(f"✓ Pinecone environment: {pinecone_env}")
    
    # Test API
    try:
        url = "https://api.pinecone.io/indexes"
        req = urllib.request.Request(url)
        req.add_header("Api-Key", pinecone_key)
        req.add_header("Content-Type", "application/json")
        
        context = ssl.create_default_context()
        with urllib.request.urlopen(req, context=context) as response:
            data = json.loads(response.read())
            print(f"✓ Pinecone API is accessible! Found {len(data.get('indexes', []))} indexes")
            for idx in data.get('indexes', []):
                print(f"  - {idx.get('name')}: {idx.get('dimension')}D, {idx.get('metric')}")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("✗ Pinecone API key is invalid")
        else:
            print(f"✗ Pinecone API error: {e.code}")
    except Exception as e:
        print(f"✗ Pinecone API error: {str(e)}")
else:
    print("✗ No Pinecone API key found")

# 3. Check other configurations
print("\n3. Other Configurations...")
configs = [
    ("VECTOR_DB_TYPE", "Vector database type"),
    ("EMBEDDING_MODEL", "Embedding model"),
    ("CHUNK_SIZE", "Chunk size"),
    ("NEO4J_URI", "Neo4j URI"),
]

for key, desc in configs:
    value = os.getenv(key)
    if value:
        print(f"✓ {desc}: {value}")
    else:
        print(f"✗ {desc}: Not set")

# 4. Check directories and files
print("\n4. File System Check...")
dirs_to_check = ["sample_pdfs", "src", "tests", "logs"]
for dir_name in dirs_to_check:
    if Path(dir_name).exists():
        file_count = len(list(Path(dir_name).iterdir()))
        print(f"✓ {dir_name}/ exists ({file_count} items)")
    else:
        print(f"✗ {dir_name}/ missing")

# 5. Summary
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

can_run_basic = all([
    openai_key is not None,
    Path("sample_pdfs").exists(),
    Path("src").exists(),
])

if can_run_basic:
    print("✓ Basic system requirements met!")
    print("\nYou can now:")
    print("1. Install Python packages: pip install -r requirements_minimal.txt")
    print("2. Process documents and test queries")
    print("3. Use the OpenAI API for Q&A generation")
    
    if pinecone_key:
        print("4. Store and search vectors in Pinecone")
else:
    print("✗ Some requirements missing. Please check:")
    if not openai_key:
        print("  - Add OPENAI_API_KEY to .env")
    if not Path("src").exists():
        print("  - Source code directory missing")

print("\n" + "=" * 60)