#!/usr/bin/env python3
"""
Demo using real APIs with minimal dependencies.
This shows how the system works with actual OpenAI and Pinecone connections.
"""

import os
import json
import urllib.request
import urllib.parse
import ssl
from pathlib import Path
from datetime import datetime

# Load environment
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

print("=" * 60)
print("Ijon PDF RAG System - Live API Demo")
print("=" * 60)

# Configuration
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")

# 1. Load sample document
print("\n1. Loading sample document...")
sample_doc_path = Path("sample_pdfs/ml_textbook.txt")
if sample_doc_path.exists():
    with open(sample_doc_path, 'r') as f:
        content = f.read()
    
    # Extract first chapter
    lines = content.split('\n')
    chapter1 = []
    in_chapter1 = False
    
    for line in lines:
        if "Chapter 1:" in line:
            in_chapter1 = True
        elif "Chapter 2:" in line:
            break
        if in_chapter1:
            chapter1.append(line)
    
    chapter1_text = '\n'.join(chapter1)
    print(f"✓ Loaded Chapter 1 ({len(chapter1_text)} characters)")
    print(f"Preview: {chapter1_text[:150]}...")
else:
    print("✗ Sample document not found")
    exit(1)

# 2. Create chunks (simple splitting)
print("\n2. Creating text chunks...")
chunk_size = 500
chunks = []
words = chapter1_text.split()
current_chunk = []
current_size = 0

for word in words:
    current_chunk.append(word)
    current_size += len(word) + 1
    
    if current_size >= chunk_size:
        chunks.append(' '.join(current_chunk))
        current_chunk = []
        current_size = 0

if current_chunk:
    chunks.append(' '.join(current_chunk))

print(f"✓ Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks[:3]):
    print(f"  Chunk {i+1}: {chunk[:80]}...")

# 3. Generate embeddings using OpenAI
print("\n3. Generating embeddings with OpenAI...")
embeddings = []

for i, chunk in enumerate(chunks[:3]):  # Only first 3 for demo
    try:
        # Prepare request
        url = "https://api.openai.com/v1/embeddings"
        data = {
            "input": chunk,
            "model": "text-embedding-ada-002"
        }
        
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {OPENAI_KEY}")
        req.add_header("Content-Type", "application/json")
        
        context = ssl.create_default_context()
        response = urllib.request.urlopen(
            req, 
            data=json.dumps(data).encode('utf-8'),
            context=context
        )
        
        result = json.loads(response.read())
        embedding = result['data'][0]['embedding']
        embeddings.append({
            "id": f"chunk_{i}",
            "text": chunk,
            "embedding": embedding
        })
        
        print(f"  ✓ Generated embedding for chunk {i+1} (dimension: {len(embedding)})")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

# 4. Test query
print("\n4. Testing query system...")
query = "What is supervised learning?"
print(f"Query: '{query}'")

# Generate query embedding
try:
    url = "https://api.openai.com/v1/embeddings"
    data = {
        "input": query,
        "model": "text-embedding-ada-002"
    }
    
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {OPENAI_KEY}")
    req.add_header("Content-Type", "application/json")
    
    context = ssl.create_default_context()
    response = urllib.request.urlopen(
        req, 
        data=json.dumps(data).encode('utf-8'),
        context=context
    )
    
    result = json.loads(response.read())
    query_embedding = result['data'][0]['embedding']
    print(f"✓ Generated query embedding")
    
except Exception as e:
    print(f"✗ Error generating query embedding: {str(e)}")
    query_embedding = None

# 5. Find similar chunks (cosine similarity)
if query_embedding and embeddings:
    print("\n5. Finding similar chunks...")
    
    # Simple cosine similarity
    import math
    
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot_product / (norm_a * norm_b)
    
    similarities = []
    for emb in embeddings:
        sim = cosine_similarity(query_embedding, emb['embedding'])
        similarities.append((sim, emb['text']))
    
    similarities.sort(reverse=True)
    
    print("Top relevant chunks:")
    for i, (sim, text) in enumerate(similarities[:2]):
        print(f"\n  Chunk {i+1} (similarity: {sim:.3f}):")
        print(f"  {text[:150]}...")

# 6. Generate answer using OpenAI
print("\n6. Generating answer with OpenAI...")
if similarities:
    context = "\n\n".join([text for _, text in similarities[:2]])
    
    try:
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {OPENAI_KEY}")
        req.add_header("Content-Type", "application/json")
        
        context_ssl = ssl.create_default_context()
        response = urllib.request.urlopen(
            req, 
            data=json.dumps(data).encode('utf-8'),
            context=context_ssl
        )
        
        result = json.loads(response.read())
        answer = result['choices'][0]['message']['content']
        
        print(f"\n✓ Generated Answer:")
        print("-" * 40)
        print(answer)
        print("-" * 40)
        
    except Exception as e:
        print(f"✗ Error generating answer: {str(e)}")

# Save demo results
print("\n7. Saving results...")
results = {
    "timestamp": datetime.now().isoformat(),
    "query": query,
    "chunks_processed": len(chunks),
    "embeddings_generated": len(embeddings),
    "api_calls": {
        "openai_embeddings": len(embeddings) + 1,  # chunks + query
        "openai_completion": 1
    }
}

results_dir = Path("demo_results")
results_dir.mkdir(exist_ok=True)
result_file = results_dir / f"api_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {result_file}")

print("\n" + "=" * 60)
print("Demo complete! This shows:")
print("• Document chunking")
print("• OpenAI embeddings generation")
print("• Similarity search")
print("• Answer generation with context")
print("\nThe full system adds:")
print("• PDF extraction")
print("• Vector database storage")
print("• Knowledge graphs")
print("• Agent-based reasoning")
print("=" * 60)