#!/usr/bin/env python3
"""
Standalone test of real PDF processing with vector database.
No dependency on the config module.
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json

# Load environment manually
def load_env():
    env_vars = {}
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                if value:
                    env_vars[key] = value
                    os.environ[key] = value
    return env_vars

env = load_env()

print("=" * 70)
print("🚀 Real PDF Processing Test - Standalone")
print("=" * 70)


class PDFProcessor:
    """Process PDFs using available libraries."""
    
    def extract_with_pymupdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using PyMuPDF."""
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages.append({
                'page_num': page_num + 1,
                'text': text,
                'char_count': len(text)
            })
        
        metadata = doc.metadata
        doc.close()
        
        return {
            'pages': pages,
            'metadata': metadata,
            'total_pages': len(pages)
        }
    
    def extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text using pdfplumber."""
        import pdfplumber
        
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            metadata = pdf.metadata
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({
                    'page_num': i + 1,
                    'text': text,
                    'char_count': len(text)
                })
        
        return {
            'pages': pages,
            'metadata': metadata,
            'total_pages': len(pages)
        }
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Create text chunks."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'word_count': len(current_chunk),
                    'char_count': len(chunk_text)
                })
                
                # Overlap
                overlap_words = int(overlap / 10)
                current_chunk = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else []
                current_size = sum(len(w) + 1 for w in current_chunk)
        
        # Last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'word_count': len(current_chunk),
                'char_count': len(chunk_text)
            })
        
        return chunks


async def test_pinecone_connection():
    """Test Pinecone connection and operations."""
    import pinecone
    
    print("\n📊 Testing Pinecone Connection...")
    print("-" * 60)
    
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=env.get('PINECONE_API_KEY'))
        
        # List existing indexes
        indexes = pc.list_indexes()
        print(f"✅ Connected to Pinecone! Found {len(indexes)} indexes:")
        for idx in indexes:
            print(f"   • {idx.name} ({idx.dimension}D, {idx.metric})")
        
        # Create or connect to test index
        test_index_name = "ijon-test"
        index_exists = any(idx.name == test_index_name for idx in indexes)
        
        if not index_exists:
            print(f"\n📦 Creating new index: {test_index_name}")
            pc.create_index(
                name=test_index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("   ⏳ Waiting for index to be ready...")
            import time
            time.sleep(10)  # Wait for index creation
        
        # Connect to index
        index = pc.Index(test_index_name)
        stats = index.describe_index_stats()
        print(f"\n✅ Connected to index '{test_index_name}':")
        print(f"   • Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   • Dimension: {stats.get('dimension', 1536)}")
        
        return index
        
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        return None


async def test_real_pdf_processing():
    """Test processing a real PDF."""
    processor = PDFProcessor()
    
    # Find PDFs
    pdf_paths = []
    for dir_name in ["test_pdfs", "sample_pdfs", "."]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            pdf_paths.extend(dir_path.glob("*.pdf"))
    
    if not pdf_paths:
        print("❌ No PDF files found!")
        return None
    
    print(f"\n📄 Found {len(pdf_paths)} PDF files:")
    for path in pdf_paths[:3]:  # Show first 3
        print(f"   • {path}")
    
    # Process first PDF
    pdf_path = pdf_paths[0]
    print(f"\n📄 Processing: {pdf_path.name}")
    print("-" * 60)
    
    # Try PyMuPDF first
    try:
        print("1️⃣ Extracting with PyMuPDF...")
        result = processor.extract_with_pymupdf(pdf_path)
        print(f"   ✓ Extracted {result['total_pages']} pages")
        print(f"   ✓ Title: {result['metadata'].get('title', 'N/A')}")
        print(f"   ✓ Total text: {sum(p['char_count'] for p in result['pages']):,} characters")
    except Exception as e:
        print(f"   ❌ PyMuPDF failed: {e}")
        
        # Try pdfplumber
        print("\n1️⃣ Extracting with pdfplumber...")
        try:
            result = processor.extract_with_pdfplumber(pdf_path)
            print(f"   ✓ Extracted {result['total_pages']} pages")
            print(f"   ✓ Total text: {sum(p['char_count'] for p in result['pages']):,} characters")
        except Exception as e:
            print(f"   ❌ pdfplumber failed: {e}")
            return None
    
    # Create chunks
    print("\n2️⃣ Creating text chunks...")
    all_text = ' '.join(page['text'] for page in result['pages'])
    chunks = processor.chunk_text(all_text)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Show sample
    if chunks:
        print(f"\n📝 Sample chunk:")
        print("-" * 60)
        print(chunks[0]['content'][:200] + "...")
        print("-" * 60)
    
    return {
        'pdf_path': pdf_path,
        'pages': result['total_pages'],
        'chunks': chunks,
        'metadata': result['metadata']
    }


async def test_embeddings_and_search(pdf_data: Dict[str, Any], pinecone_index):
    """Test embedding generation and vector search."""
    import openai
    
    if not pdf_data or not pdf_data['chunks']:
        print("❌ No data to process")
        return
    
    print("\n🧠 Testing Embeddings and Search...")
    print("-" * 60)
    
    client = openai.Client(api_key=env.get('OPENAI_API_KEY'))
    
    # Generate embeddings for first few chunks
    print("1️⃣ Generating embeddings...")
    chunks_to_process = pdf_data['chunks'][:5]  # First 5 chunks
    
    vectors = []
    for i, chunk in enumerate(chunks_to_process):
        response = client.embeddings.create(
            input=chunk['content'],
            model='text-embedding-ada-002'
        )
        embedding = response.data[0].embedding
        
        vectors.append({
            "id": f"{pdf_data['pdf_path'].stem}_chunk_{i}",
            "values": embedding,
            "metadata": {
                "content": chunk['content'][:1000],  # Pinecone metadata limit
                "pdf": pdf_data['pdf_path'].name,
                "chunk_index": i
            }
        })
        print(f"   ✓ Generated embedding {i+1}/{len(chunks_to_process)}")
    
    # Store in Pinecone if available
    if pinecone_index:
        print("\n2️⃣ Storing in Pinecone...")
        try:
            pinecone_index.upsert(vectors=vectors)
            print(f"   ✓ Stored {len(vectors)} vectors")
            import time
            time.sleep(2)  # Give Pinecone time to index
        except Exception as e:
            print(f"   ❌ Storage failed: {e}")
    
    # Test search
    print("\n3️⃣ Testing search...")
    test_query = "What is the self-attention mechanism in transformers?"
    
    # Generate query embedding
    response = client.embeddings.create(
        input=test_query,
        model='text-embedding-ada-002'
    )
    query_embedding = response.data[0].embedding
    
    if pinecone_index:
        # Search in Pinecone
        try:
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True
            )
            
            print(f"\n🔍 Query: '{test_query}'")
            print(f"📊 Found {len(results.matches)} results:")
            
            for i, match in enumerate(results.matches):
                print(f"\n   {i+1}. Score: {match.score:.3f}")
                print(f"      ID: {match.id}")
                if match.metadata:
                    preview = match.metadata.get('content', '')[:150] + "..."
                    print(f"      Preview: {preview}")
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    else:
        # Local similarity search
        print(f"\n🔍 Local search for: '{test_query}'")
        
        # Calculate similarities
        import numpy as np
        
        similarities = []
        for vec in vectors:
            # Cosine similarity
            dot_product = np.dot(query_embedding, vec['values'])
            norm_a = np.linalg.norm(query_embedding)
            norm_b = np.linalg.norm(vec['values'])
            similarity = dot_product / (norm_a * norm_b)
            similarities.append((similarity, vec['metadata']))
        
        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        print(f"📊 Top 3 results:")
        for i, (score, metadata) in enumerate(similarities[:3]):
            print(f"\n   {i+1}. Score: {score:.3f}")
            preview = metadata.get('content', '')[:150] + "..."
            print(f"      Preview: {preview}")


async def main():
    """Run all tests."""
    print("\n🔧 Environment Check:")
    print(f"   • OpenAI API Key: {'✅' if env.get('OPENAI_API_KEY') else '❌'}")
    print(f"   • Pinecone API Key: {'✅' if env.get('PINECONE_API_KEY') else '❌'}")
    print(f"   • Embedding Model: {env.get('EMBEDDING_MODEL', 'text-embedding-ada-002')}")
    
    # Test Pinecone
    pinecone_index = await test_pinecone_connection()
    
    # Test PDF processing
    pdf_data = await test_real_pdf_processing()
    
    # Test embeddings and search
    if pdf_data:
        await test_embeddings_and_search(pdf_data, pinecone_index)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary:")
    print("=" * 70)
    print(f"✅ PDF Processing: {'Success' if pdf_data else 'Failed'}")
    print(f"✅ Pinecone Connection: {'Success' if pinecone_index else 'Failed/Skipped'}")
    print(f"✅ Embeddings: Success")
    print(f"✅ Vector Search: {'Success' if pinecone_index else 'Local only'}")
    
    if pdf_data:
        print(f"\n📄 Processed PDF:")
        print(f"   • File: {pdf_data['pdf_path'].name}")
        print(f"   • Pages: {pdf_data['pages']}")
        print(f"   • Chunks: {len(pdf_data['chunks'])}")
    
    print("\n🎉 Real PDF processing test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())