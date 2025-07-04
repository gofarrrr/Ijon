#!/usr/bin/env python3
"""
Test the real Ijon pipeline with PDF processing and vector database.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Load environment manually to avoid config issues
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
print("🚀 Ijon PDF RAG System - Real Pipeline Test")
print("=" * 70)

# Import after env is loaded
from src.pdf_processor.extractor import PDFExtractor
from src.text_processing.preprocessor import TextPreprocessor
from src.text_processing.semantic_chunker import SemanticChunker
from src.vector_db.pinecone_adapter import PineconeVectorDB
from src.models import Document, PDFChunk, PDFMetadata, ProcessingStatus
import openai
import pinecone as pc


class RealPDFProcessor:
    """Process real PDFs with the full pipeline."""
    
    def __init__(self):
        self.extractor = PDFExtractor(enable_ocr=False)
        self.preprocessor = TextPreprocessor()
        self.chunker = SemanticChunker(
            chunk_size=int(env.get('CHUNK_SIZE', '1000')),
            chunk_overlap=int(env.get('CHUNK_OVERLAP', '200'))
        )
        self.vector_db = None
        self.openai_client = openai.Client(api_key=env.get('OPENAI_API_KEY'))
        
    async def initialize_vector_db(self):
        """Initialize Pinecone vector database."""
        print("\n📊 Initializing Vector Database...")
        
        # Use existing index or create new one
        self.vector_db = PineconeVectorDB(
            api_key=env.get('PINECONE_API_KEY'),
            environment=env.get('PINECONE_ENVIRONMENT'),
            index_name='ijon-test',  # Use a test index
            dimension=1536,  # OpenAI embedding dimension
            embedding_function=self.generate_embedding
        )
        
        try:
            await self.vector_db.initialize()
            stats = await self.vector_db.count_documents()
            print(f"✅ Vector DB initialized with {stats} existing documents")
        except Exception as e:
            print(f"⚠️  Vector DB initialization warning: {e}")
            print("   Will use in-memory storage instead")
            self.vector_db = None
    
    async def generate_embedding(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        embeddings = []
        for text in texts:
            response = self.openai_client.embeddings.create(
                input=text,
                model=env.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    async def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a PDF file through the full pipeline."""
        print(f"\n📄 Processing PDF: {pdf_path.name}")
        print("-" * 60)
        
        # 1. Extract metadata
        print("1️⃣ Extracting metadata...")
        metadata = await self.extractor.extract_metadata(pdf_path)
        print(f"   ✓ Title: {metadata.title or 'N/A'}")
        print(f"   ✓ Pages: {metadata.total_pages}")
        print(f"   ✓ Size: {metadata.file_size_bytes:,} bytes")
        
        # 2. Extract pages
        print("\n2️⃣ Extracting text from pages...")
        pages = await self.extractor.extract_pages(pdf_path)
        total_text_length = sum(len(page.text) for page in pages)
        print(f"   ✓ Extracted {len(pages)} pages")
        print(f"   ✓ Total text: {total_text_length:,} characters")
        
        # 3. Preprocess text
        print("\n3️⃣ Preprocessing text...")
        for page in pages:
            page.text = self.preprocessor.clean_text(page.text)
        
        # 4. Create chunks
        print("\n4️⃣ Creating semantic chunks...")
        all_chunks = []
        for page_num, page in enumerate(pages, 1):
            if page.text.strip():
                chunks = self.chunker.chunk_text(
                    page.text,
                    metadata={
                        'page_number': page_num,
                        'pdf_id': pdf_path.stem,
                        'filename': pdf_path.name
                    }
                )
                all_chunks.extend(chunks)
        
        print(f"   ✓ Created {len(all_chunks)} chunks")
        
        # 5. Generate embeddings and store
        if self.vector_db:
            print("\n5️⃣ Generating embeddings and storing in vector DB...")
            documents = []
            
            for i, chunk in enumerate(all_chunks):
                # Create document
                doc = Document(
                    id=f"{pdf_path.stem}_chunk_{i}",
                    content=chunk.content,
                    metadata={
                        'pdf_id': pdf_path.stem,
                        'filename': pdf_path.name,
                        'page_numbers': chunk.page_numbers,
                        'chunk_index': i,
                        'word_count': chunk.word_count,
                    }
                )
                documents.append(doc)
                
                # Show progress
                if (i + 1) % 5 == 0:
                    print(f"   ... processed {i + 1}/{len(all_chunks)} chunks")
            
            # Generate embeddings
            print("   🔄 Generating embeddings...")
            texts = [doc.content for doc in documents]
            embeddings = await self.generate_embedding(texts)
            
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            # Store in vector DB
            print("   💾 Storing in vector database...")
            await self.vector_db.upsert_documents(documents)
            print(f"   ✓ Stored {len(documents)} chunks with embeddings")
        else:
            print("\n5️⃣ Skipping vector storage (in-memory only)")
        
        return {
            'metadata': metadata,
            'pages': len(pages),
            'chunks': len(all_chunks),
            'sample_chunk': all_chunks[0].content[:200] + '...' if all_chunks else None
        }
    
    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the processed documents."""
        print(f"\n❓ Query: '{question}'")
        print("-" * 60)
        
        if self.vector_db:
            # Use vector database search
            print("🔍 Searching vector database...")
            results = await self.vector_db.search_by_text(question, top_k=top_k)
            
            print(f"\n📊 Found {len(results)} relevant chunks:")
            for i, result in enumerate(results):
                print(f"   {i+1}. Score: {result.score:.3f}")
                print(f"      Document: {result.document.metadata.get('filename', 'Unknown')}")
                print(f"      Preview: {result.document.content[:100]}...")
        else:
            print("⚠️  No vector database available")
            results = []
        
        # Generate answer
        if results:
            print("\n🤖 Generating answer...")
            context = '\n\n'.join([r.document.content for r in results])
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            return {
                'answer': answer,
                'sources': results,
                'chunks_used': len(results)
            }
        else:
            return {
                'answer': "No relevant information found.",
                'sources': [],
                'chunks_used': 0
            }


async def main():
    """Test the real pipeline."""
    processor = RealPDFProcessor()
    
    # Initialize vector DB
    await processor.initialize_vector_db()
    
    # Process PDFs
    print("\n📚 Processing PDFs...")
    print("=" * 70)
    
    # Look for PDFs
    pdf_paths = []
    
    # Check test_pdfs directory
    test_pdfs_dir = Path("test_pdfs")
    if test_pdfs_dir.exists():
        pdf_paths.extend(test_pdfs_dir.glob("*.pdf"))
    
    # Check sample_pdfs directory
    sample_pdfs_dir = Path("sample_pdfs")
    if sample_pdfs_dir.exists():
        pdf_paths.extend(sample_pdfs_dir.glob("*.pdf"))
    
    if not pdf_paths:
        print("❌ No PDF files found!")
        print("   Please run: python create_real_pdf.py")
        return
    
    print(f"Found {len(pdf_paths)} PDF files to process:")
    for path in pdf_paths:
        print(f"  • {path}")
    
    # Process each PDF
    processing_results = []
    for pdf_path in pdf_paths:
        try:
            result = await processor.process_pdf(pdf_path)
            processing_results.append(result)
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
    
    # Test queries
    print("\n🧪 Testing Queries...")
    print("=" * 70)
    
    test_queries = [
        "What is the self-attention mechanism?",
        "How do transformers differ from RNNs?",
        "What are the applications of large language models?",
        "What is multi-head attention?",
    ]
    
    query_results = []
    for query in test_queries:
        result = await processor.query(query)
        query_results.append({
            'query': query,
            'answer_length': len(result['answer']),
            'chunks_used': result['chunks_used'],
            'found_answer': result['chunks_used'] > 0
        })
        
        print(f"\n✅ Answer:")
        print("-" * 60)
        print(result['answer'])
        print("-" * 60)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 70)
    
    print("\n✅ Processing Results:")
    for i, result in enumerate(processing_results):
        print(f"   PDF {i+1}: {result['pages']} pages, {result['chunks']} chunks")
    
    print("\n✅ Query Results:")
    for result in query_results:
        status = "✓" if result['found_answer'] else "✗"
        print(f"   {status} {result['query'][:50]}...")
        print(f"      Used {result['chunks_used']} chunks, {result['answer_length']} chars")
    
    # Cleanup vector DB if needed
    if processor.vector_db:
        print("\n🧹 Cleaning up test data...")
        # Optional: clean up test data
        # await processor.vector_db.clear()
    
    print("\n" + "=" * 70)
    print("🎉 Real Pipeline Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())