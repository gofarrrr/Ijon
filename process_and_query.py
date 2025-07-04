#!/usr/bin/env python3
"""
Simple script to process documents and run queries using the Ijon system.
"""

import asyncio
import os
from pathlib import Path
from typing import List
import json

# Load environment variables manually
def load_env():
    env_vars = {}
    with open('.env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                if value:  # Only set if value is not empty
                    env_vars[key] = value
    return env_vars

env = load_env()
OPENAI_API_KEY = env.get('OPENAI_API_KEY')
EMBEDDING_MODEL = env.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
CHUNK_SIZE = int(env.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(env.get('CHUNK_OVERLAP', '200'))

print("=" * 60)
print("Ijon PDF RAG System - Process and Query")
print("=" * 60)

# Simple document processor
class DocumentProcessor:
    def __init__(self):
        self.documents = {}
        
    def chunk_text(self, text: str) -> List[dict]:
        """Create text chunks."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= CHUNK_SIZE:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'content': chunk_text,
                    'word_count': len(current_chunk)
                })
                
                # Overlap
                overlap_words = int(CHUNK_OVERLAP / 10)
                current_chunk = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else []
                current_size = sum(len(w) + 1 for w in current_chunk)
        
        # Last chunk
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'word_count': len(current_chunk)
            })
        
        return chunks
    
    def process_file(self, file_path: Path) -> dict:
        """Process a single file."""
        print(f"\nProcessing: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.chunk_text(content)
        
        self.documents[file_path.stem] = {
            'path': str(file_path),
            'chunks': chunks,
            'embeddings': None
        }
        
        print(f"✓ Created {len(chunks)} chunks")
        return {'file': file_path.name, 'chunks': len(chunks)}


# Query system
async def query_system(processor: DocumentProcessor, query: str):
    """Query the processed documents."""
    import openai
    
    print(f"\nQuery: '{query}'")
    print("-" * 40)
    
    client = openai.Client(api_key=OPENAI_API_KEY)
    
    # Generate query embedding
    print("Generating query embedding...")
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding
    
    # Find similar chunks
    print("Searching for relevant chunks...")
    all_scores = []
    
    for doc_id, doc_data in processor.documents.items():
        # Generate embeddings if needed
        if doc_data['embeddings'] is None:
            print(f"Generating embeddings for {doc_id}...")
            embeddings = []
            
            for chunk in doc_data['chunks']:
                resp = client.embeddings.create(
                    input=chunk['content'],
                    model=EMBEDDING_MODEL
                )
                embeddings.append(resp.data[0].embedding)
            
            doc_data['embeddings'] = embeddings
        
        # Calculate similarities
        for i, (chunk, embedding) in enumerate(zip(doc_data['chunks'], doc_data['embeddings'])):
            # Cosine similarity
            import math
            dot_product = sum(x * y for x, y in zip(query_embedding, embedding))
            norm_a = math.sqrt(sum(x * x for x in query_embedding))
            norm_b = math.sqrt(sum(y * y for y in embedding))
            score = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
            
            all_scores.append((score, chunk['content'], doc_id))
    
    # Get top results
    all_scores.sort(reverse=True, key=lambda x: x[0])
    top_results = all_scores[:5]
    
    print(f"\nTop {len(top_results)} relevant chunks found:")
    for i, (score, _, doc_id) in enumerate(top_results):
        print(f"  {i+1}. Document: {doc_id}, Score: {score:.3f}")
    
    # Create context
    context = '\n\n'.join([chunk for _, chunk, _ in top_results])
    
    # Generate answer
    print("\nGenerating answer...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        max_tokens=300,
        temperature=0.7
    )
    
    answer = response.choices[0].message.content
    
    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    
    return answer, top_results


async def main():
    """Main function."""
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process sample documents
    print("\n1. Processing Documents")
    print("-" * 40)
    
    sample_dir = Path("sample_pdfs")
    text_files = list(sample_dir.glob("*.txt"))
    
    if not text_files:
        print("No text files found in sample_pdfs/")
        return
    
    for file_path in text_files:
        if file_path.name != "README.txt":
            processor.process_file(file_path)
    
    print(f"\n✓ Processed {len(processor.documents)} documents")
    
    # Run test queries
    print("\n2. Testing Queries")
    print("-" * 40)
    
    test_queries = [
        "What is supervised learning?",
        "What are the symptoms of diabetes?",
        "What makes a contract valid?",
    ]
    
    for query in test_queries:
        await query_system(processor, query)
        print("\n" + "="*60 + "\n")
    
    # Interactive mode
    print("\n3. Interactive Query Mode")
    print("-" * 40)
    print("Enter your questions (type 'quit' to exit):\n")
    
    while True:
        try:
            user_query = input("Query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            if user_query:
                await query_system(processor, user_query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found in .env file")
    else:
        asyncio.run(main())