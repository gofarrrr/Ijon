#!/usr/bin/env python3
"""
Query PDFs synced from Google Drive.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Load environment
def load_env():
    env_vars = {}
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    if value:
                        env_vars[key] = value
                        os.environ[key] = value
    return env_vars

env = load_env()


async def query_pdfs(query: str, top_k: int = 5):
    """Query the Drive PDFs."""
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 60)
    
    # Initialize components
    from src.vector_db.pinecone_adapter import PineconeVectorDB
    import openai
    
    openai_client = openai.Client(api_key=env.get('OPENAI_API_KEY'))
    
    async def generate_embedding(texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = openai_client.embeddings.create(
                input=text,
                model=env.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    # Initialize vector DB
    vector_db = PineconeVectorDB(
        api_key=env.get('PINECONE_API_KEY'),
        environment=env.get('PINECONE_ENVIRONMENT'),
        index_name='ijon-drive-pdfs',
        dimension=1536,
        embedding_function=generate_embedding
    )
    
    try:
        await vector_db.initialize()
        
        # Search
        results = await vector_db.search_by_text(query, top_k=top_k)
        
        if not results:
            print("❌ No results found")
            return
        
        print(f"\n📊 Found {len(results)} relevant chunks:\n")
        
        # Display results
        for i, result in enumerate(results):
            doc = result.document
            score = result.score
            
            print(f"{i+1}. Score: {score:.3f}")
            print(f"   File: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   Pages: {doc.metadata.get('page_numbers', 'Unknown')}")
            print(f"   Preview: {doc.content[:150]}...")
            print()
        
        # Generate comprehensive answer
        print("\n🤖 Generating answer...")
        print("-" * 60)
        
        context = '\n\n'.join([
            f"[From {r.document.metadata.get('filename', 'Unknown')}]\n{r.document.content}"
            for r in results
        ])
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the question based on the provided context from PDF documents. Be accurate and cite which document the information comes from when relevant."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        print(answer)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Run sync_drive_pdfs.py to process PDFs")
        print("2. Set up Pinecone with the correct API key")


async def interactive_query():
    """Interactive query mode."""
    print("=" * 70)
    print("🔍 Google Drive PDF Query Interface")
    print("=" * 70)
    print("\nType your questions or 'quit' to exit")
    print("Examples:")
    print("  - What is machine learning?")
    print("  - Summarize the key points about transformers")
    print("  - What are the main findings in the research?")
    print()
    
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            await query_pdfs(query)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


async def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Query provided as command line argument
        query = ' '.join(sys.argv[1:])
        await query_pdfs(query)
    else:
        # Interactive mode
        await interactive_query()


if __name__ == "__main__":
    asyncio.run(main())