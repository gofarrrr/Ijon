#!/usr/bin/env python3
"""
Add RAG-ready full text chunks to the existing Mental Models document in Neon.

This script takes the already processed Mental Models document and adds
the full text content as chunked, RAG-ready content chunks.
"""

import asyncio
import os
import sys
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from extraction.pdf_processor import PDFProcessor
from extraction.v2.neon_storage import NeonStorage
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)

# Find the Mental Models PDF dynamically
PDF_PATTERN = "/Users/marcin/Desktop/aplikacje/The Great Mental Models*.pdf"
pdf_files = glob.glob(PDF_PATTERN)
PDF_PATH = pdf_files[0] if pdf_files else ""

# The existing document ID we want to add RAG chunks to
EXISTING_DOC_ID = "b00d0bf5-124f-49e2-973f-c88eccc49ea9"


async def add_rag_chunks():
    """Add RAG chunks to existing document."""
    print("🔧 Adding RAG-ready full text chunks to existing Mental Models document...")
    
    # Verify PDF exists
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found. Searched for: {PDF_PATTERN}")
        print(f"Current path: {PDF_PATH}")
        return
    
    print(f"📄 Using PDF: {os.path.basename(PDF_PATH)}")
    
    try:
        # 1. Process PDF to get full text chunks
        print("📄 Processing PDF for full text chunks...")
        pdf_processor = PDFProcessor()
        full_text_chunks = await pdf_processor.process_pdf(PDF_PATH)
        
        print(f"📊 Found {len(full_text_chunks)} text chunks")
        
        # 2. Convert to RAG format
        rag_chunks = []
        total_words = 0
        for chunk in full_text_chunks:
            if chunk.content.strip():  # Only include non-empty chunks
                word_count = len(chunk.content.split())
                total_words += word_count
                rag_chunks.append({
                    'content': chunk.content,
                    'page_numbers': getattr(chunk, 'page_numbers', []),
                    'chunk_metadata': {
                        'chunk_id': getattr(chunk, 'chunk_id', ''),
                        'token_count': word_count
                    }
                })
        
        avg_words = total_words // len(rag_chunks) if rag_chunks else 0
        print(f"📊 Prepared {len(rag_chunks)} RAG chunks (avg {avg_words} words/chunk)")
        
        # 3. Store in Neon database
        print("💾 Storing RAG chunks in Neon database...")
        neon_storage = NeonStorage()
        
        # Store chunks directly using database connection
        import psycopg2
        import uuid
        from datetime import datetime
        
        with psycopg2.connect(neon_storage.connection_string) as conn:
            with conn.cursor() as cur:
                # Check if document exists
                cur.execute("SELECT title FROM documents WHERE id = %s", (EXISTING_DOC_ID,))
                doc = cur.fetchone()
                if not doc:
                    print(f"❌ Document {EXISTING_DOC_ID} not found")
                    return
                
                print(f"📋 Adding RAG chunks to document: {doc[0]}")
                
                # Store RAG chunks
                chunk_count = 0
                for i, chunk_data in enumerate(rag_chunks):
                    try:
                        cur.execute("""
                            INSERT INTO content_chunks (
                                id, document_id, chunk_index, content, 
                                chunk_type, page_numbers, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            str(uuid.uuid4()),
                            EXISTING_DOC_ID,
                            i + 3000,  # High offset to avoid conflicts
                            chunk_data['content'],
                            "full_text_rag_chunk",
                            chunk_data.get('page_numbers', []),
                            datetime.utcnow()
                        ))
                        chunk_count += 1
                        
                        # Commit every 50 chunks to avoid memory issues
                        if chunk_count % 50 == 0:
                            conn.commit()
                            print(f"   ✅ Stored {chunk_count}/{len(rag_chunks)} chunks...")
                    
                    except Exception as e:
                        print(f"   ⚠️  Error storing chunk {i}: {e}")
                        continue
                
                conn.commit()
                print(f"✅ Successfully stored {chunk_count} RAG chunks!")
        
        # 4. Verify storage
        print("\n🔍 Verifying RAG chunk storage...")
        with psycopg2.connect(neon_storage.connection_string) as conn:
            with conn.cursor() as cur:
                # Get chunk counts by type
                cur.execute("""
                    SELECT chunk_type, COUNT(*) 
                    FROM content_chunks 
                    WHERE document_id = %s 
                    GROUP BY chunk_type 
                    ORDER BY chunk_type
                """, (EXISTING_DOC_ID,))
                
                chunk_counts = cur.fetchall()
                print("📊 Content chunks by type:")
                for chunk_type, count in chunk_counts:
                    print(f"   - {chunk_type}: {count}")
                
                # Test a sample query
                cur.execute("""
                    SELECT content 
                    FROM content_chunks 
                    WHERE document_id = %s AND chunk_type = 'full_text_rag_chunk' 
                    AND content ILIKE %s
                    LIMIT 3
                """, (EXISTING_DOC_ID, "%mental model%"))
                
                results = cur.fetchall()
                if results:
                    print(f"\n🔎 Sample query for 'mental model' found {len(results)} matches:")
                    for i, (content,) in enumerate(results, 1):
                        print(f"   {i}. {content[:150]}...")
                else:
                    print("\n🔎 Sample query for 'mental model' found no matches")
        
        print("\n🎉 RAG chunk integration complete!")
        print("The Mental Models document now includes full text content for RAG retrieval.")
        
    except Exception as e:
        logger.error(f"Error adding RAG chunks: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(add_rag_chunks())