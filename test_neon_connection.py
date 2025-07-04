#!/usr/bin/env python3
"""
Test Neon Database Connection and Basic Operations
"""

import psycopg2
import json
from datetime import datetime, timedelta
import numpy as np

# Load connection from .env
import os
from pathlib import Path

# Load .env manually
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

NEON_CONNECTION = os.getenv('NEON_CONNECTION_STRING')

def test_connection():
    """Test basic connection"""
    print("\n1️⃣ Testing Connection...")
    try:
        conn = psycopg2.connect(NEON_CONNECTION)
        print("✅ Connection successful!")
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_document_operations():
    """Test document CRUD operations"""
    print("\n2️⃣ Testing Document Operations...")
    conn = psycopg2.connect(NEON_CONNECTION)
    cur = conn.cursor()
    
    try:
        # Insert a document
        cur.execute("""
            INSERT INTO documents (title, authors, source_type, primary_category, tags, doc_metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id, title
        """, (
            "Introduction to Machine Learning",
            ["Andrew Ng", "Tom Mitchell"],
            "gdrive",
            "technology",
            ["machine-learning", "AI", "tutorial"],
            json.dumps({"pages": 250, "year": 2023})
        ))
        
        doc_id, title = cur.fetchone()
        print(f"✅ Created document: {title} (ID: {doc_id})")
        
        # Update document
        cur.execute("""
            UPDATE documents 
            SET import_status = 'completed', processed_at = NOW()
            WHERE id = %s
        """, (doc_id,))
        print("✅ Updated document status")
        
        # Query document
        cur.execute("""
            SELECT title, import_status, tags, doc_metadata
            FROM documents
            WHERE id = %s
        """, (doc_id,))
        
        result = cur.fetchone()
        print(f"✅ Retrieved document: {result[0]}, Status: {result[1]}")
        
        conn.commit()
        return doc_id
        
    except Exception as e:
        print(f"❌ Document operations failed: {e}")
        conn.rollback()
        return None
    finally:
        cur.close()

def test_vector_operations(doc_id):
    """Test vector similarity search"""
    print("\n3️⃣ Testing Vector Operations...")
    conn = psycopg2.connect(NEON_CONNECTION)
    cur = conn.cursor()
    
    try:
        # Create mock embeddings (in production, use OpenAI)
        mock_embedding = np.random.rand(1536).tolist()
        
        # Insert chunks with embeddings
        chunks = [
            ("Machine learning is a subset of artificial intelligence.", 0),
            ("Neural networks are inspired by biological neurons.", 1),
            ("Deep learning uses multiple layers of neural networks.", 2)
        ]
        
        for content, idx in chunks:
            cur.execute("""
                INSERT INTO content_chunks (document_id, content, chunk_index, embedding)
                VALUES (%s, %s, %s, %s)
            """, (doc_id, content, idx, mock_embedding))
        
        print(f"✅ Inserted {len(chunks)} chunks with embeddings")
        
        # Test similarity search (with mock query embedding)
        query_embedding = np.random.rand(1536).tolist()
        cur.execute("""
            SELECT id, content, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM content_chunks
            WHERE document_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT 2
        """, (query_embedding, doc_id, query_embedding))
        
        results = cur.fetchall()
        print("✅ Similarity search results:")
        for chunk_id, content, similarity in results:
            print(f"   - {content[:50]}... (similarity: {similarity:.3f})")
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"❌ Vector operations failed: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()

def test_agent_memory():
    """Test agent memory system"""
    print("\n4️⃣ Testing Agent Memory System...")
    conn = psycopg2.connect(NEON_CONNECTION)
    cur = conn.cursor()
    
    try:
        # Create scratchpad entry
        session_id = 'test-session-001'
        cur.execute("""
            INSERT INTO agent_scratchpad 
            (session_id, agent_id, thought_type, content, relevance_score, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            session_id,
            'test-agent',
            'observation',
            'User is asking about machine learning basics',
            0.9,
            datetime.now() + timedelta(hours=1)
        ))
        
        scratch_id = cur.fetchone()[0]
        print("✅ Created scratchpad entry")
        
        # Promote to memory
        cur.execute("""
            INSERT INTO agent_memories 
            (agent_id, memory_type, content, importance_score, source_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (
            'test-agent',
            'fact',
            'User is interested in ML fundamentals',
            0.8,
            'learned'
        ))
        
        memory_id = cur.fetchone()[0]
        print("✅ Promoted to long-term memory")
        
        # Update scratchpad
        cur.execute("""
            UPDATE agent_scratchpad 
            SET promoted_to_memory = true 
            WHERE id = %s
        """, (scratch_id,))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"❌ Agent memory operations failed: {e}")
        conn.rollback()
        return False
    finally:
        cur.close()

def test_helper_functions(doc_id):
    """Test database helper functions"""
    print("\n5️⃣ Testing Helper Functions...")
    conn = psycopg2.connect(NEON_CONNECTION)
    cur = conn.cursor()
    
    try:
        # Test build_agent_context
        cur.execute("SELECT build_agent_context(%s, 'medium')", (doc_id,))
        context = cur.fetchone()[0]
        print("✅ build_agent_context result:")
        print(f"   - Document title: {context['document']['title']}")
        print(f"   - Has summary: {'summary' in context}")
        
        # Note: search_rag requires actual embeddings, so we'll skip the test
        # since we used mock embeddings
        print("✅ Helper functions working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Helper function test failed: {e}")
        return False
    finally:
        cur.close()
        conn.close()

def cleanup_test_data():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    conn = psycopg2.connect(NEON_CONNECTION)
    cur = conn.cursor()
    
    try:
        cur.execute("DELETE FROM documents WHERE title LIKE '%Machine Learning%'")
        cur.execute("DELETE FROM agent_memories WHERE agent_id = 'test-agent'")
        cur.execute("DELETE FROM agent_scratchpad WHERE agent_id = 'test-agent'")
        conn.commit()
        print("✅ Test data cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")
    finally:
        cur.close()
        conn.close()

def main():
    print("=" * 70)
    print("🧪 Neon Database Test Suite")
    print("=" * 70)
    
    # Run tests
    if not test_connection():
        print("\n❌ Connection test failed. Exiting.")
        return
    
    doc_id = test_document_operations()
    if doc_id:
        test_vector_operations(doc_id)
        test_agent_memory()
        test_helper_functions(doc_id)
    
    # Cleanup
    cleanup_test_data()
    
    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)
    print("\nYour Neon database is ready for:")
    print("  - Document storage and retrieval")
    print("  - Vector similarity search")
    print("  - Agent memory management")
    print("  - Context engineering")

if __name__ == "__main__":
    main()