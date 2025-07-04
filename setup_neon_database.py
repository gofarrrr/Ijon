#!/usr/bin/env python3
"""
Setup Neon Database for Ijon PDF RAG System
Cleans existing tables and creates new schema
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os

# Neon connection string
NEON_CONNECTION = 'postgresql://neondb_owner:npg_FSb5jh0EypNP@ep-round-leaf-a8mfukld-pooler.eastus2.azure.neon.tech/neondb?sslmode=require'

def execute_sql(conn, sql, description=""):
    """Execute SQL with error handling"""
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            if description:
                print(f"✅ {description}")
    except Exception as e:
        print(f"❌ Error in {description}: {e}")
        raise

def main():
    print("=" * 70)
    print("🚀 Ijon PDF RAG System - Neon Database Setup")
    print("=" * 70)
    
    try:
        # Connect to Neon
        print("\n📊 Connecting to Neon database...")
        conn = psycopg2.connect(NEON_CONNECTION)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("✅ Connected to Neon successfully!")
        
        # Check existing tables
        print("\n🔍 Checking existing tables...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            existing_tables = cur.fetchall()
            
            if existing_tables:
                print(f"Found {len(existing_tables)} existing tables:")
                for table in existing_tables:
                    print(f"  - {table[0]}")
                
                # Ask for confirmation to drop
                response = input("\n⚠️  Do you want to DROP all existing tables? (yes/no): ")
                if response.lower() == 'yes':
                    print("\n🗑️  Dropping existing tables...")
                    for table in existing_tables:
                        execute_sql(conn, f'DROP TABLE IF EXISTS "{table[0]}" CASCADE', f"Dropped {table[0]}")
                    print("✅ All tables dropped successfully!")
                else:
                    print("❌ Setup cancelled. No changes made.")
                    return
            else:
                print("✅ No existing tables found. Clean slate!")
        
        # Create extensions
        print("\n🔧 Creating required extensions...")
        execute_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector", "Vector extension created")
        execute_sql(conn, "CREATE EXTENSION IF NOT EXISTS pg_trgm", "Trigram extension created")
        
        # Create our schema
        print("\n📦 Creating new schema...")
        
        # Read and execute the SQL schema
        schema_sql = """
-- ============================================
-- SHARED CORE: Used by both pipelines
-- ============================================

-- Documents/Books Registry
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE,
    title TEXT NOT NULL,
    authors TEXT[],
    
    -- Source tracking
    source_type VARCHAR(50),
    source_path TEXT,
    
    -- Classification
    primary_category VARCHAR(100),
    tags TEXT[],
    language VARCHAR(10) DEFAULT 'en',
    
    -- Processing
    import_status VARCHAR(50) DEFAULT 'pending',
    processed_at TIMESTAMP,
    
    -- Metadata
    page_count INTEGER,
    file_size_bytes BIGINT,
    doc_metadata JSONB,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Content Chunks (Base for both pipelines)
CREATE TABLE content_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Content
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash VARCHAR(64),
    
    -- Location
    page_numbers INTEGER[],
    section_hierarchy TEXT[],
    
    -- Semantic info
    chunk_type VARCHAR(50),
    summary_sentence TEXT,
    
    -- Embeddings
    embedding vector(1536),
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-ada-002',
    
    -- Usage metrics
    rag_retrieval_count INTEGER DEFAULT 0,
    agent_retrieval_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);

-- ============================================
-- GENERAL RAG PIPELINE TABLES
-- ============================================

-- Pre-computed Q&A pairs for fast retrieval
CREATE TABLE qa_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    
    -- Question
    question TEXT NOT NULL,
    question_embedding vector(1536),
    question_variations TEXT[],
    
    -- Answer
    answer TEXT NOT NULL,
    answer_confidence FLOAT,
    answer_type VARCHAR(50),
    
    -- Sources
    source_chunk_ids UUID[],
    
    -- Quality metrics
    human_verified BOOLEAN DEFAULT FALSE,
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    
    -- Usage
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================
-- AGENT CONTEXT PIPELINE TABLES
-- ============================================

-- Distilled knowledge for agents
CREATE TABLE distilled_knowledge (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    
    -- Type and scope
    knowledge_type VARCHAR(50),
    scope VARCHAR(50),
    scope_reference TEXT,
    
    -- Content
    content TEXT,
    structured_data JSONB,
    
    -- Quality
    distillation_method VARCHAR(50),
    quality_score FLOAT,
    
    -- Relationships
    related_chunks UUID[],
    related_knowledge UUID[],
    
    -- Embeddings for similarity
    embedding vector(1536),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent Scratchpad (temporary working memory)
CREATE TABLE agent_scratchpad (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    agent_id VARCHAR(100),
    
    -- Scratch content
    thought_type VARCHAR(50),
    content TEXT NOT NULL,
    
    -- Context metadata
    relevance_score FLOAT,
    confidence_level FLOAT,
    
    -- Lifecycle
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    promoted_to_memory BOOLEAN DEFAULT FALSE
);

-- Long-term Agent Memories
CREATE TABLE agent_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100),
    
    -- Memory content
    memory_type VARCHAR(50),
    content TEXT NOT NULL,
    importance_score FLOAT,
    
    -- Source tracking
    source_type VARCHAR(50),
    source_session_id UUID,
    source_documents UUID[],
    
    -- Semantic indexing
    embedding vector(1536),
    tags TEXT[],
    
    -- Memory management
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    decay_rate FLOAT DEFAULT 0.0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Categories with hierarchy
CREATE TABLE categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    parent_id UUID REFERENCES categories(id),
    path TEXT,
    
    -- Usage stats
    document_count INTEGER DEFAULT 0,
    query_frequency INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_chunks_embedding ON content_chunks USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_qa_embedding ON qa_pairs USING ivfflat(question_embedding vector_cosine_ops);
CREATE INDEX idx_memories_embedding ON agent_memories USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX idx_distilled_embedding ON distilled_knowledge USING ivfflat(embedding vector_cosine_ops);

-- Text search indexes
CREATE INDEX idx_documents_title ON documents USING GIN(to_tsvector('english', title));
CREATE INDEX idx_chunks_content ON content_chunks USING GIN(to_tsvector('english', content));

-- Metadata indexes
CREATE INDEX idx_documents_tags ON documents USING GIN(tags);
CREATE INDEX idx_documents_status ON documents(import_status);
CREATE INDEX idx_chunks_document ON content_chunks(document_id);
"""
        
        execute_sql(conn, schema_sql, "Created all tables successfully")
        
        # Create helper functions
        print("\n🔧 Creating helper functions...")
        
        functions_sql = """
-- RAG: Fast similarity search
CREATE OR REPLACE FUNCTION search_rag(query_embedding vector, limit_count INT DEFAULT 10)
RETURNS TABLE(chunk_id UUID, content TEXT, similarity FLOAT)
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.content,
        1 - (c.embedding <=> query_embedding) as similarity
    FROM content_chunks c
    ORDER BY c.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Agent: Build progressive context
CREATE OR REPLACE FUNCTION build_agent_context(doc_id UUID, depth VARCHAR DEFAULT 'medium')
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'document', row_to_json(d),
        'summary', CASE WHEN depth IN ('medium', 'deep') 
            THEN (SELECT content FROM distilled_knowledge 
                  WHERE document_id = doc_id 
                  AND knowledge_type = 'summary' LIMIT 1) 
        END,
        'key_concepts', (
            SELECT array_agg(content) 
            FROM distilled_knowledge 
            WHERE document_id = doc_id 
            AND knowledge_type = 'concepts' 
            LIMIT 10
        )
    ) INTO result
    FROM documents d
    WHERE d.id = doc_id;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;
"""
        
        execute_sql(conn, functions_sql, "Created helper functions")
        
        # Test the setup
        print("\n🧪 Testing the setup...")
        
        # Test 1: Insert a test document
        test_sql = """
INSERT INTO documents (title, authors, source_type, primary_category, tags)
VALUES ('Test Document', ARRAY['Test Author'], 'upload', 'test', ARRAY['test', 'setup'])
RETURNING id;
"""
        
        with conn.cursor() as cur:
            cur.execute(test_sql)
            doc_id = cur.fetchone()[0]
            print(f"✅ Test document created with ID: {doc_id}")
        
        # Test 2: Check tables
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            tables = cur.fetchall()
            
            print("\n📊 Created tables:")
            for schema, table, size in tables:
                print(f"  - {table} ({size})")
        
        # Test 3: Check indexes
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    indexname,
                    tablename
                FROM pg_indexes 
                WHERE schemaname = 'public'
                AND indexname NOT LIKE '%_pkey'
                ORDER BY tablename, indexname;
            """)
            indexes = cur.fetchall()
            
            print(f"\n📑 Created {len(indexes)} indexes")
        
        # Clean up test data
        execute_sql(conn, "DELETE FROM documents WHERE title = 'Test Document'", "Cleaned test data")
        
        print("\n" + "=" * 70)
        print("🎉 Database setup complete!")
        print("=" * 70)
        print("\nYour Neon database is ready with:")
        print(f"  - {len(tables)} tables")
        print(f"  - {len(indexes)} indexes")
        print("  - 2 helper functions")
        print("\nConnection string saved for future use.")
        
        # Save connection string to .env
        env_path = "/Users/marcin/Desktop/aplikacje/Ijon/.env"
        with open(env_path, 'a') as f:
            f.write("\n# Neon Database Configuration\n")
            f.write("NEON_CONNECTION_STRING=postgresql://neondb_owner:npg_FSb5jh0EypNP@ep-round-leaf-a8mfukld-pooler.eastus2.azure.neon.tech/neondb?sslmode=require\n")
        print(f"\n✅ Connection string added to {env_path}")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()