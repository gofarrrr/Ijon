#!/usr/bin/env python3
"""
Add enhanced prompt metadata support to existing Neon database schema.
This adds a single JSONB column to store enhanced extraction metadata.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()

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
    print("🚀 Adding Enhanced Prompt Metadata Support")
    print("=" * 70)
    
    # Get connection string from environment
    connection_string = os.getenv('NEON_CONNECTION_STRING')
    if not connection_string:
        print("❌ NEON_CONNECTION_STRING not found in environment variables")
        return
    
    try:
        # Connect to Neon
        print("\n📊 Connecting to Neon database...")
        conn = psycopg2.connect(connection_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        print("✅ Connected to Neon successfully!")
        
        # Check if content_chunks table exists
        print("\n🔍 Checking existing schema...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'content_chunks'
                );
            """)
            chunks_exists = cur.fetchone()[0]
            
            if not chunks_exists:
                print("❌ Content_chunks table not found. Please run setup_neon_database.py first.")
                return
            
            # Check if extraction_metadata column already exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'content_chunks'
                    AND column_name = 'extraction_metadata'
                );
            """)
            metadata_exists = cur.fetchone()[0]
            
            if metadata_exists:
                print("ℹ️  Extraction_metadata column already exists. Skipping creation.")
            else:
                # Add the extraction_metadata column
                print("\n📝 Adding extraction_metadata column...")
                execute_sql(conn, 
                    "ALTER TABLE content_chunks ADD COLUMN extraction_metadata JSONB",
                    "Added extraction_metadata column to content_chunks")
        
        # Create helper functions for enhanced metadata
        print("\n🔧 Creating enhanced metadata functions...")
        
        # Function to store enhanced extraction results
        store_enhanced_sql = """
        CREATE OR REPLACE FUNCTION store_enhanced_extraction(
            chunk_id UUID,
            thinking_process TEXT,
            extraction_quality FLOAT,
            confidence_details JSONB,
            processing_time_ms INTEGER DEFAULT NULL
        )
        RETURNS VOID AS $$
        BEGIN
            UPDATE content_chunks 
            SET extraction_metadata = jsonb_build_object(
                'enhanced_extraction', true,
                'thinking_process', thinking_process,
                'extraction_quality', extraction_quality,
                'confidence_details', confidence_details,
                'processing_time_ms', processing_time_ms,
                'enhanced_at', NOW()
            )
            WHERE id = chunk_id;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        execute_sql(conn, store_enhanced_sql, "Created store_enhanced_extraction function")
        
        # Function to get enhanced extraction stats
        stats_function_sql = """
        CREATE OR REPLACE FUNCTION get_enhanced_extraction_stats()
        RETURNS TABLE(
            total_chunks BIGINT,
            enhanced_chunks BIGINT,
            avg_quality_score FLOAT,
            avg_processing_time_ms FLOAT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(*) FILTER (WHERE extraction_metadata->>'enhanced_extraction' = 'true') as enhanced_chunks,
                AVG((extraction_metadata->>'extraction_quality')::FLOAT) FILTER (WHERE extraction_metadata->>'enhanced_extraction' = 'true') as avg_quality_score,
                AVG((extraction_metadata->>'processing_time_ms')::FLOAT) FILTER (WHERE extraction_metadata->>'enhanced_extraction' = 'true') as avg_processing_time_ms
            FROM content_chunks
            WHERE extraction_metadata IS NOT NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        execute_sql(conn, stats_function_sql, "Created get_enhanced_extraction_stats function")
        
        # Function to find chunks by quality score
        quality_search_sql = """
        CREATE OR REPLACE FUNCTION find_chunks_by_quality(min_quality FLOAT DEFAULT 0.8)
        RETURNS TABLE(
            chunk_id UUID,
            document_title TEXT,
            quality_score FLOAT,
            thinking_preview TEXT,
            content_preview TEXT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                cc.id,
                d.title,
                (cc.extraction_metadata->>'extraction_quality')::FLOAT,
                LEFT(cc.extraction_metadata->>'thinking_process', 100) || '...',
                LEFT(cc.content, 100) || '...'
            FROM content_chunks cc
            JOIN documents d ON cc.document_id = d.id
            WHERE cc.extraction_metadata->>'enhanced_extraction' = 'true'
            AND (cc.extraction_metadata->>'extraction_quality')::FLOAT >= min_quality
            ORDER BY (cc.extraction_metadata->>'extraction_quality')::FLOAT DESC;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        execute_sql(conn, quality_search_sql, "Created find_chunks_by_quality function")
        
        # Create an enhanced view of content chunks
        print("\n📊 Creating enhanced chunks view...")
        
        enhanced_view_sql = """
        CREATE OR REPLACE VIEW enhanced_chunks AS
        SELECT 
            cc.id,
            cc.document_id,
            d.title as document_title,
            cc.chunk_index,
            LEFT(cc.content, 200) || '...' as content_preview,
            -- Enhanced metadata fields
            CASE WHEN cc.extraction_metadata->>'enhanced_extraction' = 'true' THEN 'Enhanced' ELSE 'Standard' END as extraction_type,
            (cc.extraction_metadata->>'extraction_quality')::FLOAT as quality_score,
            (cc.extraction_metadata->>'processing_time_ms')::INTEGER as processing_time_ms,
            cc.extraction_metadata->>'enhanced_at' as enhanced_at,
            LEFT(cc.extraction_metadata->>'thinking_process', 100) as thinking_preview,
            -- Standard fields
            cc.rag_retrieval_count,
            cc.agent_retrieval_count,
            cc.created_at
        FROM content_chunks cc
        LEFT JOIN documents d ON cc.document_id = d.id
        ORDER BY 
            CASE WHEN cc.extraction_metadata->>'enhanced_extraction' = 'true' THEN 1 ELSE 2 END,
            (cc.extraction_metadata->>'extraction_quality')::FLOAT DESC NULLS LAST,
            cc.document_id, 
            cc.chunk_index;
        """
        
        execute_sql(conn, enhanced_view_sql, "Created enhanced_chunks view")
        
        # Test the new functionality
        print("\n🧪 Testing enhanced metadata support...")
        
        with conn.cursor() as cur:
            # Check current chunks
            cur.execute("SELECT COUNT(*) FROM content_chunks")
            chunk_count = cur.fetchone()[0]
            print(f"📊 Found {chunk_count} chunks in the system")
            
            if chunk_count > 0:
                # Test the stats function
                cur.execute("SELECT * FROM get_enhanced_extraction_stats()")
                stats = cur.fetchone()
                if stats:
                    total, enhanced, avg_quality, avg_time = stats
                    print(f"📈 Stats: {total} total chunks, {enhanced or 0} enhanced")
                    if avg_quality:
                        print(f"   Average quality: {avg_quality:.2f}")
                    if avg_time:
                        print(f"   Average processing time: {avg_time:.0f}ms")
                
                # Show sample from enhanced view
                cur.execute("SELECT COUNT(*) FROM enhanced_chunks")
                enhanced_count = cur.fetchone()[0]
                print(f"🔍 Enhanced chunks view contains {enhanced_count} entries")
        
        print("\n✅ Enhanced metadata support added successfully!")
        print("\nHow to use:")
        print("1. Store enhanced extraction:")
        print("   SELECT store_enhanced_extraction(chunk_id, 'thinking...', 0.95, '{\"confidence\": 0.9}', 1500);")
        print("2. View enhanced chunks: SELECT * FROM enhanced_chunks;")
        print("3. Get quality stats: SELECT * FROM get_enhanced_extraction_stats();")
        print("4. Find high-quality chunks: SELECT * FROM find_chunks_by_quality(0.9);")
        
        # Create a sample integration example
        print("\n📝 Creating integration example...")
        
        integration_example = """
# Integration Example for Enhanced Prompts

## In your extraction code:

```python
import json
import time

async def store_enhanced_result(chunk_id: str, extraction_result: dict, processing_time: float):
    \"\"\"Store enhanced extraction results in database.\"\"\"
    
    # Extract enhanced metadata
    thinking_process = extraction_result.get("thinking_process", "")
    extraction_quality = extraction_result.get("extraction_quality", 0.0)
    confidence_details = {
        "overall_confidence": extraction_result.get("overall_confidence", 0.0),
        "topics_confidence": extraction_result.get("topics_confidence", 0.0),
        "facts_confidence": extraction_result.get("facts_confidence", 0.0)
    }
    processing_time_ms = int(processing_time * 1000)
    
    # Store in database
    await db.execute_sql(
        "SELECT store_enhanced_extraction(%s, %s, %s, %s, %s)",
        (chunk_id, thinking_process, extraction_quality, 
         json.dumps(confidence_details), processing_time_ms)
    )
```

## Query examples:

```sql
-- Find your best extractions
SELECT * FROM find_chunks_by_quality(0.9);

-- Get processing statistics
SELECT * FROM get_enhanced_extraction_stats();

-- View all enhanced vs standard extractions
SELECT extraction_type, COUNT(*), AVG(quality_score) 
FROM enhanced_chunks 
GROUP BY extraction_type;
```
"""
        
        with open("enhanced_metadata_integration_example.md", "w") as f:
            f.write(integration_example)
        
        print("📄 Created enhanced_metadata_integration_example.md with usage examples")
        
    except Exception as e:
        print(f"❌ Error setting up enhanced metadata: {e}")
        sys.exit(1)
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()