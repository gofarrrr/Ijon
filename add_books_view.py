#!/usr/bin/env python3
"""
Add user-friendly books view to existing Neon database schema.
This creates a simple, human-readable interface for browsing processed documents.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os
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
    print("📚 Adding User-Friendly Books View to Ijon Database")
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
        
        # Check if documents table exists
        print("\n🔍 Checking existing schema...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'documents'
                );
            """)
            documents_exists = cur.fetchone()[0]
            
            if not documents_exists:
                print("❌ Documents table not found. Please run setup_neon_database.py first.")
                return
            
            # Check if books view already exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.views 
                    WHERE table_schema = 'public' 
                    AND table_name = 'books'
                );
            """)
            books_view_exists = cur.fetchone()[0]
            
            if books_view_exists:
                print("ℹ️  Books view already exists. Dropping and recreating...")
                execute_sql(conn, "DROP VIEW IF EXISTS books", "Dropped existing books view")
        
        # Create the user-friendly books view
        print("\n📚 Creating books view...")
        
        books_view_sql = """
        CREATE VIEW books AS
        SELECT 
            id,
            title,
            authors,
            primary_category as category,
            tags,
            page_count,
            file_size_bytes,
            language,
            source_type,
            processed_at,
            created_at,
            -- Extraction status information
            (SELECT COUNT(*) FROM content_chunks WHERE document_id = documents.id) as chunks_extracted,
            (SELECT COUNT(*) FROM qa_pairs WHERE document_id = documents.id) as qa_pairs_generated,
            (SELECT COUNT(*) FROM distilled_knowledge WHERE document_id = documents.id) as knowledge_items,
            -- Human-readable status
            CASE 
                WHEN import_status = 'completed' THEN 'Ready for Use'
                WHEN import_status = 'processing' THEN 'Processing...'
                WHEN import_status = 'failed' THEN 'Processing Failed'
                WHEN import_status = 'pending' THEN 'Waiting to Process'
                ELSE 'Unknown Status'
            END as status,
            -- Processing summary
            CASE 
                WHEN import_status = 'completed' THEN 
                    CONCAT(
                        COALESCE((SELECT COUNT(*) FROM content_chunks WHERE document_id = documents.id), 0),
                        ' text chunks, ',
                        COALESCE((SELECT COUNT(*) FROM qa_pairs WHERE document_id = documents.id), 0),
                        ' Q&A pairs'
                    )
                ELSE 'Not processed yet'
            END as extraction_summary,
            -- Easy identification
            CASE 
                WHEN page_count IS NOT NULL AND page_count > 0 THEN CONCAT(page_count, ' pages')
                WHEN file_size_bytes IS NOT NULL THEN 
                    CASE 
                        WHEN file_size_bytes > 1048576 THEN CONCAT(ROUND(file_size_bytes/1048576.0, 1), ' MB')
                        WHEN file_size_bytes > 1024 THEN CONCAT(ROUND(file_size_bytes/1024.0, 1), ' KB')
                        ELSE CONCAT(file_size_bytes, ' bytes')
                    END
                ELSE 'Size unknown'
            END as size_info
        FROM documents
        WHERE import_status IS NOT NULL
        ORDER BY 
            CASE import_status 
                WHEN 'completed' THEN 1
                WHEN 'processing' THEN 2
                WHEN 'pending' THEN 3
                ELSE 4
            END,
            title;
        """
        
        execute_sql(conn, books_view_sql, "Created books view")
        
        # Create helper function for easy book searching
        print("\n🔍 Creating book search function...")
        
        search_function_sql = """
        CREATE OR REPLACE FUNCTION search_books(search_term TEXT)
        RETURNS TABLE(
            book_id UUID,
            title TEXT,
            authors TEXT[],
            category TEXT,
            status TEXT,
            chunks_extracted BIGINT,
            relevance_score FLOAT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                b.id,
                b.title,
                b.authors,
                b.category,
                b.status,
                b.chunks_extracted,
                -- Simple relevance scoring based on title and author matches
                (
                    CASE WHEN LOWER(b.title) LIKE LOWER('%' || search_term || '%') THEN 2.0 ELSE 0.0 END +
                    CASE WHEN array_to_string(b.authors, ' ') ILIKE '%' || search_term || '%' THEN 1.5 ELSE 0.0 END +
                    CASE WHEN b.category ILIKE '%' || search_term || '%' THEN 1.0 ELSE 0.0 END +
                    CASE WHEN array_to_string(b.tags, ' ') ILIKE '%' || search_term || '%' THEN 0.5 ELSE 0.0 END
                ) as relevance_score
            FROM books b
            WHERE 
                LOWER(b.title) LIKE LOWER('%' || search_term || '%') OR
                array_to_string(b.authors, ' ') ILIKE '%' || search_term || '%' OR
                b.category ILIKE '%' || search_term || '%' OR
                array_to_string(b.tags, ' ') ILIKE '%' || search_term || '%'
            ORDER BY relevance_score DESC, b.title;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        execute_sql(conn, search_function_sql, "Created search_books function")
        
        # Create a simple book details function
        print("\n📖 Creating book details function...")
        
        book_details_sql = """
        CREATE OR REPLACE FUNCTION get_book_details(book_id UUID)
        RETURNS TABLE(
            title TEXT,
            authors TEXT[],
            category TEXT,
            tags TEXT[],
            status TEXT,
            size_info TEXT,
            extraction_summary TEXT,
            processed_at TIMESTAMP,
            top_chunks TEXT[]
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                b.title,
                b.authors,
                b.category,
                b.tags,
                b.status,
                b.size_info,
                b.extraction_summary,
                b.processed_at,
                -- Get top 3 content chunks as preview
                ARRAY(
                    SELECT SUBSTRING(c.content, 1, 200) || '...'
                    FROM content_chunks c 
                    WHERE c.document_id = book_id 
                    ORDER BY c.chunk_index 
                    LIMIT 3
                ) as top_chunks
            FROM books b
            WHERE b.id = book_id;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        execute_sql(conn, book_details_sql, "Created get_book_details function")
        
        # Test the new view
        print("\n🧪 Testing the books view...")
        
        with conn.cursor() as cur:
            # Check if we have any books
            cur.execute("SELECT COUNT(*) FROM books")
            book_count = cur.fetchone()[0]
            print(f"📚 Found {book_count} books in the system")
            
            if book_count > 0:
                # Show a sample of books
                cur.execute("""
                    SELECT title, status, chunks_extracted, extraction_summary 
                    FROM books 
                    LIMIT 5
                """)
                books = cur.fetchall()
                
                print("\n📋 Sample books:")
                for book in books:
                    title, status, chunks, summary = book
                    print(f"  • {title[:50]}{'...' if len(title) > 50 else ''}")
                    print(f"    Status: {status}")
                    print(f"    Content: {summary}")
                    print()
            
            # Test search function
            if book_count > 0:
                print("🔍 Testing search function...")
                cur.execute("SELECT search_books('test')")
                search_results = cur.fetchall()
                print(f"   Found {len(search_results)} results for 'test'")
        
        print("\n✅ Books view setup completed successfully!")
        print("\nHow to use:")
        print("1. View all books: SELECT * FROM books;")
        print("2. Search books: SELECT * FROM search_books('keyword');")
        print("3. Book details: SELECT * FROM get_book_details('book-id');")
        print("4. Ready books only: SELECT * FROM books WHERE status = 'Ready for Use';")
        
    except Exception as e:
        print(f"❌ Error setting up books view: {e}")
        sys.exit(1)
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()