#!/usr/bin/env python3
"""
Monitor the progress of ksiazki PDF processing.
"""

import json
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def check_progress():
    """Check processing progress."""
    
    # Check progress file
    progress_file = Path("ksiazki_processing_progress.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        processed_count = len(progress["processed"])
        failed_count = len(progress["failed"])
        
        print(f"📋 PROGRESS FILE STATUS")
        print(f"Processed: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"Last session: {progress['session_stats'].get('last_session', 'Never')}")
    else:
        print("📋 No progress file found")
        processed_count = 0
        failed_count = 0
    
    # Check database
    try:
        conn = psycopg2.connect(os.getenv('NEON_CONNECTION_STRING'))
        cur = conn.cursor()
        
        print(f"\n📊 DATABASE STATUS")
        
        # Count documents
        cur.execute("SELECT COUNT(*) FROM documents WHERE source_type = 'ksiazki_pdf'")
        db_docs = cur.fetchone()[0]
        print(f"Documents in database: {db_docs}")
        
        # Count chunks
        cur.execute("""
            SELECT COUNT(*) FROM content_chunks cc
            JOIN documents d ON cc.document_id = d.id
            WHERE d.source_type = 'ksiazki_pdf'
        """)
        db_chunks = cur.fetchone()[0]
        print(f"Chunks in database: {db_chunks}")
        
        # Show recent documents
        print(f"\n📚 RECENT BOOKS PROCESSED:")
        cur.execute("""
            SELECT title, primary_category, processed_at
            FROM documents 
            WHERE source_type = 'ksiazki_pdf'
            ORDER BY processed_at DESC 
            LIMIT 5
        """)
        
        for title, category, processed_at in cur.fetchall():
            print(f"  • {title[:50]}{'...' if len(title) > 50 else ''} ({category}) - {processed_at}")
        
        # Books view summary
        print(f"\n📖 BOOKS VIEW SUMMARY:")
        cur.execute("""
            SELECT category, COUNT(*), SUM(chunks_extracted)
            FROM books 
            GROUP BY category 
            ORDER BY COUNT(*) DESC
        """)
        
        for category, count, total_chunks in cur.fetchall():
            print(f"  • {category}: {count} books, {total_chunks or 0} chunks")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
    
    # Check total files vs processed
    ksiazki_path = Path("/Users/marcin/Desktop/aplikacje/ksiazki pdf")
    if ksiazki_path.exists():
        total_pdfs = len(list(ksiazki_path.rglob("*.pdf")))
        remaining = total_pdfs - processed_count
        
        print(f"\n🎯 OVERALL PROGRESS")
        print(f"Total PDFs in ksiazki folder: {total_pdfs}")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"Remaining: {remaining}")
        
        if total_pdfs > 0:
            percentage = (processed_count / total_pdfs) * 100
            print(f"Progress: {percentage:.1f}%")

def show_failed_files():
    """Show failed files if any."""
    failed_file = Path("ksiazki_failed_pdfs.json")
    if failed_file.exists():
        with open(failed_file, 'r') as f:
            failed_files = json.load(f)
        
        if failed_files:
            print(f"\n❌ FAILED FILES ({len(failed_files)}):")
            for failure in failed_files[-5:]:  # Show last 5 failures
                filename = Path(failure["filepath"]).name
                print(f"  • {filename[:50]}{'...' if len(filename) > 50 else ''}")
                print(f"    Error: {failure['error'][:100]}{'...' if len(failure['error']) > 100 else ''}")
                print(f"    Retries: {failure['retry_count']}")
                print()

if __name__ == "__main__":
    print("=" * 70)
    print("📚 KSIAZKI PDF PROCESSING MONITOR")
    print("=" * 70)
    
    check_progress()
    show_failed_files()
    
    print(f"\n🔄 To continue processing:")
    print("python batch_ksiazki_processor.py --batch-size 2 --no-enhanced")
    print(f"\n🚀 To run with enhanced prompts:")
    print("python batch_ksiazki_processor.py --batch-size 1")
    print(f"\n📊 To check progress again:")
    print("python monitor_ksiazki_progress.py")