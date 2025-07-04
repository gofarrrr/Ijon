#!/usr/bin/env python3
"""
Batch PDF Processing System for ksiazki pdf folder only.
Processes all PDFs in the ksiazki pdf directory with progress tracking and error handling.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import hashlib
import psycopg2
from psycopg2.extras import Json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extraction.pdf_processor import PDFProcessor
from extraction.v2.extractors import StatelessExtractor
from src.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

class KsiazkiBatchProcessor:
    """
    Batch PDF processing system specifically for ksiazki pdf folder.
    """
    
    def __init__(self):
        """Initialize batch processor for ksiazki pdf folder."""
        self.ksiazki_path = Path("/Users/marcin/Desktop/aplikacje/ksiazki pdf")
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
        
        # Progress tracking
        self.progress_file = Path("ksiazki_processing_progress.json")
        self.failed_files_log = Path("ksiazki_failed_pdfs.json")
        
        # Processing configuration
        self.batch_size = 3  # Process 3 files at a time
        self.max_retries = 2
        self.retry_delay = 5  # seconds
        self.use_enhanced_prompts = True  # Default to enhanced
        self.model = "gemini-2.5-pro"
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        
        # State tracking
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = 0
        self.skipped_files = 0
        
    def generate_file_id(self, filepath: Path) -> str:
        """Generate a unique ID for a file based on path and size."""
        try:
            stat = filepath.stat()
            content = f"{filepath.absolute()}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(filepath.absolute()).encode()).hexdigest()
    
    def load_progress(self) -> Dict:
        """Load processing progress from file."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load progress file: {e}")
        
        return {
            "processed": {},  # file_id -> status
            "failed": {},     # file_id -> error_info
            "session_stats": {
                "total_sessions": 0,
                "last_session": None,
                "total_processed": 0,
                "total_failed": 0
            }
        }
    
    def save_progress(self, progress: Dict):
        """Save processing progress to file."""
        try:
            progress["session_stats"]["last_session"] = datetime.now().isoformat()
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def log_failed_file(self, filepath: Path, error: str, retry_count: int):
        """Log failed file for analysis."""
        failed_info = {
            "filepath": str(filepath),
            "file_id": self.generate_file_id(filepath),
            "error": error,
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat(),
            "file_size": filepath.stat().st_size if filepath.exists() else 0
        }
        
        # Load existing failed files
        failed_files = []
        if self.failed_files_log.exists():
            try:
                with open(self.failed_files_log, 'r') as f:
                    failed_files = json.load(f)
            except Exception:
                pass
        
        # Add new failure
        failed_files.append(failed_info)
        
        # Save updated list
        try:
            with open(self.failed_files_log, 'w') as f:
                json.dump(failed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log failed file: {e}")
    
    def discover_ksiazki_pdfs(self) -> List[Path]:
        """Discover all PDF files in the ksiazki pdf folder."""
        logger.info(f"Discovering PDF files in {self.ksiazki_path}")
        
        if not self.ksiazki_path.exists():
            logger.error(f"ksiazki pdf folder not found: {self.ksiazki_path}")
            return []
        
        pdf_files = []
        for pdf_file in self.ksiazki_path.rglob("*.pdf"):
            if pdf_file.is_file():
                pdf_files.append(pdf_file)
        
        # Sort by size (smallest first for easier processing)
        pdf_files.sort(key=lambda f: f.stat().st_size if f.exists() else float('inf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files in ksiazki folder")
        return pdf_files
    
    def check_file_in_database(self, filepath: Path) -> Optional[str]:
        """Check if file is already processed in database."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # Check by source path first (most reliable)
                    cur.execute("""
                        SELECT id, import_status FROM documents 
                        WHERE source_path = %s
                    """, (str(filepath),))
                    
                    result = cur.fetchone()
                    if result:
                        doc_id, status = result
                        if status == 'completed':
                            return doc_id
                    
                    # Also check by title as fallback
                    title = self.extract_title_from_filename(filepath)
                    cur.execute("""
                        SELECT id, import_status FROM documents 
                        WHERE title = %s AND import_status = 'completed'
                    """, (title,))
                    
                    result = cur.fetchone()
                    if result:
                        return result[0]
                        
        except Exception as e:
            logger.warning(f"Database check failed for {filepath.name}: {e}")
        
        return None
    
    def extract_title_from_filename(self, filepath: Path) -> str:
        """Extract clean title from filename."""
        filename = filepath.stem
        
        # Common format: "Title -- Author -- Year -- Publisher"
        if " -- " in filename:
            parts = filename.split(" -- ")
            title = parts[0]
        else:
            title = filename
        
        # Clean up title
        title = title.replace("_", " ").strip()
        
        return title
    
    def extract_category_from_path(self, filepath: Path) -> str:
        """Extract category from folder structure."""
        try:
            # Get relative path from ksiazki pdf folder
            relative_path = filepath.relative_to(self.ksiazki_path)
            
            if len(relative_path.parts) > 1:
                # Use the immediate parent folder as category
                category = relative_path.parts[0]
                
                # Clean up category name
                category = category.replace("_", " ").replace("-", " ").strip()
                category = " ".join(word.capitalize() for word in category.split())
                
                return category
        except Exception:
            pass
        
        return "General"
    
    async def process_single_ksiazki_pdf(self, filepath: Path, progress: Dict, retry_count: int = 0) -> Tuple[bool, str]:
        """
        Process a single PDF file from ksiazki folder.
        
        Returns:
            (success, message)
        """
        start_time = time.time()
        file_id = self.generate_file_id(filepath)
        
        try:
            logger.info(f"Processing {filepath.name} (attempt {retry_count + 1})")
            
            # Check if already in database
            existing_id = self.check_file_in_database(filepath)
            if existing_id:
                logger.info(f"File {filepath.name} already processed (ID: {existing_id})")
                progress["processed"][file_id] = {
                    "status": "already_exists",
                    "document_id": existing_id,
                    "timestamp": datetime.now().isoformat()
                }
                return True, "Already exists in database"
            
            # Extract basic metadata
            file_stat = filepath.stat()
            file_size = file_stat.st_size
            
            # Process PDF
            pdf_chunks = await self.pdf_processor.process_pdf(str(filepath))
            
            if not pdf_chunks:
                raise ValueError("No content extracted from PDF")
            
            logger.info(f"Extracted {len(pdf_chunks)} chunks from {filepath.name}")
            
            # Store document in database
            document_id = await self.store_ksiazki_document(filepath, pdf_chunks, file_size)
            
            # Process chunks with enhanced extraction (sample only for speed)
            if self.use_enhanced_prompts and len(pdf_chunks) > 0:
                await self.process_chunks_enhanced(document_id, pdf_chunks[:2])  # Only first 2 chunks
            
            processing_time = time.time() - start_time
            
            # Update progress
            progress["processed"][file_id] = {
                "status": "completed",
                "document_id": document_id,
                "chunks_count": len(pdf_chunks),
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(filepath)
            }
            
            logger.info(f"✅ Successfully processed {filepath.name} in {processing_time:.1f}s")
            return True, f"Processed successfully ({len(pdf_chunks)} chunks)"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error processing {filepath.name}: {error_msg}")
            
            # Log detailed error for debugging
            if retry_count == 0:  # Only log detailed error on first attempt
                logger.error(f"Detailed error: {traceback.format_exc()}")
            
            # Check if we should retry
            if retry_count < self.max_retries:
                logger.info(f"Retrying {filepath.name} in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                return await self.process_single_ksiazki_pdf(filepath, progress, retry_count + 1)
            else:
                # Final failure - log it
                self.log_failed_file(filepath, error_msg, retry_count)
                progress["failed"][file_id] = {
                    "error": error_msg,
                    "retry_count": retry_count,
                    "timestamp": datetime.now().isoformat(),
                    "file_path": str(filepath)
                }
                return False, f"Failed after {retry_count + 1} attempts: {error_msg}"
    
    async def store_ksiazki_document(self, filepath: Path, pdf_chunks: List, file_size: int) -> str:
        """Store ksiazki document and chunks in database."""
        
        # Extract metadata
        title = self.extract_title_from_filename(filepath)
        category = self.extract_category_from_path(filepath)
        
        # Try to parse author from filename
        authors = []
        filename = filepath.stem
        if " -- " in filename:
            parts = filename.split(" -- ")
            if len(parts) >= 2:
                authors = [parts[1]]
        
        # Store in database
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Insert document
                cur.execute("""
                    INSERT INTO documents (
                        title, authors, source_type, source_path, primary_category,
                        page_count, file_size_bytes, import_status, processed_at,
                        doc_metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    title, authors, 'ksiazki_pdf', str(filepath), category,
                    len(pdf_chunks), file_size, 'processing', datetime.now(),
                    Json({"original_filename": filepath.name, "category_folder": category})
                ))
                
                document_id = cur.fetchone()[0]
                
                # Insert chunks
                for i, chunk in enumerate(pdf_chunks):
                    cur.execute("""
                        INSERT INTO content_chunks (
                            document_id, content, chunk_index, page_numbers,
                            chunk_type, summary_sentence
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        document_id, chunk.content, i, [chunk.page_num],
                        'text', chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
                    ))
                
                # Update document status
                cur.execute("""
                    UPDATE documents 
                    SET import_status = 'completed', processed_at = %s
                    WHERE id = %s
                """, (datetime.now(), document_id))
                
                conn.commit()
                
        return document_id
    
    async def process_chunks_enhanced(self, document_id: str, pdf_chunks: List):
        """Process chunks with enhanced extraction (limited for performance)."""
        logger.info(f"Running enhanced extraction for document {document_id} (sample chunks)")
        
        for i, chunk in enumerate(pdf_chunks):
            try:
                # Enhanced extraction prompt
                enhanced_prompt = f"""You are a specialized Knowledge Extraction Agent operating in an iterative extraction loop.

## Agent Loop Architecture
You operate in systematic phases:
1. **Analyze Content**: Understand the text structure and key themes
2. **Plan Extraction**: Determine extraction strategy based on content type
3. **Execute Extraction**: Apply extraction techniques systematically
4. **Validate Results**: Ensure quality and completeness
5. **Refine Output**: Improve based on confidence assessment

Extract structured knowledge from this text chunk:

{chunk.content}

Respond with a JSON object containing:
- thinking_process: Your analysis thoughts
- topics: List of main topics/concepts with definitions
- facts: List of factual claims with evidence and confidence
- relationships: How concepts relate with relationship types
- questions: Questions across cognitive levels
- summary: Academic prose summary
- overall_confidence: Your confidence (0.0-1.0)
- extraction_quality: Self-assessment of extraction quality
"""
                
                start_time = time.time()
                
                # Call enhanced extraction
                extracted_data = await StatelessExtractor.call_llm(
                    client=None,
                    prompt=enhanced_prompt,
                    model=self.model,
                    temperature=0.3
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Store enhanced metadata
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE content_chunks 
                            SET extraction_metadata = %s
                            WHERE document_id = %s AND chunk_index = %s
                        """, (
                            Json({
                                "enhanced_extraction": True,
                                "thinking_process": extracted_data.get("thinking_process", ""),
                                "extraction_quality": extracted_data.get("extraction_quality", 0.0),
                                "confidence_details": {
                                    "overall_confidence": extracted_data.get("overall_confidence", 0.0)
                                },
                                "processing_time_ms": processing_time,
                                "enhanced_at": datetime.now().isoformat()
                            }),
                            document_id, i
                        ))
                        conn.commit()
                
                logger.info(f"Enhanced extraction completed for chunk {i} in {processing_time}ms")
                
            except Exception as e:
                logger.warning(f"Enhanced extraction failed for chunk {i}: {e}")
    
    def print_progress_report(self, processed: int, total: int, failed: int, current_file: str = ""):
        """Print current progress report."""
        percentage = (processed / total * 100) if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"📚 KSIAZKI PDF PROCESSING PROGRESS")
        print(f"{'='*70}")
        print(f"Progress: {processed}/{total} ({percentage:.1f}%)")
        print(f"✅ Successful: {processed - failed}")
        print(f"❌ Failed: {failed}")
        if current_file:
            print(f"🔄 Currently processing: {current_file}")
        print(f"{'='*70}\n")
    
    async def process_batch(self, pdf_files: List[Path]):
        """Process a batch of ksiazki PDF files."""
        
        # Load existing progress
        progress = self.load_progress()
        
        # Filter already processed files
        remaining_files = []
        for pdf_file in pdf_files:
            file_id = self.generate_file_id(pdf_file)
            if file_id not in progress["processed"]:
                remaining_files.append(pdf_file)
            else:
                self.skipped_files += 1
        
        if not remaining_files:
            logger.info("All ksiazki files already processed!")
            return
        
        self.total_files = len(remaining_files)
        logger.info(f"Processing {self.total_files} remaining ksiazki files in batches of {self.batch_size}")
        
        # Process in batches
        for i in range(0, len(remaining_files), self.batch_size):
            batch = remaining_files[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(remaining_files) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            # Process batch sequentially for better stability
            for pdf_file in batch:
                result = await self.process_single_ksiazki_pdf(pdf_file, progress)
                
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    self.failed_files += 1
                else:
                    success, message = result
                    if success:
                        self.processed_files += 1
                    else:
                        self.failed_files += 1
                    
                    logger.info(f"  {pdf_file.name}: {message}")
            
            # Save progress after each batch
            progress["session_stats"]["total_processed"] = self.processed_files
            progress["session_stats"]["total_failed"] = self.failed_files
            self.save_progress(progress)
            
            # Print progress report
            self.print_progress_report(
                self.processed_files + self.skipped_files,
                len(pdf_files),
                self.failed_files
            )
            
            # Small delay between batches
            if i + self.batch_size < len(remaining_files):
                await asyncio.sleep(3)
    
    async def run_ksiazki_processing(self):
        """Run the complete ksiazki PDF processing pipeline."""
        start_time = time.time()
        
        logger.info("🚀 Starting ksiazki PDF processing")
        
        # Discover files
        pdf_files = self.discover_ksiazki_pdfs()
        if not pdf_files:
            logger.warning("No PDF files found in ksiazki folder!")
            return
        
        # Initial progress report
        progress = self.load_progress()
        already_processed = len(progress["processed"])
        
        print(f"\n📚 KSIAZKI PDF PROCESSING SUMMARY")
        print(f"Total PDF files found: {len(pdf_files)}")
        print(f"Already processed: {already_processed}")
        print(f"Remaining to process: {len(pdf_files) - already_processed}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max retries per file: {self.max_retries}")
        print(f"Enhanced prompts: {self.use_enhanced_prompts}")
        print(f"Model: {self.model}")
        print(f"Starting automatically in 3 seconds...")
        
        await asyncio.sleep(3)
        
        # Process files
        await self.process_batch(pdf_files)
        
        # Final report
        total_time = time.time() - start_time
        
        print(f"\n🎉 KSIAZKI PROCESSING COMPLETE!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"✅ Successfully processed: {self.processed_files}")
        print(f"⏭️  Already existed: {self.skipped_files}")
        print(f"❌ Failed: {self.failed_files}")
        
        if self.processed_files + self.failed_files > 0:
            print(f"📊 Success rate: {(self.processed_files/(self.processed_files + self.failed_files)*100):.1f}%")
        
        if self.failed_files > 0:
            print(f"\n❌ Failed files logged in: {self.failed_files_log}")
        
        print(f"📋 Progress saved in: {self.progress_file}")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process ksiazki PDF files")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of files to process concurrently")
    parser.add_argument("--no-enhanced", action="store_true", help="Skip enhanced extraction")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model to use for enhanced extraction")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    try:
        processor = KsiazkiBatchProcessor()
        processor.batch_size = args.batch_size
        processor.use_enhanced_prompts = not args.no_enhanced
        processor.model = args.model
        
        # If max_files specified, limit processing
        if args.max_files:
            original_process_batch = processor.process_batch
            
            async def limited_process_batch(pdf_files):
                limited_files = pdf_files[:args.max_files]
                print(f"🔢 Limiting to first {args.max_files} files for testing")
                return await original_process_batch(limited_files)
            
            processor.process_batch = limited_process_batch
        
        await processor.run_ksiazki_processing()
        
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())