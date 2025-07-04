#!/usr/bin/env python3
"""
Enhanced Batch PDF Processing System for ksiazki pdf folder.
Includes complete extraction pipeline: documents -> chunks -> QA pairs -> distilled knowledge.
Uses enhanced prompts for superior quality extraction.
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
import uuid

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extraction.pdf_processor import PDFProcessor
from extraction.v2.extractors import StatelessExtractor
from src.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

class EnhancedKsiazkiProcessor:
    """
    Complete enhanced PDF processing system for ksiazki pdf folder.
    Processes: PDF -> Chunks -> Enhanced Extraction -> QA Pairs -> Distilled Knowledge
    """
    
    def __init__(self):
        """Initialize enhanced batch processor for ksiazki pdf folder."""
        self.ksiazki_path = Path("/Users/marcin/Desktop/aplikacje/ksiazki pdf")
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
        
        # Progress tracking
        self.progress_file = Path("enhanced_ksiazki_progress.json")
        self.failed_files_log = Path("enhanced_ksiazki_failed.json")
        
        # Processing configuration
        self.batch_size = 1  # Process one book at a time for quality
        self.max_retries = 2
        self.retry_delay = 5  # seconds
        self.model = "gemini-2.5-pro"
        self.temperature = 0.3
        
        # Enhanced extraction settings
        self.max_chunks_for_qa = 10  # Process first 10 chunks for QA pairs
        self.max_chunks_for_knowledge = 15  # Process first 15 chunks for distilled knowledge
        
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
            "processed": {},  # file_id -> detailed status
            "failed": {},     # file_id -> error_info
            "session_stats": {
                "total_sessions": 0,
                "last_session": None,
                "total_processed": 0,
                "total_failed": 0,
                "total_qa_pairs": 0,
                "total_knowledge_items": 0
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
    
    def check_file_processing_status(self, filepath: Path) -> Optional[Dict]:
        """Check if file is already fully processed (including QA pairs and knowledge)."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # Check by source path
                    cur.execute("""
                        SELECT d.id, d.import_status, 
                               (SELECT COUNT(*) FROM qa_pairs WHERE document_id = d.id) as qa_count,
                               (SELECT COUNT(*) FROM distilled_knowledge WHERE document_id = d.id) as knowledge_count
                        FROM documents d
                        WHERE d.source_path = %s
                    """, (str(filepath),))
                    
                    result = cur.fetchone()
                    if result:
                        doc_id, status, qa_count, knowledge_count = result
                        return {
                            "document_id": doc_id,
                            "status": status,
                            "qa_pairs": qa_count,
                            "knowledge_items": knowledge_count,
                            "fully_processed": status == 'completed' and qa_count > 0 and knowledge_count > 0
                        }
                        
        except Exception as e:
            logger.warning(f"Database check failed for {filepath.name}: {e}")
        
        return None
    
    def extract_metadata_from_filename(self, filepath: Path) -> Dict:
        """Extract metadata from filename and path."""
        filename = filepath.stem
        
        # Parse filename format: "Title -- Author -- Year -- Publisher"
        title = filename
        authors = []
        year = None
        publisher = None
        
        if " -- " in filename:
            parts = filename.split(" -- ")
            if len(parts) >= 1:
                title = parts[0]
            if len(parts) >= 2:
                authors = [parts[1]]
            if len(parts) >= 3:
                year = parts[2]
            if len(parts) >= 4:
                publisher = parts[3]
        
        # Extract category from folder structure
        try:
            relative_path = filepath.relative_to(self.ksiazki_path)
            if len(relative_path.parts) > 1:
                category = relative_path.parts[0]
                category = category.replace("_", " ").replace("-", " ").strip()
                category = " ".join(word.capitalize() for word in category.split())
            else:
                category = "General"
        except Exception:
            category = "General"
        
        return {
            "title": title.replace("_", " ").strip(),
            "authors": authors,
            "year": year,
            "publisher": publisher,
            "category": category,
            "original_filename": filename
        }
    
    async def process_single_enhanced_pdf(self, filepath: Path, progress: Dict, retry_count: int = 0) -> Tuple[bool, str]:
        """
        Process a single PDF file with complete enhanced extraction pipeline.
        
        Returns:
            (success, message)
        """
        start_time = time.time()
        file_id = self.generate_file_id(filepath)
        
        try:
            logger.info(f"🚀 Starting enhanced processing: {filepath.name} (attempt {retry_count + 1})")
            
            # Check if already fully processed
            status = self.check_file_processing_status(filepath)
            if status and status["fully_processed"]:
                logger.info(f"File {filepath.name} already fully processed (QA: {status['qa_pairs']}, Knowledge: {status['knowledge_items']})")
                progress["processed"][file_id] = {
                    "status": "already_complete",
                    "document_id": status["document_id"],
                    "qa_pairs": status["qa_pairs"],
                    "knowledge_items": status["knowledge_items"],
                    "timestamp": datetime.now().isoformat()
                }
                return True, f"Already fully processed (QA: {status['qa_pairs']}, Knowledge: {status['knowledge_items']})"
            
            # Step 1: Process PDF and store document/chunks
            logger.info(f"📄 Step 1: Processing PDF content...")
            document_id, pdf_chunks = await self.process_and_store_pdf(filepath)
            
            # Step 2: Generate QA pairs using enhanced prompts
            logger.info(f"❓ Step 2: Generating QA pairs...")
            qa_count = await self.generate_enhanced_qa_pairs(document_id, pdf_chunks[:self.max_chunks_for_qa])
            
            # Step 3: Extract distilled knowledge using enhanced prompts
            logger.info(f"🧠 Step 3: Extracting distilled knowledge...")
            knowledge_count = await self.extract_enhanced_knowledge(document_id, pdf_chunks[:self.max_chunks_for_knowledge])
            
            # Step 4: Update document status
            await self.finalize_document_processing(document_id)
            
            processing_time = time.time() - start_time
            
            # Update progress
            progress["processed"][file_id] = {
                "status": "completed",
                "document_id": document_id,
                "chunks_count": len(pdf_chunks),
                "qa_pairs": qa_count,
                "knowledge_items": knowledge_count,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(filepath)
            }
            
            # Update session stats
            progress["session_stats"]["total_qa_pairs"] += qa_count
            progress["session_stats"]["total_knowledge_items"] += knowledge_count
            
            logger.info(f"✅ Enhanced processing complete for {filepath.name}")
            logger.info(f"   📊 Results: {len(pdf_chunks)} chunks, {qa_count} QA pairs, {knowledge_count} knowledge items")
            logger.info(f"   ⏱️  Time: {processing_time:.1f}s")
            
            return True, f"Enhanced processing complete ({len(pdf_chunks)} chunks, {qa_count} QA, {knowledge_count} knowledge)"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error in enhanced processing {filepath.name}: {error_msg}")
            
            # Log detailed error for debugging
            if retry_count == 0:
                logger.error(f"Detailed error: {traceback.format_exc()}")
            
            # Check if we should retry
            if retry_count < self.max_retries:
                logger.info(f"Retrying {filepath.name} in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
                return await self.process_single_enhanced_pdf(filepath, progress, retry_count + 1)
            else:
                # Final failure
                progress["failed"][file_id] = {
                    "error": error_msg,
                    "retry_count": retry_count,
                    "timestamp": datetime.now().isoformat(),
                    "file_path": str(filepath)
                }
                return False, f"Failed after {retry_count + 1} attempts: {error_msg}"
    
    async def process_and_store_pdf(self, filepath: Path) -> Tuple[str, List]:
        """Process PDF and store document/chunks. Returns (document_id, chunks)."""
        
        # Extract metadata
        metadata = self.extract_metadata_from_filename(filepath)
        file_stat = filepath.stat()
        
        # Process PDF
        pdf_chunks = await self.pdf_processor.process_pdf(str(filepath))
        if not pdf_chunks:
            raise ValueError("No content extracted from PDF")
        
        logger.info(f"   📄 Extracted {len(pdf_chunks)} chunks")
        
        # Store in database
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Check if document already exists
                cur.execute("SELECT id FROM documents WHERE source_path = %s", (str(filepath),))
                existing = cur.fetchone()
                
                if existing:
                    # Update existing document
                    document_id = existing[0]
                    cur.execute("""
                        UPDATE documents SET
                            processed_at = %s,
                            import_status = %s
                        WHERE id = %s
                    """, (datetime.now(), 'processing', document_id))
                else:
                    # Insert new document
                    cur.execute("""
                        INSERT INTO documents (
                            title, authors, source_type, source_path, primary_category,
                            page_count, file_size_bytes, import_status, processed_at,
                            doc_metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        metadata["title"], metadata["authors"], 'ksiazki_pdf', str(filepath), 
                        metadata["category"], len(pdf_chunks), file_stat.st_size, 
                        'processing', datetime.now(),
                        Json({
                            "original_filename": metadata["original_filename"],
                            "category_folder": metadata["category"],
                            "year": metadata["year"],
                            "publisher": metadata["publisher"]
                        })
                    ))
                    
                    document_id = cur.fetchone()[0]
                
                # Clear existing chunks for this document
                cur.execute("DELETE FROM content_chunks WHERE document_id = %s", (document_id,))
                
                # Insert chunks with enhanced metadata
                for i, chunk in enumerate(pdf_chunks):
                    cur.execute("""
                        INSERT INTO content_chunks (
                            document_id, content, chunk_index, page_numbers,
                            chunk_type, summary_sentence, extraction_metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        document_id, chunk.content, i, [chunk.page_num],
                        'text', chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                        Json({
                            "source": "enhanced_pipeline",
                            "chunk_length": len(chunk.content),
                            "page_num": chunk.page_num,
                            "created_at": datetime.now().isoformat()
                        })
                    ))
                
                conn.commit()
        
        return document_id, pdf_chunks
    
    async def generate_enhanced_qa_pairs(self, document_id: str, chunks: List) -> int:
        """Generate QA pairs using enhanced prompts."""
        logger.info(f"   ❓ Generating QA pairs for {len(chunks)} chunks...")
        
        qa_count = 0
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Clear existing QA pairs for this document
                cur.execute("DELETE FROM qa_pairs WHERE document_id = %s", (document_id,))
                
                for i, chunk in enumerate(chunks):
                    try:
                        # Enhanced QA generation prompt
                        qa_prompt = f"""You are a specialized Question-Answer Generation Agent using enhanced extraction techniques.

## Agent Loop Architecture for QA Generation
1. **Analyze Content**: Understand the text structure and key information
2. **Plan Questions**: Determine what questions would best test understanding
3. **Generate Q&A**: Create high-quality questions with comprehensive answers
4. **Validate Quality**: Ensure questions are clear and answers are accurate

## QA Generation Guidelines
- Generate 3-5 high-quality question-answer pairs
- Questions should span different cognitive levels (factual, conceptual, analytical)
- Answers should be comprehensive but concise
- Include confidence scores for each Q&A pair
- Ensure questions are answerable from the given text

Text to analyze:
{chunk.content}

Respond with a JSON object containing:
{{
  "thinking_process": "Your analysis of the content and QA strategy",
  "qa_pairs": [
    {{
      "question": "Clear, specific question",
      "answer": "Comprehensive answer based on the text",
      "question_type": "factual|conceptual|analytical|application",
      "confidence": 0.0-1.0,
      "cognitive_level": "remember|understand|apply|analyze|evaluate|create"
    }}
  ],
  "overall_confidence": 0.0-1.0
}}"""
                        
                        # Generate QA pairs with enhanced extraction
                        start_time = time.time()
                        qa_result = await StatelessExtractor.call_llm(
                            client=None,
                            prompt=qa_prompt,
                            model=self.model,
                            temperature=self.temperature
                        )
                        processing_time_ms = int((time.time() - start_time) * 1000)
                        
                        # Update chunk with enhanced metadata
                        cur.execute("""
                            UPDATE content_chunks 
                            SET extraction_metadata = %s
                            WHERE document_id = %s AND chunk_index = %s
                        """, (
                            Json({
                                "enhanced_extraction": True,
                                "thinking_process": qa_result.get("thinking_process", ""),
                                "extraction_quality": qa_result.get("overall_confidence", 0.8),
                                "confidence_details": {
                                    "overall_confidence": qa_result.get("overall_confidence", 0.8),
                                    "qa_pairs_generated": len(qa_result.get("qa_pairs", []))
                                },
                                "processing_time_ms": processing_time_ms,
                                "enhanced_at": datetime.now().isoformat(),
                                "extraction_type": "qa_generation"
                            }),
                            document_id, i
                        ))
                        
                        # Store QA pairs
                        qa_pairs = qa_result.get("qa_pairs", [])
                        for qa_pair in qa_pairs:
                            qa_id = str(uuid.uuid4())
                            
                            # Get chunk UUID for source_chunk_ids
                            cur.execute("SELECT id FROM content_chunks WHERE document_id = %s AND chunk_index = %s", (document_id, i))
                            chunk_result = cur.fetchone()
                            chunk_uuid = chunk_result[0] if chunk_result else None
                            
                            cur.execute("""
                                INSERT INTO qa_pairs (
                                    id, document_id, question, answer, answer_confidence,
                                    answer_type, source_chunk_ids, human_verified, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s)
                            """, (
                                qa_id, document_id, qa_pair.get("question", ""),
                                qa_pair.get("answer", ""), qa_pair.get("confidence", 0.8),
                                qa_pair.get("question_type", "factual"), 
                                [str(chunk_uuid)] if chunk_uuid else [], False,
                                datetime.now()
                            ))
                            
                            qa_count += 1
                        
                        logger.info(f"      Generated {len(qa_pairs)} QA pairs from chunk {i}")
                        
                    except Exception as e:
                        logger.warning(f"      Failed to generate QA for chunk {i}: {e}")
                
                conn.commit()
        
        logger.info(f"   ✅ Generated {qa_count} total QA pairs")
        return qa_count
    
    async def extract_enhanced_knowledge(self, document_id: str, chunks: List) -> int:
        """Extract distilled knowledge using enhanced prompts."""
        logger.info(f"   🧠 Extracting knowledge from {len(chunks)} chunks...")
        
        knowledge_count = 0
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Clear existing knowledge for this document
                cur.execute("DELETE FROM distilled_knowledge WHERE document_id = %s", (document_id,))
                
                # First, extract overall document summary
                try:
                    combined_content = "\n\n".join([chunk.content for chunk in chunks[:5]])  # First 5 chunks for summary
                    
                    summary_prompt = f"""You are a specialized Knowledge Distillation Agent using enhanced extraction techniques.

## Agent Loop Architecture for Knowledge Distillation
1. **Analyze Content**: Understand the overall themes and structure
2. **Plan Distillation**: Determine key knowledge types to extract
3. **Execute Distillation**: Extract structured knowledge systematically
4. **Validate Quality**: Ensure knowledge is accurate and useful

## Knowledge Distillation Task
Extract high-level knowledge from this document content:

{combined_content}

Respond with a JSON object containing:
{{
  "thinking_process": "Your analysis approach and strategy",
  "document_summary": "Comprehensive summary of the document's main points",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "main_themes": ["theme1", "theme2", "theme3"],
  "knowledge_type": "summary|concepts|themes|principles",
  "quality_score": 0.0-1.0,
  "confidence": 0.0-1.0
}}"""
                    
                    summary_result = await StatelessExtractor.call_llm(
                        client=None,
                        prompt=summary_prompt,
                        model=self.model,
                        temperature=self.temperature
                    )
                    
                    # Store document summary
                    knowledge_id = str(uuid.uuid4())
                    cur.execute("""
                        INSERT INTO distilled_knowledge (
                            id, document_id, knowledge_type, scope, content,
                            structured_data, distillation_method, quality_score,
                            related_chunks, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        knowledge_id, document_id, "summary", "document",
                        summary_result.get("document_summary", ""),
                        Json({
                            "key_concepts": summary_result.get("key_concepts", []),
                            "main_themes": summary_result.get("main_themes", []),
                            "thinking_process": summary_result.get("thinking_process", "")
                        }),
                        "enhanced_agent_loop", summary_result.get("quality_score", 0.8),
                        "{}", datetime.now()  # Empty PostgreSQL array
                    ))
                    knowledge_count += 1
                    
                except Exception as e:
                    logger.warning(f"      Failed to extract document summary: {e}")
                
                # Extract detailed knowledge from individual chunks
                for i, chunk in enumerate(chunks[:10]):  # Process first 10 chunks for detailed knowledge
                    try:
                        knowledge_prompt = f"""You are a specialized Knowledge Extraction Agent using enhanced techniques.

Extract detailed knowledge from this text chunk:

{chunk.content}

Respond with a JSON object containing:
{{
  "facts": ["Important fact 1", "Important fact 2"],
  "concepts": ["Key concept 1", "Key concept 2"],
  "relationships": [
    {{"source": "concept1", "relationship": "causes", "target": "concept2"}},
    {{"source": "concept2", "relationship": "leads_to", "target": "outcome1"}}
  ],
  "principles": ["Principle 1", "Principle 2"],
  "quality_score": 0.0-1.0
}}"""
                        
                        knowledge_result = await StatelessExtractor.call_llm(
                            client=None,
                            prompt=knowledge_prompt,
                            model=self.model,
                            temperature=self.temperature
                        )
                        
                        # Store facts
                        facts = knowledge_result.get("facts", [])
                        for fact in facts:
                            knowledge_id = str(uuid.uuid4())
                            cur.execute("""
                                INSERT INTO distilled_knowledge (
                                    id, document_id, knowledge_type, scope, content,
                                    structured_data, distillation_method, quality_score,
                                    related_chunks, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                knowledge_id, document_id, "facts", f"chunk_{i}",
                                fact, Json({"chunk_index": i}),
                                "enhanced_agent_loop", knowledge_result.get("quality_score", 0.8),
                                "{}", datetime.now()  # Empty PostgreSQL array
                            ))
                            knowledge_count += 1
                        
                        # Store concepts
                        concepts = knowledge_result.get("concepts", [])
                        for concept in concepts:
                            knowledge_id = str(uuid.uuid4())
                            cur.execute("""
                                INSERT INTO distilled_knowledge (
                                    id, document_id, knowledge_type, scope, content,
                                    structured_data, distillation_method, quality_score,
                                    related_chunks, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                knowledge_id, document_id, "concepts", f"chunk_{i}",
                                concept, Json({"chunk_index": i}),
                                "enhanced_agent_loop", knowledge_result.get("quality_score", 0.8),
                                "{}", datetime.now()  # Empty PostgreSQL array
                            ))
                            knowledge_count += 1
                        
                        # Store relationships
                        relationships = knowledge_result.get("relationships", [])
                        for relationship in relationships:
                            knowledge_id = str(uuid.uuid4())
                            cur.execute("""
                                INSERT INTO distilled_knowledge (
                                    id, document_id, knowledge_type, scope, content,
                                    structured_data, distillation_method, quality_score,
                                    related_chunks, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                knowledge_id, document_id, "relationships", f"chunk_{i}",
                                f"{relationship.get('source', '')} {relationship.get('relationship', '')} {relationship.get('target', '')}",
                                Json(relationship), "enhanced_agent_loop", 
                                knowledge_result.get("quality_score", 0.8),
                                "{}", datetime.now()
                            ))
                            knowledge_count += 1
                        
                        logger.info(f"      Extracted knowledge from chunk {i}: {len(facts)} facts, {len(concepts)} concepts, {len(relationships)} relationships")
                        
                    except Exception as e:
                        logger.warning(f"      Failed to extract knowledge from chunk {i}: {e}")
                
                conn.commit()
        
        logger.info(f"   ✅ Extracted {knowledge_count} total knowledge items")
        return knowledge_count
    
    async def finalize_document_processing(self, document_id: str):
        """Update document status to completed."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents 
                    SET import_status = 'completed', processed_at = %s
                    WHERE id = %s
                """, (datetime.now(), document_id))
                conn.commit()
    
    def print_progress_report(self, processed: int, total: int, failed: int, current_file: str = ""):
        """Print current progress report."""
        percentage = (processed / total * 100) if total > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"🚀 ENHANCED KSIAZKI PROCESSING PROGRESS")
        print(f"{'='*80}")
        print(f"Progress: {processed}/{total} ({percentage:.1f}%)")
        print(f"✅ Successful: {processed - failed}")
        print(f"❌ Failed: {failed}")
        if current_file:
            print(f"🔄 Currently processing: {current_file}")
        print(f"{'='*80}\n")
    
    async def process_batch(self, pdf_files: List[Path]):
        """Process a batch of ksiazki PDF files with enhanced extraction."""
        
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
        logger.info(f"Processing {self.total_files} remaining ksiazki files with enhanced extraction")
        
        # Process files one by one for quality
        for i, pdf_file in enumerate(remaining_files):
            logger.info(f"\n🔄 Processing file {i+1}/{len(remaining_files)}")
            
            result = await self.process_single_enhanced_pdf(pdf_file, progress)
            
            if isinstance(result, Exception):
                logger.error(f"Processing error: {result}")
                self.failed_files += 1
            else:
                success, message = result
                if success:
                    self.processed_files += 1
                else:
                    self.failed_files += 1
                
                logger.info(f"  {pdf_file.name}: {message}")
            
            # Save progress after each file
            progress["session_stats"]["total_processed"] = self.processed_files
            progress["session_stats"]["total_failed"] = self.failed_files
            self.save_progress(progress)
            
            # Print progress report
            self.print_progress_report(
                self.processed_files + self.skipped_files,
                len(pdf_files),
                self.failed_files,
                pdf_file.name if i < len(remaining_files) - 1 else ""
            )
            
            # Small delay between files
            if i < len(remaining_files) - 1:
                await asyncio.sleep(2)
    
    async def run_enhanced_processing(self):
        """Run the complete enhanced ksiazki PDF processing pipeline."""
        start_time = time.time()
        
        logger.info("🚀 Starting enhanced ksiazki PDF processing with complete pipeline")
        
        # Discover files
        pdf_files = self.discover_ksiazki_pdfs()
        if not pdf_files:
            logger.warning("No PDF files found in ksiazki folder!")
            return
        
        # Initial progress report
        progress = self.load_progress()
        already_processed = len(progress["processed"])
        
        print(f"\n📚 ENHANCED KSIAZKI PROCESSING SUMMARY")
        print(f"Total PDF files found: {len(pdf_files)}")
        print(f"Already processed: {already_processed}")
        print(f"Remaining to process: {len(pdf_files) - already_processed}")
        print(f"Model: {self.model}")
        print(f"Enhanced extraction: ✅ Full pipeline (chunks + QA pairs + knowledge)")
        print(f"Processing approach: One file at a time for maximum quality")
        print(f"Starting automatically in 5 seconds...")
        
        await asyncio.sleep(5)
        
        # Process files
        await self.process_batch(pdf_files)
        
        # Final report
        total_time = time.time() - start_time
        
        print(f"\n🎉 ENHANCED PROCESSING COMPLETE!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"✅ Successfully processed: {self.processed_files}")
        print(f"⏭️  Already existed: {self.skipped_files}")
        print(f"❌ Failed: {self.failed_files}")
        
        if self.processed_files + self.failed_files > 0:
            print(f"📊 Success rate: {(self.processed_files/(self.processed_files + self.failed_files)*100):.1f}%")
        
        print(f"📋 Progress saved in: {self.progress_file}")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced batch process ksiazki PDF files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model to use for enhanced extraction")
    
    args = parser.parse_args()
    
    try:
        processor = EnhancedKsiazkiProcessor()
        processor.model = args.model
        
        # If max_files specified, limit processing
        if args.max_files:
            original_process_batch = processor.process_batch
            
            async def limited_process_batch(pdf_files):
                limited_files = pdf_files[:args.max_files]
                print(f"🔢 Limiting to first {args.max_files} files for testing")
                return await original_process_batch(limited_files)
            
            processor.process_batch = limited_process_batch
        
        await processor.run_enhanced_processing()
        
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())