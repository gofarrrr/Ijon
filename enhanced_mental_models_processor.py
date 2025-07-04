#!/usr/bin/env python3
"""
Enhanced Mental Models PDF Processing System.
Focuses on processing PDFs from the mental models folder with complete enhanced pipeline.
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

class EnhancedMentalModelsProcessor:
    """
    Enhanced PDF processing system for mental models folder.
    Processes: PDF -> Chunks -> Enhanced Extraction -> QA Pairs -> Distilled Knowledge
    """
    
    def __init__(self):
        """Initialize enhanced mental models processor."""
        self.mental_models_path = Path("/Users/marcin/Desktop/aplikacje/ksiazki pdf/modele mentalne psychologia selfhelp podejmowanie decyzji")
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
        
        # Progress tracking
        self.progress_file = Path("enhanced_mental_models_progress.json")
        self.failed_files_log = Path("enhanced_mental_models_failed.json")
        
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
    
    def discover_mental_models_pdfs(self) -> List[Path]:
        """Discover all PDF files in the mental models folder."""
        logger.info(f"Discovering PDF files in {self.mental_models_path}")
        
        if not self.mental_models_path.exists():
            logger.error(f"Mental models folder not found: {self.mental_models_path}")
            return []
        
        pdf_files = []
        for pdf_file in self.mental_models_path.glob("*.pdf"):
            if pdf_file.is_file():
                pdf_files.append(pdf_file)
        
        # Sort by size (smallest first for easier processing)
        pdf_files.sort(key=lambda f: f.stat().st_size if f.exists() else float('inf'))
        
        logger.info(f"Found {len(pdf_files)} PDF files in mental models folder")
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
        
        return {
            "title": title.replace("_", " ").strip(),
            "authors": authors,
            "year": year,
            "publisher": publisher,
            "category": "Mental Models & Decision Making",
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
                        metadata["title"], metadata["authors"], 'mental_models_pdf', str(filepath), 
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
                            "source": "enhanced_mental_models_pipeline",
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
                        # Enhanced QA generation prompt for mental models content
                        qa_prompt = f"""You are a specialized Question-Answer Generation Agent for Mental Models and Decision Making content.

## Agent Loop Architecture for Mental Models QA Generation
1. **Analyze Content**: Understand mental model concepts, decision frameworks, and psychological principles
2. **Plan Questions**: Create questions that test practical application of mental models
3. **Generate Q&A**: Focus on "how-to" and "when-to-use" questions for mental models
4. **Validate Quality**: Ensure questions help readers apply mental models in real situations

## Mental Models QA Guidelines
- Generate 3-5 high-quality question-answer pairs focused on practical application
- Questions should help readers understand WHEN and HOW to use specific mental models
- Include questions about biases, decision frameworks, and cognitive principles
- Answers should be actionable and include examples when possible
- Focus on practical wisdom that readers can immediately apply

Text to analyze:
{chunk.content}

Respond with a JSON object containing:
{{
  "thinking_process": "Your analysis of mental models and decision concepts in this text",
  "qa_pairs": [
    {{
      "question": "How can this mental model be applied in real-world decisions?",
      "answer": "Practical answer with examples and application guidance",
      "question_type": "application|conceptual|bias_identification|framework_usage",
      "confidence": 0.0-1.0,
      "cognitive_level": "apply|analyze|evaluate|create",
      "mental_model_focus": "specific mental model or bias discussed"
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
                                "extraction_type": "mental_models_qa_generation"
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
                                qa_pair.get("question_type", "application"), 
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
                
                # First, extract overall mental models summary
                try:
                    combined_content = "\n\n".join([chunk.content for chunk in chunks[:5]])  # First 5 chunks for summary
                    
                    summary_prompt = f"""You are a specialized Mental Models Knowledge Distillation Agent.

## Agent Loop Architecture for Mental Models Knowledge Distillation
1. **Analyze Content**: Identify mental models, cognitive biases, and decision frameworks
2. **Plan Distillation**: Determine key mental models and their applications
3. **Execute Distillation**: Extract actionable mental models and decision tools
4. **Validate Quality**: Ensure knowledge is practically useful for decision making

## Mental Models Knowledge Distillation Task
Extract key mental models and decision-making insights from this content:

{combined_content}

Respond with a JSON object containing:
{{
  "thinking_process": "Your analysis of mental models and decision concepts",
  "document_summary": "Summary focused on practical mental models and decision tools",
  "key_mental_models": ["Model 1", "Model 2", "Model 3"],
  "cognitive_biases": ["Bias 1", "Bias 2", "Bias 3"],
  "decision_frameworks": ["Framework 1", "Framework 2"],
  "practical_applications": ["Application 1", "Application 2"],
  "knowledge_type": "mental_models_summary",
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
                        knowledge_id, document_id, "mental_models_summary", "document",
                        summary_result.get("document_summary", ""),
                        Json({
                            "key_mental_models": summary_result.get("key_mental_models", []),
                            "cognitive_biases": summary_result.get("cognitive_biases", []),
                            "decision_frameworks": summary_result.get("decision_frameworks", []),
                            "practical_applications": summary_result.get("practical_applications", []),
                            "thinking_process": summary_result.get("thinking_process", "")
                        }),
                        "enhanced_mental_models_agent", summary_result.get("quality_score", 0.8),
                        "{}", datetime.now()  # Empty PostgreSQL array
                    ))
                    knowledge_count += 1
                    
                except Exception as e:
                    logger.warning(f"      Failed to extract document summary: {e}")
                
                # Extract detailed mental models from individual chunks
                for i, chunk in enumerate(chunks[:10]):  # Process first 10 chunks for detailed knowledge
                    try:
                        knowledge_prompt = f"""You are a Mental Models Knowledge Extraction Agent.

Extract specific mental models and decision tools from this text chunk:

{chunk.content}

Respond with a JSON object containing:
{{
  "mental_models": ["Specific mental model 1", "Specific mental model 2"],
  "cognitive_biases": ["Bias mentioned in text"],
  "decision_tools": ["Tool 1", "Tool 2"],
  "key_insights": ["Insight 1", "Insight 2"],
  "practical_tips": ["Tip 1", "Tip 2"],
  "quality_score": 0.0-1.0
}}"""
                        
                        knowledge_result = await StatelessExtractor.call_llm(
                            client=None,
                            prompt=knowledge_prompt,
                            model=self.model,
                            temperature=self.temperature
                        )
                        
                        # Store mental models
                        mental_models = knowledge_result.get("mental_models", [])
                        for model in mental_models:
                            knowledge_id = str(uuid.uuid4())
                            cur.execute("""
                                INSERT INTO distilled_knowledge (
                                    id, document_id, knowledge_type, scope, content,
                                    structured_data, distillation_method, quality_score,
                                    related_chunks, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                knowledge_id, document_id, "mental_models", f"chunk_{i}",
                                model, Json({"chunk_index": i}),
                                "enhanced_mental_models_agent", knowledge_result.get("quality_score", 0.8),
                                "{}", datetime.now()  # Empty PostgreSQL array
                            ))
                            knowledge_count += 1
                        
                        # Store decision tools
                        decision_tools = knowledge_result.get("decision_tools", [])
                        for tool in decision_tools:
                            knowledge_id = str(uuid.uuid4())
                            cur.execute("""
                                INSERT INTO distilled_knowledge (
                                    id, document_id, knowledge_type, scope, content,
                                    structured_data, distillation_method, quality_score,
                                    related_chunks, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                knowledge_id, document_id, "decision_tools", f"chunk_{i}",
                                tool, Json({"chunk_index": i}),
                                "enhanced_mental_models_agent", knowledge_result.get("quality_score", 0.8),
                                "{}", datetime.now()  # Empty PostgreSQL array
                            ))
                            knowledge_count += 1
                        
                        # Store key insights
                        insights = knowledge_result.get("key_insights", [])
                        for insight in insights:
                            knowledge_id = str(uuid.uuid4())
                            cur.execute("""
                                INSERT INTO distilled_knowledge (
                                    id, document_id, knowledge_type, scope, content,
                                    structured_data, distillation_method, quality_score,
                                    related_chunks, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                knowledge_id, document_id, "insights", f"chunk_{i}",
                                insight, Json({"chunk_index": i}),
                                "enhanced_mental_models_agent", knowledge_result.get("quality_score", 0.8),
                                "{}", datetime.now()
                            ))
                            knowledge_count += 1
                        
                        logger.info(f"      Extracted from chunk {i}: {len(mental_models)} models, {len(decision_tools)} tools, {len(insights)} insights")
                        
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
        print(f"🧠 ENHANCED MENTAL MODELS PROCESSING PROGRESS")
        print(f"{'='*80}")
        print(f"Progress: {processed}/{total} ({percentage:.1f}%)")
        print(f"✅ Successful: {processed - failed}")
        print(f"❌ Failed: {failed}")
        if current_file:
            print(f"🔄 Currently processing: {current_file}")
        print(f"{'='*80}\n")
    
    async def process_batch(self, pdf_files: List[Path]):
        """Process a batch of mental models PDF files with enhanced extraction."""
        
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
            logger.info("All mental models files already processed!")
            return
        
        self.total_files = len(remaining_files)
        logger.info(f"Processing {self.total_files} remaining mental models files with enhanced extraction")
        
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
        """Run the complete enhanced mental models PDF processing pipeline."""
        start_time = time.time()
        
        logger.info("🧠 Starting enhanced mental models PDF processing with complete pipeline")
        
        # Discover files
        pdf_files = self.discover_mental_models_pdfs()
        if not pdf_files:
            logger.warning("No PDF files found in mental models folder!")
            return
        
        # Initial progress report
        progress = self.load_progress()
        already_processed = len(progress["processed"])
        
        print(f"\n🧠 ENHANCED MENTAL MODELS PROCESSING SUMMARY")
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
        
        print(f"\n🎉 ENHANCED MENTAL MODELS PROCESSING COMPLETE!")
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
    
    parser = argparse.ArgumentParser(description="Enhanced batch process mental models PDF files")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process (for testing)")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model to use for enhanced extraction")
    
    args = parser.parse_args()
    
    try:
        processor = EnhancedMentalModelsProcessor()
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