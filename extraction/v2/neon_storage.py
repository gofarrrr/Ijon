"""
Neon PostgreSQL storage adapter for extracted knowledge.

Provides persistent storage of extraction results in Neon database.
"""

import os
import json
import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from dotenv import load_dotenv

from extraction.models import ExtractedKnowledge
from extraction.v2.state import ExtractionState
from src.utils.logging import get_logger

load_dotenv()
logger = get_logger(__name__)


class NeonStorage:
    """Neon PostgreSQL storage for extraction knowledge using existing schema."""
    
    def __init__(self):
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
        
        logger.info("Using existing Neon database schema")
    
    async def _verify_existing_schema(self):
        """Verify existing schema is compatible with our needs."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # Check existing tables
                    cur.execute("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    
                    tables = [row[0] for row in cur.fetchall()]
                    expected_tables = ['documents', 'content_chunks', 'distilled_knowledge', 'agent_memories', 'qa_pairs']
                    
                    missing_tables = [t for t in expected_tables if t not in tables]
                    if missing_tables:
                        logger.warning(f"Missing expected tables: {missing_tables}")
                    
                    logger.info(f"Found existing tables: {tables}")
                    logger.info("Using existing Neon database schema for context engineering")
                    
        except Exception as e:
            logger.error(f"Error verifying Neon schema: {e}")
            raise
    
    async def store_extraction(self, state: ExtractionState, full_text_chunks: list = None) -> bool:
        """
        Store extraction knowledge AND full text chunks using existing Neon schema.
        
        Args:
            state: ExtractionState with extracted knowledge
            full_text_chunks: List of text chunks for RAG retrieval (optional)
        """
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # Extract filename from path
                    pdf_filename = os.path.basename(state.pdf_path) if state.pdf_path else None
                    
                    # 1. Store document record (using existing documents table schema)
                    cur.execute("""
                        INSERT INTO documents (
                            id, title, source_path, source_type, doc_metadata, 
                            import_status, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            source_path = EXCLUDED.source_path,
                            doc_metadata = EXCLUDED.doc_metadata,
                            import_status = EXCLUDED.import_status,
                            updated_at = EXCLUDED.updated_at
                    """, (
                        state.id,
                        pdf_filename or "Unknown Document",
                        state.pdf_path,
                        "pdf_extraction",
                        Json({
                            "document_type": state.metadata.get('document_type'),
                            "status": state.status,
                            "current_step": state.current_step,
                            "quality_score": state.quality_report.get('overall_score') if state.quality_report else None,
                            "processing_time": state.metadata.get('processing_time'),
                            "model_used": state.metadata.get('model_used'),
                            "extraction_metadata": state.metadata,
                            "summary": state.extraction.get('summary', '') if state.extraction else ''
                        }),
                        "completed" if state.status == "completed" else "processing",
                        datetime.fromisoformat(state.created_at) if state.created_at else datetime.utcnow(),
                        datetime.fromisoformat(state.updated_at) if state.updated_at else datetime.utcnow()
                    ))
                    
                    # 2. Store extracted knowledge as content chunks
                    if state.extraction:
                        extraction = state.extraction
                        
                        # Store topics as content chunks
                        if extraction.get('topics'):
                            for i, topic in enumerate(extraction['topics']):
                                cur.execute("""
                                    INSERT INTO content_chunks (
                                        id, document_id, chunk_index, content, 
                                        chunk_type, created_at
                                    ) VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (id) DO UPDATE SET
                                        content = EXCLUDED.content,
                                        chunk_type = EXCLUDED.chunk_type
                                """, (
                                    str(uuid.uuid4()),
                                    state.id,
                                    i,
                                    f"Topic: {topic.get('name', '')} - {topic.get('description', '')}",
                                    "extracted_topic",
                                    datetime.utcnow()
                                ))
                        
                        # Store facts as content chunks
                        if extraction.get('facts'):
                            for i, fact in enumerate(extraction['facts']):
                                cur.execute("""
                                    INSERT INTO content_chunks (
                                        id, document_id, chunk_index, content, 
                                        chunk_type, created_at
                                    ) VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (id) DO UPDATE SET
                                        content = EXCLUDED.content,
                                        chunk_type = EXCLUDED.chunk_type
                                """, (
                                    str(uuid.uuid4()),
                                    state.id,
                                    i + 1000,  # Offset to avoid conflicts
                                    f"Fact: {fact.get('claim', '')} Evidence: {fact.get('evidence', '')}",
                                    "extracted_fact",
                                    datetime.utcnow()
                                ))
                        
                        # Store relationships as distilled knowledge
                        if extraction.get('relationships'):
                            for i, rel in enumerate(extraction['relationships']):
                                cur.execute("""
                                    INSERT INTO distilled_knowledge (
                                        id, document_id, knowledge_type, 
                                        content, structured_data, quality_score, created_at
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (id) DO UPDATE SET
                                        content = EXCLUDED.content,
                                        structured_data = EXCLUDED.structured_data,
                                        quality_score = EXCLUDED.quality_score
                                """, (
                                    str(uuid.uuid4()),
                                    state.id,
                                    "relationship",
                                    f"Relationship: {rel.get('source_entity', '')} {rel.get('relationship_type', '')} {rel.get('target_entity', '')}",
                                    Json({
                                        "source_entity": rel.get('source_entity', ''),
                                        "target_entity": rel.get('target_entity', ''),
                                        "relationship_type": rel.get('relationship_type', ''),
                                        "description": rel.get('description', '')
                                    }),
                                    rel.get('confidence', 0.0),
                                    datetime.utcnow()
                                ))
                        
                        # Store questions as QA pairs for future reference
                        if extraction.get('questions'):
                            for i, question in enumerate(extraction['questions']):
                                cur.execute("""
                                    INSERT INTO qa_pairs (
                                        id, document_id, question, answer, 
                                        answer_confidence, answer_type, created_at
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (id) DO UPDATE SET
                                        question = EXCLUDED.question,
                                        answer_confidence = EXCLUDED.answer_confidence,
                                        answer_type = EXCLUDED.answer_type
                                """, (
                                    str(uuid.uuid4()),
                                    state.id,
                                    question.get('question_text', ''),
                                    "",  # Will be filled by questioning system later
                                    question.get('confidence', 0.0),
                                    "extracted_question",
                                    datetime.utcnow()
                                ))
                    
                    # 3. Store extraction summary as distilled knowledge
                    if state.extraction and state.extraction.get('summary'):
                        cur.execute("""
                            INSERT INTO distilled_knowledge (
                                id, document_id, knowledge_type, 
                                content, structured_data, quality_score, created_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET
                                content = EXCLUDED.content,
                                structured_data = EXCLUDED.structured_data,
                                quality_score = EXCLUDED.quality_score
                        """, (
                            str(uuid.uuid4()),
                            state.id,
                            "summary",
                            state.extraction['summary'],
                            Json({
                                "quality_score": state.quality_report.get('overall_score') if state.quality_report else None,
                                "extraction_metadata": state.metadata
                            }),
                            state.extraction.get('overall_confidence', 0.5),
                            datetime.utcnow()
                        ))
                    
                    # 4. Store full text chunks for RAG retrieval (if provided)
                    if full_text_chunks:
                        logger.info(f"Storing {len(full_text_chunks)} full text chunks for RAG")
                        for i, chunk_data in enumerate(full_text_chunks):
                            # chunk_data should be a dict with 'content', 'page_numbers', etc.
                            chunk_content = chunk_data.get('content', '') if isinstance(chunk_data, dict) else str(chunk_data)
                            page_numbers = chunk_data.get('page_numbers', []) if isinstance(chunk_data, dict) else []
                            
                            cur.execute("""
                                INSERT INTO content_chunks (
                                    id, document_id, chunk_index, content, 
                                    chunk_type, page_numbers, created_at
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (id) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    chunk_type = EXCLUDED.chunk_type,
                                    page_numbers = EXCLUDED.page_numbers
                            """, (
                                str(uuid.uuid4()),
                                state.id,
                                i + 2000,  # Offset to avoid conflicts with extracted content
                                chunk_content,
                                "full_text_chunk",
                                page_numbers,
                                datetime.utcnow()
                            ))
                    
                    conn.commit()
                    logger.info(f"Successfully stored extraction {state.id} in Neon database using existing schema")
                    if full_text_chunks:
                        logger.info(f"Stored {len(full_text_chunks)} additional full text chunks for RAG")
                    return True
                    
        except Exception as e:
            logger.error(f"Error storing extraction in Neon: {e}")
            logger.exception("Full error details:")
            return False
    
    async def get_extraction_summary(self, extraction_id: str) -> Optional[Dict[str, Any]]:
        """Get extraction summary from Neon database using existing schema."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get document info and count related content
                    cur.execute("""
                        SELECT d.*, 
                               COUNT(DISTINCT cc.id) as content_chunks_count,
                               COUNT(DISTINCT dk.id) as distilled_knowledge_count,
                               COUNT(DISTINCT qa.id) as qa_pairs_count
                        FROM documents d
                        LEFT JOIN content_chunks cc ON d.id = cc.document_id
                        LEFT JOIN distilled_knowledge dk ON d.id = dk.document_id  
                        LEFT JOIN qa_pairs qa ON d.id = qa.document_id
                        WHERE d.id = %s
                        GROUP BY d.id
                    """, (extraction_id,))
                    
                    result = cur.fetchone()
                    if result:
                        return dict(result)
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting extraction summary: {e}")
            return None
    
    async def query_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query extracted knowledge using existing schema and full-text search."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Search across content chunks, distilled knowledge, and QA pairs
                    search_query = f"%{query}%"
                    
                    cur.execute("""
                        (SELECT 'content_chunk' as type, cc.content as content, 
                                cc.chunk_type as knowledge_type, d.title as document_title,
                                0.5 as confidence, 
                                d.id as document_id
                         FROM content_chunks cc 
                         JOIN documents d ON cc.document_id = d.id
                         WHERE cc.content ILIKE %s)
                        UNION ALL
                        (SELECT 'distilled_knowledge' as type, dk.content as content, 
                                dk.knowledge_type, d.title as document_title,
                                dk.quality_score as confidence, d.id as document_id
                         FROM distilled_knowledge dk 
                         JOIN documents d ON dk.document_id = d.id
                         WHERE dk.content ILIKE %s)
                        UNION ALL
                        (SELECT 'qa_pair' as type, qa.question as content, 
                                COALESCE(qa.answer, 'No answer yet') as knowledge_type, 
                                d.title as document_title,
                                qa.answer_confidence as confidence, d.id as document_id
                         FROM qa_pairs qa 
                         JOIN documents d ON qa.document_id = d.id
                         WHERE qa.question ILIKE %s OR qa.answer ILIKE %s)
                        ORDER BY confidence DESC
                        LIMIT %s
                    """, (search_query, search_query, search_query, search_query, limit))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            logger.error(f"Error querying knowledge: {e}")
            return []
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics using existing schema."""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            (SELECT COUNT(*) FROM documents) as total_documents,
                            (SELECT COUNT(*) FROM content_chunks) as total_content_chunks,
                            (SELECT COUNT(*) FROM distilled_knowledge) as total_distilled_knowledge,
                            (SELECT COUNT(*) FROM qa_pairs) as total_qa_pairs,
                            (SELECT COUNT(*) FROM agent_memories) as total_agent_memories,
                            (SELECT AVG(quality_score) FROM distilled_knowledge WHERE quality_score IS NOT NULL) as avg_confidence,
                            (SELECT COUNT(*) FROM documents WHERE doc_metadata->>'status' = 'completed') as completed_extractions
                    """)
                    
                    result = cur.fetchone()
                    return dict(result) if result else {}
                    
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


# Global instance
neon_storage = NeonStorage()