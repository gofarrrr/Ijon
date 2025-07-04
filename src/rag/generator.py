"""
Answer generation module for RAG pipeline.

This module handles generating answers from retrieved context,
including proper citation formatting and confidence scoring.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from src.config import get_settings
from src.models import GeneratedAnswer, PDFChunk, RAGContext
from src.utils.errors import ContextWindowExceededError, GenerationError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class AnswerGenerator:
    """
    Generate answers from retrieved context using LLMs.
    
    Features:
    - Context-aware answer generation
    - Automatic citation insertion
    - Confidence scoring
    - Token limit management
    - Multiple LLM provider support
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_context_tokens: int = 3000,
        max_answer_tokens: int = 500,
        temperature: float = 0.3,
        include_confidence: bool = True,
    ) -> None:
        """
        Initialize answer generator.

        Args:
            model_name: LLM model name
            max_context_tokens: Maximum tokens for context
            max_answer_tokens: Maximum tokens for answer
            temperature: Generation temperature
            include_confidence: Whether to calculate confidence scores
        """
        self.settings = get_settings()
        self.model_name = model_name or "gpt-3.5-turbo"
        self.max_context_tokens = max_context_tokens
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.include_confidence = include_confidence
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except:
            # Fallback to cl100k_base encoding
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # LLM client will be initialized lazily
        self._llm_client = None

    def _ensure_llm_client(self) -> None:
        """Ensure LLM client is initialized."""
        if self._llm_client is None:
            try:
                import openai
                self._llm_client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            except Exception as e:
                raise GenerationError(f"Failed to initialize OpenAI client: {str(e)}")

    @log_performance
    async def generate_answer(
        self,
        query: str,
        chunks: List[Tuple[PDFChunk, float]],
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        """
        Generate an answer from query and retrieved chunks.

        Args:
            query: User query
            chunks: List of (chunk, score) tuples
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer with citations

        Raises:
            GenerationError: If generation fails
            ContextWindowExceededError: If context too long
        """
        start_time = time.time()
        
        try:
            # Prepare context
            rag_context = self._prepare_context(query, chunks)
            
            # Check token limits
            self._check_token_limits(rag_context)
            
            # Generate answer
            answer_text = await self._generate_llm_answer(
                rag_context,
                system_prompt or self._get_default_system_prompt(),
            )
            
            # Extract and format citations
            answer_with_citations, citations = self._process_citations(
                answer_text,
                chunks,
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(chunks) if self.include_confidence else 0.8
            
            # Create result
            result = GeneratedAnswer(
                query=query,
                answer=answer_with_citations,
                citations=citations,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                model_used=self.model_name,
            )
            
            logger.info(
                f"Generated answer in {result.processing_time:.2f}s",
                extra={
                    "query_length": len(query),
                    "chunks_used": len(chunks),
                    "answer_length": len(answer_with_citations),
                },
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise GenerationError(f"Answer generation failed: {str(e)}")

    def _prepare_context(
        self,
        query: str,
        chunks: List[Tuple[PDFChunk, float]],
    ) -> RAGContext:
        """Prepare context for generation."""
        # Sort chunks by score (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        
        # Extract just the chunks
        chunk_objects = [chunk for chunk, _ in sorted_chunks]
        
        # Count tokens
        context_text = "\n\n".join(chunk.content for chunk in chunk_objects)
        total_tokens = len(self.tokenizer.encode(context_text))
        
        return RAGContext(
            query=query,
            relevant_chunks=chunk_objects,
            total_tokens=total_tokens,
        )

    def _check_token_limits(self, context: RAGContext) -> None:
        """Check if context fits within token limits."""
        # Account for system prompt and query
        overhead_tokens = 200  # Approximate tokens for prompts
        available_tokens = self.max_context_tokens - overhead_tokens
        
        if context.total_tokens > available_tokens:
            raise ContextWindowExceededError(
                context_length=context.total_tokens,
                max_length=available_tokens,
            )

    async def _generate_llm_answer(
        self,
        context: RAGContext,
        system_prompt: str,
    ) -> str:
        """Generate answer using LLM."""
        self._ensure_llm_client()
        
        # Format context for LLM
        context_text = context.get_context_text()
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {context.query}\n\nAnswer:",
            },
        ]
        
        try:
            # Generate completion
            response = await self._llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_answer_tokens,
                temperature=self.temperature,
                n=1,
            )
            
            answer = response.choices[0].message.content
            return answer.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise GenerationError(f"LLM generation failed: {str(e)}")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for RAG."""
        return """You are a helpful assistant that answers questions based on the provided context.

Instructions:
1. Answer the question using ONLY the information provided in the context
2. Be concise and direct
3. If the context doesn't contain enough information, say so
4. Reference specific sources using [Source N] format
5. Maintain factual accuracy - do not add information not present in context
6. Use clear, professional language

Important: Your answer must be grounded in the provided context only."""

    def _process_citations(
        self,
        answer: str,
        chunks: List[Tuple[PDFChunk, float]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process citations in the answer.

        Args:
            answer: Generated answer with [Source N] citations
            chunks: Original chunks with scores

        Returns:
            Tuple of (formatted answer, citation list)
        """
        citations = []
        citation_map = {}
        
        # Find all [Source N] patterns
        citation_pattern = r'\[Source (\d+)\]'
        matches = re.finditer(citation_pattern, answer)
        
        for match in matches:
            source_num = int(match.group(1))
            
            # Get corresponding chunk (1-indexed in text)
            if 1 <= source_num <= len(chunks):
                chunk, score = chunks[source_num - 1]
                
                # Create citation entry
                citation_id = f"cite_{len(citations) + 1}"
                citation = {
                    "id": citation_id,
                    "source_number": source_num,
                    "pdf_id": chunk.pdf_id,
                    "chunk_id": chunk.id,
                    "pages": chunk.page_numbers,
                    "score": score,
                    "excerpt": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    "metadata": {
                        "filename": chunk.metadata.get("filename", "Unknown"),
                        "section": chunk.section_title,
                        "chapter": chunk.chapter_title,
                    },
                }
                
                # Check if we already have this citation
                existing = False
                for existing_cite in citations:
                    if existing_cite["chunk_id"] == chunk.id:
                        citation_map[source_num] = existing_cite["id"]
                        existing = True
                        break
                
                if not existing:
                    citations.append(citation)
                    citation_map[source_num] = citation_id
        
        # Replace [Source N] with [N] for cleaner output
        formatted_answer = re.sub(
            citation_pattern,
            lambda m: f"[{m.group(1)}]",
            answer,
        )
        
        return formatted_answer, citations

    def _calculate_confidence(self, chunks: List[Tuple[PDFChunk, float]]) -> float:
        """
        Calculate confidence score for the answer.

        Args:
            chunks: Retrieved chunks with scores

        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Factors for confidence calculation
        scores = [score for _, score in chunks]
        
        # Average score of top chunks
        top_scores = scores[:3] if len(scores) >= 3 else scores
        avg_score = sum(top_scores) / len(top_scores)
        
        # Score distribution (prefer concentrated high scores)
        if len(scores) > 1:
            score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            distribution_factor = 1.0 - min(score_variance, 0.5)
        else:
            distribution_factor = 1.0
        
        # Number of chunks (more context = higher confidence)
        chunk_count_factor = min(len(chunks) / 5.0, 1.0)
        
        # Combine factors
        confidence = (
            avg_score * 0.6 +
            distribution_factor * 0.2 +
            chunk_count_factor * 0.2
        )
        
        return min(max(confidence, 0.0), 1.0)

    async def generate_summary(
        self,
        chunks: List[PDFChunk],
        max_length: int = 500,
    ) -> str:
        """
        Generate a summary of multiple chunks.

        Args:
            chunks: List of chunks to summarize
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        # Combine chunk content
        combined_text = "\n\n".join(chunk.content for chunk in chunks)
        
        # Create summary prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates concise summaries.",
            },
            {
                "role": "user",
                "content": f"Please provide a concise summary of the following text in no more than {max_length} characters:\n\n{combined_text}",
            },
        ]
        
        try:
            self._ensure_llm_client()
            
            response = await self._llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_length // 4,  # Rough token estimate
                temperature=0.3,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed."

    async def check_answer_quality(
        self,
        answer: GeneratedAnswer,
    ) -> Dict[str, Any]:
        """
        Check the quality of a generated answer.

        Args:
            answer: Generated answer to check

        Returns:
            Quality metrics dictionary
        """
        quality_metrics = {
            "has_citations": bool(answer.citations),
            "citation_count": len(answer.citations),
            "answer_length": len(answer.answer),
            "confidence_score": answer.confidence_score,
            "processing_time": answer.processing_time,
        }
        
        # Check if answer addresses the query
        # Simple heuristic: check for question keywords in answer
        query_terms = set(answer.query.lower().split())
        answer_terms = set(answer.answer.lower().split())
        term_overlap = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
        
        quality_metrics["addresses_query"] = term_overlap > 0.3
        
        # Check for common quality issues
        quality_metrics["too_short"] = len(answer.answer) < 50
        quality_metrics["too_long"] = len(answer.answer) > 2000
        quality_metrics["no_specifics"] = not any(
            word in answer.answer.lower()
            for word in ["specifically", "for example", "such as", "including"]
        )
        
        # Overall quality score
        quality_score = (
            0.3 * quality_metrics["confidence_score"] +
            0.2 * (1.0 if quality_metrics["has_citations"] else 0.0) +
            0.2 * (1.0 if quality_metrics["addresses_query"] else 0.0) +
            0.2 * (1.0 if not quality_metrics["too_short"] else 0.0) +
            0.1 * (1.0 if not quality_metrics["no_specifics"] else 0.0)
        )
        
        quality_metrics["overall_quality"] = quality_score
        
        return quality_metrics


def create_answer_generator() -> AnswerGenerator:
    """Create an answer generator with default settings."""
    settings = get_settings()
    return AnswerGenerator(
        model_name=settings.question_gen_model,
        include_confidence=True,
    )