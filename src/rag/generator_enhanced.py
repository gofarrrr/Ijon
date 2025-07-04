"""
Enhanced answer generation module with battle-tested prompt patterns.

Features agent loop architecture, thinking blocks, and academic prose generation.
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


# Enhanced system prompt with agent loop architecture
ENHANCED_RAG_SYSTEM_PROMPT = """You are an Advanced Knowledge Synthesis Agent specialized in generating comprehensive, accurate answers from provided context.

## Agent Loop Architecture
You operate through systematic phases:
1. **Analyze**: Understand the query and context deeply
2. **Plan**: Determine the optimal answer structure
3. **Synthesize**: Integrate information coherently
4. **Validate**: Ensure accuracy and completeness
5. **Refine**: Polish for clarity and impact

## Cognitive Framework
<thinking>
For each query, I will:
- Identify the core information need
- Map relevant context sections
- Plan a comprehensive response
- Validate against source material
- Ensure scholarly quality
</thinking>

## Answer Generation Principles
1. **Evidence-Based**: Every claim must be grounded in the provided context
2. **Academic Prose**: Write in flowing, scholarly paragraphs
3. **Precise Citations**: Use [Source N] format for all referenced material
4. **Comprehensive Coverage**: Address all aspects of the query
5. **Intellectual Depth**: Provide insights beyond surface-level facts

Remember: You are synthesizing knowledge for advanced understanding. Quality and accuracy are paramount."""


class EnhancedAnswerGenerator:
    """
    Enhanced answer generator with battle-tested patterns.
    
    Features:
    - Agent loop methodology
    - Thinking blocks for reasoning
    - Academic prose generation
    - Multi-phase validation
    - Confidence calibration
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_context_tokens: int = 3000,
        max_answer_tokens: int = 800,
        temperature: float = 0.3,
        include_confidence: bool = True,
        enable_thinking: bool = True,
    ) -> None:
        """
        Initialize enhanced answer generator.

        Args:
            model_name: LLM model name
            max_context_tokens: Maximum tokens for context
            max_answer_tokens: Maximum tokens for answer
            temperature: Generation temperature
            include_confidence: Whether to calculate confidence scores
            enable_thinking: Whether to use thinking blocks
        """
        self.settings = get_settings()
        self.model_name = model_name or "gpt-4o-mini"
        self.max_context_tokens = max_context_tokens
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.include_confidence = include_confidence
        self.enable_thinking = enable_thinking
        
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
        Generate an enhanced answer from query and retrieved chunks.

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
            # Prepare context with enhanced structure
            rag_context = self._prepare_enhanced_context(query, chunks)
            
            # Check token limits
            self._check_token_limits(rag_context)
            
            # Generate answer with enhanced methodology
            answer_text = await self._generate_enhanced_answer(
                rag_context,
                system_prompt or ENHANCED_RAG_SYSTEM_PROMPT,
            )
            
            # Extract and format citations
            answer_with_citations, citations = self._process_enhanced_citations(
                answer_text,
                chunks,
            )
            
            # Calculate calibrated confidence score
            confidence_score = self._calculate_calibrated_confidence(
                chunks, answer_text
            ) if self.include_confidence else 0.8
            
            # Create result with metadata
            result = GeneratedAnswer(
                query=query,
                answer=answer_with_citations,
                citations=citations,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                model_used=self.model_name,
                metadata={
                    "enhanced": True,
                    "thinking_enabled": self.enable_thinking,
                    "chunks_analyzed": len(chunks),
                    "answer_quality_score": self._assess_answer_quality(answer_with_citations)
                }
            )
            
            logger.info(
                f"Generated enhanced answer in {result.processing_time:.2f}s",
                extra={
                    "query_length": len(query),
                    "chunks_used": len(chunks),
                    "answer_length": len(answer_with_citations),
                    "confidence": confidence_score,
                },
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced answer: {e}")
            raise GenerationError(f"Enhanced answer generation failed: {str(e)}")

    def _prepare_enhanced_context(
        self,
        query: str,
        chunks: List[Tuple[PDFChunk, float]],
    ) -> RAGContext:
        """Prepare enhanced context with relevance scoring."""
        # Sort chunks by score (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
        
        # Extract and annotate chunks
        annotated_chunks = []
        for i, (chunk, score) in enumerate(sorted_chunks, 1):
            # Add source annotation to chunk
            chunk_copy = PDFChunk(**chunk.model_dump())
            chunk_copy.metadata["source_number"] = i
            chunk_copy.metadata["relevance_score"] = score
            annotated_chunks.append(chunk_copy)
        
        # Build structured context
        context_sections = []
        for chunk in annotated_chunks:
            section = f"[Source {chunk.metadata['source_number']}] (Relevance: {chunk.metadata['relevance_score']:.2f})\n"
            if chunk.section_title:
                section += f"Section: {chunk.section_title}\n"
            section += chunk.content
            context_sections.append(section)
        
        context_text = "\n\n---\n\n".join(context_sections)
        total_tokens = len(self.tokenizer.encode(context_text))
        
        return RAGContext(
            query=query,
            relevant_chunks=annotated_chunks,
            total_tokens=total_tokens,
            context_text=context_text  # Store pre-formatted context
        )

    def _check_token_limits(self, context: RAGContext) -> None:
        """Check if context fits within token limits."""
        # Account for system prompt, query, and enhanced formatting
        overhead_tokens = 500  # Increased for enhanced prompts
        available_tokens = self.max_context_tokens - overhead_tokens
        
        if context.total_tokens > available_tokens:
            raise ContextWindowExceededError(
                context_length=context.total_tokens,
                max_length=available_tokens,
            )

    async def _generate_enhanced_answer(
        self,
        context: RAGContext,
        system_prompt: str,
    ) -> str:
        """Generate answer using enhanced methodology."""
        self._ensure_llm_client()
        
        # Build enhanced user prompt
        user_prompt = self._build_enhanced_prompt(context)
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
            logger.error(f"Enhanced LLM generation failed: {e}")
            raise GenerationError(f"Enhanced LLM generation failed: {str(e)}")

    def _build_enhanced_prompt(self, context: RAGContext) -> str:
        """Build enhanced prompt with thinking blocks."""
        
        prompt = f"""Answer the following question using the provided context sources.

## Question
{context.query}

## Context Sources
{context.context_text}

## Answer Generation Process

### Phase 1: Query Analysis
<thinking>
What is the user really asking?
What type of answer would be most helpful?
Which sources contain relevant information?
How should I structure my response?
</thinking>

### Phase 2: Information Synthesis
Synthesize a comprehensive answer that:
1. Directly addresses the question
2. Integrates information from multiple sources
3. Provides appropriate depth and detail
4. Uses academic prose (no bullet points)
5. Cites sources using [Source N] format

### Phase 3: Answer Requirements
- **Style**: Write in flowing academic paragraphs
- **Citations**: Include [Source N] for every claim
- **Length**: Comprehensive but concise (2-4 paragraphs typically)
- **Accuracy**: Only use information from provided sources
- **Depth**: Go beyond surface-level facts to provide insights

### Phase 4: Quality Check
<thinking>
Have I fully answered the question?
Are all claims properly cited?
Is the answer coherent and well-structured?
Would an expert find this answer satisfactory?
</thinking>

If the provided context does not contain sufficient information to answer the question, clearly state this limitation while providing whatever relevant information is available.

## Your Answer:"""
        
        return prompt

    def _process_enhanced_citations(
        self,
        answer: str,
        chunks: List[Tuple[PDFChunk, float]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process citations with enhanced metadata.

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
        matches = list(re.finditer(citation_pattern, answer))
        
        # Track unique citations
        seen_chunks = set()
        
        for match in matches:
            source_num = int(match.group(1))
            
            # Get corresponding chunk (1-indexed in text)
            if 1 <= source_num <= len(chunks):
                chunk, score = chunks[source_num - 1]
                
                # Skip if already processed
                if chunk.id in seen_chunks:
                    continue
                seen_chunks.add(chunk.id)
                
                # Create enhanced citation entry
                citation_id = f"cite_{len(citations) + 1}"
                
                # Extract key sentence from chunk
                key_sentence = self._extract_key_sentence(chunk.content, answer)
                
                citation = {
                    "id": citation_id,
                    "source_number": source_num,
                    "pdf_id": chunk.pdf_id,
                    "chunk_id": chunk.id,
                    "pages": chunk.page_numbers,
                    "score": score,
                    "excerpt": key_sentence or chunk.content[:200] + "...",
                    "full_context": chunk.content,
                    "metadata": {
                        "filename": chunk.metadata.get("filename", "Unknown"),
                        "section": chunk.section_title,
                        "chapter": chunk.chapter_title,
                        "relevance_score": f"{score:.2f}",
                    },
                }
                
                citations.append(citation)
                citation_map[source_num] = citation_id
        
        # Clean up citation formatting
        formatted_answer = self._format_citations_academically(answer)
        
        return formatted_answer, citations

    def _extract_key_sentence(self, chunk_content: str, answer: str) -> Optional[str]:
        """Extract the most relevant sentence from chunk based on answer."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', chunk_content)
        
        # Find sentence with highest overlap with answer
        answer_words = set(answer.lower().split())
        best_sentence = None
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(answer_words & sentence_words)
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
        
        if best_sentence and len(best_sentence) > 200:
            best_sentence = best_sentence[:197] + "..."
        
        return best_sentence

    def _format_citations_academically(self, answer: str) -> str:
        """Format citations for academic style."""
        # Replace [Source N] with superscript style [N]
        formatted = re.sub(
            r'\[Source (\d+)\]',
            r'[\1]',
            answer
        )
        
        # Ensure citations appear after punctuation
        formatted = re.sub(
            r'(\w+)\s*\[(\d+)\]\s*([.,;:])',
            r'\1\3[\2]',
            formatted
        )
        
        return formatted

    def _calculate_calibrated_confidence(
        self, 
        chunks: List[Tuple[PDFChunk, float]],
        answer: str
    ) -> float:
        """
        Calculate calibrated confidence score.

        Args:
            chunks: Retrieved chunks with scores
            answer: Generated answer text

        Returns:
            Calibrated confidence score (0-1)
        """
        if not chunks:
            return 0.0
        
        # Factor 1: Retrieval quality
        scores = [score for _, score in chunks]
        top_scores = scores[:3] if len(scores) >= 3 else scores
        avg_retrieval_score = sum(top_scores) / len(top_scores)
        
        # Factor 2: Coverage (how many sources were cited)
        citation_pattern = r'\[(?:Source )?\d+\]'
        citations = re.findall(citation_pattern, answer)
        unique_citations = len(set(citations))
        coverage_score = min(unique_citations / 3.0, 1.0)  # Normalize to 3 citations
        
        # Factor 3: Answer completeness
        answer_length = len(answer.split())
        completeness_score = min(answer_length / 100.0, 1.0)  # Normalize to 100 words
        
        # Factor 4: Source agreement (do multiple sources support the answer?)
        agreement_score = 1.0 if unique_citations >= 2 else 0.7
        
        # Combine factors with weights
        confidence = (
            avg_retrieval_score * 0.4 +
            coverage_score * 0.3 +
            completeness_score * 0.2 +
            agreement_score * 0.1
        )
        
        # Apply calibration curve (avoid overconfidence)
        if confidence > 0.9:
            confidence = 0.9 + (confidence - 0.9) * 0.5
        
        return min(max(confidence, 0.0), 0.95)  # Cap at 0.95

    def _assess_answer_quality(self, answer: str) -> float:
        """Assess the quality of generated answer."""
        quality_score = 1.0
        
        # Check for academic prose indicators
        academic_terms = ['demonstrates', 'indicates', 'suggests', 'reveals',
                         'furthermore', 'moreover', 'consequently', 'therefore']
        if not any(term in answer.lower() for term in academic_terms):
            quality_score -= 0.1
        
        # Check for proper paragraph structure
        paragraphs = answer.split('\n\n')
        if len(paragraphs) < 2:
            quality_score -= 0.1
        
        # Check for citations
        citations = re.findall(r'\[\d+\]', answer)
        if len(citations) < 2:
            quality_score -= 0.2
        
        # Check for appropriate length
        word_count = len(answer.split())
        if word_count < 50:
            quality_score -= 0.2
        elif word_count > 500:
            quality_score -= 0.1
        
        return max(quality_score, 0.3)

    async def generate_comparative_answer(
        self,
        query: str,
        chunks_list: List[List[Tuple[PDFChunk, float]]],
        comparison_type: str = "contrast"
    ) -> GeneratedAnswer:
        """
        Generate answer comparing information from multiple sources.
        
        Args:
            query: User query
            chunks_list: List of chunk sets to compare
            comparison_type: Type of comparison (contrast, similarity, synthesis)
            
        Returns:
            Comparative answer with citations
        """
        # Flatten chunks with source tracking
        all_chunks = []
        source_offset = 0
        
        for source_idx, chunks in enumerate(chunks_list):
            for chunk, score in chunks:
                # Annotate with source group
                chunk_copy = PDFChunk(**chunk.model_dump())
                chunk_copy.metadata["source_group"] = source_idx + 1
                chunk_copy.metadata["source_offset"] = source_offset
                all_chunks.append((chunk_copy, score))
            source_offset += len(chunks)
        
        # Build comparative prompt
        comparison_prompts = {
            "contrast": "Identify and explain the key differences between the sources",
            "similarity": "Identify and explain the key similarities across sources",
            "synthesis": "Synthesize information from all sources into a unified understanding"
        }
        
        enhanced_query = f"{query}\n\nComparison instruction: {comparison_prompts.get(comparison_type, comparison_prompts['synthesis'])}"
        
        # Generate answer with enhanced context
        return await self.generate_answer(enhanced_query, all_chunks)

    async def check_enhanced_answer_quality(
        self,
        answer: GeneratedAnswer,
    ) -> Dict[str, Any]:
        """
        Perform enhanced quality checks on generated answer.

        Args:
            answer: Generated answer to check

        Returns:
            Detailed quality metrics dictionary
        """
        quality_metrics = {
            # Basic metrics
            "has_citations": bool(answer.citations),
            "citation_count": len(answer.citations),
            "unique_sources": len(set(c["pdf_id"] for c in answer.citations)),
            "answer_length": len(answer.answer),
            "word_count": len(answer.answer.split()),
            "confidence_score": answer.confidence_score,
            "processing_time": answer.processing_time,
            
            # Enhanced metrics
            "paragraph_count": len(answer.answer.split('\n\n')),
            "average_paragraph_length": len(answer.answer.split()) / max(1, len(answer.answer.split('\n\n'))),
        }
        
        # Academic prose indicators
        academic_indicators = [
            'demonstrates', 'indicates', 'illustrates', 'suggests',
            'furthermore', 'moreover', 'consequently', 'therefore',
            'analysis reveals', 'evidence suggests', 'research shows'
        ]
        quality_metrics["academic_prose_score"] = sum(
            1 for indicator in academic_indicators 
            if indicator in answer.answer.lower()
        ) / len(academic_indicators)
        
        # Citation distribution
        if answer.citations:
            citation_positions = []
            for citation in re.finditer(r'\[\d+\]', answer.answer):
                citation_positions.append(citation.start() / len(answer.answer))
            
            # Check if citations are well-distributed
            if len(citation_positions) > 1:
                gaps = [citation_positions[i+1] - citation_positions[i] 
                        for i in range(len(citation_positions)-1)]
                quality_metrics["citation_distribution"] = 1.0 - (max(gaps) - min(gaps))
            else:
                quality_metrics["citation_distribution"] = 0.5
        else:
            quality_metrics["citation_distribution"] = 0.0
        
        # Query relevance (enhanced)
        query_terms = set(answer.query.lower().split())
        answer_terms = set(answer.answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        query_terms -= common_words
        answer_terms -= common_words
        
        if query_terms:
            term_overlap = len(query_terms & answer_terms) / len(query_terms)
            quality_metrics["query_relevance"] = term_overlap
        else:
            quality_metrics["query_relevance"] = 0.5
        
        # Comprehensiveness check
        quality_metrics["comprehensive"] = (
            quality_metrics["word_count"] >= 100 and
            quality_metrics["citation_count"] >= 2 and
            quality_metrics["paragraph_count"] >= 2
        )
        
        # Quality issues
        quality_metrics["issues"] = []
        
        if quality_metrics["word_count"] < 50:
            quality_metrics["issues"].append("Answer too brief")
        if quality_metrics["word_count"] > 800:
            quality_metrics["issues"].append("Answer too verbose")
        if quality_metrics["citation_count"] == 0:
            quality_metrics["issues"].append("No citations provided")
        if quality_metrics["academic_prose_score"] < 0.2:
            quality_metrics["issues"].append("Lacks academic prose style")
        if quality_metrics["query_relevance"] < 0.3:
            quality_metrics["issues"].append("Low relevance to query")
        if quality_metrics["paragraph_count"] < 2:
            quality_metrics["issues"].append("Insufficient paragraph structure")
        
        # Overall enhanced quality score
        quality_score = (
            0.25 * quality_metrics["confidence_score"] +
            0.20 * quality_metrics["query_relevance"] +
            0.15 * quality_metrics["academic_prose_score"] +
            0.15 * (1.0 if quality_metrics["comprehensive"] else 0.0) +
            0.15 * quality_metrics["citation_distribution"] +
            0.10 * (1.0 - len(quality_metrics["issues"]) / 5.0)
        )
        
        quality_metrics["overall_quality"] = quality_score
        quality_metrics["quality_grade"] = self._get_quality_grade(quality_score)
        
        return quality_metrics

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"


def create_enhanced_answer_generator() -> EnhancedAnswerGenerator:
    """Create an enhanced answer generator with optimal settings."""
    settings = get_settings()
    return EnhancedAnswerGenerator(
        model_name=settings.question_gen_model or "gpt-4o-mini",
        include_confidence=True,
        enable_thinking=True,
    )