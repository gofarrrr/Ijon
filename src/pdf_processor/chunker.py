"""
Semantic text chunking with sentence boundary awareness.

This module handles intelligent splitting of text into chunks that respect
sentence boundaries and maintain semantic coherence.
"""

import re
from typing import List, Optional, Tuple

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from src.config import get_settings
from src.models import PDFChunk
from src.utils.errors import ChunkingError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class SemanticChunker:
    """
    Split text into semantic chunks respecting sentence boundaries.
    
    This chunker ensures that:
    1. Chunks don't break in the middle of sentences
    2. Chunks have appropriate overlap for context
    3. Chunks are roughly equal in size
    4. Section boundaries are respected when possible
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_sections: bool = True,
        use_nltk: bool = True,
    ) -> None:
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk
            respect_sections: Whether to avoid splitting sections
            use_nltk: Whether to use NLTK for sentence tokenization
        """
        self.settings = get_settings()
        self.chunk_size = chunk_size or self.settings.chunk_size
        self.chunk_overlap = chunk_overlap or self.settings.chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_sections = respect_sections
        self.use_nltk = use_nltk
        
        # Validate parameters
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        
        # Download NLTK data if needed
        if self.use_nltk:
            self._ensure_nltk_data()

    def _ensure_nltk_data(self) -> None:
        """Download required NLTK data if not present."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt", quiet=True)

    @log_performance
    async def chunk_text(
        self,
        text: str,
        pdf_id: str,
        metadata: Optional[dict] = None,
        page_numbers: Optional[List[int]] = None,
        section_title: Optional[str] = None,
        chapter_title: Optional[str] = None,
    ) -> List[PDFChunk]:
        """
        Split text into semantic chunks.

        Args:
            text: Text to chunk
            pdf_id: Source PDF ID
            metadata: Additional metadata for chunks
            page_numbers: Page numbers this text spans
            section_title: Section title if available
            chapter_title: Chapter title if available

        Returns:
            List of PDFChunk objects

        Raises:
            ChunkingError: If chunking fails
        """
        if not text or not text.strip():
            return []
        
        try:
            # Split into sentences
            sentences = self._split_sentences(text)
            
            if not sentences:
                return []
            
            # Group sentences into chunks
            chunks = self._group_sentences_into_chunks(sentences)
            
            # Create PDFChunk objects
            pdf_chunks = []
            for i, (chunk_text, chunk_sentences) in enumerate(chunks):
                chunk = PDFChunk(
                    pdf_id=pdf_id,
                    content=chunk_text,
                    page_numbers=page_numbers or [1],
                    chunk_index=i,
                    metadata=metadata or {},
                    word_count=len(word_tokenize(chunk_text)) if self.use_nltk else len(chunk_text.split()),
                    char_count=len(chunk_text),
                    section_title=section_title,
                    chapter_title=chapter_title,
                )
                pdf_chunks.append(chunk)
            
            logger.debug(
                f"Created {len(pdf_chunks)} chunks from {len(sentences)} sentences",
                extra={"pdf_id": pdf_id, "text_length": len(text)},
            )
            
            return pdf_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise ChunkingError(f"Failed to chunk text: {str(e)}")

    @log_performance
    async def chunk_pages(
        self,
        pages: List[Tuple[int, str]],
        pdf_id: str,
        metadata: Optional[dict] = None,
    ) -> List[PDFChunk]:
        """
        Chunk multiple pages of text.

        Args:
            pages: List of (page_number, text) tuples
            pdf_id: Source PDF ID
            metadata: Additional metadata

        Returns:
            List of PDFChunk objects
        """
        all_chunks = []
        
        # Process pages maintaining page boundaries
        current_text = ""
        current_pages = []
        
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            
            # Check if adding this page would exceed chunk size significantly
            combined_text = current_text + "\n\n" + page_text if current_text else page_text
            
            if len(combined_text) > self.chunk_size * 1.5 and current_text:
                # Chunk current accumulated text
                chunks = await self.chunk_text(
                    current_text,
                    pdf_id=pdf_id,
                    metadata=metadata,
                    page_numbers=current_pages,
                )
                all_chunks.extend(chunks)
                
                # Start new accumulation
                current_text = page_text
                current_pages = [page_num]
            else:
                # Accumulate
                current_text = combined_text
                current_pages.append(page_num)
        
        # Process remaining text
        if current_text:
            chunks = await self.chunk_text(
                current_text,
                pdf_id=pdf_id,
                metadata=metadata,
                page_numbers=current_pages,
            )
            all_chunks.extend(chunks)
        
        # Update chunk indices
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        
        return all_chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.use_nltk:
            # Use NLTK's sophisticated sentence tokenizer
            sentences = sent_tokenize(text)
        else:
            # Fallback to regex-based splitting
            sentences = self._regex_sentence_split(text)
        
        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _regex_sentence_split(self, text: str) -> List[str]:
        """Simple regex-based sentence splitting as fallback."""
        # Split on sentence endings followed by space and capital letter
        sentence_endings = r"[.!?]+[\s]+"
        sentences = re.split(sentence_endings, text)
        
        # Handle edge cases
        cleaned_sentences = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            
            # Add back the sentence ending if it was removed
            if i < len(sentences) - 1:
                # Find the ending that was removed
                match = re.search(sentence_endings, text[text.find(sent) + len(sent):])
                if match:
                    sent += match.group().strip()
            
            cleaned_sentences.append(sent)
        
        return cleaned_sentences

    def _group_sentences_into_chunks(
        self, sentences: List[str]
    ) -> List[Tuple[str, List[str]]]:
        """
        Group sentences into chunks respecting size constraints.

        Returns:
            List of (chunk_text, sentences_in_chunk) tuples
        """
        chunks = []
        current_chunk_sentences = []
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_chunk_size + sentence_size > self.chunk_size and current_chunk_sentences:
                # Create chunk with current sentences
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append((chunk_text, current_chunk_sentences[:]))
                
                # Determine overlap sentences
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.chunk_overlap
                )
                
                # Start new chunk with overlap
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_size = sum(len(s) for s in current_chunk_sentences)
            else:
                # Add sentence to current chunk
                current_chunk_sentences.append(sentence)
                current_chunk_size += sentence_size
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append((chunk_text, current_chunk_sentences))
            elif chunks:
                # Merge with previous chunk if too small
                prev_text, prev_sentences = chunks[-1]
                merged_text = prev_text + " " + chunk_text
                merged_sentences = prev_sentences + current_chunk_sentences
                chunks[-1] = (merged_text, merged_sentences)
        
        return chunks

    def _get_overlap_sentences(
        self, sentences: List[str], target_overlap: int
    ) -> List[str]:
        """
        Get sentences from the end of the chunk for overlap.

        Args:
            sentences: List of sentences in the chunk
            target_overlap: Target overlap size in characters

        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        overlap_size = 0
        
        # Work backwards through sentences
        for sentence in reversed(sentences):
            overlap_size += len(sentence)
            overlap_sentences.insert(0, sentence)
            
            if overlap_size >= target_overlap:
                break
        
        return overlap_sentences

    def chunk_by_sections(
        self,
        sections: List[Tuple[str, str]],
        pdf_id: str,
        metadata: Optional[dict] = None,
    ) -> List[PDFChunk]:
        """
        Chunk text by sections, respecting section boundaries.

        Args:
            sections: List of (section_title, section_content) tuples
            pdf_id: Source PDF ID
            metadata: Additional metadata

        Returns:
            List of PDFChunk objects
        """
        all_chunks = []
        
        for section_title, section_content in sections:
            if not section_content.strip():
                continue
            
            # Chunk section content
            section_chunks = self.chunk_text(
                section_content,
                pdf_id=pdf_id,
                metadata=metadata,
                section_title=section_title,
            )
            
            all_chunks.extend(section_chunks)
        
        # Update chunk indices
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
        
        return all_chunks


def create_semantic_chunker() -> SemanticChunker:
    """Create a semantic chunker with settings."""
    settings = get_settings()
    return SemanticChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_chunk_size=100,
        respect_sections=True,
        use_nltk=True,
    )