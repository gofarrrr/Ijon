"""
Simple PDF processor for extracting text from PDFs.

This is a minimal implementation for the baseline evaluation.
"""

import PyPDF2
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class PDFChunk:
    """Represents a chunk of text from a PDF."""
    content: str
    page_num: int
    chunk_index: int
    metadata: Dict[str, Any] = None


class PDFProcessor:
    """Simple PDF text extraction and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def process_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """
        Extract text from PDF and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of PDFChunk objects
        """
        try:
            # Try PyPDF2 first
            chunks = self._extract_with_pypdf2(pdf_path)
            if chunks:
                return chunks
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # If PyPDF2 fails, try pdfplumber
        try:
            import pdfplumber
            chunks = self._extract_with_pdfplumber(pdf_path)
            if chunks:
                return chunks
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
        
        # If both fail, try PyMuPDF
        try:
            import fitz
            chunks = self._extract_with_pymupdf(pdf_path)
            if chunks:
                return chunks
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")
        
        raise ValueError(f"Could not extract text from {pdf_path}")
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[PDFChunk]:
        """Extract text using PyPDF2."""
        chunks = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    # Clean the text
                    text = self._clean_text(text)
                    
                    # Split into chunks
                    page_chunks = self._split_into_chunks(text, page_num + 1)
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[PDFChunk]:
        """Extract text using pdfplumber."""
        import pdfplumber
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if text and text.strip():
                    # Clean the text
                    text = self._clean_text(text)
                    
                    # Split into chunks
                    page_chunks = self._split_into_chunks(text, page_num + 1)
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[PDFChunk]:
        """Extract text using PyMuPDF."""
        import fitz
        chunks = []
        
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text = page.get_text()
            
            if text.strip():
                # Clean the text
                text = self._clean_text(text)
                
                # Split into chunks
                page_chunks = self._split_into_chunks(text, page_num + 1)
                chunks.extend(page_chunks)
        
        pdf_document.close()
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip very short lines (likely page numbers)
            if len(line.strip()) < 5:
                continue
            
            # Skip lines that are just numbers
            if re.match(r'^\d+$', line.strip()):
                continue
            
            cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines).strip()
    
    def _split_into_chunks(self, text: str, page_num: int) -> List[PDFChunk]:
        """Split text into overlapping chunks."""
        chunks = []
        chunk_index = 0
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            chunks.append(PDFChunk(
                content=text,
                page_num=page_num,
                chunk_index=chunk_index,
                metadata={"full_page": True}
            ))
            return chunks
        
        # Split into overlapping chunks
        start = 0
        while start < len(text):
            # Find end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Look for other break points
                    for delimiter in ['\n', '!', '?', ';']:
                        delim_pos = text.rfind(delimiter, start, end)
                        if delim_pos > start:
                            end = delim_pos + 1
                            break
            
            # Create chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(PDFChunk(
                    content=chunk_text,
                    page_num=page_num,
                    chunk_index=chunk_index,
                    metadata={
                        "start_char": start,
                        "end_char": end
                    }
                ))
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks