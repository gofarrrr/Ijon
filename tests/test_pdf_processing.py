"""
Tests for PDF processing reliability and edge cases.
"""

import asyncio
import io
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PyPDF2 import PdfWriter

from src.models import PDFMetadata, PDFPage, ProcessingStatus
from src.pdf_processor.extractor import PDFExtractor
from src.pdf_processor.processor import PDFProcessor
from src.utils.errors import PDFExtractionError, PDFProcessingError


class TestPDFProcessingReliability:
    """Test PDF processing with various edge cases and formats."""
    
    @pytest.fixture
    def pdf_extractor(self):
        """Create PDF extractor instance."""
        return PDFExtractor(enable_ocr=False)
    
    @pytest.fixture
    def pdf_processor(self, mock_vector_db):
        """Create PDF processor instance."""
        return PDFProcessor(vector_db=mock_vector_db)
    
    def create_test_pdf(self, num_pages: int = 5, text_per_page: str = None) -> bytes:
        """Create a test PDF with specified content."""
        pdf_writer = PdfWriter()
        
        for i in range(num_pages):
            page_text = text_per_page or f"This is page {i+1} content."
            # Note: PyPDF2 doesn't directly support adding text,
            # so this is a simplified version
            pdf_writer.add_blank_page(width=612, height=792)
        
        buffer = io.BytesIO()
        pdf_writer.write(buffer)
        return buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_empty_pdf_handling(self, pdf_extractor, temp_dir):
        """Test handling of empty PDFs."""
        # Create empty PDF
        empty_pdf = self.create_test_pdf(num_pages=0)
        pdf_path = temp_dir / "empty.pdf"
        pdf_path.write_bytes(empty_pdf)
        
        # Should handle gracefully
        metadata = await pdf_extractor.extract_metadata(pdf_path)
        assert metadata.total_pages == 0
        assert metadata.processing_status == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_large_pdf_handling(self, pdf_processor, temp_dir):
        """Test handling of large PDFs."""
        # Create a large PDF (100 pages)
        large_text = "Lorem ipsum " * 100  # Substantial text per page
        large_pdf = self.create_test_pdf(num_pages=100, text_per_page=large_text)
        
        pdf_path = temp_dir / "large.pdf"
        pdf_path.write_bytes(large_pdf)
        
        # Mock extraction to simulate large PDF
        mock_pages = [
            PDFPage(
                page_number=i,
                text=large_text,
                images=[],
                tables=[],
                has_ocr=False,
            )
            for i in range(1, 101)
        ]
        
        with patch.object(pdf_processor.extractor, 'extract_pages', return_value=mock_pages):
            result = await pdf_processor.process_pdf(
                pdf_path,
                pdf_id="large-pdf-test",
            )
            
            assert result.metadata.total_pages == 100
            assert len(result.chunks) > 0
            assert result.metadata.processing_status == ProcessingStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_corrupted_pdf_handling(self, pdf_extractor, temp_dir):
        """Test handling of corrupted PDFs."""
        # Create corrupted PDF data
        corrupted_pdf = b"%PDF-1.4\n%%corrupted data here\n%%EOF"
        pdf_path = temp_dir / "corrupted.pdf"
        pdf_path.write_bytes(corrupted_pdf)
        
        # Should raise appropriate error
        with pytest.raises(PDFExtractionError):
            await pdf_extractor.extract_metadata(pdf_path)
    
    @pytest.mark.asyncio
    async def test_scanned_pdf_without_ocr(self, pdf_extractor, temp_dir):
        """Test handling of scanned PDFs when OCR is disabled."""
        # Create a PDF that simulates scanned content (no extractable text)
        scanned_pdf = self.create_test_pdf(num_pages=3, text_per_page="")
        pdf_path = temp_dir / "scanned.pdf"
        pdf_path.write_bytes(scanned_pdf)
        
        # Extract without OCR
        extractor_no_ocr = PDFExtractor(enable_ocr=False)
        pages = await extractor_no_ocr.extract_pages(pdf_path)
        
        # Should extract but with empty text
        assert len(pages) == 3
        for page in pages:
            assert page.text.strip() == ""
            assert not page.has_ocr
    
    @pytest.mark.asyncio
    async def test_pdf_with_special_characters(self, pdf_processor, temp_dir):
        """Test handling of PDFs with special characters and unicode."""
        special_text = "Test with émojis 🚀🎯 and spëcial cháracters: α β γ δ"
        
        mock_pages = [
            PDFPage(
                page_number=1,
                text=special_text,
                images=[],
                tables=[],
                has_ocr=False,
            )
        ]
        
        with patch.object(pdf_processor.extractor, 'extract_pages', return_value=mock_pages):
            result = await pdf_processor.process_pdf(
                temp_dir / "special.pdf",
                pdf_id="special-char-test",
            )
            
            # Check that special characters are preserved
            assert any("émojis" in chunk.content for chunk in result.chunks)
            assert any("α β γ" in chunk.content for chunk in result.chunks)
    
    @pytest.mark.asyncio
    async def test_pdf_memory_efficiency(self, pdf_processor, temp_dir):
        """Test memory efficiency with streaming processing."""
        # Create a moderately large PDF
        large_pdf = self.create_test_pdf(num_pages=50)
        pdf_path = temp_dir / "memory_test.pdf"
        pdf_path.write_bytes(large_pdf)
        
        # Mock to track memory usage patterns
        chunks_processed = []
        
        async def mock_upsert(chunks):
            chunks_processed.append(len(chunks))
        
        pdf_processor.vector_db.upsert_documents = AsyncMock(side_effect=mock_upsert)
        
        with patch.object(pdf_processor.extractor, 'extract_pages') as mock_extract:
            # Simulate page-by-page extraction
            async def page_generator():
                for i in range(50):
                    yield PDFPage(
                        page_number=i+1,
                        text=f"Page {i+1} content " * 100,
                        images=[],
                        tables=[],
                        has_ocr=False,
                    )
            
            mock_extract.return_value = [page async for page in page_generator()]
            
            await pdf_processor.process_pdf(pdf_path, pdf_id="memory-test")
            
            # Verify batch processing occurred
            assert len(chunks_processed) > 1  # Multiple batches
            assert all(batch_size <= 100 for batch_size in chunks_processed)
    
    @pytest.mark.asyncio
    async def test_concurrent_pdf_processing(self, pdf_processor, temp_dir):
        """Test concurrent processing of multiple PDFs."""
        # Create multiple test PDFs
        pdf_files = []
        for i in range(5):
            pdf_data = self.create_test_pdf(num_pages=10)
            pdf_path = temp_dir / f"concurrent_{i}.pdf"
            pdf_path.write_bytes(pdf_data)
            pdf_files.append(pdf_path)
        
        # Process concurrently
        tasks = [
            pdf_processor.process_pdf(pdf_path, pdf_id=f"concurrent-{i}")
            for i, pdf_path in enumerate(pdf_files)
        ]
        
        # Mock extraction for consistent results
        with patch.object(pdf_processor.extractor, 'extract_pages') as mock_extract:
            mock_extract.return_value = [
                PDFPage(
                    page_number=j+1,
                    text=f"Content for page {j+1}",
                    images=[],
                    tables=[],
                    has_ocr=False,
                )
                for j in range(10)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        assert all(not isinstance(r, Exception) for r in results)
        assert len(results) == 5
    
    @pytest.mark.asyncio 
    async def test_pdf_extraction_timeout(self, pdf_processor, temp_dir):
        """Test timeout handling for slow PDF processing."""
        pdf_path = temp_dir / "slow.pdf"
        pdf_path.write_bytes(self.create_test_pdf(num_pages=5))
        
        # Mock slow extraction
        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow processing
            return []
        
        with patch.object(pdf_processor.extractor, 'extract_pages', side_effect=slow_extract):
            with patch.object(pdf_processor.settings, 'processing_timeout', 1):
                with pytest.raises(asyncio.TimeoutError):
                    await pdf_processor.process_pdf(pdf_path, pdf_id="timeout-test")
    
    @pytest.mark.asyncio
    async def test_pdf_with_images_and_tables(self, pdf_processor):
        """Test handling of PDFs with mixed content."""
        # Mock pages with images and tables
        mock_pages = [
            PDFPage(
                page_number=1,
                text="Page with image and table",
                images=[{"type": "image/png", "data": b"fake-image-data"}],
                tables=[{
                    "headers": ["Column 1", "Column 2"],
                    "rows": [["A", "B"], ["C", "D"]],
                }],
                has_ocr=False,
            )
        ]
        
        with patch.object(pdf_processor.extractor, 'extract_pages', return_value=mock_pages):
            result = await pdf_processor.process_pdf(
                Path("fake.pdf"),
                pdf_id="mixed-content-test",
            )
            
            # Verify mixed content is handled
            assert len(result.chunks) > 0
            chunk_content = result.chunks[0].content
            assert "Page with image and table" in chunk_content
            assert "[Image]" in chunk_content or "image" in chunk_content.lower()
            assert "Column 1" in chunk_content  # Table content


@pytest.mark.asyncio
class TestPDFChunkingStrategies:
    """Test different chunking strategies for reliability."""
    
    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for chunking tests."""
        return """
        Introduction: This is the introduction paragraph. It provides context and background.
        
        Section 1: Main Content
        This section contains the main content. It has multiple sentences. Each sentence 
        contributes to the overall meaning. The content flows logically from one point 
        to another.
        
        1.1 Subsection
        This is a subsection with detailed information. It expands on the main topic.
        Additional details are provided here. The explanation is thorough and complete.
        
        Section 2: Supporting Information
        This section provides supporting details. It reinforces the main points.
        Examples and evidence are included. The argumentation is clear.
        
        Conclusion: This wraps up the document. Key points are summarized.
        """
    
    async def test_sentence_boundary_chunking(self, sample_text):
        """Test chunking respects sentence boundaries."""
        from src.pdf_processor.chunking import chunk_text_with_overlap
        
        chunks = chunk_text_with_overlap(
            sample_text,
            chunk_size=200,
            chunk_overlap=50,
        )
        
        # Verify chunks end at sentence boundaries
        for chunk in chunks:
            # Should end with sentence terminator or be the last chunk
            assert (
                chunk.content.rstrip().endswith(('.', '!', '?', ':'))
                or chunk == chunks[-1]
            )
    
    async def test_section_aware_chunking(self, sample_text):
        """Test chunking preserves section structure."""
        from src.pdf_processor.chunking import chunk_text_with_overlap
        
        chunks = chunk_text_with_overlap(
            sample_text,
            chunk_size=300,
            chunk_overlap=50,
            preserve_sections=True,
        )
        
        # Verify section headers are preserved
        section_headers = ["Introduction:", "Section 1:", "Section 2:", "Conclusion:"]
        for header in section_headers:
            assert any(header in chunk.content for chunk in chunks)
    
    async def test_overlap_consistency(self):
        """Test overlap between chunks is consistent."""
        from src.pdf_processor.chunking import chunk_text_with_overlap
        
        text = " ".join([f"Sentence {i}." for i in range(100)])
        chunks = chunk_text_with_overlap(
            text,
            chunk_size=200,
            chunk_overlap=50,
        )
        
        # Verify overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].content[-50:]
            chunk2_start = chunks[i+1].content[:50]
            
            # There should be some overlap
            assert any(
                word in chunk2_start 
                for word in chunk1_end.split() 
                if len(word) > 3
            )


@pytest.mark.asyncio
class TestPDFMetadataExtraction:
    """Test PDF metadata extraction accuracy."""
    
    async def test_metadata_extraction_completeness(self):
        """Test all metadata fields are extracted."""
        extractor = PDFExtractor()
        
        # Mock PDF with complete metadata
        with patch('PyPDF2.PdfReader') as mock_reader:
            mock_pdf = MagicMock()
            mock_pdf.metadata = {
                '/Title': 'Test Document',
                '/Author': 'Test Author',
                '/Subject': 'Test Subject',
                '/Keywords': 'test, pdf, metadata',
                '/CreationDate': "D:20231201120000",
                '/ModDate': "D:20231202120000",
            }
            mock_pdf.pages = [MagicMock() for _ in range(5)]
            mock_reader.return_value = mock_pdf
            
            metadata = await extractor.extract_metadata(Path("test.pdf"))
        
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.subject == "Test Subject"
        assert metadata.keywords == ["test", "pdf", "metadata"]
        assert metadata.total_pages == 5
    
    async def test_malformed_metadata_handling(self):
        """Test handling of malformed metadata."""
        extractor = PDFExtractor()
        
        with patch('PyPDF2.PdfReader') as mock_reader:
            mock_pdf = MagicMock()
            # Malformed metadata
            mock_pdf.metadata = {
                '/Title': b'\x00\x01Invalid bytes',
                '/Author': None,
                '/CreationDate': 'Invalid date format',
            }
            mock_pdf.pages = [MagicMock()]
            mock_reader.return_value = mock_pdf
            
            metadata = await extractor.extract_metadata(Path("test.pdf"))
        
        # Should handle gracefully
        assert metadata.total_pages == 1
        assert metadata.processing_status == ProcessingStatus.COMPLETED
