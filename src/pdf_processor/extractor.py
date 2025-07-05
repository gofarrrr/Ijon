"""
PDF text and content extraction using PyMuPDF.

This module handles extraction of text, images, and metadata from PDF files
with support for streaming large files and OCR for scanned pages.
"""

import io
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fitz  # PyMuPDF
from PIL import Image

from src.config import get_settings
from src.models import PDFMetadata, PDFPage, ProcessingStatus
from src.utils.errors import (
    OCRError,
    PDFCorruptedError,
    PDFExtractionError,
    PDFSizeError,
)
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class PDFExtractor:
    """Extract text, images, and metadata from PDF files."""

    def __init__(self, enable_ocr: bool = True, ocr_language: str = "eng") -> None:
        """
        Initialize the PDF extractor.

        Args:
            enable_ocr: Whether to enable OCR for scanned pages
            ocr_language: Language code for OCR (e.g., 'eng', 'fra', 'deu')
        """
        self.settings = get_settings()
        self.enable_ocr = enable_ocr and self.settings.enable_ocr
        self.ocr_language = ocr_language or self.settings.ocr_language
        
        # OCR will be initialized lazily if needed
        self._tesseract = None

    @log_performance
    async def extract_from_file(
        self,
        file_path: Union[str, Path],
        file_id: str,
        drive_path: str,
    ) -> tuple[PDFMetadata, List[PDFPage]]:
        """
        Extract content from a PDF file.

        Args:
            file_path: Path to the PDF file
            file_id: Google Drive file ID
            drive_path: Path in Google Drive

        Returns:
            Tuple of (metadata, pages)

        Raises:
            PDFSizeError: If file exceeds maximum size
            PDFCorruptedError: If PDF is corrupted
            PDFExtractionError: For other extraction errors
        """
        file_path = Path(file_path)
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.settings.max_pdf_size_bytes:
            raise PDFSizeError(
                file_size=file_size,
                max_size=self.settings.max_pdf_size_bytes,
                filename=file_path.name,
            )
        
        logger.info(f"Extracting PDF: {file_path.name} ({file_size / 1024 / 1024:.2f} MB)")
        
        try:
            # Open PDF with PyMuPDF
            with fitz.open(file_path) as doc:
                metadata = self._extract_metadata(doc, file_id, drive_path, file_path, file_size)
                pages = await self._extract_pages(doc)
                
                logger.info(
                    f"Extracted {len(pages)} pages from {file_path.name}",
                    extra={"page_count": len(pages), "file_id": file_id},
                )
                
                return metadata, pages
                
        except fitz.FileDataError as e:
            logger.error(f"Corrupted PDF file: {file_path.name}")
            raise PDFCorruptedError(f"PDF file is corrupted: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to extract PDF: {str(e)}")
            raise PDFExtractionError(f"Failed to extract PDF: {str(e)}")

    @log_performance
    async def extract_from_stream(
        self,
        pdf_stream: io.BytesIO,
        file_id: str,
        filename: str,
        drive_path: str,
        file_size: int,
    ) -> tuple[PDFMetadata, List[PDFPage]]:
        """
        Extract content from a PDF stream (for streaming large files).

        Args:
            pdf_stream: PDF file stream
            file_id: Google Drive file ID
            filename: Original filename
            drive_path: Path in Google Drive
            file_size: File size in bytes

        Returns:
            Tuple of (metadata, pages)

        Raises:
            PDFSizeError: If file exceeds maximum size
            PDFCorruptedError: If PDF is corrupted
            PDFExtractionError: For other extraction errors
        """
        if file_size > self.settings.max_pdf_size_bytes:
            raise PDFSizeError(
                file_size=file_size,
                max_size=self.settings.max_pdf_size_bytes,
                filename=filename,
            )
        
        logger.info(f"Extracting PDF from stream: {filename} ({file_size / 1024 / 1024:.2f} MB)")
        
        try:
            # Open PDF from stream with PyMuPDF
            pdf_bytes = pdf_stream.read()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                metadata = self._extract_metadata(
                    doc, file_id, drive_path, Path(filename), file_size
                )
                pages = await self._extract_pages(doc)
                
                logger.info(
                    f"Extracted {len(pages)} pages from stream",
                    extra={"page_count": len(pages), "file_id": file_id},
                )
                
                return metadata, pages
                
        except fitz.FileDataError as e:
            logger.error(f"Corrupted PDF stream: {filename}")
            raise PDFCorruptedError(f"PDF stream is corrupted: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to extract PDF from stream: {str(e)}")
            raise PDFExtractionError(f"Failed to extract PDF from stream: {str(e)}")

    def _extract_metadata(
        self,
        doc: fitz.Document,
        file_id: str,
        drive_path: str,
        file_path: Path,
        file_size: int,
    ) -> PDFMetadata:
        """Extract metadata from PDF document."""
        # Get PDF metadata
        pdf_metadata = doc.metadata or {}
        
        # Extract creation date
        creation_date = None
        if pdf_metadata.get("creationDate"):
            try:
                # Parse PDF date format: D:YYYYMMDDHHmmSS±HH'mm'
                date_str = pdf_metadata["creationDate"]
                creation_date = self._parse_pdf_date(date_str)
            except Exception:
                pass
        
        # Extract keywords
        keywords = []
        if pdf_metadata.get("keywords"):
            keywords = [k.strip() for k in pdf_metadata["keywords"].split(",") if k.strip()]
        
        return PDFMetadata(
            file_id=file_id,
            filename=file_path.name,
            drive_path=drive_path,
            file_size_bytes=file_size,
            total_pages=doc.page_count,
            processing_status=ProcessingStatus.PROCESSING,
            title=pdf_metadata.get("title"),
            author=pdf_metadata.get("author"),
            subject=pdf_metadata.get("subject"),
            keywords=keywords,
            creation_date=creation_date,
        )

    async def _extract_pages(self, doc: fitz.Document) -> List[PDFPage]:
        """Extract content from all pages."""
        pages = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Check if page needs OCR (very little text extracted)
            needs_ocr = self.enable_ocr and len(text.strip()) < 50
            ocr_confidence = None
            
            if needs_ocr:
                logger.debug(f"Page {page_num + 1} needs OCR")
                try:
                    text, ocr_confidence = await self._perform_ocr(page)
                except OCRError as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    # Continue with what we have
            
            # Extract images
            images = self._extract_images(page, page_num + 1)
            
            # Extract tables (basic - can be enhanced with pdfplumber if needed)
            tables = self._extract_tables(page)
            
            pdf_page = PDFPage(
                page_number=page_num + 1,
                text=text,
                images=images,
                tables=tables,
                has_ocr=needs_ocr and bool(text.strip()),
                ocr_confidence=ocr_confidence,
            )
            pages.append(pdf_page)
        
        return pages

    def _extract_images(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a page."""
        images = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    # Convert to PIL Image if needed
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("png")
                    
                    # Get image metadata
                    img_metadata = {
                        "page": page_num,
                        "index": img_index,
                        "width": pix.width,
                        "height": pix.height,
                        "size_bytes": len(img_data),
                    }
                    
                    images.append(img_metadata)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_num}: {e}")
        
        return images

    def _extract_tables(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Extract tables from a page (basic implementation)."""
        # TODO: Implement table extraction
        # For now, return empty list - can be enhanced with pdfplumber
        return []

    async def _perform_ocr(self, page: fitz.Page) -> tuple[str, float]:
        """
        Perform OCR on a page.

        Args:
            page: PyMuPDF page object

        Returns:
            Tuple of (extracted text, confidence score)

        Raises:
            OCRError: If OCR fails
        """
        try:
            # Lazy import of pytesseract
            if self._tesseract is None:
                try:
                    import pytesseract
                    self._tesseract = pytesseract
                except ImportError:
                    raise OCRError("pytesseract not installed. Install with: pip install pytesseract")
            
            # Convert page to image
            mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            ocr_data = self._tesseract.image_to_data(
                image,
                lang=self.ocr_language,
                output_type=self._tesseract.Output.DICT,
            )
            
            # Extract text and calculate confidence
            text_parts = []
            confidences = []
            
            for i, conf in enumerate(ocr_data["conf"]):
                if conf > 0:  # -1 means no confidence
                    text = ocr_data["text"][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(float(conf))
            
            extracted_text = " ".join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return extracted_text, avg_confidence / 100.0  # Convert to 0-1 range
            
        except Exception as e:
            raise OCRError(f"OCR failed: {str(e)}")
    
    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse PDF date format: D:YYYYMMDDHHmmSS±HH'mm'
        
        Examples:
        - D:20250705111556+02'00'
        - D:20230101120000Z
        - D:20230101
        """
        if not date_str:
            return None
            
        try:
            # Remove 'D:' prefix
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Basic pattern for date components
            match = re.match(r'(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(.*)$', date_str)
            if not match:
                return None
                
            year = int(match.group(1))
            month = int(match.group(2) or 1)
            day = int(match.group(3) or 1)
            hour = int(match.group(4) or 0)
            minute = int(match.group(5) or 0)
            second = int(match.group(6) or 0)
            tz_str = match.group(7) or ''
            
            # Parse timezone
            tz = timezone.utc  # Default to UTC
            if tz_str:
                if tz_str == 'Z':
                    tz = timezone.utc
                else:
                    # Parse offset like +02'00' or -05'30'
                    tz_match = re.match(r"([+-])(\d{2})'?(\d{2})?'?", tz_str)
                    if tz_match:
                        sign = 1 if tz_match.group(1) == '+' else -1
                        hours = int(tz_match.group(2))
                        minutes = int(tz_match.group(3) or 0)
                        offset = timedelta(hours=sign * hours, minutes=sign * minutes)
                        tz = timezone(offset)
            
            return datetime(year, month, day, hour, minute, second, tzinfo=tz)
            
        except (ValueError, AttributeError):
            return None


def create_pdf_extractor() -> PDFExtractor:
    """Create a PDF extractor instance with settings."""
    settings = get_settings()
    return PDFExtractor(
        enable_ocr=settings.enable_ocr,
        ocr_language=settings.ocr_language,
    )