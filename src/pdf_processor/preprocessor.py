"""
Text preprocessing and cleaning for PDF content.

This module handles text normalization, cleaning, and preparation
for chunking and embedding generation.
"""

import re
import unicodedata
from typing import List, Optional, Set

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Clean and normalize text extracted from PDFs."""

    def __init__(
        self,
        remove_headers: bool = True,
        remove_footers: bool = True,
        remove_page_numbers: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False,
        lowercase: bool = False,
    ) -> None:
        """
        Initialize the text preprocessor.

        Args:
            remove_headers: Remove common header patterns
            remove_footers: Remove common footer patterns
            remove_page_numbers: Remove page number patterns
            normalize_whitespace: Normalize whitespace and newlines
            remove_special_chars: Remove special characters (keep alphanumeric only)
            lowercase: Convert text to lowercase
        """
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.remove_page_numbers = remove_page_numbers
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        
        # Common header/footer patterns
        self.header_patterns = [
            r"^\s*\d+\s*$",  # Just page numbers
            r"^\s*Page\s+\d+\s*$",  # "Page N"
            r"^\s*\d+\s*of\s*\d+\s*$",  # "N of M"
            r"^\s*Chapter\s+\d+.*$",  # Chapter headers
            r"^\s*Section\s+\d+.*$",  # Section headers
        ]
        
        self.footer_patterns = [
            r"^\s*\d+\s*$",  # Just page numbers
            r"^\s*Page\s+\d+\s*$",  # "Page N"
            r"^\s*\d+\s*of\s*\d+\s*$",  # "N of M"
            r"^\s*©.*$",  # Copyright lines
            r"^\s*Copyright.*$",  # Copyright lines
        ]

    def preprocess(self, text: str, page_number: Optional[int] = None) -> str:
        """
        Preprocess text with all configured cleaning steps.

        Args:
            text: Raw text to preprocess
            page_number: Optional page number for context

        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 2: Remove headers/footers if configured
        if self.remove_headers or self.remove_footers:
            text = self._remove_headers_footers(text)
        
        # Step 3: Remove page numbers
        if self.remove_page_numbers and page_number:
            text = self._remove_page_numbers(text, page_number)
        
        # Step 4: Clean special characters
        text = self._clean_special_characters(text)
        
        # Step 5: Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        # Step 6: Remove special chars if configured
        if self.remove_special_chars:
            text = self._remove_special_chars(text)
        
        # Step 7: Lowercase if configured
        if self.lowercase:
            text = text.lower()
        
        # Step 8: Final cleanup
        text = text.strip()
        
        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of texts to preprocess

        Returns:
            List of cleaned texts
        """
        return [self.preprocess(text, idx + 1) for idx, text in enumerate(texts)]

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFKD form
        text = unicodedata.normalize("NFKD", text)
        
        # Replace common Unicode characters with ASCII equivalents
        replacements = {
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
            "–": "-",
            "—": "-",
            "…": "...",
            "•": "*",
            "·": "*",
            "°": " degrees",
            "±": "+/-",
            "×": "x",
            "÷": "/",
            "≈": "~",
            "≤": "<=",
            "≥": ">=",
            "≠": "!=",
            "∞": "infinity",
            "π": "pi",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            "$": "USD",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header and footer patterns."""
        lines = text.split("\n")
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Check if line matches header patterns (typically in first 3 lines)
            if self.remove_headers and i < 3:
                if any(re.match(pattern, line, re.IGNORECASE) for pattern in self.header_patterns):
                    continue
            
            # Check if line matches footer patterns (typically in last 3 lines)
            if self.remove_footers and i >= len(lines) - 3:
                if any(re.match(pattern, line, re.IGNORECASE) for pattern in self.footer_patterns):
                    continue
            
            cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)

    def _remove_page_numbers(self, text: str, page_number: int) -> str:
        """Remove page numbers from text."""
        # Common page number patterns
        patterns = [
            rf"\b{page_number}\b",  # Just the number
            rf"Page\s+{page_number}\b",  # "Page N"
            rf"\b{page_number}\s+of\s+\d+",  # "N of M"
            rf"-\s*{page_number}\s*-",  # "- N -"
            rf"\[{page_number}\]",  # "[N]"
            rf"\({page_number}\)",  # "(N)"
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        return text

    def _clean_special_characters(self, text: str) -> str:
        """Clean special characters while preserving readability."""
        # Remove control characters except newlines and tabs
        text = "".join(
            char for char in text 
            if char == "\n" or char == "\t" or not unicodedata.category(char).startswith("C")
        )
        
        # Fix common OCR errors
        ocr_fixes = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "Ĳ": "IJ",
            "ĳ": "ij",
            "Œ": "OE",
            "œ": "oe",
            "Æ": "AE",
            "æ": "ae",
        }
        
        for old, new in ocr_fixes.items():
            text = text.replace(old, new)
        
        # Remove multiple consecutive punctuation marks
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        
        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.,;:!?])(?=[A-Za-z])", r"\1 ", text)
        
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and newlines."""
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Replace tabs with spaces
        text = text.replace("\t", " ")
        
        # Remove spaces at the beginning and end of lines
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        text = "\n".join(lines)
        
        # Remove empty lines at the beginning and end
        text = text.strip()
        
        return text

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and basic punctuation."""
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s.,;:!?'\"-]", " ", text)
        
        # Clean up multiple spaces created by removal
        text = re.sub(r" +", " ", text)
        
        return text

    def extract_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Extract sections with titles from text.

        Args:
            text: Preprocessed text

        Returns:
            List of (section_title, section_content) tuples
        """
        # Common section patterns
        section_patterns = [
            r"^#+\s+(.+)$",  # Markdown headers
            r"^Chapter\s+\d+[:\s]+(.+)$",  # Chapter N: Title
            r"^Section\s+\d+[:\s]+(.+)$",  # Section N: Title
            r"^Part\s+\d+[:\s]+(.+)$",  # Part N: Title
            r"^(\d+\.)+\s+(.+)$",  # Numbered sections (1.2.3 Title)
            r"^([A-Z][A-Z\s]+)$",  # ALL CAPS TITLES
        ]
        
        sections = []
        current_section = None
        current_content = []
        
        lines = text.split("\n")
        
        for line in lines:
            # Check if line is a section header
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    # Save previous section
                    if current_section:
                        sections.append((current_section, "\n".join(current_content).strip()))
                    
                    # Start new section
                    current_section = match.group(1) if match.lastindex else line.strip()
                    current_content = []
                    is_header = True
                    break
            
            if not is_header and line.strip():
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections.append((current_section, "\n".join(current_content).strip()))
        
        # If no sections found, return the whole text as one section
        if not sections:
            sections.append(("Document", text))
        
        return sections


def create_text_preprocessor() -> TextPreprocessor:
    """Create a text preprocessor with default settings."""
    return TextPreprocessor(
        remove_headers=True,
        remove_footers=True,
        remove_page_numbers=True,
        normalize_whitespace=True,
        remove_special_chars=False,  # Keep special chars by default
        lowercase=False,  # Keep original case by default
    )