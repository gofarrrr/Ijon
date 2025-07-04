"""
Document profiler for analyzing PDF structure and determining document type.

This profiler examines various document characteristics to classify the document
and recommend an appropriate extraction strategy.
"""

import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import statistics

from extraction.models import DocumentProfile, DocumentType, ConfidenceLevel
from extraction.pdf_processor import PDFProcessor, PDFChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentProfiler:
    """Analyzes documents to determine type and characteristics."""
    
    def __init__(self):
        """Initialize the document profiler."""
        self.pdf_processor = PDFProcessor()
        
        # Keywords and patterns for document classification
        self.type_indicators = {
            DocumentType.ACADEMIC: {
                "keywords": ["abstract", "introduction", "methodology", "results", 
                           "conclusion", "references", "doi:", "et al.", "figure", 
                           "table", "hypothesis", "literature review", "citation"],
                "patterns": [
                    r'\[\d+\]',  # Citation brackets [1]
                    r'\(\d{4}\)',  # Year in parentheses (2024)
                    r'doi:\s*\S+',  # DOI pattern
                    r'[A-Z][a-z]+,\s*[A-Z]\.',  # Author name format
                ]
            },
            DocumentType.TECHNICAL: {
                "keywords": ["installation", "configuration", "api", "function", 
                           "parameter", "syntax", "example", "usage", "command",
                           "error", "troubleshooting", "requirements", "setup"],
                "patterns": [
                    r'```[\s\S]*?```',  # Code blocks
                    r'`[^`]+`',  # Inline code
                    r'\$\s*\w+',  # Command line
                    r'def\s+\w+\(',  # Function definitions
                    r'class\s+\w+[:\(]',  # Class definitions
                ]
            },
            DocumentType.BUSINESS: {
                "keywords": ["strategy", "market", "revenue", "customer", "growth",
                           "competitive", "analysis", "forecast", "budget", "roi",
                           "stakeholder", "objective", "kpi", "quarterly"],
                "patterns": [
                    r'\$[\d,]+\.?\d*[MBK]?',  # Money amounts
                    r'\d+\.?\d*%',  # Percentages
                    r'Q[1-4]\s*\d{4}',  # Quarter notation
                    r'FY\s*\d{4}',  # Fiscal year
                ]
            },
            DocumentType.TUTORIAL: {
                "keywords": ["step", "tutorial", "how to", "guide", "learn",
                           "exercise", "example", "practice", "lesson", "tip",
                           "beginner", "advanced", "walkthrough"],
                "patterns": [
                    r'Step\s+\d+:?',  # Step numbering
                    r'^\d+\.\s+',  # Numbered lists
                    r'•\s+',  # Bullet points
                    r'Note:',  # Notes
                    r'Tip:',  # Tips
                ]
            },
            DocumentType.NARRATIVE: {
                "keywords": ["chapter", "story", "character", "narrative", "plot",
                           "scene", "dialogue", "description", "novel", "memoir"],
                "patterns": [
                    r'Chapter\s+\d+',  # Chapter markers
                    r'"[^"]{10,}"',  # Dialogue
                    r'—',  # Em dash (common in narratives)
                ]
            },
            DocumentType.LEGAL: {
                "keywords": ["whereas", "hereby", "thereof", "pursuant", "clause",
                           "section", "article", "agreement", "contract", "liability"],
                "patterns": [
                    r'§\s*\d+',  # Section symbol
                    r'Article\s+[IVX]+',  # Roman numerals
                    r'\d+\.\d+\.\d+',  # Legal numbering
                ]
            },
            DocumentType.MEDICAL: {
                "keywords": ["patient", "diagnosis", "treatment", "symptom", "clinical",
                           "medical", "disease", "therapy", "dosage", "trial"],
                "patterns": [
                    r'mg/kg',  # Dosage
                    r'p\s*[<>]\s*0\.\d+',  # P-values
                    r'n\s*=\s*\d+',  # Sample size
                ]
            }
        }
    
    async def profile_document(self, pdf_path: str) -> DocumentProfile:
        """
        Analyze a PDF document and create a comprehensive profile.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            DocumentProfile with classification and characteristics
        """
        logger.info(f"Profiling document: {pdf_path}")
        
        try:
            # Extract text chunks for analysis
            chunks = await self.pdf_processor.process_pdf(pdf_path)
            
            if not chunks:
                logger.warning(f"No text extracted from {pdf_path}")
                return self._create_unknown_profile(pdf_path)
            
            # Analyze document characteristics
            structure_score = self._analyze_structure(chunks)
            doc_type, type_confidence = self._classify_document_type(chunks)
            has_tables, has_figures, has_citations = self._detect_special_elements(chunks)
            language = self._detect_language(chunks)
            ocr_quality = self._assess_text_quality(chunks)
            
            # Determine recommended strategy
            strategy = self._recommend_strategy(doc_type, structure_score)
            
            profile = DocumentProfile(
                document_id=Path(pdf_path).stem,
                document_type=doc_type,
                structure_score=structure_score,
                ocr_quality=ocr_quality,
                language=language,
                page_count=max(chunk.page_num for chunk in chunks),
                has_tables=has_tables,
                has_figures=has_figures,
                has_citations=has_citations,
                metadata={
                    "chunk_count": len(chunks),
                    "avg_chunk_size": statistics.mean(len(c.content) for c in chunks),
                    "type_confidence": type_confidence,
                    "filename": Path(pdf_path).name
                },
                recommended_strategy=strategy
            )
            
            logger.info(f"Document profile complete: {doc_type.value} (confidence: {type_confidence:.2f})")
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling document: {str(e)}")
            return self._create_unknown_profile(pdf_path)
    
    def _analyze_structure(self, chunks: List[PDFChunk]) -> ConfidenceLevel:
        """Analyze how well-structured the document is."""
        indicators = {
            "consistent_formatting": 0,
            "clear_sections": 0,
            "logical_flow": 0,
            "clean_text": 0
        }
        
        # Check for consistent formatting
        chunk_lengths = [len(c.content) for c in chunks]
        if chunk_lengths:
            cv = statistics.stdev(chunk_lengths) / statistics.mean(chunk_lengths) if len(chunk_lengths) > 1 else 0
            indicators["consistent_formatting"] = 1.0 if cv < 0.5 else 0.5
        
        # Check for clear sections (headings, etc.)
        section_pattern = r'^[A-Z][^.!?]*$|^\d+\.\s+[A-Z]|^Chapter\s+\d+|^Section\s+\d+'
        section_count = sum(1 for c in chunks if re.search(section_pattern, c.content, re.MULTILINE))
        indicators["clear_sections"] = min(1.0, section_count / (len(chunks) * 0.1))
        
        # Check for logical flow (paragraphs, sentences)
        avg_sentences = statistics.mean(
            len(re.findall(r'[.!?]+', c.content)) for c in chunks
        )
        indicators["logical_flow"] = min(1.0, avg_sentences / 10)
        
        # Check text cleanliness
        clean_ratio = statistics.mean(
            len(re.findall(r'\w+', c.content)) / max(1, len(c.content.split()))
            for c in chunks[:5]  # Sample first 5 chunks
        )
        indicators["clean_text"] = clean_ratio
        
        # Calculate overall structure score
        structure_score = statistics.mean(indicators.values())
        return ConfidenceLevel(structure_score)
    
    def _classify_document_type(self, chunks: List[PDFChunk]) -> Tuple[DocumentType, float]:
        """Classify the document type based on content analysis."""
        # Combine first few chunks for classification
        sample_text = " ".join(c.content for c in chunks[:5]).lower()
        
        type_scores = {}
        
        for doc_type, indicators in self.type_indicators.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for kw in indicators["keywords"] if kw in sample_text)
            keyword_score = min(1.0, keyword_matches / len(indicators["keywords"]))
            
            # Check patterns
            pattern_matches = sum(
                1 for pattern in indicators["patterns"] 
                if re.search(pattern, sample_text, re.IGNORECASE)
            )
            pattern_score = min(1.0, pattern_matches / len(indicators["patterns"])) if indicators["patterns"] else 0
            
            # Combined score (weighted average)
            score = keyword_score * 0.7 + pattern_score * 0.3
            type_scores[doc_type] = score
        
        # Get the best match
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            confidence = type_scores[best_type]
            
            # If confidence is too low, mark as unknown
            if confidence < 0.3:
                return DocumentType.UNKNOWN, confidence
            
            return best_type, confidence
        
        return DocumentType.UNKNOWN, 0.0
    
    def _detect_special_elements(self, chunks: List[PDFChunk]) -> Tuple[bool, bool, bool]:
        """Detect presence of tables, figures, and citations."""
        combined_text = " ".join(c.content for c in chunks).lower()
        
        # Detect tables
        table_indicators = ["table", "column", "row", "|"]
        has_tables = any(indicator in combined_text for indicator in table_indicators)
        
        # Detect figures
        figure_indicators = ["figure", "fig.", "diagram", "chart", "graph", "image"]
        has_figures = any(indicator in combined_text for indicator in figure_indicators)
        
        # Detect citations
        citation_patterns = [r'\[\d+\]', r'\(\d{4}\)', r'et al\.', r'doi:']
        has_citations = any(re.search(pattern, combined_text) for pattern in citation_patterns)
        
        return has_tables, has_figures, has_citations
    
    def _detect_language(self, chunks: List[PDFChunk]) -> str:
        """Detect the primary language of the document."""
        # Simple English detection for now
        # Could be extended with langdetect library
        sample_text = " ".join(c.content for c in chunks[:3])
        
        english_words = ["the", "is", "and", "of", "to", "in", "a", "that"]
        word_list = sample_text.lower().split()
        english_count = sum(1 for word in word_list[:100] if word in english_words)
        
        return "en" if english_count > 5 else "unknown"
    
    def _assess_text_quality(self, chunks: List[PDFChunk]) -> ConfidenceLevel:
        """Assess the quality of extracted text (potential OCR issues)."""
        quality_indicators = []
        
        for chunk in chunks[:5]:  # Sample first 5 chunks
            # Check for garbled text patterns
            garbled_ratio = len(re.findall(r'[^\w\s.,!?;:\'"()-]', chunk.content)) / max(1, len(chunk.content))
            
            # Check for reasonable word lengths
            words = chunk.content.split()
            if words:
                avg_word_length = statistics.mean(len(w) for w in words)
                reasonable_length = 1.0 if 3 <= avg_word_length <= 10 else 0.5
            else:
                reasonable_length = 0.0
            
            # Check for proper spacing
            multiple_spaces = len(re.findall(r'\s{3,}', chunk.content))
            spacing_quality = 1.0 if multiple_spaces < 5 else 0.5
            
            # Combined quality score for this chunk
            chunk_quality = (1 - garbled_ratio) * 0.5 + reasonable_length * 0.3 + spacing_quality * 0.2
            quality_indicators.append(chunk_quality)
        
        return ConfidenceLevel(statistics.mean(quality_indicators) if quality_indicators else 0.5)
    
    def _recommend_strategy(self, doc_type: DocumentType, structure_score: float) -> str:
        """Recommend an extraction strategy based on document characteristics."""
        if structure_score < 0.5:
            # Poor structure - use baseline with extra validation
            return "baseline_validated"
        
        # Map document types to strategies
        strategy_map = {
            DocumentType.ACADEMIC: "academic",
            DocumentType.TECHNICAL: "technical",
            DocumentType.BUSINESS: "narrative",  # Business often has narrative elements
            DocumentType.TUTORIAL: "technical",  # Similar extraction needs
            DocumentType.NARRATIVE: "narrative",
            DocumentType.HISTORICAL: "narrative",
            DocumentType.LEGAL: "technical",  # Structured like technical docs
            DocumentType.MEDICAL: "academic",  # Similar to academic papers
            DocumentType.UNKNOWN: "baseline"
        }
        
        return strategy_map.get(doc_type, "baseline")
    
    def _create_unknown_profile(self, pdf_path: str) -> DocumentProfile:
        """Create a profile for documents that couldn't be analyzed."""
        return DocumentProfile(
            document_id=Path(pdf_path).stem,
            document_type=DocumentType.UNKNOWN,
            structure_score=0.5,
            ocr_quality=0.5,
            language="unknown",
            page_count=0,
            metadata={"error": "Could not analyze document", "filename": Path(pdf_path).name},
            recommended_strategy="baseline"
        )