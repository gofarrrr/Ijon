"""
Base extraction strategy class.

All document-specific strategies inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from extraction.models import ExtractedKnowledge, DocumentProfile
from extraction.pdf_processor import PDFChunk


class ExtractionStrategy(ABC):
    """Abstract base class for extraction strategies."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the extraction strategy.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for extraction
        """
        self.api_key = openai_api_key
        self.model = model
        self.temperature = 0.3  # Low temperature for consistency
    
    @abstractmethod
    def build_extraction_prompt(self, content: str, profile: DocumentProfile) -> str:
        """
        Build a document-type-specific extraction prompt.
        
        Args:
            content: Text content to extract from
            profile: Document profile with type and characteristics
            
        Returns:
            Customized prompt for the document type
        """
        pass
    
    @abstractmethod
    def post_process_extraction(self, 
                              extraction: ExtractedKnowledge, 
                              profile: DocumentProfile) -> ExtractedKnowledge:
        """
        Post-process the extraction based on document type.
        
        Args:
            extraction: Raw extraction from the model
            profile: Document profile
            
        Returns:
            Enhanced extraction with type-specific improvements
        """
        pass
    
    def get_extraction_parameters(self, profile: DocumentProfile) -> Dict[str, Any]:
        """
        Get model parameters optimized for the document type.
        
        Args:
            profile: Document profile
            
        Returns:
            Dictionary of model parameters
        """
        # Default parameters - can be overridden by subclasses
        return {
            "temperature": self.temperature,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
        }
    
    def should_validate_facts(self, profile: DocumentProfile) -> bool:
        """
        Determine if facts should be validated for this document type.
        
        Args:
            profile: Document profile
            
        Returns:
            Whether to validate facts
        """
        # More validation for low-quality documents
        return profile.ocr_quality < 0.7 or profile.structure_score < 0.6
    
    def get_confidence_adjustment(self, profile: DocumentProfile) -> float:
        """
        Get confidence score adjustment based on document quality.
        
        Args:
            profile: Document profile
            
        Returns:
            Multiplier for confidence scores (0.5-1.0)
        """
        # Adjust confidence based on document quality
        quality_factors = [
            profile.structure_score,
            profile.ocr_quality,
            0.8 if profile.document_type != "unknown" else 0.5
        ]
        
        return max(0.5, min(1.0, sum(quality_factors) / len(quality_factors)))