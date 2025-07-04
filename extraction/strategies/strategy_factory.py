"""
Factory for creating document-specific extraction strategies.
"""

from typing import Dict, Type

from extraction.models import DocumentProfile
from extraction.strategies.base_strategy import ExtractionStrategy
from extraction.strategies.academic_strategy import AcademicStrategy
from extraction.strategies.technical_strategy import TechnicalStrategy
from extraction.strategies.narrative_strategy import NarrativeStrategy
from extraction.baseline.extractor import BaselineExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StrategyFactory:
    """Factory for creating extraction strategies based on document profile."""
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the strategy factory.
        
        Args:
            openai_api_key: OpenAI API key for all strategies
        """
        self.api_key = openai_api_key
        
        # Map strategy names to classes
        self.strategy_map: Dict[str, Type[ExtractionStrategy]] = {
            "academic": AcademicStrategy,
            "technical": TechnicalStrategy,
            "narrative": NarrativeStrategy,
        }
        
        # Cache for strategy instances
        self._strategy_cache = {}
    
    def get_strategy(self, profile: DocumentProfile) -> ExtractionStrategy:
        """
        Get the appropriate extraction strategy for a document.
        
        Args:
            profile: Document profile with classification
            
        Returns:
            Extraction strategy instance
        """
        strategy_name = profile.recommended_strategy
        
        # Handle special cases
        if strategy_name == "baseline" or strategy_name == "baseline_validated":
            # Return baseline extractor (not a strategy subclass)
            logger.info(f"Using baseline extractor for document type: {profile.document_type.value}")
            return None  # Caller should use BaselineExtractor directly
        
        # Get strategy class
        strategy_class = self.strategy_map.get(strategy_name)
        
        if not strategy_class:
            logger.warning(f"Unknown strategy '{strategy_name}', falling back to narrative")
            strategy_class = NarrativeStrategy
            strategy_name = "narrative"
        
        # Check cache
        if strategy_name not in self._strategy_cache:
            logger.info(f"Creating new {strategy_name} strategy instance")
            self._strategy_cache[strategy_name] = strategy_class(self.api_key)
        
        return self._strategy_cache[strategy_name]
    
    def get_available_strategies(self) -> Dict[str, str]:
        """
        Get list of available strategies with descriptions.
        
        Returns:
            Dictionary of strategy names to descriptions
        """
        return {
            "baseline": "Simple prompt-based extraction (default)",
            "academic": "Optimized for research papers and academic texts",
            "technical": "Optimized for technical documentation and manuals",
            "narrative": "Optimized for business books and narrative texts",
            "baseline_validated": "Baseline with extra validation for poor quality documents"
        }