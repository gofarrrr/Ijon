"""
Document-aware extraction strategies for Stage 2.

This module implements document profiling and type-specific extraction strategies
to improve extraction quality over the baseline.
"""

from .document_profiler import DocumentProfiler, DocumentProfile
from .base_strategy import ExtractionStrategy
from .academic_strategy import AcademicStrategy
from .technical_strategy import TechnicalStrategy
from .narrative_strategy import NarrativeStrategy
from .strategy_factory import StrategyFactory

__all__ = [
    "DocumentProfiler",
    "DocumentProfile", 
    "ExtractionStrategy",
    "AcademicStrategy",
    "TechnicalStrategy",
    "NarrativeStrategy",
    "StrategyFactory"
]