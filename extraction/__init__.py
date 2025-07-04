"""
Quality Knowledge Extraction System

A progressive, research-based approach to extracting high-quality knowledge from PDFs.
Implements stages from simple baseline to complex orchestration, with validation at each step.
"""

__version__ = "0.1.0"

from .models import (
    ExtractedKnowledge,
    Topic,
    Fact,
    Relationship,
    Question,
    QualityScore,
    DocumentProfile,
    ExtractionMetadata
)

__all__ = [
    "ExtractedKnowledge",
    "Topic",
    "Fact",
    "Relationship",
    "Question",
    "QualityScore",
    "DocumentProfile",
    "ExtractionMetadata"
]