"""
Data models for the Quality Knowledge Extraction System.

These models represent the core data structures used throughout the extraction pipeline,
with built-in validation and confidence scoring.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


from pydantic import conint, confloat
from typing import Annotated

# Confidence level type - float between 0.0 and 1.0
ConfidenceLevel = Annotated[float, confloat(ge=0.0, le=1.0)]


class DocumentType(str, Enum):
    """Types of documents we can process"""
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    BUSINESS = "business"
    TUTORIAL = "tutorial"
    NARRATIVE = "narrative"
    LEGAL = "legal"
    MEDICAL = "medical"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class CognitiveLevel(str, Enum):
    """Bloom's Taxonomy levels for questions"""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class Topic(BaseModel):
    """A topic or concept extracted from the document"""
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    confidence: ConfidenceLevel = Field(..., description="Extraction confidence")
    keywords: List[str] = Field(default_factory=list)
    source_chunks: List[str] = Field(default_factory=list, description="Chunk IDs where topic appears")
    
    class Config:
        json_encoders = {UUID: str}


class Fact(BaseModel):
    """A factual claim extracted from the document"""
    id: UUID = Field(default_factory=uuid4)
    claim: str = Field(..., min_length=1)
    evidence: Optional[str] = Field(None, description="Supporting evidence from text")
    confidence: ConfidenceLevel = Field(..., description="Extraction confidence")
    topics: List[str] = Field(default_factory=list, description="Related topic IDs")
    source_page: Optional[int] = None
    source_chunk: Optional[str] = None
    
    class Config:
        json_encoders = {UUID: str}


class Relationship(BaseModel):
    """A relationship between entities or concepts"""
    id: UUID = Field(default_factory=uuid4)
    source_entity: str = Field(..., description="Source entity/concept")
    target_entity: str = Field(..., description="Target entity/concept")
    relationship_type: str = Field(..., description="Type of relationship")
    description: Optional[str] = None
    confidence: ConfidenceLevel = Field(..., description="Extraction confidence")
    bidirectional: bool = Field(default=False)
    source_chunks: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {UUID: str}


class Question(BaseModel):
    """A question that can be answered from the document"""
    id: UUID = Field(default_factory=uuid4)
    question_text: str = Field(..., min_length=10)
    expected_answer: Optional[str] = None
    answer_chunks: List[str] = Field(default_factory=list, description="Chunks containing answer")
    cognitive_level: CognitiveLevel = Field(default=CognitiveLevel.UNDERSTAND)
    difficulty: int = Field(ge=1, le=5, default=3)
    topics: List[str] = Field(default_factory=list, description="Related topic IDs")
    confidence: ConfidenceLevel = Field(..., description="Question quality confidence")
    
    class Config:
        json_encoders = {UUID: str}


class ExtractedKnowledge(BaseModel):
    """Complete knowledge extracted from a document chunk"""
    id: UUID = Field(default_factory=uuid4)
    chunk_id: Optional[str] = Field(None, description="Source chunk identifier")
    topics: List[Topic] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    summary: Optional[str] = None
    overall_confidence: ConfidenceLevel = Field(..., description="Overall extraction confidence")
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('overall_confidence', pre=True, always=True)
    def calculate_overall_confidence(cls, v, values):
        """Calculate overall confidence if not provided"""
        if v is not None:
            return v
        
        # Average confidence across all components
        confidences = []
        for field in ['topics', 'facts', 'relationships', 'questions']:
            if field in values and values[field]:
                confidences.extend([item.confidence for item in values[field]])
        
        if confidences:
            return sum(confidences) / len(confidences)
        return 0.5  # Default medium confidence
    
    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class DocumentProfile(BaseModel):
    """Profile of a document for extraction strategy selection"""
    id: UUID = Field(default_factory=uuid4)
    document_id: str
    document_type: DocumentType
    structure_score: ConfidenceLevel = Field(..., description="How well-structured the document is")
    ocr_quality: ConfidenceLevel = Field(..., description="OCR quality if applicable")
    language: str = Field(default="en")
    page_count: int = Field(gt=0)
    has_tables: bool = False
    has_figures: bool = False
    has_citations: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recommended_strategy: str = Field(default="baseline")
    type_confidence: ConfidenceLevel = Field(default=0.5, description="Confidence in document type classification")
    special_elements: Dict[str, Any] = Field(default_factory=dict, description="Special elements found in document")
    profiled_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class QualityDimension(BaseModel):
    """Individual quality dimension score"""
    dimension: str
    score: ConfidenceLevel
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class QualityScore(BaseModel):
    """Quality assessment of extracted knowledge"""
    id: UUID = Field(default_factory=uuid4)
    extraction_id: UUID
    overall_score: ConfidenceLevel
    consistency: QualityDimension
    grounding: QualityDimension
    coherence: QualityDimension
    completeness: QualityDimension
    issues_found: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    passed_threshold: bool = Field(default=False)
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('passed_threshold', pre=True, always=True)
    def check_threshold(cls, v, values):
        """Check if quality passes minimum threshold"""
        if 'overall_score' in values:
            return values['overall_score'] >= 0.7
        return False
    
    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process"""
    extraction_id: UUID
    stage: int = Field(ge=1, le=8, description="Extraction system stage used")
    strategy_used: str
    attempts: int = Field(ge=1, default=1)
    processing_time_ms: float = Field(gt=0)
    tokens_used: int = Field(ge=0)
    model_used: str = Field(default="gpt-4")
    temperature: float = Field(ge=0, le=2, default=0.3)
    success: bool = True
    error_message: Optional[str] = None
    
    class Config:
        json_encoders = {UUID: str}


class ValidationResult(BaseModel):
    """Result of extraction validation"""
    extraction_id: UUID
    validation_type: str  # "manual", "automated", "collaborative"
    is_valid: bool
    confidence: ConfidenceLevel
    issues: List[str] = Field(default_factory=list)
    corrections: Dict[str, Any] = Field(default_factory=dict)
    validated_by: str  # Agent ID or human identifier
    validated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}