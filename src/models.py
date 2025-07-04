"""
Core data models for Ijon PDF RAG System.

This module defines all the Pydantic models used throughout the application
for data validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class ProcessingStatus(str, Enum):
    """Status of PDF processing."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionType(str, Enum):
    """Types of questions for generation."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    INFERENTIAL = "inferential"
    EVALUATIVE = "evaluative"


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


# =============================================================================
# Base Models
# =============================================================================


class TimestampedModel(BaseModel):
    """Base model with timestamp fields."""

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


# =============================================================================
# PDF Models
# =============================================================================


class PDFMetadata(TimestampedModel):
    """Metadata for a PDF document."""

    file_id: str = Field(..., description="Google Drive file ID")
    filename: str = Field(..., description="Original filename")
    drive_path: str = Field(..., description="Path in Google Drive")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status",
    )
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Extracted metadata
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    subject: Optional[str] = Field(None, description="Document subject")
    keywords: List[str] = Field(default_factory=list, description="Document keywords")
    creation_date: Optional[datetime] = Field(None, description="Document creation date")

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Ensure file size is positive."""
        if v < 0:
            raise ValueError("File size must be non-negative")
        return v


class PDFPage(BaseModel):
    """Represents a single page from a PDF."""

    page_number: int = Field(..., ge=1, description="Page number (1-indexed)")
    text: str = Field(..., description="Extracted text content")
    images: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of images with metadata",
    )
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of tables with data",
    )
    has_ocr: bool = Field(False, description="Whether OCR was used")
    ocr_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="OCR confidence score",
    )


class PDFChunk(BaseModel):
    """A chunk of text from a PDF document."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique chunk ID")
    pdf_id: str = Field(..., description="Source PDF file ID")
    content: str = Field(..., min_length=1, description="Chunk text content")
    page_numbers: List[int] = Field(..., min_length=1, description="Pages this chunk spans")
    chunk_index: int = Field(..., ge=0, description="Position in document")
    
    # Metadata for retrieval
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    word_count: int = Field(..., ge=0, description="Number of words in chunk")
    char_count: int = Field(..., ge=0, description="Number of characters in chunk")
    
    # Hierarchical information
    section_title: Optional[str] = Field(None, description="Section title if available")
    chapter_title: Optional[str] = Field(None, description="Chapter title if available")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Chunk content cannot be empty or just whitespace")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Calculate word and character counts if not provided."""
        if not self.word_count:
            self.word_count = len(self.content.split())
        if not self.char_count:
            self.char_count = len(self.content)


# =============================================================================
# Vector Database Models
# =============================================================================


class Document(BaseModel):
    """Document for vector database storage."""

    id: str = Field(..., description="Unique document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding dimensions if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("Embedding cannot be empty")
        return v


class SearchResult(BaseModel):
    """Result from vector search."""

    document: Document = Field(..., description="Retrieved document")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(..., ge=1, description="Result rank")


class SearchQuery(BaseModel):
    """Query for vector search."""

    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    include_embeddings: bool = Field(False, description="Include embeddings in results")


# =============================================================================
# RAG Models
# =============================================================================


class RAGContext(BaseModel):
    """Context for RAG generation."""

    query: str = Field(..., description="User query")
    relevant_chunks: List[PDFChunk] = Field(..., description="Retrieved chunks")
    total_tokens: int = Field(..., ge=0, description="Total context tokens")
    
    def get_context_text(self) -> str:
        """Get formatted context text for generation."""
        context_parts = []
        for i, chunk in enumerate(self.relevant_chunks, 1):
            source = f"[Source {i}: {chunk.metadata.get('filename', 'Unknown')}, "
            source += f"Pages {min(chunk.page_numbers)}-{max(chunk.page_numbers)}]"
            context_parts.append(f"{source}\n{chunk.content}")
        return "\n\n".join(context_parts)


class GeneratedAnswer(BaseModel):
    """Generated answer with citations."""

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    citations: List[Dict[str, Any]] = Field(..., description="Source citations")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Answer confidence")
    processing_time: float = Field(..., ge=0.0, description="Generation time in seconds")
    model_used: str = Field(..., description="LLM model used")


# =============================================================================
# Question Generation Models
# =============================================================================


class GeneratedQuestion(BaseModel):
    """A generated question with metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Question ID")
    question: str = Field(..., min_length=10, description="Question text")
    question_type: QuestionType = Field(..., description="Type of question")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty level")
    source_chunk_ids: List[str] = Field(..., description="Source chunk IDs")
    expected_answer_length: str = Field(..., description="Expected answer length")
    keywords: List[str] = Field(default_factory=list, description="Question keywords")


class QuestionBatch(TimestampedModel):
    """Batch of generated questions."""

    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="Batch ID")
    pdf_id: str = Field(..., description="Source PDF ID")
    questions: List[GeneratedQuestion] = Field(..., description="Generated questions")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Batch diversity score")
    generation_time: float = Field(..., ge=0.0, description="Total generation time")


# =============================================================================
# MCP Server Models
# =============================================================================


class MCPRequest(BaseModel):
    """MCP server request."""

    tool: str = Field(..., description="Tool name to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID")


class MCPResponse(BaseModel):
    """MCP server response."""

    request_id: str = Field(..., description="Original request ID")
    success: bool = Field(..., description="Whether request succeeded")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")


class ProcessingJob(TimestampedModel):
    """Background processing job."""

    job_id: str = Field(default_factory=lambda: str(uuid4()), description="Job ID")
    job_type: str = Field(..., description="Type of job")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result")
    error: Optional[str] = Field(None, description="Error if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Job metadata")


# =============================================================================
# Export Models
# =============================================================================


class ExportRequest(BaseModel):
    """Request to export data."""

    export_type: str = Field(..., description="Type of data to export")
    format: ExportFormat = Field(..., description="Export format")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Export filters")
    include_embeddings: bool = Field(False, description="Include embeddings in export")


class ExportResult(BaseModel):
    """Result of export operation."""

    export_id: str = Field(default_factory=lambda: str(uuid4()), description="Export ID")
    file_path: str = Field(..., description="Path to exported file")
    format: ExportFormat = Field(..., description="Export format")
    record_count: int = Field(..., ge=0, description="Number of records exported")
    file_size_bytes: int = Field(..., ge=0, description="Export file size")
    export_time: float = Field(..., ge=0.0, description="Export time in seconds")


# =============================================================================
# Knowledge Graph Models
# =============================================================================


class EntityType(str, Enum):
    """Types of entities in the knowledge graph."""
    
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    DOCUMENT = "document"
    TOPIC = "topic"
    OTHER = "other"


class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph."""
    
    # Document relationships
    MENTIONS = "mentions"
    DESCRIBES = "describes"
    REFERENCES = "references"
    
    # Entity relationships
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    BELONGS_TO = "belongs_to"
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    
    # Temporal relationships
    HAPPENED_BEFORE = "happened_before"
    HAPPENED_AFTER = "happened_after"
    CONCURRENT_WITH = "concurrent_with"
    
    # Conceptual relationships
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    CAUSES = "causes"
    INFLUENCES = "influences"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"


class GraphEntity(BaseModel):
    """Entity in the knowledge graph."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Entity ID")
    name: str = Field(..., min_length=1, description="Entity name")
    type: EntityType = Field(..., description="Entity type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    
    # Source tracking
    source_pdf_ids: List[str] = Field(default_factory=list, description="Source PDF IDs")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Source chunk IDs")
    
    # Embeddings for similarity search
    embedding: Optional[List[float]] = Field(None, description="Entity embedding")
    
    # Metadata
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def __hash__(self):
        """Make entity hashable for deduplication."""
        return hash((self.name.lower(), self.type))


class GraphRelationship(BaseModel):
    """Relationship between entities in the knowledge graph."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Relationship ID")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: RelationshipType = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    
    # Source tracking
    source_pdf_ids: List[str] = Field(default_factory=list, description="Source PDF IDs")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Source chunk IDs")
    
    # Metadata
    confidence_score: float = Field(1.0, ge=0.0, le=1.0, description="Extraction confidence")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphQuery(BaseModel):
    """Query for the knowledge graph."""
    
    query_type: str = Field(..., description="Type of graph query")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    limit: int = Field(10, ge=1, le=1000, description="Maximum results")


class GraphResult(BaseModel):
    """Result from a graph query."""
    
    entities: List[GraphEntity] = Field(default_factory=list, description="Entities in result")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="Relationships in result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")
    query_time: float = Field(..., ge=0.0, description="Query execution time")


class GraphTriple(BaseModel):
    """Triple representation for knowledge graph."""
    
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship/predicate")
    object: str = Field(..., description="Object entity")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Triple properties")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Triple confidence")


class GraphSchema(BaseModel):
    """Schema information for the knowledge graph."""
    
    entity_types: List[str] = Field(..., description="Available entity types")
    relationship_types: List[str] = Field(..., description="Available relationship types")
    entity_properties: Dict[str, List[str]] = Field(default_factory=dict, description="Properties by entity type")
    relationship_properties: Dict[str, List[str]] = Field(default_factory=dict, description="Properties by relationship type")
    indexes: List[Dict[str, Any]] = Field(default_factory=list, description="Graph indexes")
    constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Graph constraints")