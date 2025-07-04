"""
Custom exceptions for Ijon PDF RAG System.

This module defines all custom exceptions used throughout the application
for better error handling and debugging.
"""

from typing import Any, Optional


class IjonException(Exception):
    """Base exception for all Ijon-specific errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# PDF Processing Exceptions
# =============================================================================


class PDFProcessingError(IjonException):
    """Base exception for PDF processing errors."""

    pass


class PDFExtractionError(PDFProcessingError):
    """Error during PDF text/content extraction."""

    pass


class PDFSizeError(PDFProcessingError):
    """PDF exceeds maximum allowed size."""

    def __init__(self, file_size: int, max_size: int, filename: str) -> None:
        """Initialize with size information."""
        message = f"PDF '{filename}' size ({file_size} bytes) exceeds maximum ({max_size} bytes)"
        super().__init__(message, {"file_size": file_size, "max_size": max_size, "filename": filename})


class PDFCorruptedError(PDFProcessingError):
    """PDF file is corrupted or invalid."""

    pass


class OCRError(PDFProcessingError):
    """Error during OCR processing."""

    pass


# =============================================================================
# Google Drive Exceptions
# =============================================================================


class GoogleDriveError(IjonException):
    """Base exception for Google Drive operations."""

    pass


class DriveAuthenticationError(GoogleDriveError):
    """Authentication with Google Drive failed."""

    pass


class DriveQuotaExceededError(GoogleDriveError):
    """Google Drive API quota exceeded."""

    def __init__(self, retry_after: Optional[int] = None) -> None:
        """Initialize with retry information."""
        message = "Google Drive API quota exceeded"
        details = {}
        if retry_after:
            message += f". Retry after {retry_after} seconds"
            details["retry_after"] = retry_after
        super().__init__(message, details)


class DriveFileNotFoundError(GoogleDriveError):
    """File not found in Google Drive."""

    def __init__(self, file_id: str) -> None:
        """Initialize with file ID."""
        message = f"File with ID '{file_id}' not found in Google Drive"
        super().__init__(message, {"file_id": file_id})


# =============================================================================
# Vector Database Exceptions
# =============================================================================


class VectorDatabaseError(IjonException):
    """Base exception for vector database operations."""

    pass


class VectorDatabaseConnectionError(VectorDatabaseError):
    """Failed to connect to vector database."""

    pass


class VectorDatabaseNotInitializedError(VectorDatabaseError):
    """Vector database not properly initialized."""

    pass


class EmbeddingGenerationError(VectorDatabaseError):
    """Error generating embeddings."""

    pass


class VectorSearchError(VectorDatabaseError):
    """Error during vector search operation."""

    pass


class IndexNotFoundError(VectorDatabaseError):
    """Vector index not found."""

    def __init__(self, index_name: str) -> None:
        """Initialize with index name."""
        message = f"Vector index '{index_name}' not found"
        super().__init__(message, {"index_name": index_name})


# =============================================================================
# Graph Database Exceptions
# =============================================================================


class GraphDatabaseError(IjonException):
    """Base exception for graph database operations."""

    pass


class GraphDatabaseConnectionError(GraphDatabaseError):
    """Failed to connect to graph database."""

    pass


class GraphDatabaseNotInitializedError(GraphDatabaseError):
    """Graph database not properly initialized."""

    pass


class GraphQueryError(GraphDatabaseError):
    """Error executing graph query."""

    pass


class GraphSchemaError(GraphDatabaseError):
    """Error with graph schema operations."""

    pass


# =============================================================================
# RAG Pipeline Exceptions
# =============================================================================


class RAGPipelineError(IjonException):
    """Base exception for RAG pipeline errors."""

    pass


class ChunkingError(RAGPipelineError):
    """Error during text chunking."""

    pass


class RetrievalError(RAGPipelineError):
    """Error during document retrieval."""

    pass


class GenerationError(RAGPipelineError):
    """Error during answer generation."""

    pass


class ContextWindowExceededError(RAGPipelineError):
    """Context window limit exceeded."""

    def __init__(self, context_length: int, max_length: int) -> None:
        """Initialize with context information."""
        message = f"Context length ({context_length} tokens) exceeds maximum ({max_length} tokens)"
        super().__init__(message, {"context_length": context_length, "max_length": max_length})


# =============================================================================
# MCP Server Exceptions
# =============================================================================


class MCPServerError(IjonException):
    """Base exception for MCP server errors."""

    pass


class MCPAuthenticationError(MCPServerError):
    """MCP authentication failed."""

    pass


class MCPToolNotFoundError(MCPServerError):
    """Requested MCP tool not found."""

    def __init__(self, tool_name: str) -> None:
        """Initialize with tool name."""
        message = f"MCP tool '{tool_name}' not found"
        super().__init__(message, {"tool_name": tool_name})


class MCPRequestError(MCPServerError):
    """Invalid MCP request."""

    pass


# =============================================================================
# Agent Exceptions
# =============================================================================


class AgentError(IjonException):
    """Base exception for agent operations."""

    pass


class QuestionGenerationError(AgentError):
    """Error generating questions."""

    pass


class AgentToolError(AgentError):
    """Error executing agent tool."""

    def __init__(self, tool_name: str, error: str) -> None:
        """Initialize with tool information."""
        message = f"Error executing tool '{tool_name}': {error}"
        super().__init__(message, {"tool_name": tool_name, "error": error})


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(IjonException):
    """Configuration error."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration missing."""

    def __init__(self, config_name: str) -> None:
        """Initialize with config name."""
        message = f"Required configuration '{config_name}' is missing"
        super().__init__(message, {"config_name": config_name})


# =============================================================================
# Processing Exceptions
# =============================================================================


class ProcessingTimeoutError(IjonException):
    """Processing operation timed out."""

    def __init__(self, operation: str, timeout: int) -> None:
        """Initialize with timeout information."""
        message = f"Operation '{operation}' timed out after {timeout} seconds"
        super().__init__(message, {"operation": operation, "timeout": timeout})


class ConcurrencyLimitError(IjonException):
    """Concurrent processing limit exceeded."""

    def __init__(self, current: int, limit: int) -> None:
        """Initialize with concurrency information."""
        message = f"Concurrent operations ({current}) exceeds limit ({limit})"
        super().__init__(message, {"current": current, "limit": limit})


# =============================================================================
# Export Exceptions
# =============================================================================


class ExportError(IjonException):
    """Base exception for export operations."""

    pass


class UnsupportedFormatError(ExportError):
    """Unsupported export format."""

    def __init__(self, format: str, supported: list[str]) -> None:
        """Initialize with format information."""
        message = f"Format '{format}' not supported. Supported formats: {', '.join(supported)}"
        super().__init__(message, {"format": format, "supported": supported})