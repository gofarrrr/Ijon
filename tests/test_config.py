"""
Tests for configuration module.
"""

import os
from pathlib import Path

import pytest

from src.config import Settings, get_settings


class TestSettings:
    """Test the Settings configuration class."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        
        assert settings.vector_db_type == "pinecone"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.log_level == "INFO"
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("VECTOR_DB_TYPE", "supabase")
        monkeypatch.setenv("CHUNK_SIZE", "2000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        settings = Settings()
        
        assert settings.vector_db_type == "supabase"
        assert settings.chunk_size == 2000
        assert settings.log_level == "DEBUG"

    def test_chunk_overlap_validation(self):
        """Test that chunk overlap must be smaller than chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
            Settings(chunk_size=100, chunk_overlap=200)

    def test_vector_db_validation_pinecone(self):
        """Test Pinecone configuration validation."""
        with pytest.raises(ValueError, match="pinecone_api_key required"):
            Settings(vector_db_type="pinecone", pinecone_api_key=None)

    def test_vector_db_validation_supabase(self):
        """Test Supabase configuration validation."""
        with pytest.raises(ValueError, match="supabase_url required"):
            Settings(
                vector_db_type="supabase",
                supabase_url=None,
                supabase_key="test-key"
            )

    def test_vector_db_validation_neon(self):
        """Test Neon configuration validation."""
        with pytest.raises(ValueError, match="neon_connection_string required"):
            Settings(vector_db_type="neon", neon_connection_string=None)

    def test_drive_folder_ids_parsing(self):
        """Test parsing of comma-separated folder IDs."""
        # Test string input
        settings = Settings(drive_folder_ids="folder1,folder2,folder3")
        assert settings.drive_folder_ids == ["folder1", "folder2", "folder3"]
        
        # Test list input
        settings = Settings(drive_folder_ids=["folder1", "folder2"])
        assert settings.drive_folder_ids == ["folder1", "folder2"]
        
        # Test empty string
        settings = Settings(drive_folder_ids="")
        assert settings.drive_folder_ids == []

    def test_helper_properties(self):
        """Test helper properties."""
        settings = Settings(
            max_pdf_size_mb=10,
            embedding_model="all-MiniLM-L6-v2"
        )
        
        assert settings.max_pdf_size_bytes == 10 * 1024 * 1024
        assert settings.embedding_dimension == 384
        assert settings.mcp_url == "http://localhost:8080"

    def test_embedding_dimensions(self):
        """Test embedding dimension calculation for different models."""
        test_cases = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
            ("text-embedding-ada-002", 1536),
            ("unknown-model", 384),  # Default
        ]
        
        for model, expected_dim in test_cases:
            settings = Settings(embedding_model=model)
            assert settings.embedding_dimension == expected_dim

    def test_log_file_path_creation(self, tmp_path):
        """Test log file path directory creation."""
        log_path = tmp_path / "logs" / "test.log"
        settings = Settings(log_file_path=log_path)
        
        # Directory should not exist yet
        assert not log_path.parent.exists()
        
        # Call get_log_file_path should create directory
        result = settings.get_log_file_path()
        
        assert result == log_path
        assert log_path.parent.exists()

    def test_get_settings_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2