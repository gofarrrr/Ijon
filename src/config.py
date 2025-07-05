# Config
"""
Configuration for Ijon RAG system.
Updated to use Gemini embeddings by default.
"""

import os
from pathlib import Path

class Settings:
    # Logging
    log_level = "INFO"
    dev_mode = False
    
    # Embeddings (DEFAULT: Gemini)
    embedding_model = "text-embedding-004"  # Gemini 768D embeddings
    
    # Cache settings
    enable_cache = True
    cache_dir = Path(".cache")
    
    # Vector DB (using Neon PostgreSQL)
    vector_db_type = "neon"
    
    # PDF processing
    enable_ocr = False
    chunk_size = 1000
    chunk_overlap = 200
    max_pdf_size_bytes = 500 * 1024 * 1024  # 500MB
    ocr_language = "eng"
    
    # Optional settings for compatibility
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Legacy, not used
    
    def get_log_file_path(self):
        return None

# Singleton instance
_settings = None

def get_settings():
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings