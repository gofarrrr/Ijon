"""
Embedding generation module for RAG pipeline.

This module handles text embedding generation using Gemini embeddings,
with caching and batch processing support.

Updated: Now uses Gemini text-embedding-004 by default.
Legacy OpenAI support removed.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.config import get_settings
from src.utils.errors import EmbeddingGenerationError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using Gemini or other API models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize embedding generator.

        Args:
            model_name: Model name (default: 'text-embedding-004')
            device: Device to use (ignored for API models)
            cache_dir: Directory for caching embeddings
            use_cache: Whether to cache embeddings
            batch_size: Batch size for encoding
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.device = device  # Kept for compatibility
        self.cache_dir = cache_dir or self.settings.cache_dir / "embeddings"
        self.use_cache = use_cache and self.settings.enable_cache
        self.batch_size = batch_size
        
        # API configuration
        self._gemini_configured = False
        
        # Setup cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized embedding generator with model: {self.model_name}")

    @property
    def model(self):
        """Compatibility property - returns None for API models."""
        return None

    @property
    def dimension(self) -> int:
        """Get embedding dimension for current model."""
        if self.model_name == "text-embedding-004":
            return 768  # Gemini embeddings
        else:
            # Default for compatibility
            return 768

    def _ensure_gemini_client(self) -> None:
        """Ensure Gemini is configured for embeddings."""
        if self._gemini_configured:
            return
            
        try:
            import google.generativeai as genai
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=gemini_api_key)
            self._gemini_configured = True
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to configure Gemini: {str(e)}")

    @log_performance
    async def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embeddings

        Raises:
            EmbeddingGenerationError: If generation fails
        """
        if not texts:
            return []
        
        try:
            # Check cache for existing embeddings
            if self.use_cache:
                embeddings, uncached_indices = self._check_cache(texts)
                if not uncached_indices:
                    logger.debug(f"All {len(texts)} embeddings found in cache")
                    return embeddings
                
                # Get uncached texts
                uncached_texts = [texts[i] for i in uncached_indices]
                logger.debug(f"Generating {len(uncached_texts)} embeddings (cached: {len(texts) - len(uncached_texts)})")
            else:
                embeddings = [None] * len(texts)
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
            
            # Generate embeddings using Gemini
            new_embeddings = await self._generate_gemini_embeddings(uncached_texts)
            
            # Combine cached and new embeddings
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                
                # Cache the new embedding
                if self.use_cache:
                    self._cache_embedding(texts[idx], embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {str(e)}")
    
    async def _generate_gemini_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings using Gemini API."""
        self._ensure_gemini_client()
        
        import google.generativeai as genai
        
        # Gemini has a batch limit of 100
        max_batch_size = min(self.batch_size, 100)
        all_embeddings = []
        
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            
            try:
                # Create embedding requests
                embeddings = []
                for text in batch:
                    result = genai.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type="retrieval_document",
                    )
                    embeddings.append(result['embedding'])
                
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                logger.error(f"Gemini embedding generation failed: {e}")
                raise EmbeddingGenerationError(f"Gemini API error: {str(e)}")
        
        return all_embeddings

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            EmbeddingGenerationError: If generation fails
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include model name in cache key
        key_string = f"{self.model_name}:{text}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _check_cache(self, texts: List[str]) -> tuple[List[Optional[List[float]]], List[int]]:
        """
        Check cache for existing embeddings.

        Returns:
            Tuple of (embeddings list with None for uncached, indices of uncached texts)
        """
        embeddings = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            
            if cache_file.exists():
                try:
                    # Load from cache
                    embedding = np.load(cache_file).tolist()
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
                    embeddings.append(None)
                    uncached_indices.append(i)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
        
        return embeddings, uncached_indices

    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        try:
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            
            # Save as numpy array for efficiency
            np.save(cache_file, np.array(embedding))
            
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared embedding cache")

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        metric: str = "cosine",
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score
        """
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        elif metric == "euclidean":
            # Negative euclidean distance (higher is more similar)
            similarity = -np.linalg.norm(e1 - e2)
        elif metric == "dot":
            # Dot product
            similarity = np.dot(e1, e2)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        return float(similarity)


def create_embedding_generator() -> EmbeddingGenerator:
    """Create an embedding generator with default settings."""
    settings = get_settings()
    return EmbeddingGenerator(
        model_name=settings.embedding_model,
        use_cache=settings.enable_cache,
    )