"""
Gemini-specific embedding generator for RAG pipeline.
This version only supports Gemini embeddings without sentence-transformers dependency.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import google.generativeai as genai

from src.config import get_settings
from src.utils.errors import EmbeddingGenerationError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiEmbeddingGenerator:
    """Generate embeddings using Gemini text-embedding-004."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize Gemini embedding generator.

        Args:
            cache_dir: Directory for caching embeddings
            use_cache: Whether to cache embeddings
            batch_size: Batch size for encoding
        """
        self.settings = get_settings()
        self.model_name = "text-embedding-004"
        self.cache_dir = cache_dir or Path(".cache") / "embeddings"
        self.use_cache = use_cache
        self.batch_size = batch_size
        
        # Configure Gemini
        self._configure_gemini()
        
        # Setup cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Gemini embedding generator")

    @property
    def dimension(self) -> int:
        """Get embedding dimension for Gemini model."""
        return 768

    def _configure_gemini(self) -> None:
        """Configure Gemini API."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            logger.info("Gemini API configured successfully")
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to configure Gemini: {str(e)}")

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
            
            # Generate embeddings for uncached texts
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
        # Gemini has a batch limit of 100
        max_batch_size = 100
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