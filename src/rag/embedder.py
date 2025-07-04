"""
Embedding generation module for RAG pipeline.

This module handles text embedding generation using sentence-transformers
or OpenAI embeddings, with caching and batch processing support.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.utils.errors import EmbeddingGenerationError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using various models."""

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
            model_name: Model name (e.g., 'all-MiniLM-L6-v2')
            device: Device to use ('cuda', 'cpu', or None for auto)
            cache_dir: Directory for caching embeddings
            use_cache: Whether to cache embeddings
            batch_size: Batch size for encoding
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model
        self.device = device
        self.cache_dir = cache_dir or self.settings.cache_dir / "embeddings"
        self.use_cache = use_cache and self.settings.enable_cache
        self.batch_size = batch_size
        
        # Model will be loaded lazily
        self._model: Optional[SentenceTransformer] = None
        self._openai_client = None
        
        # Setup cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized embedding generator with model: {self.model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Get or load the sentence transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension for current model."""
        if self.model_name.startswith("text-embedding"):
            # OpenAI models
            return {"text-embedding-ada-002": 1536}.get(self.model_name, 1536)
        else:
            # Load model to get dimension
            return self.model.get_sentence_embedding_dimension()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            if self.model_name.startswith("text-embedding"):
                # OpenAI embeddings - don't load sentence transformer
                self._model = None
                self._ensure_openai_client()
            else:
                # Sentence transformer model
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                
                # Log model info
                logger.info(
                    f"Loaded model with dimension: {self._model.get_sentence_embedding_dimension()}"
                )
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingGenerationError(f"Failed to load model {self.model_name}: {str(e)}")

    def _ensure_openai_client(self) -> None:
        """Ensure OpenAI client is initialized."""
        if self._openai_client is None:
            try:
                import openai
                self._openai_client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            except Exception as e:
                raise EmbeddingGenerationError(f"Failed to initialize OpenAI client: {str(e)}")

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
            
            # Generate embeddings for uncached texts
            if self.model_name.startswith("text-embedding"):
                new_embeddings = await self._generate_openai_embeddings(uncached_texts)
            else:
                new_embeddings = await self._generate_local_embeddings(uncached_texts, show_progress)
            
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

    async def _generate_local_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Generate embeddings using local sentence transformer."""
        import asyncio
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate embeddings
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress and len(batch) > 10,
                )
            )
            
            # Convert to list
            batch_embeddings = embeddings.tolist()
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    async def _generate_openai_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        self._ensure_openai_client()
        
        # OpenAI has a limit on batch size
        max_batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            
            try:
                response = await self._openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"OpenAI embedding generation failed: {e}")
                raise EmbeddingGenerationError(f"OpenAI API error: {str(e)}")
        
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