"""
HyDE (Hypothetical Document Embeddings) query enhancement for better retrieval.

This module implements the HyDE technique which generates hypothetical documents
that would answer the query, then uses those for more effective retrieval.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

from src.config import get_settings
from src.utils.logging import get_logger, log_performance
from src.utils.errors import RetrievalError

load_dotenv()

logger = get_logger(__name__)


class HyDEEnhancer:
    """
    Stateless HyDE query enhancer following 12-factor principles.
    
    HyDE improves retrieval by:
    1. Generating hypothetical documents that would answer the query
    2. Using these hypothetical documents instead of/alongside the query
    3. Better semantic matching through document-to-document similarity
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_hypothetical_docs: int = 3,
        use_multi_perspective: bool = True,
    ):
        """
        Initialize HyDE enhancer.
        
        Args:
            model: LLM model to use for generation
            temperature: Generation temperature (higher = more diverse)
            max_hypothetical_docs: Number of hypothetical documents to generate
            use_multi_perspective: Whether to generate from multiple perspectives
        """
        self.settings = get_settings()
        self.model = model
        self.temperature = temperature
        self.max_hypothetical_docs = max_hypothetical_docs
        self.use_multi_perspective = use_multi_perspective
    
    @log_performance
    async def enhance_query(
        self,
        query: str,
        client: AsyncOpenAI,
        doc_type: Optional[str] = None,
        domain_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance query with hypothetical documents - pure function.
        
        Args:
            query: Original user query
            client: OpenAI client (passed in, not stored)
            doc_type: Document type for context (academic, technical, etc.)
            domain_context: Additional domain context
            
        Returns:
            Dict containing:
            - original_query: The original query
            - hypothetical_docs: List of generated hypothetical documents
            - enhanced_queries: List of enhanced search queries
            - generation_metadata: Metadata about the generation process
        """
        try:
            # Generate hypothetical documents
            hypothetical_docs = await self._generate_hypothetical_documents(
                query=query,
                client=client,
                doc_type=doc_type,
                domain_context=domain_context,
            )
            
            # Create enhanced queries combining original + hypothetical
            enhanced_queries = self._create_enhanced_queries(query, hypothetical_docs)
            
            logger.info(
                f"Generated {len(hypothetical_docs)} hypothetical documents "
                f"and {len(enhanced_queries)} enhanced queries"
            )
            
            return {
                "original_query": query,
                "hypothetical_docs": hypothetical_docs,
                "enhanced_queries": enhanced_queries,
                "generation_metadata": {
                    "model_used": self.model,
                    "temperature": self.temperature,
                    "doc_type": doc_type,
                    "multi_perspective": self.use_multi_perspective,
                },
            }
            
        except Exception as e:
            logger.error(f"HyDE enhancement failed: {e}")
            # Graceful fallback to original query
            return {
                "original_query": query,
                "hypothetical_docs": [],
                "enhanced_queries": [query],
                "generation_metadata": {"error": str(e)},
            }
    
    async def _generate_hypothetical_documents(
        self,
        query: str,
        client: AsyncOpenAI,
        doc_type: Optional[str] = None,
        domain_context: Optional[str] = None,
    ) -> List[str]:
        """Generate hypothetical documents that would answer the query."""
        
        prompts = self._build_generation_prompts(query, doc_type, domain_context)
        
        # Generate documents concurrently for speed
        tasks = []
        for prompt in prompts[:self.max_hypothetical_docs]:
            task = self._generate_single_document(client, prompt)
            tasks.append(task)
        
        # Wait for all generations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failures and return successful generations
        hypothetical_docs = []
        for result in results:
            if isinstance(result, str) and result.strip():
                hypothetical_docs.append(result.strip())
            elif isinstance(result, Exception):
                logger.warning(f"Failed to generate hypothetical document: {result}")
        
        return hypothetical_docs
    
    def _build_generation_prompts(
        self,
        query: str,
        doc_type: Optional[str] = None,
        domain_context: Optional[str] = None,
    ) -> List[str]:
        """Build prompts for generating hypothetical documents."""
        
        # Base context for different document types
        doc_contexts = {
            "academic": "academic research paper or scientific article",
            "technical": "technical documentation, manual, or guide", 
            "medical": "medical literature, clinical study, or health guideline",
            "legal": "legal document, regulation, or case study",
            "business": "business report, analysis, or strategic document",
        }
        
        doc_context = doc_contexts.get(doc_type, "informative document")
        
        # Base prompt for direct answering with academic prose
        base_prompt = f"""Write a substantive passage from an {doc_context} that would directly answer this question: {query}

Write in continuous academic prose with varied sentence structures. Begin with a clear topic sentence that addresses the query, then develop the answer through flowing paragraphs. Include specific details, facts, and examples woven naturally into the narrative. Write as if excerpting from an authoritative scholarly source, using formal academic language without lists or bullet points. The passage should read as a cohesive excerpt that could appear in a peer-reviewed publication."""
        
        prompts = [base_prompt]
        
        # Add multi-perspective prompts if enabled
        if self.use_multi_perspective:
            # Alternative perspective prompt with academic prose
            alt_prompt = f"""Write a comprehensive section from an {doc_context} that provides context and background information relevant to: {query}

Develop your explanation through flowing academic paragraphs that build understanding progressively. Begin with foundational concepts and definitions, then expand to show relationships and implications. Use sophisticated transitions between ideas to create a seamless narrative. Write in the formal, scholarly style of an authoritative academic reference, maintaining continuous prose throughout without resorting to lists or fragmented structures."""
            
            prompts.append(alt_prompt)
            
            # Example-focused prompt with academic prose
            example_prompt = f"""Write a detailed passage from an {doc_context} that includes specific examples, case studies, or practical applications related to: {query}

Present examples within flowing academic prose, integrating concrete details, numbers, and real-world instances naturally into the narrative. Each example should be introduced with context, explored in depth, and connected back to the main topic through sophisticated transitions. Maintain the scholarly tone of peer-reviewed literature while making the examples vivid and illustrative. Avoid listing examples; instead, weave them into a cohesive analytical narrative."""
            
            prompts.append(example_prompt)
        
        # Add domain-specific context if provided
        if domain_context:
            domain_prompt = f"""Write an in-depth excerpt from an {doc_context} in the {domain_context} domain that answers: {query}

Craft your response in sophisticated academic prose appropriate to the {domain_context} field. Employ domain-specific terminology precisely while maintaining readability through clear explanations woven into the narrative. Present specialized methods and insights as they would appear in high-impact journals of the field, using the writing conventions and analytical depth expected in {domain_context} scholarship. Build your argument through interconnected paragraphs that demonstrate mastery of the domain's discourse."""
            
            prompts.append(domain_prompt)
        
        return prompts
    
    async def _generate_single_document(
        self,
        client: AsyncOpenAI,
        prompt: str,
    ) -> str:
        """Generate a single hypothetical document."""
        
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic writer skilled in crafting sophisticated scholarly prose. Generate substantial passages in continuous paragraphs with varied sentence structures. Your writing should demonstrate the depth and nuance found in peer-reviewed publications, using formal academic language while maintaining clarity. Never use bullet points or lists; instead, weave information into flowing narrative paragraphs with smooth transitions between ideas."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=300,  # Keep documents concise
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate hypothetical document: {e}")
            raise
    
    def _create_enhanced_queries(
        self,
        original_query: str,
        hypothetical_docs: List[str],
    ) -> List[str]:
        """Create enhanced queries combining original and hypothetical content."""
        
        enhanced_queries = [original_query]  # Always include original
        
        for doc in hypothetical_docs:
            if doc:
                # Extract key phrases from hypothetical document
                key_phrases = self._extract_key_phrases(doc)
                
                # Create enhanced query by combining original with key phrases
                if key_phrases:
                    enhanced_query = f"{original_query} {' '.join(key_phrases[:3])}"
                    enhanced_queries.append(enhanced_query)
                
                # Also add the hypothetical document itself as a "query"
                # This enables document-to-document matching
                enhanced_queries.append(doc[:200])  # Truncate for efficiency
        
        return enhanced_queries
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text - simple implementation."""
        import re
        
        # Simple extraction - in production, could use NLP libraries
        # Remove common words and extract meaningful phrases
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        key_phrases = []
        for sentence in sentences[:2]:  # First 2 sentences
            # Remove common words
            words = sentence.strip().split()
            meaningful_words = []
            
            # Simple filter for meaningful words
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            for word in words:
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word and clean_word not in stopwords and len(clean_word) > 2:
                    meaningful_words.append(clean_word)
            
            if meaningful_words:
                # Take phrases of 2-3 words
                for i in range(len(meaningful_words) - 1):
                    phrase = ' '.join(meaningful_words[i:i+2])
                    key_phrases.append(phrase)
        
        return key_phrases[:5]  # Return top 5 phrases


class HyDERetrievalWrapper:
    """
    Wrapper that integrates HyDE enhancement with existing retrievers.
    
    This maintains the same interface as standard retrievers but adds
    HyDE enhancement as an optional feature.
    """
    
    def __init__(
        self,
        base_retriever,
        hyde_enhancer: Optional[HyDEEnhancer] = None,
        client: Optional[AsyncOpenAI] = None,
        enable_hyde: bool = True,
        hybrid_weight: float = 0.7,  # Weight for original query vs hypothetical
    ):
        """
        Initialize HyDE wrapper.
        
        Args:
            base_retriever: Base retriever to wrap
            hyde_enhancer: HyDE enhancer instance
            client: OpenAI client for generation
            enable_hyde: Whether HyDE is enabled
            hybrid_weight: Weight for combining original and enhanced results
        """
        self.base_retriever = base_retriever
        self.hyde_enhancer = hyde_enhancer or HyDEEnhancer()
        self.client = client
        self.enable_hyde = enable_hyde
        self.hybrid_weight = hybrid_weight
        
        # Initialize client if not provided
        if not self.client and enable_hyde:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = AsyncOpenAI(api_key=api_key)
            else:
                logger.warning("No OpenAI API key found, HyDE will be disabled")
                self.enable_hyde = False
    
    async def initialize(self) -> None:
        """Initialize the base retriever."""
        await self.base_retriever.initialize()
    
    @log_performance
    async def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        use_hyde: Optional[bool] = None,
        doc_type: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve chunks with optional HyDE enhancement.
        
        Args:
            query: Search query
            top_k: Number of chunks to return
            min_score: Minimum similarity score
            filters: Metadata filters
            use_hyde: Override global HyDE setting
            doc_type: Document type for HyDE context
            **kwargs: Additional arguments for base retriever
            
        Returns:
            List of (chunk, score) tuples
        """
        # Determine if we should use HyDE
        should_use_hyde = use_hyde if use_hyde is not None else self.enable_hyde
        
        if not should_use_hyde or not self.client:
            # Fall back to base retriever
            return await self.base_retriever.retrieve_chunks(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filters=filters,
                **kwargs
            )
        
        try:
            # Generate HyDE enhancement
            hyde_result = await self.hyde_enhancer.enhance_query(
                query=query,
                client=self.client,
                doc_type=doc_type,
            )
            
            # Perform retrieval with multiple queries
            all_results = {}  # chunk_id -> (chunk, best_score)
            
            # Retrieve with original query (higher weight)
            original_results = await self.base_retriever.retrieve_chunks(
                query=query,
                top_k=top_k * 2,  # Get more for merging
                min_score=min_score,
                filters=filters,
                **kwargs
            )
            
            # Add original results with full weight
            for chunk, score in original_results:
                chunk_id = getattr(chunk, 'id', str(hash(chunk.content)))
                weighted_score = score * self.hybrid_weight
                all_results[chunk_id] = (chunk, weighted_score)
            
            # Retrieve with enhanced queries
            for enhanced_query in hyde_result["enhanced_queries"][1:]:  # Skip original
                try:
                    enhanced_results = await self.base_retriever.retrieve_chunks(
                        query=enhanced_query,
                        top_k=top_k,
                        min_score=min_score,
                        filters=filters,
                        **kwargs
                    )
                    
                    # Add enhanced results with reduced weight
                    enhanced_weight = (1.0 - self.hybrid_weight) / len(hyde_result["enhanced_queries"][1:])
                    
                    for chunk, score in enhanced_results:
                        chunk_id = getattr(chunk, 'id', str(hash(chunk.content)))
                        weighted_score = score * enhanced_weight
                        
                        if chunk_id in all_results:
                            # Take maximum score
                            existing_chunk, existing_score = all_results[chunk_id]
                            all_results[chunk_id] = (existing_chunk, max(existing_score, weighted_score))
                        else:
                            all_results[chunk_id] = (chunk, weighted_score)
                            
                except Exception as e:
                    logger.warning(f"Enhanced query failed: {e}")
                    continue
            
            # Sort by score and return top results
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(
                f"HyDE retrieval: {len(original_results)} original + "
                f"{len(final_results) - len(original_results)} enhanced = "
                f"{len(final_results)} total, returning top {top_k}"
            )
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"HyDE retrieval failed, falling back to standard: {e}")
            # Fall back to standard retrieval
            return await self.base_retriever.retrieve_chunks(
                query=query,
                top_k=top_k,
                min_score=min_score,
                filters=filters,
                **kwargs
            )
    
    def __getattr__(self, name):
        """Delegate other methods to base retriever."""
        return getattr(self.base_retriever, name)


def create_hyde_enhanced_retriever(base_retriever, **kwargs):
    """
    Factory function to create HyDE-enhanced retriever.
    
    Args:
        base_retriever: Base retriever to enhance
        **kwargs: Arguments for HyDERetrievalWrapper
        
    Returns:
        HyDE-enhanced retriever
    """
    return HyDERetrievalWrapper(base_retriever, **kwargs)