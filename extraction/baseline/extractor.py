"""
Baseline extractor implementation.

Simple prompt-based extraction using OpenAI to establish baseline performance.
Target: 70%+ quality on well-structured PDFs.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from openai import AsyncOpenAI
from pydantic import ValidationError

from ..models import (
    ExtractedKnowledge,
    Topic,
    Fact,
    Relationship,
    Question,
    CognitiveLevel,
    ExtractionMetadata
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineExtractor:
    """Simple prompt-based extraction system."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the baseline extractor.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for extraction
        """
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.temperature = 0.3  # Low temperature for consistency
        
    async def extract(self, chunk_id: str, content: str, chunk_metadata: Optional[Dict[str, Any]] = None) -> ExtractedKnowledge:
        """
        Extract knowledge from a text chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content to extract from
            chunk_metadata: Optional metadata about the chunk
            
        Returns:
            ExtractedKnowledge object with all extracted information
        """
        start_time = time.time()
        
        try:
            # Build the extraction prompt
            prompt = self._build_extraction_prompt(content)
            
            # Call OpenAI
            response = await self._call_openai(prompt)
            
            # Parse the response
            extracted_data = self._parse_response(response)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create ExtractedKnowledge object
            knowledge = ExtractedKnowledge(
                chunk_id=chunk_id,
                topics=extracted_data.get("topics", []),
                facts=extracted_data.get("facts", []),
                relationships=extracted_data.get("relationships", []),
                questions=extracted_data.get("questions", []),
                summary=extracted_data.get("summary"),
                overall_confidence=extracted_data.get("overall_confidence", 0.7),
                extraction_metadata={
                    "stage": 1,
                    "strategy": "baseline",
                    "model": self.model,
                    "temperature": self.temperature,
                    "processing_time_ms": processing_time_ms,
                    "chunk_metadata": chunk_metadata or {}
                }
            )
            
            logger.info(f"Successfully extracted knowledge from chunk {chunk_id}", 
                       extra={"topics": len(knowledge.topics), 
                             "facts": len(knowledge.facts),
                             "questions": len(knowledge.questions)})
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk_id}: {str(e)}")
            # Return minimal extraction on failure
            return ExtractedKnowledge(
                chunk_id=chunk_id,
                overall_confidence=0.0,
                extraction_metadata={
                    "stage": 1,
                    "strategy": "baseline",
                    "error": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            )
    
    def _build_extraction_prompt(self, content: str) -> str:
        """Build the extraction prompt."""
        return f"""Extract structured knowledge from the following text. Provide your response in JSON format.

Text to analyze:
{content}

Extract the following information:

1. TOPICS: Main topics and concepts discussed
   - name: Short name of the topic
   - description: Detailed description
   - keywords: Related keywords
   - confidence: Your confidence in this extraction (0.0-1.0)

2. FACTS: Key factual claims made in the text
   - claim: The factual statement
   - evidence: Supporting evidence from the text (quote if possible)
   - confidence: Your confidence (0.0-1.0)

3. RELATIONSHIPS: How entities/concepts relate to each other
   - source_entity: First entity
   - target_entity: Second entity
   - relationship_type: Type of relationship (e.g., "causes", "is part of", "contradicts")
   - description: Brief description of the relationship
   - confidence: Your confidence (0.0-1.0)

4. QUESTIONS: Generate 5 questions that this text can answer
   - question_text: The question
   - expected_answer: Brief answer based on the text
   - cognitive_level: One of [remember, understand, apply, analyze, evaluate, create]
   - difficulty: 1-5 (1=easy, 5=hard)
   - confidence: Your confidence in question quality (0.0-1.0)

5. SUMMARY: A concise summary of the main points (2-3 sentences)

6. OVERALL_CONFIDENCE: Your overall confidence in the extraction quality (0.0-1.0)

Respond with a JSON object containing these fields: topics, facts, relationships, questions, summary, overall_confidence.

Important guidelines:
- Be precise and accurate
- Only extract information clearly supported by the text
- Provide confidence scores that reflect your certainty
- Generate diverse questions at different cognitive levels
- If information is unclear or ambiguous, use lower confidence scores
"""

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API and return the response."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured knowledge from text. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the OpenAI response."""
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Convert to model objects
            parsed_data = {
                "topics": [],
                "facts": [],
                "relationships": [],
                "questions": [],
                "summary": data.get("summary", ""),
                "overall_confidence": float(data.get("overall_confidence", 0.7))
            }
            
            # Parse topics
            for topic_data in data.get("topics", []):
                try:
                    topic = Topic(
                        name=topic_data["name"],
                        description=topic_data["description"],
                        keywords=topic_data.get("keywords", []),
                        confidence=float(topic_data.get("confidence", 0.7))
                    )
                    parsed_data["topics"].append(topic)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse topic: {e}")
            
            # Parse facts
            for fact_data in data.get("facts", []):
                try:
                    fact = Fact(
                        claim=fact_data["claim"],
                        evidence=fact_data.get("evidence"),
                        confidence=float(fact_data.get("confidence", 0.7))
                    )
                    parsed_data["facts"].append(fact)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse fact: {e}")
            
            # Parse relationships
            for rel_data in data.get("relationships", []):
                try:
                    relationship = Relationship(
                        source_entity=rel_data["source_entity"],
                        target_entity=rel_data["target_entity"],
                        relationship_type=rel_data["relationship_type"],
                        description=rel_data.get("description"),
                        confidence=float(rel_data.get("confidence", 0.7))
                    )
                    parsed_data["relationships"].append(relationship)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse relationship: {e}")
            
            # Parse questions
            for q_data in data.get("questions", []):
                try:
                    # Map cognitive level
                    cognitive_level = CognitiveLevel(q_data.get("cognitive_level", "understand").lower())
                    
                    question = Question(
                        question_text=q_data["question_text"],
                        expected_answer=q_data.get("expected_answer"),
                        cognitive_level=cognitive_level,
                        difficulty=int(q_data.get("difficulty", 3)),
                        confidence=float(q_data.get("confidence", 0.7))
                    )
                    parsed_data["questions"].append(question)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse question: {e}")
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise

    async def extract_batch(self, chunks: List[Dict[str, Any]], max_concurrent: int = 5) -> List[ExtractedKnowledge]:
        """
        Extract knowledge from multiple chunks concurrently.
        
        Args:
            chunks: List of dicts with 'id' and 'content' keys
            max_concurrent: Maximum concurrent extractions
            
        Returns:
            List of ExtractedKnowledge objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def extract_with_semaphore(chunk):
            async with semaphore:
                return await self.extract(
                    chunk_id=chunk["id"],
                    content=chunk["content"],
                    chunk_metadata=chunk.get("metadata")
                )
        
        tasks = [extract_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        return results