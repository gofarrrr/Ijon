"""
Enhanced baseline extractor with battle-tested prompt patterns.

This implementation incorporates agent loop architecture, thinking blocks,
and academic prose patterns for improved extraction quality.
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


ENHANCED_EXTRACTION_SYSTEM_PROMPT = """You are a specialized Knowledge Extraction Agent operating in an iterative extraction loop.

## Agent Loop Architecture
You operate in systematic phases:
1. **Analyze Content**: Understand the text structure and key themes
2. **Plan Extraction**: Determine extraction strategy based on content type
3. **Execute Extraction**: Apply extraction techniques systematically
4. **Validate Results**: Ensure quality and completeness
5. **Refine Output**: Improve based on confidence assessment

## Extraction Methodology

### Phase 1: Content Analysis (Thinking)
Before extracting, analyze:
```
What type of content is this?
What are the main themes?
What extraction challenges exist?
What level of detail is appropriate?
```

### Phase 2: Systematic Extraction
Extract knowledge in layers:
- First pass: Identify major topics and themes
- Second pass: Extract detailed facts and evidence
- Third pass: Discover relationships and connections
- Fourth pass: Generate comprehensive questions

### Phase 3: Quality Validation
After extraction, reflect:
```
Have I captured all key information?
Are my confidence scores accurate?
Do the relationships make sense?
Are the questions diverse and valuable?
```

## Output Requirements
- Provide structured JSON output
- Include confidence scores reflecting true certainty
- Ensure all claims have supporting evidence
- Generate questions across all cognitive levels
- Never expose extraction mechanics to end users

Remember: Quality over quantity. Extract only well-supported information."""


class EnhancedBaselineExtractor:
    """Enhanced extraction system with battle-tested patterns."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", use_gemini: bool = False):
        """
        Initialize the enhanced extractor.
        
        Args:
            api_key: API key (OpenAI or Google)
            model: Model to use for extraction
            use_gemini: Whether to use Gemini instead of OpenAI
        """
        self.use_gemini = use_gemini
        self.model = model
        self.temperature = 0.3  # Low temperature for consistency
        self.enable_thinking_blocks = True
        self.use_academic_prose = True
        
        if use_gemini:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        
    async def extract(self, chunk_id: str, content: str, chunk_metadata: Optional[Dict[str, Any]] = None) -> ExtractedKnowledge:
        """
        Extract knowledge using enhanced methodology.
        
        Args:
            chunk_id: Unique identifier for the chunk
            content: Text content to extract from
            chunk_metadata: Optional metadata about the chunk
            
        Returns:
            ExtractedKnowledge object with all extracted information
        """
        start_time = time.time()
        
        try:
            # Build the enhanced extraction prompt
            prompt = self._build_enhanced_prompt(content, chunk_metadata)
            
            # Call OpenAI with enhanced system prompt
            response = await self._call_openai_enhanced(prompt)
            
            # Parse the response with validation
            extracted_data = self._parse_enhanced_response(response)
            
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
                    "strategy": "enhanced_baseline",
                    "model": self.model,
                    "temperature": self.temperature,
                    "processing_time_ms": processing_time_ms,
                    "thinking_enabled": self.enable_thinking_blocks,
                    "chunk_metadata": chunk_metadata or {},
                    "extraction_phases": extracted_data.get("extraction_phases", [])
                }
            )
            
            logger.info(f"Successfully extracted knowledge from chunk {chunk_id}", 
                       extra={"topics": len(knowledge.topics), 
                             "facts": len(knowledge.facts),
                             "questions": len(knowledge.questions),
                             "confidence": knowledge.overall_confidence})
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Failed to extract from chunk {chunk_id}: {str(e)}")
            # Return minimal extraction on failure
            return ExtractedKnowledge(
                chunk_id=chunk_id,
                overall_confidence=0.0,
                extraction_metadata={
                    "stage": 1,
                    "strategy": "enhanced_baseline",
                    "error": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            )
    
    def _build_enhanced_prompt(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Build enhanced extraction prompt with thinking blocks."""
        
        context_info = ""
        if metadata:
            if "document_type" in metadata:
                context_info += f"\nDocument Type: {metadata['document_type']}"
            if "section" in metadata:
                context_info += f"\nSection: {metadata['section']}"
        
        prompt = f"""Extract comprehensive structured knowledge from the following text.

{context_info}

Text to analyze:
{content}

## Extraction Requirements

### Phase 1: Content Analysis
First, analyze the content:
```
Content Type: [Identify the type of content]
Main Themes: [List 2-3 primary themes]
Complexity Level: [Simple/Moderate/Complex]
Extraction Strategy: [Your planned approach]
```

### Phase 2: Knowledge Extraction

Extract the following with careful attention to quality:

1. **TOPICS** - Main concepts and themes
   For each topic provide:
   - name: Concise, meaningful name
   - description: Comprehensive description in 2-3 sentences of flowing prose
   - keywords: 3-5 most relevant keywords
   - confidence: Your true confidence (0.0-1.0)

2. **FACTS** - Verifiable claims with evidence
   For each fact provide:
   - claim: The specific factual statement
   - evidence: Direct quote or paraphrase from text
   - context: Brief context explaining significance
   - confidence: Based on evidence strength (0.0-1.0)

3. **RELATIONSHIPS** - Meaningful connections between entities
   For each relationship provide:
   - source_entity: First entity (be specific)
   - target_entity: Second entity (be specific)
   - relationship_type: Precise type (e.g., "causes", "enables", "contradicts")
   - description: Explain the relationship in one flowing sentence
   - evidence: What in the text supports this relationship
   - confidence: How certain you are (0.0-1.0)

4. **QUESTIONS** - Generate 6-8 diverse questions across cognitive levels
   For each question provide:
   - question_text: Clear, well-formed question
   - expected_answer: Comprehensive answer based on the text
   - cognitive_level: One of [remember, understand, apply, analyze, evaluate, create]
   - difficulty: 1-5 scale based on complexity
   - rationale: Why this question is valuable
   - confidence: Quality assessment (0.0-1.0)

   Ensure questions are distributed across cognitive levels:
   - At least 1 "remember" question (factual recall)
   - At least 2 "understand" questions (comprehension)
   - At least 1 "apply" question (practical application)
   - At least 1 "analyze" question (breaking down concepts)
   - Include "evaluate" or "create" if content supports it

5. **SUMMARY** - Comprehensive summary in academic prose
   Write 3-4 sentences that:
   - Capture the essential message
   - Flow naturally without bullet points
   - Connect key concepts coherently
   - Provide scholarly synthesis

### Phase 3: Quality Reflection
After extraction, assess your work:
```
Completeness: [Have I captured all key information?]
Accuracy: [Are all claims well-supported?]
Relationships: [Do connections make logical sense?]
Questions: [Do they cover diverse cognitive levels?]
Overall Quality: [Rate 0.0-1.0 with justification]
```

### Output Format
Provide your response as a JSON object with these fields:
- topics: Array of topic objects
- facts: Array of fact objects  
- relationships: Array of relationship objects
- questions: Array of question objects
- summary: String with academic prose summary
- overall_confidence: Float 0.0-1.0
- extraction_phases: Array of strings documenting your extraction process
- quality_reflection: Object with your quality assessment

Important: Maintain academic quality throughout. Focus on precision, evidence, and meaningful connections."""
        
        return prompt

    async def _call_openai_enhanced(self, prompt: str) -> str:
        """Call LLM API with enhanced system prompt."""
        try:
            if self.use_gemini:
                # Combine system prompt and user prompt for Gemini
                full_prompt = f"{ENHANCED_EXTRACTION_SYSTEM_PROMPT}\n\n{prompt}\n\nRespond with valid JSON only."
                
                # Generate with Gemini
                response = await self.client.generate_content_async(
                    full_prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "response_mime_type": "application/json"
                    }
                )
                
                return response.text
            else:
                # OpenAI path
                messages = [
                    {
                        "role": "system", 
                        "content": ENHANCED_EXTRACTION_SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {str(e)}")
            raise

    def _parse_enhanced_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the enhanced response."""
        try:
            # Parse JSON
            data = json.loads(response)
            
            # Convert to model objects with enhanced validation
            parsed_data = {
                "topics": [],
                "facts": [],
                "relationships": [],
                "questions": [],
                "summary": data.get("summary", ""),
                "overall_confidence": float(data.get("overall_confidence", 0.7)),
                "extraction_phases": data.get("extraction_phases", [])
            }
            
            # Parse topics with enhanced structure
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
            
            # Parse facts with context
            for fact_data in data.get("facts", []):
                try:
                    fact = Fact(
                        claim=fact_data["claim"],
                        evidence=fact_data.get("evidence"),
                        confidence=float(fact_data.get("confidence", 0.7)),
                        metadata={"context": fact_data.get("context", "")}
                    )
                    parsed_data["facts"].append(fact)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse fact: {e}")
            
            # Parse relationships with evidence
            for rel_data in data.get("relationships", []):
                try:
                    relationship = Relationship(
                        source_entity=rel_data["source_entity"],
                        target_entity=rel_data["target_entity"],
                        relationship_type=rel_data["relationship_type"],
                        description=rel_data.get("description"),
                        confidence=float(rel_data.get("confidence", 0.7)),
                        metadata={"evidence": rel_data.get("evidence", "")}
                    )
                    parsed_data["relationships"].append(relationship)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse relationship: {e}")
            
            # Parse questions with rationale
            for q_data in data.get("questions", []):
                try:
                    cognitive_level = CognitiveLevel(q_data.get("cognitive_level", "understand").lower())
                    
                    question = Question(
                        question_text=q_data["question_text"],
                        expected_answer=q_data.get("expected_answer"),
                        cognitive_level=cognitive_level,
                        difficulty=int(q_data.get("difficulty", 3)),
                        confidence=float(q_data.get("confidence", 0.7)),
                        metadata={"rationale": q_data.get("rationale", "")}
                    )
                    parsed_data["questions"].append(question)
                except (KeyError, ValidationError) as e:
                    logger.warning(f"Failed to parse question: {e}")
            
            # Log quality reflection if present
            if "quality_reflection" in data:
                logger.info(f"Quality reflection: {data['quality_reflection']}")
            
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
        
        # Log batch statistics
        total_topics = sum(len(r.topics) for r in results)
        total_facts = sum(len(r.facts) for r in results)
        total_questions = sum(len(r.questions) for r in results)
        avg_confidence = sum(r.overall_confidence for r in results) / len(results)
        
        logger.info(
            f"Batch extraction complete: {len(chunks)} chunks, "
            f"{total_topics} topics, {total_facts} facts, "
            f"{total_questions} questions, avg confidence: {avg_confidence:.2f}"
        )
        
        return results