"""
Entity and relationship extraction for knowledge graph construction.

This module handles extracting structured information from text chunks
to build a knowledge graph representation of the content.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from src.config import get_settings
from src.models import (
    EntityType,
    GraphEntity,
    GraphRelationship,
    GraphTriple,
    PDFChunk,
    RelationshipType,
)
from src.utils.errors import GraphDatabaseError
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


class ExtractionPrompt(BaseModel):
    """Prompt configuration for entity extraction."""
    
    system_prompt: str = Field(..., description="System prompt for extraction")
    user_prompt_template: str = Field(..., description="User prompt template")
    examples: List[Dict[str, Any]] = Field(default_factory=list, description="Few-shot examples")


class KnowledgeExtractor:
    """
    Extract entities and relationships from text for knowledge graph construction.
    
    This class uses LLMs to identify entities, their types, and relationships
    between them in text chunks. It supports multiple extraction strategies
    and can be customized with domain-specific prompts.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_entities_per_chunk: int = 50,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize knowledge extractor.
        
        Args:
            model_name: LLM model for extraction
            temperature: Generation temperature
            max_entities_per_chunk: Maximum entities to extract per chunk
            confidence_threshold: Minimum confidence for extraction
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.question_gen_model
        self.temperature = temperature
        self.max_entities_per_chunk = max_entities_per_chunk
        self.confidence_threshold = confidence_threshold
        
        # LLM client will be initialized lazily
        self._llm_client = None
        
        # Entity deduplication cache
        self._entity_cache: Dict[str, GraphEntity] = {}
        self._relationship_cache: Set[Tuple[str, str, str]] = set()

    def _ensure_llm_client(self) -> None:
        """Ensure LLM client is initialized."""
        if self._llm_client is None:
            try:
                import openai
                self._llm_client = openai.AsyncOpenAI(
                    api_key=self.settings.openai_api_key
                )
            except Exception as e:
                raise GraphDatabaseError(f"Failed to initialize OpenAI client: {str(e)}")

    @log_performance
    async def extract_from_chunk(
        self,
        chunk: PDFChunk,
        context_chunks: Optional[List[PDFChunk]] = None,
    ) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """
        Extract entities and relationships from a PDF chunk.
        
        Args:
            chunk: The chunk to process
            context_chunks: Additional chunks for context
            
        Returns:
            Tuple of (entities, relationships)
        """
        try:
            # Prepare context
            text = chunk.content
            if context_chunks:
                context = "\n\n".join(c.content for c in context_chunks[:2])
                text = f"Context:\n{context}\n\nMain text:\n{text}"
            
            # Extract entities first
            entities = await self._extract_entities(text, chunk)
            
            # Then extract relationships between found entities
            relationships = await self._extract_relationships(text, entities, chunk)
            
            # Deduplicate
            entities = self._deduplicate_entities(entities)
            relationships = self._deduplicate_relationships(relationships)
            
            logger.info(
                f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk",
                extra={"chunk_id": chunk.id, "pdf_id": chunk.pdf_id},
            )
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Failed to extract from chunk: {e}")
            return [], []

    async def _extract_entities(
        self,
        text: str,
        chunk: PDFChunk,
    ) -> List[GraphEntity]:
        """Extract entities from text using LLM."""
        self._ensure_llm_client()
        
        prompt = self._get_entity_extraction_prompt()
        
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt_template.format(text=text)},
        ]
        
        try:
            response = await self._llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            
            result = self._parse_entity_response(
                response.choices[0].message.content,
                chunk,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    async def _extract_relationships(
        self,
        text: str,
        entities: List[GraphEntity],
        chunk: PDFChunk,
    ) -> List[GraphRelationship]:
        """Extract relationships between entities using LLM."""
        if len(entities) < 2:
            return []
        
        self._ensure_llm_client()
        
        # Create entity reference for the prompt
        entity_list = [
            f"{i+1}. {e.name} ({e.type.value})"
            for i, e in enumerate(entities)
        ]
        
        prompt = self._get_relationship_extraction_prompt()
        
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {
                "role": "user",
                "content": prompt.user_prompt_template.format(
                    text=text,
                    entities="\n".join(entity_list),
                ),
            },
        ]
        
        try:
            response = await self._llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            
            result = self._parse_relationship_response(
                response.choices[0].message.content,
                entities,
                chunk,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    def _get_entity_extraction_prompt(self) -> ExtractionPrompt:
        """Get prompt for entity extraction."""
        return ExtractionPrompt(
            system_prompt="""You are an expert at extracting entities from text for knowledge graph construction.

Extract all important entities from the text, including:
- People (PERSON)
- Organizations/Companies (ORGANIZATION)
- Locations/Places (LOCATION)
- Concepts/Ideas (CONCEPT)
- Events (EVENT)
- Technologies/Tools (TECHNOLOGY)
- Products/Services (PRODUCT)
- Topics/Subjects (TOPIC)

For each entity, provide:
1. The entity name (as it appears in text)
2. The entity type (from the list above)
3. A brief description (1-2 sentences)
4. Confidence score (0.0-1.0)

Return the result as a JSON object with an "entities" array.""",
            user_prompt_template="""Extract entities from the following text:

{text}

Return a JSON object in this format:
{{
  "entities": [
    {{
      "name": "Entity Name",
      "type": "ENTITY_TYPE",
      "description": "Brief description",
      "confidence": 0.9
    }}
  ]
}}""",
        )

    def _get_relationship_extraction_prompt(self) -> ExtractionPrompt:
        """Get prompt for relationship extraction."""
        return ExtractionPrompt(
            system_prompt="""You are an expert at identifying relationships between entities in text.

Given a text and a list of entities, identify relationships between them such as:
- MENTIONS (document mentions entity)
- DESCRIBES (detailed description)
- RELATED_TO (general relationship)
- PART_OF (hierarchical relationship)
- WORKS_FOR (employment)
- LOCATED_IN (location relationship)
- IS_A (type relationship)
- CAUSES (causal relationship)
- INFLUENCES (influence relationship)

Only extract relationships that are explicitly stated or strongly implied in the text.

Return the result as a JSON object with a "relationships" array.""",
            user_prompt_template="""Given the following entities:
{entities}

Extract relationships from this text:
{text}

Return a JSON object in this format:
{{
  "relationships": [
    {{
      "source": 1,
      "target": 2,
      "type": "RELATIONSHIP_TYPE",
      "description": "Brief description of the relationship",
      "confidence": 0.9
    }}
  ]
}}

Use entity numbers from the list (1-based indexing).""",
        )

    def _parse_entity_response(
        self,
        response: str,
        chunk: PDFChunk,
    ) -> List[GraphEntity]:
        """Parse LLM response for entities."""
        try:
            import json
            data = json.loads(response)
            
            entities = []
            for item in data.get("entities", [])[:self.max_entities_per_chunk]:
                # Skip low confidence entities
                if item.get("confidence", 0) < self.confidence_threshold:
                    continue
                
                # Map type string to enum
                entity_type = self._map_entity_type(item.get("type", "OTHER"))
                
                entity = GraphEntity(
                    name=item["name"],
                    type=entity_type,
                    properties={
                        "description": item.get("description", ""),
                        "first_seen": chunk.metadata.get("filename", "Unknown"),
                    },
                    source_pdf_ids=[chunk.pdf_id],
                    source_chunk_ids=[chunk.id],
                    confidence_score=item.get("confidence", 0.8),
                )
                
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Failed to parse entity response: {e}")
            return []

    def _parse_relationship_response(
        self,
        response: str,
        entities: List[GraphEntity],
        chunk: PDFChunk,
    ) -> List[GraphRelationship]:
        """Parse LLM response for relationships."""
        try:
            import json
            data = json.loads(response)
            
            relationships = []
            for item in data.get("relationships", []):
                # Skip low confidence relationships
                if item.get("confidence", 0) < self.confidence_threshold:
                    continue
                
                # Get source and target entities
                source_idx = item.get("source", 0) - 1  # Convert to 0-based
                target_idx = item.get("target", 0) - 1
                
                if 0 <= source_idx < len(entities) and 0 <= target_idx < len(entities):
                    source_entity = entities[source_idx]
                    target_entity = entities[target_idx]
                    
                    # Map type string to enum
                    rel_type = self._map_relationship_type(item.get("type", "RELATED_TO"))
                    
                    relationship = GraphRelationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        type=rel_type,
                        properties={
                            "description": item.get("description", ""),
                            "context": chunk.content[:200],
                        },
                        source_pdf_ids=[chunk.pdf_id],
                        source_chunk_ids=[chunk.id],
                        confidence_score=item.get("confidence", 0.8),
                    )
                    
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to parse relationship response: {e}")
            return []

    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map string to EntityType enum."""
        type_mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "LOCATION": EntityType.LOCATION,
            "CONCEPT": EntityType.CONCEPT,
            "EVENT": EntityType.EVENT,
            "TECHNOLOGY": EntityType.TECHNOLOGY,
            "PRODUCT": EntityType.PRODUCT,
            "TOPIC": EntityType.TOPIC,
        }
        return type_mapping.get(type_str.upper(), EntityType.OTHER)

    def _map_relationship_type(self, type_str: str) -> RelationshipType:
        """Map string to RelationshipType enum."""
        type_mapping = {
            "MENTIONS": RelationshipType.MENTIONS,
            "DESCRIBES": RelationshipType.DESCRIBES,
            "REFERENCES": RelationshipType.REFERENCES,
            "RELATED_TO": RelationshipType.RELATED_TO,
            "PART_OF": RelationshipType.PART_OF,
            "BELONGS_TO": RelationshipType.BELONGS_TO,
            "WORKS_FOR": RelationshipType.WORKS_FOR,
            "LOCATED_IN": RelationshipType.LOCATED_IN,
            "IS_A": RelationshipType.IS_A,
            "CAUSES": RelationshipType.CAUSES,
            "INFLUENCES": RelationshipType.INFLUENCES,
        }
        return type_mapping.get(type_str.upper(), RelationshipType.RELATED_TO)

    def _deduplicate_entities(
        self,
        entities: List[GraphEntity],
    ) -> List[GraphEntity]:
        """Deduplicate entities based on name and type."""
        unique_entities = []
        seen = set()
        
        for entity in entities:
            key = (entity.name.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # Merge source tracking
                for existing in unique_entities:
                    if (existing.name.lower(), existing.type) == key:
                        existing.source_pdf_ids.extend(entity.source_pdf_ids)
                        existing.source_chunk_ids.extend(entity.source_chunk_ids)
                        existing.source_pdf_ids = list(set(existing.source_pdf_ids))
                        existing.source_chunk_ids = list(set(existing.source_chunk_ids))
                        break
        
        return unique_entities

    def _deduplicate_relationships(
        self,
        relationships: List[GraphRelationship],
    ) -> List[GraphRelationship]:
        """Deduplicate relationships."""
        unique_relationships = []
        seen = set()
        
        for rel in relationships:
            key = (rel.source_id, rel.target_id, rel.type)
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships

    async def extract_from_chunks(
        self,
        chunks: List[PDFChunk],
        batch_size: int = 5,
        use_context: bool = True,
    ) -> Tuple[List[GraphEntity], List[GraphRelationship]]:
        """
        Extract entities and relationships from multiple chunks.
        
        Args:
            chunks: List of chunks to process
            batch_size: Number of chunks to process concurrently
            use_context: Whether to use surrounding chunks as context
            
        Returns:
            Tuple of (all entities, all relationships)
        """
        all_entities = []
        all_relationships = []
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = []
            for j, chunk in enumerate(batch):
                # Get context chunks if requested
                context_chunks = None
                if use_context:
                    context_chunks = []
                    # Previous chunk
                    if i + j > 0:
                        context_chunks.append(chunks[i + j - 1])
                    # Next chunk
                    if i + j < len(chunks) - 1:
                        context_chunks.append(chunks[i + j + 1])
                
                task = self.extract_from_chunk(chunk, context_chunks)
                tasks.append(task)
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks)
            
            # Collect results
            for entities, relationships in results:
                all_entities.extend(entities)
                all_relationships.extend(relationships)
        
        # Global deduplication
        all_entities = self._deduplicate_entities(all_entities)
        all_relationships = self._deduplicate_relationships(all_relationships)
        
        logger.info(
            f"Extracted {len(all_entities)} unique entities and "
            f"{len(all_relationships)} relationships from {len(chunks)} chunks"
        )
        
        return all_entities, all_relationships

    def clear_cache(self) -> None:
        """Clear entity and relationship caches."""
        self._entity_cache.clear()
        self._relationship_cache.clear()
        logger.info("Cleared extraction caches")