"""
Entity extraction module for graph RAG.

Extracts entities and relationships from text chunks using LLMs.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

import google.generativeai as genai

from src.config import get_settings
from src.rag.graph_store import Entity, Relationship
from src.models import PDFChunk
from src.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity extraction from a chunk."""
    entities: List[Entity]
    relationships: List[Relationship]
    chunk_id: str
    confidence: float = 1.0


class EntityExtractor:
    """
    Extract entities and relationships from text chunks.
    
    Uses Gemini or other LLMs to identify entities and their relationships
    in document chunks for building the knowledge graph.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        batch_size: int = 5,
    ):
        """Initialize entity extractor."""
        self.settings = get_settings()
        self.model_name = model_name
        self.temperature = temperature
        self.batch_size = batch_size
        
        # Entity types to extract
        self.entity_types = [
            "CONCEPT",      # Mental models, theories, principles
            "PERSON",       # People, researchers, authors
            "TECHNIQUE",    # Methods, approaches, strategies
            "BIAS",         # Cognitive biases
            "PRINCIPLE",    # Core principles, rules
            "FRAMEWORK",    # Frameworks, models, systems
            "EXAMPLE",      # Specific examples, cases
            "DOMAIN",       # Fields, areas of application
        ]
        
        # Relationship types
        self.relationship_types = [
            "DESCRIBES",      # Entity describes another
            "APPLIES_TO",     # Entity applies to domain/context
            "CREATED_BY",     # Created or discovered by person
            "RELATED_TO",     # General relationship
            "PART_OF",        # Component relationship
            "CONTRASTS_WITH", # Opposing concepts
            "LEADS_TO",       # Causal relationship
            "EXAMPLE_OF",     # Instance relationship
        ]
        
        logger.info(f"Initialized entity extractor with model: {self.model_name}")
    
    @log_performance
    async def extract_from_chunk(
        self,
        chunk: PDFChunk,
        context: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a single chunk.
        
        Args:
            chunk: PDF chunk to process
            context: Additional context about the document
            
        Returns:
            Extraction result with entities and relationships
        """
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(chunk.content, context)
            
            # Call Gemini for extraction
            model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": 1000,
                }
            )
            
            # Parse the response
            extraction_result = self._parse_extraction_response(
                response.text, chunk.id
            )
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Entity extraction failed for chunk {chunk.id}: {e}")
            # Return empty result on failure
            return ExtractionResult(
                entities=[],
                relationships=[],
                chunk_id=chunk.id,
                confidence=0.0
            )
    
    @log_performance
    async def extract_from_chunks(
        self,
        chunks: List[PDFChunk],
        context: Optional[str] = None
    ) -> List[ExtractionResult]:
        """Extract entities from multiple chunks in batches."""
        results = []
        
        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            # Extract concurrently within batch
            batch_tasks = [
                self.extract_from_chunk(chunk, context)
                for chunk in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i//self.batch_size + 1}, extracted from {len(batch)} chunks")
        
        return results
    
    def _build_extraction_prompt(
        self,
        content: str,
        context: Optional[str] = None
    ) -> str:
        """Build prompt for entity extraction."""
        entity_types_str = ", ".join(self.entity_types)
        relationship_types_str = ", ".join(self.relationship_types)
        
        prompt = f"""Extract entities and relationships from the following text.

Entity Types to identify: {entity_types_str}
Relationship Types to identify: {relationship_types_str}

{f"Document Context: {context}" if context else ""}

Text to analyze:
{content}

Instructions:
1. Identify key entities mentioned in the text
2. Classify each entity into one of the given types
3. Provide a brief description for each entity
4. Identify relationships between entities
5. Only extract entities and relationships that are explicitly mentioned or clearly implied

Output Format (JSON):
{{
    "entities": [
        {{
            "name": "entity name",
            "type": "ENTITY_TYPE",
            "description": "brief description"
        }}
    ],
    "relationships": [
        {{
            "source": "source entity name",
            "target": "target entity name", 
            "type": "RELATIONSHIP_TYPE"
        }}
    ]
}}

Extract entities and relationships:"""
        
        return prompt
    
    def _parse_extraction_response(
        self,
        response_text: str,
        chunk_id: str
    ) -> ExtractionResult:
        """Parse the LLM response into entities and relationships."""
        entities = []
        relationships = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in extraction response for chunk {chunk_id}")
                return ExtractionResult([], [], chunk_id, 0.5)
            
            data = json.loads(json_match.group())
            
            # Parse entities
            entity_map = {}  # name -> Entity object
            
            for entity_data in data.get('entities', []):
                entity = Entity(
                    name=entity_data.get('name', '').strip(),
                    type=entity_data.get('type', 'CONCEPT'),
                    description=entity_data.get('description', ''),
                    chunk_ids=[chunk_id],
                    confidence=0.9  # High confidence for LLM extraction
                )
                
                if entity.name and entity.type in self.entity_types:
                    entities.append(entity)
                    entity_map[entity.name.lower()] = entity
            
            # Parse relationships (will need entity IDs later)
            relationship_data = []
            for rel_data in data.get('relationships', []):
                source_name = rel_data.get('source', '').strip().lower()
                target_name = rel_data.get('target', '').strip().lower()
                rel_type = rel_data.get('type', 'RELATED_TO')
                
                if (source_name in entity_map and 
                    target_name in entity_map and
                    rel_type in self.relationship_types):
                    relationship_data.append({
                        'source_name': source_name,
                        'target_name': target_name,
                        'type': rel_type
                    })
            
            # We'll create Relationship objects after entities are stored
            # For now, store the data in entity properties
            for rel in relationship_data:
                source = entity_map[rel['source_name']]
                if 'relationships' not in source.properties:
                    source.properties['relationships'] = []
                source.properties['relationships'].append(rel)
            
            return ExtractionResult(
                entities=entities,
                relationships=[],  # Will be created after entities are stored
                chunk_id=chunk_id,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            return ExtractionResult([], [], chunk_id, 0.0)
    
    def merge_extraction_results(
        self,
        results: List[ExtractionResult]
    ) -> Tuple[List[Entity], List[Dict[str, Any]]]:
        """
        Merge extraction results, deduplicating entities.
        
        Returns:
            Tuple of (merged entities, relationship data to be processed)
        """
        # Deduplicate entities by (name, type)
        entity_map = {}
        relationship_data = []
        
        for result in results:
            for entity in result.entities:
                key = (entity.name.lower(), entity.type)
                
                if key in entity_map:
                    # Merge with existing entity
                    existing = entity_map[key]
                    existing.chunk_ids.extend(entity.chunk_ids)
                    existing.chunk_ids = list(set(existing.chunk_ids))
                    
                    # Merge properties
                    if 'relationships' in entity.properties:
                        if 'relationships' not in existing.properties:
                            existing.properties['relationships'] = []
                        existing.properties['relationships'].extend(
                            entity.properties['relationships']
                        )
                else:
                    entity_map[key] = entity
            
            # Collect relationship data
            for entity in result.entities:
                if 'relationships' in entity.properties:
                    for rel in entity.properties['relationships']:
                        rel['chunk_id'] = result.chunk_id
                        relationship_data.append(rel)
        
        # Clean up relationship data from properties
        entities = list(entity_map.values())
        for entity in entities:
            entity.properties.pop('relationships', None)
        
        return entities, relationship_data
    
    async def build_relationships(
        self,
        relationship_data: List[Dict[str, Any]],
        entity_name_to_id: Dict[str, int]
    ) -> List[Relationship]:
        """
        Build Relationship objects from extracted data.
        
        Args:
            relationship_data: List of relationship data dicts
            entity_name_to_id: Mapping from entity names to database IDs
            
        Returns:
            List of Relationship objects
        """
        relationships = []
        seen = set()  # Deduplicate relationships
        
        for rel_data in relationship_data:
            source_name = rel_data['source_name'].lower()
            target_name = rel_data['target_name'].lower()
            rel_type = rel_data['type']
            chunk_id = rel_data.get('chunk_id', '')
            
            # Get entity IDs
            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)
            
            if source_id and target_id:
                # Create unique key for deduplication
                key = (source_id, target_id, rel_type)
                
                if key not in seen:
                    seen.add(key)
                    
                    relationship = Relationship(
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        type=rel_type,
                        chunk_ids=[chunk_id] if chunk_id else [],
                        confidence=0.9
                    )
                    relationships.append(relationship)
                else:
                    # Update existing relationship with new chunk
                    for rel in relationships:
                        if (rel.source_entity_id == source_id and
                            rel.target_entity_id == target_id and
                            rel.type == rel_type):
                            if chunk_id and chunk_id not in rel.chunk_ids:
                                rel.chunk_ids.append(chunk_id)
                            break
        
        return relationships


def create_entity_extractor() -> EntityExtractor:
    """Create an entity extractor instance."""
    return EntityExtractor()