"""
Narrative document extraction strategy.

Optimized for business books, historical texts, and narrative content.
"""

from typing import Dict, Any
import re

from extraction.models import (
    ExtractedKnowledge, DocumentProfile, Topic, Fact, 
    Relationship, Question, CognitiveLevel
)
from extraction.strategies.base_strategy import ExtractionStrategy
from src.utils.logging import get_logger

logger = get_logger(__name__)


class NarrativeStrategy(ExtractionStrategy):
    """Extraction strategy optimized for narrative documents."""
    
    def build_extraction_prompt(self, content: str, profile: DocumentProfile) -> str:
        """Build prompt optimized for narrative content."""
        
        return f"""Extract structured knowledge from this narrative text. Focus on key concepts, lessons, and insights.

Text to analyze:
{content}

Extract the following with attention to narrative elements:

1. TOPICS: Main themes, concepts, and ideas
   - name: Core concept or theme
   - description: Explain the concept in context of the narrative
   - keywords: Include related themes, metaphors, or examples used
   - confidence: Your confidence (0.0-1.0)

2. FACTS: Key insights, principles, and lessons
   - claim: The main point, lesson, or principle
   - evidence: Supporting examples, anecdotes, or reasoning from the text
   - confidence: Your confidence (0.0-1.0)
   
   Focus on:
   - Core principles or rules presented
   - Lessons learned from examples
   - Insights from case studies or stories
   - Author's main arguments
   - Historical facts or business strategies

3. RELATIONSHIPS: How concepts and ideas connect
   - source_entity: Concept/person/organization
   - target_entity: Related concept/person/organization
   - relationship_type: Use narrative relationships like "illustrates", "leads to", "contrasts with", "exemplifies", "influences"
   - description: Context of the relationship in the narrative
   - confidence: Your confidence (0.0-1.0)

4. QUESTIONS: Generate thought-provoking questions
   - question_text: Focus on understanding, application, and critical thinking
   - expected_answer: Based on the text's perspective
   - cognitive_level: Emphasize "understand", "apply", "analyze", and "evaluate"
   - difficulty: 2-5 (based on conceptual depth)
   - confidence: Your confidence (0.0-1.0)
   
   Include questions about:
   - Main arguments and their validity
   - How to apply concepts in practice
   - Implications of the ideas presented
   - Comparisons with other approaches
   - Critical evaluation of claims

5. SUMMARY: Narrative summary capturing the main message and key takeaways

6. OVERALL_CONFIDENCE: Your confidence in the extraction quality (0.0-1.0)

Additionally, extract these narrative-specific elements if present:
- examples: Key examples or case studies mentioned
- quotes: Important quotes or memorable phrases
- timeline: Chronological elements if present

Respond with a JSON object. Capture the essence of the narrative while maintaining factual accuracy."""

    def post_process_extraction(self, 
                              extraction: ExtractedKnowledge, 
                              profile: DocumentProfile) -> ExtractedKnowledge:
        """Enhance extraction with narrative-specific processing."""
        
        # Enhance topics with narrative categorization
        for topic in extraction.topics:
            # Identify conceptual topics
            if any(term in topic.description.lower() for term in ["principle", "concept", "idea", "theory"]):
                topic.keywords.append("conceptual")
            
            # Identify practical topics
            if any(term in topic.description.lower() for term in ["strategy", "approach", "method", "technique"]):
                topic.keywords.append("practical")
            
            # Identify example-based topics
            if any(term in topic.description.lower() for term in ["case", "example", "story", "anecdote"]):
                topic.keywords.append("example_based")
        
        # Process facts for narrative elements
        for fact in extraction.facts:
            # Identify facts from examples
            if any(indicator in fact.claim.lower() for indicator in ["for example", "for instance", "case in point"]):
                if fact.evidence:
                    fact.evidence = f"[From Example] {fact.evidence}"
                else:
                    fact.evidence = "[Derived from example in text]"
            
            # Identify quoted facts
            if re.search(r'"[^"]{20,}"', fact.claim):
                fact.confidence = min(1.0, fact.confidence * 1.1)  # Quotes are usually accurate
                if not fact.evidence:
                    fact.evidence = "[Direct quote from text]"
            
            # Mark business/historical facts
            if re.search(r'\$[\d,]+|%\s*\w+|\d{4}s?', fact.claim):
                fact.topics = fact.topics or []
                if "$" in fact.claim:
                    fact.topics.append("business_metric")
                if re.search(r'\d{4}s?', fact.claim):
                    fact.topics.append("historical_reference")
        
        # Enhance relationships for narrative flow
        narrative_relationships = {
            "illustrates": "provides an example of",
            "leads to": "results in",
            "contrasts with": "shows the opposite of",
            "exemplifies": "is a perfect example of",
            "influences": "has an impact on"
        }
        
        for rel in extraction.relationships:
            if rel.relationship_type in narrative_relationships and not rel.description:
                rel.description = f"In this narrative, {rel.source_entity} {narrative_relationships[rel.relationship_type]} {rel.target_entity}"
        
        # Adjust questions for narrative content
        for question in extraction.questions:
            # Narrative questions should encourage deeper thinking
            if question.cognitive_level in [CognitiveLevel.REMEMBER, CognitiveLevel.UNDERSTAND]:
                # Upgrade questions about implications or applications
                if any(word in question.question_text.lower() for word in ["why", "how", "implications", "apply"]):
                    question.cognitive_level = CognitiveLevel.ANALYZE
                    question.difficulty = max(3, question.difficulty)
        
        # Add narrative metadata
        extraction.extraction_metadata["narrative_elements"] = {
            "has_quotes": any(re.search(r'"[^"]{20,}"', f.claim) for f in extraction.facts),
            "has_examples": any("example" in f.evidence.lower() if f.evidence else False for f in extraction.facts),
            "conceptual_topics": sum(1 for t in extraction.topics if "conceptual" in t.keywords),
            "practical_topics": sum(1 for t in extraction.topics if "practical" in t.keywords),
            "example_based_topics": sum(1 for t in extraction.topics if "example_based" in t.keywords)
        }
        
        # Adjust confidence based on narrative clarity
        if extraction.extraction_metadata["narrative_elements"]["has_examples"]:
            extraction.overall_confidence = min(1.0, extraction.overall_confidence * 1.05)
        
        logger.info(f"Narrative post-processing complete. Found "
                   f"{extraction.extraction_metadata['narrative_elements']['example_based_topics']} example-based topics")
        
        return extraction
    
    def get_extraction_parameters(self, profile: DocumentProfile) -> Dict[str, Any]:
        """Get parameters optimized for narrative extraction."""
        params = super().get_extraction_parameters(profile)
        
        # Narrative needs more nuanced understanding
        params["temperature"] = 0.5  # Slightly higher for conceptual extraction
        
        # Narratives can have longer explanations
        params["max_tokens"] = 2000
        
        return params
    
    def should_validate_facts(self, profile: DocumentProfile) -> bool:
        """Narrative facts benefit from validation, especially claims."""
        # Validate more for business/historical narratives
        return profile.document_type in ["business", "historical"] or super().should_validate_facts(profile)