"""
Academic document extraction strategy.

Optimized for research papers, journal articles, and academic texts.
"""

from typing import List, Dict, Any
import re

from extraction.models import (
    ExtractedKnowledge, DocumentProfile, Topic, Fact, 
    Relationship, Question, CognitiveLevel
)
from extraction.strategies.base_strategy import ExtractionStrategy
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AcademicStrategy(ExtractionStrategy):
    """Extraction strategy optimized for academic documents."""
    
    def build_extraction_prompt(self, content: str, profile: DocumentProfile) -> str:
        """Build prompt optimized for academic content."""
        
        return f"""Extract structured knowledge from this academic text. Focus on research contributions, methodology, and findings.

Text to analyze:
{content}

Extract the following information with special attention to academic elements:

1. TOPICS: Main research topics and concepts
   - name: Topic name (use academic terminology)
   - description: Detailed academic description
   - keywords: Include technical terms and field-specific vocabulary
   - confidence: Your confidence (0.0-1.0)

2. FACTS: Key findings, claims, and contributions
   - claim: The academic claim or finding
   - evidence: Supporting evidence, data, or citations
   - confidence: Your confidence (0.0-1.0)
   
   Pay special attention to:
   - Research hypotheses
   - Experimental results
   - Statistical findings (include p-values, confidence intervals)
   - Novel contributions to the field

3. RELATIONSHIPS: How concepts, theories, and findings relate
   - source_entity: First concept/theory/method
   - target_entity: Second concept/theory/method
   - relationship_type: Use academic relationships like "supports", "contradicts", "extends", "applies", "validates"
   - description: Academic context of the relationship
   - confidence: Your confidence (0.0-1.0)

4. QUESTIONS: Generate research-oriented questions
   - question_text: The question (focus on methodology, implications, limitations)
   - expected_answer: Based on the text
   - cognitive_level: Use higher levels - "analyze", "evaluate", "create"
   - difficulty: 1-5 (lean towards 3-5 for academic content)
   - confidence: Your confidence (0.0-1.0)
   
   Include questions about:
   - Research methodology
   - Validity of conclusions
   - Future research directions
   - Theoretical implications

5. SUMMARY: Academic summary focusing on contributions and methodology

6. OVERALL_CONFIDENCE: Your confidence in the extraction quality (0.0-1.0)

Additionally, extract these academic-specific elements if present:
- methodology: Research methods used
- limitations: Stated limitations or caveats
- future_work: Suggested future research

Respond with a JSON object containing all fields. Be precise with academic terminology."""

    def post_process_extraction(self, 
                              extraction: ExtractedKnowledge, 
                              profile: DocumentProfile) -> ExtractedKnowledge:
        """Enhance extraction with academic-specific processing."""
        
        # Enhance topics with academic categorization
        for topic in extraction.topics:
            # Add academic field indicators
            if any(term in topic.name.lower() for term in ["hypothesis", "theory", "model"]):
                topic.keywords.append("theoretical")
            if any(term in topic.name.lower() for term in ["experiment", "study", "analysis"]):
                topic.keywords.append("empirical")
        
        # Enhance facts with citation detection
        for fact in extraction.facts:
            # Look for citation patterns
            if re.search(r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)', fact.claim):
                fact.confidence = min(1.0, fact.confidence * 1.1)  # Boost confidence for cited facts
            
            # Mark statistical findings
            if re.search(r'p\s*[<>=]\s*0\.\d+|CI:|confidence interval', fact.claim, re.IGNORECASE):
                if "statistical_finding" not in str(fact.evidence):
                    fact.evidence = f"[Statistical Finding] {fact.evidence or fact.claim}"
        
        # Enhance relationships for academic context
        academic_relationship_types = {
            "supports": "provides empirical support for",
            "contradicts": "challenges the findings of",
            "extends": "builds upon the work of",
            "applies": "applies the methodology of",
            "validates": "independently validates"
        }
        
        for rel in extraction.relationships:
            # Enhance relationship descriptions
            if rel.relationship_type in academic_relationship_types:
                if not rel.description:
                    rel.description = f"{rel.source_entity} {academic_relationship_types[rel.relationship_type]} {rel.target_entity}"
        
        # Ensure questions are academically rigorous
        for question in extraction.questions:
            # Upgrade cognitive levels for academic content
            if question.cognitive_level in [CognitiveLevel.REMEMBER, CognitiveLevel.UNDERSTAND]:
                # Only basic questions should be about definitions
                if "define" not in question.question_text.lower():
                    question.cognitive_level = CognitiveLevel.ANALYZE
                    question.difficulty = max(3, question.difficulty)
        
        # Add metadata about academic elements found
        extraction.extraction_metadata["academic_elements"] = {
            "has_citations": any(re.search(r'\[\d+\]|\([A-Z][a-z]+.*\d{4}\)', f.claim) for f in extraction.facts),
            "has_statistics": any(re.search(r'p\s*[<>=]\s*0\.\d+|n\s*=\s*\d+', f.claim) for f in extraction.facts),
            "theoretical_topics": sum(1 for t in extraction.topics if "theoretical" in t.keywords),
            "empirical_topics": sum(1 for t in extraction.topics if "empirical" in t.keywords)
        }
        
        # Adjust overall confidence based on academic indicators
        if extraction.extraction_metadata["academic_elements"]["has_citations"]:
            extraction.overall_confidence = min(1.0, extraction.overall_confidence * 1.05)
        
        logger.info(f"Academic post-processing complete. Found {len(extraction.facts)} facts with "
                   f"{extraction.extraction_metadata['academic_elements']['has_citations']} citations")
        
        return extraction
    
    def get_extraction_parameters(self, profile: DocumentProfile) -> Dict[str, Any]:
        """Get parameters optimized for academic extraction."""
        params = super().get_extraction_parameters(profile)
        
        # Academic papers often need more tokens for complex content
        params["max_tokens"] = 2500
        
        # Slightly higher temperature for nuanced academic concepts
        params["temperature"] = 0.4
        
        return params
    
    def should_validate_facts(self, profile: DocumentProfile) -> bool:
        """Academic facts should always be validated if possible."""
        return True  # Always validate academic claims