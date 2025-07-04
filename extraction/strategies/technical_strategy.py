"""
Technical document extraction strategy.

Optimized for technical manuals, API documentation, and how-to guides.
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


class TechnicalStrategy(ExtractionStrategy):
    """Extraction strategy optimized for technical documents."""
    
    def build_extraction_prompt(self, content: str, profile: DocumentProfile) -> str:
        """Build prompt optimized for technical content."""
        
        return f"""Extract structured knowledge from this technical documentation. Focus on procedures, specifications, and practical information.

Text to analyze:
{content}

Extract the following with attention to technical details:

1. TOPICS: Technical concepts, features, and components
   - name: Feature/concept name (use exact technical terms)
   - description: Technical description with specifications
   - keywords: Include version numbers, technical specs, related technologies
   - confidence: Your confidence (0.0-1.0)

2. FACTS: Technical specifications, requirements, and procedures
   - claim: The technical fact or specification
   - evidence: Exact values, commands, or configuration details
   - confidence: Your confidence (0.0-1.0)
   
   Focus on:
   - System requirements
   - Configuration parameters
   - Command syntax
   - Error codes and solutions
   - Performance specifications
   - Best practices and warnings

3. RELATIONSHIPS: Technical dependencies and interactions
   - source_entity: Component/feature/system
   - target_entity: Related component/feature/system
   - relationship_type: Use technical relationships like "depends on", "configures", "implements", "requires", "incompatible with"
   - description: Technical context
   - confidence: Your confidence (0.0-1.0)

4. QUESTIONS: Generate practical, implementation-focused questions
   - question_text: Focus on "how to", troubleshooting, and configuration
   - expected_answer: Specific technical answer
   - cognitive_level: Mix of "understand", "apply", and "analyze"
   - difficulty: 1-5 (based on technical complexity)
   - confidence: Your confidence (0.0-1.0)
   
   Include questions about:
   - Implementation steps
   - Common errors and solutions
   - Configuration options
   - Performance optimization
   - Compatibility issues

5. SUMMARY: Technical summary with key specifications and procedures

6. OVERALL_CONFIDENCE: Your confidence in the extraction quality (0.0-1.0)

Additionally, extract these technical-specific elements if present:
- code_snippets: Any code examples (preserve formatting)
- commands: Specific commands or CLI syntax
- parameters: Configuration parameters with types and defaults
- warnings: Important warnings or cautions

Respond with a JSON object. Preserve exact technical terminology and values."""

    def post_process_extraction(self, 
                              extraction: ExtractedKnowledge, 
                              profile: DocumentProfile) -> ExtractedKnowledge:
        """Enhance extraction with technical-specific processing."""
        
        # Process topics for technical categorization
        for topic in extraction.topics:
            # Identify configuration topics
            if any(term in topic.name.lower() for term in ["config", "setting", "parameter", "option"]):
                topic.keywords.append("configuration")
            
            # Identify API/interface topics
            if any(term in topic.name.lower() for term in ["api", "endpoint", "interface", "method"]):
                topic.keywords.append("api")
            
            # Identify installation/setup topics
            if any(term in topic.name.lower() for term in ["install", "setup", "deploy", "initialize"]):
                topic.keywords.append("setup")
        
        # Enhance facts with technical markers
        code_pattern = r'`[^`]+`|```[\s\S]*?```'
        command_pattern = r'^\$\s*\w+|^>\s*\w+|^#\s*\w+'
        
        for fact in extraction.facts:
            # Mark facts containing code
            if re.search(code_pattern, fact.claim):
                if not fact.evidence:
                    fact.evidence = "[Contains Code]"
                fact.topics = fact.topics or []
                fact.topics.append("code_example")
            
            # Mark command-line facts
            if re.search(command_pattern, fact.claim, re.MULTILINE):
                if not fact.evidence:
                    fact.evidence = "[Command Line]"
                fact.topics = fact.topics or []
                fact.topics.append("cli_command")
            
            # Boost confidence for facts with specific values
            if re.search(r'\d+\.\d+|\d+[KMG]B|v\d+\.\d+', fact.claim):
                fact.confidence = min(1.0, fact.confidence * 1.1)
        
        # Enhance relationships for technical context
        technical_relationships = {
            "depends on": "has dependency on",
            "configures": "provides configuration for",
            "implements": "is an implementation of",
            "requires": "has requirement for",
            "extends": "extends functionality of"
        }
        
        for rel in extraction.relationships:
            if rel.relationship_type in technical_relationships and not rel.description:
                rel.description = f"{rel.source_entity} {technical_relationships[rel.relationship_type]} {rel.target_entity}"
        
        # Adjust questions for technical focus
        for question in extraction.questions:
            # Technical questions should be practical
            if question.cognitive_level == CognitiveLevel.REMEMBER:
                # Upgrade to application level for how-to questions
                if any(phrase in question.question_text.lower() for phrase in ["how to", "how do", "steps to"]):
                    question.cognitive_level = CognitiveLevel.APPLY
                    question.difficulty = max(2, question.difficulty)
        
        # Add technical metadata
        extraction.extraction_metadata["technical_elements"] = {
            "has_code": any(re.search(code_pattern, f.claim) for f in extraction.facts),
            "has_commands": any(re.search(command_pattern, f.claim, re.MULTILINE) for f in extraction.facts),
            "config_topics": sum(1 for t in extraction.topics if "configuration" in t.keywords),
            "api_topics": sum(1 for t in extraction.topics if "api" in t.keywords),
            "setup_topics": sum(1 for t in extraction.topics if "setup" in t.keywords)
        }
        
        # Boost confidence for technical documents with clear specifications
        if extraction.extraction_metadata["technical_elements"]["has_code"]:
            extraction.overall_confidence = min(1.0, extraction.overall_confidence * 1.05)
        
        logger.info(f"Technical post-processing complete. Found "
                   f"{extraction.extraction_metadata['technical_elements']['config_topics']} config topics")
        
        return extraction
    
    def get_extraction_parameters(self, profile: DocumentProfile) -> Dict[str, Any]:
        """Get parameters optimized for technical extraction."""
        params = super().get_extraction_parameters(profile)
        
        # Technical docs need precise extraction
        params["temperature"] = 0.2  # Lower temperature for accuracy
        
        # May need more tokens for code examples
        params["max_tokens"] = 2500
        
        return params