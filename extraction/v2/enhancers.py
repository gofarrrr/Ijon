"""
Focused enhancers - each does ONE thing well.

Following 12-factor principle: small, focused micro-agents.
"""

from typing import List, Dict, Set, Optional
import re
from openai import AsyncOpenAI

from extraction.models import (
    ExtractedKnowledge, Fact, Question, Relationship,
    CognitiveLevel
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CitationEnhancer:
    """Enhance extraction with citation information - single responsibility."""
    
    @staticmethod
    def enhance(extraction: ExtractedKnowledge, source_text: str) -> ExtractedKnowledge:
        """
        Add citations to facts - pure function.
        
        Args:
            extraction: Current extraction
            source_text: Original document text
            
        Returns:
            Enhanced extraction with citations
        """
        # Find all citations in source
        citations = CitationEnhancer._extract_citations(source_text)
        
        # Match citations to facts
        enhanced_count = 0
        for fact in extraction.facts:
            # First try formal citations
            matched_citations = CitationEnhancer._match_citations(
                fact.claim, 
                citations, 
                source_text
            )
            
            if matched_citations and not fact.evidence:
                fact.evidence = f"Supported by: {', '.join(matched_citations[:3])}"
                enhanced_count += 1
            elif matched_citations and fact.evidence:
                fact.evidence += f" (Citations: {', '.join(matched_citations[:2])})"
                enhanced_count += 1
            
            # If no formal citations, find evidence in source text
            if not fact.evidence:
                evidence = CitationEnhancer._find_evidence(fact.claim, source_text)
                if evidence:
                    fact.evidence = evidence
                    enhanced_count += 1
            
            # Boost confidence for cited facts
            if fact.evidence:
                fact.confidence = min(1.0, fact.confidence * 1.15)
        
        logger.info(f"Enhanced {enhanced_count} facts with citations/evidence")
        return extraction
    
    @staticmethod
    def _extract_citations(text: str) -> List[str]:
        """Extract citations from text - pure function."""
        citations = []
        
        # Pattern 1: (Author, Year)
        pattern1 = r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)'
        citations.extend(re.findall(pattern1, text))
        
        # Pattern 2: [Number]
        pattern2 = r'\[\d+\]'
        citations.extend(re.findall(pattern2, text))
        
        return list(set(citations))  # Unique citations
    
    @staticmethod
    def _match_citations(claim: str, citations: List[str], source: str) -> List[str]:
        """Match citations to a claim - pure function."""
        matched = []
        
        # Find claim in source and look for nearby citations
        claim_words = claim.lower().split()[:5]  # First 5 words
        search_text = ' '.join(claim_words)
        
        if search_text in source.lower():
            # Find position and check citations within 100 chars
            pos = source.lower().index(search_text)
            nearby_text = source[max(0, pos-50):pos+200]
            
            for citation in citations:
                if citation in nearby_text:
                    matched.append(citation)
        
        return matched
    
    @staticmethod
    def _find_evidence(claim: str, source_text: str) -> Optional[str]:
        """Find evidence for a claim in source text."""
        # Get key terms from claim
        claim_lower = claim.lower()
        
        # Look for sentences containing key parts of the claim
        sentences = source_text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence contains key parts of the claim
            # Simple heuristic: check for significant overlap
            claim_words = set(word for word in claim_lower.split() if len(word) > 3)
            sentence_words = set(word for word in sentence_lower.split() if len(word) > 3)
            
            # If significant overlap, use as evidence
            overlap = len(claim_words & sentence_words)
            if overlap >= min(3, len(claim_words) // 2):
                # Clean up the sentence
                evidence = sentence.strip()
                if not evidence.endswith('.'):
                    evidence += '.'
                return f"Source: {evidence[:200]}"
        
        return None


class QuestionEnhancer:
    """Generate high-quality questions - single responsibility."""
    
    @staticmethod
    async def enhance(extraction: ExtractedKnowledge,
                     client: AsyncOpenAI,
                     target_levels: List[CognitiveLevel] = None) -> ExtractedKnowledge:
        """
        Add Bloom's taxonomy questions - pure async function.
        
        Args:
            extraction: Current extraction
            client: OpenAI client (passed in)
            target_levels: Desired cognitive levels
            
        Returns:
            Enhanced extraction with questions
        """
        if not target_levels:
            target_levels = [
                CognitiveLevel.UNDERSTAND,
                CognitiveLevel.APPLY,
                CognitiveLevel.ANALYZE
            ]
        
        # Generate questions for main topics
        new_questions = []
        
        for topic in extraction.topics[:3]:  # Top 3 topics
            for level in target_levels:
                question = await QuestionEnhancer._generate_question(
                    topic.name,
                    topic.description,
                    level,
                    client
                )
                if question:
                    new_questions.append(question)
        
        # Add to existing questions
        extraction.questions.extend(new_questions)
        
        # Remove duplicates
        seen = set()
        unique_questions = []
        for q in extraction.questions:
            if q.question_text not in seen:
                seen.add(q.question_text)
                unique_questions.append(q)
        
        extraction.questions = unique_questions
        
        logger.info(f"Added {len(new_questions)} questions")
        return extraction
    
    @staticmethod
    async def _generate_question(topic: str, 
                               description: str,
                               level: CognitiveLevel,
                               client: AsyncOpenAI) -> Question:
        """Generate single question - pure async function."""
        level_prompts = {
            CognitiveLevel.UNDERSTAND: "that tests comprehension",
            CognitiveLevel.APPLY: "that requires applying the concept",
            CognitiveLevel.ANALYZE: "that requires analysis and critical thinking",
            CognitiveLevel.EVALUATE: "that requires evaluation and judgment"
        }
        
        prompt = f"""Generate a {level.value} level question {level_prompts.get(level, '')} about:
Topic: {topic}
Description: {description}

Return JSON with:
- question_text: The question
- expected_answer: Brief answer
- difficulty: 1-5"""

        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            return Question(
                question_text=data["question_text"],
                expected_answer=data["expected_answer"],
                cognitive_level=level,
                difficulty=data.get("difficulty", 3),
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Failed to generate question: {str(e)}")
            return None


class RelationshipEnhancer:
    """Find and validate relationships - single responsibility."""
    
    @staticmethod
    def enhance(extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """
        Find relationships between entities - pure function.
        
        Args:
            extraction: Current extraction
            
        Returns:
            Enhanced extraction with relationships
        """
        # Extract all entities
        entities = RelationshipEnhancer._extract_entities(extraction)
        
        # Find missing relationships
        for entity1 in entities:
            for entity2 in entities:
                if entity1 != entity2:
                    if not RelationshipEnhancer._has_relationship(
                        extraction.relationships, entity1, entity2
                    ):
                        # Check if entities are related
                        rel = RelationshipEnhancer._infer_relationship(
                            entity1, entity2, extraction
                        )
                        if rel:
                            extraction.relationships.append(rel)
        
        logger.info(f"Total relationships: {len(extraction.relationships)}")
        return extraction
    
    @staticmethod
    def _extract_entities(extraction: ExtractedKnowledge) -> Set[str]:
        """Extract all entities - pure function."""
        entities = set()
        
        # From topics
        for topic in extraction.topics:
            entities.add(topic.name)
        
        # From existing relationships
        for rel in extraction.relationships:
            entities.add(rel.source_entity)
            entities.add(rel.target_entity)
        
        return entities
    
    @staticmethod
    def _has_relationship(relationships: List[Relationship], 
                         entity1: str, entity2: str) -> bool:
        """Check if relationship exists - pure function."""
        for rel in relationships:
            if (rel.source_entity == entity1 and rel.target_entity == entity2) or \
               (rel.source_entity == entity2 and rel.target_entity == entity1):
                return True
        return False
    
    @staticmethod
    def _infer_relationship(entity1: str, entity2: str,
                           extraction: ExtractedKnowledge) -> Relationship:
        """Infer relationship between entities - pure function."""
        # Simple inference based on co-occurrence in facts
        co_occurrences = 0
        
        for fact in extraction.facts:
            fact_text = fact.claim.lower()
            if entity1.lower() in fact_text and entity2.lower() in fact_text:
                co_occurrences += 1
        
        if co_occurrences >= 1:
            # Check for semantic relationships
            rel_type, description = RelationshipEnhancer._determine_relationship_type(
                entity1, entity2, extraction
            )
            
            return Relationship(
                source_entity=entity1,
                target_entity=entity2,
                relationship_type=rel_type,
                description=description,
                confidence=min(0.9, 0.4 + 0.2 * co_occurrences)
            )
        
        return None
    
    @staticmethod
    def _determine_relationship_type(entity1: str, entity2: str, 
                                   extraction: ExtractedKnowledge) -> tuple[str, str]:
        """Determine relationship type based on context."""
        # Simple heuristics
        entity1_lower = entity1.lower()
        entity2_lower = entity2.lower()
        
        # Check in facts for relationship hints
        for fact in extraction.facts:
            fact_lower = fact.claim.lower()
            if entity1_lower in fact_lower and entity2_lower in fact_lower:
                # Look for relationship indicators
                if any(word in fact_lower for word in ["uses", "applies", "leverages"]):
                    return "uses", f"{entity1} uses {entity2}"
                elif any(word in fact_lower for word in ["improves", "enhances", "reduces"]):
                    return "impacts", f"{entity1} impacts {entity2}"
                elif any(word in fact_lower for word in ["predicts", "analyzes", "detects"]):
                    return "analyzes", f"{entity1} analyzes aspects of {entity2}"
        
        return "related_to", f"{entity1} is related to {entity2}"


class SummaryEnhancer:
    """Improve summary quality - single responsibility."""
    
    @staticmethod
    async def enhance(extraction: ExtractedKnowledge,
                     client: AsyncOpenAI) -> ExtractedKnowledge:
        """
        Improve summary - pure async function.
        
        Args:
            extraction: Current extraction
            client: OpenAI client
            
        Returns:
            Enhanced extraction with better summary
        """
        # Build context from extraction
        context = SummaryEnhancer._build_context(extraction)
        
        prompt = f"""Improve this summary to better capture the key points:

Current summary: {extraction.summary or 'No summary yet'}

Key information:
{context}

Write a concise, comprehensive summary (2-3 sentences) that captures the main topics, key findings, and overall significance."""

        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=150
            )
            
            extraction.summary = response.choices[0].message.content.strip()
            logger.info("Summary enhanced")
            
        except Exception as e:
            logger.error(f"Failed to enhance summary: {str(e)}")
        
        return extraction
    
    @staticmethod
    def _build_context(extraction: ExtractedKnowledge) -> str:
        """Build context from extraction - pure function."""
        lines = []
        
        # Top topics
        lines.append("Main topics:")
        for topic in extraction.topics[:3]:
            lines.append(f"- {topic.name}")
        
        # Key facts
        lines.append("\nKey facts:")
        for fact in sorted(extraction.facts, key=lambda f: f.confidence, reverse=True)[:3]:
            lines.append(f"- {fact.claim[:80]}...")
        
        return "\n".join(lines)