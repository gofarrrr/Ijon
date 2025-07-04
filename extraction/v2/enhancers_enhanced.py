"""
Enhanced V2 enhancers with battle-tested prompt patterns.

Each enhancer uses focused agent loops and thinking blocks for quality.
"""

from typing import List, Dict, Set, Optional, Tuple
import re
import json
from openai import AsyncOpenAI

from extraction.models import (
    ExtractedKnowledge, Fact, Question, Relationship,
    CognitiveLevel
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancedCitationEnhancer:
    """Enhanced citation enhancer with agent-based validation."""
    
    CITATION_AGENT_PROMPT = """You are a Citation Validation Agent specializing in evidence verification.

## Validation Methodology
1. **Source Analysis**: Identify all citation patterns and references
2. **Claim Mapping**: Match citations to specific claims
3. **Evidence Strengthening**: Enhance facts with proper attribution
4. **Quality Assurance**: Ensure academic citation standards

## Thinking Process
<thinking>
For each fact, I will:
- Check for existing citations
- Find supporting evidence in source
- Validate citation accuracy
- Strengthen attribution
</thinking>

Focus on precision and academic integrity."""
    
    def __init__(self):
        self.citation_patterns = [
            r'\([A-Z][a-z]+(?:\s+et\s+al\.)?,\s*\d{4}\)',  # (Author, Year)
            r'\[[\d,\s]+\]',  # [1] or [1,2,3]
            r'[A-Z][a-z]+\s+\(\d{4}\)',  # Author (Year)
            r'\b(?:see|cf\.)\s+[A-Z][a-z]+',  # see Author
        ]
    
    @staticmethod
    def enhance(extraction: ExtractedKnowledge, source_text: str) -> ExtractedKnowledge:
        """
        Enhance extraction with validated citations.
        
        Args:
            extraction: Current extraction
            source_text: Original document text
            
        Returns:
            Enhanced extraction with citations
        """
        enhancer = EnhancedCitationEnhancer()
        
        # Extract all citations with context
        citations_with_context = enhancer._extract_citations_with_context(source_text)
        
        # Process each fact with enhanced validation
        enhanced_count = 0
        for fact in extraction.facts:
            # First check for formal citations
            matched_citations = enhancer._match_citations_intelligently(
                fact.claim, citations_with_context, source_text
            )
            
            if matched_citations:
                # Enhance with proper attribution
                if not fact.evidence:
                    fact.evidence = f"Supported by: {', '.join(matched_citations[:3])}"
                else:
                    fact.evidence = enhancer._integrate_citations(fact.evidence, matched_citations)
                enhanced_count += 1
            
            # Find contextual evidence if no citations
            if not fact.evidence or len(fact.evidence) < 20:
                evidence = enhancer._find_contextual_evidence(fact.claim, source_text)
                if evidence:
                    fact.evidence = evidence
                    enhanced_count += 1
            
            # Adjust confidence based on evidence quality
            fact.confidence = enhancer._calculate_evidence_confidence(fact)
        
        logger.info(f"Enhanced {enhanced_count} facts with citations/evidence")
        
        # Add citation metadata
        extraction.extraction_metadata["citation_enhancement"] = {
            "total_citations_found": len(citations_with_context),
            "facts_enhanced": enhanced_count,
            "citation_patterns_used": len(enhancer.citation_patterns)
        }
        
        return extraction
    
    def _extract_citations_with_context(self, text: str) -> List[Dict[str, any]]:
        """Extract citations with surrounding context."""
        citations = []
        
        for pattern in self.citation_patterns:
            for match in re.finditer(pattern, text):
                citation = match.group()
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                
                citations.append({
                    "citation": citation,
                    "position": match.start(),
                    "context": text[start:end],
                    "pattern_type": pattern
                })
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for c in citations:
            if c["citation"] not in seen:
                seen.add(c["citation"])
                unique_citations.append(c)
        
        return unique_citations
    
    def _match_citations_intelligently(self, claim: str, citations: List[Dict], 
                                     source: str) -> List[str]:
        """Match citations using intelligent proximity and semantic analysis."""
        matched = []
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        key_terms = [word for word in claim_lower.split() 
                    if len(word) > 4 and word not in ['that', 'this', 'with', 'from']]
        
        for citation_info in citations:
            context_lower = citation_info["context"].lower()
            
            # Check term overlap in citation context
            overlap_score = sum(1 for term in key_terms if term in context_lower)
            
            if overlap_score >= min(3, len(key_terms) // 2):
                matched.append(citation_info["citation"])
        
        return matched[:3]  # Return top 3 matches
    
    def _find_contextual_evidence(self, claim: str, source_text: str) -> Optional[str]:
        """Find evidence using intelligent context matching."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', source_text)
        claim_words = set(word.lower() for word in claim.split() if len(word) > 3)
        
        best_match = None
        best_score = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_words = set(word.lower() for word in sentence.split() if len(word) > 3)
            
            # Calculate overlap score
            overlap = len(claim_words & sentence_words)
            score = overlap / max(len(claim_words), 1)
            
            if score > best_score and score > 0.3:
                best_score = score
                # Include surrounding context
                context_sentences = []
                if i > 0:
                    context_sentences.append(sentences[i-1].strip())
                context_sentences.append(sentence)
                if i < len(sentences) - 1:
                    context_sentences.append(sentences[i+1].strip())
                
                best_match = " ".join(context_sentences)
        
        if best_match:
            # Clean and format
            best_match = re.sub(r'\s+', ' ', best_match).strip()
            if len(best_match) > 200:
                best_match = best_match[:197] + "..."
            return f"Context: {best_match}"
        
        return None
    
    def _integrate_citations(self, existing_evidence: str, citations: List[str]) -> str:
        """Intelligently integrate citations into existing evidence."""
        # Check if citations already present
        for citation in citations:
            if citation in existing_evidence:
                return existing_evidence
        
        # Add citations appropriately
        citation_str = ", ".join(citations[:2])
        if existing_evidence.endswith("."):
            return f"{existing_evidence[:-1]} ({citation_str})."
        else:
            return f"{existing_evidence} ({citation_str})"
    
    def _calculate_evidence_confidence(self, fact: Fact) -> float:
        """Calculate confidence based on evidence quality."""
        if not fact.evidence:
            return fact.confidence * 0.7
        
        evidence_len = len(fact.evidence)
        has_citation = bool(re.search(r'\([^)]+\d{4}\)|\[\d+\]', fact.evidence))
        
        # Base confidence
        confidence = fact.confidence
        
        # Boost for citations
        if has_citation:
            confidence = min(1.0, confidence * 1.2)
        
        # Boost for substantial evidence
        if evidence_len > 50:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence


class EnhancedQuestionEnhancer:
    """Enhanced question generator with cognitive depth analysis."""
    
    QUESTION_ENHANCEMENT_PROMPT = """You are a Cognitive Assessment Specialist enhancing questions for deep learning.

## Enhancement Framework
1. **Cognitive Mapping**: Ensure proper distribution across Bloom's taxonomy
2. **Depth Analysis**: Create questions that probe understanding
3. **Practical Relevance**: Connect to real-world applications
4. **Assessment Design**: Enable meaningful evaluation

## Thinking Process
<thinking>
For each topic, I will:
- Identify knowledge gaps in current questions
- Design questions for missing cognitive levels
- Ensure intellectual stimulation
- Validate question quality
</thinking>

Create questions that challenge and educate advanced learners."""
    
    @staticmethod
    async def enhance(extraction: ExtractedKnowledge,
                     client: AsyncOpenAI,
                     target_levels: List[CognitiveLevel] = None) -> ExtractedKnowledge:
        """
        Enhance with cognitively diverse questions.
        
        Args:
            extraction: Current extraction
            client: OpenAI client
            target_levels: Desired cognitive levels
            
        Returns:
            Enhanced extraction with better questions
        """
        if not target_levels:
            target_levels = list(CognitiveLevel)
        
        # Analyze current question distribution
        current_distribution = EnhancedQuestionEnhancer._analyze_distribution(
            extraction.questions
        )
        
        # Identify gaps
        gaps = EnhancedQuestionEnhancer._identify_gaps(
            current_distribution, target_levels
        )
        
        # Generate questions for gaps
        new_questions = []
        for level in gaps:
            # Select appropriate topics
            topics = extraction.topics[:3] if extraction.topics else []
            
            for topic in topics:
                question = await EnhancedQuestionEnhancer._generate_targeted_question(
                    topic, level, extraction, client
                )
                if question:
                    new_questions.append(question)
        
        # Enhance existing questions
        enhanced_questions = await EnhancedQuestionEnhancer._enhance_existing_questions(
            extraction.questions, client
        )
        
        # Combine and deduplicate
        all_questions = enhanced_questions + new_questions
        extraction.questions = EnhancedQuestionEnhancer._deduplicate_questions(all_questions)
        
        # Sort by cognitive level and difficulty
        extraction.questions.sort(
            key=lambda q: (list(CognitiveLevel).index(q.cognitive_level), q.difficulty)
        )
        
        logger.info(f"Enhanced to {len(extraction.questions)} total questions")
        return extraction
    
    @staticmethod
    def _analyze_distribution(questions: List[Question]) -> Dict[CognitiveLevel, int]:
        """Analyze current cognitive level distribution."""
        distribution = {level: 0 for level in CognitiveLevel}
        for q in questions:
            distribution[q.cognitive_level] += 1
        return distribution
    
    @staticmethod
    def _identify_gaps(current: Dict[CognitiveLevel, int], 
                      target_levels: List[CognitiveLevel]) -> List[CognitiveLevel]:
        """Identify which cognitive levels need more questions."""
        gaps = []
        
        # Minimum targets
        min_targets = {
            CognitiveLevel.REMEMBER: 1,
            CognitiveLevel.UNDERSTAND: 2,
            CognitiveLevel.APPLY: 1,
            CognitiveLevel.ANALYZE: 1,
            CognitiveLevel.EVALUATE: 1,
            CognitiveLevel.CREATE: 0  # Optional
        }
        
        for level in target_levels:
            if current.get(level, 0) < min_targets.get(level, 1):
                gaps.append(level)
        
        return gaps
    
    @staticmethod
    async def _generate_targeted_question(topic: any, level: CognitiveLevel,
                                        extraction: ExtractedKnowledge,
                                        client: AsyncOpenAI) -> Optional[Question]:
        """Generate question for specific topic and cognitive level."""
        
        # Build context
        context = f"Topic: {topic.name}\nDescription: {topic.description}\n"
        
        # Add relevant facts
        relevant_facts = [f for f in extraction.facts 
                         if topic.name.lower() in f.claim.lower()][:3]
        if relevant_facts:
            context += "\nRelated facts:\n"
            context += "\n".join([f"- {f.claim}" for f in relevant_facts])
        
        level_prompts = {
            CognitiveLevel.REMEMBER: "that tests precise factual recall",
            CognitiveLevel.UNDERSTAND: "that requires explaining the concept in their own words",
            CognitiveLevel.APPLY: "that requires applying this concept to a new scenario",
            CognitiveLevel.ANALYZE: "that requires breaking down and examining components",
            CognitiveLevel.EVALUATE: "that requires critical judgment and assessment",
            CognitiveLevel.CREATE: "that requires synthesizing new ideas or solutions"
        }
        
        prompt = f"""Generate ONE high-quality {level.value} level question {level_prompts[level]}.

Context:
{context}

Requirements:
1. Question must be clear and unambiguous
2. Should genuinely test the {level.value} cognitive level
3. Include a comprehensive expected answer (2-3 sentences)
4. Explain why this question is valuable
5. Assign appropriate difficulty (1-5)

Output as JSON with: question_text, expected_answer, rationale, difficulty"""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EnhancedQuestionEnhancer.QUESTION_ENHANCEMENT_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            return Question(
                question_text=data["question_text"],
                expected_answer=data["expected_answer"],
                cognitive_level=level,
                difficulty=data.get("difficulty", 3),
                confidence=0.85,
                metadata={"rationale": data.get("rationale", "")}
            )
            
        except Exception as e:
            logger.error(f"Failed to generate {level.value} question: {e}")
            return None
    
    @staticmethod
    async def _enhance_existing_questions(questions: List[Question], 
                                        client: AsyncOpenAI) -> List[Question]:
        """Enhance existing questions for better quality."""
        enhanced = []
        
        for q in questions:
            # Skip if already high quality
            if q.expected_answer and len(q.expected_answer) > 50:
                enhanced.append(q)
                continue
            
            try:
                prompt = f"""Enhance this question to be more intellectually stimulating.

Current question: {q.question_text}
Cognitive level: {q.cognitive_level.value}
Current answer: {q.expected_answer or "No answer provided"}

Improve by:
1. Making the question more thought-provoking
2. Providing a comprehensive answer (2-3 sentences)
3. Ensuring it truly tests the {q.cognitive_level.value} level
4. Adding a learning objective

Output as JSON with: question_text, expected_answer, learning_objective"""

                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
                
                data = json.loads(response.choices[0].message.content)
                
                # Create enhanced question
                enhanced_q = Question(
                    question_text=data.get("question_text", q.question_text),
                    expected_answer=data.get("expected_answer", q.expected_answer),
                    cognitive_level=q.cognitive_level,
                    difficulty=q.difficulty,
                    confidence=min(1.0, q.confidence * 1.1),
                    metadata={
                        **q.metadata,
                        "learning_objective": data.get("learning_objective", ""),
                        "enhanced": True
                    }
                )
                enhanced.append(enhanced_q)
                
            except Exception as e:
                logger.error(f"Failed to enhance question: {e}")
                enhanced.append(q)  # Keep original
        
        return enhanced
    
    @staticmethod
    def _deduplicate_questions(questions: List[Question]) -> List[Question]:
        """Remove duplicate questions intelligently."""
        seen = set()
        unique = []
        
        for q in questions:
            # Create normalized key
            key = q.question_text.lower().strip()
            key = re.sub(r'[^\w\s]', '', key)
            key = ' '.join(key.split())
            
            if key not in seen:
                seen.add(key)
                unique.append(q)
        
        return unique


class EnhancedRelationshipEnhancer:
    """Enhanced relationship discovery with semantic analysis."""
    
    RELATIONSHIP_ANALYSIS_PROMPT = """You are a Semantic Relationship Analyst discovering meaningful connections.

## Analysis Framework
1. **Entity Recognition**: Identify all significant entities
2. **Relationship Types**: Discover diverse connection types
3. **Semantic Validation**: Ensure relationships are meaningful
4. **Evidence Grounding**: Support with textual evidence

## Thinking Process
<thinking>
I will analyze entities and their interactions:
- What entities are present?
- How do they relate to each other?
- What evidence supports these relationships?
- Are there implicit connections?
</thinking>

Focus on discovering non-obvious but important relationships."""
    
    @staticmethod
    def enhance(extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """
        Enhance extraction with discovered relationships.
        
        Args:
            extraction: Current extraction
            
        Returns:
            Enhanced extraction with more relationships
        """
        enhancer = EnhancedRelationshipEnhancer()
        
        # Extract all entities
        entities = enhancer._extract_comprehensive_entities(extraction)
        
        # Build entity context map
        entity_contexts = enhancer._build_entity_contexts(extraction, entities)
        
        # Discover new relationships
        new_relationships = []
        
        for entity1 in entities:
            for entity2 in entities:
                if entity1 >= entity2:  # Avoid duplicates and self-relationships
                    continue
                
                # Check if relationship already exists
                if enhancer._relationship_exists(extraction.relationships, entity1, entity2):
                    continue
                
                # Analyze potential relationship
                rel = enhancer._analyze_relationship(
                    entity1, entity2, entity_contexts, extraction
                )
                
                if rel and rel.confidence > 0.5:
                    new_relationships.append(rel)
        
        # Add discovered relationships
        extraction.relationships.extend(new_relationships)
        
        # Enhance existing relationships
        for rel in extraction.relationships:
            if not rel.description or len(rel.description) < 10:
                rel.description = enhancer._generate_relationship_description(
                    rel, entity_contexts
                )
        
        logger.info(f"Added {len(new_relationships)} new relationships, "
                   f"total: {len(extraction.relationships)}")
        
        return extraction
    
    def _extract_comprehensive_entities(self, extraction: ExtractedKnowledge) -> Set[str]:
        """Extract all entities from various sources."""
        entities = set()
        
        # From topics
        for topic in extraction.topics:
            entities.add(topic.name)
            # Extract from description
            desc_entities = self._extract_entities_from_text(topic.description)
            entities.update(desc_entities)
        
        # From facts
        for fact in extraction.facts:
            fact_entities = self._extract_entities_from_text(fact.claim)
            entities.update(fact_entities)
        
        # From existing relationships
        for rel in extraction.relationships:
            entities.add(rel.source_entity)
            entities.add(rel.target_entity)
        
        # Clean and normalize
        entities = {e.strip() for e in entities if len(e.strip()) > 2}
        
        return entities
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract entities using pattern matching."""
        entities = set()
        
        # Capitalized phrases (likely entities)
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities.update(re.findall(cap_pattern, text))
        
        # Technical terms in quotes
        quote_pattern = r'"([^"]+)"|\'([^\']+)\''
        for match in re.findall(quote_pattern, text):
            entities.add(match[0] or match[1])
        
        return entities
    
    def _build_entity_contexts(self, extraction: ExtractedKnowledge, 
                             entities: Set[str]) -> Dict[str, List[str]]:
        """Build context for each entity."""
        contexts = {entity: [] for entity in entities}
        
        # From topics
        for topic in extraction.topics:
            if topic.name in entities:
                contexts[topic.name].append(topic.description)
        
        # From facts
        for fact in extraction.facts:
            for entity in entities:
                if entity.lower() in fact.claim.lower():
                    contexts[entity].append(fact.claim)
        
        return contexts
    
    def _relationship_exists(self, relationships: List[Relationship], 
                           entity1: str, entity2: str) -> bool:
        """Check if relationship already exists."""
        for rel in relationships:
            if (rel.source_entity == entity1 and rel.target_entity == entity2) or \
               (rel.source_entity == entity2 and rel.target_entity == entity1):
                return True
        return False
    
    def _analyze_relationship(self, entity1: str, entity2: str,
                            contexts: Dict[str, List[str]],
                            extraction: ExtractedKnowledge) -> Optional[Relationship]:
        """Analyze potential relationship between entities."""
        # Get contexts
        context1 = contexts.get(entity1, [])
        context2 = contexts.get(entity2, [])
        
        # Check co-occurrence in facts
        co_occurrences = []
        for fact in extraction.facts:
            fact_lower = fact.claim.lower()
            if entity1.lower() in fact_lower and entity2.lower() in fact_lower:
                co_occurrences.append(fact.claim)
        
        if not co_occurrences:
            return None
        
        # Determine relationship type and description
        rel_type, description, confidence = self._infer_relationship_details(
            entity1, entity2, co_occurrences, context1, context2
        )
        
        if rel_type:
            return Relationship(
                source_entity=entity1,
                target_entity=entity2,
                relationship_type=rel_type,
                description=description,
                confidence=confidence,
                metadata={"evidence": co_occurrences[0][:100]}
            )
        
        return None
    
    def _infer_relationship_details(self, entity1: str, entity2: str,
                                  co_occurrences: List[str],
                                  context1: List[str], 
                                  context2: List[str]) -> Tuple[str, str, float]:
        """Infer relationship type and description from context."""
        # Analyze first co-occurrence for relationship indicators
        text = co_occurrences[0].lower()
        
        # Relationship patterns
        patterns = {
            "enables": ["enables", "allows", "facilitates", "permits"],
            "uses": ["uses", "utilizes", "employs", "applies", "leverages"],
            "improves": ["improves", "enhances", "optimizes", "boosts"],
            "depends_on": ["depends on", "requires", "needs", "relies on"],
            "contrasts_with": ["contrasts", "differs", "opposes", "conflicts"],
            "complements": ["complements", "supports", "reinforces"],
            "implements": ["implements", "realizes", "executes"],
            "analyzes": ["analyzes", "examines", "evaluates", "studies"],
        }
        
        for rel_type, indicators in patterns.items():
            for indicator in indicators:
                if indicator in text:
                    # Generate description
                    description = self._generate_contextual_description(
                        entity1, entity2, rel_type, co_occurrences[0]
                    )
                    
                    # Calculate confidence
                    confidence = 0.6 + (0.1 * len(co_occurrences))
                    confidence = min(0.9, confidence)
                    
                    return rel_type, description, confidence
        
        # Default relationship
        if co_occurrences:
            return ("related_to", 
                   f"{entity1} is related to {entity2} in the context of the discussion",
                   0.5)
        
        return None, None, 0.0
    
    def _generate_contextual_description(self, entity1: str, entity2: str,
                                       rel_type: str, evidence: str) -> str:
        """Generate contextual relationship description."""
        templates = {
            "enables": f"{entity1} enables {entity2} by providing necessary capabilities",
            "uses": f"{entity1} uses {entity2} as part of its implementation",
            "improves": f"{entity1} improves {entity2} through enhancement mechanisms",
            "depends_on": f"{entity1} depends on {entity2} for its functionality",
            "contrasts_with": f"{entity1} contrasts with {entity2} in approach or implementation",
            "complements": f"{entity1} complements {entity2} to achieve better results",
            "implements": f"{entity1} implements the concepts defined by {entity2}",
            "analyzes": f"{entity1} analyzes aspects of {entity2} for insights",
        }
        
        base_description = templates.get(rel_type, f"{entity1} is related to {entity2}")
        
        # Try to extract more specific information from evidence
        if len(evidence) > 50:
            # Look for more specific relationship description in evidence
            sentences = evidence.split('.')
            for sentence in sentences:
                if entity1 in sentence and entity2 in sentence:
                    # Use this sentence as basis for description
                    return sentence.strip()
        
        return base_description
    
    def _generate_relationship_description(self, rel: Relationship,
                                         contexts: Dict[str, List[str]]) -> str:
        """Generate description for existing relationship."""
        # Get contexts
        source_context = contexts.get(rel.source_entity, [])
        target_context = contexts.get(rel.target_entity, [])
        
        # Build description based on relationship type
        if rel.relationship_type == "uses":
            return f"{rel.source_entity} utilizes {rel.target_entity} for enhanced functionality"
        elif rel.relationship_type == "enables":
            return f"{rel.source_entity} enables {rel.target_entity} through direct support"
        elif rel.relationship_type == "related_to":
            return f"{rel.source_entity} and {rel.target_entity} are conceptually connected"
        else:
            return f"{rel.source_entity} {rel.relationship_type.replace('_', ' ')} {rel.target_entity}"


class EnhancedSummaryEnhancer:
    """Enhanced summary generator with academic prose."""
    
    SUMMARY_ENHANCEMENT_PROMPT = """You are an Academic Summary Specialist creating scholarly syntheses.

## Summary Principles
1. **Comprehensive Coverage**: Capture all essential insights
2. **Academic Prose**: Write in flowing, scholarly style
3. **Conceptual Integration**: Connect ideas meaningfully
4. **Significance Highlighting**: Emphasize what matters

## Writing Process
<thinking>
I will synthesize:
- Main topics and their relationships
- Key findings and their implications
- Overall significance of the content
- Connections to broader contexts
</thinking>

Create summaries that inform and educate at an advanced level."""
    
    @staticmethod
    async def enhance(extraction: ExtractedKnowledge,
                     client: AsyncOpenAI) -> ExtractedKnowledge:
        """
        Enhance summary with academic quality.
        
        Args:
            extraction: Current extraction
            client: OpenAI client
            
        Returns:
            Enhanced extraction with better summary
        """
        # Build comprehensive context
        context = EnhancedSummaryEnhancer._build_rich_context(extraction)
        
        prompt = f"""Create a scholarly summary that synthesizes this knowledge.

Extracted Knowledge:
{context}

Requirements:
1. Write 3-4 sentences of flowing academic prose
2. NO bullet points or lists - continuous narrative only
3. Connect major themes coherently
4. Highlight significance and implications
5. Use precise, scholarly language
6. Ensure conceptual completeness

The summary should read like a paragraph from an academic paper, providing both synthesis and insight."""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": EnhancedSummaryEnhancer.SUMMARY_ENHANCEMENT_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            
            new_summary = response.choices[0].message.content.strip()
            
            # Validate summary quality
            if EnhancedSummaryEnhancer._validate_summary(new_summary):
                extraction.summary = new_summary
                logger.info("Summary enhanced with academic prose")
            else:
                logger.warning("Generated summary did not meet quality standards")
            
        except Exception as e:
            logger.error(f"Failed to enhance summary: {e}")
        
        return extraction
    
    @staticmethod
    def _build_rich_context(extraction: ExtractedKnowledge) -> str:
        """Build rich context from extraction."""
        lines = []
        
        # Main topics with descriptions
        if extraction.topics:
            lines.append("Primary Topics:")
            for i, topic in enumerate(extraction.topics[:4], 1):
                lines.append(f"{i}. {topic.name}: {topic.description}")
        
        # Key facts with high confidence
        high_conf_facts = [f for f in extraction.facts if f.confidence > 0.7]
        if high_conf_facts:
            lines.append("\nKey Findings:")
            for fact in high_conf_facts[:4]:
                lines.append(f"- {fact.claim}")
        
        # Important relationships
        if extraction.relationships:
            lines.append("\nSignificant Relationships:")
            for rel in extraction.relationships[:3]:
                lines.append(f"- {rel.source_entity} {rel.relationship_type} "
                           f"{rel.target_entity}")
        
        # Current summary if exists
        if extraction.summary:
            lines.append(f"\nCurrent Summary: {extraction.summary}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _validate_summary(summary: str) -> bool:
        """Validate summary meets quality standards."""
        # Check length
        if len(summary) < 100 or len(summary) > 400:
            return False
        
        # Check for bullet points or lists
        if any(marker in summary for marker in ['•', '- ', '* ', '1.', '2.']):
            return False
        
        # Check for multiple sentences
        sentences = summary.count('.') + summary.count('!') + summary.count('?')
        if sentences < 2:
            return False
        
        # Check for academic language indicators
        academic_terms = ['demonstrates', 'indicates', 'suggests', 'reveals',
                         'encompasses', 'constitutes', 'represents', 'illustrates']
        if not any(term in summary.lower() for term in academic_terms):
            return False
        
        return True


# Convenience function to get enhanced enhancers
def get_enhanced_enhancers() -> Dict[str, any]:
    """Get instances of all enhanced enhancers."""
    return {
        "citation": EnhancedCitationEnhancer(),
        "question": EnhancedQuestionEnhancer(),
        "relationship": EnhancedRelationshipEnhancer(),
        "summary": EnhancedSummaryEnhancer()
    }