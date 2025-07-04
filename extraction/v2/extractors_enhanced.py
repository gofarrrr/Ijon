"""
Enhanced V2 extractors with battle-tested prompt patterns.

Incorporates agent loop architecture, thinking blocks, and academic prose.
"""

from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
import json

from extraction.models import ExtractedKnowledge, Topic, Fact, Relationship, Question, CognitiveLevel
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Enhanced system prompt with agent loop architecture
ENHANCED_EXTRACTION_SYSTEM_PROMPT = """You are a specialized Knowledge Extraction Agent operating with advanced cognitive capabilities.

## Agent Loop Architecture
You follow a systematic extraction methodology:
1. **Analyze**: Understand content structure and identify key patterns
2. **Plan**: Develop extraction strategy based on content type
3. **Execute**: Apply focused extraction techniques
4. **Validate**: Ensure quality and completeness
5. **Iterate**: Refine based on confidence assessment

## Cognitive Processing
Use structured thinking to ensure high-quality extraction:
- Analyze content deeply before extracting
- Validate claims against source material
- Ensure logical consistency across extractions
- Generate insights beyond surface-level information

## Academic Standards
Maintain scholarly rigor throughout:
- Write in flowing academic prose
- Support all claims with evidence
- Ensure intellectual coherence
- Focus on quality over quantity

Remember: You are extracting knowledge for advanced reasoning systems. Precision and depth matter."""


class EnhancedBaseExtractor:
    """Enhanced base extractor with battle-tested patterns."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = "gpt-4o-mini"
        self.temperature = 0.3
        
    async def extract(self, content: str, focus: str = None) -> ExtractedKnowledge:
        """
        Extract knowledge using enhanced methodology.
        
        Args:
            content: Text to extract from
            focus: Optional extraction focus
            
        Returns:
            ExtractedKnowledge with high-quality extractions
        """
        prompt = self._build_enhanced_prompt(content, focus)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": ENHANCED_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            return self._parse_enhanced_response(data)
            
        except Exception as e:
            logger.error(f"Enhanced extraction failed: {e}")
            return ExtractedKnowledge(
                topics=[], facts=[], relationships=[], questions=[],
                overall_confidence=0.0, summary="Extraction failed"
            )
    
    def _build_enhanced_prompt(self, content: str, focus: Optional[str]) -> str:
        """Build enhanced extraction prompt with thinking blocks."""
        
        focus_instruction = f"\nExtraction Focus: {focus}" if focus else ""
        
        return f"""Extract comprehensive knowledge from the following content using systematic analysis.

{focus_instruction}

Content to analyze:
{content}

## Extraction Process

### Phase 1: Content Analysis
<thinking>
Analyze the content structure and identify:
- Content type and domain
- Key themes and concepts
- Information density
- Extraction challenges
</thinking>

### Phase 2: Strategic Extraction

#### Topics (Main Concepts)
Extract 3-5 primary topics with:
- **name**: Precise, meaningful identifier
- **description**: 2-3 sentences of flowing academic prose explaining the concept's significance and context
- **keywords**: 3-5 most relevant terms
- **confidence**: Your genuine assessment (0.0-1.0)

#### Facts (Verifiable Claims)
Extract key facts with:
- **claim**: Specific, verifiable statement
- **evidence**: Direct support from the text
- **context**: Why this fact matters
- **confidence**: Based on evidence strength

#### Relationships (Conceptual Connections)
Identify meaningful relationships:
- **source_entity**: First concept (specific)
- **target_entity**: Second concept (specific)
- **relationship_type**: Precise type (e.g., "enables", "contradicts", "depends_on")
- **description**: One sentence explaining the connection
- **evidence**: Textual support
- **confidence**: Certainty level

#### Questions (Cognitive Engagement)
Generate 6-8 questions distributed across Bloom's taxonomy:
- **question_text**: Clear, thought-provoking question
- **expected_answer**: Comprehensive answer based on the text
- **cognitive_level**: [remember, understand, apply, analyze, evaluate, create]
- **difficulty**: 1-5 scale
- **rationale**: Why this question adds value
- **confidence**: Quality assessment

Ensure cognitive level distribution:
- 1-2 "remember" (factual recall)
- 2 "understand" (comprehension)
- 1-2 "apply" (practical application)
- 1-2 "analyze" (breaking down concepts)
- 1 "evaluate" or "create" (if content supports)

### Phase 3: Synthesis
**Summary**: Write 3-4 sentences of flowing academic prose that synthesizes the key insights, connects major themes, and highlights the content's significance. Avoid bullet points or lists.

### Phase 4: Quality Reflection
<thinking>
Assess your extraction:
- Have I captured the essential knowledge?
- Are all claims well-supported?
- Do relationships reveal meaningful connections?
- Are questions intellectually stimulating?
- Overall quality score: [0.0-1.0 with justification]
</thinking>

## Output Format
Provide a JSON object with:
- topics: Array of topic objects
- facts: Array of fact objects
- relationships: Array of relationship objects
- questions: Array of question objects
- summary: String (academic prose)
- overall_confidence: Float (0.0-1.0)
- extraction_metadata: Object with process details
- quality_assessment: Your reflection on extraction quality

Focus on depth and precision. Extract only well-supported, meaningful information."""
    
    def _parse_enhanced_response(self, data: Dict[str, Any]) -> ExtractedKnowledge:
        """Parse enhanced response with quality validation."""
        
        # Parse topics with academic descriptions
        topics = []
        for t in data.get("topics", []):
            topics.append(Topic(
                name=t["name"],
                description=t["description"],
                keywords=t.get("keywords", []),
                confidence=float(t.get("confidence", 0.7))
            ))
        
        # Parse facts with context
        facts = []
        for f in data.get("facts", []):
            facts.append(Fact(
                claim=f["claim"],
                evidence=f.get("evidence"),
                confidence=float(f.get("confidence", 0.7)),
                metadata={"context": f.get("context", "")}
            ))
        
        # Parse relationships with evidence
        relationships = []
        for r in data.get("relationships", []):
            relationships.append(Relationship(
                source_entity=r["source_entity"],
                target_entity=r["target_entity"],
                relationship_type=r["relationship_type"],
                description=r.get("description"),
                confidence=float(r.get("confidence", 0.7)),
                metadata={"evidence": r.get("evidence", "")}
            ))
        
        # Parse questions with rationale
        questions = []
        for q in data.get("questions", []):
            questions.append(Question(
                question_text=q["question_text"],
                expected_answer=q.get("expected_answer"),
                cognitive_level=CognitiveLevel(q.get("cognitive_level", "understand")),
                difficulty=int(q.get("difficulty", 3)),
                confidence=float(q.get("confidence", 0.7)),
                metadata={"rationale": q.get("rationale", "")}
            ))
        
        # Build metadata
        metadata = data.get("extraction_metadata", {})
        metadata["quality_assessment"] = data.get("quality_assessment", {})
        metadata["enhanced"] = True
        
        return ExtractedKnowledge(
            topics=topics,
            facts=facts,
            relationships=relationships,
            questions=questions,
            summary=data.get("summary", ""),
            overall_confidence=float(data.get("overall_confidence", 0.7)),
            extraction_metadata=metadata
        )


class EnhancedTopicExtractor:
    """Enhanced topic-focused extractor with deep analysis."""
    
    ENHANCED_TOPIC_PROMPT = """You are a Topic Analysis Specialist employing advanced extraction techniques.

## Analytical Framework
Your approach combines:
1. **Conceptual Mapping**: Identify core concepts and their boundaries
2. **Hierarchical Analysis**: Understand topic relationships and dependencies
3. **Contextual Grounding**: Ensure topics are well-supported by evidence
4. **Semantic Clustering**: Group related concepts effectively

## Extraction Methodology
<thinking>
Before extracting topics, I will:
- Identify the domain and discourse level
- Map conceptual boundaries
- Assess information density
- Plan extraction strategy
</thinking>

Extract topics with scholarly depth and precision."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = "gpt-4o-mini"
        
    async def extract(self, content: str) -> List[Topic]:
        """Extract topics with enhanced methodology."""
        
        prompt = f"""Perform deep topic analysis on the following content.

Content:
{content}

## Analysis Process

### Phase 1: Domain Analysis
<thinking>
What domain does this content belong to?
What is the appropriate level of abstraction?
What are the conceptual boundaries?
</thinking>

### Phase 2: Topic Extraction
Extract 3-7 primary topics. For each topic provide:

1. **name**: Precise, domain-appropriate identifier
2. **description**: Write 2-3 sentences of flowing academic prose that:
   - Defines the concept clearly
   - Explains its significance in context
   - Connects to broader themes
3. **keywords**: 3-5 most relevant terms that capture the concept
4. **importance_score**: 0.0-1.0 based on centrality to the content
5. **confidence**: Your genuine assessment of extraction quality

### Phase 3: Relationship Mapping
For each topic, identify:
- How it relates to other topics
- Its hierarchical position (foundational, derived, applied)
- Dependencies and interactions

### Phase 4: Quality Validation
<thinking>
Assess the topic extraction:
- Have I captured the essential concepts?
- Are descriptions academically rigorous?
- Do topics form a coherent conceptual map?
- What is my confidence in this extraction?
</thinking>

Output as JSON with an array of topic objects. Maintain academic standards throughout."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.ENHANCED_TOPIC_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            topics = []
            for t in data.get("topics", []):
                topic = Topic(
                    name=t["name"],
                    description=t["description"],
                    keywords=t.get("keywords", []),
                    confidence=float(t.get("confidence", 0.7))
                )
                # Add importance to metadata
                topic.metadata = {
                    "importance_score": t.get("importance_score", 0.5),
                    "hierarchical_level": t.get("hierarchical_level", "primary")
                }
                topics.append(topic)
            
            logger.info(f"Enhanced topic extraction found {len(topics)} topics")
            return topics
            
        except Exception as e:
            logger.error(f"Enhanced topic extraction failed: {e}")
            return []


class EnhancedFactExtractor:
    """Enhanced fact extractor with evidence validation."""
    
    ENHANCED_FACT_PROMPT = """You are a Fact Verification Specialist using rigorous extraction methods.

## Verification Framework
Your approach ensures:
1. **Evidence-Based Claims**: Every fact must have clear textual support
2. **Precision in Language**: Claims must be specific and verifiable
3. **Contextual Relevance**: Facts must be significant, not trivial
4. **Logical Consistency**: Facts must not contradict each other

## Critical Thinking Process
<thinking>
For each potential fact, I will verify:
- Is there direct evidence in the text?
- Is the claim specific and falsifiable?
- Does it add meaningful knowledge?
- What is my confidence level?
</thinking>

Extract only well-supported, significant facts."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = "gpt-4o-mini"
        
    async def extract(self, content: str, existing_facts: List[Fact] = None) -> List[Fact]:
        """Extract facts with enhanced validation."""
        
        existing_context = ""
        if existing_facts:
            existing_context = "\n\nExisting facts to avoid duplication:\n"
            existing_context += "\n".join([f"- {f.claim}" for f in existing_facts[:5]])
        
        prompt = f"""Extract verifiable facts with rigorous evidence standards.

Content:
{content}
{existing_context}

## Extraction Process

### Phase 1: Claim Identification
<thinking>
What are the key factual claims in this content?
Which claims are most significant?
What evidence supports each claim?
</thinking>

### Phase 2: Fact Extraction
For each fact, provide:

1. **claim**: Specific, verifiable statement
   - Must be falsifiable
   - Should be self-contained
   - Avoid vague language
   
2. **evidence**: Direct textual support
   - Quote relevant portions
   - Provide context if needed
   - Explain how it supports the claim
   
3. **significance**: Why this fact matters
   - Its importance to the topic
   - Implications or applications
   - Connection to broader themes
   
4. **confidence**: 0.0-1.0 based on:
   - Strength of evidence
   - Clarity of claim
   - Absence of contradictions

### Phase 3: Consistency Check
<thinking>
Do these facts form a consistent picture?
Are there any contradictions?
Have I avoided redundancy?
What is the overall quality?
</thinking>

Output as JSON with an array of fact objects. Prioritize quality over quantity."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.ENHANCED_FACT_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for factual accuracy
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            facts = []
            for f in data.get("facts", []):
                fact = Fact(
                    claim=f["claim"],
                    evidence=f.get("evidence", ""),
                    confidence=float(f.get("confidence", 0.7))
                )
                fact.metadata = {
                    "significance": f.get("significance", ""),
                    "verification_notes": f.get("verification_notes", "")
                }
                facts.append(fact)
            
            logger.info(f"Enhanced fact extraction found {len(facts)} verified facts")
            return facts
            
        except Exception as e:
            logger.error(f"Enhanced fact extraction failed: {e}")
            return []


class EnhancedQuestionGenerator:
    """Enhanced question generator with cognitive depth."""
    
    ENHANCED_QUESTION_PROMPT = """You are a Cognitive Assessment Specialist creating intellectually stimulating questions.

## Question Design Principles
1. **Cognitive Progression**: Questions should scaffold from basic to complex
2. **Intellectual Engagement**: Stimulate critical thinking and analysis
3. **Practical Relevance**: Connect to real-world applications
4. **Assessment Value**: Enable meaningful evaluation of understanding

## Bloom's Taxonomy Framework
Design questions that truly test each cognitive level:
- **Remember**: Factual recall with precision
- **Understand**: Conceptual comprehension and explanation
- **Apply**: Practical use in new contexts
- **Analyze**: Breaking down complex ideas
- **Evaluate**: Critical judgment and assessment
- **Create**: Synthesis and innovation

Generate questions that would challenge and educate advanced learners."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = "gpt-4o-mini"
        
    async def generate(self, knowledge: ExtractedKnowledge, target_count: int = 8) -> List[Question]:
        """Generate cognitively diverse questions."""
        
        # Build context from knowledge
        context = self._build_context(knowledge)
        
        prompt = f"""Generate {target_count} intellectually stimulating questions based on this knowledge.

Knowledge Context:
{context}

## Question Generation Process

### Phase 1: Cognitive Planning
<thinking>
What are the key concepts to assess?
How can I create meaningful cognitive progression?
What real-world applications exist?
What would challenge advanced learners?
</thinking>

### Phase 2: Question Creation
Generate questions with:

1. **question_text**: Clear, thought-provoking question
   - Avoid yes/no questions
   - Be specific and unambiguous
   - Challenge assumptions
   
2. **expected_answer**: Comprehensive response
   - 2-4 sentences of academic prose
   - Include key concepts
   - Show depth of understanding
   
3. **cognitive_level**: Bloom's taxonomy level
   - Ensure distribution across all levels
   - Match question complexity to level
   - Progress logically
   
4. **difficulty**: 1-5 scale
   - 1: Basic recall
   - 3: Moderate analysis
   - 5: Complex synthesis
   
5. **learning_objective**: What this question teaches
   - Specific skill or concept
   - Why it matters
   - Connection to broader learning
   
6. **assessment_rubric**: How to evaluate answers
   - Key points required
   - Quality indicators
   - Common misconceptions

### Phase 3: Quality Review
<thinking>
Do these questions form a coherent assessment?
Is there appropriate cognitive diversity?
Would these challenge and educate?
What is the overall quality?
</thinking>

Ensure distribution:
- 1-2 Remember (but make them precise)
- 2 Understand (conceptual depth)
- 1-2 Apply (real scenarios)
- 1-2 Analyze (complex reasoning)
- 1-2 Evaluate/Create (if content supports)

Output as JSON with array of question objects."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.ENHANCED_QUESTION_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,  # Higher for creativity
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            questions = []
            for q in data.get("questions", []):
                question = Question(
                    question_text=q["question_text"],
                    expected_answer=q.get("expected_answer", ""),
                    cognitive_level=CognitiveLevel(q.get("cognitive_level", "understand")),
                    difficulty=int(q.get("difficulty", 3)),
                    confidence=0.8  # High confidence for generated questions
                )
                question.metadata = {
                    "learning_objective": q.get("learning_objective", ""),
                    "assessment_rubric": q.get("assessment_rubric", "")
                }
                questions.append(question)
            
            logger.info(f"Enhanced question generation created {len(questions)} questions")
            return questions
            
        except Exception as e:
            logger.error(f"Enhanced question generation failed: {e}")
            return []
    
    def _build_context(self, knowledge: ExtractedKnowledge) -> str:
        """Build context from extracted knowledge."""
        lines = []
        
        # Topics
        if knowledge.topics:
            lines.append("Main Topics:")
            for t in knowledge.topics[:5]:
                lines.append(f"- {t.name}: {t.description}")
        
        # Key facts
        if knowledge.facts:
            lines.append("\nKey Facts:")
            for f in sorted(knowledge.facts, key=lambda x: x.confidence, reverse=True)[:5]:
                lines.append(f"- {f.claim}")
        
        # Summary
        if knowledge.summary:
            lines.append(f"\nSummary: {knowledge.summary}")
        
        return "\n".join(lines)


# Convenience function to get all enhanced extractors
def get_enhanced_extractors(client: AsyncOpenAI) -> Dict[str, Any]:
    """Get instances of all enhanced extractors."""
    return {
        "base": EnhancedBaseExtractor(client),
        "topic": EnhancedTopicExtractor(client),
        "fact": EnhancedFactExtractor(client),
        "question": EnhancedQuestionGenerator(client)
    }