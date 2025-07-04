"""
Agent Educator - Teaching other agents to extract knowledge effectively.

This module provides educational resources and interactive learning for agents
to improve their knowledge extraction capabilities.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from extraction.v2.knowledge_service import (
    extract_knowledge,
    analyze_extraction_quality,
    get_extraction_examples,
    get_extraction_insights,
    get_extraction_guidelines
)
from extraction.models import ExtractedKnowledge
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LearningProgress:
    """Track an agent's learning progress."""
    agent_id: str
    lessons_completed: List[str]
    practice_scores: List[float]
    current_level: str  # "beginner", "intermediate", "advanced"
    strengths: List[str]
    weaknesses: List[str]
    last_updated: datetime


class AgentEducator:
    """
    Educator for teaching agents knowledge extraction skills.
    
    Provides structured lessons, practice exercises, and feedback.
    """
    
    def __init__(self):
        self.curriculum = self._build_curriculum()
        self.agent_progress = {}  # Track progress for each agent
        
    def _build_curriculum(self) -> Dict[str, Any]:
        """Build the educational curriculum."""
        return {
            "beginner": {
                "lessons": [
                    {
                        "id": "basics_1",
                        "title": "Understanding Knowledge Extraction",
                        "objectives": [
                            "Identify topics vs facts",
                            "Understand confidence scoring",
                            "Recognize good evidence"
                        ],
                        "content": self._get_lesson_basics_1()
                    },
                    {
                        "id": "basics_2", 
                        "title": "Document Types and Strategies",
                        "objectives": [
                            "Identify document types",
                            "Choose appropriate extraction strategy",
                            "Understand quality dimensions"
                        ],
                        "content": self._get_lesson_basics_2()
                    }
                ],
                "exercises": [
                    {
                        "id": "exercise_1",
                        "title": "Extract from Simple Text",
                        "difficulty": "easy",
                        "test_content": """
                        Python is a high-level programming language. It was created by 
                        Guido van Rossum and released in 1991. Python emphasizes code 
                        readability and uses significant whitespace. It supports multiple 
                        programming paradigms including procedural, object-oriented, and 
                        functional programming.
                        """,
                        "expected_topics": ["Python", "Programming Language"],
                        "expected_facts_count": 3
                    }
                ]
            },
            "intermediate": {
                "lessons": [
                    {
                        "id": "enhance_1",
                        "title": "Using Enhancers Effectively",
                        "objectives": [
                            "Apply citation enhancement",
                            "Discover relationships",
                            "Generate quality questions"
                        ],
                        "content": self._get_lesson_enhance_1()
                    },
                    {
                        "id": "quality_1",
                        "title": "Quality Assessment and Improvement",
                        "objectives": [
                            "Analyze extraction quality",
                            "Identify weaknesses",
                            "Apply improvements"
                        ],
                        "content": self._get_lesson_quality_1()
                    }
                ],
                "exercises": [
                    {
                        "id": "exercise_2",
                        "title": "Extract from Academic Text",
                        "difficulty": "medium",
                        "test_content": """
                        Recent studies (Smith et al., 2023) have shown that transformer 
                        models achieve state-of-the-art performance on natural language 
                        processing tasks. The attention mechanism allows models to focus 
                        on relevant parts of the input. BERT achieved 92.4% accuracy on 
                        the GLUE benchmark, while GPT-3 demonstrated few-shot learning 
                        capabilities with 175 billion parameters.
                        """,
                        "expected_topics": ["Transformer Models", "NLP", "Attention Mechanism"],
                        "expected_citations": True,
                        "minimum_quality": 0.7
                    }
                ]
            },
            "advanced": {
                "lessons": [
                    {
                        "id": "domain_1",
                        "title": "Domain-Specific Extraction",
                        "objectives": [
                            "Master domain terminology",
                            "Extract specialized knowledge",
                            "Handle complex relationships"
                        ],
                        "content": self._get_lesson_domain_1()
                    }
                ],
                "exercises": [
                    {
                        "id": "exercise_3",
                        "title": "Extract Complex Relationships",
                        "difficulty": "hard",
                        "test_content": """
                        The integration of quantum computing with machine learning presents 
                        unique challenges. Quantum algorithms like Grover's search and Shor's 
                        factorization demonstrate exponential speedup. However, quantum 
                        decoherence limits practical applications. Hybrid classical-quantum 
                        approaches show promise, with variational quantum eigensolvers (VQE) 
                        being used for optimization problems in drug discovery.
                        """,
                        "expected_relationships": 3,
                        "minimum_quality": 0.8
                    }
                ]
            }
        }
    
    async def teach_agent(self, agent_id: str, lesson_id: str) -> Dict[str, Any]:
        """
        Teach a specific lesson to an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            lesson_id: ID of the lesson to teach
            
        Returns:
            Lesson content and materials
        """
        # Find the lesson
        lesson = None
        for level in self.curriculum.values():
            for l in level.get("lessons", []):
                if l["id"] == lesson_id:
                    lesson = l
                    break
        
        if not lesson:
            return {"error": f"Lesson {lesson_id} not found"}
        
        # Track progress
        if agent_id not in self.agent_progress:
            self.agent_progress[agent_id] = LearningProgress(
                agent_id=agent_id,
                lessons_completed=[],
                practice_scores=[],
                current_level="beginner",
                strengths=[],
                weaknesses=[],
                last_updated=datetime.utcnow()
            )
        
        # Record lesson completion
        progress = self.agent_progress[agent_id]
        if lesson_id not in progress.lessons_completed:
            progress.lessons_completed.append(lesson_id)
            progress.last_updated = datetime.utcnow()
        
        logger.info(f"Teaching lesson {lesson_id} to agent {agent_id}")
        
        return {
            "lesson": lesson,
            "examples": await self._get_lesson_examples(lesson_id),
            "practice_tips": self._get_practice_tips(lesson_id),
            "next_steps": self._get_next_steps(agent_id, lesson_id)
        }
    
    async def practice_extraction(self, agent_id: str, exercise_id: str, 
                                agent_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an agent's practice extraction.
        
        Args:
            agent_id: Unique identifier for the agent
            exercise_id: ID of the exercise
            agent_extraction: The agent's extraction attempt
            
        Returns:
            Feedback and score
        """
        # Find the exercise
        exercise = None
        for level in self.curriculum.values():
            for ex in level.get("exercises", []):
                if ex["id"] == exercise_id:
                    exercise = ex
                    break
        
        if not exercise:
            return {"error": f"Exercise {exercise_id} not found"}
        
        # Analyze the extraction
        quality_result = await analyze_extraction_quality(
            agent_extraction,
            exercise["test_content"]
        )
        
        # Evaluate against expectations
        feedback = {
            "exercise_id": exercise_id,
            "quality_score": quality_result["overall_score"],
            "passed": quality_result["overall_score"] >= exercise.get("minimum_quality", 0.6),
            "strengths": [],
            "improvements_needed": [],
            "detailed_feedback": {}
        }
        
        # Check topic extraction
        if "expected_topics" in exercise:
            extracted_topics = [t["name"] for t in agent_extraction.get("topics", [])]
            expected_topics = exercise["expected_topics"]
            
            topic_coverage = len(set(extracted_topics) & set(expected_topics)) / len(expected_topics)
            feedback["detailed_feedback"]["topic_coverage"] = topic_coverage
            
            if topic_coverage >= 0.8:
                feedback["strengths"].append("Good topic identification")
            else:
                feedback["improvements_needed"].append(
                    f"Missing topics: {set(expected_topics) - set(extracted_topics)}"
                )
        
        # Check fact count
        if "expected_facts_count" in exercise:
            fact_count = len(agent_extraction.get("facts", []))
            expected_count = exercise["expected_facts_count"]
            
            if fact_count >= expected_count:
                feedback["strengths"].append("Comprehensive fact extraction")
            else:
                feedback["improvements_needed"].append(
                    f"Extract more facts (found {fact_count}, expected at least {expected_count})"
                )
        
        # Check citations
        if exercise.get("expected_citations", False):
            facts_with_evidence = sum(
                1 for f in agent_extraction.get("facts", []) 
                if f.get("evidence")
            )
            if facts_with_evidence > 0:
                feedback["strengths"].append("Good use of citations")
            else:
                feedback["improvements_needed"].append("Add evidence/citations to facts")
        
        # Check relationships
        if "expected_relationships" in exercise:
            rel_count = len(agent_extraction.get("relationships", []))
            expected_rels = exercise["expected_relationships"]
            
            if rel_count >= expected_rels:
                feedback["strengths"].append("Strong relationship identification")
            else:
                feedback["improvements_needed"].append(
                    f"Identify more relationships (found {rel_count}, expected at least {expected_rels})"
                )
        
        # Update agent progress
        if agent_id in self.agent_progress:
            progress = self.agent_progress[agent_id]
            progress.practice_scores.append(quality_result["overall_score"])
            
            # Update strengths/weaknesses
            for dim, score in quality_result["dimensions"].items():
                if score >= 0.8 and dim not in progress.strengths:
                    progress.strengths.append(dim)
                elif score < 0.6 and dim not in progress.weaknesses:
                    progress.weaknesses.append(dim)
            
            # Check for level advancement
            if len(progress.practice_scores) >= 3 and \
               sum(progress.practice_scores[-3:]) / 3 >= 0.75:
                if progress.current_level == "beginner":
                    progress.current_level = "intermediate"
                    feedback["achievement"] = "🎉 Advanced to Intermediate level!"
                elif progress.current_level == "intermediate":
                    progress.current_level = "advanced"
                    feedback["achievement"] = "🏆 Advanced to Advanced level!"
        
        # Add specific recommendations
        feedback["recommendations"] = quality_result.get("recommendations", [])
        
        logger.info(f"Agent {agent_id} scored {quality_result['overall_score']:.2f} on {exercise_id}")
        
        return feedback
    
    def get_agent_progress(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get learning progress for an agent."""
        if agent_id not in self.agent_progress:
            return None
        
        progress = self.agent_progress[agent_id]
        
        # Calculate statistics
        avg_score = sum(progress.practice_scores) / len(progress.practice_scores) if progress.practice_scores else 0
        recent_trend = "improving" if len(progress.practice_scores) >= 2 and progress.practice_scores[-1] > progress.practice_scores[-2] else "stable"
        
        return {
            "agent_id": agent_id,
            "current_level": progress.current_level,
            "lessons_completed": len(progress.lessons_completed),
            "total_lessons": sum(len(level["lessons"]) for level in self.curriculum.values()),
            "average_score": avg_score,
            "recent_trend": recent_trend,
            "strengths": progress.strengths,
            "weaknesses": progress.weaknesses,
            "last_active": progress.last_updated.isoformat()
        }
    
    async def recommend_next_lesson(self, agent_id: str) -> Dict[str, Any]:
        """Recommend the next lesson for an agent."""
        progress = self.agent_progress.get(agent_id)
        
        if not progress:
            # New agent - start with basics
            return {
                "recommended_lesson": "basics_1",
                "reason": "Start with understanding knowledge extraction fundamentals"
            }
        
        # Find uncompleted lessons at current level
        current_level_lessons = self.curriculum[progress.current_level]["lessons"]
        uncompleted = [
            l for l in current_level_lessons 
            if l["id"] not in progress.lessons_completed
        ]
        
        if uncompleted:
            return {
                "recommended_lesson": uncompleted[0]["id"],
                "reason": f"Complete {progress.current_level} curriculum"
            }
        
        # Check if ready for next level
        if progress.current_level == "beginner" and len(progress.practice_scores) >= 2:
            avg_score = sum(progress.practice_scores[-2:]) / 2
            if avg_score >= 0.7:
                return {
                    "recommended_lesson": "enhance_1",
                    "reason": "Ready for intermediate enhancer techniques"
                }
        
        # Recommend practice based on weaknesses
        if progress.weaknesses:
            weakness = progress.weaknesses[0]
            if weakness == "grounding":
                return {
                    "recommended_exercise": "exercise_2",
                    "reason": "Practice adding citations and evidence"
                }
            elif weakness == "coherence":
                return {
                    "recommended_exercise": "exercise_3", 
                    "reason": "Practice identifying relationships"
                }
        
        return {
            "recommended_lesson": "domain_1",
            "reason": "Explore advanced domain-specific techniques"
        }
    
    # Lesson content methods
    def _get_lesson_basics_1(self) -> str:
        return """
# Lesson 1: Understanding Knowledge Extraction

## What is Knowledge Extraction?
Knowledge extraction is the process of identifying and structuring important information from text.

## Core Components:

### 1. Topics
- **What**: Main subjects or themes
- **Example**: "Machine Learning", "Climate Change"
- **Best Practice**: 3-7 topics per document
- **Confidence**: Based on prominence and repetition

### 2. Facts
- **What**: Specific, verifiable claims
- **Example**: "Python was created in 1991"
- **Best Practice**: Include evidence when available
- **Confidence**: Based on clarity and support

### 3. Questions
- **What**: Knowledge gaps or learning prompts
- **Levels**: Remember → Understand → Apply → Analyze
- **Example**: "How does the attention mechanism work?"

### 4. Relationships
- **What**: Connections between entities
- **Types**: causes, uses, improves, contradicts
- **Example**: "Deep Learning" → uses → "Neural Networks"

## Quality Indicators:
- High confidence (>0.8) for main topics
- Evidence-backed facts
- Logical relationships
- Comprehensive coverage

## Practice Exercise:
Try extracting knowledge from a simple paragraph and identify:
- 2-3 main topics
- 3-5 specific facts
- 1-2 relationships
"""

    def _get_lesson_basics_2(self) -> str:
        return """
# Lesson 2: Document Types and Strategies

## Document Types:

### Academic Papers
- **Characteristics**: Citations, methodology, formal structure
- **Strategy**: Focus on findings, preserve citations
- **Key Extractions**: Hypotheses, methods, results, conclusions
- **Quality Focus**: High grounding with evidence

### Technical Documentation
- **Characteristics**: Implementation details, code examples
- **Strategy**: Extract specifications, parameters, procedures
- **Key Extractions**: Requirements, APIs, configurations
- **Quality Focus**: Precision and completeness

### Narrative/Business
- **Characteristics**: Stories, strategies, opinions
- **Strategy**: Identify stakeholders, goals, challenges
- **Key Extractions**: Problems, solutions, outcomes
- **Quality Focus**: Coherence and context

## Model Selection:
- **Long documents (>50K)**: Use Claude for context
- **Academic + Quality**: Use GPT-4 for accuracy
- **Quick extraction**: Use GPT-3.5 for speed
- **Technical precision**: Use specialized extractors

## Quality Dimensions:

### 1. Consistency (25%)
- Facts support topics
- No contradictions
- Uniform detail level

### 2. Grounding (35%)
- Evidence for claims
- Proper citations
- Traceable sources

### 3. Coherence (20%)
- Logical flow
- Clear relationships
- Connected ideas

### 4. Completeness (20%)
- Major points covered
- No critical gaps
- Balanced perspective

## Red Flags:
- ❌ Vague claims without evidence
- ❌ Missing main topics
- ❌ Contradictory facts
- ❌ No relationships identified
"""

    def _get_lesson_enhance_1(self) -> str:
        return """
# Lesson 3: Using Enhancers Effectively

## Citation Enhancer

### When to Use:
- Academic content with references
- Claims needing evidence
- Controversial statements

### How it Works:
1. Matches claims to source text
2. Finds formal citations (Author, Year)
3. Extracts supporting sentences
4. Boosts confidence for cited facts

### Example:
Before: "AI improves diagnosis"
After: "AI improves diagnosis (Source: Smith et al., 2023 found 94% accuracy)"

## Relationship Enhancer

### When to Use:
- Complex systems or processes
- Multiple interconnected topics
- Cause-effect scenarios

### Types of Relationships:
- **uses**: Technology applications
- **impacts**: Effects and influences  
- **analyzes**: Research subjects
- **contradicts**: Opposing views
- **enables**: Prerequisites

### Discovery Method:
1. Find entity co-occurrences
2. Analyze connecting words
3. Determine relationship type
4. Assign confidence

## Question Enhancer

### Bloom's Taxonomy Levels:
1. **Remember**: What is X?
2. **Understand**: How does X work?
3. **Apply**: How can X be used for Y?
4. **Analyze**: What are the components of X?
5. **Evaluate**: Is X better than Y?

### Best Practices:
- Match questions to content depth
- Target knowledge gaps
- Encourage critical thinking
- Provide expected answers

## Combining Enhancers:
1. Extract base knowledge
2. Add citations for grounding
3. Discover relationships for coherence
4. Generate questions for engagement
5. Create summary for overview

## Quality Impact:
- Citations: +15-20% grounding score
- Relationships: +10-15% coherence score
- Questions: Improves learning retention
"""

    def _get_lesson_quality_1(self) -> str:
        return """
# Lesson 4: Quality Assessment and Improvement

## Understanding Quality Scores

### Score Ranges:
- **0.8+**: Production ready
- **0.6-0.8**: Good, minor improvements needed
- **0.4-0.6**: Significant issues, enhancement required
- **<0.4**: Major problems, consider re-extraction

## Analyzing Weaknesses

### Low Consistency (<0.6):
- **Symptoms**: Facts don't relate to topics
- **Fixes**: 
  - Align facts with identified topics
  - Remove off-topic content
  - Ensure uniform detail level

### Low Grounding (<0.6):
- **Symptoms**: Claims lack evidence
- **Fixes**:
  - Apply citation enhancer
  - Add "According to..." attributions
  - Include specific numbers/dates
  - Reference source sections

### Low Coherence (<0.6):
- **Symptoms**: Disconnected information
- **Fixes**:
  - Apply relationship enhancer
  - Create logical flow
  - Group related facts
  - Add transitional context

### Low Completeness (<0.6):
- **Symptoms**: Missing key information
- **Fixes**:
  - Extract more facts
  - Cover all document sections
  - Include limitations/caveats
  - Add minority viewpoints

## Improvement Workflow:

1. **Get Quality Report**:
   ```python
   quality = await analyze_extraction_quality(extraction, source)
   ```

2. **Identify Weakest Dimension**:
   ```python
   weakest = min(quality['dimensions'], key=quality['dimensions'].get)
   ```

3. **Apply Targeted Enhancement**:
   - Grounding → Citation Enhancer
   - Coherence → Relationship Enhancer
   - Completeness → Re-extract with focus

4. **Verify Improvement**:
   ```python
   new_quality = await analyze_extraction_quality(enhanced, source)
   improved = new_quality['overall_score'] > quality['overall_score']
   ```

## Common Patterns:

### Academic Papers:
- Often need citation enhancement
- Benefit from question generation
- Require methodology extraction

### Technical Docs:
- Need complete parameter coverage
- Benefit from example extraction
- Require precise terminology

### Business Documents:
- Need stakeholder identification
- Benefit from outcome extraction
- Require balanced perspective
"""

    def _get_lesson_domain_1(self) -> str:
        return """
# Lesson 5: Domain-Specific Extraction

## Machine Learning Domain

### Key Terminology:
- Models: CNN, RNN, Transformer, GAN
- Metrics: Accuracy, F1, AUC, Perplexity
- Datasets: MNIST, ImageNet, COCO, GLUE
- Techniques: Transfer learning, Fine-tuning

### Extraction Focus:
1. Architecture details (layers, parameters)
2. Performance metrics with context
3. Training details (epochs, batch size)
4. Comparison with baselines
5. Limitations and failure modes

### Quality Indicators:
- Specific numbers (not just "improved")
- Dataset context (size, domain)
- Computational requirements
- Reproducibility information

## Healthcare Domain

### Key Terminology:
- Clinical phases (I, II, III, IV)
- Metrics: Efficacy, Safety, QoL
- Standards: FDA, EMA, GCP
- Study types: RCT, Cohort, Case-control

### Extraction Focus:
1. Patient populations (n=, demographics)
2. Interventions and controls
3. Primary/secondary endpoints
4. Statistical significance (p-values, CI)
5. Adverse events and risks

### Quality Indicators:
- Sample sizes and power
- Blinding and randomization
- Follow-up duration
- Regulatory status

## Financial Domain

### Key Terminology:
- Metrics: ROI, NPV, IRR, EBITDA
- Instruments: Equity, Debt, Derivatives
- Analysis: Fundamental, Technical
- Risk: Market, Credit, Operational

### Extraction Focus:
1. Financial metrics with timeframes
2. Market conditions and assumptions
3. Risk factors and mitigation
4. Regulatory compliance
5. Forward-looking statements

## Best Practices by Domain:

### Scientific/Technical:
- Preserve exact terminology
- Include error bars/confidence intervals
- Note experimental conditions
- Extract replication information

### Business/Strategic:
- Identify all stakeholders
- Extract goals and KPIs
- Note assumptions and risks
- Include competitive context

### Legal/Regulatory:
- Preserve exact quotes
- Note effective dates
- Extract obligations and rights
- Include jurisdiction information

## Advanced Techniques:

### Cross-Domain Connections:
- ML + Healthcare = Medical AI
- Finance + Regulatory = FinTech compliance
- Technical + Business = Product strategy

### Handling Ambiguity:
- Note conflicting information
- Indicate uncertainty levels
- Provide alternative interpretations
- Flag areas needing human review
"""

    async def _get_lesson_examples(self, lesson_id: str) -> List[Dict[str, Any]]:
        """Get examples for a specific lesson."""
        if lesson_id == "basics_1":
            return get_extraction_examples("all")
        elif lesson_id == "enhance_1":
            # Show before/after enhancement examples
            return [
                {
                    "title": "Citation Enhancement Example",
                    "before": {
                        "claim": "Neural networks achieve high accuracy",
                        "confidence": 0.7
                    },
                    "after": {
                        "claim": "Neural networks achieve high accuracy",
                        "evidence": "LeCun et al. (2015) reported 99.2% on MNIST",
                        "confidence": 0.9
                    }
                }
            ]
        return []
    
    def _get_practice_tips(self, lesson_id: str) -> List[str]:
        """Get practice tips for a lesson."""
        tips_map = {
            "basics_1": [
                "Start with clear, well-structured text",
                "Aim for 3-5 topics and 5-10 facts",
                "Focus on confidence scoring accuracy"
            ],
            "basics_2": [
                "Practice identifying document types quickly",
                "Match extraction strategy to content",
                "Pay attention to quality dimension balance"
            ],
            "enhance_1": [
                "Try each enhancer separately first",
                "Combine enhancers for maximum effect",
                "Measure quality improvement"
            ],
            "quality_1": [
                "Always analyze before trying to improve",
                "Focus on the weakest dimension first",
                "Verify improvements with re-analysis"
            ],
            "domain_1": [
                "Build domain vocabulary lists",
                "Study high-quality domain examples",
                "Practice cross-domain extraction"
            ]
        }
        return tips_map.get(lesson_id, ["Practice regularly", "Analyze your mistakes"])
    
    def _get_next_steps(self, agent_id: str, completed_lesson: str) -> List[str]:
        """Get recommended next steps after a lesson."""
        progress = self.agent_progress.get(agent_id)
        
        if not progress:
            return ["Complete the practice exercise", "Move to Lesson 2"]
        
        steps = []
        
        # Recommend practice if scores are low
        if progress.practice_scores and progress.practice_scores[-1] < 0.7:
            steps.append("Practice more with the current exercise")
        
        # Recommend next lesson
        if completed_lesson == "basics_1":
            steps.append("Continue to Lesson 2: Document Types")
        elif completed_lesson == "basics_2":
            steps.append("Try Exercise 1 to test your skills")
        
        # Recommend enhancement if ready
        if progress.current_level == "intermediate":
            steps.append("Experiment with enhancer combinations")
        
        return steps


# Standalone functions for agent interaction

async def get_personalized_curriculum(agent_id: str, 
                                    agent_capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a personalized curriculum based on agent capabilities.
    
    Args:
        agent_id: Unique agent identifier
        agent_capabilities: Dict describing what the agent can do
        
    Returns:
        Personalized learning path
    """
    curriculum = []
    
    # Assess current level
    can_extract = agent_capabilities.get("can_extract_text", False)
    has_llm_access = agent_capabilities.get("has_llm_access", False)
    understands_json = agent_capabilities.get("understands_json", False)
    
    if not can_extract:
        curriculum.append({
            "priority": 1,
            "lesson": "prerequisites",
            "content": "Learn to extract text from documents first"
        })
    
    if not understands_json:
        curriculum.append({
            "priority": 2,
            "lesson": "json_structures",
            "content": "Understanding ExtractedKnowledge JSON structure"
        })
    
    # Main curriculum
    curriculum.extend([
        {
            "priority": 3,
            "lesson": "basics_1",
            "estimated_time": "30 minutes",
            "prerequisites": ["text_extraction", "json"]
        },
        {
            "priority": 4,
            "lesson": "basics_2",
            "estimated_time": "45 minutes",
            "prerequisites": ["basics_1"]
        }
    ])
    
    # Advanced topics if capable
    if has_llm_access:
        curriculum.append({
            "priority": 5,
            "lesson": "llm_prompting",
            "content": "Optimizing prompts for extraction"
        })
    
    return {
        "agent_id": agent_id,
        "recommended_path": sorted(curriculum, key=lambda x: x["priority"]),
        "estimated_total_time": "3-4 hours",
        "customization_notes": f"Curriculum adapted for agent with {sum(1 for v in agent_capabilities.values() if v)} capabilities"
    }


async def create_agent_study_group(agent_ids: List[str]) -> Dict[str, Any]:
    """
    Create a collaborative learning environment for multiple agents.
    
    Args:
        agent_ids: List of agents to form a study group
        
    Returns:
        Study group configuration and activities
    """
    return {
        "group_id": f"study_group_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        "members": agent_ids,
        "activities": [
            {
                "type": "peer_review",
                "description": "Agents review each other's extractions",
                "benefit": "Learn from different approaches"
            },
            {
                "type": "competition", 
                "description": "Extraction quality competition",
                "benefit": "Motivate improvement through gamification"
            },
            {
                "type": "collaboration",
                "description": "Joint extraction of complex documents",
                "benefit": "Combine strengths for better results"
            }
        ],
        "communication_protocol": {
            "share_extractions": True,
            "share_quality_scores": True,
            "share_techniques": True,
            "privacy": "within_group_only"
        },
        "goals": [
            "All agents reach 0.75+ average quality",
            "Share 3+ unique techniques",
            "Complete advanced curriculum together"
        ]
    }


def get_certification_criteria() -> Dict[str, Any]:
    """
    Get criteria for agent certification in knowledge extraction.
    
    Returns:
        Certification requirements and levels
    """
    return {
        "levels": {
            "certified_basic": {
                "requirements": [
                    "Complete all beginner lessons",
                    "Pass 3 beginner exercises with 0.7+ score",
                    "Demonstrate consistent quality improvement"
                ],
                "benefits": [
                    "Can extract from simple documents",
                    "Understands quality dimensions",
                    "Can apply basic enhancements"
                ]
            },
            "certified_intermediate": {
                "requirements": [
                    "Complete intermediate curriculum",
                    "Average quality score 0.75+",
                    "Successfully use all enhancers",
                    "Pass domain-specific exercise"
                ],
                "benefits": [
                    "Can handle complex documents",
                    "Optimizes extraction strategies",
                    "Provides quality feedback"
                ]
            },
            "certified_advanced": {
                "requirements": [
                    "Complete advanced curriculum",
                    "Average quality score 0.85+",
                    "Demonstrate innovation in extraction",
                    "Teach other agents successfully"
                ],
                "benefits": [
                    "Expert-level extraction",
                    "Can create new techniques",
                    "Qualified to educate others"
                ]
            }
        },
        "assessment_method": "Automated scoring + peer review",
        "validity_period": "6 months (requires re-certification)",
        "certification_benefits": [
            "Quality trust marker",
            "Access to advanced resources",
            "Eligible for teaching roles"
        ]
    }