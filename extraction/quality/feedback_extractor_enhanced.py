"""
Enhanced feedback-based extraction system with battle-tested patterns.

Uses agent loops, thinking blocks, and iterative improvement strategies.
"""

import asyncio
from typing import Dict, Any, List, Optional
import json

from openai import AsyncOpenAI

from extraction.models import ExtractedKnowledge, DocumentProfile
from extraction.quality.scorer import QualityScorer
from extraction.document_aware.extractor import DocumentAwareExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Enhanced system prompt for feedback-driven improvement
ENHANCED_FEEDBACK_SYSTEM_PROMPT = """You are a Quality-Driven Extraction Specialist using iterative refinement techniques.

## Agent Loop Architecture
Your improvement process follows these phases:
1. **Analyze**: Understand quality feedback and identify weaknesses
2. **Plan**: Develop targeted improvement strategies
3. **Execute**: Apply focused enhancements
4. **Validate**: Ensure improvements address issues
5. **Iterate**: Refine until quality threshold is met

## Quality Framework
<thinking>
For each improvement iteration, I will:
- Identify specific quality issues
- Understand root causes
- Apply targeted fixes
- Validate improvements
- Maintain extraction integrity
</thinking>

## Improvement Principles
1. **Evidence-Based**: All improvements must strengthen grounding
2. **Consistency First**: Resolve contradictions before adding new content
3. **Precision Focus**: Enhance accuracy over quantity
4. **Coherent Integration**: Ensure all parts work together
5. **Academic Standards**: Maintain scholarly quality throughout

Your goal is systematic quality improvement through intelligent refinement."""


class EnhancedFeedbackExtractor:
    """Enhanced extraction system with quality feedback loops."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize enhanced feedback extractor.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for extraction
        """
        self.api_key = openai_api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize components
        self.scorer = QualityScorer()
        self.base_extractor = DocumentAwareExtractor(openai_api_key, model)
        
        # Enhanced configuration
        self.max_iterations = 4  # Allow more iterations
        self.quality_threshold = 0.75  # Higher quality bar
        self.improvement_threshold = 0.05  # Minimum improvement to continue
        
        logger.info(f"Initialized EnhancedFeedbackExtractor with model: {model}")
    
    async def extract_with_enhanced_feedback(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract knowledge with enhanced quality feedback loop.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Final extraction with detailed quality analysis
        """
        logger.info(f"Starting enhanced extraction with feedback for: {pdf_path}")
        
        iteration_history = []
        best_extraction = None
        best_score = 0.0
        improvement_strategies = []
        
        # Get source content for grounding checks
        source_content = await self._get_source_content(pdf_path)
        
        for iteration in range(self.max_iterations):
            logger.info(f"Enhanced iteration {iteration + 1}/{self.max_iterations}")
            
            # Phase 1: Extract with appropriate strategy
            if iteration == 0:
                # Initial extraction
                extraction = await self.base_extractor.extract(pdf_path)
            else:
                # Improved extraction based on feedback
                extraction = await self._extract_with_targeted_improvements(
                    pdf_path,
                    previous_extraction=best_extraction,
                    quality_analysis=iteration_history[-1]["quality_analysis"],
                    improvement_strategies=improvement_strategies,
                    source_content=source_content
                )
            
            # Phase 2: Comprehensive quality analysis
            quality_analysis = await self._analyze_extraction_quality(
                extraction, source_content
            )
            
            # Phase 3: Strategic planning
            if quality_analysis["overall_score"] < self.quality_threshold:
                improvement_strategies = self._develop_improvement_strategies(
                    quality_analysis, extraction
                )
            
            # Record iteration
            iteration_data = {
                "iteration": iteration + 1,
                "extraction": extraction,
                "quality_analysis": quality_analysis,
                "overall_score": quality_analysis["overall_score"],
                "improvement_strategies": improvement_strategies
            }
            iteration_history.append(iteration_data)
            
            # Update best if improved
            if quality_analysis["overall_score"] > best_score:
                improvement = quality_analysis["overall_score"] - best_score
                best_extraction = extraction
                best_score = quality_analysis["overall_score"]
                logger.info(f"New best score: {best_score:.3f} (+{improvement:.3f})")
            
            # Check termination conditions
            if quality_analysis["overall_score"] >= self.quality_threshold:
                logger.info(f"Quality threshold reached: {quality_analysis['overall_score']:.3f}")
                break
            
            if iteration > 0:
                improvement = quality_analysis["overall_score"] - iteration_history[-2]["overall_score"]
                if improvement < self.improvement_threshold:
                    logger.info(f"Improvement below threshold: {improvement:.3f}")
                    break
            
            if not improvement_strategies:
                logger.info("No improvement strategies identified")
                break
        
        # Final comprehensive analysis
        final_analysis = await self._perform_final_analysis(
            best_extraction, source_content, iteration_history
        )
        
        return {
            "extraction": best_extraction,
            "quality_analysis": final_analysis,
            "iterations": len(iteration_history),
            "iteration_history": self._summarize_iteration_history(iteration_history),
            "total_improvement": best_score - iteration_history[0]["overall_score"],
            "improvement_trajectory": [h["overall_score"] for h in iteration_history],
            "key_improvements": self._identify_key_improvements(iteration_history),
            "remaining_issues": final_analysis.get("remaining_issues", [])
        }
    
    async def _analyze_extraction_quality(self, extraction: ExtractedKnowledge, 
                                        source_content: str) -> Dict[str, Any]:
        """Perform comprehensive quality analysis."""
        # Basic scoring
        basic_scores = self.scorer.score_extraction(extraction, source_content)
        
        # Enhanced analysis
        enhanced_analysis = {
            **basic_scores,
            "detailed_metrics": {}
        }
        
        # Topic quality analysis
        topic_analysis = self._analyze_topic_quality(extraction.topics)
        enhanced_analysis["detailed_metrics"]["topics"] = topic_analysis
        
        # Fact quality analysis
        fact_analysis = self._analyze_fact_quality(extraction.facts, source_content)
        enhanced_analysis["detailed_metrics"]["facts"] = fact_analysis
        
        # Relationship quality analysis
        relationship_analysis = self._analyze_relationship_quality(
            extraction.relationships, extraction.topics, extraction.facts
        )
        enhanced_analysis["detailed_metrics"]["relationships"] = relationship_analysis
        
        # Question quality analysis
        question_analysis = self._analyze_question_quality(extraction.questions)
        enhanced_analysis["detailed_metrics"]["questions"] = question_analysis
        
        # Summary quality analysis
        summary_analysis = self._analyze_summary_quality(
            extraction.summary, extraction.topics, extraction.facts
        )
        enhanced_analysis["detailed_metrics"]["summary"] = summary_analysis
        
        # Academic quality assessment
        academic_score = self._assess_academic_quality(extraction)
        enhanced_analysis["academic_quality"] = academic_score
        
        return enhanced_analysis
    
    def _analyze_topic_quality(self, topics: List) -> Dict[str, Any]:
        """Analyze topic extraction quality."""
        if not topics:
            return {"score": 0.0, "issues": ["No topics extracted"]}
        
        analysis = {
            "count": len(topics),
            "avg_description_length": sum(len(t.description) for t in topics) / len(topics),
            "avg_keyword_count": sum(len(t.keywords) for t in topics) / len(topics),
            "issues": []
        }
        
        # Check for quality issues
        for topic in topics:
            if len(topic.description) < 50:
                analysis["issues"].append(f"Topic '{topic.name}' has brief description")
            if len(topic.keywords) < 3:
                analysis["issues"].append(f"Topic '{topic.name}' has few keywords")
            if topic.confidence < 0.5:
                analysis["issues"].append(f"Topic '{topic.name}' has low confidence")
        
        # Calculate score
        score = 1.0
        if analysis["avg_description_length"] < 100:
            score -= 0.2
        if analysis["avg_keyword_count"] < 3:
            score -= 0.1
        if len(analysis["issues"]) > 0:
            score -= 0.1 * min(len(analysis["issues"]), 3)
        
        analysis["score"] = max(0.0, score)
        return analysis
    
    def _analyze_fact_quality(self, facts: List, source_content: str) -> Dict[str, Any]:
        """Analyze fact extraction quality."""
        if not facts:
            return {"score": 0.0, "issues": ["No facts extracted"]}
        
        analysis = {
            "count": len(facts),
            "with_evidence": sum(1 for f in facts if f.evidence),
            "avg_confidence": sum(f.confidence for f in facts) / len(facts),
            "issues": []
        }
        
        # Check each fact
        for fact in facts:
            # Evidence check
            if not fact.evidence:
                analysis["issues"].append(f"Fact lacks evidence: '{fact.claim[:50]}...'")
            elif len(fact.evidence) < 20:
                analysis["issues"].append(f"Fact has weak evidence: '{fact.claim[:50]}...'")
            
            # Grounding check (simplified)
            if source_content:
                claim_words = set(fact.claim.lower().split())
                source_words = set(source_content.lower().split())
                overlap = len(claim_words & source_words) / len(claim_words)
                if overlap < 0.5:
                    analysis["issues"].append(f"Fact poorly grounded: '{fact.claim[:50]}...'")
        
        # Calculate score
        evidence_ratio = analysis["with_evidence"] / analysis["count"]
        score = (
            evidence_ratio * 0.5 +
            analysis["avg_confidence"] * 0.3 +
            (1.0 - min(len(analysis["issues"]) / analysis["count"], 1.0)) * 0.2
        )
        
        analysis["score"] = score
        return analysis
    
    def _analyze_relationship_quality(self, relationships: List, topics: List, facts: List) -> Dict[str, Any]:
        """Analyze relationship extraction quality."""
        analysis = {
            "count": len(relationships),
            "with_description": sum(1 for r in relationships if r.description),
            "avg_confidence": sum(r.confidence for r in relationships) / len(relationships) if relationships else 0,
            "issues": []
        }
        
        # Build entity sets
        topic_entities = {t.name for t in topics}
        fact_entities = set()
        for fact in facts:
            # Extract entities from facts (simplified)
            words = fact.claim.split()
            fact_entities.update(w for w in words if w[0].isupper() and len(w) > 2)
        
        all_entities = topic_entities | fact_entities
        
        # Check relationships
        for rel in relationships:
            # Check if entities are defined
            if rel.source_entity not in all_entities:
                analysis["issues"].append(f"Undefined source entity: {rel.source_entity}")
            if rel.target_entity not in all_entities:
                analysis["issues"].append(f"Undefined target entity: {rel.target_entity}")
            
            # Check description quality
            if not rel.description or len(rel.description) < 10:
                analysis["issues"].append(
                    f"Poor relationship description: {rel.source_entity} -> {rel.target_entity}"
                )
        
        # Calculate score
        if not relationships:
            score = 0.3  # Penalty for no relationships
        else:
            score = (
                (analysis["with_description"] / analysis["count"]) * 0.4 +
                analysis["avg_confidence"] * 0.3 +
                (1.0 - min(len(analysis["issues"]) / (analysis["count"] * 2), 1.0)) * 0.3
            )
        
        analysis["score"] = score
        return analysis
    
    def _analyze_question_quality(self, questions: List) -> Dict[str, Any]:
        """Analyze question quality and distribution."""
        if not questions:
            return {"score": 0.0, "issues": ["No questions generated"]}
        
        # Cognitive level distribution
        level_counts = {}
        for q in questions:
            level = q.cognitive_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        analysis = {
            "count": len(questions),
            "level_distribution": level_counts,
            "with_answers": sum(1 for q in questions if q.expected_answer),
            "avg_difficulty": sum(q.difficulty for q in questions) / len(questions),
            "issues": []
        }
        
        # Check distribution
        expected_levels = ["remember", "understand", "apply", "analyze"]
        for level in expected_levels:
            if level not in level_counts:
                analysis["issues"].append(f"Missing {level} level questions")
        
        # Check question quality
        for q in questions:
            if not q.expected_answer:
                analysis["issues"].append(f"Question lacks answer: '{q.question_text[:50]}...'")
            elif len(q.expected_answer) < 20:
                analysis["issues"].append(f"Question has brief answer: '{q.question_text[:50]}...'")
        
        # Calculate score
        distribution_score = len(level_counts) / 6.0  # 6 possible levels
        answer_ratio = analysis["with_answers"] / analysis["count"]
        
        score = (
            distribution_score * 0.4 +
            answer_ratio * 0.4 +
            (1.0 - min(len(analysis["issues"]) / analysis["count"], 1.0)) * 0.2
        )
        
        analysis["score"] = score
        return analysis
    
    def _analyze_summary_quality(self, summary: str, topics: List, facts: List) -> Dict[str, Any]:
        """Analyze summary quality."""
        if not summary:
            return {"score": 0.0, "issues": ["No summary provided"]}
        
        analysis = {
            "length": len(summary),
            "sentence_count": summary.count('.') + summary.count('!') + summary.count('?'),
            "covers_topics": 0,
            "issues": []
        }
        
        # Check topic coverage
        summary_lower = summary.lower()
        for topic in topics[:3]:  # Check top 3 topics
            if topic.name.lower() in summary_lower:
                analysis["covers_topics"] += 1
        
        # Check quality issues
        if analysis["length"] < 100:
            analysis["issues"].append("Summary too brief")
        if analysis["length"] > 400:
            analysis["issues"].append("Summary too long")
        if analysis["sentence_count"] < 2:
            analysis["issues"].append("Summary needs more sentences")
        if '•' in summary or '-' in summary[:3]:
            analysis["issues"].append("Summary uses bullet points instead of prose")
        
        # Check academic prose
        academic_terms = ['demonstrates', 'indicates', 'suggests', 'reveals']
        if not any(term in summary_lower for term in academic_terms):
            analysis["issues"].append("Summary lacks academic prose")
        
        # Calculate score
        topic_coverage = analysis["covers_topics"] / min(3, len(topics)) if topics else 0
        length_score = 1.0 if 100 <= analysis["length"] <= 300 else 0.7
        prose_score = 0.0 if '•' in summary else 1.0
        
        score = (
            topic_coverage * 0.4 +
            length_score * 0.3 +
            prose_score * 0.2 +
            (1.0 - min(len(analysis["issues"]) / 4, 1.0)) * 0.1
        )
        
        analysis["score"] = score
        return analysis
    
    def _assess_academic_quality(self, extraction: ExtractedKnowledge) -> float:
        """Assess overall academic quality of extraction."""
        score = 1.0
        
        # Check topic descriptions
        for topic in extraction.topics:
            if topic.description and len(topic.description.split('.')) < 2:
                score -= 0.05
        
        # Check fact evidence
        facts_with_evidence = sum(1 for f in extraction.facts if f.evidence)
        if extraction.facts:
            evidence_ratio = facts_with_evidence / len(extraction.facts)
            if evidence_ratio < 0.8:
                score -= 0.1
        
        # Check summary prose
        if extraction.summary:
            if any(marker in extraction.summary for marker in ['•', '*', '-']):
                score -= 0.15
        
        # Check question quality
        high_level_questions = sum(
            1 for q in extraction.questions 
            if q.cognitive_level.value in ['analyze', 'evaluate', 'create']
        )
        if extraction.questions and high_level_questions < 2:
            score -= 0.1
        
        return max(0.3, score)
    
    async def _extract_with_targeted_improvements(
        self,
        pdf_path: str,
        previous_extraction: ExtractedKnowledge,
        quality_analysis: Dict[str, Any],
        improvement_strategies: List[Dict[str, Any]],
        source_content: str
    ) -> ExtractedKnowledge:
        """
        Re-extract with targeted improvements based on analysis.
        
        Args:
            pdf_path: Path to PDF
            previous_extraction: Previous extraction attempt
            quality_analysis: Detailed quality analysis
            improvement_strategies: Specific improvement strategies
            source_content: Source content for reference
            
        Returns:
            Improved extraction
        """
        logger.info("Performing targeted improvement extraction")
        
        # Build comprehensive improvement prompt
        improvement_prompt = self._build_targeted_improvement_prompt(
            quality_analysis,
            improvement_strategies,
            previous_extraction,
            source_content
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": ENHANCED_FEEDBACK_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": improvement_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Parse improved extraction
            response_text = response.choices[0].message.content
            return self._parse_improved_extraction(response_text, previous_extraction)
            
        except Exception as e:
            logger.error(f"Targeted improvement failed: {str(e)}")
            return previous_extraction
    
    def _build_targeted_improvement_prompt(
        self,
        quality_analysis: Dict[str, Any],
        improvement_strategies: List[Dict[str, Any]],
        previous_extraction: ExtractedKnowledge,
        source_content: str
    ) -> str:
        """Build comprehensive improvement prompt."""
        
        prompt_parts = [
            "Improve the extraction based on the following quality analysis and strategies.",
            f"\nCurrent Quality Score: {quality_analysis['overall_score']:.2f}",
            "\n## Quality Analysis"
        ]
        
        # Add dimension scores
        prompt_parts.append("\n### Dimension Scores:")
        for dim, score in quality_analysis["dimension_scores"].items():
            status = "✓" if score > 0.7 else "✗"
            prompt_parts.append(f"{status} {dim.title()}: {score:.2f}")
        
        # Add detailed issues
        prompt_parts.append("\n### Specific Issues:")
        
        # Topic issues
        topic_analysis = quality_analysis["detailed_metrics"]["topics"]
        if topic_analysis["issues"]:
            prompt_parts.append("\nTopic Issues:")
            for issue in topic_analysis["issues"][:3]:
                prompt_parts.append(f"- {issue}")
        
        # Fact issues
        fact_analysis = quality_analysis["detailed_metrics"]["facts"]
        if fact_analysis["issues"]:
            prompt_parts.append("\nFact Issues:")
            for issue in fact_analysis["issues"][:5]:
                prompt_parts.append(f"- {issue}")
        
        # Relationship issues
        rel_analysis = quality_analysis["detailed_metrics"]["relationships"]
        if rel_analysis["issues"]:
            prompt_parts.append("\nRelationship Issues:")
            for issue in rel_analysis["issues"][:3]:
                prompt_parts.append(f"- {issue}")
        
        # Question issues
        q_analysis = quality_analysis["detailed_metrics"]["questions"]
        if q_analysis["issues"]:
            prompt_parts.append("\nQuestion Issues:")
            for issue in q_analysis["issues"][:3]:
                prompt_parts.append(f"- {issue}")
        
        # Add improvement strategies
        prompt_parts.append("\n## Improvement Strategies")
        prompt_parts.append("<thinking>")
        prompt_parts.append("Based on the analysis, I need to:")
        for strategy in improvement_strategies:
            prompt_parts.append(f"- {strategy['action']} (Priority: {strategy['priority']})")
        prompt_parts.append("</thinking>")
        
        # Add specific instructions
        prompt_parts.append("\n## Improvement Instructions")
        for i, strategy in enumerate(improvement_strategies, 1):
            prompt_parts.append(f"\n{i}. {strategy['instruction']}")
        
        # Add previous extraction
        prompt_parts.append("\n## Previous Extraction to Improve")
        prompt_parts.append(json.dumps({
            "topics": [{"name": t.name, "description": t.description, "keywords": t.keywords}
                      for t in previous_extraction.topics],
            "facts": [{"claim": f.claim, "evidence": f.evidence, "confidence": f.confidence}
                     for f in previous_extraction.facts],
            "relationships": [{"source": r.source_entity, "target": r.target_entity,
                             "type": r.relationship_type, "description": r.description}
                            for r in previous_extraction.relationships],
            "questions": [{"text": q.question_text, "answer": q.expected_answer,
                         "level": q.cognitive_level.value, "difficulty": q.difficulty}
                        for q in previous_extraction.questions],
            "summary": previous_extraction.summary
        }, indent=2))
        
        # Add source excerpt
        if source_content:
            prompt_parts.append(f"\n## Source Content Excerpt (first 1500 chars)")
            prompt_parts.append(source_content[:1500] + "...")
        
        # Final instructions
        prompt_parts.append("\n## Output Requirements")
        prompt_parts.append("1. Address ALL identified issues systematically")
        prompt_parts.append("2. Maintain academic prose throughout (no bullet points in descriptions/summary)")
        prompt_parts.append("3. Ensure all facts have strong evidence")
        prompt_parts.append("4. Create questions across all cognitive levels")
        prompt_parts.append("5. Write a flowing, comprehensive summary")
        prompt_parts.append("\nProvide the improved extraction as a JSON object.")
        
        return "\n".join(prompt_parts)
    
    def _develop_improvement_strategies(
        self,
        quality_analysis: Dict[str, Any],
        extraction: ExtractedKnowledge
    ) -> List[Dict[str, Any]]:
        """Develop specific improvement strategies based on analysis."""
        strategies = []
        
        # Prioritize by dimension scores
        dim_scores = quality_analysis["dimension_scores"]
        
        # Consistency improvements
        if dim_scores.get("consistency", 1.0) < 0.7:
            strategies.append({
                "dimension": "consistency",
                "priority": "high",
                "action": "Resolve contradictions and align all elements",
                "instruction": "Review all facts and relationships for contradictions. Ensure entities in relationships are defined in topics or facts. Make all elements work together coherently."
            })
        
        # Grounding improvements
        if dim_scores.get("grounding", 1.0) < 0.7:
            strategies.append({
                "dimension": "grounding",
                "priority": "high",
                "action": "Strengthen evidence for all claims",
                "instruction": "For each fact without evidence, find supporting text from the source. Replace vague claims with specific, verifiable statements. Ensure all content is traceable to the source material."
            })
        
        # Completeness improvements
        if dim_scores.get("completeness", 1.0) < 0.7:
            strategies.append({
                "dimension": "completeness",
                "priority": "medium",
                "action": "Expand coverage of key concepts",
                "instruction": "Add missing important topics, facts about uncovered aspects, and relationships between key entities. Ensure questions cover all major topics."
            })
        
        # Academic quality improvements
        if quality_analysis.get("academic_quality", 1.0) < 0.7:
            strategies.append({
                "dimension": "academic_quality",
                "priority": "high",
                "action": "Enhance academic prose and depth",
                "instruction": "Rewrite all descriptions and the summary in flowing academic prose (no bullet points). Add scholarly language and ensure each topic description is 2-3 complete sentences."
            })
        
        # Question distribution improvements
        q_analysis = quality_analysis["detailed_metrics"]["questions"]
        if "level_distribution" in q_analysis:
            missing_levels = ["remember", "understand", "apply", "analyze"]
            for level in missing_levels:
                if level not in q_analysis["level_distribution"]:
                    strategies.append({
                        "dimension": "questions",
                        "priority": "medium",
                        "action": f"Add {level}-level questions",
                        "instruction": f"Create at least one {level}-level question that truly tests this cognitive level. Include comprehensive expected answers."
                    })
        
        # Summary improvements
        summary_analysis = quality_analysis["detailed_metrics"]["summary"]
        if summary_analysis["score"] < 0.7:
            strategies.append({
                "dimension": "summary",
                "priority": "medium",
                "action": "Rewrite summary with academic prose",
                "instruction": "Write a 3-4 sentence summary in flowing academic prose that synthesizes main topics, key findings, and significance. No bullet points or lists."
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        strategies.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return strategies[:5]  # Limit to top 5 strategies
    
    def _parse_improved_extraction(self, response_text: str, 
                                  previous: ExtractedKnowledge) -> ExtractedKnowledge:
        """Parse improved extraction with fallbacks."""
        try:
            # Clean response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Create improved extraction with validation
            from extraction.models import Topic, Fact, Relationship, Question, CognitiveLevel
            
            # Parse topics
            topics = []
            for t in data.get("topics", []):
                if isinstance(t, dict) and "name" in t:
                    topics.append(Topic(
                        name=t["name"],
                        description=t.get("description", ""),
                        keywords=t.get("keywords", []),
                        confidence=float(t.get("confidence", 0.7))
                    ))
            
            # Parse facts
            facts = []
            for f in data.get("facts", []):
                if isinstance(f, dict) and "claim" in f:
                    facts.append(Fact(
                        claim=f["claim"],
                        evidence=f.get("evidence", ""),
                        confidence=float(f.get("confidence", 0.7))
                    ))
            
            # Parse relationships
            relationships = []
            for r in data.get("relationships", []):
                if isinstance(r, dict) and all(k in r for k in ["source", "target"]):
                    relationships.append(Relationship(
                        source_entity=r["source"],
                        target_entity=r["target"],
                        relationship_type=r.get("type", "related_to"),
                        description=r.get("description", ""),
                        confidence=float(r.get("confidence", 0.7))
                    ))
            
            # Parse questions
            questions = []
            for q in data.get("questions", []):
                if isinstance(q, dict) and "text" in q:
                    try:
                        level = CognitiveLevel(q.get("level", "understand"))
                    except:
                        level = CognitiveLevel.UNDERSTAND
                    
                    questions.append(Question(
                        question_text=q["text"],
                        expected_answer=q.get("answer", ""),
                        cognitive_level=level,
                        difficulty=int(q.get("difficulty", 3)),
                        confidence=float(q.get("confidence", 0.7))
                    ))
            
            # Create improved extraction
            return ExtractedKnowledge(
                topics=topics,
                facts=facts,
                relationships=relationships,
                questions=questions,
                summary=data.get("summary", previous.summary),
                overall_confidence=float(data.get("overall_confidence", 0.7)),
                extraction_metadata={
                    **previous.extraction_metadata,
                    "improved": True,
                    "improvement_iteration": previous.extraction_metadata.get("improvement_iteration", 0) + 1
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse improved extraction: {str(e)}")
            return previous
    
    async def _perform_final_analysis(
        self,
        extraction: ExtractedKnowledge,
        source_content: str,
        iteration_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive final analysis."""
        # Get standard analysis
        analysis = await self._analyze_extraction_quality(extraction, source_content)
        
        # Add improvement summary
        analysis["improvement_summary"] = {
            "iterations": len(iteration_history),
            "initial_score": iteration_history[0]["overall_score"],
            "final_score": analysis["overall_score"],
            "total_improvement": analysis["overall_score"] - iteration_history[0]["overall_score"],
            "improvements_by_dimension": {}
        }
        
        # Track dimension improvements
        initial_dims = iteration_history[0]["quality_analysis"]["dimension_scores"]
        final_dims = analysis["dimension_scores"]
        
        for dim in final_dims:
            if dim in initial_dims:
                improvement = final_dims[dim] - initial_dims[dim]
                analysis["improvement_summary"]["improvements_by_dimension"][dim] = {
                    "initial": initial_dims[dim],
                    "final": final_dims[dim],
                    "improvement": improvement
                }
        
        # Identify remaining issues
        remaining_issues = []
        
        for dim, score in final_dims.items():
            if score < 0.7:
                remaining_issues.append({
                    "dimension": dim,
                    "score": score,
                    "description": f"{dim.title()} still needs improvement"
                })
        
        # Add specific remaining issues
        for metric_type, metric_data in analysis["detailed_metrics"].items():
            if isinstance(metric_data, dict) and "issues" in metric_data:
                for issue in metric_data["issues"][:2]:  # Top 2 issues per type
                    remaining_issues.append({
                        "type": metric_type,
                        "issue": issue
                    })
        
        analysis["remaining_issues"] = remaining_issues
        
        return analysis
    
    def _summarize_iteration_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create concise summary of iteration history."""
        summary = []
        
        for h in history:
            summary.append({
                "iteration": h["iteration"],
                "overall_score": h["overall_score"],
                "dimension_scores": h["quality_analysis"]["dimension_scores"],
                "key_improvements": [s["action"] for s in h.get("improvement_strategies", [])[:3]],
                "issues_count": sum(
                    len(m.get("issues", [])) 
                    for m in h["quality_analysis"]["detailed_metrics"].values()
                    if isinstance(m, dict)
                )
            })
        
        return summary
    
    def _identify_key_improvements(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify the most significant improvements made."""
        if len(history) < 2:
            return []
        
        improvements = []
        
        # Compare first and last iterations
        first = history[0]["quality_analysis"]
        last = history[-1]["quality_analysis"]
        
        # Dimension improvements
        for dim in first["dimension_scores"]:
            if dim in last["dimension_scores"]:
                improvement = last["dimension_scores"][dim] - first["dimension_scores"][dim]
                if improvement > 0.1:
                    improvements.append(
                        f"{dim.title()} improved by {improvement:.2f}"
                    )
        
        # Specific metric improvements
        first_facts = first["detailed_metrics"]["facts"]
        last_facts = last["detailed_metrics"]["facts"]
        
        if "with_evidence" in first_facts and "with_evidence" in last_facts:
            if last_facts["with_evidence"] > first_facts["with_evidence"]:
                improvements.append(
                    f"Added evidence to {last_facts['with_evidence'] - first_facts['with_evidence']} facts"
                )
        
        # Question improvements
        first_q = first["detailed_metrics"]["questions"]
        last_q = last["detailed_metrics"]["questions"]
        
        if "level_distribution" in last_q:
            new_levels = set(last_q["level_distribution"].keys()) - set(first_q.get("level_distribution", {}).keys())
            if new_levels:
                improvements.append(f"Added questions for levels: {', '.join(new_levels)}")
        
        # Academic quality
        if "academic_quality" in last and "academic_quality" in first:
            if last["academic_quality"] > first["academic_quality"] + 0.1:
                improvements.append("Enhanced academic prose quality")
        
        return improvements[:5]  # Top 5 improvements
    
    async def _get_source_content(self, pdf_path: str) -> str:
        """Get source content from PDF for grounding checks."""
        try:
            from extraction.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            chunks = await processor.process_pdf(pdf_path)
            return "\n".join([chunk.content for chunk in chunks])
        except Exception as e:
            logger.error(f"Failed to get source content: {str(e)}")
            return ""
    
    async def compare_with_baseline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Compare enhanced feedback extraction with baseline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Detailed comparison results
        """
        logger.info(f"Running enhanced feedback comparison for: {pdf_path}")
        
        # Run baseline extraction
        baseline_result = await self.base_extractor.extract(pdf_path)
        
        # Run enhanced feedback extraction
        feedback_result = await self.extract_with_enhanced_feedback(pdf_path)
        
        # Get source for scoring
        source_content = await self._get_source_content(pdf_path)
        
        # Score both
        baseline_analysis = await self._analyze_extraction_quality(baseline_result, source_content)
        
        # Detailed comparison
        comparison = {
            "pdf_path": pdf_path,
            "baseline": {
                "overall_score": baseline_analysis["overall_score"],
                "dimension_scores": baseline_analysis["dimension_scores"],
                "academic_quality": baseline_analysis["academic_quality"],
                "topics": len(baseline_result.topics),
                "facts": len(baseline_result.facts),
                "questions": len(baseline_result.questions),
                "iterations": 1
            },
            "enhanced_feedback": {
                "overall_score": feedback_result["quality_analysis"]["overall_score"],
                "dimension_scores": feedback_result["quality_analysis"]["dimension_scores"],
                "academic_quality": feedback_result["quality_analysis"]["academic_quality"],
                "topics": len(feedback_result["extraction"].topics),
                "facts": len(feedback_result["extraction"].facts),
                "questions": len(feedback_result["extraction"].questions),
                "iterations": feedback_result["iterations"]
            },
            "improvements": {
                "overall_gain": (feedback_result["quality_analysis"]["overall_score"] - 
                               baseline_analysis["overall_score"]),
                "dimension_gains": {},
                "quality_gains": {
                    "facts_with_evidence": sum(1 for f in feedback_result["extraction"].facts if f.evidence) -
                                         sum(1 for f in baseline_result.facts if f.evidence),
                    "question_levels": len(set(q.cognitive_level for q in feedback_result["extraction"].questions)) -
                                     len(set(q.cognitive_level for q in baseline_result.questions)),
                    "academic_improvement": (feedback_result["quality_analysis"]["academic_quality"] -
                                           baseline_analysis["academic_quality"])
                }
            },
            "efficiency": {
                "iterations_used": feedback_result["iterations"],
                "improvement_per_iteration": feedback_result["total_improvement"] / feedback_result["iterations"]
            }
        }
        
        # Calculate dimension gains
        for dim in baseline_analysis["dimension_scores"]:
            if dim in feedback_result["quality_analysis"]["dimension_scores"]:
                gain = (feedback_result["quality_analysis"]["dimension_scores"][dim] -
                       baseline_analysis["dimension_scores"][dim])
                comparison["improvements"]["dimension_gains"][dim] = gain
        
        return comparison