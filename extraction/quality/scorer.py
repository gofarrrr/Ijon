"""
Quality scorer for extracted knowledge.

Evaluates extraction quality across multiple dimensions:
- Consistency: Internal logical consistency
- Grounding: Evidence and support for claims
- Coherence: Overall narrative and structure
- Completeness: Coverage of important information
"""

import asyncio
from typing import List, Dict, Any, Tuple
import statistics
from datetime import datetime

from extraction.models import (
    ExtractedKnowledge, Fact, Topic, Relationship, 
    Question, ConfidenceLevel, CognitiveLevel
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QualityScorer:
    """Scores extraction quality across multiple dimensions."""
    
    def __init__(self):
        """Initialize the quality scorer."""
        self.weights = {
            "consistency": 0.25,
            "grounding": 0.30,
            "coherence": 0.25,
            "completeness": 0.20
        }
        logger.info("Initialized QualityScorer")
    
    def score_extraction(self, extraction: ExtractedKnowledge, 
                        source_content: str = None) -> Dict[str, Any]:
        """
        Score extraction quality across all dimensions.
        
        Args:
            extraction: The extracted knowledge to score
            source_content: Original document content for grounding checks
            
        Returns:
            Quality scores and detailed feedback
        """
        logger.info(f"Scoring extraction with {len(extraction.facts)} facts, "
                   f"{len(extraction.topics)} topics")
        
        # Score individual dimensions
        consistency_score = self._score_consistency(extraction)
        grounding_score = self._score_grounding(extraction, source_content)
        coherence_score = self._score_coherence(extraction)
        completeness_score = self._score_completeness(extraction)
        
        # Calculate weighted overall score
        dimension_scores = {
            "consistency": consistency_score["score"],
            "grounding": grounding_score["score"],
            "coherence": coherence_score["score"],
            "completeness": completeness_score["score"]
        }
        
        overall_score = sum(
            score * self.weights[dimension] 
            for dimension, score in dimension_scores.items()
        )
        
        # Identify weaknesses
        weaknesses = []
        for dimension, score in dimension_scores.items():
            if score < 0.6:
                weaknesses.append({
                    "dimension": dimension,
                    "score": score,
                    "severity": "high" if score < 0.4 else "medium"
                })
        
        # Generate improvement suggestions
        suggestions = self._generate_suggestions(
            consistency_score, grounding_score, 
            coherence_score, completeness_score
        )
        
        return {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "detailed_scores": {
                "consistency": consistency_score,
                "grounding": grounding_score,
                "coherence": coherence_score,
                "completeness": completeness_score
            },
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "needs_reextraction": overall_score < 0.6,
            "scored_at": datetime.utcnow().isoformat()
        }
    
    def _score_consistency(self, extraction: ExtractedKnowledge) -> Dict[str, Any]:
        """Score internal logical consistency."""
        issues = []
        scores = []
        
        # Check fact consistency
        fact_texts = [f.claim.lower() for f in extraction.facts]
        for i, fact1 in enumerate(extraction.facts):
            for j, fact2 in enumerate(extraction.facts[i+1:], i+1):
                # Check for contradictions
                if self._facts_contradict(fact1.claim, fact2.claim):
                    issues.append({
                        "type": "contradiction",
                        "fact1": fact1.claim[:100],
                        "fact2": fact2.claim[:100]
                    })
                    scores.append(0.0)
                else:
                    scores.append(1.0)
        
        # Check relationship consistency
        for rel in extraction.relationships:
            # Verify entities exist in topics or facts
            entities_found = 0
            for topic in extraction.topics:
                if rel.source_entity.lower() in topic.name.lower() or \
                   rel.target_entity.lower() in topic.name.lower():
                    entities_found += 1
            
            entity_score = min(1.0, entities_found / 2)
            scores.append(entity_score)
            
            if entity_score < 0.5:
                issues.append({
                    "type": "missing_entity",
                    "relationship": f"{rel.source_entity} -> {rel.target_entity}"
                })
        
        # Check confidence consistency
        confidence_values = [f.confidence for f in extraction.facts]
        if confidence_values:
            cv = statistics.stdev(confidence_values) if len(confidence_values) > 1 else 0
            confidence_consistency = 1.0 - min(1.0, cv)
            scores.append(confidence_consistency)
        
        final_score = statistics.mean(scores) if scores else 0.5
        
        return {
            "score": final_score,
            "issues": issues,
            "total_checks": len(scores),
            "passed_checks": sum(1 for s in scores if s > 0.5)
        }
    
    def _score_grounding(self, extraction: ExtractedKnowledge, 
                        source_content: str = None) -> Dict[str, Any]:
        """Score how well claims are grounded in evidence."""
        issues = []
        scores = []
        
        # Check facts have evidence
        for fact in extraction.facts:
            if fact.evidence:
                # Evidence exists
                evidence_score = 0.8
                
                # Bonus for specific evidence markers
                if any(marker in fact.evidence.lower() for marker in 
                      ["study", "research", "data", "report", "survey"]):
                    evidence_score = 1.0
                
                scores.append(evidence_score)
            else:
                # No evidence
                scores.append(0.2)
                issues.append({
                    "type": "missing_evidence",
                    "fact": fact.claim[:100]
                })
        
        # Check if facts appear in source (if available)
        if source_content:
            source_lower = source_content.lower()
            for fact in extraction.facts:
                # Simple substring check
                key_terms = self._extract_key_terms(fact.claim)
                terms_found = sum(1 for term in key_terms if term in source_lower)
                source_score = min(1.0, terms_found / max(1, len(key_terms)))
                scores.append(source_score)
                
                if source_score < 0.3:
                    issues.append({
                        "type": "not_in_source",
                        "fact": fact.claim[:100]
                    })
        
        # Check topic grounding
        for topic in extraction.topics:
            if topic.description and len(topic.description) > 20:
                scores.append(0.8)
            else:
                scores.append(0.3)
                issues.append({
                    "type": "vague_topic",
                    "topic": topic.name
                })
        
        final_score = statistics.mean(scores) if scores else 0.5
        
        return {
            "score": final_score,
            "issues": issues,
            "facts_with_evidence": sum(1 for f in extraction.facts if f.evidence),
            "total_facts": len(extraction.facts)
        }
    
    def _score_coherence(self, extraction: ExtractedKnowledge) -> Dict[str, Any]:
        """Score overall narrative coherence and structure."""
        issues = []
        scores = []
        
        # Check summary coherence
        if extraction.summary:
            # Summary should reference main topics
            summary_lower = extraction.summary.lower()
            topics_referenced = sum(
                1 for topic in extraction.topics 
                if topic.name.lower() in summary_lower
            )
            topic_coverage = min(1.0, topics_referenced / max(1, len(extraction.topics)))
            scores.append(topic_coverage)
            
            if topic_coverage < 0.5:
                issues.append({
                    "type": "summary_topic_mismatch",
                    "topics_in_summary": topics_referenced,
                    "total_topics": len(extraction.topics)
                })
        else:
            scores.append(0.0)
            issues.append({"type": "missing_summary"})
        
        # Check question relevance
        for question in extraction.questions:
            # Questions should relate to topics or facts
            q_lower = question.question_text.lower()
            relevance_score = 0.0
            
            for topic in extraction.topics:
                if any(word in q_lower for word in topic.name.lower().split()):
                    relevance_score = max(relevance_score, 0.8)
            
            for fact in extraction.facts[:5]:  # Check first 5 facts
                if any(word in q_lower for word in fact.claim.lower().split()[:5]):
                    relevance_score = max(relevance_score, 0.7)
            
            scores.append(relevance_score)
            
            if relevance_score < 0.3:
                issues.append({
                    "type": "irrelevant_question",
                    "question": question.question_text[:100]
                })
        
        # Check relationship coherence
        if extraction.relationships:
            # Relationships should form some connected structure
            entity_graph = {}
            for rel in extraction.relationships:
                if rel.source_entity not in entity_graph:
                    entity_graph[rel.source_entity] = []
                entity_graph[rel.source_entity].append(rel.target_entity)
            
            # Simple connectivity check
            connected_entities = len(entity_graph)
            total_entities = len(set(
                [r.source_entity for r in extraction.relationships] +
                [r.target_entity for r in extraction.relationships]
            ))
            
            connectivity_score = min(1.0, connected_entities / max(1, total_entities))
            scores.append(connectivity_score)
        
        final_score = statistics.mean(scores) if scores else 0.5
        
        return {
            "score": final_score,
            "issues": issues,
            "has_summary": bool(extraction.summary),
            "question_relevance": statistics.mean([s for s in scores[-len(extraction.questions):]])
                                if extraction.questions else 0.0
        }
    
    def _score_completeness(self, extraction: ExtractedKnowledge) -> Dict[str, Any]:
        """Score extraction completeness."""
        issues = []
        scores = []
        
        # Check component presence
        components = {
            "topics": len(extraction.topics) >= 2,
            "facts": len(extraction.facts) >= 3,
            "relationships": len(extraction.relationships) >= 1,
            "questions": len(extraction.questions) >= 2,
            "summary": bool(extraction.summary)
        }
        
        component_score = sum(components.values()) / len(components)
        scores.append(component_score)
        
        for comp, present in components.items():
            if not present:
                issues.append({
                    "type": "missing_component",
                    "component": comp
                })
        
        # Check topic coverage
        if extraction.topics:
            # Topics should have good descriptions
            desc_scores = []
            for topic in extraction.topics:
                if topic.description:
                    desc_length = len(topic.description)
                    desc_score = min(1.0, desc_length / 100)  # 100 chars is good
                    desc_scores.append(desc_score)
                else:
                    desc_scores.append(0.0)
            
            topic_quality = statistics.mean(desc_scores)
            scores.append(topic_quality)
        
        # Check question diversity
        if extraction.questions:
            # Questions should span cognitive levels
            cognitive_levels = set(q.cognitive_level for q in extraction.questions)
            diversity_score = len(cognitive_levels) / 6  # 6 levels in Bloom's
            scores.append(diversity_score)
            
            if diversity_score < 0.3:
                issues.append({
                    "type": "low_question_diversity",
                    "levels_used": len(cognitive_levels)
                })
        
        # Check fact detail
        if extraction.facts:
            detailed_facts = sum(1 for f in extraction.facts 
                               if len(f.claim) > 50 and f.evidence)
            detail_score = detailed_facts / len(extraction.facts)
            scores.append(detail_score)
        
        final_score = statistics.mean(scores) if scores else 0.5
        
        return {
            "score": final_score,
            "issues": issues,
            "component_scores": components,
            "total_items": sum([
                len(extraction.topics),
                len(extraction.facts),
                len(extraction.relationships),
                len(extraction.questions)
            ])
        }
    
    def _facts_contradict(self, claim1: str, claim2: str) -> bool:
        """Simple contradiction detection."""
        # Very basic implementation - looks for opposite indicators
        negations = ["not", "no", "never", "none", "neither"]
        opposites = [("increase", "decrease"), ("rise", "fall"), 
                    ("positive", "negative"), ("success", "failure")]
        
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        # Check for direct negation
        for neg in negations:
            if neg in claim1_lower and neg not in claim2_lower:
                # Check if they're about the same thing
                if self._claims_overlap(claim1_lower, claim2_lower) > 0.5:
                    return True
        
        # Check for opposites
        for word1, word2 in opposites:
            if word1 in claim1_lower and word2 in claim2_lower:
                if self._claims_overlap(claim1_lower, claim2_lower) > 0.5:
                    return True
        
        return False
    
    def _claims_overlap(self, claim1: str, claim2: str) -> float:
        """Calculate overlap between two claims."""
        words1 = set(claim1.split())
        words2 = set(claim2.split())
        
        # Remove common words
        common_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at"}
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return 0.0
        
        overlap = len(words1 & words2)
        return overlap / min(len(words1), len(words2))
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple implementation - takes longer words
        words = text.lower().split()
        return [w for w in words if len(w) > 4 and w.isalpha()][:5]
    
    def _generate_suggestions(self, consistency: Dict, grounding: Dict,
                            coherence: Dict, completeness: Dict) -> List[Dict[str, str]]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Consistency suggestions
        if consistency["score"] < 0.6:
            if any(i["type"] == "contradiction" for i in consistency["issues"]):
                suggestions.append({
                    "dimension": "consistency",
                    "suggestion": "Review and resolve contradictory facts",
                    "priority": "high"
                })
            if any(i["type"] == "missing_entity" for i in consistency["issues"]):
                suggestions.append({
                    "dimension": "consistency",
                    "suggestion": "Ensure all relationship entities are defined as topics",
                    "priority": "medium"
                })
        
        # Grounding suggestions
        if grounding["score"] < 0.6:
            if grounding["facts_with_evidence"] < grounding["total_facts"] * 0.5:
                suggestions.append({
                    "dimension": "grounding",
                    "suggestion": "Add supporting evidence for facts",
                    "priority": "high"
                })
            if any(i["type"] == "not_in_source" for i in grounding["issues"]):
                suggestions.append({
                    "dimension": "grounding",
                    "suggestion": "Verify facts against source document",
                    "priority": "high"
                })
        
        # Coherence suggestions
        if coherence["score"] < 0.6:
            if not coherence["has_summary"]:
                suggestions.append({
                    "dimension": "coherence",
                    "suggestion": "Add a comprehensive summary",
                    "priority": "medium"
                })
            if coherence.get("question_relevance", 1.0) < 0.5:
                suggestions.append({
                    "dimension": "coherence",
                    "suggestion": "Ensure questions relate to main topics",
                    "priority": "medium"
                })
        
        # Completeness suggestions
        if completeness["score"] < 0.6:
            missing = [k for k, v in completeness["component_scores"].items() if not v]
            if missing:
                suggestions.append({
                    "dimension": "completeness",
                    "suggestion": f"Add missing components: {', '.join(missing)}",
                    "priority": "high"
                })
        
        return suggestions