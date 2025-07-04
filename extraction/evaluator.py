"""
Evaluation framework for the extraction system.

Provides metrics calculation, manual review interface, and reporting capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

from .models import ExtractedKnowledge, Fact, Question, Topic, Relationship


@dataclass
class ExtractionMetrics:
    """Metrics for extraction evaluation."""
    chunk_id: str
    total_topics: int
    total_facts: int
    total_relationships: int
    total_questions: int
    
    # Quality metrics
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    
    # Question diversity
    question_cognitive_levels: Dict[str, int]
    avg_question_difficulty: float
    
    # Validation results
    facts_validated: int = 0
    facts_accurate: int = 0
    facts_accuracy_rate: float = 0.0
    
    questions_validated: int = 0
    questions_answerable: int = 0
    questions_answerability_rate: float = 0.0
    
    # Processing metrics
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    
    # Issues found
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


@dataclass
class StageMetrics:
    """Aggregate metrics for a stage."""
    stage: int
    total_chunks: int
    total_documents: int
    
    # Aggregate counts
    total_facts_extracted: int
    total_questions_generated: int
    total_topics_identified: int
    total_relationships_found: int
    
    # Quality metrics
    overall_accuracy: float
    overall_answerability: float
    avg_confidence: float
    
    # Performance metrics
    avg_processing_time_ms: float
    total_tokens_used: int
    estimated_cost_usd: float
    
    # Success criteria
    meets_accuracy_target: bool  # 70%+ for Stage 1
    meets_answerability_target: bool  # 60%+ for Stage 1
    
    # Failure analysis
    failure_patterns: Dict[str, int]
    document_type_performance: Dict[str, Dict[str, float]]


class ExtractionEvaluator:
    """Evaluates extraction quality and generates metrics."""
    
    def __init__(self, output_dir: str = "extraction/evaluation"):
        """Initialize evaluator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation results storage
        self.validation_results = []
        
    def calculate_metrics(self, extraction: ExtractedKnowledge) -> ExtractionMetrics:
        """Calculate metrics for a single extraction."""
        
        # Collect all confidence scores
        all_confidences = []
        all_confidences.extend([t.confidence for t in extraction.topics])
        all_confidences.extend([f.confidence for f in extraction.facts])
        all_confidences.extend([r.confidence for r in extraction.relationships])
        all_confidences.extend([q.confidence for q in extraction.questions])
        
        # Question cognitive level distribution
        cognitive_levels = {}
        for q in extraction.questions:
            level = q.cognitive_level.value
            cognitive_levels[level] = cognitive_levels.get(level, 0) + 1
        
        # Calculate metrics
        metrics = ExtractionMetrics(
            chunk_id=extraction.chunk_id,
            total_topics=len(extraction.topics),
            total_facts=len(extraction.facts),
            total_relationships=len(extraction.relationships),
            total_questions=len(extraction.questions),
            avg_confidence=statistics.mean(all_confidences) if all_confidences else 0.0,
            min_confidence=min(all_confidences) if all_confidences else 0.0,
            max_confidence=max(all_confidences) if all_confidences else 0.0,
            question_cognitive_levels=cognitive_levels,
            avg_question_difficulty=statistics.mean([q.difficulty for q in extraction.questions]) if extraction.questions else 0.0,
            processing_time_ms=extraction.extraction_metadata.get("processing_time_ms", 0.0)
        )
        
        # Check for potential issues
        if metrics.avg_confidence < 0.5:
            metrics.issues.append("Low average confidence")
        
        if metrics.total_facts < 3:
            metrics.issues.append("Few facts extracted")
            
        if metrics.total_questions < 3:
            metrics.issues.append("Few questions generated")
            
        if len(cognitive_levels) < 2:
            metrics.issues.append("Low question diversity")
        
        return metrics
    
    def validate_facts_manual(self, extraction: ExtractedKnowledge, source_text: str) -> List[Dict[str, Any]]:
        """
        Interface for manual fact validation.
        
        Returns list of validation results for each fact.
        """
        validation_results = []
        
        print("\n" + "="*70)
        print("MANUAL FACT VALIDATION")
        print("="*70)
        print(f"Chunk ID: {extraction.chunk_id}")
        print(f"Total facts to validate: {len(extraction.facts)}")
        print("\nSource text preview:")
        print(source_text[:500] + "..." if len(source_text) > 500 else source_text)
        print("\n" + "-"*70)
        
        for i, fact in enumerate(extraction.facts, 1):
            print(f"\n[Fact {i}/{len(extraction.facts)}]")
            print(f"Claim: {fact.claim}")
            print(f"Evidence: {fact.evidence}")
            print(f"Confidence: {fact.confidence:.2f}")
            
            # Get validation input
            while True:
                result = input("\nIs this fact accurate? (y/n/s to skip): ").lower()
                if result in ['y', 'n', 's']:
                    break
                print("Please enter 'y' for yes, 'n' for no, or 's' to skip")
            
            if result != 's':
                validation_results.append({
                    "fact_id": str(fact.id),
                    "claim": fact.claim,
                    "is_accurate": result == 'y',
                    "confidence": fact.confidence,
                    "validated_at": datetime.utcnow().isoformat()
                })
        
        return validation_results
    
    def validate_questions_manual(self, extraction: ExtractedKnowledge, source_text: str) -> List[Dict[str, Any]]:
        """
        Interface for manual question validation.
        
        Returns list of validation results for each question.
        """
        validation_results = []
        
        print("\n" + "="*70)
        print("MANUAL QUESTION VALIDATION")
        print("="*70)
        print(f"Chunk ID: {extraction.chunk_id}")
        print(f"Total questions to validate: {len(extraction.questions)}")
        print("\n" + "-"*70)
        
        for i, question in enumerate(extraction.questions, 1):
            print(f"\n[Question {i}/{len(extraction.questions)}]")
            print(f"Question: {question.question_text}")
            print(f"Expected Answer: {question.expected_answer}")
            print(f"Cognitive Level: {question.cognitive_level.value}")
            print(f"Difficulty: {question.difficulty}/5")
            print(f"Confidence: {question.confidence:.2f}")
            
            # Get validation input
            while True:
                result = input("\nIs this question answerable from the source text? (y/n/s to skip): ").lower()
                if result in ['y', 'n', 's']:
                    break
                print("Please enter 'y' for yes, 'n' for no, or 's' to skip")
            
            if result != 's':
                validation_results.append({
                    "question_id": str(question.id),
                    "question_text": question.question_text,
                    "is_answerable": result == 'y',
                    "cognitive_level": question.cognitive_level.value,
                    "difficulty": question.difficulty,
                    "confidence": question.confidence,
                    "validated_at": datetime.utcnow().isoformat()
                })
        
        return validation_results
    
    def update_metrics_with_validation(self, metrics: ExtractionMetrics, 
                                     fact_validations: List[Dict[str, Any]],
                                     question_validations: List[Dict[str, Any]]) -> None:
        """Update metrics with validation results."""
        
        # Update fact validation metrics
        if fact_validations:
            metrics.facts_validated = len(fact_validations)
            metrics.facts_accurate = sum(1 for v in fact_validations if v["is_accurate"])
            metrics.facts_accuracy_rate = metrics.facts_accurate / metrics.facts_validated
        
        # Update question validation metrics
        if question_validations:
            metrics.questions_validated = len(question_validations)
            metrics.questions_answerable = sum(1 for v in question_validations if v["is_answerable"])
            metrics.questions_answerability_rate = metrics.questions_answerable / metrics.questions_validated
    
    def generate_chunk_report(self, metrics: ExtractionMetrics) -> Dict[str, Any]:
        """Generate a report for a single chunk."""
        return {
            "chunk_id": metrics.chunk_id,
            "extraction_summary": {
                "topics": metrics.total_topics,
                "facts": metrics.total_facts,
                "relationships": metrics.total_relationships,
                "questions": metrics.total_questions
            },
            "quality_metrics": {
                "avg_confidence": round(metrics.avg_confidence, 3),
                "confidence_range": [round(metrics.min_confidence, 3), round(metrics.max_confidence, 3)],
                "facts_accuracy": round(metrics.facts_accuracy_rate, 3) if metrics.facts_validated > 0 else None,
                "questions_answerability": round(metrics.questions_answerability_rate, 3) if metrics.questions_validated > 0 else None
            },
            "question_analysis": {
                "cognitive_distribution": metrics.question_cognitive_levels,
                "avg_difficulty": round(metrics.avg_question_difficulty, 2)
            },
            "performance": {
                "processing_time_ms": round(metrics.processing_time_ms, 2),
                "tokens_used": metrics.tokens_used
            },
            "issues": metrics.issues,
            "validation_stats": {
                "facts_validated": metrics.facts_validated,
                "questions_validated": metrics.questions_validated
            }
        }
    
    def generate_stage_report(self, all_metrics: List[ExtractionMetrics], 
                            stage: int = 1) -> StageMetrics:
        """Generate aggregate report for a stage."""
        
        # Aggregate counts
        total_facts = sum(m.total_facts for m in all_metrics)
        total_questions = sum(m.total_questions for m in all_metrics)
        total_topics = sum(m.total_topics for m in all_metrics)
        total_relationships = sum(m.total_relationships for m in all_metrics)
        
        # Calculate accuracy metrics
        validated_facts = [m for m in all_metrics if m.facts_validated > 0]
        if validated_facts:
            total_facts_validated = sum(m.facts_validated for m in validated_facts)
            total_facts_accurate = sum(m.facts_accurate for m in validated_facts)
            overall_accuracy = total_facts_accurate / total_facts_validated
        else:
            overall_accuracy = 0.0
        
        # Calculate answerability metrics
        validated_questions = [m for m in all_metrics if m.questions_validated > 0]
        if validated_questions:
            total_questions_validated = sum(m.questions_validated for m in validated_questions)
            total_questions_answerable = sum(m.questions_answerable for m in validated_questions)
            overall_answerability = total_questions_answerable / total_questions_validated
        else:
            overall_answerability = 0.0
        
        # Performance metrics
        avg_processing_time = statistics.mean(m.processing_time_ms for m in all_metrics)
        total_tokens = sum(m.tokens_used for m in all_metrics)
        
        # Estimate cost (GPT-4 pricing approximate)
        estimated_cost = (total_tokens / 1000) * 0.03  # $0.03 per 1K tokens
        
        # Failure pattern analysis
        failure_patterns = {}
        for m in all_metrics:
            for issue in m.issues:
                failure_patterns[issue] = failure_patterns.get(issue, 0) + 1
        
        # Create stage metrics
        stage_metrics = StageMetrics(
            stage=stage,
            total_chunks=len(all_metrics),
            total_documents=len(set(m.chunk_id.split("_")[0] for m in all_metrics)),  # Assuming chunk_id format
            total_facts_extracted=total_facts,
            total_questions_generated=total_questions,
            total_topics_identified=total_topics,
            total_relationships_found=total_relationships,
            overall_accuracy=overall_accuracy,
            overall_answerability=overall_answerability,
            avg_confidence=statistics.mean(m.avg_confidence for m in all_metrics),
            avg_processing_time_ms=avg_processing_time,
            total_tokens_used=total_tokens,
            estimated_cost_usd=estimated_cost,
            meets_accuracy_target=overall_accuracy >= 0.7,  # 70% target for Stage 1
            meets_answerability_target=overall_answerability >= 0.6,  # 60% target for Stage 1
            failure_patterns=failure_patterns,
            document_type_performance={}  # TODO: Implement per-document-type analysis
        )
        
        return stage_metrics
    
    def save_report(self, report: Dict[str, Any], filename: str) -> Path:
        """Save report to JSON file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return output_path
    
    def generate_failure_analysis(self, all_metrics: List[ExtractionMetrics]) -> Dict[str, Any]:
        """Analyze extraction failures and patterns."""
        
        # Categorize chunks by performance
        high_performing = [m for m in all_metrics if m.avg_confidence >= 0.8 and len(m.issues) == 0]
        low_performing = [m for m in all_metrics if m.avg_confidence < 0.5 or len(m.issues) >= 2]
        
        # Analyze common issues
        all_issues = []
        for m in all_metrics:
            all_issues.extend(m.issues)
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        # Identify patterns
        patterns = {
            "low_confidence_correlation": {
                "avg_facts_when_low_confidence": statistics.mean(
                    m.total_facts for m in low_performing
                ) if low_performing else 0,
                "avg_facts_when_high_confidence": statistics.mean(
                    m.total_facts for m in high_performing
                ) if high_performing else 0
            },
            "extraction_completeness": {
                "chunks_with_no_facts": sum(1 for m in all_metrics if m.total_facts == 0),
                "chunks_with_no_questions": sum(1 for m in all_metrics if m.total_questions == 0),
                "chunks_with_no_relationships": sum(1 for m in all_metrics if m.total_relationships == 0)
            }
        }
        
        # Recommendations
        recommendations = []
        if issue_frequency.get("Low average confidence", 0) > len(all_metrics) * 0.3:
            recommendations.append("Consider adjusting prompts to be more specific")
        
        if patterns["extraction_completeness"]["chunks_with_no_facts"] > len(all_metrics) * 0.2:
            recommendations.append("Improve fact extraction prompts or handle edge cases")
        
        return {
            "performance_distribution": {
                "high_performing_chunks": len(high_performing),
                "low_performing_chunks": len(low_performing),
                "total_chunks": len(all_metrics)
            },
            "issue_frequency": issue_frequency,
            "patterns": patterns,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat()
        }