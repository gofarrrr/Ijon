"""
Evaluation framework for PDF RAG system.

This module provides comprehensive evaluation metrics and testing utilities
for assessing the quality of the RAG system.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer, util

from src.config import get_settings
from src.models import PDFChunk, SearchResult
from src.rag.pipeline import RAGPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Answer quality metrics
    answer_relevance: float  # 0-1, how relevant is the answer
    answer_completeness: float  # 0-1, how complete is the answer
    answer_correctness: float  # 0-1, factual correctness
    answer_coherence: float  # 0-1, logical coherence
    
    # Retrieval quality metrics
    retrieval_precision: float  # Precision of retrieved chunks
    retrieval_recall: float  # Recall of relevant chunks
    retrieval_f1: float  # F1 score
    retrieval_mrr: float  # Mean Reciprocal Rank
    retrieval_ndcg: float  # Normalized Discounted Cumulative Gain
    
    # Graph quality metrics (if applicable)
    graph_coverage: Optional[float] = None  # Entity/relation coverage
    graph_accuracy: Optional[float] = None  # Accuracy of extracted relationships
    graph_connectivity: Optional[float] = None  # Graph connectivity score
    
    # Performance metrics
    latency_ms: float  # End-to-end latency
    tokens_used: int  # Total tokens consumed
    
    # Confidence scores
    confidence_calibration: float  # How well calibrated are confidence scores
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        answer_score = (
            self.answer_relevance * 0.3 +
            self.answer_completeness * 0.2 +
            self.answer_correctness * 0.3 +
            self.answer_coherence * 0.2
        )
        retrieval_score = self.retrieval_f1
        
        # Weight answer quality more heavily
        return answer_score * 0.7 + retrieval_score * 0.3


@dataclass
class TestCase:
    """Evaluation test case."""
    
    id: str
    question: str
    expected_answer: str
    relevant_chunks: List[str]  # IDs of relevant chunks
    required_entities: List[str]  # Entities that should be mentioned
    required_facts: List[str]  # Key facts that should be included
    difficulty: str  # easy, medium, hard
    category: str  # factual, analytical, comparative, etc.
    

@dataclass
class TestDataset:
    """Collection of test cases for evaluation."""
    
    name: str
    description: str
    test_cases: List[TestCase]
    pdf_ids: List[str]  # PDFs used in this dataset
    created_at: datetime
    

class RAGEvaluator:
    """Comprehensive evaluator for the RAG system."""
    
    def __init__(self, pipeline: RAGPipeline):
        """Initialize evaluator with RAG pipeline."""
        self.pipeline = pipeline
        self.settings = get_settings()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    async def evaluate_single(
        self,
        test_case: TestCase,
        use_agent: bool = True,
    ) -> Tuple[str, List[SearchResult], EvaluationMetrics]:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Test case to evaluate
            use_agent: Whether to use agent-based query
            
        Returns:
            Tuple of (answer, search_results, metrics)
        """
        import time
        start_time = time.time()
        
        # Generate answer
        if use_agent:
            answer, search_results = await self.pipeline.query_with_agent(
                test_case.question
            )
        else:
            answer, search_results = await self.pipeline.query(
                test_case.question
            )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate metrics
        metrics = await self._calculate_metrics(
            test_case,
            answer,
            search_results,
            latency_ms,
        )
        
        return answer, search_results, metrics
    
    async def evaluate_dataset(
        self,
        dataset: TestDataset,
        use_agent: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            dataset: Test dataset
            use_agent: Whether to use agent-based queries
            save_results: Whether to save detailed results
            
        Returns:
            Evaluation results summary
        """
        results = []
        
        for test_case in dataset.test_cases:
            try:
                answer, search_results, metrics = await self.evaluate_single(
                    test_case, use_agent
                )
                
                result = {
                    "test_case_id": test_case.id,
                    "question": test_case.question,
                    "generated_answer": answer,
                    "expected_answer": test_case.expected_answer,
                    "metrics": metrics.__dict__,
                    "search_results": [
                        {
                            "chunk_id": sr.document.id,
                            "score": sr.score,
                            "content_preview": sr.document.content[:200],
                        }
                        for sr in search_results[:5]
                    ],
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate test case {test_case.id}: {e}")
                results.append({
                    "test_case_id": test_case.id,
                    "error": str(e),
                })
        
        # Calculate aggregate metrics
        summary = self._calculate_summary(results, dataset)
        
        if save_results:
            self._save_results(dataset.name, results, summary)
        
        return summary
    
    async def _calculate_metrics(
        self,
        test_case: TestCase,
        answer: str,
        search_results: List[SearchResult],
        latency_ms: float,
    ) -> EvaluationMetrics:
        """Calculate comprehensive metrics for a test case."""
        # Answer quality metrics
        answer_relevance = self._calculate_semantic_similarity(
            answer, test_case.expected_answer
        )
        
        answer_completeness = self._calculate_completeness(
            answer, test_case.required_facts
        )
        
        answer_correctness = self._calculate_correctness(
            answer, test_case.expected_answer, test_case.required_facts
        )
        
        answer_coherence = self._calculate_coherence(answer)
        
        # Retrieval quality metrics
        retrieved_ids = [sr.document.id for sr in search_results]
        retrieval_metrics = self._calculate_retrieval_metrics(
            retrieved_ids, test_case.relevant_chunks
        )
        
        # TODO: Calculate graph metrics if applicable
        
        # Estimate tokens (rough approximation)
        tokens_used = len(test_case.question.split()) + len(answer.split()) * 2
        
        # TODO: Calculate confidence calibration
        confidence_calibration = 0.8  # Placeholder
        
        return EvaluationMetrics(
            answer_relevance=answer_relevance,
            answer_completeness=answer_completeness,
            answer_correctness=answer_correctness,
            answer_coherence=answer_coherence,
            retrieval_precision=retrieval_metrics["precision"],
            retrieval_recall=retrieval_metrics["recall"],
            retrieval_f1=retrieval_metrics["f1"],
            retrieval_mrr=retrieval_metrics["mrr"],
            retrieval_ndcg=retrieval_metrics["ndcg"],
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            confidence_calibration=confidence_calibration,
        )
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return max(0.0, similarity)
    
    def _calculate_completeness(
        self, answer: str, required_facts: List[str]
    ) -> float:
        """Calculate how many required facts are covered."""
        if not required_facts:
            return 1.0
        
        covered = 0
        answer_lower = answer.lower()
        
        for fact in required_facts:
            # Simple keyword matching - could be improved with semantic matching
            if fact.lower() in answer_lower:
                covered += 1
            else:
                # Check semantic similarity
                similarity = self._calculate_semantic_similarity(answer, fact)
                if similarity > 0.7:
                    covered += 0.5
        
        return covered / len(required_facts)
    
    def _calculate_correctness(
        self,
        answer: str,
        expected_answer: str,
        required_facts: List[str],
    ) -> float:
        """Calculate factual correctness."""
        # Combine semantic similarity with fact coverage
        semantic_score = self._calculate_semantic_similarity(answer, expected_answer)
        fact_score = self._calculate_completeness(answer, required_facts)
        
        return (semantic_score * 0.6 + fact_score * 0.4)
    
    def _calculate_coherence(self, answer: str) -> float:
        """Calculate answer coherence."""
        # Simple heuristics for now
        sentences = answer.split('.')
        
        if len(sentences) < 2:
            return 1.0
        
        # Check for logical flow between sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            if sentences[i].strip() and sentences[i+1].strip():
                similarity = self._calculate_semantic_similarity(
                    sentences[i], sentences[i+1]
                )
                coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 1.0
    
    def _calculate_retrieval_metrics(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
    ) -> Dict[str, float]:
        """Calculate retrieval metrics."""
        if not relevant_ids:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
            }
        
        # Convert to sets for easier calculation
        retrieved_set = set(retrieved_ids)
        relevant_set = set(relevant_ids)
        
        # Precision and Recall
        true_positives = len(retrieved_set & relevant_set)
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        
        # F1 Score
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        
        # Mean Reciprocal Rank
        mrr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                mrr = 1 / (i + 1)
                break
        
        # NDCG (simplified)
        dcg = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                dcg += 1 / np.log2(i + 2)
        
        idcg = sum(1 / np.log2(i + 2) for i in range(len(relevant_set)))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "ndcg": ndcg,
        }
    
    def _calculate_summary(
        self, results: List[Dict], dataset: TestDataset
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        valid_results = [r for r in results if "metrics" in r]
        
        if not valid_results:
            return {
                "error": "No valid results to summarize",
                "total_cases": len(dataset.test_cases),
                "successful_cases": 0,
            }
        
        # Extract all metrics
        all_metrics = [r["metrics"] for r in valid_results]
        
        # Calculate averages
        metric_names = list(EvaluationMetrics.__dataclass_fields__.keys())
        avg_metrics = {}
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics if m.get(metric) is not None]
            if values:
                avg_metrics[f"avg_{metric}"] = np.mean(values)
                avg_metrics[f"std_{metric}"] = np.std(values)
        
        # Calculate overall scores
        overall_scores = [
            EvaluationMetrics(**m).overall_score
            for m in all_metrics
        ]
        
        # Group by difficulty and category
        by_difficulty = {}
        by_category = {}
        
        for result, test_case in zip(valid_results, dataset.test_cases):
            score = EvaluationMetrics(**result["metrics"]).overall_score
            
            # By difficulty
            if test_case.difficulty not in by_difficulty:
                by_difficulty[test_case.difficulty] = []
            by_difficulty[test_case.difficulty].append(score)
            
            # By category
            if test_case.category not in by_category:
                by_category[test_case.category] = []
            by_category[test_case.category].append(score)
        
        return {
            "dataset_name": dataset.name,
            "total_cases": len(dataset.test_cases),
            "successful_cases": len(valid_results),
            "failed_cases": len(results) - len(valid_results),
            "overall_score": {
                "mean": np.mean(overall_scores),
                "std": np.std(overall_scores),
                "min": np.min(overall_scores),
                "max": np.max(overall_scores),
            },
            "metrics": avg_metrics,
            "by_difficulty": {
                k: {"mean": np.mean(v), "count": len(v)}
                for k, v in by_difficulty.items()
            },
            "by_category": {
                k: {"mean": np.mean(v), "count": len(v)}
                for k, v in by_category.items()
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def _save_results(
        self,
        dataset_name: str,
        results: List[Dict],
        summary: Dict[str, Any],
    ) -> None:
        """Save evaluation results to disk."""
        # Create evaluation results directory
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        details_path = results_dir / f"{dataset_name}_{timestamp}_details.json"
        with open(details_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_path = results_dir / f"{dataset_name}_{timestamp}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(
            f"Saved evaluation results",
            extra={
                "details_path": str(details_path),
                "summary_path": str(summary_path),
            },
        )


# Test utilities
def create_sample_test_dataset() -> TestDataset:
    """Create a sample test dataset for evaluation."""
    test_cases = [
        # Easy factual questions
        TestCase(
            id="easy_1",
            question="What is the main topic of this document?",
            expected_answer="The document discusses machine learning techniques for natural language processing.",
            relevant_chunks=["chunk_1", "chunk_2"],
            required_entities=["machine learning", "NLP"],
            required_facts=["machine learning", "natural language processing"],
            difficulty="easy",
            category="factual",
        ),
        # Medium analytical questions
        TestCase(
            id="medium_1",
            question="How do transformers differ from RNNs in processing sequences?",
            expected_answer="Transformers process sequences in parallel using attention mechanisms, while RNNs process sequentially. This makes transformers faster and better at capturing long-range dependencies.",
            relevant_chunks=["chunk_5", "chunk_7", "chunk_9"],
            required_entities=["transformers", "RNN", "attention"],
            required_facts=["parallel processing", "attention mechanisms", "sequential processing"],
            difficulty="medium",
            category="analytical",
        ),
        # Hard comparative questions
        TestCase(
            id="hard_1",
            question="Compare and contrast the architectures and use cases of BERT, GPT, and T5 models.",
            expected_answer="BERT uses bidirectional encoding for understanding tasks, GPT uses autoregressive generation for text completion, and T5 frames all tasks as text-to-text. BERT excels at classification, GPT at generation, and T5 provides a unified framework.",
            relevant_chunks=["chunk_12", "chunk_15", "chunk_18", "chunk_20"],
            required_entities=["BERT", "GPT", "T5"],
            required_facts=["bidirectional", "autoregressive", "text-to-text", "classification", "generation"],
            difficulty="hard",
            category="comparative",
        ),
    ]
    
    return TestDataset(
        name="sample_ml_dataset",
        description="Sample dataset for testing ML document understanding",
        test_cases=test_cases,
        pdf_ids=["ml_paper_1.pdf", "ml_paper_2.pdf"],
        created_at=datetime.utcnow(),
    )
