"""
Evaluation orchestrator that coordinates all evaluation components.

This module provides a unified interface for quality assessment, metrics collection,
and continuous improvement across the RAG pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import asyncio

from src.evaluation.metrics_collector import get_metrics_collector, collect_metrics
from src.models import SearchResult, PDFChunk
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationOrchestrator:
    """
    Unified evaluation system that coordinates all quality assessment components.
    
    Integrates:
    - Metrics collection
    - Extraction evaluation
    - Quality scoring
    - Adaptive quality management
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize evaluation orchestrator."""
        self.metrics_collector = get_metrics_collector()
        
        # Quality thresholds
        self.min_quality_score = 0.7
        self.min_confidence = 0.6
        
    async def initialize(self):
        """Initialize all evaluation components."""
        await self.metrics_collector.initialize()
        logger.info("Evaluation orchestrator initialized")
    
    @collect_metrics('evaluation', 'evaluate_retrieval')
    async def evaluate_retrieval(
        self,
        query: str,
        results: List[SearchResult],
        expected_docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval quality.
        
        Args:
            query: Search query
            results: Retrieved results
            expected_docs: Expected document IDs (for testing)
            
        Returns:
            Evaluation metrics including precision, recall, NDCG
        """
        evaluation = {
            'query': query,
            'num_results': len(results),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Record basic metrics
        await self.metrics_collector.record_metric(
            'retrieval_count', len(results), 'retriever', 'search'
        )
        
        if results:
            # Score distribution
            scores = [r.score for r in results]
            evaluation['score_distribution'] = {
                'min': min(scores),
                'max': max(scores),
                'avg': sum(scores) / len(scores)
            }
            
            # Record score metrics
            await self.metrics_collector.record_metric(
                'avg_retrieval_score', evaluation['score_distribution']['avg'],
                'retriever', 'search'
            )
        
        # Calculate precision/recall if ground truth provided
        if expected_docs:
            retrieved_ids = [r.document.id for r in results]
            
            # Precision: relevant retrieved / total retrieved
            relevant_retrieved = len(set(retrieved_ids) & set(expected_docs))
            precision = relevant_retrieved / len(results) if results else 0
            
            # Recall: relevant retrieved / total relevant
            recall = relevant_retrieved / len(expected_docs) if expected_docs else 0
            
            # F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            evaluation['precision'] = precision
            evaluation['recall'] = recall
            evaluation['f1_score'] = f1
            
            # Record metrics
            await self.metrics_collector.record_metric(
                'retrieval_precision', precision, 'retriever', 'search'
            )
            await self.metrics_collector.record_metric(
                'retrieval_recall', recall, 'retriever', 'search'
            )
        
        return evaluation
    
    @collect_metrics('evaluation', 'evaluate_extraction')
    async def evaluate_extraction(
        self,
        extracted: Dict[str, Any],
        chunk: PDFChunk
    ) -> Dict[str, Any]:
        """
        Evaluate extraction quality with simplified metrics.
        
        Args:
            extracted: Extracted data
            chunk: Source chunk
            
        Returns:
            Quality assessment
        """
        # Simple quality assessment
        report = {
            'chunk_id': chunk.id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Calculate simple quality score based on extraction completeness
        quality_score = 0.5  # Base score
        if extracted.get('entities', []):
            quality_score += 0.3
        if extracted.get('relationships', []):
            quality_score += 0.2
        
        report['quality_score'] = quality_score
        
        # Record metrics
        await self.metrics_collector.record_metric(
            'extraction_quality', quality_score, 'extractor', 'extract'
        )
        
        return report
    
    @collect_metrics('evaluation', 'evaluate_graph_quality')
    async def evaluate_graph_quality(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate knowledge graph quality.
        
        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            
        Returns:
            Graph quality metrics
        """
        evaluation = {
            'entity_count': len(entities),
            'relationship_count': len(relationships),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if entities:
            # Entity type distribution
            type_counts = {}
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            
            evaluation['entity_types'] = type_counts
            
            # Calculate graph density
            max_relationships = len(entities) * (len(entities) - 1) / 2
            density = len(relationships) / max_relationships if max_relationships > 0 else 0
            evaluation['graph_density'] = density
            
            # Record metrics
            await self.metrics_collector.record_metric(
                'entity_count', len(entities), 'graph', 'extract'
            )
            await self.metrics_collector.record_metric(
                'relationship_count', len(relationships), 'graph', 'extract'
            )
            await self.metrics_collector.record_metric(
                'graph_density', density, 'graph', 'extract'
            )
        
        return evaluation
    
    @collect_metrics('evaluation', 'evaluate_end_to_end')
    async def evaluate_end_to_end(
        self,
        query: str,
        answer: str,
        retrieval_results: List[SearchResult],
        extraction_results: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end RAG pipeline quality.
        
        Args:
            query: User query
            answer: Generated answer
            retrieval_results: Retrieved documents
            extraction_results: Extracted knowledge (if available)
            
        Returns:
            Comprehensive pipeline evaluation
        """
        evaluation = {
            'query': query,
            'answer_length': len(answer),
            'retrieval_count': len(retrieval_results),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Answer quality assessment
        answer_quality = await self._assess_answer_quality(query, answer, retrieval_results)
        evaluation['answer_quality'] = answer_quality
        
        # Record end-to-end metrics
        await self.metrics_collector.record_metric(
            'answer_quality', answer_quality['overall_score'],
            'pipeline', 'end_to_end'
        )
        await self.metrics_collector.record_metric(
            'answer_length', len(answer), 'pipeline', 'end_to_end'
        )
        
        # Track performance metric
        await self.metrics_collector.record_metric(
            'pipeline_quality', answer_quality['overall_score'],
            'pipeline', 'performance'
        )
        
        return evaluation
    
    async def _assess_answer_quality(
        self,
        query: str,
        answer: str,
        sources: List[SearchResult]
    ) -> Dict[str, Any]:
        """Assess answer quality based on multiple criteria."""
        assessment = {
            'relevance': 0.0,
            'completeness': 0.0,
            'coherence': 0.0,
            'grounding': 0.0,
            'overall_score': 0.0
        }
        
        # Simple heuristics for now (can be enhanced with LLM-based evaluation)
        
        # Relevance: Does answer address the query?
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        term_overlap = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
        assessment['relevance'] = min(term_overlap * 2, 1.0)  # Scale up
        
        # Completeness: Answer length relative to query complexity
        expected_length = len(query.split()) * 10  # Rough heuristic
        length_ratio = min(len(answer.split()) / expected_length, 1.0)
        assessment['completeness'] = length_ratio
        
        # Coherence: Basic structure checks
        has_sentences = '.' in answer
        has_structure = len(answer.split('\n')) > 1 or len(answer) > 100
        assessment['coherence'] = 0.5 if has_sentences else 0.2
        assessment['coherence'] += 0.5 if has_structure else 0.3
        
        # Grounding: Answer supported by sources
        if sources:
            source_text = ' '.join([s.document.content[:500] for s in sources[:3]])
            answer_words = set(answer.lower().split())
            source_words = set(source_text.lower().split())
            grounding_ratio = len(answer_words & source_words) / len(answer_words) if answer_words else 0
            assessment['grounding'] = min(grounding_ratio * 1.5, 1.0)  # Scale up
        
        # Overall score (weighted average)
        assessment['overall_score'] = (
            assessment['relevance'] * 0.3 +
            assessment['completeness'] * 0.2 +
            assessment['coherence'] * 0.2 +
            assessment['grounding'] * 0.3
        )
        
        return assessment
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and performance metrics."""
        # Get aggregated metrics
        metrics = self.metrics_collector.get_aggregated_metrics()
        
        # Calculate health indicators
        health = {
            'status': 'healthy',
            'components': {},
            'metrics_summary': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check component health
        components = ['retriever', 'extractor', 'graph', 'pipeline']
        for component in components:
            component_metrics = {k: v for k, v in metrics.items() if k.startswith(f"{component}.")}
            
            if component_metrics:
                # Calculate success rate
                success_key = f"{component}.*.success_rate"
                success_metrics = [v for k, v in component_metrics.items() if 'success_rate' in k]
                if success_metrics:
                    avg_success = sum(m['average'] for m in success_metrics) / len(success_metrics)
                    
                    health['components'][component] = {
                        'status': 'healthy' if avg_success > 0.9 else 'degraded' if avg_success > 0.7 else 'unhealthy',
                        'success_rate': avg_success,
                        'metrics': component_metrics
                    }
        
        # Overall status
        unhealthy_count = sum(1 for c in health['components'].values() if c['status'] == 'unhealthy')
        if unhealthy_count > 0:
            health['status'] = 'unhealthy'
        elif any(c['status'] == 'degraded' for c in health['components'].values()):
            health['status'] = 'degraded'
        
        return health
    
    async def close(self):
        """Close evaluation orchestrator."""
        await self.metrics_collector.close()


# Global instance
_orchestrator = None


def get_evaluation_orchestrator() -> EvaluationOrchestrator:
    """Get the global evaluation orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = EvaluationOrchestrator()
    return _orchestrator