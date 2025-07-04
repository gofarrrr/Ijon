"""
A/B testing framework for comparing extraction strategies.

Tests baseline vs document-aware extraction approaches.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import statistics

from extraction.baseline.extractor import BaselineExtractor
from extraction.document_aware.extractor import DocumentAwareExtractor
from extraction.models import ExtractedKnowledge
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ABTestFramework:
    """Framework for A/B testing extraction strategies."""
    
    def __init__(self, openai_api_key: str, output_dir: str = "test_results/ab_tests"):
        """
        Initialize A/B testing framework.
        
        Args:
            openai_api_key: OpenAI API key
            output_dir: Directory for test results
        """
        self.baseline_extractor = BaselineExtractor(openai_api_key)
        self.aware_extractor = DocumentAwareExtractor(openai_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized A/B test framework. Output: {output_dir}")
    
    async def run_ab_test(self, pdf_paths: List[str], test_name: str = "ab_test") -> Dict[str, Any]:
        """
        Run A/B test comparing baseline and document-aware extraction.
        
        Args:
            pdf_paths: List of PDF paths to test
            test_name: Name for this test run
            
        Returns:
            Test results and analysis
        """
        logger.info(f"Starting A/B test '{test_name}' with {len(pdf_paths)} PDFs")
        
        results = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "pdf_count": len(pdf_paths),
            "individual_results": [],
            "aggregate_metrics": {},
            "statistical_analysis": {}
        }
        
        # Run tests for each PDF
        for pdf_path in pdf_paths:
            logger.info(f"Testing: {pdf_path}")
            try:
                comparison = await self.aware_extractor.compare_with_baseline(pdf_path)
                
                # Add quality metrics
                quality_metrics = await self._evaluate_quality_difference(
                    pdf_path, comparison
                )
                comparison["quality_metrics"] = quality_metrics
                
                results["individual_results"].append(comparison)
                
            except Exception as e:
                logger.error(f"Test failed for {pdf_path}: {str(e)}")
                results["individual_results"].append({
                    "pdf_path": pdf_path,
                    "error": str(e)
                })
        
        # Calculate aggregate metrics
        results["aggregate_metrics"] = self._calculate_aggregate_metrics(
            results["individual_results"]
        )
        
        # Statistical analysis
        results["statistical_analysis"] = self._perform_statistical_analysis(
            results["individual_results"]
        )
        
        # Save results
        output_file = self.output_dir / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"A/B test complete. Results saved to: {output_file}")
        
        # Generate summary report
        self._generate_summary_report(results, test_name)
        
        return results
    
    async def _evaluate_quality_difference(self, pdf_path: str, comparison: Dict) -> Dict[str, Any]:
        """
        Evaluate quality differences between extractions.
        
        Args:
            pdf_path: Path to PDF
            comparison: Basic comparison metrics
            
        Returns:
            Quality evaluation metrics
        """
        # For now, use confidence and quantity metrics
        # In a real system, we'd have human evaluation or more sophisticated metrics
        
        baseline = comparison["baseline"]
        aware = comparison["document_aware"]
        
        quality_metrics = {
            "confidence_improvement": aware["confidence"] - baseline["confidence"],
            "fact_confidence_improvement": aware["avg_fact_confidence"] - baseline["avg_fact_confidence"],
            "extraction_completeness": {
                "baseline": (baseline["topics"] + baseline["facts"]) / 100,  # Normalized estimate
                "aware": (aware["topics"] + aware["facts"]) / 100
            },
            "strategy_benefit": aware["strategy"] != "baseline",
            "specialized_extraction": aware["document_type"] != "unknown"
        }
        
        # Calculate overall quality score (0-1)
        improvements = [
            quality_metrics["confidence_improvement"] > 0,
            quality_metrics["fact_confidence_improvement"] > 0,
            aware["facts"] >= baseline["facts"],
            quality_metrics["strategy_benefit"]
        ]
        
        quality_metrics["overall_improvement_score"] = sum(improvements) / len(improvements)
        
        return quality_metrics
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all tests."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        # Extract metrics
        confidence_improvements = [r["improvements"]["confidence_delta"] for r in valid_results]
        facts_improvements = [r["improvements"]["facts_delta"] for r in valid_results]
        quality_scores = [r["quality_metrics"]["overall_improvement_score"] for r in valid_results]
        
        # Calculate aggregates
        return {
            "total_tests": len(results),
            "successful_tests": len(valid_results),
            "average_confidence_improvement": statistics.mean(confidence_improvements),
            "average_facts_improvement": statistics.mean(facts_improvements),
            "average_quality_score": statistics.mean(quality_scores),
            "improvement_rate": sum(1 for s in quality_scores if s > 0.5) / len(quality_scores),
            "strategy_distribution": self._get_strategy_distribution(valid_results),
            "document_type_distribution": self._get_document_type_distribution(valid_results)
        }
    
    def _get_strategy_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Get distribution of strategies used."""
        strategies = {}
        for r in results:
            strategy = r["document_aware"]["strategy"]
            strategies[strategy] = strategies.get(strategy, 0) + 1
        return strategies
    
    def _get_document_type_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Get distribution of document types."""
        types = {}
        for r in results:
            doc_type = r["document_aware"]["document_type"]
            types[doc_type] = types.get(doc_type, 0) + 1
        return types
    
    def _perform_statistical_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Perform statistical analysis on results."""
        valid_results = [r for r in results if "error" not in r]
        
        if len(valid_results) < 2:
            return {"error": "Insufficient data for statistical analysis"}
        
        # Confidence improvements
        confidence_deltas = [r["improvements"]["confidence_delta"] for r in valid_results]
        
        # Basic statistics
        analysis = {
            "confidence_improvement": {
                "mean": statistics.mean(confidence_deltas),
                "median": statistics.median(confidence_deltas),
                "stdev": statistics.stdev(confidence_deltas) if len(confidence_deltas) > 1 else 0,
                "min": min(confidence_deltas),
                "max": max(confidence_deltas)
            },
            "significant_improvement": statistics.mean(confidence_deltas) > 0.05,
            "consistency": statistics.stdev(confidence_deltas) < 0.2 if len(confidence_deltas) > 1 else True
        }
        
        # Success rate by document type
        success_by_type = {}
        for r in valid_results:
            doc_type = r["document_aware"]["document_type"]
            success = r["quality_metrics"]["overall_improvement_score"] > 0.5
            
            if doc_type not in success_by_type:
                success_by_type[doc_type] = []
            success_by_type[doc_type].append(success)
        
        analysis["success_rate_by_type"] = {
            doc_type: sum(successes) / len(successes)
            for doc_type, successes in success_by_type.items()
        }
        
        return analysis
    
    def _generate_summary_report(self, results: Dict, test_name: str):
        """Generate human-readable summary report."""
        report_file = self.output_dir / f"{test_name}_summary.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# A/B Test Summary: {test_name}\n\n")
            f.write(f"**Date**: {results['timestamp']}\n")
            f.write(f"**PDFs Tested**: {results['pdf_count']}\n\n")
            
            # Aggregate metrics
            agg = results["aggregate_metrics"]
            f.write("## Overall Results\n\n")
            f.write(f"- **Success Rate**: {agg.get('successful_tests', 0)}/{results['pdf_count']}\n")
            f.write(f"- **Improvement Rate**: {agg.get('improvement_rate', 0):.1%}\n")
            f.write(f"- **Average Confidence Gain**: {agg.get('average_confidence_improvement', 0):.3f}\n")
            f.write(f"- **Average Quality Score**: {agg.get('average_quality_score', 0):.2f}\n\n")
            
            # Strategy distribution
            f.write("## Strategy Usage\n\n")
            for strategy, count in agg.get('strategy_distribution', {}).items():
                f.write(f"- {strategy}: {count}\n")
            f.write("\n")
            
            # Statistical analysis
            stats = results["statistical_analysis"]
            if "error" not in stats:
                f.write("## Statistical Analysis\n\n")
                f.write(f"- **Significant Improvement**: {'Yes' if stats.get('significant_improvement', False) else 'No'}\n")
                f.write(f"- **Consistency**: {'High' if stats.get('consistency', False) else 'Low'}\n\n")
                
                # Success by type
                f.write("### Success Rate by Document Type\n\n")
                for doc_type, rate in stats.get('success_rate_by_type', {}).items():
                    f.write(f"- {doc_type}: {rate:.1%}\n")
            
            # Individual results
            f.write("\n## Individual Results\n\n")
            for i, result in enumerate(results["individual_results"], 1):
                if "error" in result:
                    f.write(f"{i}. {result['pdf_path']} - ERROR: {result['error']}\n")
                else:
                    f.write(f"{i}. {Path(result['pdf_path']).name}\n")
                    f.write(f"   - Strategy: {result['document_aware']['strategy']}\n")
                    f.write(f"   - Type: {result['document_aware']['document_type']}\n")
                    f.write(f"   - Quality Score: {result['quality_metrics']['overall_improvement_score']:.2f}\n")
                    f.write(f"   - Confidence Δ: {result['improvements']['confidence_delta']:+.3f}\n\n")
        
        logger.info(f"Summary report saved to: {report_file}")


async def run_stage2_validation(openai_api_key: str):
    """Run Stage 2 validation tests."""
    logger.info("Running Stage 2 validation...")
    
    # Use test PDFs
    test_pdfs = [
        "test_documents/technical_manual.pdf",
        "test_documents/research_paper.pdf", 
        "test_documents/business_book.pdf",
        "test_documents/tutorial_guide.pdf",
        "test_documents/historical_document.pdf"
    ]
    
    # Create framework
    framework = ABTestFramework(openai_api_key)
    
    # Run A/B test
    results = await framework.run_ab_test(
        test_pdfs,
        test_name="stage2_validation"
    )
    
    # Check if we meet Stage 2 success criteria
    agg = results["aggregate_metrics"]
    success_criteria = {
        "improvement_rate": agg.get("improvement_rate", 0) > 0.6,  # 60% improvement
        "confidence_gain": agg.get("average_confidence_improvement", 0) > 0.05,
        "quality_score": agg.get("average_quality_score", 0) > 0.5
    }
    
    logger.info("Stage 2 Validation Results:")
    logger.info(f"- Improvement Rate: {agg.get('improvement_rate', 0):.1%} (Target: >60%)")
    logger.info(f"- Confidence Gain: {agg.get('average_confidence_improvement', 0):.3f} (Target: >0.05)")
    logger.info(f"- Quality Score: {agg.get('average_quality_score', 0):.2f} (Target: >0.5)")
    logger.info(f"- PASS: {all(success_criteria.values())}")
    
    return results, all(success_criteria.values())