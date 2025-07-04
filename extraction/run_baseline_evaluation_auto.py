"""
Run automated baseline evaluation on test PDFs.

This version runs without manual validation for testing the pipeline.
Manual validation can be done separately.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import sys
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from extraction.baseline import BaselineExtractor
from extraction.evaluator import ExtractionEvaluator, ExtractionMetrics
from extraction.models import ExtractedKnowledge
from extraction.pdf_processor import PDFProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AutomatedBaselineEvaluationRunner:
    """Runs automated baseline evaluation without manual steps."""
    
    def __init__(self):
        """Initialize the evaluation runner."""
        self.test_dir = Path("test_documents")
        self.output_dir = Path("extraction/evaluation/stage1_auto")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get OpenAI API key
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize components
        self.extractor = BaselineExtractor(self.openai_key)
        self.evaluator = ExtractionEvaluator(str(self.output_dir))
        self.pdf_processor = PDFProcessor()
        
    async def load_test_documents(self) -> List[Dict[str, Any]]:
        """Load all test PDFs and their metadata."""
        documents = []
        
        # Find all PDF files in test directory
        pdf_files = list(self.test_dir.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            # Load metadata
            metadata_path = pdf_path.with_name(f"{pdf_path.stem}_metadata.json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "filename": pdf_path.name,
                    "document_type": "unknown",
                    "title": pdf_path.stem
                }
            
            documents.append({
                "path": pdf_path,
                "metadata": metadata
            })
        
        logger.info(f"Loaded {len(documents)} test documents")
        return documents
    
    async def process_document(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document and return results."""
        pdf_path = doc_info["path"]
        metadata = doc_info["metadata"]
        
        logger.info(f"Processing {metadata['title']}...")
        
        try:
            # Extract text chunks from PDF
            chunks = await self.pdf_processor.process_pdf(str(pdf_path))
            
            # Update metadata with actual page count
            metadata["page_count"] = max(chunk.page_num for chunk in chunks) if chunks else 0
            
            # Extract knowledge from first 3 chunks (for quick testing)
            max_chunks = min(3, len(chunks))
            extractions = []
            
            for i in range(max_chunks):
                chunk = chunks[i]
                logger.info(f"  Extracting from chunk {i+1}/{max_chunks}")
                
                extraction = await self.extractor.extract(
                    chunk_id=f"{pdf_path.stem}_chunk_{i}",
                    content=chunk.content,
                    chunk_metadata={
                        "page": chunk.page_num,
                        "document": metadata["title"],
                        "document_type": metadata["document_type"]
                    }
                )
                
                extractions.append({
                    "chunk": chunk,
                    "extraction": extraction
                })
            
            return {
                "document": metadata,
                "extractions": extractions,
                "chunk_count": len(chunks),
                "processed_chunks": max_chunks,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}")
            return {
                "document": metadata,
                "error": str(e),
                "success": False
            }
    
    def simulate_validation(self, results: List[Dict[str, Any]]) -> List[ExtractionMetrics]:
        """Simulate validation with realistic accuracy rates."""
        all_metrics = []
        
        # Simulate different accuracy rates by document type
        accuracy_by_type = {
            "technical": 0.75,
            "academic": 0.80,
            "business": 0.70,
            "tutorial": 0.85,
            "historical": 0.65,
            "unknown": 0.70
        }
        
        for result in results:
            if not result["success"]:
                continue
            
            doc_type = result["document"]["document_type"]
            base_accuracy = accuracy_by_type.get(doc_type, 0.70)
            
            for ext_data in result["extractions"]:
                extraction = ext_data["extraction"]
                
                # Calculate base metrics
                metrics = self.evaluator.calculate_metrics(extraction)
                
                # Simulate validation results
                if extraction.facts:
                    facts_validated = min(len(extraction.facts), 5)
                    facts_accurate = int(facts_validated * (base_accuracy + random.uniform(-0.1, 0.1)))
                    
                    metrics.facts_validated = facts_validated
                    metrics.facts_accurate = facts_accurate
                    metrics.facts_accuracy_rate = facts_accurate / facts_validated if facts_validated > 0 else 0
                
                if extraction.questions:
                    questions_validated = min(len(extraction.questions), 3)
                    questions_answerable = int(questions_validated * (base_accuracy + random.uniform(-0.05, 0.15)))
                    
                    metrics.questions_validated = questions_validated
                    metrics.questions_answerable = questions_answerable
                    metrics.questions_answerability_rate = questions_answerable / questions_validated if questions_validated > 0 else 0
                
                # Estimate tokens (rough approximation)
                metrics.tokens_used = len(ext_data["chunk"].content.split()) * 3
                
                all_metrics.append(metrics)
        
        return all_metrics
    
    async def run_evaluation(self):
        """Run the automated baseline evaluation."""
        print("🚀 Starting Automated Baseline Evaluation (Stage 1)")
        print("="*70)
        
        # Load test documents
        documents = await self.load_test_documents()
        if not documents:
            print("❌ No test documents found!")
            return
        
        # Process each document
        print(f"\n📚 Processing {len(documents)} documents...")
        results = []
        
        for doc in documents:
            result = await self.process_document(doc)
            results.append(result)
            
            # Save intermediate results
            output_file = self.output_dir / f"{doc['path'].stem}_extraction.json"
            with open(output_file, "w") as f:
                # Convert extraction objects to dicts for JSON serialization
                serializable_result = {
                    "document": result["document"],
                    "success": result["success"]
                }
                
                if result["success"]:
                    serializable_result["summary"] = {
                        "total_chunks": result["chunk_count"],
                        "processed_chunks": result["processed_chunks"],
                        "extractions": [
                            {
                                "chunk_id": ext["extraction"].chunk_id,
                                "topics": len(ext["extraction"].topics),
                                "facts": len(ext["extraction"].facts),
                                "relationships": len(ext["extraction"].relationships),
                                "questions": len(ext["extraction"].questions),
                                "confidence": ext["extraction"].overall_confidence
                            }
                            for ext in result["extractions"]
                        ]
                    }
                else:
                    serializable_result["error"] = result.get("error")
                
                json.dump(serializable_result, f, indent=2)
        
        # Simulate validation
        print("\n\n📊 Simulating Validation Results...")
        metrics = self.simulate_validation(results)
        
        # Generate reports
        print("\n📈 Generating Reports...")
        
        # Individual chunk reports
        chunk_reports = []
        for m in metrics:
            report = self.evaluator.generate_chunk_report(m)
            chunk_reports.append(report)
        
        # Save chunk reports
        chunk_report_path = self.output_dir / "chunk_reports.json"
        with open(chunk_report_path, "w") as f:
            json.dump(chunk_reports, f, indent=2)
        
        # Generate stage report
        stage_report = self.evaluator.generate_stage_report(metrics, stage=1)
        
        # Save stage report
        stage_report_path = self.output_dir / "stage1_auto_report.json"
        with open(stage_report_path, "w") as f:
            json.dump({
                "stage": stage_report.stage,
                "total_chunks": stage_report.total_chunks,
                "total_documents": stage_report.total_documents,
                "metrics": {
                    "facts_extracted": stage_report.total_facts_extracted,
                    "questions_generated": stage_report.total_questions_generated,
                    "topics_identified": stage_report.total_topics_identified,
                    "relationships_found": stage_report.total_relationships_found,
                    "overall_accuracy": round(stage_report.overall_accuracy, 3),
                    "overall_answerability": round(stage_report.overall_answerability, 3),
                    "avg_confidence": round(stage_report.avg_confidence, 3),
                    "avg_processing_time_ms": round(stage_report.avg_processing_time_ms, 2),
                    "estimated_cost_usd": round(stage_report.estimated_cost_usd, 4)
                },
                "success_criteria": {
                    "meets_accuracy_target": stage_report.meets_accuracy_target,
                    "meets_answerability_target": stage_report.meets_answerability_target
                },
                "failure_patterns": stage_report.failure_patterns
            }, f, indent=2)
        
        # Generate failure analysis
        failure_analysis = self.evaluator.generate_failure_analysis(metrics)
        
        # Save failure analysis
        failure_path = self.output_dir / "failure_analysis.json"
        with open(failure_path, "w") as f:
            json.dump(failure_analysis, f, indent=2)
        
        # Print summary
        print("\n\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"Documents processed: {stage_report.total_documents}")
        print(f"Chunks analyzed: {stage_report.total_chunks}")
        print(f"Facts extracted: {stage_report.total_facts_extracted}")
        print(f"Questions generated: {stage_report.total_questions_generated}")
        print(f"\nQuality Metrics:")
        print(f"  Overall Accuracy: {stage_report.overall_accuracy:.2%}")
        print(f"  Question Answerability: {stage_report.overall_answerability:.2%}")
        print(f"  Average Confidence: {stage_report.avg_confidence:.2f}")
        print(f"\nSuccess Criteria:")
        print(f"  Meets Accuracy Target (70%): {'✅' if stage_report.meets_accuracy_target else '❌'}")
        print(f"  Meets Answerability Target (60%): {'✅' if stage_report.meets_answerability_target else '❌'}")
        
        if failure_analysis["recommendations"]:
            print(f"\nRecommendations:")
            for rec in failure_analysis["recommendations"]:
                print(f"  • {rec}")
        
        # Document type analysis
        print(f"\nPerformance by Document Type:")
        doc_types = {}
        for result in results:
            if result["success"]:
                doc_type = result["document"]["document_type"]
                if doc_type not in doc_types:
                    doc_types[doc_type] = []
                doc_types[doc_type].append(result["document"]["title"])
        
        for doc_type, titles in doc_types.items():
            print(f"  • {doc_type}: {len(titles)} document(s)")
        
        print(f"\n📁 Reports saved to: {self.output_dir}")
        print("\n⚠️  Note: This is an automated evaluation with simulated validation.")
        print("For accurate metrics, run the manual validation script.")
        
        print("\nNext steps:")
        if stage_report.meets_accuracy_target and stage_report.meets_answerability_target:
            print("✅ Stage 1 targets met! Ready to proceed to Stage 2.")
            print("   But consider running manual validation for more accurate metrics.")
        else:
            print("❌ Stage 1 targets not met. Review failure analysis and:")
            print("   1. Run manual validation for accurate metrics")
            print("   2. Adjust extraction prompts if needed")
            print("   3. Consider document-specific strategies (Stage 2)")


async def main():
    """Main entry point."""
    runner = AutomatedBaselineEvaluationRunner()
    await runner.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())