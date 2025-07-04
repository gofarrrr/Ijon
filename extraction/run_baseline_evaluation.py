"""
Run baseline evaluation on test PDFs.

This script processes all test PDFs using the baseline extractor,
conducts manual validation, and generates comprehensive reports.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from extraction.baseline import BaselineExtractor
from extraction.evaluator import ExtractionEvaluator, ExtractionMetrics
from extraction.models import ExtractedKnowledge
from extraction.pdf_processor import PDFProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaselineEvaluationRunner:
    """Runs the complete baseline evaluation pipeline."""
    
    def __init__(self):
        """Initialize the evaluation runner."""
        self.test_dir = Path("test_documents")
        self.output_dir = Path("extraction/evaluation/stage1")
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
            metadata_path = pdf_path.with_suffix("_metadata.json")
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
            
            # Extract knowledge from first 5 chunks (for manageable validation)
            max_chunks = min(5, len(chunks))
            extractions = []
            
            for i in range(max_chunks):
                chunk = chunks[i]
                logger.info(f"  Extracting from chunk {i+1}/{max_chunks}")
                
                extraction = await self.extractor.extract(
                    chunk_id=f"{pdf_path.stem}_chunk_{i}",
                    content=chunk.content,
                    chunk_metadata={
                        "page": chunk.page_num,
                        "document": metadata["title"]
                    }
                )
                
                extractions.append({
                    "chunk": chunk,
                    "extraction": extraction
                })
            
            return {
                "document": metadata,
                "extractions": extractions,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {str(e)}")
            return {
                "document": metadata,
                "error": str(e),
                "success": False
            }
    
    def conduct_manual_validation(self, results: List[Dict[str, Any]]) -> List[ExtractionMetrics]:
        """Conduct manual validation for a subset of extractions."""
        print("\n" + "="*70)
        print("MANUAL VALIDATION PHASE")
        print("="*70)
        print("We'll validate a sample of facts and questions from each document.")
        print("This helps establish baseline accuracy metrics.")
        
        all_metrics = []
        
        # Validate 20 facts and 10 questions per document type
        facts_per_doc = 20
        questions_per_doc = 10
        
        for result in results:
            if not result["success"]:
                continue
            
            doc_metadata = result["document"]
            print(f"\n\n📄 Document: {doc_metadata['title']}")
            print(f"Type: {doc_metadata['document_type']}")
            print("-" * 50)
            
            # Collect all facts and questions from this document
            all_facts = []
            all_questions = []
            doc_metrics = []
            
            for ext_data in result["extractions"]:
                chunk = ext_data["chunk"]
                extraction = ext_data["extraction"]
                
                # Calculate base metrics
                metrics = self.evaluator.calculate_metrics(extraction)
                
                # Collect facts and questions for validation
                for fact in extraction.facts[:facts_per_doc // len(result["extractions"])]:
                    all_facts.append((fact, chunk.content, extraction, metrics))
                
                for question in extraction.questions[:questions_per_doc // len(result["extractions"])]:
                    all_questions.append((question, chunk.content, extraction, metrics))
                
                doc_metrics.append(metrics)
            
            # Validate facts
            if all_facts:
                print(f"\n📋 Validating {len(all_facts)} facts...")
                fact_validations = []
                
                for fact, source_text, extraction, metrics in all_facts:
                    validation = self.evaluator.validate_facts_manual(
                        extraction, source_text
                    )
                    fact_validations.extend(validation)
                    
                    # Update metrics with validation results
                    if validation:
                        self.evaluator.update_metrics_with_validation(
                            metrics, validation, []
                        )
            
            # Validate questions
            if all_questions:
                print(f"\n❓ Validating {len(all_questions)} questions...")
                question_validations = []
                
                for question, source_text, extraction, metrics in all_questions:
                    validation = self.evaluator.validate_questions_manual(
                        extraction, source_text
                    )
                    question_validations.extend(validation)
                    
                    # Update metrics with validation results
                    if validation:
                        self.evaluator.update_metrics_with_validation(
                            metrics, [], validation
                        )
            
            all_metrics.extend(doc_metrics)
        
        return all_metrics
    
    async def run_evaluation(self):
        """Run the complete baseline evaluation."""
        print("🚀 Starting Baseline Evaluation (Stage 1)")
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
                    serializable_result["extractions"] = [
                        {
                            "chunk_id": ext["extraction"].chunk_id,
                            "topics": len(ext["extraction"].topics),
                            "facts": len(ext["extraction"].facts),
                            "questions": len(ext["extraction"].questions),
                            "confidence": ext["extraction"].overall_confidence
                        }
                        for ext in result["extractions"]
                    ]
                else:
                    serializable_result["error"] = result.get("error")
                
                json.dump(serializable_result, f, indent=2)
        
        # Conduct manual validation
        print("\n\n📊 Starting Manual Validation...")
        print("Please answer honestly - this establishes our baseline metrics.")
        
        metrics = self.conduct_manual_validation(results)
        
        # Generate reports
        print("\n\n📈 Generating Reports...")
        
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
        stage_report_path = self.output_dir / "stage1_report.json"
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
                    "overall_accuracy": stage_report.overall_accuracy,
                    "overall_answerability": stage_report.overall_answerability,
                    "avg_confidence": stage_report.avg_confidence,
                    "avg_processing_time_ms": stage_report.avg_processing_time_ms,
                    "estimated_cost_usd": stage_report.estimated_cost_usd
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
        
        print(f"\n📁 Reports saved to: {self.output_dir}")
        print("\nNext steps:")
        if stage_report.meets_accuracy_target and stage_report.meets_answerability_target:
            print("✅ Stage 1 targets met! Ready to proceed to Stage 2.")
        else:
            print("❌ Stage 1 targets not met. Review failure analysis and adjust approach.")


async def main():
    """Main entry point."""
    runner = BaselineEvaluationRunner()
    await runner.run_evaluation()


if __name__ == "__main__":
    asyncio.run(main())