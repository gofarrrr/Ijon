"""
Main extraction pipeline - owns control flow explicitly.

Following 12-factor principles:
- Explicit control flow (no hidden magic)
- Stateless components
- Small, focused loops
- Human-in-the-loop capability
"""

from typing import Dict, Any, Optional, List
import asyncio
from dataclasses import dataclass
from datetime import datetime

from openai import AsyncOpenAI

from extraction.models import ExtractedKnowledge, DocumentProfile
from extraction.quality.scorer import QualityScorer
from extraction.pdf_processor import PDFProcessor
from extraction.v2.extractors import (
    select_model_for_document,
    BaselineExtractor,
    AcademicExtractor,
    TechnicalExtractor
)
from extraction.v2.enhancers import (
    CitationEnhancer,
    QuestionEnhancer,
    RelationshipEnhancer,
    SummaryEnhancer
)
from extraction.v2.error_handling import ErrorHandler, SmartRetryExtractor
from extraction.v2.human_contact import HumanValidationService, ConsoleChannel
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for extraction - passed explicitly."""
    api_key: str
    max_enhancer_loops: int = 2
    quality_threshold: float = 0.7
    enable_human_validation: bool = True
    enhancers_enabled: List[str] = None
    
    def __post_init__(self):
        if self.enhancers_enabled is None:
            self.enhancers_enabled = ["citation", "question", "relationship", "summary"]


class ExtractionPipeline:
    """
    Main extraction pipeline - owns control flow.
    
    This is NOT a framework - it's explicit, understandable code.
    """
    
    def __init__(self, config: ExtractionConfig):
        """Initialize with explicit configuration."""
        self.config = config
        self.scorer = QualityScorer()
        self.pdf_processor = PDFProcessor()
        self.error_handler = ErrorHandler(max_retries=3)
        
        # Create client once, pass to stateless functions
        self.client = AsyncOpenAI(api_key=config.api_key)
        
        # Human validation service
        if config.enable_human_validation:
            self.human_validator = HumanValidationService([ConsoleChannel()])
        
        logger.info("Initialized extraction pipeline")
    
    async def extract(self, 
                     pdf_path: str,
                     requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main extraction flow - explicit steps.
        
        Args:
            pdf_path: Path to PDF
            requirements: User requirements (speed, quality, etc.)
            
        Returns:
            Dictionary with extraction and metadata
        """
        start_time = datetime.utcnow()
        
        # Step 1: Extract and profile document
        logger.info(f"Step 1: Processing {pdf_path}")
        chunks = await self.pdf_processor.process_pdf(pdf_path)
        content = "\n".join([chunk.content for chunk in chunks])
        
        if not content.strip():
            return self._error_result("No content extracted from PDF")
        
        # Simple profiling (could use micro-agents here)
        doc_profile = self._profile_document(chunks)
        logger.info(f"Document type: {doc_profile['type']}, length: {doc_profile['length']}")
        
        # Step 2: Select model and extractor
        logger.info("Step 2: Selecting extraction strategy")
        model_config = select_model_for_document(
            doc_type=doc_profile["type"],
            doc_length=doc_profile["length"],
            quality_required=requirements.get("quality", False) if requirements else False,
            budget_conscious=requirements.get("budget", False) if requirements else False
        )
        
        logger.info(f"Selected: {model_config['model']} - {model_config['reason']}")
        
        # Step 3: Initial extraction with smart retry
        logger.info("Step 3: Initial extraction")
        extractor_class = model_config["extractor"]
        
        # Wrap with smart retry
        smart_extractor = SmartRetryExtractor(extractor_class, self.error_handler)
        
        try:
            extraction = await smart_extractor.extract(
                content=content[:10000],  # Limit context
                client=self.client,
                model=model_config["model"]
            )
        except Exception as e:
            logger.error(f"Extraction failed after retries: {e}")
            error_summary = self.error_handler.get_error_summary()
            return self._error_result(f"Extraction failed: {str(e)}", error_summary)
        
        # Step 4: Score quality
        logger.info("Step 4: Quality assessment")
        quality_report = self.scorer.score_extraction(extraction, content)
        logger.info(f"Initial quality score: {quality_report['overall_score']:.3f}")
        
        # Step 5: Enhancement loop (small, focused)
        if quality_report["overall_score"] < self.config.quality_threshold:
            logger.info("Step 5: Enhancement loop")
            extraction = await self._enhancement_loop(
                extraction=extraction,
                quality_report=quality_report,
                source_content=content,
                max_iterations=self.config.max_enhancer_loops
            )
            
            # Re-score after enhancements
            quality_report = self.scorer.score_extraction(extraction, content)
            logger.info(f"Enhanced quality score: {quality_report['overall_score']:.3f}")
        
        # Step 6: Human validation (if enabled and needed)
        human_feedback = None
        if (self.config.enable_human_validation and 
            quality_report["overall_score"] < 0.6):
            logger.info("Step 6: Requesting human validation")
            human_feedback = await self._request_human_validation(
                extraction, quality_report, pdf_path
            )
        
        # Final result
        end_time = datetime.utcnow()
        
        return {
            "extraction": extraction,
            "quality_report": quality_report,
            "metadata": {
                "pdf_path": pdf_path,
                "document_profile": doc_profile,
                "model_used": model_config["model"],
                "extractor_used": extractor_class.__name__,
                "processing_time": (end_time - start_time).total_seconds(),
                "enhancements_applied": self.config.enhancers_enabled,
                "human_validated": human_feedback is not None
            },
            "human_feedback": human_feedback
        }
    
    def _profile_document(self, chunks: List[Any]) -> Dict[str, Any]:
        """Simple document profiling - pure function."""
        total_length = sum(len(chunk.content) for chunk in chunks)
        
        # Simple type detection
        content_sample = " ".join([chunk.content[:500] for chunk in chunks[:3]])
        doc_type = "unknown"
        
        if any(term in content_sample.lower() for term in ["abstract", "methodology", "hypothesis"]):
            doc_type = "academic"
        elif any(term in content_sample.lower() for term in ["config", "install", "api", "function"]):
            doc_type = "technical"
        elif any(term in content_sample.lower() for term in ["chapter", "story", "business"]):
            doc_type = "narrative"
        
        return {
            "type": doc_type,
            "length": total_length,
            "chunks": len(chunks)
        }
    
    async def _enhancement_loop(self,
                               extraction: ExtractedKnowledge,
                               quality_report: Dict[str, Any],
                               source_content: str,
                               max_iterations: int) -> ExtractedKnowledge:
        """
        Small enhancement loop - explicit control.
        
        This is a small, focused loop (not a complex agent).
        """
        for iteration in range(max_iterations):
            logger.info(f"Enhancement iteration {iteration + 1}/{max_iterations}")
            
            # Apply enhancers based on weaknesses
            weaknesses = quality_report["weaknesses"]
            
            for weakness in weaknesses:
                if weakness["dimension"] == "grounding" and "citation" in self.config.enhancers_enabled:
                    logger.info("Applying citation enhancement")
                    extraction = CitationEnhancer.enhance(extraction, source_content)
                
                elif weakness["dimension"] == "completeness" and "question" in self.config.enhancers_enabled:
                    logger.info("Applying question enhancement")
                    extraction = await QuestionEnhancer.enhance(extraction, self.client)
                
                elif weakness["dimension"] == "coherence":
                    if "relationship" in self.config.enhancers_enabled:
                        logger.info("Applying relationship enhancement")
                        extraction = RelationshipEnhancer.enhance(extraction)
                    
                    if "summary" in self.config.enhancers_enabled and not extraction.summary:
                        logger.info("Applying summary enhancement")
                        extraction = await SummaryEnhancer.enhance(extraction, self.client)
            
            # Re-score to check improvement
            new_quality = self.scorer.score_extraction(extraction, source_content)
            
            # Stop if good enough
            if new_quality["overall_score"] >= self.config.quality_threshold:
                logger.info(f"Quality threshold reached: {new_quality['overall_score']:.3f}")
                break
            
            # Stop if no improvement
            if new_quality["overall_score"] <= quality_report["overall_score"]:
                logger.info("No improvement, stopping enhancement")
                break
            
            quality_report = new_quality
        
        return extraction
    
    async def _request_human_validation(self,
                                      extraction: ExtractedKnowledge,
                                      quality_report: Dict[str, Any],
                                      pdf_path: str = None) -> Dict[str, Any]:
        """
        Request human validation through validation UI.
        """
        from extraction.v2.validation_ui import create_validation_request, validation_store
        import uuid
        
        logger.info("Human validation requested due to low quality score")
        
        # Create validation request
        validation_id = str(uuid.uuid4())
        val_state = create_validation_request(
            state_id=validation_id,
            extraction=extraction,
            quality_report=quality_report,
            pdf_path=pdf_path or "unknown.pdf"
        )
        
        # Save to validation store
        await validation_store.save(val_state)
        
        logger.info(f"Validation request created: {validation_id}")
        logger.info(f"Visit http://localhost:8001/validator/{validation_id} to review")
        
        # In a real system, this would wait for validation or send notifications
        # For now, return a placeholder indicating validation is pending
        return {
            "status": "pending_validation",
            "validation_id": validation_id,
            "validation_url": f"http://localhost:8001/validator/{validation_id}",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Human validation requested. Visit the URL to provide feedback."
        }
    
    def _error_result(self, error: str, error_summary: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create error result - pure function."""
        metadata = {
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error_summary:
            metadata["error_details"] = error_summary
        
        return {
            "extraction": None,
            "quality_report": None,
            "metadata": metadata
        }


# Example usage - explicit and clear
async def extract_document(pdf_path: str, api_key: str):
    """Example of using the pipeline - no magic."""
    
    # Explicit configuration
    config = ExtractionConfig(
        api_key=api_key,
        max_enhancer_loops=2,
        quality_threshold=0.7,
        enable_human_validation=False,  # For testing
        enhancers_enabled=["citation", "question", "summary"]
    )
    
    # Create pipeline
    pipeline = ExtractionPipeline(config)
    
    # Extract with requirements
    result = await pipeline.extract(
        pdf_path=pdf_path,
        requirements={
            "quality": True,
            "budget": False
        }
    )
    
    return result