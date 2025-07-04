"""
Document-aware extraction system.

Integrates document profiling with specialized extraction strategies.
"""

import asyncio
from typing import Optional, Dict, Any
import json

from openai import AsyncOpenAI
from pydantic import ValidationError

from extraction.models import ExtractedKnowledge, DocumentProfile
from extraction.strategies.document_profiler import DocumentProfiler
from extraction.strategies.strategy_factory import StrategyFactory
from extraction.baseline.extractor import BaselineExtractor
from extraction.pdf_processor import PDFProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentAwareExtractor:
    """Document-aware extraction that adapts to document types."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the document-aware extractor.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for extraction
        """
        self.api_key = openai_api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize components
        self.profiler = DocumentProfiler()
        self.strategy_factory = StrategyFactory(openai_api_key)
        self.baseline_extractor = BaselineExtractor(openai_api_key, model)
        self.pdf_processor = PDFProcessor()
        
        logger.info(f"Initialized DocumentAwareExtractor with model: {model}")
    
    async def extract(self, pdf_path: str) -> ExtractedKnowledge:
        """
        Extract knowledge using document-aware strategies.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted knowledge adapted to document type
        """
        try:
            # Step 1: Profile the document
            logger.info(f"Profiling document: {pdf_path}")
            profile = await self.profiler.profile_document(pdf_path)
            
            logger.info(f"Document type: {profile.document_type} "
                       f"(confidence: {profile.type_confidence:.2f})")
            logger.info(f"Recommended strategy: {profile.recommended_strategy}")
            
            # Step 2: Get appropriate strategy
            strategy = self.strategy_factory.get_strategy(profile)
            
            # Step 3: Extract content
            logger.info("Extracting PDF content...")
            chunks = await self.pdf_processor.process_pdf(pdf_path)
            content = "\n".join([chunk.content for chunk in chunks])
            
            if not content.strip():
                logger.error("No text content extracted from PDF")
                return self._create_empty_extraction("No text content found")
            
            # Truncate if too long
            if len(content) > 10000:
                logger.warning(f"Content too long ({len(content)} chars), truncating...")
                content = content[:10000]
            
            # Step 4: Apply extraction strategy
            if strategy is None:
                # Use baseline extractor
                logger.info("Using baseline extraction strategy")
                extraction = await self.baseline_extractor.extract(
                    chunk_id=f"doc_{pdf_path}",
                    content=content
                )
            else:
                # Use specialized strategy
                logger.info(f"Using {type(strategy).__name__} strategy")
                
                # Build custom prompt
                prompt = strategy.build_extraction_prompt(content, profile)
                
                # Get optimized parameters
                params = strategy.get_extraction_parameters(profile)
                
                # Call OpenAI with strategy-specific prompt
                response = await self._call_openai(prompt, params)
                
                logger.debug(f"Raw OpenAI response: {response[:500]}...")
                
                # Parse response
                extraction = self._parse_response(response)
                
                # Post-process with strategy
                extraction = strategy.post_process_extraction(extraction, profile)
                
                # Apply confidence adjustment
                confidence_adj = strategy.get_confidence_adjustment(profile)
                extraction.overall_confidence *= confidence_adj
            
            # Step 5: Add profiling metadata
            extraction.extraction_metadata.update({
                "document_type": profile.document_type,
                "type_confidence": profile.type_confidence,
                "strategy_used": profile.recommended_strategy,
                "structure_score": profile.structure_score,
                "ocr_quality": profile.ocr_quality,
                "special_elements": profile.special_elements
            })
            
            logger.info(f"Extraction complete. Overall confidence: {extraction.overall_confidence:.2f}")
            
            return extraction
            
        except Exception as e:
            logger.error(f"Document-aware extraction failed: {str(e)}")
            return self._create_empty_extraction(f"Extraction failed: {str(e)}")
    
    async def _call_openai(self, prompt: str, params: Dict[str, Any]) -> str:
        """Call OpenAI API with given prompt and parameters."""
        try:
            messages = [
                {"role": "system", "content": "You are an expert knowledge extractor. Extract structured information from documents and respond in JSON format."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _parse_response(self, response: str) -> ExtractedKnowledge:
        """Parse OpenAI response into ExtractedKnowledge."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse JSON
            data = json.loads(response)
            
            logger.info(f"Parsed JSON keys: {list(data.keys())}")
            
            # Normalize keys to lowercase
            if any(key.isupper() for key in data.keys()):
                logger.info("Converting uppercase keys to lowercase")
                data = {k.lower(): v for k, v in data.items()}
            
            # Handle different response formats
            if "topics" not in data and "facts" not in data:
                # Try to find nested structure
                logger.warning("No topics/facts at top level, searching nested...")
                for key in data:
                    if isinstance(data[key], dict) and "topics" in data[key]:
                        logger.info(f"Found extraction data under key: {key}")
                        data = data[key]
                        break
                else:
                    logger.warning("No nested extraction data found")
            
            # Log what we're extracting
            logger.info(f"Creating ExtractedKnowledge with {len(data.get('topics', []))} topics, "
                       f"{len(data.get('facts', []))} facts")
            
            # Process summary - handle dict or string
            summary = data.get("summary", "")
            if isinstance(summary, dict):
                # Try to extract a string from dict
                summary = summary.get("text", "") or summary.get("content", "") or str(summary)
            
            # Create ExtractedKnowledge
            return ExtractedKnowledge(
                topics=data.get("topics", []),
                facts=data.get("facts", []),
                relationships=data.get("relationships", []),
                questions=data.get("questions", []),
                overall_confidence=data.get("overall_confidence", 0.5),
                summary=summary,
                extraction_metadata=data.get("extraction_metadata", {})
            )
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse response: {str(e)}")
            logger.error(f"Response was: {response[:500]}...")
            return self._create_empty_extraction(f"Parse error: {str(e)}")
    
    def _create_empty_extraction(self, error_msg: str) -> ExtractedKnowledge:
        """Create empty extraction with error message."""
        return ExtractedKnowledge(
            topics=[],
            facts=[],
            relationships=[],
            questions=[],
            overall_confidence=0.0,
            summary=f"Extraction failed: {error_msg}",
            extraction_metadata={"error": error_msg}
        )
    
    async def _extract_baseline(self, pdf_path: str) -> ExtractedKnowledge:
        """Extract using baseline method for comparison."""
        try:
            # Extract content
            chunks = await self.pdf_processor.process_pdf(pdf_path)
            content = "\n".join([chunk.content for chunk in chunks])
            
            if not content.strip():
                return self._create_empty_extraction("No text content found")
            
            # Truncate if too long
            if len(content) > 10000:
                content = content[:10000]
            
            # Use baseline extractor
            return await self.baseline_extractor.extract(
                chunk_id=f"baseline_{pdf_path}",
                content=content
            )
            
        except Exception as e:
            logger.error(f"Baseline extraction failed: {str(e)}")
            return self._create_empty_extraction(f"Baseline extraction failed: {str(e)}")
    
    async def compare_with_baseline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Compare document-aware extraction with baseline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Comparison results
        """
        logger.info(f"Running comparison for: {pdf_path}")
        
        # Run both extractions
        baseline_task = self._extract_baseline(pdf_path)
        aware_task = self.extract(pdf_path)
        
        baseline_result, aware_result = await asyncio.gather(
            baseline_task, aware_task
        )
        
        # Compare results
        comparison = {
            "pdf_path": pdf_path,
            "baseline": {
                "topics": len(baseline_result.topics),
                "facts": len(baseline_result.facts),
                "relationships": len(baseline_result.relationships),
                "questions": len(baseline_result.questions),
                "confidence": baseline_result.overall_confidence,
                "avg_fact_confidence": sum(f.confidence for f in baseline_result.facts) / max(1, len(baseline_result.facts))
            },
            "document_aware": {
                "topics": len(aware_result.topics),
                "facts": len(aware_result.facts),
                "relationships": len(aware_result.relationships),
                "questions": len(aware_result.questions),
                "confidence": aware_result.overall_confidence,
                "avg_fact_confidence": sum(f.confidence for f in aware_result.facts) / max(1, len(aware_result.facts)),
                "strategy": aware_result.extraction_metadata.get("strategy_used", "unknown"),
                "document_type": aware_result.extraction_metadata.get("document_type", "unknown")
            },
            "improvements": {
                "topics_delta": len(aware_result.topics) - len(baseline_result.topics),
                "facts_delta": len(aware_result.facts) - len(baseline_result.facts),
                "confidence_delta": aware_result.overall_confidence - baseline_result.overall_confidence,
                "has_special_elements": bool(aware_result.extraction_metadata.get("special_elements", {}))
            }
        }
        
        return comparison