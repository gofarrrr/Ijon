"""
Feedback-based extraction system.

Uses quality scores to identify weaknesses and re-extract with targeted improvements.
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


class FeedbackExtractor:
    """Extraction system with quality feedback loop."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize feedback extractor.
        
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
        
        # Feedback configuration
        self.max_iterations = 3
        self.quality_threshold = 0.7
        
        logger.info(f"Initialized FeedbackExtractor with model: {model}")
    
    async def extract_with_feedback(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract knowledge with quality feedback loop.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Final extraction with quality scores and iteration history
        """
        logger.info(f"Starting extraction with feedback for: {pdf_path}")
        
        iteration_history = []
        best_extraction = None
        best_score = 0.0
        
        # Get source content for grounding checks
        source_content = await self._get_source_content(pdf_path)
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Extract
            if iteration == 0:
                # First iteration - use standard extraction
                extraction = await self.base_extractor.extract(pdf_path)
            else:
                # Subsequent iterations - use feedback
                extraction = await self._extract_with_improvements(
                    pdf_path, 
                    previous_extraction=best_extraction,
                    quality_scores=iteration_history[-1]["quality_scores"],
                    source_content=source_content
                )
            
            # Score quality
            quality_scores = self.scorer.score_extraction(extraction, source_content)
            
            # Record iteration
            iteration_data = {
                "iteration": iteration + 1,
                "extraction": extraction,
                "quality_scores": quality_scores,
                "overall_score": quality_scores["overall_score"]
            }
            iteration_history.append(iteration_data)
            
            # Update best if improved
            if quality_scores["overall_score"] > best_score:
                best_extraction = extraction
                best_score = quality_scores["overall_score"]
                logger.info(f"New best score: {best_score:.3f}")
            
            # Check if good enough
            if quality_scores["overall_score"] >= self.quality_threshold:
                logger.info(f"Quality threshold reached: {quality_scores['overall_score']:.3f}")
                break
            
            # Check if no improvement needed
            if not quality_scores["needs_reextraction"]:
                logger.info("No re-extraction needed based on quality scores")
                break
        
        # Prepare final result
        final_scores = self.scorer.score_extraction(best_extraction, source_content)
        
        return {
            "extraction": best_extraction,
            "quality_scores": final_scores,
            "iterations": len(iteration_history),
            "iteration_history": [
                {
                    "iteration": h["iteration"],
                    "overall_score": h["overall_score"],
                    "dimension_scores": h["quality_scores"]["dimension_scores"]
                }
                for h in iteration_history
            ],
            "improvement": best_score - iteration_history[0]["overall_score"],
            "final_suggestions": final_scores["suggestions"]
        }
    
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
    
    async def _extract_with_improvements(self, pdf_path: str,
                                        previous_extraction: ExtractedKnowledge,
                                        quality_scores: Dict[str, Any],
                                        source_content: str) -> ExtractedKnowledge:
        """
        Re-extract with targeted improvements based on quality feedback.
        
        Args:
            pdf_path: Path to PDF
            previous_extraction: Previous extraction attempt
            quality_scores: Quality scores identifying weaknesses
            source_content: Source content for reference
            
        Returns:
            Improved extraction
        """
        logger.info("Performing targeted re-extraction based on feedback")
        
        # Build improvement prompt
        improvement_prompt = self._build_improvement_prompt(
            quality_scores, previous_extraction, source_content
        )
        
        # Re-extract with improvements
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert knowledge extractor. Improve the extraction based on specific feedback."
                    },
                    {
                        "role": "user",
                        "content": improvement_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            return self._parse_improved_extraction(response_text, previous_extraction)
            
        except Exception as e:
            logger.error(f"Re-extraction failed: {str(e)}")
            return previous_extraction  # Return previous if re-extraction fails
    
    def _build_improvement_prompt(self, quality_scores: Dict[str, Any],
                                 previous_extraction: ExtractedKnowledge,
                                 source_content: str) -> str:
        """Build targeted improvement prompt based on quality feedback."""
        
        # Identify main issues
        weaknesses = quality_scores["weaknesses"]
        suggestions = quality_scores["suggestions"]
        detailed_scores = quality_scores["detailed_scores"]
        
        prompt_parts = [
            "Improve the following knowledge extraction based on quality feedback.\n",
            f"Current overall quality score: {quality_scores['overall_score']:.2f}\n",
            "\nWEAKNESSES IDENTIFIED:"
        ]
        
        # Add specific weaknesses
        for weakness in weaknesses:
            prompt_parts.append(f"- {weakness['dimension'].title()}: "
                              f"score {weakness['score']:.2f} ({weakness['severity']})")
        
        # Add specific issues
        prompt_parts.append("\nSPECIFIC ISSUES:")
        
        # Consistency issues
        if detailed_scores["consistency"]["issues"]:
            prompt_parts.append("\nConsistency Issues:")
            for issue in detailed_scores["consistency"]["issues"][:3]:
                if issue["type"] == "contradiction":
                    prompt_parts.append(f"- Contradiction between facts: "
                                      f"'{issue['fact1']}' vs '{issue['fact2']}'")
                elif issue["type"] == "missing_entity":
                    prompt_parts.append(f"- Relationship references undefined entity: "
                                      f"{issue['relationship']}")
        
        # Grounding issues
        if detailed_scores["grounding"]["issues"]:
            prompt_parts.append("\nGrounding Issues:")
            for issue in detailed_scores["grounding"]["issues"][:3]:
                if issue["type"] == "missing_evidence":
                    prompt_parts.append(f"- Fact lacks evidence: '{issue['fact']}'")
                elif issue["type"] == "not_in_source":
                    prompt_parts.append(f"- Fact not found in source: '{issue['fact']}'")
        
        # Coherence issues
        if detailed_scores["coherence"]["issues"]:
            prompt_parts.append("\nCoherence Issues:")
            for issue in detailed_scores["coherence"]["issues"][:3]:
                if issue["type"] == "irrelevant_question":
                    prompt_parts.append(f"- Question not related to content: "
                                      f"'{issue['question']}'")
                elif issue["type"] == "summary_topic_mismatch":
                    prompt_parts.append(f"- Summary doesn't cover main topics")
        
        # Add improvement instructions
        prompt_parts.append("\nIMPROVEMENT INSTRUCTIONS:")
        for suggestion in suggestions:
            prompt_parts.append(f"- {suggestion['suggestion']} "
                              f"(Priority: {suggestion['priority']})")
        
        # Add previous extraction
        prompt_parts.append("\nPREVIOUS EXTRACTION:")
        prompt_parts.append(json.dumps({
            "topics": [{"name": t.name, "description": t.description} 
                      for t in previous_extraction.topics],
            "facts": [{"claim": f.claim, "evidence": f.evidence} 
                     for f in previous_extraction.facts],
            "relationships": [{"source": r.source_entity, "target": r.target_entity,
                             "type": r.relationship_type}
                            for r in previous_extraction.relationships],
            "questions": [{"text": q.question_text, "answer": q.expected_answer}
                         for q in previous_extraction.questions],
            "summary": previous_extraction.summary
        }, indent=2))
        
        # Add source excerpt for grounding
        if source_content:
            prompt_parts.append(f"\nSOURCE EXCERPT (first 1000 chars):")
            prompt_parts.append(source_content[:1000] + "...")
        
        # Final instructions
        prompt_parts.append("\nProvide an improved extraction that addresses these issues.")
        prompt_parts.append("Respond with a JSON object with the same structure but improved content.")
        prompt_parts.append("Focus especially on the identified weaknesses and follow the improvement instructions.")
        
        return "\n".join(prompt_parts)
    
    def _parse_improved_extraction(self, response_text: str, 
                                  previous: ExtractedKnowledge) -> ExtractedKnowledge:
        """Parse improved extraction from response."""
        try:
            # Clean response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Parse JSON
            data = json.loads(response_text)
            
            # Normalize keys
            if any(key.isupper() for key in data.keys()):
                data = {k.lower(): v for k, v in data.items()}
            
            # Process topics
            topics = []
            for topic_data in data.get("topics", []):
                if isinstance(topic_data, str):
                    # Convert string to dict
                    topics.append({
                        "name": topic_data,
                        "description": "",
                        "confidence": 0.5
                    })
                elif isinstance(topic_data, dict):
                    topics.append(topic_data)
            
            # Process facts
            facts = []
            for fact_data in data.get("facts", []):
                if isinstance(fact_data, str):
                    facts.append({
                        "claim": fact_data,
                        "evidence": "",
                        "confidence": 0.5
                    })
                elif isinstance(fact_data, dict):
                    facts.append(fact_data)
            
            # Process relationships
            relationships = []
            for rel_data in data.get("relationships", []):
                if isinstance(rel_data, str):
                    # Try to parse relationship string
                    parts = rel_data.split(" ")
                    if len(parts) >= 3:
                        relationships.append({
                            "source_entity": parts[0],
                            "target_entity": parts[-1],
                            "relationship_type": "relates_to",
                            "confidence": 0.5
                        })
                elif isinstance(rel_data, dict):
                    relationships.append(rel_data)
            
            # Process questions
            questions = []
            for q_data in data.get("questions", []):
                if isinstance(q_data, str):
                    questions.append({
                        "question_text": q_data,
                        "expected_answer": "",
                        "cognitive_level": "understand",
                        "difficulty": 3,
                        "confidence": 0.5
                    })
                elif isinstance(q_data, dict):
                    questions.append(q_data)
            
            # Create improved extraction
            return ExtractedKnowledge(
                topics=topics,
                facts=facts,
                relationships=relationships,
                questions=questions,
                summary=data.get("summary", previous.summary) if isinstance(data.get("summary"), str) else previous.summary,
                overall_confidence=data.get("overall_confidence", previous.overall_confidence),
                extraction_metadata={
                    **previous.extraction_metadata,
                    "improved": True,
                    "improvement_iteration": 1
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse improved extraction: {str(e)}")
            return previous
    
    async def compare_with_baseline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Compare feedback extraction with baseline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Comparison results
        """
        logger.info(f"Running feedback comparison for: {pdf_path}")
        
        # Run baseline extraction
        baseline_result = await self.base_extractor.extract(pdf_path)
        
        # Run feedback extraction
        feedback_result = await self.extract_with_feedback(pdf_path)
        
        # Score both
        source_content = await self._get_source_content(pdf_path)
        baseline_scores = self.scorer.score_extraction(baseline_result, source_content)
        
        # Compare
        comparison = {
            "pdf_path": pdf_path,
            "baseline": {
                "overall_score": baseline_scores["overall_score"],
                "dimension_scores": baseline_scores["dimension_scores"],
                "topics": len(baseline_result.topics),
                "facts": len(baseline_result.facts),
                "iterations": 1
            },
            "feedback": {
                "overall_score": feedback_result["quality_scores"]["overall_score"],
                "dimension_scores": feedback_result["quality_scores"]["dimension_scores"],
                "topics": len(feedback_result["extraction"].topics),
                "facts": len(feedback_result["extraction"].facts),
                "iterations": feedback_result["iterations"]
            },
            "improvements": {
                "overall_score_gain": (feedback_result["quality_scores"]["overall_score"] - 
                                     baseline_scores["overall_score"]),
                "iteration_improvement": feedback_result["improvement"],
                "dimensions_improved": sum(
                    1 for dim in ["consistency", "grounding", "coherence", "completeness"]
                    if feedback_result["quality_scores"]["dimension_scores"][dim] > 
                       baseline_scores["dimension_scores"][dim]
                )
            }
        }
        
        return comparison