#!/usr/bin/env python3
"""
Improved QA Generator with Better Source Grounding.
Addresses critical quality issues: source alignment, content hallucination, and confidence calibration.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
import psycopg2
from psycopg2.extras import Json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extraction.v2.extractors import StatelessExtractor
from src.utils.logging import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

class ImprovedQAGenerator:
    """
    Improved QA generation system with strict source grounding.
    Fixes issues: source mismatches, hallucinations, overconfident scoring.
    """
    
    def __init__(self, model: str = "gemini-2.5-pro", temperature: float = 0.2):
        """Initialize improved QA generator."""
        self.model = model
        self.temperature = temperature  # Lower temperature for more focused responses
        self.connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("NEON_CONNECTION_STRING not found in environment")
    
    def create_source_grounded_prompt(self, chunk_content: str, chunk_index: int) -> str:
        """Create a strictly source-grounded QA generation prompt."""
        return f"""You are a specialized Question-Answer Generation Agent with STRICT SOURCE GROUNDING requirements.

## CRITICAL REQUIREMENTS - MUST FOLLOW EXACTLY:
1. **ONLY use information present in the provided text chunk**
2. **DO NOT reference information from other parts of the book**
3. **DO NOT use your general knowledge or training data**
4. **Every fact in your questions and answers must be verifiable in the source text**

## SOURCE TEXT CHUNK (Chunk #{chunk_index}):
'''
{chunk_content}
'''

## TASK:
Generate 2-4 high-quality question-answer pairs using ONLY the information above.

## VALIDATION CHECKLIST (You must verify each Q&A):
- ✅ Can this question be answered using ONLY the text above?
- ✅ Does every fact in the answer appear in the source text?
- ✅ Am I avoiding references to people/events not mentioned in this chunk?
- ✅ Am I not adding external knowledge about mental models?

## OUTPUT FORMAT:
{{
  "source_validation": "I have verified that all content comes from the provided text chunk",
  "qa_pairs": [
    {{
      "question": "Question that can be answered from the text above",
      "answer": "Answer using only facts from the provided text",
      "question_type": "comprehension|application|analysis",
      "confidence": 0.1-1.0,
      "source_evidence": "Direct quote or paraphrase from the text that supports this Q&A",
      "mental_model_mentioned": "specific mental model name if explicitly mentioned in text, or 'none'"
    }}
  ],
  "chunk_summary": "Brief summary of what this specific chunk contains",
  "mental_models_found": ["list of mental models explicitly mentioned in this chunk"],
  "confidence_reasoning": "Why I assigned these confidence scores"
}}

## CONFIDENCE SCORING GUIDELINES:
- 0.9-1.0: Facts directly quoted from text, obvious comprehension questions
- 0.7-0.9: Clear inferences from text, well-supported applications
- 0.5-0.7: Reasonable interpretations, some uncertainty in application
- 0.3-0.5: Speculative applications, unclear connections
- 0.1-0.3: Uncertain or potentially inaccurate

## EXAMPLES OF WHAT NOT TO DO:
❌ "This demonstrates the mental model of Asymmetric Risk..." (if "Asymmetric Risk" isn't in the text)
❌ "Nicholas Winton's story shows..." (if Nicholas Winton isn't mentioned in THIS chunk)
❌ "This principle can be applied to business..." (if no business application is in the text)

## EXAMPLES OF GOOD PRACTICE:
✅ "The text states that..." (directly referencing the provided text)
✅ "According to this passage..." (clear attribution to source)
✅ "The author mentions..." (if the author explicitly mentions something)

Generate questions that help readers understand and apply the specific content in this chunk."""

    def create_validation_prompt(self, qa_pair: Dict, source_chunk: str) -> str:
        """Create a prompt to validate QA pair against source chunk."""
        return f"""You are a Quality Validation Agent. Your task is to verify if a question-answer pair is accurate based on its source text.

## SOURCE TEXT:
'''
{source_chunk}
'''

## QUESTION-ANSWER PAIR TO VALIDATE:
Question: {qa_pair['question']}
Answer: {qa_pair['answer']}

## VALIDATION TASK:
Check if this Q&A pair meets quality standards:

1. **Source Accuracy**: Does every fact in the Q&A come from the source text?
2. **No Hallucination**: Are there any details not present in the source?
3. **Proper Attribution**: Does the Q&A correctly represent what's in the text?

## OUTPUT FORMAT:
{{
  "is_valid": true/false,
  "accuracy_score": 0.0-1.0,
  "issues_found": ["list of specific problems"],
  "facts_not_in_source": ["facts mentioned in Q&A but not in source text"],
  "keyword_overlap_score": 0.0-1.0,
  "recommendation": "approve|revise|reject",
  "validation_explanation": "Detailed explanation of validation decision"
}}

Be strict in your validation. If ANY fact in the Q&A cannot be verified in the source text, mark it as invalid."""

    async def generate_validated_qa_pairs(self, chunk_content: str, chunk_index: int, document_id: str) -> Tuple[List[Dict], Dict]:
        """Generate QA pairs with validation step."""
        logger.info(f"   🎯 Generating validated QA for chunk {chunk_index}")
        
        # Step 1: Generate QA pairs with strict source grounding
        start_time = time.time()
        qa_prompt = self.create_source_grounded_prompt(chunk_content, chunk_index)
        
        try:
            qa_result = await StatelessExtractor.call_llm(
                client=None,
                prompt=qa_prompt,
                model=self.model,
                temperature=self.temperature
            )
            generation_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Failed to generate QA for chunk {chunk_index}: {e}")
            return [], {"error": str(e)}
        
        # Step 2: Validate each QA pair
        validated_qa_pairs = []
        validation_results = []
        
        qa_pairs = qa_result.get("qa_pairs", [])
        
        for i, qa_pair in enumerate(qa_pairs):
            try:
                # Validate the QA pair
                validation_prompt = self.create_validation_prompt(qa_pair, chunk_content)
                validation_result = await StatelessExtractor.call_llm(
                    client=None,
                    prompt=validation_prompt,
                    model=self.model,
                    temperature=0.1  # Very low temperature for validation
                )
                
                # Check validation result
                if validation_result.get("is_valid", False) and validation_result.get("accuracy_score", 0) >= 0.7:
                    # Adjust confidence based on validation
                    original_confidence = qa_pair.get("confidence", 0.5)
                    validation_score = validation_result.get("accuracy_score", 0.7)
                    adjusted_confidence = (original_confidence + validation_score) / 2
                    
                    validated_qa_pair = {
                        **qa_pair,
                        "confidence": round(adjusted_confidence, 2),
                        "validation_score": validation_result.get("accuracy_score", 0),
                        "validation_passed": True,
                        "source_evidence": qa_pair.get("source_evidence", ""),
                        "mental_model_mentioned": qa_pair.get("mental_model_mentioned", "none")
                    }
                    validated_qa_pairs.append(validated_qa_pair)
                    logger.info(f"      ✅ QA pair {i+1} passed validation (score: {validation_result.get('accuracy_score', 0):.2f})")
                else:
                    logger.warning(f"      ❌ QA pair {i+1} failed validation: {validation_result.get('validation_explanation', 'Unknown reason')}")
                
                validation_results.append(validation_result)
                
            except Exception as e:
                logger.warning(f"      ❌ Validation failed for QA pair {i+1}: {e}")
                validation_results.append({"error": str(e)})
        
        total_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            "source_grounded_generation": True,
            "validation_enabled": True,
            "generation_time_ms": int(generation_time * 1000),
            "total_time_ms": int(total_time * 1000),
            "original_qa_count": len(qa_pairs),
            "validated_qa_count": len(validated_qa_pairs),
            "validation_pass_rate": len(validated_qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            "chunk_summary": qa_result.get("chunk_summary", ""),
            "mental_models_found": qa_result.get("mental_models_found", []),
            "confidence_reasoning": qa_result.get("confidence_reasoning", ""),
            "validation_results": validation_results,
            "improved_generation": True,
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"      📊 Generated {len(validated_qa_pairs)}/{len(qa_pairs)} validated QA pairs ({metadata['validation_pass_rate']:.1%} pass rate)")
        
        return validated_qa_pairs, metadata
    
    async def process_document_chunks(self, document_id: str, max_chunks: int = 10) -> Tuple[int, Dict]:
        """Process chunks for a document with improved QA generation."""
        logger.info(f"🚀 Processing document {document_id} with improved QA generation")
        
        # Get document chunks
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, chunk_index, page_numbers
                    FROM content_chunks 
                    WHERE document_id = %s 
                    ORDER BY chunk_index ASC 
                    LIMIT %s
                """, (document_id, max_chunks))
                
                chunks = cur.fetchall()
        
        if not chunks:
            logger.warning(f"No chunks found for document {document_id}")
            return 0, {"error": "No chunks found"}
        
        logger.info(f"   📄 Processing {len(chunks)} chunks with improved QA generation")
        
        # Clear existing QA pairs for this document
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM qa_pairs WHERE document_id = %s", (document_id,))
                conn.commit()
        
        total_qa_pairs = 0
        processing_stats = {
            "chunks_processed": 0,
            "total_qa_generated": 0,
            "total_qa_validated": 0,
            "average_validation_rate": 0.0,
            "chunk_processing_times": [],
            "validation_pass_rates": [],
            "mental_models_found": [],
            "processing_errors": []
        }
        
        # Process each chunk
        for chunk_id, content, chunk_index, page_numbers in chunks:
            try:
                logger.info(f"   🔄 Processing chunk {chunk_index}")
                
                # Generate validated QA pairs
                validated_qa_pairs, metadata = await self.generate_validated_qa_pairs(
                    content, chunk_index, document_id
                )
                
                # Store validated QA pairs in database
                if validated_qa_pairs:
                    qa_count = await self.store_qa_pairs(
                        document_id, validated_qa_pairs, chunk_id, chunk_index
                    )
                    total_qa_pairs += qa_count
                
                # Update chunk with enhanced metadata
                await self.update_chunk_metadata(chunk_id, metadata)
                
                # Update processing stats
                processing_stats["chunks_processed"] += 1
                processing_stats["total_qa_generated"] += metadata.get("original_qa_count", 0)
                processing_stats["total_qa_validated"] += len(validated_qa_pairs)
                processing_stats["chunk_processing_times"].append(metadata.get("total_time_ms", 0))
                processing_stats["validation_pass_rates"].append(metadata.get("validation_pass_rate", 0))
                processing_stats["mental_models_found"].extend(metadata.get("mental_models_found", []))
                
                logger.info(f"   ✅ Chunk {chunk_index} complete: {len(validated_qa_pairs)} validated QA pairs")
                
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_index}: {e}"
                logger.error(error_msg)
                processing_stats["processing_errors"].append(error_msg)
        
        # Calculate final statistics
        if processing_stats["validation_pass_rates"]:
            processing_stats["average_validation_rate"] = sum(processing_stats["validation_pass_rates"]) / len(processing_stats["validation_pass_rates"])
        
        processing_stats["unique_mental_models"] = list(set(processing_stats["mental_models_found"]))
        processing_stats["average_processing_time_ms"] = sum(processing_stats["chunk_processing_times"]) / len(processing_stats["chunk_processing_times"]) if processing_stats["chunk_processing_times"] else 0
        
        logger.info(f"🎉 Document processing complete!")
        logger.info(f"   📊 Total validated QA pairs: {total_qa_pairs}")
        logger.info(f"   📈 Average validation rate: {processing_stats['average_validation_rate']:.1%}")
        logger.info(f"   🧠 Mental models found: {len(processing_stats['unique_mental_models'])}")
        
        return total_qa_pairs, processing_stats
    
    async def store_qa_pairs(self, document_id: str, qa_pairs: List[Dict], chunk_id: str, chunk_index: int) -> int:
        """Store validated QA pairs in database."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                stored_count = 0
                
                for qa_pair in qa_pairs:
                    qa_id = str(uuid.uuid4())
                    
                    cur.execute("""
                        INSERT INTO qa_pairs (
                            id, document_id, question, answer, answer_confidence,
                            answer_type, source_chunk_ids, human_verified, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s)
                    """, (
                        qa_id, document_id, qa_pair["question"], qa_pair["answer"],
                        qa_pair["confidence"], qa_pair.get("question_type", "comprehension"),
                        [chunk_id], False, datetime.now()
                    ))
                    stored_count += 1
                
                conn.commit()
                return stored_count
    
    async def update_chunk_metadata(self, chunk_id: str, metadata: Dict):
        """Update chunk with enhanced processing metadata."""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE content_chunks 
                    SET extraction_metadata = %s
                    WHERE id = %s
                """, (Json(metadata), chunk_id))
                conn.commit()
    
    async def test_improved_generation(self, document_id: str, max_chunks: int = 3) -> Dict:
        """Test the improved QA generation on a document."""
        logger.info(f"🧪 Testing improved QA generation on document {document_id}")
        
        start_time = time.time()
        qa_count, processing_stats = await self.process_document_chunks(document_id, max_chunks)
        total_time = time.time() - start_time
        
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "document_id": document_id,
            "chunks_tested": max_chunks,
            "total_processing_time": total_time,
            "qa_pairs_generated": qa_count,
            "processing_statistics": processing_stats,
            "improvement_metrics": {
                "validation_enabled": True,
                "source_grounding": True,
                "confidence_calibrated": True,
                "hallucination_prevention": True
            }
        }
        
        logger.info(f"✅ Test complete! Generated {qa_count} validated QA pairs in {total_time:.1f}s")
        
        return test_results

async def main():
    """Main function for testing improved QA generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test improved QA generation system")
    parser.add_argument("--document-id", required=True, help="Document ID to test")
    parser.add_argument("--max-chunks", type=int, default=3, help="Maximum chunks to process")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    
    args = parser.parse_args()
    
    try:
        generator = ImprovedQAGenerator(model=args.model, temperature=args.temperature)
        
        # Test improved generation
        test_results = await generator.test_improved_generation(
            args.document_id, args.max_chunks
        )
        
        # Save test results
        results_file = Path(f"improved_qa_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n🎉 IMPROVED QA GENERATION TEST COMPLETE!")
        print(f"📊 QA pairs generated: {test_results['qa_pairs_generated']}")
        print(f"📈 Validation rate: {test_results['processing_statistics']['average_validation_rate']:.1%}")
        print(f"🧠 Mental models found: {len(test_results['processing_statistics']['unique_mental_models'])}")
        print(f"💾 Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"❌ Error in improved QA generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())