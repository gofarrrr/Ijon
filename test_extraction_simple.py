#!/usr/bin/env python3
"""
Simple extraction test - focused test of the Mental Models PDF.

This script tests the core extraction pipeline with better error handling
and shorter content for faster testing.
"""

import asyncio
import os
import sys
import glob
import uuid
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import AsyncOpenAI

from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
from extraction.v2.state import StateStore, ExtractionState
from extraction.v2.neon_storage import NeonStorage
from extraction.v2.extractors import select_model_for_document
from src.utils.logging import get_logger

# Load environment variables
load_dotenv()
logger = get_logger(__name__)

# Find the Mental Models PDF dynamically
PDF_PATTERN = "/Users/marcin/Desktop/aplikacje/The Great Mental Models*.pdf"
pdf_files = glob.glob(PDF_PATTERN)
PDF_PATH = pdf_files[0] if pdf_files else ""


async def simple_extraction_test():
    """Simple extraction test with better error handling."""
    print("🚀 Simple Extraction Test - Mental Models PDF")
    print("=" * 50)
    
    # Verify PDF exists
    if not PDF_PATH or not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATTERN}")
        return
    
    print(f"📄 Processing PDF: {os.path.basename(PDF_PATH)}")
    
    try:
        # 1. Initialize components
        print("\n🔧 Initializing extraction pipeline...")
        
        # Initialize state store and Neon storage
        state_store = StateStore()
        neon_storage = NeonStorage()
        
        # Verify Neon schema
        await neon_storage._verify_existing_schema()
        
        # Create extraction config with shorter timeouts
        config = ExtractionConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            enable_human_validation=False,  # Disable for automated processing
            enhancers_enabled=["question"]  # Just questions for speed
        )
        
        # Create extraction pipeline
        pipeline = ExtractionPipeline(config)
        
        # 2. Select optimal model
        print("\n🎯 Selecting extraction model...")
        
        # Force Gemini 2.5 Pro for quality
        model_config = {
            "model": "gemini-2.5-pro",
            "extractor": "BaselineExtractor",
            "reason": "Testing with Gemini 2.5 Pro"
        }
        
        print(f"🤖 Using model: {model_config['model']} ({model_config['reason']})")
        
        # 3. Run extraction with timeout protection
        print("\n⚙️  Starting knowledge extraction...")
        print("This should complete in 1-2 minutes...")
        
        # Build requirements for the extraction
        requirements = {
            "model": model_config["model"],
            "extractor_class": model_config.get("extractor"),
            "quality_required": True,
            "use_gemini": True
        }
        
        try:
            # Run extraction with asyncio timeout
            result = await asyncio.wait_for(
                pipeline.extract(pdf_path=PDF_PATH, requirements=requirements),
                timeout=300  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            print("⏰ Extraction timed out after 5 minutes")
            # Check if we have partial results in state store
            states = await state_store.list_active()
            if states:
                print(f"📋 Found {len(states)} partial extraction states")
                for state in states:
                    print(f"   - State {state.id[:8]}: {state.current_step} ({state.status})")
            return
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            logger.exception("Extraction error details:")
            return
        
        if not result.get("extraction"):
            print("❌ Extraction failed - no results returned")
            return
        
        extraction = result["extraction"]
        quality_report = result.get("quality_report", {})
        
        print(f"\n✅ Extraction completed!")
        print(f"📈 Quality Score: {quality_report.get('overall_score', 0):.3f}")
        print(f"📊 Extracted Content:")
        print(f"   - Topics: {len(extraction.topics if hasattr(extraction, 'topics') else [])}")
        print(f"   - Facts: {len(extraction.facts if hasattr(extraction, 'facts') else [])}")
        print(f"   - Relationships: {len(extraction.relationships if hasattr(extraction, 'relationships') else [])}")
        print(f"   - Questions: {len(extraction.questions if hasattr(extraction, 'questions') else [])}")
        
        # 4. Store in Neon database
        print("\n💾 Storing knowledge in Neon database...")
        
        # Create a state object for the current extraction
        current_state = ExtractionState(
            id=str(uuid.uuid4()),
            pdf_path=PDF_PATH,
            current_step="completed",
            extraction=extraction.model_dump() if hasattr(extraction, 'model_dump') else (extraction.dict() if hasattr(extraction, 'dict') else extraction.__dict__),
            quality_report=quality_report,
            metadata={
                "model_used": model_config["model"],
                "processing_time": result.get("processing_time", 0),
                "document_type": "academic"
            },
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            status="completed"
        )
        
        # Store in Neon database
        success = await neon_storage.store_extraction(current_state)
        
        if success:
            print("✅ Successfully stored extraction in Neon database!")
            
            # 5. Verify storage by querying
            print("\n🔍 Verifying storage...")
            
            # Get extraction summary
            summary = await neon_storage.get_extraction_summary(current_state.id)
            if summary:
                print(f"📋 Document stored: {summary.get('title', 'Unknown')}")
                print(f"   - Content chunks: {summary.get('content_chunks_count', 0)}")
                print(f"   - Distilled knowledge: {summary.get('distilled_knowledge_count', 0)}")
                print(f"   - QA pairs: {summary.get('qa_pairs_count', 0)}")
            
            # 6. Test knowledge queries
            print("\n🔍 Testing knowledge queries...")
            
            test_queries = ["mental models", "thinking"]
            
            for query in test_queries:
                results = await neon_storage.query_knowledge(query, limit=2)
                print(f"\n🔎 Query: '{query}'")
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. [{result['type']}] {result['content'][:80]}...")
                        print(f"      Confidence: {result['confidence']:.3f}")
                else:
                    print(f"   No results found")
            
            # 7. Get database statistics
            print("\n📊 Database Statistics:")
            stats = await neon_storage.get_database_stats()
            for key, value in stats.items():
                print(f"   - {key.replace('_', ' ').title()}: {value}")
            
            print("\n🎉 Simple Test Complete!")
            print("The Mental Models PDF has been successfully processed and stored.")
            
        else:
            print("❌ Failed to store extraction in Neon database")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"❌ Processing failed: {e}")
        return


if __name__ == "__main__":
    asyncio.run(simple_extraction_test())