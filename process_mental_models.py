#!/usr/bin/env python3
"""
Process The Great Mental Models PDF with Neon database integration.

This script demonstrates the complete pipeline:
1. Extract knowledge from the PDF using Gemini 2.5 Pro
2. Store extracted knowledge in Neon database using existing schema  
3. Query the stored knowledge
4. Generate additional insights using the questioning system
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
from extraction.v2.state import StateStore
from extraction.v2.neon_storage import NeonStorage
from extraction.v2.extractors import select_model_for_document
from src.utils.logging import get_logger

# Load environment variables
load_dotenv()
logger = get_logger(__name__)

# PDF path - updated to match the actual filename in aplikacje folder
# Find the Mental Models PDF dynamically
PDF_PATTERN = "/Users/marcin/Desktop/aplikacje/The Great Mental Models*.pdf"
pdf_files = glob.glob(PDF_PATTERN)
PDF_PATH = pdf_files[0] if pdf_files else ""


async def main():
    """Main processing function."""
    print("🚀 Starting Mental Models PDF Processing with Neon Database Integration")
    print("=" * 80)
    
    # Verify PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        print("Please ensure the PDF is in the correct location.")
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
        
        # Create extraction config
        config = ExtractionConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            enable_human_validation=False,  # Disable for automated processing
            enhancers_enabled=["citation", "question", "relationship"]
        )
        
        # Create extraction pipeline
        pipeline = ExtractionPipeline(config)
        
        # 2. Select optimal model for this document
        print("\n🎯 Selecting optimal extraction model...")
        
        # Get PDF size for model selection
        pdf_size = os.path.getsize(PDF_PATH)
        print(f"📊 PDF size: {pdf_size / (1024*1024):.1f} MB")
        
        # Select model - force Gemini 2.5 Pro for quality
        model_config = select_model_for_document(
            doc_type="academic",  # Mental models is academic/educational
            doc_length=pdf_size,
            quality_required=True,  # We want the best quality
            budget_conscious=False,  # Use the best model
            use_gemini=True  # Use Gemini
        )
        
        print(f"🤖 Selected model: {model_config['model']} ({model_config['reason']})")
        
        # 3. Run extraction
        print("\n⚙️  Starting knowledge extraction...")
        print("This may take several minutes for a large document...")
        
        # Build requirements for the extraction
        requirements = {
            "model": model_config["model"],
            "extractor_class": model_config["extractor"],
            "quality_required": True,
            "use_gemini": True
        }
        
        result = await pipeline.extract(
            pdf_path=PDF_PATH,
            requirements=requirements
        )
        
        if not result.get("extraction"):
            print("❌ Extraction failed - no results returned")
            return
        
        extraction = result["extraction"]
        quality_report = result.get("quality_report", {})
        
        # Get full text chunks for RAG storage
        print("\n📄 Processing full text for RAG storage...")
        from extraction.pdf_processor import PDFProcessor
        pdf_processor = PDFProcessor()
        full_text_chunks = await pdf_processor.process_pdf(PDF_PATH)
        
        # Convert chunks to the format expected by Neon storage
        rag_chunks = []
        for chunk in full_text_chunks:
            rag_chunks.append({
                'content': chunk.content,
                'page_numbers': getattr(chunk, 'page_numbers', []),
                'chunk_metadata': {
                    'chunk_id': getattr(chunk, 'chunk_id', ''),
                    'token_count': len(chunk.content.split()) if chunk.content else 0
                }
            })
        
        print(f"📊 RAG Text Chunks: {len(rag_chunks)} chunks (avg {sum(len(c['content'].split()) for c in rag_chunks) // len(rag_chunks) if rag_chunks else 0} words/chunk)")
        
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
        from extraction.v2.state import ExtractionState
        
        # Create state with the current extraction results
        current_state = ExtractionState(
            id=str(uuid.uuid4()),
            pdf_path=PDF_PATH,
            current_step="completed",
            extraction=extraction.model_dump() if hasattr(extraction, 'model_dump') else (extraction.dict() if hasattr(extraction, 'dict') else extraction.__dict__),
            quality_report=quality_report,
            metadata={
                "model_used": model_config["model"],
                "processing_time": (datetime.utcnow() - result.get("start_time", datetime.utcnow())).total_seconds() if result.get("start_time") else 0,
                "document_type": "academic"
            },
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            status="completed"
        )
        
        # Store in Neon database (both extracted knowledge and full text chunks)
        success = await neon_storage.store_extraction(current_state, full_text_chunks=rag_chunks)
        
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
                
                test_queries = [
                    "mental models",
                    "thinking concepts", 
                    "decision making",
                    "cognitive biases",
                    "problem solving"
                ]
                
                for query in test_queries:
                    results = await neon_storage.query_knowledge(query, limit=3)
                    print(f"\n🔎 Query: '{query}'")
                    if results:
                        for i, result in enumerate(results, 1):
                            print(f"   {i}. [{result['type']}] {result['content'][:100]}...")
                            print(f"      Confidence: {result['confidence']:.3f}")
                    else:
                        print(f"   No results found")
                
                # 7. Get database statistics
                print("\n📊 Database Statistics:")
                stats = await neon_storage.get_database_stats()
                for key, value in stats.items():
                    print(f"   - {key.replace('_', ' ').title()}: {value}")
                
                print("\n🎉 Processing Complete!")
                print("The Mental Models book knowledge has been successfully extracted and stored in Neon database.")
                print("You can now query this knowledge through the database or use it for agent reasoning.")
                
            else:
                print("❌ Failed to store extraction in Neon database")
        else:
            print("❌ No extraction state found")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        print(f"❌ Processing failed: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())