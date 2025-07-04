"""
Script to prepare test PDFs for Stage 1 evaluation.

This script helps set up the test documents by creating symbolic links
to existing PDFs and generating metadata files.
"""

import os
import json
import shutil
from pathlib import Path


# Map of test categories to source PDFs
TEST_PDFS = {
    "business_book": {
        "source": "/Users/marcin/Desktop/ksiazki pdf/ekonomia bitcoin /Principles_ Life and Work -- Ray Dalio -- 2017 -- Simon & Schuster -- 369e3f09bf06b69467904a5b77e562c3 -- Anna's Archive.pdf",
        "title": "Principles: Life and Work",
        "author": "Ray Dalio",
        "document_type": "business",
        "characteristics": ["business_strategy", "case_studies", "principles", "narrative"],
        "expected_difficulty": "medium"
    },
    "tutorial_guide": {
        "source": "/Users/marcin/Desktop/ksiazki pdf/storytelling i pisanie sprzedaż viral komunikacja/HOW TO TELL A STORY _ the essential guide to memorable -- The Moth; Meg Bowles; Catherine Burns; Jenifer Hixson; Sarah -- London, 2022 -- Short Books, -- 9781780725673 -- 11fd182a87eee8009e88b27f73a67976 -- Anna's Archive.pdf",
        "title": "How to Tell a Story: The Essential Guide",
        "author": "The Moth",
        "document_type": "tutorial",
        "characteristics": ["step_by_step", "examples", "practical_guide", "techniques"],
        "expected_difficulty": "easy"
    },
    "academic_paper": {
        "source": "/Users/marcin/Desktop/ksiazki pdf/storytelling i pisanie sprzedaż viral komunikacja/Scientific Advertising -- Claude C_ Hopkins -- 2022 -- SANAGE PUBLISHING HOUSE LLP -- 77b29e9ddb4eee6a1aa319f44e6da2a2 -- Anna's Archive.pdf",
        "title": "Scientific Advertising",
        "author": "Claude C. Hopkins",
        "document_type": "academic",
        "characteristics": ["research_based", "analytical", "data_driven", "methodology"],
        "expected_difficulty": "medium"
    },
    "technical_manual": {
        "source": "/Users/marcin/Desktop/ksiazki pdf/storytelling i pisanie sprzedaż viral komunikacja/The Art & Business Of Ghostwriting_ How To Make $10,000+ Per Month Writing For Other People Online-Different Publishing (2023).pdf",
        "title": "The Art & Business of Ghostwriting",
        "author": "Nicolas Cole",
        "document_type": "technical",
        "characteristics": ["processes", "techniques", "practical_steps", "business_model"],
        "expected_difficulty": "medium"
    },
    "historical_text": {
        "source": "/Users/marcin/Desktop/ksiazki pdf/storytelling i pisanje sprzedaż viral komunikacja/The Art of Public Speaking -- Dale Breckenridge Carnegie -- 1905 -- Feedbooks -- 46ef43eab080f1394301d84fca3e59f7 -- Anna's Archive.pdf",
        "title": "The Art of Public Speaking",
        "author": "Dale Carnegie",
        "document_type": "historical",
        "characteristics": ["classical_text", "historical_context", "rhetoric", "1905_publication"],
        "expected_difficulty": "hard"
    }
}


def prepare_test_documents():
    """Prepare test documents by copying PDFs and creating metadata."""
    
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for category, info in TEST_PDFS.items():
        source_path = Path(info["source"])
        
        if not source_path.exists():
            print(f"❌ Source not found for {category}: {source_path}")
            failed += 1
            continue
        
        # Copy PDF to test directory
        dest_pdf = test_dir / f"{category}.pdf"
        try:
            shutil.copy2(source_path, dest_pdf)
            print(f"✅ Copied {category}.pdf")
            
            # Create metadata file
            metadata = {
                "filename": f"{category}.pdf",
                "document_type": info["document_type"],
                "title": info["title"],
                "author": info["author"],
                "source": "Local collection",
                "page_count": "TBD",  # Will be determined during processing
                "language": "en",
                "characteristics": info["characteristics"],
                "expected_extraction_difficulty": info["expected_difficulty"],
                "notes": f"Test document for {category} category"
            }
            
            metadata_path = test_dir / f"{category}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Created metadata for {category}")
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed to prepare {category}: {str(e)}")
            failed += 1
    
    print(f"\n📊 Summary: {successful} successful, {failed} failed")
    
    if successful == 5:
        print("✅ All test documents prepared successfully!")
        print("\nNext steps:")
        print("1. Run: python extraction/run_baseline_evaluation.py")
        print("2. Complete manual validation when prompted")
        print("3. Review the generated reports in extraction/evaluation/")
    else:
        print("\n⚠️  Some documents failed to prepare. You may need to:")
        print("1. Find alternative PDFs for the missing categories")
        print("2. Update the paths in this script")
        print("3. Or download sample PDFs from the suggested sources in README.md")


if __name__ == "__main__":
    prepare_test_documents()