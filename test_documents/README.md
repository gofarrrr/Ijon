# Test Documents for Quality Extraction System

This directory contains test PDFs for validating the extraction system across different document types.

## Required Documents

For Stage 1 baseline testing, we need 5 diverse PDF types:

1. **Technical Manual** (`technical_manual.pdf`)
   - Example: Software documentation, API reference, or hardware manual
   - Characteristics: Structured format, code examples, specifications
   - Sources: Open source project docs, technical specifications

2. **Academic Paper** (`academic_paper.pdf`)
   - Example: Research paper, journal article
   - Characteristics: Abstract, citations, methodology, results
   - Sources: arXiv, PubMed Central, open access journals

3. **Business Book** (`business_book.pdf`)
   - Example: Management, strategy, or entrepreneurship book chapter
   - Characteristics: Narrative style, case studies, business concepts
   - Sources: Open textbooks, sample chapters

4. **Tutorial/Guide** (`tutorial_guide.pdf`)
   - Example: How-to guide, programming tutorial, cookbook
   - Characteristics: Step-by-step instructions, examples, tips
   - Sources: Open educational resources, documentation sites

5. **Historical Text** (`historical_text.pdf`)
   - Example: Historical document, biography excerpt, cultural text
   - Characteristics: Dense prose, contextual information, dates/events
   - Sources: Project Gutenberg, Internet Archive

## Document Metadata

Each PDF should have an accompanying metadata file (`{filename}_metadata.json`) with:

```json
{
  "filename": "technical_manual.pdf",
  "document_type": "technical",
  "title": "Document Title",
  "source": "URL or source description",
  "page_count": 50,
  "language": "en",
  "characteristics": ["structured", "code_examples", "diagrams"],
  "expected_extraction_difficulty": "medium",
  "notes": "Any special considerations for this document"
}
```

## Sample Sources for Test Documents

### Free/Open Sources:
- **Technical**: https://github.com/EbookFoundation/free-programming-books
- **Academic**: https://arxiv.org/
- **Business**: https://opentextbc.ca/businessopentexts/
- **Tutorial**: https://www.tutorialspoint.com/
- **Historical**: https://www.gutenberg.org/

## Usage

Place your test PDFs in this directory before running the baseline evaluation:

```bash
test_documents/
├── README.md
├── technical_manual.pdf
├── technical_manual_metadata.json
├── academic_paper.pdf
├── academic_paper_metadata.json
├── business_book.pdf
├── business_book_metadata.json
├── tutorial_guide.pdf
├── tutorial_guide_metadata.json
├── historical_text.pdf
└── historical_text_metadata.json
```

The evaluation scripts will automatically discover and process these documents.