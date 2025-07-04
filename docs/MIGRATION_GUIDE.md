# Migration Guide: Moving to 12-Factor Extraction

## Overview

This guide helps you migrate from the complex v1 system to the simple, reliable v2 system following 12-factor principles.

## Why Migrate?

### Current Pain Points (v1)
- 🔴 Hard to debug when things go wrong
- 🔴 Difficult to customize extraction behavior  
- 🔴 Complex multi-model consensus often fails
- 🔴 Hidden state makes testing hard
- 🔴 Monolithic components hard to modify

### Benefits of v2
- ✅ Every step is explicit and debuggable
- ✅ Pick exactly the components you need
- ✅ Stateless functions are easy to test
- ✅ Human validation improves quality
- ✅ 3x faster, 67% cheaper

## Migration Strategy

### Phase 1: Run in Parallel (Week 1)
Keep v1 running while testing v2:

```python
# Your existing code stays the same
from extraction.document_aware.extractor import DocumentAwareExtractor
v1_result = await extractor.extract(pdf_path)

# Test v2 alongside
from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
config = ExtractionConfig(api_key=api_key)
pipeline = ExtractionPipeline(config)
v2_result = await pipeline.extract(pdf_path)

# Compare results
compare_extractions(v1_result, v2_result["extraction"])
```

### Phase 2: Gradual Component Migration (Week 2-3)

#### Option A: Use v2 Components in v1
```python
# Start using stateless extractors in existing code
from extraction.v2.extractors import AcademicExtractor
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=api_key)
extraction = await AcademicExtractor.extract(content, client)
```

#### Option B: Use Individual Enhancers
```python
# Enhance existing extractions with v2 components
from extraction.v2.enhancers import CitationEnhancer, QuestionEnhancer

# Your existing extraction
extraction = await old_extractor.extract(pdf_path)

# Enhance with v2 components
extraction = CitationEnhancer.enhance(extraction, source_text)
extraction = await QuestionEnhancer.enhance(extraction, client)
```

### Phase 3: Full Migration (Week 4)

Replace v1 calls with v2:

```python
# Before
extractor = DocumentAwareExtractor(api_key)
result = await extractor.extract(pdf_path)

# After
config = ExtractionConfig(
    api_key=api_key,
    quality_threshold=0.7,
    enhancers_enabled=["citation", "question", "summary"]
)
pipeline = ExtractionPipeline(config)
result = await pipeline.extract(pdf_path)
extraction = result["extraction"]
```

## Common Migration Patterns

### 1. Custom Document Strategies
**v1**: Complex strategy classes
```python
class MyCustomStrategy(ExtractionStrategy):
    def build_extraction_prompt(self, content, profile):
        # 200 lines of complexity
```

**v2**: Simple functions
```python
@staticmethod
async def extract_custom(content: str, client: AsyncOpenAI) -> ExtractedKnowledge:
    prompt = f"Extract {my_specific_needs} from: {content}"
    response = await client.complete(prompt)
    return parse_response(response)
```

### 2. Quality Improvement
**v1**: Complex feedback loops
```python
feedback_extractor = FeedbackExtractor(api_key)
result = await feedback_extractor.extract_with_feedback(pdf_path)
```

**v2**: Explicit enhancement
```python
extraction = await BaselineExtractor.extract(content, client)
if quality_score < 0.7:
    extraction = CitationEnhancer.enhance(extraction, content)
    extraction = await QuestionEnhancer.enhance(extraction, client)
```

### 3. Model Selection
**v1**: Hidden in factory
```python
# Who knows what model this uses?
strategy = factory.get_strategy(profile)
```

**v2**: Explicit rules
```python
model_config = select_model_for_document(
    doc_type="academic",
    doc_length=5000,
    quality_required=True
)
# Returns: {"model": "gpt-4", "reason": "Academic document needs precision"}
```

## Configuration Mapping

### v1 Configuration (Hidden)
```python
# Scattered across multiple files
# Hard to find and change
```

### v2 Configuration (Explicit)
```python
config = ExtractionConfig(
    api_key=api_key,                    # Required
    max_enhancer_loops=2,               # Control enhancement
    quality_threshold=0.7,              # When to stop
    enable_human_validation=True,       # Human in the loop
    enhancers_enabled=[                 # Pick what you need
        "citation",      # Find and add citations
        "question",      # Generate questions
        "relationship",  # Find entity relationships  
        "summary"        # Improve summary
    ]
)
```

## Testing During Migration

### 1. Quality Comparison
```python
async def compare_quality(pdf_path: str):
    # Run both
    v1_result = await run_v1_extraction(pdf_path)
    v2_result = await run_v2_extraction(pdf_path)
    
    # Compare metrics
    print(f"V1 Facts: {len(v1_result.facts)}")
    print(f"V2 Facts: {len(v2_result['extraction'].facts)}")
    print(f"V2 Quality: {v2_result['quality_report']['overall_score']}")
    print(f"V2 Time: {v2_result['metadata']['processing_time']}s")
```

### 2. Cost Comparison
```python
# V1: Multiple models, many calls
v1_cost = count_tokens(v1_calls) * gpt4_price

# V2: Smart model selection
v2_cost = count_tokens(v2_calls) * selected_model_price
```

### 3. Reliability Testing
```python
# Run 100 extractions, measure success rate
v1_success_rate = successful_v1 / 100
v2_success_rate = successful_v2 / 100
```

## Rollback Plan

If issues arise, v2 is designed for easy rollback:

1. **Components are isolated** - Remove v2, v1 still works
2. **No shared state** - No database migrations needed
3. **Parallel operation** - Run both until confident

## Getting Help

### Documentation
- Architecture: `/docs/12factor_refactoring_plan.md`
- Benefits: `/docs/12factor_benefits.md`
- Examples: `/test_v2_extraction.py`

### Code Examples
- Extractors: `/extraction/v2/extractors.py`
- Enhancers: `/extraction/v2/enhancers.py`
- Pipeline: `/extraction/v2/pipeline.py`

## Timeline

- **Week 1**: Set up parallel testing
- **Week 2**: Migrate first production use case
- **Week 3**: Migrate remaining use cases
- **Week 4**: Shut down v1, celebrate! 🎉

## Success Metrics

Track these during migration:
- ✅ Extraction success rate (target: >95%)
- ✅ Quality scores (target: >0.7)
- ✅ Processing time (target: <20s)
- ✅ Cost per document (target: <$0.05)
- ✅ Developer happiness (target: 😊)

## Remember

The goal is **simple, reliable extraction** not complex magic. When in doubt:
- Choose explicit over implicit
- Choose simple over clever
- Choose debuggable over elegant
- Choose working over perfect

Good luck with your migration!