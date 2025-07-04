# Ijon V2 System Overview

## 🎯 Overview

The Ijon V2 system is a complete rewrite following 12-factor agent principles. It provides high-quality knowledge extraction from PDFs with human-in-the-loop validation, stateless components, and a service-oriented architecture for agent education.

## 🏗️ Architecture

### Core Principles (12-Factor Agent)
1. **Stateless Functions**: All extractors and enhancers are pure functions
2. **Explicit Control Flow**: No hidden magic, clear pipeline steps
3. **Human-in-the-Loop**: Validation UI for quality assurance
4. **Error Compaction**: Smart retry with exponential backoff
5. **Multiple Triggers**: CLI, API, scheduled, event-based
6. **State Management**: Pause/resume capability
7. **Service-Oriented**: Knowledge extraction as a service

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Entry Points                          │
├──────────────┬────────────┬────────────┬───────────────────┤
│     CLI      │    API     │  Schedule  │  Event Triggers   │
└──────┬───────┴──────┬─────┴──────┬─────┴─────────┬─────────┘
       │              │            │               │
       └──────────────┴────────────┴───────────────┘
                             │
                    ┌────────▼────────┐
                    │ Pipeline Manager │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ Document │       │  Model   │       │ Quality  │
   │ Profiler │       │  Router  │       │  Scorer  │
   └──────────┘       └──────────┘       └──────────┘
                             │
                    ┌────────▼────────┐
                    │   Extractors    │
                    │ • Baseline      │
                    │ • Academic      │
                    │ • Technical     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Enhancers     │
                    │ • Citation      │
                    │ • Relationship  │
                    │ • Question      │
                    │ • Summary       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Quality Check   │
                    └────────┬────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
           ┌──────────┐          ┌──────────┐
           │ Output   │          │Validation│
           │          │          │    UI    │
           └──────────┘          └──────────┘
```

## 📁 File Structure

```
extraction/v2/
├── __init__.py              # Package initialization
├── extractors.py            # Stateless extractors & model routing
├── enhancers.py             # Focused enhancement functions
├── pipeline.py              # Main extraction pipeline
├── state.py                 # State management for pause/resume
├── error_handling.py        # Smart retry & error compaction
├── human_contact.py         # Human validation channels
├── triggers.py              # Multiple trigger mechanisms
├── validation_ui.py         # Web UI for human validators
└── knowledge_service.py     # Service endpoints for agents

tests/
├── test_model_routing.py    # Model selection tests
├── test_enhancer_composition.py  # Enhancer tests
└── test_pipeline_integration.py  # Full pipeline tests
```

## 🔧 Key Components

### 1. Stateless Extractors
```python
# Pure functions with no side effects
async def extract(content: str, client: AsyncOpenAI, model: str = "gpt-3.5-turbo") -> ExtractedKnowledge:
    # Extract topics, facts, questions, relationships
    # Return immutable result
```

### 2. Model Router
```python
def select_model_for_document(doc_type: str, doc_length: int, quality_required: bool = False) -> Dict[str, Any]:
    # Deterministic routing based on:
    # - Document type (academic, technical, narrative)
    # - Document length (>50K chars → Claude)
    # - Quality requirements (high → GPT-4)
    # - Budget constraints (low → GPT-3.5)
```

### 3. Focused Enhancers
- **CitationEnhancer**: Adds evidence and citations to facts
- **RelationshipEnhancer**: Discovers entity relationships
- **QuestionEnhancer**: Generates Bloom's taxonomy questions
- **SummaryEnhancer**: Creates comprehensive summaries

### 4. Quality Scorer
```python
# Four dimensions of quality:
- Consistency: Facts align with topics (weight: 0.25)
- Grounding: Claims have evidence (weight: 0.35)
- Coherence: Logical relationships (weight: 0.20)
- Completeness: Coverage of content (weight: 0.20)
```

### 5. Human Validation UI
- Web interface for reviewing extractions
- Quality score visualization
- Structured feedback forms
- Confidence adjustment
- Item-level review

### 6. Knowledge Service
Provides extraction capabilities as a service:
- `extract_knowledge()`: Main extraction endpoint
- `analyze_extraction_quality()`: Quality analysis
- `get_extraction_examples()`: High-quality examples
- `get_extraction_insights()`: Domain-specific tips
- `check_extraction_status()`: Job status monitoring

## 🚀 Usage Examples

### Basic Extraction
```python
from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig

config = ExtractionConfig(
    api_key="your-api-key",
    quality_threshold=0.7,
    enable_human_validation=True,
    enhancers_enabled=["citation", "relationship"]
)

pipeline = ExtractionPipeline(config)
result = await pipeline.extract("document.pdf")
```

### Using the Knowledge Service
```python
from extraction.v2.knowledge_service import extract_knowledge, analyze_extraction_quality

# Extract knowledge
result = await extract_knowledge(
    content=document_text,
    doc_type="academic",
    apply_citation_enhancer=True
)

# Check quality
if result['quality_score'] < 0.7:
    quality = await analyze_extraction_quality(
        result['extraction'],
        document_text
    )
    # Apply recommendations
    for rec in quality['recommendations']:
        print(f"Improve: {rec}")
```

### Human Validation
```python
# Start validation UI server
python -m extraction.v2.validation_ui

# Extractions with quality < 0.6 automatically trigger validation
# Validators can:
# - Review extraction quality
# - Adjust confidence scores
# - Provide structured feedback
# - Approve/reject/request revision
```

## 📊 Quality Metrics

### Extraction Quality Scoring
- **High Quality (0.8+)**: Ready for use
- **Medium Quality (0.6-0.8)**: May need enhancement
- **Low Quality (<0.6)**: Requires human validation

### Performance Metrics
- Model routing decision: <0.1ms
- Citation enhancement: ~0.94ms per extraction
- Relationship enhancement: ~0.87ms per extraction
- Full pipeline: 2-10s depending on document size

## 🧪 Testing

### Test Coverage
- **Model Routing**: 8 comprehensive tests
- **Enhancer Composition**: 4 tests + performance benchmarks
- **Pipeline Integration**: End-to-end tests
- **Human Validation**: UI interaction tests

### Running Tests
```bash
# Model routing tests
python run_model_routing_tests.py

# Enhancer tests
python run_enhancer_tests.py

# Integration tests
python test_simple_v2.py
```

## 🔄 Continuous Improvement

### Planned Enhancements
1. **Metrics Dashboard**: Real-time monitoring
2. **Agent Education**: Multi-agent learning system
3. **Advanced Enhancers**: 
   - Semantic search enhancement
   - Multi-document relationship detection
   - Contradiction detection

### Feedback Loop
1. Human validators provide structured feedback
2. Feedback analyzed for patterns
3. System improvements based on common issues
4. Retraining of extraction strategies

## 🛡️ Security & Reliability

### Security Features
- No hardcoded API keys
- Environment variable configuration
- Secure state management
- Input validation

### Reliability Features
- Smart retry with exponential backoff
- Error compaction for debugging
- State persistence for pause/resume
- Graceful degradation

## 📚 Documentation

### For Developers
- [API Reference](./api_reference.md)
- [Extension Guide](./extension_guide.md)
- [Testing Guide](./testing_guide.md)

### For Users
- [Quick Start](./quickstart.md)
- [Configuration](./configuration.md)
- [Troubleshooting](./troubleshooting.md)

### For Validators
- [Validation Guide](./validation_guide.md)
- [Quality Criteria](./quality_criteria.md)

## 🎯 Key Achievements

1. **12-Factor Compliance**: Full implementation of all principles
2. **Stateless Architecture**: All components are pure functions
3. **Human-in-the-Loop**: Comprehensive validation UI
4. **Service-Oriented**: Ready for agent education
5. **High Quality**: Sophisticated scoring and enhancement
6. **Production Ready**: Error handling, monitoring, persistence

## 🚦 Current Status

✅ **Completed**:
- Stage 1-7: All core functionality
- 12-factor refactoring
- Model routing and enhancers
- Human validation UI
- Knowledge service for agents

🔄 **In Progress**:
- Metrics dashboard
- Multi-agent education system

The Ijon V2 system represents a complete, production-ready knowledge extraction platform that balances automation with human oversight, providing high-quality extractions that other agents can learn from and build upon.