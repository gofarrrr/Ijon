# Testing and Evaluation Guide for Ijon PDF RAG System

This guide describes the comprehensive testing, evaluation, and calibration system for the Ijon PDF RAG system.

## Overview

The testing framework consists of three main components:

1. **Component Testing**: Unit tests for individual modules
2. **Evaluation System**: Quality metrics for RAG performance
3. **Calibration System**: Parameter optimization for best results

## Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --test-type component
python run_tests.py --test-type evaluation
python run_tests.py --test-type calibration
```

## Component Testing

### Test Coverage

The system includes comprehensive tests for:

- **PDF Processing**: Edge cases, corrupted files, large documents
- **Vector Databases**: All three adapters (Pinecone, Neon, Supabase)
- **Knowledge Graph**: Entity extraction, relationship building
- **Agents**: Query planning, multi-hop reasoning
- **Configuration**: Settings validation, environment handling

### Running Component Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_pdf_processing.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Key Test Scenarios

1. **PDF Reliability Tests**
   - Empty PDFs
   - Corrupted PDFs
   - Large PDFs (100+ pages)
   - Scanned PDFs (with/without OCR)
   - Special characters and Unicode
   - Concurrent processing

2. **Chunking Strategy Tests**
   - Sentence boundary preservation
   - Section-aware chunking
   - Overlap consistency
   - Memory efficiency

## Evaluation System

### Metrics

The evaluation system measures:

1. **Answer Quality**
   - Relevance (0-1): Semantic similarity to expected answer
   - Completeness (0-1): Coverage of required facts
   - Correctness (0-1): Factual accuracy
   - Coherence (0-1): Logical flow and consistency

2. **Retrieval Quality**
   - Precision: Relevant chunks / Retrieved chunks
   - Recall: Relevant chunks found / Total relevant chunks
   - F1 Score: Harmonic mean of precision and recall
   - MRR: Mean Reciprocal Rank
   - NDCG: Normalized Discounted Cumulative Gain

3. **Graph Quality** (if enabled)
   - Coverage: Percentage of entities/relations extracted
   - Accuracy: Correctness of extracted relationships
   - Connectivity: Graph connectivity score

4. **Performance**
   - Latency: End-to-end query time (ms)
   - Tokens: Total tokens consumed
   - Throughput: Queries per second

### Test Datasets

Create test datasets using:

```python
from tests.create_test_data import save_test_datasets
save_test_datasets()
```

This creates:
- `ml_comprehensive.json`: 7 ML-focused test cases
- `medical_basic.json`: Medical domain questions
- `legal_basic.json`: Legal domain questions

### Running Evaluations

```python
from src.rag.pipeline import RAGPipeline
from tests.test_evaluation import RAGEvaluator, create_sample_test_dataset

# Initialize pipeline
pipeline = await RAGPipeline.create()

# Create evaluator
evaluator = RAGEvaluator(pipeline)

# Run evaluation
dataset = create_sample_test_dataset()
results = await evaluator.evaluate_dataset(
    dataset,
    use_agent=True,
    save_results=True
)

print(f"Overall Score: {results['overall_score']['mean']:.2f}")
```

## Calibration System

### Tunable Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| chunk_size | 200-2000 | Size of text chunks |
| chunk_overlap | 0-500 | Overlap between chunks |
| retrieval_top_k | 3-15 | Number of chunks to retrieve |
| retrieval_min_score | 0.0-0.9 | Minimum similarity score |
| entity_confidence_threshold | 0.5-0.95 | Entity extraction confidence |
| graph_traversal_depth | 1-4 | Graph traversal depth |
| agent_temperature | 0.0-1.0 | Agent response temperature |
| agent_max_iterations | 1-5 | Maximum reasoning iterations |

### Calibration Methods

1. **Single Parameter Optimization**
   ```python
   from src.calibration import RAGCalibrator
   
   calibrator = RAGCalibrator(pipeline, evaluator)
   result = await calibrator.calibrate_parameter(
       parameter,
       test_dataset,
       optimization_metric="overall_score"
   )
   ```

2. **Auto-Calibration**
   ```python
   profile = await calibrator.auto_calibrate(
       test_dataset,
       parameters=["chunk_size", "retrieval_top_k"],
       optimization_metric="retrieval_f1"
   )
   ```

3. **Grid Search**
   ```python
   profile = await calibrator.grid_search(
       parameters,
       test_dataset,
       optimization_metric="answer_relevance"
   )
   ```

### Confidence Calibration

Calibrate confidence scores for better reliability:

```python
from src.calibration import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Add samples during evaluation
for prediction in predictions:
    calibrator.add_sample(
        predicted_confidence=prediction.confidence,
        was_correct=prediction.is_correct
    )

# Calculate calibration
results = calibrator.calibrate()
print(f"Expected Calibration Error: {results['expected_calibration_error']:.3f}")
```

## Debugging Tools

### Query Tracing

```python
from tests.test_runner import DebugTracer

tracer = DebugTracer()

# Add traces during execution
tracer.trace("retrieval_start", {"query": query})
tracer.trace("chunks_retrieved", {"chunks_retrieved": len(chunks)})
tracer.trace("answer_generated", {"answer_length": len(answer)})

# Get summary
print(tracer.get_trace_summary())
```

### Performance Monitoring

```python
from src.utils.logging import log_performance

@log_performance
async def process_query(query: str):
    # Function automatically logs execution time
    pass
```

## Best Practices

1. **Test Data Quality**
   - Create diverse test cases covering edge cases
   - Include domain-specific examples
   - Verify expected answers are accurate

2. **Evaluation Strategy**
   - Run evaluations on representative datasets
   - Use multiple metrics, not just overall score
   - Compare agent vs non-agent performance

3. **Calibration Tips**
   - Start with single parameter optimization
   - Use grid search for interacting parameters
   - Save and version calibration profiles

4. **Debugging Approach**
   - Enable query tracing for failed cases
   - Check retrieval quality before answer quality
   - Monitor performance metrics for bottlenecks

## Example: Complete Testing Workflow

```python
import asyncio
from pathlib import Path

async def complete_testing_workflow():
    # 1. Initialize system
    from src.rag.pipeline import RAGPipeline
    pipeline = await RAGPipeline.create()
    
    # 2. Process test PDFs
    test_pdf = Path("test_data/sample.pdf")
    await pipeline.process_pdf(test_pdf)
    
    # 3. Run evaluation
    from tests.test_evaluation import RAGEvaluator, create_sample_test_dataset
    evaluator = RAGEvaluator(pipeline)
    dataset = create_sample_test_dataset()
    
    baseline_results = await evaluator.evaluate_dataset(dataset)
    print(f"Baseline Score: {baseline_results['overall_score']['mean']:.2f}")
    
    # 4. Calibrate parameters
    from src.calibration import RAGCalibrator
    calibrator = RAGCalibrator(pipeline, evaluator)
    
    profile = await calibrator.auto_calibrate(
        dataset,
        parameters=["chunk_size", "retrieval_top_k"]
    )
    
    # 5. Re-evaluate with optimal parameters
    final_results = await evaluator.evaluate_dataset(dataset)
    print(f"Optimized Score: {final_results['overall_score']['mean']:.2f}")
    
    # 6. Save calibration profile
    print(f"Calibration profile saved: {profile.name}")

# Run the workflow
asyncio.run(complete_testing_workflow())
```

## Continuous Testing

For production deployments:

1. **Automated Testing**
   - Run component tests on every commit
   - Schedule daily evaluation runs
   - Monitor performance metrics

2. **A/B Testing**
   - Test new calibration profiles
   - Compare different retrieval strategies
   - Measure user satisfaction

3. **Feedback Loop**
   - Collect user feedback on answers
   - Update test datasets with real queries
   - Refine calibration based on usage
