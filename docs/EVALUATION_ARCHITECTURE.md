# Ijon RAG Evaluation Architecture

## Overview

The Ijon RAG system implements a comprehensive evaluation framework that monitors, measures, and improves system performance across all components. This document describes the unified evaluation architecture implemented in Phase 3.

## Architecture Components

### 1. Metrics Collector (`src/evaluation/metrics_collector.py`)

The centralized metrics collection system that:
- Buffers metrics in memory for efficient batch processing
- Stores metrics in PostgreSQL for historical analysis
- Provides real-time aggregations for dashboards
- Integrates with existing logging infrastructure

**Key Features:**
- Async metric recording with buffering
- Automatic metric aggregation (count, sum, min, max, avg)
- Time-series data with hourly rollups
- Decorator for automatic latency and success tracking

**Usage:**
```python
# Manual metric recording
await metrics_collector.record_metric(
    'entities_extracted', 45, 'entity_extractor', 'extract_batch'
)

# Automatic collection with decorator
@collect_metrics('reranker', 'hybrid_rerank')
async def hybrid_rerank(self, query, results):
    # Method implementation
    pass
```

### 2. Evaluation Orchestrator (`src/evaluation/evaluation_orchestrator.py`)

Coordinates all evaluation components and provides unified interfaces for:
- Retrieval quality assessment (precision, recall, F1, NDCG)
- Extraction quality evaluation (using existing evaluators)
- Graph quality metrics (density, connectivity)
- End-to-end pipeline evaluation
- System health monitoring

**Key Methods:**
- `evaluate_retrieval()`: Assess search quality
- `evaluate_extraction()`: Evaluate extracted knowledge
- `evaluate_graph_quality()`: Measure graph metrics
- `evaluate_end_to_end()`: Full pipeline assessment
- `get_system_health()`: Overall system status

### 3. Monitoring Dashboard (`src/monitoring/dashboard.py`)

Web-based real-time monitoring interface that displays:
- System health status
- Component success rates
- Performance trends
- Detailed metrics tables

**Access:**
```bash
python scripts/run_monitoring.py --host localhost --port 8080
```

## Integration Points

### Phase 1: Hybrid Reranking
- Tracks input/output document counts
- Measures score improvements
- Records latency for reranking operations

### Phase 2: Graph RAG
- Monitors entity extraction rates
- Tracks relationship discovery
- Measures graph traversal performance

### Existing Systems
- **ExtractionEvaluator**: Provides detailed extraction metrics
- **QualityScorer**: Multi-dimensional quality assessment
- **AdaptiveQualityManager**: Continuous improvement tracking
- **TerminalMonitor**: Real-time extraction monitoring

## Metrics Schema

### Database Tables

```sql
-- Core metrics table
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    value FLOAT NOT NULL,
    component TEXT NOT NULL,
    operation TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata JSONB DEFAULT '{}'
);

-- Hourly aggregations
CREATE TABLE metrics_hourly (
    component TEXT,
    operation TEXT,
    metric_name TEXT,
    hour TIMESTAMP,
    count INTEGER,
    sum FLOAT,
    min FLOAT,
    max FLOAT,
    avg FLOAT,
    p50 FLOAT,
    p95 FLOAT,
    p99 FLOAT
);
```

## Key Metrics Tracked

### Retrieval Metrics
- `retrieval_count`: Number of documents retrieved
- `avg_retrieval_score`: Average similarity score
- `retrieval_precision`: Precision when ground truth available
- `retrieval_recall`: Recall when ground truth available

### Extraction Metrics
- `extraction_quality`: Overall quality score
- `entities_per_chunk`: Entity extraction rate
- `relationships_per_chunk`: Relationship discovery rate
- `topic_count`, `fact_count`, `question_count`

### Performance Metrics
- `latency_ms`: Operation execution time
- `success_rate`: Success/failure tracking
- `error_count`: Error frequency and types
- `token_usage`: LLM token consumption

### Graph Metrics
- `entity_count`: Total entities in graph
- `relationship_count`: Total relationships
- `graph_density`: Connectivity measure
- `path_length`: Multi-hop traversal depths

## Quality Thresholds

- Minimum quality score: 0.7
- Minimum confidence: 0.6
- Success rate alert: < 0.9
- Latency warning: > 1000ms

## Usage Examples

### Recording Custom Metrics
```python
from src.evaluation.metrics_collector import get_metrics_collector

collector = get_metrics_collector()
await collector.record_metric(
    'custom_metric', 42.5, 'my_component', 'my_operation',
    metadata={'user_id': 123}
)
```

### Evaluating a Complete Pipeline
```python
from src.evaluation.evaluation_orchestrator import get_evaluation_orchestrator

orchestrator = get_evaluation_orchestrator()
evaluation = await orchestrator.evaluate_end_to_end(
    query="What is flow state?",
    answer="Flow state is...",
    retrieval_results=results
)
```

### Checking System Health
```python
health = await orchestrator.get_system_health()
print(f"System status: {health['status']}")
for component, status in health['components'].items():
    print(f"{component}: {status['status']} ({status['success_rate']:.1%})")
```

## Benefits

1. **Unified Monitoring**: Single source of truth for all metrics
2. **Real-time Visibility**: Live dashboards and alerts
3. **Historical Analysis**: Time-series data for trend detection
4. **Automated Quality Gates**: Re-extraction triggers for low quality
5. **Performance Optimization**: Identify bottlenecks and regressions
6. **Continuous Improvement**: Adaptive quality management integration

## Future Enhancements

1. **Advanced Analytics**: ML-based anomaly detection
2. **Custom Dashboards**: Role-specific views
3. **Alert Management**: Configurable thresholds and notifications
4. **A/B Testing**: Built-in experiment tracking
5. **Cost Tracking**: Token usage and API cost monitoring