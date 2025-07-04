# Migration Guide: Upgrading to Cognitive RAG

This guide helps existing Ijon users upgrade to the new cognitive RAG system.

## Overview

The cognitive enhancements are **fully backward compatible**. Your existing code will continue to work, and you can gradually adopt new features.

## Migration Steps

### Step 1: Update Dependencies

The cognitive system requires a few additional dependencies:

```bash
# Core dependencies (already in requirements.txt)
pip install pydantic-ai>=0.0.9
pip install nltk>=3.8.0
pip install neo4j>=5.0.0

# Optional for full functionality
pip install sentence-transformers>=2.2.0  # For enhanced embeddings
```

### Step 2: Minimal Code Changes

#### Option A: Keep Existing Code (No Changes Required)

Your current code continues to work:

```python
# Existing code - still works!
pipeline = create_rag_pipeline(
    vector_store_type="pinecone",
    graph_store_type="neo4j"
)

result = await pipeline.query(query, client)
```

#### Option B: Enable HyDE Enhancement Only

Add HyDE for better search with one parameter:

```python
# Just add enable_hyde=True
pipeline = create_rag_pipeline(
    vector_store_type="pinecone",
    graph_store_type="neo4j",
    enable_hyde=True  # ← Add this line
)

# Everything else stays the same
result = await pipeline.query(query, client)
```

#### Option C: Full Cognitive Enhancement

Wrap your pipeline with cognitive capabilities:

```python
from src.rag.cognitive_pipeline import create_cognitive_rag_pipeline

# Your existing pipeline
base_pipeline = create_rag_pipeline(
    vector_store_type="pinecone",
    graph_store_type="neo4j",
    enable_hyde=True
)

# Wrap with cognitive enhancement
cognitive_pipeline = create_cognitive_rag_pipeline(
    rag_pipeline=base_pipeline,
    cognitive_threshold=0.6,  # Optional tuning
    enable_hybrid_mode=True   # Optional
)

# Use the same query interface
result = await cognitive_pipeline.query(query, client)
```

### Step 3: Understanding Routing Behavior

The cognitive system automatically routes queries:

| Query Type | Example | Route | Response Time |
|------------|---------|-------|---------------|
| Simple lookup | "What is Python?" | Fast RAG | 50-200ms |
| Definition | "Define machine learning" | Fast RAG | 50-200ms |
| Analysis | "Analyze market trends..." | Cognitive | 2-5 minutes |
| Creation | "Create a blog post..." | Cognitive | 2-5 minutes |
| Problem solving | "Fix this bug..." | Cognitive | 2-5 minutes |

### Step 4: Gradual Feature Adoption

#### Phase 1: Start with HyDE
```python
# Just enable HyDE for better search
enable_hyde=True
```

#### Phase 2: Add Cognitive Routing
```python
# Wrap with cognitive pipeline
cognitive_pipeline = create_cognitive_rag_pipeline(base_pipeline)
```

#### Phase 3: Enable Quality Features
```python
# Add quality enhancements
orchestrator = CognitiveOrchestrator(
    enable_self_correction=True,
    enable_reasoning_validation=True
)
```

## Configuration Migration

### Old Configuration
```python
# config.py
RAG_CONFIG = {
    "vector_store": "pinecone",
    "chunk_size": 1000,
    "overlap": 200
}
```

### New Configuration (Backward Compatible)
```python
# config.py
RAG_CONFIG = {
    "vector_store": "pinecone",
    "chunk_size": 1000,
    "overlap": 200,
    # New optional settings
    "enable_hyde": True,
    "cognitive_threshold": 0.6,
    "quality_threshold": 0.8
}
```

## API Compatibility

### Query Method - Fully Compatible
```python
# Old interface still works
result = await pipeline.query(
    query="Your query",
    client=client,
    max_results=5
)

# New optional parameters
result = await pipeline.query(
    query="Your query",
    client=client,
    max_results=5,
    quality_threshold=0.8,    # Optional
    force_cognitive=False,    # Optional
    enable_verification=True  # Optional
)
```

### Result Object - Extended but Compatible
```python
# Old attributes still work
print(result.summary)
print(result.documents)
print(result.confidence)

# New attributes available
if hasattr(result, 'quality_score'):
    print(f"Quality: {result.quality_score}")
if hasattr(result, 'agents_used'):
    print(f"Agents: {result.agents_used}")
```

## Performance Considerations

### Query Routing Impact

The system automatically optimizes performance:

1. **Simple queries**: No performance impact (fast RAG path)
2. **Complex queries**: Improved quality worth the extra time
3. **Hybrid mode**: Best of both worlds when enabled

### Resource Usage

- **Memory**: ~200MB additional for cognitive components
- **CPU**: Minimal impact for simple queries
- **API calls**: Only for complex queries requiring LLM routing

### Tuning for Your Use Case

```python
# For mostly simple queries (FAQ, definitions)
cognitive_threshold=0.8  # Higher = less cognitive usage

# For complex analysis tasks
cognitive_threshold=0.4  # Lower = more cognitive usage

# For quality-critical applications
quality_threshold=0.85   # Higher quality requirements
```

## Rollback Plan

If needed, you can instantly rollback:

```python
# Option 1: Disable cognitive features
cognitive_pipeline = create_cognitive_rag_pipeline(
    rag_pipeline=base_pipeline,
    cognitive_threshold=1.0  # Effectively disables cognitive routing
)

# Option 2: Use base pipeline directly
result = await base_pipeline.query(query, client)

# Option 3: Remove cognitive wrapper entirely
# Just use your original create_rag_pipeline() code
```

## Common Migration Scenarios

### Scenario 1: Knowledge Base Application
```python
# Add HyDE for better search, keep threshold high
cognitive_pipeline = create_cognitive_rag_pipeline(
    rag_pipeline=base_pipeline,
    cognitive_threshold=0.8,  # Mostly use fast RAG
    enable_hybrid_mode=False  # Don't need hybrid
)
```

### Scenario 2: Analysis Platform
```python
# Lower threshold for more cognitive usage
cognitive_pipeline = create_cognitive_rag_pipeline(
    rag_pipeline=base_pipeline,
    cognitive_threshold=0.5,  # Balance fast and cognitive
    enable_hybrid_mode=True   # Combine results
)
```

### Scenario 3: Content Generation
```python
# Force cognitive for quality
result = await cognitive_pipeline.query(
    query=content_request,
    client=client,
    force_cognitive=True,     # Always use agents
    quality_threshold=0.85    # High quality bar
)
```

## Monitoring Changes

### Check Routing Decisions
```python
# See how queries are routed
analysis = await cognitive_pipeline.analyze_query_complexity(
    query="Your query",
    client=client
)
print(f"Will use cognitive: {analysis['should_use_cognitive']}")
```

### Track Performance
```python
# Monitor quality improvements
if hasattr(result, 'metadata'):
    if 'self_correction' in result.metadata:
        print(f"Quality improved by: {result.metadata['self_correction']['improvement']}")
```

## FAQ

**Q: Will this break my existing code?**
A: No, the enhancements are fully backward compatible.

**Q: Can I disable cognitive features?**
A: Yes, set `cognitive_threshold=1.0` or use the base pipeline directly.

**Q: How much slower is cognitive processing?**
A: Simple queries are unaffected. Complex queries take 2-10 minutes but produce much higher quality results.

**Q: Do I need to retrain or reindex?**
A: No, your existing vector stores and indices work as-is.

**Q: Can I use only some features?**
A: Yes, enable features incrementally (HyDE → Routing → Quality).

## Support

- Check [Getting Started Guide](getting_started_cognitive_rag.md) for detailed usage
- Review [test examples](../examples/cognitive_rag_demo.py) for patterns
- See [architecture docs](agentic_rag_enhancements_final_summary.md) for deep dive

The cognitive enhancements are designed to be adopted gradually. Start with HyDE for better search, then add cognitive routing as you see benefits for your use case.