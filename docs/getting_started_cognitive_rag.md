# Getting Started with Cognitive RAG Enhancements

This guide will help you start using the new cognitive RAG enhancements in your Ijon system.

## Quick Start

### 1. Basic Setup

```python
from src.rag.cognitive_pipeline import create_cognitive_rag_pipeline
from src.rag.pipeline import create_rag_pipeline
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI(api_key="your-api-key")

# Create base RAG pipeline with HyDE
base_rag = create_rag_pipeline(
    vector_store_type="pinecone",  # or "supabase", "neon"
    enable_hyde=True,              # Enable HyDE enhancement
    graph_store_type="neo4j"       # Optional graph store
)

# Create cognitive-enhanced pipeline
cognitive_rag = create_cognitive_rag_pipeline(
    rag_pipeline=base_rag,
    cognitive_threshold=0.6,       # When to use cognitive agents
    enable_hybrid_mode=True        # Combine RAG + cognitive results
)
```

### 2. Simple Query (Fast RAG Path)

```python
# Simple queries automatically use fast RAG path
result = await cognitive_rag.query(
    query="What is machine learning?",
    client=client,
    max_results=5
)

print(f"Summary: {result.summary}")
print(f"Processing time: {result.processing_time}s")
```

### 3. Complex Task (Cognitive Agent Path)

```python
# Complex tasks automatically route to cognitive agents
result = await cognitive_rag.query(
    query="Create a comprehensive AI implementation strategy for a healthcare organization, including regulatory compliance, patient privacy, and ROI analysis",
    client=client,
    quality_threshold=0.8,  # Require high quality
    force_cognitive=True    # Optional: force cognitive path
)

# Cognitive results include additional metadata
print(f"Agents used: {result.agents_used}")
print(f"Quality score: {result.quality_score:.2f}")
print(f"Recommendations: {result.recommendations}")
```

## Common Use Cases

### 1. Research & Analysis Tasks

```python
# Automatic routing to AnalysisAgent
analysis_result = await cognitive_rag.query(
    query="Analyze the impact of large language models on software development practices",
    client=client,
    context="Focus on productivity, code quality, and developer experience"
)

# Access structured analysis
if hasattr(analysis_result.result, 'key_findings'):
    print("Key Findings:")
    for finding in analysis_result.result.key_findings:
        print(f"  • {finding}")
```

### 2. Problem Solving Tasks

```python
# Routes to SolutionAgent with self-correction
solution_result = await cognitive_rag.query(
    query="Our data pipeline is experiencing 30% failure rate. Diagnose the issue and propose solutions.",
    client=client,
    context="Python-based ETL pipeline processing 1TB daily"
)

# Solution includes problem analysis and recommendations
if hasattr(solution_result.result, 'recommended_solution'):
    print(f"Problem: {solution_result.result.problem_summary}")
    print(f"Solution: {solution_result.result.recommended_solution}")
    print(f"Implementation steps: {solution_result.result.implementation_steps}")
```

### 3. Content Creation Tasks

```python
# Routes to CreationAgent with quality validation
creation_result = await cognitive_rag.query(
    query="Create a technical blog post about zero-shot learning in NLP",
    client=client,
    quality_threshold=0.85  # High quality requirement
)

# Created content with metadata
if hasattr(creation_result.result, 'created_content'):
    print(f"Content type: {creation_result.result.content_type}")
    print(f"Quality score: {creation_result.quality_score:.2f}")
    print(f"Content:\n{creation_result.result.created_content}")
```

## Advanced Features

### 1. Analyze Query Complexity

```python
# Preview how a query will be routed
analysis = await cognitive_rag.analyze_query_complexity(
    query="Your query here",
    client=client
)

print(f"Task type: {analysis['task_type']}")
print(f"Complexity: {analysis['complexity']}")
print(f"Will use cognitive agents: {analysis['should_use_cognitive']}")
print(f"Recommended agent: {analysis['recommended_agent']}")
print(f"Estimated time: {analysis['estimated_time']}s")
```

### 2. Direct Orchestrator Usage

```python
from src.agents.cognitive_orchestrator import create_cognitive_orchestrator

# Create orchestrator with all quality features
orchestrator = create_cognitive_orchestrator(
    use_llm_router=True,  # LLM-enhanced routing
    tools=agent_tools     # Your configured tools
)

# Execute with full control
orchestration_result = await orchestrator.execute_task(
    task="Complex multi-step task",
    quality_threshold=0.8,
    enable_verification=True  # Add verification step
)

# Get quality insights
insights = await orchestrator.get_quality_insights()
for insight in insights:
    print(f"{insight.insight_type}: {insight.description}")
```

### 3. Performance Monitoring

```python
# Get agent performance summary
from src.agents.quality_feedback import create_adaptive_quality_manager

quality_manager = create_adaptive_quality_manager()
summary = quality_manager.get_performance_summary()

print(f"Overall health: {summary['overall_health']}")
print(f"Average quality: {summary['average_quality_score']:.2f}")
print(f"Improving agents: {summary['improving_agents']}")
print(f"Declining agents: {summary['declining_agents']}")
```

## Configuration Options

### Environment Variables

```bash
# Optional configuration
export COGNITIVE_MAX_ITERATIONS=3        # Max self-correction iterations
export COGNITIVE_QUALITY_THRESHOLD=0.8   # Default quality threshold
export COGNITIVE_ENABLE_HYBRID=true      # Enable hybrid mode by default
export COGNITIVE_MAX_CONCURRENT=3        # Max concurrent agents
```

### Custom Configuration

```python
# Fine-tune cognitive pipeline
cognitive_rag = create_cognitive_rag_pipeline(
    rag_pipeline=base_rag,
    cognitive_threshold=0.5,     # Lower = more cognitive usage
    enable_hybrid_mode=True,
)

# Configure orchestrator
orchestrator = CognitiveOrchestrator(
    router=custom_router,
    max_concurrent_agents=5,
    default_timeout=600.0,       # 10 minutes
    enable_self_correction=True,
    enable_reasoning_validation=True,
    enable_quality_feedback=True
)
```

## Debugging & Monitoring

### 1. Enable Detailed Logging

```python
import logging

# Enable debug logging
logging.getLogger("src.agents").setLevel(logging.DEBUG)
logging.getLogger("src.rag").setLevel(logging.DEBUG)

# Query with logging
result = await cognitive_rag.query(
    query="Your query",
    client=client,
    metadata={"debug": True}
)
```

### 2. Inspect Execution Trace

```python
# For cognitive results
if hasattr(result, 'orchestration_result'):
    trace = result.orchestration_result.execution_trace
    for step in trace:
        print(f"Agent: {step['agent']}")
        print(f"Duration: {step['duration']}s")
        print(f"Success: {step['success']}")
```

### 3. Quality Metrics

```python
# Check quality enhancements
if result.metadata.get('self_correction'):
    correction_data = result.metadata['self_correction']
    print(f"Correction applied: {correction_data['applied']}")
    print(f"Iterations: {correction_data['iterations']}")
    print(f"Quality improvement: {correction_data['improvement']:.2f}")

if result.metadata.get('reasoning_validation'):
    reasoning_data = result.metadata['reasoning_validation']
    print(f"Reasoning score: {reasoning_data['overall_score']:.2f}")
    print(f"Logic issues: {reasoning_data['issues_count']}")
```

## Best Practices

### 1. Query Formulation
- Be specific about the task type (analyze, create, solve, verify)
- Include relevant context for better routing
- Specify quality requirements when needed

### 2. Performance Optimization
- Use `force_cognitive=False` (default) to allow automatic routing
- Set appropriate timeouts for long-running tasks
- Monitor agent performance trends

### 3. Quality Assurance
- Set quality thresholds based on task criticality
- Review recommendations from quality feedback
- Use verification for critical outputs

### 4. Error Handling

```python
try:
    result = await cognitive_rag.query(query, client)
    
    if not result.success:
        print(f"Task failed: {result.metadata.get('error')}")
        print(f"Recommendations: {result.recommendations}")
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback to simple RAG if needed
    fallback_result = await base_rag.query(query, client)
```

## Troubleshooting

### Common Issues

1. **All queries going to cognitive agents**
   - Check cognitive_threshold setting (higher = less cognitive)
   - Verify query complexity analysis

2. **Slow performance**
   - Check if simple queries are being routed correctly
   - Adjust max_concurrent_agents for resource constraints
   - Disable self-correction for non-critical tasks

3. **Quality issues**
   - Enable self-correction and reasoning validation
   - Check performance metrics for declining agents
   - Review quality feedback recommendations

### Getting Help

- Check logs in `data/performance/` for performance metrics
- Review execution traces for debugging
- Use `analyze_query_complexity()` to understand routing decisions

## Next Steps

1. **Start Simple**: Test with basic queries to verify routing
2. **Experiment**: Try different task types to see agent specialization
3. **Monitor**: Track quality metrics and performance trends
4. **Optimize**: Adjust thresholds based on your use cases
5. **Extend**: Add custom agents or tools as needed

The cognitive RAG system is designed to be transparent and debuggable. Start with the defaults and adjust based on your specific needs and performance requirements.