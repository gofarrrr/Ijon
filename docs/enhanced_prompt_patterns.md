# Enhanced Prompt Patterns Documentation

## Overview

This document describes the enhanced prompt patterns implemented in the Ijon cognitive agent system, based on analysis of industry best practices from leading AI companies. These enhancements significantly improve agent capabilities through systematic thinking, event awareness, and sophisticated output generation.

## Key Enhancements

### 1. Agent Loop Architecture

Agents now operate in iterative cycles with clear phases:

```
1. Analyze Events → 2. Plan Actions → 3. Execute Tools → 4. Observe Results → 5. Iterate or Complete
```

**Benefits:**
- Systematic approach to complex tasks
- Clear state transitions
- Built-in iteration for quality improvement
- Natural stopping conditions

**Implementation:**
- Base agents automatically support event streams
- Each phase is tracked and can be monitored
- Agents can reflect on their progress

### 2. Event Stream Processing

Event streams provide context awareness and state tracking:

```python
Event Types:
- MESSAGE: User inputs requiring processing
- ACTION: Agent tool usage (hidden from user)
- OBSERVATION: Results from tool execution  
- PLAN: Task decomposition and strategy
- KNOWLEDGE: Best practices and patterns
- THINKING: Agent reasoning blocks
- STATE: Agent state changes
```

**Usage Example:**
```python
from src.agents.event_stream import EventStream

# Create event stream
event_stream = EventStream()

# Add events during execution
event_stream.add_message("Analyze Q3 performance", source="user")
event_stream.add_plan({"steps": ["gather data", "analyze", "report"]})
event_stream.add_thinking("Patterns emerging: 15% revenue growth...")

# Get context for decisions
recent_events = event_stream.get_recent_events(10)
context = event_processor.analyze_context()
```

### 3. Deep Research Mode

Specialized agent for comprehensive, academic-style analysis:

**Requirements:**
- Minimum 10,000 words for comprehensive topics
- Major sections: 1,500-2,500 words each
- Academic prose throughout (no lists)
- Extensive citations [1][2] format
- Multi-phase research methodology

**Usage:**
```python
from src.agents.deep_research_agent import create_deep_research_agent

agent = create_deep_research_agent(min_word_count=10000, min_sources=20)
result = await agent.conduct_deep_research(
    topic="Impact of AI on Healthcare",
    target_depth="comprehensive"
)
```

### 4. Thinking Blocks

Agents now include reflection points after major actions:

```
## Thinking Process
After each action, reflect:
```
What did I accomplish?
What challenges remain?
Is my approach effective?
What should I do next?
```
```

**Appears in:**
- Validation prompts (accuracy, completeness, etc.)
- Correction prompts (identifying what to fix)
- Agent decision points
- Quality assessments

### 5. Academic Prose Requirements

All content generation emphasizes scholarly writing:

**Before (List-based):**
```
Key findings:
- Revenue up 15%
- Costs down 10%
- Efficiency improved
```

**After (Academic prose):**
```
The analysis reveals substantial improvements across key financial metrics during the 
evaluation period. Revenue demonstrated a fifteen percent increase, which coincides 
with a ten percent reduction in operational costs. These combined improvements 
indicate enhanced operational efficiency, suggesting that recent strategic 
initiatives have yielded measurable positive outcomes.
```

### 6. Modular Prompt Architecture

Prompts are now composed from reusable modules:

```python
from src.agents.prompt_modules import get_prompt_builder, ModuleType

builder = get_prompt_builder()
prompt = builder.build_prompt(
    base_prompt="You are an Analysis Agent",
    context={
        "task": "Analyze customer feedback",
        "domain": "E-commerce",
        "min_words": 1500,
        "style": "academic"
    },
    module_types=[
        ModuleType.PLANNER,
        ModuleType.KNOWLEDGE,
        ModuleType.THINKING,
        ModuleType.OUTPUT
    ]
)
```

**Available Module Types:**
- **PLANNER**: Task decomposition and planning
- **KNOWLEDGE**: Domain expertise and best practices
- **DATASOURCE**: Available data APIs
- **THINKING**: Reflection and reasoning blocks
- **CONSTRAINT**: Rules and limitations
- **OUTPUT**: Formatting requirements
- **CONTEXT**: Historical context
- **EXAMPLE**: Demonstrations

### 7. Tool Abstraction

Critical principle: Never expose technical implementation to users.

**Wrong:**
```
"I'll use the search_documents tool to find information..."
"Let me query the vector database..."
```

**Correct:**
```
"I'll search for relevant information..."
"Let me look for those details..."
```

## Integration Guide

### Basic Integration

1. **Enable Event Streams:**
```python
# In your agent initialization
context = AgentContext()
context.event_stream = EventStream()
```

2. **Use Enhanced Prompts:**
```python
from src.agents.prompts import COGNITIVE_ANALYSIS_AGENT_PROMPT

agent = Agent(
    system_prompt=COGNITIVE_ANALYSIS_AGENT_PROMPT,
    # ... other config
)
```

3. **Add Thinking to Validations:**
```python
# Validation automatically includes thinking blocks
validator = QualityValidator()
issues = await validator.validate_result(result, query, task_type)
```

### Advanced Integration

1. **Custom Prompt Modules:**
```python
from src.agents.prompt_modules import PromptModule, ModuleType, add_custom_module

custom_module = PromptModule(
    module_type=ModuleType.KNOWLEDGE,
    name="industry_standards",
    content="Apply these industry standards: {standards}",
    priority=8,
    conditions=[lambda ctx: "standards" in ctx]
)
add_custom_module(custom_module)
```

2. **Deep Research Tasks:**
```python
# For comprehensive analysis
agent = create_deep_research_agent()
result = await agent.research_with_outline(
    topic="Market Analysis",
    outline={
        "Market Overview": ["Size", "Growth", "Segments"],
        "Competitive Landscape": ["Key Players", "Market Share"],
        "Future Outlook": ["Trends", "Opportunities", "Risks"]
    }
)
```

3. **Event Stream Analysis:**
```python
# Monitor agent performance
processor = EventStreamProcessor(event_stream)
context = processor.analyze_context()
patterns = context["patterns"]
if any(p["type"] == "repeated_action" for p in patterns):
    # Agent might be stuck, intervene
    pass
```

## Best Practices

### 1. Prompt Design

- Start with agent loop awareness
- Include relevant modules based on task type
- Set clear depth/quality requirements
- Add thinking blocks at decision points

### 2. Event Management

- Track all significant actions as events
- Use event streams for multi-step tasks
- Analyze patterns to detect issues
- Maintain reasonable context windows

### 3. Output Quality

- Enforce minimum depth for complex tasks
- Use academic prose for professional output
- Include confidence assessments
- Provide clear reasoning traces

### 4. Performance

- Use module conditions to avoid unnecessary prompts
- Cache frequently used prompt combinations
- Monitor token usage with deep research
- Set appropriate iteration limits

## Migration Guide

### From Basic Prompts

1. Replace simple prompts with cognitive versions:
```python
# Old
prompt = "Analyze this data and provide insights"

# New
from src.agents.prompts import COGNITIVE_ANALYSIS_AGENT_PROMPT
# Use the full cognitive prompt with agent loop
```

2. Add event stream support:
```python
# Add to existing agents
if not context.event_stream:
    context.event_stream = EventStream()
```

3. Enable thinking in validations:
```python
# Validation prompts automatically include thinking blocks
# No code changes needed, just update to new version
```

### From List-Based Output

Transform list-based outputs to academic prose:

```python
# Use output modules
context["style"] = "academic"
# Agent will automatically use academic prose module
```

## Performance Considerations

### Token Usage

Enhanced prompts use more tokens but provide better results:
- Basic prompt: ~100 tokens
- Enhanced cognitive prompt: ~500-800 tokens
- Deep research prompt: ~1000+ tokens

### Recommendations:
- Use basic prompts for simple queries
- Use cognitive prompts for complex analysis
- Reserve deep research for comprehensive reports

### Processing Time

- Event streams add minimal overhead (<5%)
- Thinking blocks add one inference call
- Deep research is intentionally thorough (10-30 min)

## Troubleshooting

### Common Issues

1. **Agent produces lists despite academic prose requirement**
   - Ensure `style: "academic"` in context
   - Verify OUTPUT modules are included
   - Check prompt has latest enhancements

2. **Event stream grows too large**
   - Set appropriate max_events limit
   - Use context_window_size for recent events
   - Implement event archiving for long tasks

3. **Deep research doesn't meet word count**
   - Verify min_word_count is set
   - Ensure sufficient source material
   - Check iteration limits aren't too low

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger("src.agents").setLevel(logging.DEBUG)
```

## Examples

See `/tests/test_enhanced_prompts_demo.py` for working examples of all enhancements.

## Future Enhancements

Planned improvements:
1. Adaptive prompt selection based on task analysis
2. Cross-agent learning from successful patterns
3. Dynamic module generation from examples
4. Prompt effectiveness tracking and optimization

## References

This implementation draws inspiration from:
- Claude's thinking blocks and artifacts system
- Perplexity's deep research methodology  
- GPT-4's multi-perspective approaches
- Gemini's structured output patterns
- Cursor's agentic coding practices
- Manus's modular architecture

For questions or contributions, please refer to the project's contribution guidelines.