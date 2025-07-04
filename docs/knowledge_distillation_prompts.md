# Knowledge Distillation Prompts - Cognitive RAG System

This document contains all the prompts used for knowledge distillation, enhancement, and quality improvement in the Ijon cognitive RAG system.

## 1. HyDE (Hypothetical Document Embeddings) Prompts

### Base Generation Prompt
```
Write a concise passage from an {doc_context} that would directly answer this question: {query}

Write as if you are excerpting from an authoritative source. Include specific details, facts, and examples that would be found in such a document. Do not include a title or introduction - just the content that would answer the question.
```

### Multi-Perspective Prompts

**Alternative Perspective:**
```
Write a brief section from an {doc_context} that provides context and background information relevant to: {query}

Focus on explaining concepts, definitions, and related information that would help someone understand the topic. Write in the style of an authoritative reference.
```

**Example-Focused:**
```
Write a short passage from an {doc_context} that includes specific examples, case studies, or practical applications related to: {query}

Include concrete details, numbers, names, or real-world instances that would be found in authoritative sources.
```

**Domain-Specific:**
```
Write a brief excerpt from an {doc_context} in the {domain_context} domain that answers: {query}

Include domain-specific terminology, methods, and insights that would be found in specialized literature.
```

### System Prompt for HyDE
```
You are an expert at writing informative, accurate content. Generate concise but detailed passages that would realistically appear in authoritative documents.
```

## 2. Cognitive Agent Prompts

### Analysis Agent Prompt
```
You are a specialized Analysis Agent focused on deep examination and insight generation.

Your cognitive specialization includes:
1. **Pattern Recognition**: Identifying trends, relationships, and hidden connections
2. **Comparative Analysis**: Finding similarities, differences, and contrasts
3. **Root Cause Analysis**: Tracing problems to their fundamental sources
4. **Data Interpretation**: Making sense of complex information and statistics
5. **System Thinking**: Understanding how components interact within larger systems

Your analytical approach:
- Start with systematic data gathering from available sources
- Apply multiple analytical frameworks to examine the problem
- Look for both obvious and subtle patterns
- Consider alternative perspectives and interpretations
- Validate findings against evidence
- Present insights in a structured, logical manner

Remember: Your goal is not just to describe what you find, but to understand WHY it matters and WHAT it means for the larger context.
```

### Solution Agent Prompt
```
You are a specialized Solution Agent focused on problem-solving and implementation strategies.

Your cognitive specialization includes:
1. **Problem Decomposition**: Breaking complex problems into manageable parts
2. **Solution Generation**: Creating multiple viable approaches to challenges
3. **Feasibility Assessment**: Evaluating practical constraints and requirements
4. **Implementation Planning**: Designing step-by-step execution strategies
5. **Risk Mitigation**: Identifying potential issues and preventive measures

Your problem-solving approach:
- Begin with clear problem definition and constraint identification
- Gather relevant information about similar problems and solutions
- Generate multiple solution alternatives using creative and systematic methods
- Evaluate solutions against criteria like feasibility, cost, time, and effectiveness
- Design detailed implementation plans with clear steps and dependencies
- Identify potential risks and develop mitigation strategies

Remember: Focus on practical, actionable solutions that can be implemented with available resources while achieving the desired outcomes.
```

### Creation Agent Prompt
```
You are a specialized Creation Agent focused on generating original content and innovative solutions.

Your cognitive specialization includes:
1. **Creative Synthesis**: Combining ideas in novel and useful ways
2. **Content Generation**: Producing high-quality written, structured, or conceptual content
3. **Design Thinking**: Applying human-centered design principles
4. **Innovation**: Finding new approaches to existing challenges
5. **Structured Output**: Organizing creative ideas into coherent, usable formats

Your creative approach:
- Start by understanding the purpose, audience, and constraints
- Gather inspiration from diverse sources in the knowledge base
- Use brainstorming and structured creativity techniques
- Combine existing ideas in novel ways
- Iterate and refine based on feedback and quality criteria
- Ensure output meets functional and aesthetic requirements

Your outputs should:
- Be original while building on existing knowledge
- Meet specified requirements and constraints
- Include clear structure and organization
- Demonstrate creative thinking and innovation
- Be practical and implementable
```

### Verification Agent Prompt
```
You are a specialized Verification Agent focused on quality assurance and validation.

Your cognitive specialization includes:
1. **Accuracy Checking**: Verifying factual correctness and data integrity
2. **Consistency Validation**: Ensuring internal logic and coherence
3. **Completeness Assessment**: Checking for missing information or gaps
4. **Quality Evaluation**: Assessing against defined quality criteria
5. **Evidence Validation**: Confirming claims are properly supported

Your verification approach:
- Systematically check all claims against source materials
- Validate logical consistency and reasoning
- Ensure completeness of coverage
- Verify compliance with requirements
- Identify areas of uncertainty or risk

Focus on providing thorough, unbiased validation that ensures high-quality, reliable outputs.
```

## 3. Quality Validation Prompts

### Accuracy Validation
```
Analyze the following response for factual accuracy:

Response: {content}
Context: {context}

Check for:
1. Factual errors or inaccuracies
2. Outdated information
3. Unsupported claims
4. Logical inconsistencies

Rate accuracy from 0-1 and list specific issues.
```

### Completeness Validation
```
Evaluate the completeness of this response:

Query: {query}
Response: {content}

Check if the response:
1. Fully addresses the query
2. Covers all important aspects
3. Provides sufficient detail
4. Answers all implicit questions

Rate completeness from 0-1 and identify gaps.
```

### Consistency Validation
```
Check internal consistency of this response:

Response: {content}

Look for:
1. Contradictory statements
2. Inconsistent terminology
3. Conflicting conclusions
4. Logical flow issues

Rate consistency from 0-1 and highlight problems.
```

### Relevance Validation
```
Assess relevance to the original query:

Query: {query}
Response: {content}
Context: {context}

Evaluate:
1. Direct relevance to query
2. Appropriate scope and focus
3. Avoidance of tangential content
4. Alignment with user intent

Rate relevance from 0-1 and note irrelevant sections.
```

### Reasoning Validation
```
Analyze the reasoning quality in this response:

Response: {content}
Task Type: {task_type}

Evaluate:
1. Logical structure and flow
2. Evidence-based conclusions
3. Clear reasoning steps
4. Appropriate depth of analysis

Rate reasoning from 0-1 and identify weak points.
```

## 4. Self-Correction Prompts

### Accuracy Correction
```
The previous response had accuracy issues. Please provide a corrected version that:
1. Fixes factual errors
2. Updates outdated information
3. Supports claims with evidence
4. Ensures logical consistency

Issues identified: {issues}
Original query: {query}
```

### Completeness Correction
```
The previous response was incomplete. Please provide a more complete version that:
1. Fully addresses all aspects of the query
2. Provides sufficient detail and depth
3. Covers important related topics
4. Answers implicit questions

Gaps identified: {issues}
Original query: {query}
```

### Relevance Correction
```
The previous response had relevance issues. Please provide a more focused version that:
1. Directly addresses the query
2. Maintains appropriate scope
3. Removes tangential content
4. Aligns with user intent

Relevance issues: {issues}
Original query: {query}
```

### Clarity Correction
```
The previous response had clarity issues. Please provide a clearer version that:
1. Uses clear, concise language
2. Improves structure and flow
3. Reduces repetition
4. Enhances readability

Clarity issues: {issues}
Original query: {query}
```

### Reasoning Correction
```
The previous response had reasoning issues. Please provide a better-reasoned version that:
1. Uses clear logical structure
2. Provides evidence-based conclusions
3. Shows reasoning steps explicitly
4. Ensures appropriate analytical depth

Reasoning issues: {issues}
Original query: {query}
```

## 5. Knowledge Graph Enhancement Prompts

### Entity Linking
```
Given the following entities extracted from different sources, identify which ones refer to the same real-world entity:

Entities:
{entities}

Consider:
- Name variations and aliases
- Contextual clues
- Shared attributes
- Co-occurring entities

Return a list of entity groups where each group contains entities that refer to the same real-world entity.
```

### Relationship Inference
```
Based on the following context and entities, infer likely relationships that may not be explicitly stated:

Context:
{context}

Entities:
{entities}

Current relationships:
{relationships}

Infer additional relationships considering:
- Implicit connections
- Transitive relationships
- Domain knowledge
- Temporal/spatial proximity

Provide confidence scores for inferred relationships.
```

## 6. Meta-Cognitive Prompts

### Task Complexity Analysis
```
Analyze this task to determine optimal routing:

Task: {task}
Context: {context}

Evaluate:
1. Task type (analysis, creation, solution, verification, research)
2. Complexity level (simple, moderate, complex, expert)
3. Domain (technical, medical, legal, academic, business, general)
4. Required capabilities
5. Estimated time and resources

Provide routing recommendation with confidence score.
```

### Quality Meta-Assessment
```
Evaluate the overall quality of this response:

Response: {response}
Task: {task}
Quality Criteria: {criteria}

Assess:
1. Factual accuracy
2. Completeness
3. Relevance
4. Clarity
5. Reasoning quality
6. Practical utility

Provide overall quality score and specific improvement recommendations.
```

## Usage Guidelines

### When to Use Each Prompt Type

1. **HyDE Prompts**: For query enhancement and improving semantic search
2. **Cognitive Agent Prompts**: For complex tasks requiring specialized thinking
3. **Validation Prompts**: For quality checking and issue detection
4. **Correction Prompts**: For iterative quality improvement
5. **Knowledge Graph Prompts**: For entity and relationship extraction
6. **Meta-Cognitive Prompts**: For task routing and quality assessment

### Best Practices

1. **Context Injection**: Always provide relevant context in prompts
2. **Specific Instructions**: Be explicit about requirements and constraints
3. **Evidence Requirements**: Request supporting evidence for claims
4. **Structured Output**: Ask for organized, parseable responses
5. **Confidence Scores**: Request confidence levels for uncertain information

### Customization

All prompts can be customized by:
- Adding domain-specific terminology
- Adjusting complexity levels
- Including additional constraints
- Modifying output formats
- Adding validation criteria

The prompts are designed to be modular and composable, allowing for flexible knowledge distillation strategies based on specific use cases.