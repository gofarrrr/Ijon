# 12-Factor Refactoring Plan

## Immediate Code Changes Needed

### 1. Make Extractors Stateless (Stage 1-2)
**Current Issue**: Extractors maintain state (client, config)
**Fix**: Pass everything as parameters

```python
# Before
class BaselineExtractor:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key)  # State!
    
    async def extract(self, content: str) -> ExtractedKnowledge:
        # Uses self.client

# After  
class BaselineExtractor:
    @staticmethod
    async def extract(content: str, client: AsyncOpenAI, 
                     model: str = "gpt-3.5-turbo") -> ExtractedKnowledge:
        # Pure function - no state
```

### 2. Simplify Document Profiler (Stage 2)
**Current Issue**: Does too much in one class
**Fix**: Split into micro-agents

```python
# New micro-agents
class StructureAnalyzer:
    """Analyze document structure only"""
    @staticmethod
    def analyze(chunks: List[PDFChunk]) -> StructureScore:
        pass

class TypeClassifier:
    """Classify document type only"""
    @staticmethod
    def classify(chunks: List[PDFChunk]) -> Tuple[DocumentType, float]:
        pass

class QualityAssessor:
    """Assess OCR/text quality only"""
    @staticmethod
    def assess(chunks: List[PDFChunk]) -> QualityScore:
        pass
```

### 3. Explicit Strategy Selection (Stage 2)
**Current Issue**: Strategy selection hidden in factory
**Fix**: Make it a pure function with clear rules

```python
def select_extraction_strategy(doc_type: DocumentType, 
                             structure_score: float,
                             requirements: Dict = None) -> str:
    """Pure function - deterministic strategy selection"""
    
    # Explicit rules
    if structure_score < 0.5:
        return "baseline_validated"
    
    if doc_type == DocumentType.ACADEMIC:
        return "academic"
    elif doc_type == DocumentType.TECHNICAL:
        return "technical"
    elif doc_type in [DocumentType.BUSINESS, DocumentType.NARRATIVE]:
        return "narrative"
    else:
        return "baseline"
```

### 4. Simplify Feedback Loop (Stage 3)
**Current Issue**: Complex re-extraction with state
**Fix**: Simple improvement functions

```python
class FeedbackImprover:
    @staticmethod
    def improve_facts(facts: List[Fact], issues: List[str], 
                     source: str) -> List[Fact]:
        """Fix specific fact issues"""
        pass
    
    @staticmethod
    def improve_coherence(extraction: ExtractedKnowledge, 
                         issues: List[str]) -> ExtractedKnowledge:
        """Fix coherence issues only"""
        pass
```

### 5. New Model Router (Stage 4 Replacement)

```python
class ModelRouter:
    """Simple, deterministic model selection"""
    
    ROUTING_RULES = {
        # (doc_type, length_range, requirements) -> model
        (DocumentType.ACADEMIC, (0, 5000), {"speed": True}): "gpt-3.5-turbo",
        (DocumentType.ACADEMIC, (0, 5000), {"quality": True}): "gpt-4",
        (DocumentType.ACADEMIC, (5000, 50000), None): "claude-3-opus",
        (DocumentType.TECHNICAL, None, {"code": True}): "gpt-4",
        # ... more rules
    }
    
    @staticmethod
    def select_model(profile: DocumentProfile, 
                    requirements: Dict = None) -> ModelConfig:
        """Deterministic model selection based on rules"""
        # Match rules, return best model
        pass
```

### 6. Focused Enhancers (Stage 5 Replacement)

```python
# Each enhancer is a single-purpose micro-agent
class CitationEnhancer:
    @staticmethod
    async def enhance(extraction: ExtractedKnowledge, 
                     source_text: str) -> ExtractedKnowledge:
        """Add/validate citations only"""
        # Find citations in source
        # Match to facts
        # Add citation evidence
        return extraction

class QuestionEnhancer:
    @staticmethod  
    async def enhance(extraction: ExtractedKnowledge,
                     bloom_levels: List[str] = None) -> ExtractedKnowledge:
        """Generate questions only"""
        # Create questions for each topic/fact
        # Assign Bloom's levels
        return extraction

class RelationshipEnhancer:
    @staticmethod
    async def enhance(extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """Find relationships between entities"""
        # Analyze entities
        # Create relationship graph
        return extraction
```

### 7. Human Validation (Stage 6 Replacement)

```python
class HumanValidation:
    @staticmethod
    async def request_review(extraction: ExtractedKnowledge,
                           quality_report: Dict,
                           channel: str = "slack") -> ValidationResult:
        """Send extraction for human review"""
        # Format for human review
        # Send via MCP/Slack/Email
        # Wait for structured response
        pass
    
    @staticmethod
    def apply_feedback(extraction: ExtractedKnowledge,
                      feedback: ValidationResult) -> ExtractedKnowledge:
        """Apply human corrections"""
        # Pure function - apply changes
        pass
```

## Refactoring Priority

1. **Week 1**: Make extractors stateless
   - Remove self.client from all extractors
   - Pass clients as parameters
   - Update all tests

2. **Week 2**: Implement model router
   - Replace Stage 4 collaborative extraction
   - Add deterministic routing rules
   - Test with different document types

3. **Week 3**: Create focused enhancers
   - Replace Stage 5 complex question generator
   - Build citation, question, relationship enhancers
   - Each < 100 lines of code

4. **Week 4**: Add human validation
   - Replace Stage 6 memory system
   - Implement MCP/Slack integration
   - Create feedback UI

## Benefits After Refactoring

1. **Testability**: Each function can be tested in isolation
2. **Debuggability**: Clear data flow, no hidden state
3. **Composability**: Mix and match components easily
4. **Reliability**: Fewer moving parts = fewer failures
5. **Performance**: Parallel execution of stateless functions

## Code Principles to Follow

1. **No hidden state**: Everything passed explicitly
2. **Single responsibility**: Each function/class does ONE thing
3. **Pure functions**: Same input = same output
4. **Explicit over implicit**: No magic, just code
5. **Fail fast**: Return errors, don't retry blindly

## Migration Strategy

1. Keep existing code working
2. Build new components alongside
3. Gradually swap old for new
4. Maintain backwards compatibility
5. Full cutover once stable