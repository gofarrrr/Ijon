# Quality Knowledge Extraction System - V2 (12-Factor Design)

## Overview

Redesigned following 12-factor agent principles:
- **Small, focused micro-agents** instead of complex orchestrators
- **Stateless, pure functions** for all extraction components  
- **Own our prompts and control flow** explicitly
- **Context engineering first** - quality through careful token management
- **Human-in-the-loop** for quality validation

## Core Principles

1. **LLMs are pure functions**: Every extraction is stateless (content + context → knowledge)
2. **Micro-agents over monoliths**: Each component does ONE thing well
3. **Explicit control flow**: No magic, just code we understand
4. **Context is everything**: Quality comes from what tokens we put in

## Revised Architecture

### Stage 1: Foundation Components ✅ (Completed)
Keep as-is - already follows principles:
- Simple baseline extractor
- Clear data models
- Pure function design

### Stage 2: Document-Aware Strategies ✅ (Completed) 
Minor refactor needed:
- Split DocumentProfiler into micro-agents
- Make strategies fully stateless
- Simplify strategy selection

### Stage 3: Quality Validation ✅ (Completed)
Good as-is:
- QualityScorer is already a focused component
- Clear scoring dimensions
- Actionable feedback

### Stage 4: Smart Model Router (Redesigned) 🔄
**Old**: Complex collaborative extraction with consensus
**New**: Simple, deterministic model router

```python
class ModelRouter:
    """Routes to best model based on document type and requirements"""
    
    def select_model(self, profile: DocumentProfile, requirements: Dict) -> str:
        # Deterministic rules, not complex consensus
        if profile.document_type == "academic" and requirements.get("citations"):
            return "gpt-4"  # Better at structured data
        elif profile.length > 10000:
            return "claude-3"  # Better context window
        elif requirements.get("speed"):
            return "gpt-3.5-turbo"
        else:
            return "gpt-4"  # Default
```

Instead of multiple models trying to reach consensus:
- Use document profile to pick BEST model for the job
- Fall back to different models only on failure
- Keep extraction attempts independent

### Stage 5: Focused Enhancers (Redesigned) 🔄
**Old**: Intelligent question generator
**New**: Suite of small, focused enhancers

```python
# Each enhancer is a micro-agent with single responsibility
class CitationExtractor:
    """Extract and validate citations"""
    def enhance(self, extraction: ExtractedKnowledge, source: str) -> ExtractedKnowledge:
        # Pure function: adds citation data
        pass

class QuestionGenerator:
    """Generate Bloom's taxonomy questions"""
    def enhance(self, extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        # Pure function: adds questions
        pass

class SummaryImprover:
    """Improve summary quality"""
    def enhance(self, extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        # Pure function: better summary
        pass
```

### Stage 6: Human Validation Loop (Redesigned) 🔄
**Old**: Memory-enhanced extraction with pattern learning
**New**: Human-in-the-loop validation

```python
class HumanValidator:
    """Allow humans to validate and improve extractions"""
    
    async def request_validation(self, extraction: ExtractedKnowledge) -> ValidationResult:
        # Send to human reviewer via Slack/Email/MCP
        # Get structured feedback
        # Return approval or required changes
        pass
```

Key features:
- Quality scores trigger human review
- Humans provide structured feedback
- System learns from corrections (simple pattern matching, not complex ML)

### Stage 7: MCP Integration ✅ (Keep)
Already aligns with "meet users where they are"

### Stage 8: Removed ❌
No orchestrator-worker system needed - violates small/focused principle

## New Pipeline Flow

```python
class ExtractionPipeline:
    """Owns control flow - explicit and understandable"""
    
    def __init__(self):
        # Small, focused components
        self.profiler = DocumentProfiler()
        self.router = ModelRouter()
        self.extractor = BaselineExtractor()
        self.scorer = QualityScorer()
        self.enhancers = [
            CitationExtractor(),
            QuestionGenerator(),
            SummaryImprover()
        ]
        self.validator = HumanValidator()
    
    async def extract(self, pdf_path: str, requirements: Dict = None) -> ExtractedKnowledge:
        # Step 1: Profile document (micro-agent)
        profile = await self.profiler.profile(pdf_path)
        
        # Step 2: Route to best model (deterministic)
        model = self.router.select_model(profile, requirements)
        
        # Step 3: Extract with chosen model
        extraction = await self.extractor.extract(pdf_path, model=model)
        
        # Step 4: Score quality
        quality = self.scorer.score(extraction)
        
        # Step 5: Enhance if needed (small loops, 2-3 iterations max)
        if quality.needs_improvement:
            for enhancer in self.enhancers:
                if self._should_enhance(enhancer, quality):
                    extraction = enhancer.enhance(extraction)
        
        # Step 6: Human validation for low confidence
        if quality.overall_score < 0.7:
            validation = await self.validator.request_validation(extraction)
            if validation.changes_required:
                extraction = self._apply_human_feedback(extraction, validation)
        
        return extraction
```

## Implementation Priority

1. **Refactor Stage 4**: Simple model router (2 days)
2. **Implement Stage 5**: Focused enhancers (3 days)
3. **Add Stage 6**: Human validation (2 days)
4. **Integration**: Connect everything (1 day)

## Benefits of 12-Factor Approach

1. **Easier to debug**: Each component has single responsibility
2. **Better reliability**: Stateless functions, explicit flow
3. **Flexible composition**: Mix and match enhancers as needed
4. **Human collaboration**: Humans improve quality where AI struggles
5. **Faster iteration**: Change prompts without touching framework

## Removed Complexity

- ❌ Multi-model consensus (Stage 4)
- ❌ Complex memory systems (Stage 6)
- ❌ Orchestrator-worker architecture (Stage 8)
- ❌ Over-abstracted tool use
- ❌ Hidden control flow

## Added Simplicity

- ✅ Deterministic model selection
- ✅ Pure function enhancers
- ✅ Human-in-the-loop validation
- ✅ Explicit state management
- ✅ Clear, ownable code

## Success Metrics

- **Quality**: 85%+ accuracy on extracted facts
- **Reliability**: 95%+ successful extractions
- **Speed**: <30s for typical document
- **Human effort**: <5% need human validation
- **Maintainability**: Any developer can understand and modify

## Next Steps

1. Update existing code to be more stateless
2. Implement simple model router
3. Create focused enhancer suite
4. Add human validation hooks
5. Test end-to-end pipeline