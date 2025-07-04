"""
Knowledge extraction service for agent education.

Provides extraction capabilities as simple functions that other agents can use.
Following 12-factor principles: stateless, focused endpoints.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from extraction.models import ExtractedKnowledge, Topic, Fact, Question, Relationship
from extraction.v2.extractors import select_model_for_document
from extraction.v2.enhancers import CitationEnhancer, RelationshipEnhancer
from extraction.quality.scorer import QualityScorer
from extraction.v2.state import StateStore, ExtractionState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Global instances
state_store = StateStore()
quality_scorer = QualityScorer()


async def extract_knowledge(content: str, 
                          doc_type: str = "unknown", 
                          apply_citation_enhancer: bool = True,
                          apply_relationship_enhancer: bool = True) -> Dict[str, Any]:
    """
    Extract structured knowledge from text content.
    
    This is the main service endpoint for knowledge extraction.
    Other agents can call this to extract topics, facts, questions, and relationships.
    
    Args:
        content: Text content to extract from
        doc_type: Document type (academic, technical, narrative, unknown)
        apply_citation_enhancer: Whether to enhance with citations
        apply_relationship_enhancer: Whether to find relationships
        
    Returns:
        Dict containing:
        - status: "success" or "error"
        - extraction: ExtractedKnowledge object (as dict)
        - quality_score: Overall quality score (0-1)
        - quality_report: Detailed quality analysis
        - model_used: Which model was selected
        - enhancers_applied: Which enhancers were used
        
    Example:
        >>> result = await extract_knowledge(
        ...     "Machine learning is transforming healthcare...",
        ...     doc_type="technical"
        ... )
        >>> print(f"Quality: {result['quality_score']}")
        >>> print(f"Topics: {len(result['extraction']['topics'])}")
    """
    try:
        # Select appropriate model based on document characteristics
        config = select_model_for_document(
            doc_type=doc_type,
            doc_length=len(content),
            quality_required=True
        )
        
        logger.info(f"Extracting with {config['model']} for {doc_type} document")
        
        # Create demo extraction (in production, use real LLM client)
        extraction = _create_demo_extraction(content)
        
        # Apply enhancers to improve extraction
        if apply_citation_enhancer:
            extraction = CitationEnhancer.enhance(extraction, content)
            logger.info("Applied citation enhancement")
        
        if apply_relationship_enhancer:
            extraction = RelationshipEnhancer.enhance(extraction)
            logger.info("Applied relationship enhancement")
        
        # Calculate quality score
        quality_report = quality_scorer.score_extraction(extraction, content)
        
        # Fix quality report format
        quality_report["dimensions"] = quality_report.get("dimensions", quality_report.get("dimension_scores", {}))
        
        return {
            "status": "success",
            "extraction": extraction.dict() if hasattr(extraction, 'dict') else _extraction_to_dict(extraction),
            "quality_score": quality_report["overall_score"],
            "quality_report": quality_report,
            "model_used": config["model"],
            "enhancers_applied": {
                "citation": apply_citation_enhancer,
                "relationship": apply_relationship_enhancer
            }
        }
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "extraction": None
        }


async def analyze_extraction_quality(extraction_data: Dict[str, Any], 
                                   source_content: str) -> Dict[str, Any]:
    """
    Analyze the quality of an extraction.
    
    Use this to evaluate extractions and get improvement recommendations.
    
    Args:
        extraction_data: Extraction data (topics, facts, etc.)
        source_content: Original source text
        
    Returns:
        Quality report with:
        - overall_score: 0-1 quality score
        - dimensions: Scores for consistency, grounding, coherence, completeness
        - weaknesses: List of quality issues
        - recommendations: Specific improvements to make
        
    Example:
        >>> quality = await analyze_extraction_quality(my_extraction, original_text)
        >>> if quality['overall_score'] < 0.7:
        ...     for rec in quality['recommendations']:
        ...         print(f"Improve: {rec}")
    """
    try:
        # Convert dict to ExtractedKnowledge object
        extraction = _dict_to_extraction(extraction_data)
        
        # Score the extraction
        quality_report = quality_scorer.score_extraction(extraction, source_content)
        
        return {
            "status": "success",
            "overall_score": quality_report["overall_score"],
            "dimensions": quality_report.get("dimensions", quality_report.get("dimension_scores", {})),
            "weaknesses": quality_report.get("weaknesses", []),
            "strengths": quality_report.get("strengths", []),
            "recommendations": _generate_recommendations(quality_report)
        }
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def get_extraction_examples(doc_type: str = "all") -> List[Dict[str, Any]]:
    """
    Get examples of high-quality extractions.
    
    Learn from these examples to understand what good extractions look like.
    
    Args:
        doc_type: Filter by document type (academic, technical, narrative, all)
        
    Returns:
        List of example extractions with explanations
    """
    examples = []
    
    # Academic example - shows proper citation usage
    if doc_type in ["academic", "all"]:
        examples.append({
            "doc_type": "academic",
            "example": {
                "topics": [
                    {
                        "name": "Neural Networks",
                        "description": "Computational models inspired by biological neural networks",
                        "confidence": 0.95
                    },
                    {
                        "name": "Deep Learning",
                        "description": "Multi-layered neural network architectures",
                        "confidence": 0.92
                    }
                ],
                "facts": [
                    {
                        "claim": "Convolutional neural networks achieve 99% accuracy on MNIST dataset",
                        "evidence": "LeCun et al. (1998) demonstrated CNN performance on handwritten digits",
                        "confidence": 0.9
                    },
                    {
                        "claim": "Transformer models revolutionized NLP with attention mechanisms",
                        "evidence": "Vaswani et al. (2017) 'Attention is All You Need' paper",
                        "confidence": 0.95
                    }
                ],
                "questions": [
                    {
                        "question_text": "How do attention mechanisms improve model performance?",
                        "cognitive_level": "analyze",
                        "expected_answer": "By allowing models to focus on relevant parts of input",
                        "difficulty": 4,
                        "confidence": 0.85
                    }
                ],
                "relationships": [
                    {
                        "source_entity": "Neural Networks",
                        "target_entity": "Deep Learning",
                        "relationship_type": "foundation_of",
                        "description": "Neural Networks form the foundation of Deep Learning",
                        "confidence": 0.95
                    }
                ],
                "summary": "Neural networks and deep learning have transformed AI through architectures like CNNs and Transformers.",
                "overall_confidence": 0.92,
                "quality_score": 0.94
            },
            "explanation": "Academic extractions should include proper citations, high confidence scores, and demonstrate deep understanding of relationships between concepts."
        })
    
    # Technical example - shows implementation focus
    if doc_type in ["technical", "all"]:
        examples.append({
            "doc_type": "technical",
            "example": {
                "topics": [
                    {
                        "name": "Docker Containers",
                        "description": "Lightweight virtualization technology for application deployment",
                        "confidence": 0.9
                    },
                    {
                        "name": "Container Orchestration",
                        "description": "Management of containerized applications at scale",
                        "confidence": 0.88
                    }
                ],
                "facts": [
                    {
                        "claim": "Docker containers share the host OS kernel for efficiency",
                        "evidence": "Technical documentation confirms kernel sharing architecture",
                        "confidence": 0.95
                    },
                    {
                        "claim": "Kubernetes can manage thousands of containers across clusters",
                        "evidence": "Production deployments demonstrate scalability",
                        "confidence": 0.9
                    },
                    {
                        "claim": "Container images use layered filesystem for space efficiency",
                        "evidence": "Docker image layer caching reduces storage by 60-80%",
                        "confidence": 0.87
                    }
                ],
                "questions": [
                    {
                        "question_text": "What are the security implications of kernel sharing?",
                        "cognitive_level": "evaluate",
                        "expected_answer": "Potential for container escape, need for security policies",
                        "difficulty": 4,
                        "confidence": 0.8
                    }
                ],
                "quality_score": 0.88
            },
            "explanation": "Technical extractions focus on implementation details, performance metrics, and practical considerations."
        })
    
    return examples


def get_extraction_insights(topic: str) -> Dict[str, Any]:
    """
    Get insights about extracting specific topics.
    
    Use these insights to improve extraction quality for specific domains.
    
    Args:
        topic: Topic to get extraction insights for
        
    Returns:
        Best practices and tips for extracting the topic
    """
    insights = {
        "machine learning": {
            "tips": [
                "Look for algorithm names and performance metrics",
                "Extract both theoretical concepts and practical applications",
                "Identify datasets mentioned (MNIST, ImageNet, COCO, etc.)",
                "Capture accuracy/performance claims with specific numbers",
                "Note computational requirements (GPU hours, memory)",
                "Extract comparison with baseline methods"
            ],
            "common_patterns": [
                "Model architecture descriptions (layers, parameters)",
                "Training methodology (optimizer, learning rate, epochs)",
                "Evaluation metrics (accuracy, F1, AUC, perplexity)",
                "Ablation study results",
                "Dataset statistics and preprocessing steps"
            ],
            "quality_indicators": [
                "Specific performance numbers with context",
                "Citations to original papers",
                "Clear problem-solution mapping",
                "Limitations and failure cases mentioned"
            ],
            "recommended_enhancers": ["citation", "relationship"]
        },
        "healthcare": {
            "tips": [
                "Extract clinical outcomes and statistics",
                "Identify medical terminology with definitions",
                "Look for patient demographics and study sizes",
                "Capture regulatory compliance mentions (FDA, CE)",
                "Note efficacy vs effectiveness distinctions",
                "Extract cost-benefit analyses"
            ],
            "common_patterns": [
                "Clinical trial phases (I, II, III, IV)",
                "Treatment efficacy rates with confidence intervals",
                "Adverse events and contraindications",
                "Patient inclusion/exclusion criteria",
                "Statistical significance (p-values, hazard ratios)"
            ],
            "quality_indicators": [
                "Sample sizes and study duration",
                "Control group comparisons",
                "Regulatory approval status",
                "Peer review and journal impact factors"
            ],
            "recommended_enhancers": ["citation", "question"]
        },
        "default": {
            "tips": [
                "Identify main concepts and provide clear definitions",
                "Extract quantitative claims with supporting evidence",
                "Look for relationships between concepts",
                "Generate questions for deeper understanding",
                "Note assumptions and prerequisites",
                "Capture both benefits and limitations"
            ],
            "common_patterns": [
                "Problem-solution pairs",
                "Cause-effect relationships",
                "Comparisons and contrasts",
                "Historical progression or timeline",
                "Key stakeholders and their roles"
            ],
            "quality_indicators": [
                "Balanced perspective (pros and cons)",
                "Concrete examples provided",
                "Clear logical flow",
                "Actionable insights"
            ],
            "recommended_enhancers": ["citation", "relationship", "question"]
        }
    }
    
    topic_lower = topic.lower()
    
    # Check for specific domain matches
    for key in ["machine learning", "healthcare"]:
        if key in topic_lower or (key == "healthcare" and any(term in topic_lower for term in ["medical", "clinical", "patient"])):
            result = insights[key].copy()
            result["topic"] = topic
            result["matched_domain"] = key
            result["recommended_doc_type"] = "academic" if key == "machine learning" else "technical"
            return result
    
    # Default insights
    result = insights["default"].copy()
    result["topic"] = topic
    result["matched_domain"] = "general"
    result["recommended_doc_type"] = "unknown"
    return result


async def check_extraction_status(extraction_id: str) -> Dict[str, Any]:
    """
    Check the status of an extraction job.
    
    Args:
        extraction_id: ID of the extraction to check
        
    Returns:
        Status information including state and progress
    """
    try:
        state = await state_store.load(extraction_id)
        
        if not state:
            return {
                "status": "not_found",
                "error": f"Extraction {extraction_id} not found"
            }
        
        return {
            "status": "success",
            "extraction_id": extraction_id,
            "state": state.status,
            "current_step": state.current_step,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "quality_score": state.quality_report.get("overall_score") if state.quality_report else None,
            "has_extraction": state.extraction is not None,
            "needs_validation": state.metadata.get("needs_validation", False),
            "validation_url": f"http://localhost:8001/validator/{extraction_id}" if state.status == "pending_validation" else None
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def get_extraction_guidelines() -> str:
    """
    Get comprehensive guidelines for high-quality extraction.
    
    Returns:
        Markdown-formatted guidelines
    """
    return """# Knowledge Extraction Guidelines

## 1. Document Analysis Phase
- **Identify document type**: Academic papers need citations, technical docs need implementation details
- **Assess quality indicators**: Author credibility, publication venue, recency
- **Determine scope**: What aspects are most important to extract?
- **Note document structure**: Abstracts, methodology sections, conclusions

## 2. Topic Extraction Best Practices
- **Optimal count**: Extract 3-7 main topics (too many dilutes focus)
- **Hierarchy**: Identify primary vs. secondary topics
- **Descriptions**: 10-20 words, capture essence without details
- **Confidence scoring**: 
  - 0.9+: Explicitly stated main theme
  - 0.7-0.9: Important supporting topic
  - 0.5-0.7: Mentioned but not central

## 3. Fact Extraction Standards
- **Verifiability**: Can this claim be fact-checked?
- **Specificity**: Include numbers, dates, names when available
- **Evidence**: Direct quotes > paraphrasing > inference
- **Objectivity**: Avoid interpretation, stick to stated claims
- **Confidence alignment**: Higher confidence for directly stated facts

## 4. Quality Dimensions

### Consistency (Internal Coherence)
- Facts should support identified topics
- No contradictory claims
- Uniform level of detail
- Logical flow between elements

### Grounding (Evidence Support)
- Every major claim needs evidence
- Citations for academic content
- Source references for technical content
- "According to..." attributions

### Coherence (Logical Structure)
- Clear relationships between topics
- Facts build upon each other
- Questions address key gaps
- Summary synthesizes main points

### Completeness (Coverage)
- All major points extracted
- Key stakeholders identified
- Important caveats noted
- Both benefits and limitations

## 5. Enhancement Strategies

### Citation Enhancement
- **When**: Academic papers, research content, controversial claims
- **How**: Match claims to references, extract author-year format
- **Quality boost**: Adds credibility and traceability

### Relationship Enhancement
- **When**: Complex topics, system descriptions, cause-effect content
- **How**: Identify entity co-occurrences, determine relationship types
- **Quality boost**: Reveals hidden connections

### Question Enhancement  
- **When**: Educational content, technical tutorials, exploratory topics
- **How**: Generate Bloom's taxonomy questions (understand → apply → analyze)
- **Quality boost**: Aids comprehension and retention

### Summary Enhancement
- **When**: Long documents, multiple topics, complex narratives
- **How**: Synthesize key points, maintain objectivity
- **Quality boost**: Provides quick overview

## 6. Common Pitfalls to Avoid
- ❌ Over-extraction: Too many minor details
- ❌ Under-extraction: Missing crucial information  
- ❌ Interpretation: Adding opinions not in source
- ❌ Decontextualization: Removing important context
- ❌ Citation errors: Wrong attributions
- ❌ Confidence inflation: Overestimating certainty

## 7. Domain-Specific Tips

### Academic/Research
- Focus on methodology and results
- Preserve statistical details
- Note limitations mentioned by authors
- Extract future work suggestions

### Technical/Documentation
- Emphasize implementation steps
- Capture version numbers, dependencies
- Note prerequisites and requirements
- Extract troubleshooting information

### Business/Strategic
- Identify stakeholders and their interests
- Extract metrics and KPIs
- Note market conditions and assumptions
- Capture strategic rationale

## 8. Quality Checkpoints
Before finalizing extraction:
1. ✓ Do facts support the identified topics?
2. ✓ Is evidence provided for major claims?
3. ✓ Are relationships logical and meaningful?
4. ✓ Would a reader understand the main points?
5. ✓ Is confidence scoring consistent and justified?
"""


# Helper functions

def _create_demo_extraction(content: str) -> ExtractedKnowledge:
    """Create a demo extraction based on content keywords."""
    extraction = ExtractedKnowledge(
        topics=[],
        facts=[],
        questions=[],
        relationships=[],
        overall_confidence=0.0
    )
    
    content_lower = content.lower()
    
    # Add relevant topics based on content
    if "machine learning" in content_lower or "ai" in content_lower:
        extraction.topics.append(
            Topic(
                name="Machine Learning",
                description="AI techniques that learn patterns from data",
                confidence=0.85
            )
        )
        extraction.facts.append(
            Fact(
                claim="Machine learning models improve through experience",
                confidence=0.9
            )
        )
    
    if "healthcare" in content_lower or "medical" in content_lower:
        extraction.topics.append(
            Topic(
                name="Healthcare Applications",
                description="Medical and health-related use cases",
                confidence=0.8
            )
        )
        
    if "deep learning" in content_lower:
        extraction.facts.append(
            Fact(
                claim="Deep learning models can process complex data like images",
                confidence=0.85
            )
        )
    
    # Calculate overall confidence
    if extraction.topics:
        topic_conf = sum(t.confidence for t in extraction.topics) / len(extraction.topics)
        fact_conf = sum(f.confidence for f in extraction.facts) / len(extraction.facts) if extraction.facts else 0.5
        extraction.overall_confidence = (topic_conf + fact_conf) / 2
    
    return extraction


def _extraction_to_dict(extraction: ExtractedKnowledge) -> Dict[str, Any]:
    """Convert ExtractedKnowledge to dict."""
    return {
        "topics": [{"name": t.name, "description": t.description, "confidence": t.confidence} for t in extraction.topics],
        "facts": [{"claim": f.claim, "evidence": f.evidence, "confidence": f.confidence} for f in extraction.facts],
        "questions": [{"question_text": q.question_text, "cognitive_level": q.cognitive_level, 
                      "expected_answer": q.expected_answer, "difficulty": q.difficulty,
                      "confidence": q.confidence} for q in extraction.questions],
        "relationships": [{"source_entity": r.source_entity, "target_entity": r.target_entity,
                         "relationship_type": r.relationship_type, "description": r.description,
                         "confidence": r.confidence} for r in extraction.relationships],
        "summary": extraction.summary,
        "overall_confidence": extraction.overall_confidence
    }


def _dict_to_extraction(data: Dict[str, Any]) -> ExtractedKnowledge:
    """Convert dict to ExtractedKnowledge."""
    return ExtractedKnowledge(
        topics=[Topic(**t) for t in data.get("topics", [])],
        facts=[Fact(**f) for f in data.get("facts", [])],
        questions=[Question(**q) for q in data.get("questions", [])],
        relationships=[Relationship(**r) for r in data.get("relationships", [])],
        summary=data.get("summary"),
        overall_confidence=data.get("overall_confidence", 0.5)
    )


def _generate_recommendations(quality_report: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on quality report."""
    recommendations = []
    
    for weakness in quality_report.get("weaknesses", []):
        dim = weakness["dimension"]
        severity = weakness.get("severity", "medium")
        
        if dim == "grounding":
            if severity == "high":
                recommendations.append("Add citations or evidence for all major claims")
            else:
                recommendations.append("Strengthen evidence for unsupported facts")
                
        elif dim == "completeness":
            recommendations.append("Extract additional facts or topics for comprehensive coverage")
            recommendations.append("Consider what key information might be missing")
            
        elif dim == "coherence":
            recommendations.append("Identify and extract relationships between topics")
            recommendations.append("Ensure logical flow from topics to facts")
            
        elif dim == "consistency":
            recommendations.append("Align facts with identified topics")
            recommendations.append("Remove or revise contradictory claims")
    
    # Add general recommendations based on score
    score = quality_report.get("overall_score", 0)
    if score < 0.5:
        recommendations.append("Consider re-extraction with different parameters")
    elif score < 0.7:
        recommendations.append("Apply enhancers to improve specific weak areas")
    
    return recommendations