"""
MCP (Model Context Protocol) server for extraction knowledge.

Provides extraction capabilities and knowledge as a service to other agents.
Following 12-factor principles: stateless, focused endpoints.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass

from mcp.server import Server
from mcp.server.models import InitializationOptions
import mcp.types as types

from extraction.models import ExtractedKnowledge
from extraction.v2.extractors import select_model_for_document, BaselineExtractor
from extraction.v2.enhancers import CitationEnhancer, RelationshipEnhancer
from extraction.quality.scorer import QualityScorer
from extraction.v2.state import StateStore, ExtractionState
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Create MCP server instance
server = Server("ijon-extraction")
state_store = StateStore()
quality_scorer = QualityScorer()


@dataclass
class ExtractionRequest:
    """Request for extraction service."""
    content: str
    doc_type: str = "unknown"
    apply_enhancers: List[str] = None
    quality_threshold: float = 0.7


# Define available tools
@server.tool()
async def extract_knowledge(content: str, doc_type: str = "unknown", 
                          apply_citation_enhancer: bool = True,
                          apply_relationship_enhancer: bool = True) -> Dict[str, Any]:
    """
    Extract structured knowledge from text content.
    
    Args:
        content: Text content to extract from
        doc_type: Document type (academic, technical, narrative, unknown)
        apply_citation_enhancer: Whether to enhance with citations
        apply_relationship_enhancer: Whether to find relationships
        
    Returns:
        Extracted knowledge with topics, facts, questions, and relationships
    """
    try:
        # Select appropriate model
        config = select_model_for_document(
            doc_type=doc_type,
            doc_length=len(content),
            quality_required=True
        )
        
        logger.info(f"Extracting with {config['model']} for {doc_type} document")
        
        # Mock extraction for MCP demo (in production, use real client)
        extraction = ExtractedKnowledge(
            topics=[],
            facts=[],
            questions=[],
            relationships=[],
            overall_confidence=0.0
        )
        
        # For demo purposes, create sample extraction based on content
        if "machine learning" in content.lower():
            from extraction.models import Topic, Fact
            extraction.topics.append(
                Topic(
                    name="Machine Learning",
                    description="AI techniques for pattern recognition",
                    confidence=0.85
                )
            )
            extraction.facts.append(
                Fact(
                    claim="Machine learning can identify patterns in data",
                    confidence=0.9
                )
            )
        
        # Apply enhancers
        if apply_citation_enhancer:
            extraction = CitationEnhancer.enhance(extraction, content)
        
        if apply_relationship_enhancer:
            extraction = RelationshipEnhancer.enhance(extraction)
        
        # Calculate quality
        quality_report = quality_scorer.score_extraction(extraction, content)
        
        return {
            "status": "success",
            "extraction": extraction.dict() if hasattr(extraction, 'dict') else extraction.__dict__,
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


@server.tool()
async def analyze_extraction_quality(extraction_data: Dict[str, Any], 
                                   source_content: str) -> Dict[str, Any]:
    """
    Analyze the quality of an extraction.
    
    Args:
        extraction_data: Extraction data (topics, facts, etc.)
        source_content: Original source text
        
    Returns:
        Quality report with scores and weaknesses
    """
    try:
        # Convert dict back to ExtractedKnowledge if needed
        if isinstance(extraction_data, dict):
            from extraction.models import Topic, Fact, Question, Relationship
            
            extraction = ExtractedKnowledge(
                topics=[Topic(**t) for t in extraction_data.get("topics", [])],
                facts=[Fact(**f) for f in extraction_data.get("facts", [])],
                questions=[Question(**q) for q in extraction_data.get("questions", [])],
                relationships=[Relationship(**r) for r in extraction_data.get("relationships", [])],
                summary=extraction_data.get("summary"),
                overall_confidence=extraction_data.get("overall_confidence", 0.5)
            )
        else:
            extraction = extraction_data
        
        # Score the extraction
        quality_report = quality_scorer.score_extraction(extraction, source_content)
        
        return {
            "status": "success",
            "overall_score": quality_report["overall_score"],
            "dimensions": quality_report["dimensions"],
            "weaknesses": quality_report["weaknesses"],
            "strengths": quality_report.get("strengths", []),
            "recommendations": _generate_recommendations(quality_report)
        }
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@server.tool()
async def get_extraction_examples(doc_type: str = "all") -> List[Dict[str, Any]]:
    """
    Get examples of high-quality extractions.
    
    Args:
        doc_type: Filter by document type (academic, technical, narrative, all)
        
    Returns:
        List of example extractions with explanations
    """
    examples = []
    
    # Academic example
    if doc_type in ["academic", "all"]:
        examples.append({
            "doc_type": "academic",
            "example": {
                "topics": [
                    {
                        "name": "Neural Networks",
                        "description": "Computational models inspired by biological neural networks",
                        "confidence": 0.95
                    }
                ],
                "facts": [
                    {
                        "claim": "Convolutional neural networks achieve 99% accuracy on MNIST",
                        "evidence": "LeCun et al. (1998) demonstrated CNN performance on handwritten digits",
                        "confidence": 0.9
                    }
                ],
                "quality_score": 0.92
            },
            "explanation": "Academic extractions should include citations and high confidence"
        })
    
    # Technical example
    if doc_type in ["technical", "all"]:
        examples.append({
            "doc_type": "technical",
            "example": {
                "topics": [
                    {
                        "name": "Docker Containers",
                        "description": "Lightweight virtualization technology for application deployment",
                        "confidence": 0.9
                    }
                ],
                "facts": [
                    {
                        "claim": "Docker containers share the host OS kernel",
                        "evidence": "Technical documentation confirms kernel sharing architecture",
                        "confidence": 0.95
                    }
                ],
                "quality_score": 0.88
            },
            "explanation": "Technical extractions focus on precise implementation details"
        })
    
    return examples


@server.tool()
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
            "metadata": state.metadata
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@server.tool()
async def get_extraction_insights(topic: str) -> Dict[str, Any]:
    """
    Get insights about extracting specific topics.
    
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
                "Identify datasets mentioned (MNIST, ImageNet, etc.)",
                "Capture accuracy/performance claims with evidence"
            ],
            "common_patterns": [
                "Model architecture descriptions",
                "Training methodology",
                "Evaluation metrics",
                "Comparison with baselines"
            ],
            "enhancers": ["citation", "relationship"]
        },
        "healthcare": {
            "tips": [
                "Extract clinical outcomes and statistics",
                "Identify medical terminology and procedures",
                "Look for patient demographics and study sizes",
                "Capture regulatory compliance mentions"
            ],
            "common_patterns": [
                "Clinical trial phases",
                "Treatment efficacy rates",
                "Side effects and contraindications",
                "Cost-benefit analyses"
            ],
            "enhancers": ["citation", "question"]
        },
        "default": {
            "tips": [
                "Identify main concepts and definitions",
                "Extract quantitative claims with evidence",
                "Look for relationships between concepts",
                "Generate questions for deeper understanding"
            ],
            "common_patterns": [
                "Problem-solution pairs",
                "Cause-effect relationships",
                "Comparisons and contrasts",
                "Historical progression"
            ],
            "enhancers": ["citation", "relationship", "question"]
        }
    }
    
    topic_lower = topic.lower()
    for key in insights:
        if key in topic_lower:
            return {
                "topic": topic,
                "insights": insights[key],
                "recommended_doc_type": "academic" if key == "machine learning" else "technical"
            }
    
    return {
        "topic": topic,
        "insights": insights["default"],
        "recommended_doc_type": "unknown"
    }


# Define available resources
@server.resource()
async def extraction_guidelines() -> Resource:
    """
    Guidelines for high-quality knowledge extraction.
    """
    return Resource(
        uri="extraction://guidelines",
        name="Extraction Guidelines",
        description="Best practices for knowledge extraction",
        mimeType="text/markdown",
        contents="""# Knowledge Extraction Guidelines

## 1. Document Analysis
- Identify document type (academic, technical, narrative)
- Assess document structure and quality
- Determine appropriate extraction strategy

## 2. Topic Extraction
- Extract 3-7 main topics
- Provide clear, concise descriptions
- Assign confidence scores based on prominence

## 3. Fact Extraction
- Extract verifiable claims
- Include supporting evidence when available
- Maintain objectivity and accuracy

## 4. Quality Criteria
- **Consistency**: Facts align with topics
- **Grounding**: Claims have evidence
- **Coherence**: Logical flow between elements
- **Completeness**: Covers main points

## 5. Enhancement Strategy
- Use Citation Enhancer for academic texts
- Apply Relationship Enhancer for complex topics
- Generate questions for educational content
"""
    )


@server.resource()
async def extraction_metrics() -> Resource:
    """
    Current extraction system metrics.
    """
    # Get current stats
    active_states = await state_store.list_active()
    
    metrics = {
        "active_extractions": len(active_states),
        "average_quality_score": 0.75,  # Would calculate from recent extractions
        "model_usage": {
            "gpt-3.5-turbo": 60,
            "gpt-4": 30,
            "claude-3-opus": 10
        },
        "enhancer_usage": {
            "citation": 80,
            "relationship": 70,
            "question": 50,
            "summary": 40
        }
    }
    
    return Resource(
        uri="extraction://metrics",
        name="Extraction Metrics",
        description="Current system performance metrics",
        mimeType="application/json",
        contents=json.dumps(metrics, indent=2)
    )


def _generate_recommendations(quality_report: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on quality report."""
    recommendations = []
    
    for weakness in quality_report.get("weaknesses", []):
        if weakness["dimension"] == "grounding":
            recommendations.append("Add citations or evidence to support claims")
        elif weakness["dimension"] == "completeness":
            recommendations.append("Extract additional facts or topics for better coverage")
        elif weakness["dimension"] == "coherence":
            recommendations.append("Identify relationships between topics and facts")
        elif weakness["dimension"] == "consistency":
            recommendations.append("Ensure facts align with identified topics")
    
    return recommendations


# Server lifecycle
@server.on_initialize()
async def on_initialize():
    """Initialize the MCP server."""
    logger.info("Ijon Extraction MCP server initialized")
    logger.info(f"Available tools: {len(server._tools)}")
    logger.info(f"Available resources: {len(server._resources)}")


if __name__ == "__main__":
    import sys
    from mcp.server.stdio import stdio_server
    
    logger.info("Starting Ijon Extraction MCP server...")
    
    # Run the server
    try:
        asyncio.run(stdio_server(server))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)