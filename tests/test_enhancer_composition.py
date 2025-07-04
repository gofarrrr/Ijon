"""
Test enhancer composition and performance.

This validates that enhancers can be composed effectively
and measures their performance impact.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any

from extraction.models import ExtractedKnowledge, Topic, Fact, Question, Relationship
from extraction.v2.enhancers import (
    CitationEnhancer,
    QuestionEnhancer,
    RelationshipEnhancer,
    SummaryEnhancer
)
from openai import AsyncOpenAI


class TestEnhancerComposition:
    """Test suite for enhancer composition."""
    
    @pytest.fixture
    def sample_extraction(self):
        """Create sample extraction for testing."""
        return ExtractedKnowledge(
            topics=[
                Topic(name="Machine Learning", description="AI subfield focusing on algorithms that improve through experience", confidence=0.9),
                Topic(name="Healthcare", description="Medical and health services", confidence=0.85),
                Topic(name="Predictive Analytics", description="Using data to predict future outcomes", confidence=0.8)
            ],
            facts=[
                Fact(claim="Machine learning algorithms can predict disease onset with 85% accuracy", confidence=0.8),
                Fact(claim="Early detection through AI reduces treatment costs by 40%", confidence=0.75),
                Fact(claim="Deep learning models analyze medical images faster than human radiologists", confidence=0.85),
                Fact(claim="AI-driven drug discovery reduces development time from 10 years to 5 years", confidence=0.7)
            ],
            overall_confidence=0.8
        )
    
    @pytest.fixture
    def sample_source_text(self):
        """Sample source text for citation enhancement."""
        return """
        Recent studies have shown that machine learning algorithms can predict disease onset 
        with 85% accuracy when analyzing patient data from electronic health records. 
        
        The implementation of AI in healthcare has led to significant cost reductions. 
        Early detection through AI reduces treatment costs by 40% according to a 2023 
        report from the Healthcare Analytics Institute.
        
        In radiology, deep learning models analyze medical images faster than human 
        radiologists, processing scans in seconds rather than minutes.
        
        Pharmaceutical companies are leveraging AI-driven drug discovery to reduce 
        development time from the traditional 10 years to just 5 years.
        """
    
    def test_citation_enhancer(self, sample_extraction, sample_source_text):
        """Test citation enhancement."""
        # Apply citation enhancement
        enhanced = CitationEnhancer.enhance(sample_extraction, sample_source_text)
        
        # Verify citations were added
        facts_with_evidence = [f for f in enhanced.facts if f.evidence]
        assert len(facts_with_evidence) >= 2
        
        # Check that evidence contains source text
        for fact in facts_with_evidence:
            assert fact.evidence is not None
            assert len(fact.evidence) > 10
    
    @pytest.mark.asyncio
    async def test_question_enhancer(self, sample_extraction):
        """Test question enhancement."""
        # Skip if no API key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OpenAI API key available")
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Apply question enhancement
        original_questions = len(sample_extraction.questions)
        enhanced = await QuestionEnhancer.enhance(sample_extraction, client)
        
        # Verify questions were added
        assert len(enhanced.questions) > original_questions
        
        # Check question quality
        for question in enhanced.questions:
            assert len(question.question_text) > 10
            assert question.confidence > 0.5
            assert question.cognitive_level is not None
    
    def test_relationship_enhancer(self, sample_extraction):
        """Test relationship enhancement."""
        # Apply relationship enhancement
        enhanced = RelationshipEnhancer.enhance(sample_extraction)
        
        # Should find relationships between entities
        assert len(enhanced.relationships) > 0
        
        # Check relationship quality
        for rel in enhanced.relationships:
            assert rel.source_entity is not None
            assert rel.target_entity is not None
            assert rel.relationship_type is not None
            assert rel.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_summary_enhancer(self, sample_extraction):
        """Test summary enhancement."""
        # Skip if no API key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OpenAI API key available")
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Apply summary enhancement
        assert sample_extraction.summary is None
        enhanced = await SummaryEnhancer.enhance(sample_extraction, client)
        
        # Verify summary was added
        assert enhanced.summary is not None
        assert len(enhanced.summary) > 50
        assert len(enhanced.summary) < 500
    
    @pytest.mark.asyncio
    async def test_sequential_composition(self, sample_extraction, sample_source_text):
        """Test applying enhancers sequentially."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OpenAI API key available")
        
        client = AsyncOpenAI(api_key=api_key)
        
        # Track changes at each step
        steps = []
        
        # Step 1: Citation enhancement
        step1 = CitationEnhancer.enhance(sample_extraction, sample_source_text)
        steps.append({
            "enhancer": "citation",
            "facts_with_evidence": len([f for f in step1.facts if f.evidence])
        })
        
        # Step 2: Relationship enhancement
        step2 = RelationshipEnhancer.enhance(step1)
        steps.append({
            "enhancer": "relationship",
            "relationships": len(step2.relationships)
        })
        
        # Step 3: Question enhancement
        step3 = await QuestionEnhancer.enhance(step2, client)
        steps.append({
            "enhancer": "question",
            "questions": len(step3.questions)
        })
        
        # Step 4: Summary enhancement
        final = await SummaryEnhancer.enhance(step3, client)
        steps.append({
            "enhancer": "summary",
            "has_summary": final.summary is not None
        })
        
        # Verify all enhancements were applied
        assert len([f for f in final.facts if f.evidence]) > 0
        assert len(final.relationships) > 0
        assert len(final.questions) > len(sample_extraction.questions)
        assert final.summary is not None
        
        # Each step should preserve previous enhancements
        for i, step in enumerate(steps):
            print(f"Step {i+1}: {step}")
    
    def test_enhancer_idempotency(self, sample_extraction, sample_source_text):
        """Test that enhancers are idempotent (applying twice has same effect)."""
        # Apply citation enhancement twice
        enhanced_once = CitationEnhancer.enhance(sample_extraction, sample_source_text)
        enhanced_twice = CitationEnhancer.enhance(enhanced_once, sample_source_text)
        
        # Should not duplicate evidence
        evidence_once = [f.evidence for f in enhanced_once.facts if f.evidence]
        evidence_twice = [f.evidence for f in enhanced_twice.facts if f.evidence]
        
        assert evidence_once == evidence_twice
    
    def test_enhancer_performance(self, sample_extraction, sample_source_text):
        """Test performance of enhancers."""
        # Measure citation enhancement time
        start = time.time()
        for _ in range(100):
            CitationEnhancer.enhance(sample_extraction, sample_source_text)
        citation_time = (time.time() - start) / 100
        
        # Measure relationship enhancement time
        start = time.time()
        for _ in range(100):
            RelationshipEnhancer.enhance(sample_extraction)
        relationship_time = (time.time() - start) / 100
        
        # Should be fast (< 10ms per enhancement)
        assert citation_time < 0.01
        assert relationship_time < 0.01
        
        print(f"Citation enhancement: {citation_time*1000:.2f}ms")
        print(f"Relationship enhancement: {relationship_time*1000:.2f}ms")


class TestEnhancerConfiguration:
    """Test different enhancer configurations."""
    
    @pytest.mark.asyncio
    async def test_selective_enhancement(self, sample_extraction, sample_source_text):
        """Test selective application of enhancers."""
        from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
        
        # Skip if no API key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OpenAI API key available")
        
        # Test different configurations
        configs = [
            (["citation"], 1),
            (["relationship"], 1),
            (["citation", "relationship"], 2),
            (["citation", "question", "summary"], 3),
            ([], 0)
        ]
        
        for enhancers, expected_count in configs:
            config = ExtractionConfig(
                api_key=api_key,
                max_enhancer_loops=1,
                quality_threshold=0.0,  # Always enhance
                enable_human_validation=False,
                enhancers_enabled=enhancers
            )
            
            # Create pipeline
            pipeline = ExtractionPipeline(config)
            
            # Mock PDF processor
            class MockProcessor:
                async def process_pdf(self, path):
                    return [type('Chunk', (), {'content': sample_source_text})]
            
            pipeline.pdf_processor = MockProcessor()
            
            # Run extraction
            result = await pipeline.extract("test.pdf")
            
            # Verify only selected enhancers were applied
            applied = result["metadata"]["enhancements_applied"]
            assert len(applied) == expected_count
            for enhancer in enhancers:
                assert enhancer in applied


class TestEnhancerQualityImpact:
    """Test impact of enhancers on quality scores."""
    
    @pytest.mark.asyncio
    async def test_quality_improvement(self, sample_extraction, sample_source_text):
        """Test that enhancers improve quality scores."""
        from extraction.quality.scorer import QualityScorer
        
        # Skip if no API key
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OpenAI API key available")
        
        client = AsyncOpenAI(api_key=api_key)
        scorer = QualityScorer()
        
        # Score original
        original_score = scorer.score_extraction(sample_extraction, sample_source_text)
        
        # Apply enhancements
        enhanced = CitationEnhancer.enhance(sample_extraction, sample_source_text)
        enhanced = RelationshipEnhancer.enhance(enhanced)
        enhanced = await QuestionEnhancer.enhance(enhanced, client)
        enhanced = await SummaryEnhancer.enhance(enhanced, client)
        
        # Score enhanced
        enhanced_score = scorer.score_extraction(enhanced, sample_source_text)
        
        # Quality should improve
        assert enhanced_score["overall_score"] >= original_score["overall_score"]
        
        # Specific dimensions should improve
        assert enhanced_score["dimension_scores"]["grounding"] > original_score["dimension_scores"]["grounding"]
        assert enhanced_score["dimension_scores"]["completeness"] >= original_score["dimension_scores"]["completeness"]
        
        print(f"Original score: {original_score['overall_score']:.2%}")
        print(f"Enhanced score: {enhanced_score['overall_score']:.2%}")
        print(f"Improvement: {enhanced_score['overall_score'] - original_score['overall_score']:.2%}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])