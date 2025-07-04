"""
Test model routing logic for different document types.

This validates that the deterministic model router selects
appropriate models based on document characteristics.
"""

import pytest
import asyncio
from typing import Dict, Any

from extraction.v2.extractors import select_model_for_document
from extraction.models import DocumentType


class TestModelRouting:
    """Test suite for model routing logic."""
    
    def test_academic_document_routing(self):
        """Test routing for academic documents."""
        # Test 1: Academic with quality requirement
        config = select_model_for_document(
            doc_type="academic",
            doc_length=5000,
            quality_required=True,
            budget_conscious=False
        )
        assert config["model"] == "gpt-4"
        assert "precision" in config["reason"].lower()
        
        # Test 2: Academic with budget constraint
        config = select_model_for_document(
            doc_type="academic",
            doc_length=5000,
            quality_required=False,
            budget_conscious=True
        )
        assert config["model"] == "gpt-3.5-turbo"
        assert "budget" in config["reason"].lower() or "3.5" in config["reason"].lower()
    
    def test_technical_document_routing(self):
        """Test routing for technical documents."""
        # Test 1: Technical with normal length
        config = select_model_for_document(
            doc_type="technical",
            doc_length=10000,
            quality_required=False,
            budget_conscious=False
        )
        assert config["model"] == "gpt-3.5-turbo"
        assert "technical" in config["reason"].lower()
        
        # Test 2: Technical with quality requirement
        config = select_model_for_document(
            doc_type="technical",
            doc_length=10000,
            quality_required=True,
            budget_conscious=False
        )
        assert config["model"] == "gpt-4"
    
    def test_long_document_routing(self):
        """Test routing for very long documents."""
        # Test 1: Very long document
        config = select_model_for_document(
            doc_type="narrative",
            doc_length=100000,
            quality_required=False,
            budget_conscious=False
        )
        assert config["model"] == "claude-3-opus"
        assert "long" in config["reason"].lower() or "context" in config["reason"].lower()
        
        # Test 2: Long document with quality requirement
        config = select_model_for_document(
            doc_type="unknown",
            doc_length=80000,
            quality_required=True,
            budget_conscious=True
        )
        assert config["model"] == "claude-3-opus"
        assert "context" in config["reason"].lower()
    
    def test_default_routing(self):
        """Test default routing for unknown documents."""
        # Test 1: Unknown type, short document
        config = select_model_for_document(
            doc_type="unknown",
            doc_length=3000,
            quality_required=False,
            budget_conscious=False
        )
        assert config["model"] == "gpt-3.5-turbo"
        assert "default" in config["reason"].lower()
        
        # Test 2: Unknown with quality requirement
        config = select_model_for_document(
            doc_type="unknown",
            doc_length=3000,
            quality_required=True,
            budget_conscious=False
        )
        assert config["model"] == "gpt-4"
    
    def test_extractor_selection(self):
        """Test that correct extractor is selected with model."""
        from extraction.v2.extractors import BaselineExtractor, AcademicExtractor, TechnicalExtractor
        
        # Academic document
        config = select_model_for_document(
            doc_type="academic",
            doc_length=5000,
            quality_required=True,
            budget_conscious=False
        )
        assert config["extractor"] == AcademicExtractor
        
        # Technical document
        config = select_model_for_document(
            doc_type="technical",
            doc_length=5000,
            quality_required=False,
            budget_conscious=False
        )
        assert config["extractor"] == TechnicalExtractor
        
        # Default
        config = select_model_for_document(
            doc_type="narrative",
            doc_length=5000,
            quality_required=False,
            budget_conscious=False
        )
        assert config["extractor"] == BaselineExtractor
    
    def test_edge_cases(self):
        """Test edge cases in routing."""
        # Test 1: Empty document type
        config = select_model_for_document(
            doc_type="",
            doc_length=5000,
            quality_required=False,
            budget_conscious=False
        )
        assert config["model"] == "gpt-3.5-turbo"
        
        # Test 2: Zero length
        config = select_model_for_document(
            doc_type="academic",
            doc_length=0,
            quality_required=False,
            budget_conscious=False
        )
        assert config["model"] in ["gpt-3.5-turbo", "gpt-4"]
        
        # Test 3: Extremely long document
        config = select_model_for_document(
            doc_type="technical",
            doc_length=500000,
            quality_required=True,
            budget_conscious=False
        )
        assert config["model"] == "claude-3-opus"
    
    def test_routing_consistency(self):
        """Test that routing is deterministic."""
        # Same inputs should always produce same outputs
        inputs = {
            "doc_type": "academic",
            "doc_length": 7500,
            "quality_required": True,
            "budget_conscious": False
        }
        
        # Run 10 times
        results = []
        for _ in range(10):
            config = select_model_for_document(**inputs)
            results.append(config["model"])
        
        # All results should be identical
        assert len(set(results)) == 1
        assert results[0] == "gpt-4"


class TestModelRoutingIntegration:
    """Integration tests for model routing in pipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_model_selection(self):
        """Test model selection within the pipeline."""
        from extraction.v2.pipeline import ExtractionPipeline, ExtractionConfig
        import os
        
        # Skip if no API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("No OpenAI API key available")
        
        # Create pipeline
        config = ExtractionConfig(
            api_key=api_key,
            max_enhancer_loops=0,  # Skip enhancement for speed
            quality_threshold=0.9,
            enable_human_validation=False,
            enhancers_enabled=[]
        )
        
        pipeline = ExtractionPipeline(config)
        
        # Mock the PDF processor to return different document types
        class MockPDFProcessor:
            def __init__(self, doc_type: str, length: int):
                self.doc_type = doc_type
                self.length = length
            
            async def process_pdf(self, path):
                # Generate content based on type
                if self.doc_type == "academic":
                    content = "This study investigates the hypothesis that machine learning algorithms can effectively predict disease outcomes. Our methodology involves systematic analysis of patient data. Results indicate significant correlation between predictive models and actual outcomes."
                elif self.doc_type == "technical":
                    content = "Installation Guide: 1. Install Python 3.8 or higher. 2. Run pip install requirements.txt. 3. Configure the API_KEY environment variable. 4. Execute python main.py to start the application. For debugging, use the --verbose flag."
                else:
                    content = "The company reported strong growth in Q4, with revenue increasing by 25% year-over-year. The CEO stated that new product launches and market expansion drove the positive results."
                
                # Repeat to reach desired length
                full_content = content * (self.length // len(content) + 1)
                
                return [type('Chunk', (), {'content': full_content[:self.length]})]
        
        # Test different document types
        test_cases = [
            ("academic", 5000, True, "gpt-4"),
            ("technical", 10000, False, "gpt-3.5-turbo"),
            ("business", 3000, False, "gpt-3.5-turbo"),
        ]
        
        for doc_type, length, quality_req, expected_model in test_cases:
            # Replace PDF processor
            pipeline.pdf_processor = MockPDFProcessor(doc_type, length)
            
            # Run extraction
            result = await pipeline.extract(
                pdf_path="test.pdf",
                requirements={"quality": quality_req}
            )
            
            # Verify model selection
            assert result["metadata"]["model_used"] == expected_model
            assert result["extraction"] is not None


def test_routing_performance():
    """Test performance of routing logic."""
    import time
    
    # Measure routing decision time
    start = time.time()
    
    for _ in range(1000):
        select_model_for_document(
            doc_type="academic",
            doc_length=5000,
            quality_required=True,
            budget_conscious=False
        )
    
    duration = time.time() - start
    avg_time = duration / 1000
    
    # Should be very fast (< 0.1ms per decision)
    assert avg_time < 0.0001
    print(f"Average routing time: {avg_time*1000:.3f}ms")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])