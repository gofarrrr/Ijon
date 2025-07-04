"""
Enhanced document-aware extraction with battle-tested patterns.

Integrates agent loops, thinking blocks, and adaptive strategies.
"""

import asyncio
from typing import Optional, Dict, Any, List
import json

from openai import AsyncOpenAI
from pydantic import ValidationError

from extraction.models import ExtractedKnowledge, DocumentProfile, Topic, Fact, Relationship, Question
from extraction.strategies.document_profiler import DocumentProfiler
from extraction.strategies.strategy_factory import StrategyFactory
from extraction.baseline.extractor import BaselineExtractor
from extraction.pdf_processor import PDFProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Enhanced system prompt for document-aware extraction
ENHANCED_DOCUMENT_AWARE_PROMPT = """You are a Document Analysis Specialist with adaptive extraction capabilities.

## Agent Loop Architecture
Your extraction process adapts to document type through these phases:
1. **Profile**: Understand document structure and type
2. **Strategize**: Select optimal extraction approach
3. **Extract**: Apply document-specific techniques
4. **Validate**: Ensure quality matches document type
5. **Refine**: Adjust based on document characteristics

## Document Awareness Framework
<thinking>
For each document, I will:
- Identify document type and structure
- Assess information density and complexity
- Choose appropriate extraction depth
- Adapt language to document style
- Validate against document norms
</thinking>

## Extraction Principles
1. **Adaptive Depth**: Match extraction detail to document type
2. **Structural Awareness**: Respect document organization
3. **Domain Expertise**: Apply field-specific knowledge
4. **Quality Calibration**: Adjust confidence by document type
5. **Coherent Output**: Maintain document's logical flow

Your goal is intelligent extraction that respects document context."""


class EnhancedDocumentAwareExtractor:
    """Enhanced document-aware extraction with adaptive strategies."""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize enhanced document-aware extractor.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for extraction
        """
        self.api_key = openai_api_key
        self.model = model
        self.client = AsyncOpenAI(api_key=openai_api_key)
        
        # Initialize components
        self.profiler = DocumentProfiler()
        self.strategy_factory = StrategyFactory(openai_api_key)
        self.baseline_extractor = BaselineExtractor(openai_api_key, model)
        self.pdf_processor = PDFProcessor()
        
        # Enhanced configuration
        self.enable_thinking = True
        self.adaptive_prompting = True
        self.multi_pass_extraction = True
        
        logger.info(f"Initialized EnhancedDocumentAwareExtractor with model: {model}")
    
    async def extract_with_enhanced_awareness(self, pdf_path: str) -> ExtractedKnowledge:
        """
        Extract knowledge with enhanced document awareness.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extraction adapted to document characteristics
        """
        try:
            # Phase 1: Comprehensive document profiling
            logger.info(f"Profiling document: {pdf_path}")
            profile = await self._enhanced_profile_document(pdf_path)
            
            logger.info(
                f"Document profile: {profile.document_type} "
                f"(confidence: {profile.type_confidence:.2f}), "
                f"complexity: {profile.complexity_score:.2f}"
            )
            
            # Phase 2: Strategy selection with thinking
            strategy_analysis = self._analyze_strategy_selection(profile)
            strategy = self.strategy_factory.get_strategy(profile)
            
            # Phase 3: Multi-pass extraction
            logger.info("Performing multi-pass extraction...")
            
            # First pass: Structure-aware extraction
            structure_extraction = await self._extract_with_structure_awareness(
                pdf_path, profile, strategy
            )
            
            # Second pass: Content-focused extraction
            content_extraction = await self._extract_with_content_focus(
                pdf_path, profile, strategy, structure_extraction
            )
            
            # Phase 4: Intelligent merging
            merged_extraction = self._merge_extractions(
                structure_extraction, content_extraction, profile
            )
            
            # Phase 5: Document-specific enhancement
            final_extraction = await self._apply_document_enhancements(
                merged_extraction, profile
            )
            
            # Phase 6: Quality validation
            final_extraction = self._validate_and_adjust_quality(
                final_extraction, profile
            )
            
            # Add comprehensive metadata
            final_extraction.extraction_metadata.update({
                "document_type": profile.document_type,
                "type_confidence": profile.type_confidence,
                "complexity_score": profile.complexity_score,
                "strategy_used": profile.recommended_strategy,
                "extraction_passes": 2,
                "adaptive_features": {
                    "thinking_enabled": self.enable_thinking,
                    "adaptive_prompting": self.adaptive_prompting,
                    "document_specific_enhancements": True
                },
                "quality_metrics": self._calculate_quality_metrics(final_extraction, profile)
            })
            
            logger.info(
                f"Enhanced extraction complete. "
                f"Topics: {len(final_extraction.topics)}, "
                f"Facts: {len(final_extraction.facts)}, "
                f"Questions: {len(final_extraction.questions)}, "
                f"Confidence: {final_extraction.overall_confidence:.2f}"
            )
            
            return final_extraction
            
        except Exception as e:
            logger.error(f"Enhanced document-aware extraction failed: {str(e)}")
            return self._create_fallback_extraction(str(e))
    
    async def _enhanced_profile_document(self, pdf_path: str) -> DocumentProfile:
        """Perform enhanced document profiling."""
        # Get base profile
        base_profile = await self.profiler.profile_document(pdf_path)
        
        # Enhance with additional analysis
        chunks = await self.pdf_processor.process_pdf(pdf_path)
        
        if chunks:
            # Analyze content complexity
            content_sample = " ".join([c.content[:200] for c in chunks[:5]])
            complexity_score = self._analyze_content_complexity(content_sample)
            
            # Detect specialized elements
            specialized_elements = self._detect_specialized_elements(chunks)
            
            # Update profile
            base_profile.metadata.update({
                "complexity_score": complexity_score,
                "specialized_elements": specialized_elements,
                "content_density": self._calculate_content_density(chunks),
                "estimated_reading_level": self._estimate_reading_level(content_sample)
            })
            
            # Store for later use
            base_profile.complexity_score = complexity_score
        
        return base_profile
    
    def _analyze_strategy_selection(self, profile: DocumentProfile) -> Dict[str, Any]:
        """Analyze and document strategy selection reasoning."""
        analysis = {
            "document_type": profile.document_type,
            "selected_strategy": profile.recommended_strategy,
            "reasoning": []
        }
        
        # Document type specific reasoning
        if profile.document_type == "research_paper":
            analysis["reasoning"].extend([
                "Research paper detected - will prioritize methodology and findings",
                "Enhanced citation tracking will be applied",
                "Questions will focus on research implications"
            ])
        elif profile.document_type == "technical_report":
            analysis["reasoning"].extend([
                "Technical report detected - will extract specifications and procedures",
                "Relationships will focus on system components",
                "Facts will emphasize quantitative data"
            ])
        elif profile.document_type == "textbook":
            analysis["reasoning"].extend([
                "Textbook detected - will structure by learning objectives",
                "Questions will cover multiple cognitive levels",
                "Topics will be hierarchically organized"
            ])
        
        # Complexity adjustments
        if hasattr(profile, 'complexity_score'):
            if profile.complexity_score > 0.7:
                analysis["reasoning"].append("High complexity - will use deeper analysis")
            elif profile.complexity_score < 0.3:
                analysis["reasoning"].append("Low complexity - will focus on clarity")
        
        logger.info(f"Strategy analysis: {analysis}")
        return analysis
    
    async def _extract_with_structure_awareness(
        self,
        pdf_path: str,
        profile: DocumentProfile,
        strategy: Any
    ) -> ExtractedKnowledge:
        """First pass: Extract with focus on document structure."""
        
        # Get content
        chunks = await self.pdf_processor.process_pdf(pdf_path)
        if not chunks:
            return self._create_fallback_extraction("No content extracted")
        
        # Build structure-aware prompt
        structure_prompt = self._build_structure_aware_prompt(chunks, profile)
        
        # Extract with structure focus
        try:
            messages = [
                {"role": "system", "content": ENHANCED_DOCUMENT_AWARE_PROMPT},
                {"role": "user", "content": structure_prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return self._parse_extraction_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Structure-aware extraction failed: {e}")
            return self._create_fallback_extraction(str(e))
    
    async def _extract_with_content_focus(
        self,
        pdf_path: str,
        profile: DocumentProfile,
        strategy: Any,
        structure_extraction: ExtractedKnowledge
    ) -> ExtractedKnowledge:
        """Second pass: Extract with focus on content depth."""
        
        # Get content
        chunks = await self.pdf_processor.process_pdf(pdf_path)
        if not chunks:
            return structure_extraction
        
        # Build content-focused prompt
        content_prompt = self._build_content_focused_prompt(
            chunks, profile, structure_extraction
        )
        
        # Extract with content focus
        try:
            messages = [
                {"role": "system", "content": ENHANCED_DOCUMENT_AWARE_PROMPT},
                {"role": "user", "content": content_prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,  # Slightly higher for content diversity
                response_format={"type": "json_object"}
            )
            
            return self._parse_extraction_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Content-focused extraction failed: {e}")
            return structure_extraction
    
    def _build_structure_aware_prompt(self, chunks: List[Any], profile: DocumentProfile) -> str:
        """Build prompt focused on document structure."""
        
        # Sample content focusing on structure
        structure_samples = []
        for chunk in chunks[:10]:  # First 10 chunks for structure
            if chunk.section_title:
                structure_samples.append(f"Section: {chunk.section_title}")
            if chunk.chapter_title:
                structure_samples.append(f"Chapter: {chunk.chapter_title}")
            structure_samples.append(f"Content preview: {chunk.content[:100]}...")
        
        prompt = f"""Analyze the structure and organization of this {profile.document_type} document.

Document Type: {profile.document_type}
Structure Elements: {', '.join(profile.structure_features)}
Complexity: {getattr(profile, 'complexity_score', 0.5):.2f}

## Document Structure Sample
{chr(10).join(structure_samples[:15])}

## Extraction Focus: Document Structure

<thinking>
This is a {profile.document_type}. I should:
- Identify the main structural elements
- Extract topics that reflect document organization
- Find relationships between sections
- Generate questions about document structure
</thinking>

### Phase 1: Structural Analysis
Extract knowledge focusing on:
1. **Topics**: Main sections and their purposes
2. **Relationships**: How sections connect and build upon each other
3. **Organization**: The document's logical flow

### Phase 2: Extraction Requirements
Provide:
- 3-5 topics representing major document sections
- Relationships showing document organization
- Facts about document structure and purpose
- Questions about the document's organization

Output as JSON with standard extraction fields. Focus on document structure and organization."""
        
        return prompt
    
    def _build_content_focused_prompt(
        self,
        chunks: List[Any],
        profile: DocumentProfile,
        structure_extraction: ExtractedKnowledge
    ) -> str:
        """Build prompt focused on content depth."""
        
        # Get full content
        content = "\n\n".join([chunk.content for chunk in chunks])
        if len(content) > 8000:
            content = content[:8000] + "..."
        
        # Reference existing structure
        existing_topics = [t.name for t in structure_extraction.topics]
        
        prompt = f"""Extract detailed content from this {profile.document_type} document.

Document Type: {profile.document_type}
Already Identified Topics: {', '.join(existing_topics)}

## Full Content
{content}

## Extraction Focus: Content Depth

<thinking>
I already have the document structure. Now I need to:
- Extract detailed facts with evidence
- Find specific relationships between concepts
- Generate diverse questions at different cognitive levels
- Create a comprehensive summary
</thinking>

### Phase 1: Deep Content Analysis
Focus on:
1. **Facts**: Specific claims, findings, and statements with evidence
2. **Detailed Relationships**: How concepts interact and influence each other
3. **Comprehensive Questions**: Cover all cognitive levels from recall to synthesis

### Phase 2: Quality Requirements
- Every fact MUST have supporting evidence
- Relationships should explain the nature of connections
- Questions should include expected answers
- Summary should be 3-4 sentences of academic prose

### Phase 3: Document-Specific Focus"""
        
        # Add document-specific instructions
        if profile.document_type == "research_paper":
            prompt += """
- Prioritize methodology, results, and conclusions
- Extract statistical findings with exact values
- Focus on research implications and limitations"""
        elif profile.document_type == "technical_report":
            prompt += """
- Extract specifications, procedures, and recommendations
- Focus on technical details and measurements
- Identify system components and their interactions"""
        elif profile.document_type == "textbook":
            prompt += """
- Extract key concepts, definitions, and examples
- Identify learning objectives and prerequisites
- Generate educational questions for each topic"""
        
        prompt += "\n\nOutput as JSON with comprehensive content extraction."
        
        return prompt
    
    def _merge_extractions(
        self,
        structure_extraction: ExtractedKnowledge,
        content_extraction: ExtractedKnowledge,
        profile: DocumentProfile
    ) -> ExtractedKnowledge:
        """Intelligently merge multiple extraction passes."""
        
        # Start with structure extraction as base
        merged = ExtractedKnowledge(
            topics=structure_extraction.topics.copy(),
            facts=[],
            relationships=[],
            questions=[],
            summary="",
            overall_confidence=0.0
        )
        
        # Merge topics (prefer structure extraction but add missing from content)
        topic_names = {t.name.lower() for t in merged.topics}
        for topic in content_extraction.topics:
            if topic.name.lower() not in topic_names:
                merged.topics.append(topic)
        
        # Merge facts (prefer content extraction for detail)
        merged.facts = content_extraction.facts.copy()
        
        # Add any unique facts from structure extraction
        fact_claims = {f.claim.lower() for f in merged.facts}
        for fact in structure_extraction.facts:
            if fact.claim.lower() not in fact_claims:
                merged.facts.append(fact)
        
        # Merge relationships (combine both, remove duplicates)
        rel_keys = set()
        for rel in content_extraction.relationships + structure_extraction.relationships:
            key = f"{rel.source_entity}|{rel.target_entity}|{rel.relationship_type}"
            if key not in rel_keys:
                merged.relationships.append(rel)
                rel_keys.add(key)
        
        # Merge questions (prefer content extraction but ensure diversity)
        merged.questions = content_extraction.questions.copy()
        
        # Add structural questions if missing
        existing_levels = {q.cognitive_level for q in merged.questions}
        for q in structure_extraction.questions:
            if q.cognitive_level not in existing_levels or len(merged.questions) < 8:
                merged.questions.append(q)
                existing_levels.add(q.cognitive_level)
        
        # Use content extraction summary (usually more comprehensive)
        merged.summary = content_extraction.summary or structure_extraction.summary
        
        # Calculate merged confidence
        merged.overall_confidence = (
            structure_extraction.overall_confidence * 0.3 +
            content_extraction.overall_confidence * 0.7
        )
        
        logger.info(
            f"Merged extraction: {len(merged.topics)} topics, "
            f"{len(merged.facts)} facts, {len(merged.questions)} questions"
        )
        
        return merged
    
    async def _apply_document_enhancements(
        self,
        extraction: ExtractedKnowledge,
        profile: DocumentProfile
    ) -> ExtractedKnowledge:
        """Apply document-specific enhancements."""
        
        if profile.document_type == "research_paper":
            extraction = await self._enhance_research_paper(extraction)
        elif profile.document_type == "technical_report":
            extraction = await self._enhance_technical_report(extraction)
        elif profile.document_type == "textbook":
            extraction = await self._enhance_textbook(extraction)
        elif profile.document_type == "legal_document":
            extraction = await self._enhance_legal_document(extraction)
        
        return extraction
    
    async def _enhance_research_paper(self, extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """Enhance extraction for research papers."""
        
        # Ensure methodology facts
        has_methodology = any("method" in f.claim.lower() for f in extraction.facts)
        if not has_methodology:
            logger.info("Adding methodology enhancement for research paper")
            # Would normally extract methodology here
        
        # Ensure research questions
        research_questions = [q for q in extraction.questions 
                            if any(term in q.question_text.lower() 
                                  for term in ["hypothesis", "research", "study"])]
        if not research_questions:
            # Add a research-focused question
            from extraction.models import Question, CognitiveLevel
            extraction.questions.append(Question(
                question_text="What are the main research contributions of this study?",
                expected_answer="Based on the extracted content, the main contributions should be identified from the conclusion or abstract sections.",
                cognitive_level=CognitiveLevel.ANALYZE,
                difficulty=4,
                confidence=0.7
            ))
        
        return extraction
    
    async def _enhance_technical_report(self, extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """Enhance extraction for technical reports."""
        
        # Ensure technical specifications in facts
        spec_facts = [f for f in extraction.facts 
                     if any(term in f.claim.lower() 
                           for term in ["specification", "requirement", "standard"])]
        
        if len(spec_facts) < 3:
            logger.info("Technical report may need more specification extraction")
        
        # Add technical relationships
        component_entities = [e for e in self._get_all_entities(extraction)
                            if any(term in e.lower() 
                                  for term in ["system", "component", "module"])]
        
        if len(component_entities) > 1 and len(extraction.relationships) < 5:
            # Would normally add component relationships here
            pass
        
        return extraction
    
    async def _enhance_textbook(self, extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """Enhance extraction for textbooks."""
        
        # Ensure educational questions at all levels
        from extraction.models import CognitiveLevel
        
        level_counts = {}
        for q in extraction.questions:
            level = q.cognitive_level
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Ensure at least one question per level
        missing_levels = set(CognitiveLevel) - set(level_counts.keys())
        if missing_levels:
            logger.info(f"Adding questions for missing levels: {missing_levels}")
            # Would normally generate questions for missing levels
        
        # Ensure learning objectives in topics
        has_objectives = any("objective" in t.description.lower() or 
                           "learn" in t.description.lower() 
                           for t in extraction.topics)
        
        if not has_objectives and extraction.topics:
            extraction.topics[0].description = (
                "Learning objectives: " + extraction.topics[0].description
            )
        
        return extraction
    
    async def _enhance_legal_document(self, extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """Enhance extraction for legal documents."""
        
        # Ensure legal terminology in facts
        legal_terms = ["shall", "whereas", "pursuant", "liability", "obligation"]
        legal_facts = [f for f in extraction.facts
                      if any(term in f.claim.lower() for term in legal_terms)]
        
        if len(legal_facts) < len(extraction.facts) * 0.3:
            logger.info("Legal document may need more precise legal language extraction")
        
        # Ensure hierarchical relationships for legal structure
        hierarchical_rels = [r for r in extraction.relationships
                           if r.relationship_type in ["contains", "supersedes", "references"]]
        
        if len(hierarchical_rels) < 2:
            # Would normally add legal structure relationships
            pass
        
        return extraction
    
    def _validate_and_adjust_quality(
        self,
        extraction: ExtractedKnowledge,
        profile: DocumentProfile
    ) -> ExtractedKnowledge:
        """Validate and adjust extraction quality based on document type."""
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(extraction, profile)
        
        # Adjust confidence based on document type and quality
        confidence_adjustment = 1.0
        
        # Document type adjustments
        if profile.document_type == "research_paper":
            if quality_metrics["has_methodology"]:
                confidence_adjustment *= 1.1
            if quality_metrics["citation_count"] > 5:
                confidence_adjustment *= 1.05
        elif profile.document_type == "technical_report":
            if quality_metrics["has_specifications"]:
                confidence_adjustment *= 1.1
            if quality_metrics["quantitative_facts_ratio"] > 0.3:
                confidence_adjustment *= 1.05
        
        # Quality adjustments
        if quality_metrics["facts_with_evidence_ratio"] > 0.8:
            confidence_adjustment *= 1.1
        if quality_metrics["question_diversity"] > 0.7:
            confidence_adjustment *= 1.05
        
        # Apply adjustment
        extraction.overall_confidence = min(
            0.95,
            extraction.overall_confidence * confidence_adjustment
        )
        
        # Add quality validation metadata
        extraction.extraction_metadata["quality_validation"] = {
            "metrics": quality_metrics,
            "confidence_adjustment": confidence_adjustment,
            "validation_passed": quality_metrics["overall_quality"] > 0.6
        }
        
        return extraction
    
    def _calculate_quality_metrics(
        self,
        extraction: ExtractedKnowledge,
        profile: DocumentProfile
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics."""
        
        metrics = {
            # Basic counts
            "topic_count": len(extraction.topics),
            "fact_count": len(extraction.facts),
            "relationship_count": len(extraction.relationships),
            "question_count": len(extraction.questions),
            
            # Quality indicators
            "facts_with_evidence_ratio": sum(1 for f in extraction.facts if f.evidence) / max(1, len(extraction.facts)),
            "avg_topic_description_length": sum(len(t.description) for t in extraction.topics) / max(1, len(extraction.topics)),
            "question_diversity": len(set(q.cognitive_level for q in extraction.questions)) / 6.0,
            
            # Document-specific metrics
            "has_methodology": any("method" in f.claim.lower() for f in extraction.facts),
            "has_specifications": any("specif" in f.claim.lower() for f in extraction.facts),
            "citation_count": sum(1 for f in extraction.facts if any(c in f.evidence for c in ["[", "("])),
            "quantitative_facts_ratio": sum(1 for f in extraction.facts if any(c.isdigit() for c in f.claim)) / max(1, len(extraction.facts)),
            
            # Summary quality
            "summary_length": len(extraction.summary) if extraction.summary else 0,
            "summary_sentences": extraction.summary.count('.') if extraction.summary else 0,
        }
        
        # Calculate overall quality score
        quality_score = (
            (metrics["facts_with_evidence_ratio"] * 0.3) +
            (min(metrics["question_diversity"], 1.0) * 0.2) +
            (min(metrics["topic_count"] / 5.0, 1.0) * 0.2) +
            (min(metrics["relationship_count"] / 5.0, 1.0) * 0.15) +
            (1.0 if metrics["summary_length"] > 100 else 0.5) * 0.15
        )
        
        metrics["overall_quality"] = quality_score
        
        return metrics
    
    def _get_all_entities(self, extraction: ExtractedKnowledge) -> List[str]:
        """Get all entities from extraction."""
        entities = []
        
        # From topics
        entities.extend(t.name for t in extraction.topics)
        
        # From relationships
        for rel in extraction.relationships:
            entities.append(rel.source_entity)
            entities.append(rel.target_entity)
        
        # Simple entity extraction from facts
        for fact in extraction.facts:
            words = fact.claim.split()
            entities.extend(w for w in words if w[0].isupper() and len(w) > 2)
        
        return list(set(entities))
    
    def _analyze_content_complexity(self, content: str) -> float:
        """Analyze content complexity (0-1 scale)."""
        # Simple heuristics for complexity
        words = content.split()
        if not words:
            return 0.5
        
        # Average word length
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Sentence complexity (words per sentence)
        sentences = content.split('.')
        avg_sentence_length = len(words) / max(1, len(sentences))
        
        # Technical term density
        technical_indicators = [
            'algorithm', 'methodology', 'hypothesis', 'analysis',
            'framework', 'implementation', 'optimization', 'evaluation'
        ]
        technical_density = sum(1 for w in words if w.lower() in technical_indicators) / len(words)
        
        # Calculate complexity score
        complexity = (
            min(avg_word_length / 8.0, 1.0) * 0.3 +
            min(avg_sentence_length / 25.0, 1.0) * 0.3 +
            min(technical_density * 20, 1.0) * 0.4
        )
        
        return complexity
    
    def _detect_specialized_elements(self, chunks: List[Any]) -> Dict[str, int]:
        """Detect specialized document elements."""
        elements = {
            "equations": 0,
            "tables": 0,
            "figures": 0,
            "citations": 0,
            "code_blocks": 0,
            "lists": 0
        }
        
        for chunk in chunks:
            content = chunk.content
            
            # Simple detection heuristics
            if any(sym in content for sym in ['∑', '∫', '∂', '=']):
                elements["equations"] += 1
            if 'Table' in content or '|' in content:
                elements["tables"] += 1
            if 'Figure' in content or 'Fig.' in content:
                elements["figures"] += 1
            if '[' in content and ']' in content:
                elements["citations"] += content.count('[')
            if '```' in content or 'def ' in content or 'function' in content:
                elements["code_blocks"] += 1
            if any(marker in content for marker in ['•', '1.', '-']):
                elements["lists"] += 1
        
        return elements
    
    def _calculate_content_density(self, chunks: List[Any]) -> float:
        """Calculate content density metric."""
        if not chunks:
            return 0.0
        
        total_chars = sum(len(c.content) for c in chunks)
        total_words = sum(len(c.content.split()) for c in chunks)
        
        # Average words per chunk
        avg_words_per_chunk = total_words / len(chunks)
        
        # Density score (normalized)
        density = min(avg_words_per_chunk / 300.0, 1.0)
        
        return density
    
    def _estimate_reading_level(self, content: str) -> str:
        """Estimate reading level of content."""
        words = content.split()
        if not words:
            return "unknown"
        
        # Simple readability heuristics
        avg_word_length = sum(len(w) for w in words) / len(words)
        complex_words = sum(1 for w in words if len(w) > 7) / len(words)
        
        if avg_word_length > 6 and complex_words > 0.3:
            return "graduate"
        elif avg_word_length > 5 and complex_words > 0.2:
            return "undergraduate"
        elif avg_word_length > 4:
            return "high_school"
        else:
            return "general"
    
    def _parse_extraction_response(self, response: str) -> ExtractedKnowledge:
        """Parse extraction response with enhanced validation."""
        try:
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            
            # Parse JSON
            data = json.loads(response)
            
            # Create extraction with validation
            return ExtractedKnowledge(
                topics=self._parse_topics(data.get("topics", [])),
                facts=self._parse_facts(data.get("facts", [])),
                relationships=self._parse_relationships(data.get("relationships", [])),
                questions=self._parse_questions(data.get("questions", [])),
                summary=data.get("summary", ""),
                overall_confidence=float(data.get("overall_confidence", 0.7)),
                extraction_metadata=data.get("extraction_metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to parse extraction response: {e}")
            return self._create_fallback_extraction(f"Parse error: {str(e)}")
    
    def _parse_topics(self, topics_data: List[Dict]) -> List[Topic]:
        """Parse topics with validation."""
        topics = []
        for t in topics_data:
            if isinstance(t, dict) and "name" in t:
                topics.append(Topic(
                    name=t["name"],
                    description=t.get("description", ""),
                    keywords=t.get("keywords", []),
                    confidence=float(t.get("confidence", 0.7))
                ))
        return topics
    
    def _parse_facts(self, facts_data: List[Dict]) -> List[Fact]:
        """Parse facts with validation."""
        facts = []
        for f in facts_data:
            if isinstance(f, dict) and "claim" in f:
                facts.append(Fact(
                    claim=f["claim"],
                    evidence=f.get("evidence", ""),
                    confidence=float(f.get("confidence", 0.7))
                ))
        return facts
    
    def _parse_relationships(self, rels_data: List[Dict]) -> List[Relationship]:
        """Parse relationships with validation."""
        relationships = []
        for r in rels_data:
            if isinstance(r, dict) and all(k in r for k in ["source_entity", "target_entity"]):
                relationships.append(Relationship(
                    source_entity=r["source_entity"],
                    target_entity=r["target_entity"],
                    relationship_type=r.get("relationship_type", "related_to"),
                    description=r.get("description", ""),
                    confidence=float(r.get("confidence", 0.7))
                ))
        return relationships
    
    def _parse_questions(self, questions_data: List[Dict]) -> List[Question]:
        """Parse questions with validation."""
        from extraction.models import CognitiveLevel
        
        questions = []
        for q in questions_data:
            if isinstance(q, dict) and "question_text" in q:
                try:
                    level = CognitiveLevel(q.get("cognitive_level", "understand"))
                except:
                    level = CognitiveLevel.UNDERSTAND
                
                questions.append(Question(
                    question_text=q["question_text"],
                    expected_answer=q.get("expected_answer", ""),
                    cognitive_level=level,
                    difficulty=int(q.get("difficulty", 3)),
                    confidence=float(q.get("confidence", 0.7))
                ))
        return questions
    
    def _create_fallback_extraction(self, error_msg: str) -> ExtractedKnowledge:
        """Create fallback extraction for errors."""
        return ExtractedKnowledge(
            topics=[],
            facts=[],
            relationships=[],
            questions=[],
            overall_confidence=0.0,
            summary=f"Extraction failed: {error_msg}",
            extraction_metadata={"error": error_msg, "fallback": True}
        )
    
    async def compare_extraction_methods(self, pdf_path: str) -> Dict[str, Any]:
        """Compare enhanced extraction with baseline methods."""
        logger.info(f"Running extraction method comparison for: {pdf_path}")
        
        # Run different extraction methods
        baseline_task = self.baseline_extractor.extract(
            chunk_id=f"baseline_{pdf_path}",
            content=await self._get_full_content(pdf_path)
        )
        
        standard_task = self.extract(pdf_path)  # Standard document-aware
        enhanced_task = self.extract_with_enhanced_awareness(pdf_path)
        
        # Gather results
        results = await asyncio.gather(
            baseline_task,
            standard_task,
            enhanced_task,
            return_exceptions=True
        )
        
        baseline_result, standard_result, enhanced_result = results
        
        # Handle errors
        if isinstance(baseline_result, Exception):
            baseline_result = self._create_fallback_extraction(str(baseline_result))
        if isinstance(standard_result, Exception):
            standard_result = self._create_fallback_extraction(str(standard_result))
        if isinstance(enhanced_result, Exception):
            enhanced_result = self._create_fallback_extraction(str(enhanced_result))
        
        # Compare results
        comparison = {
            "pdf_path": pdf_path,
            "baseline": self._summarize_extraction(baseline_result),
            "standard_aware": self._summarize_extraction(standard_result),
            "enhanced_aware": self._summarize_extraction(enhanced_result),
            "improvements": {
                "standard_over_baseline": self._calculate_improvement(baseline_result, standard_result),
                "enhanced_over_standard": self._calculate_improvement(standard_result, enhanced_result),
                "enhanced_over_baseline": self._calculate_improvement(baseline_result, enhanced_result)
            },
            "quality_analysis": {
                "baseline_quality": self._analyze_extraction_quality(baseline_result),
                "standard_quality": self._analyze_extraction_quality(standard_result),
                "enhanced_quality": self._analyze_extraction_quality(enhanced_result)
            }
        }
        
        return comparison
    
    def _summarize_extraction(self, extraction: ExtractedKnowledge) -> Dict[str, Any]:
        """Create summary of extraction for comparison."""
        return {
            "topics": len(extraction.topics),
            "facts": len(extraction.facts),
            "facts_with_evidence": sum(1 for f in extraction.facts if f.evidence),
            "relationships": len(extraction.relationships),
            "questions": len(extraction.questions),
            "question_levels": len(set(q.cognitive_level for q in extraction.questions)),
            "confidence": extraction.overall_confidence,
            "has_summary": bool(extraction.summary and len(extraction.summary) > 50),
            "metadata": extraction.extraction_metadata
        }
    
    def _calculate_improvement(self, before: ExtractedKnowledge, after: ExtractedKnowledge) -> Dict[str, Any]:
        """Calculate improvement metrics between extractions."""
        return {
            "topics_added": len(after.topics) - len(before.topics),
            "facts_added": len(after.facts) - len(before.facts),
            "evidence_improvement": (
                sum(1 for f in after.facts if f.evidence) - 
                sum(1 for f in before.facts if f.evidence)
            ),
            "relationships_added": len(after.relationships) - len(before.relationships),
            "questions_added": len(after.questions) - len(before.questions),
            "confidence_gain": after.overall_confidence - before.overall_confidence,
            "quality_improvement": (
                self._analyze_extraction_quality(after)["overall"] -
                self._analyze_extraction_quality(before)["overall"]
            )
        }
    
    def _analyze_extraction_quality(self, extraction: ExtractedKnowledge) -> Dict[str, float]:
        """Analyze extraction quality for comparison."""
        return {
            "completeness": min(
                (len(extraction.topics) / 5.0) * 0.25 +
                (len(extraction.facts) / 10.0) * 0.25 +
                (len(extraction.relationships) / 5.0) * 0.25 +
                (len(extraction.questions) / 8.0) * 0.25,
                1.0
            ),
            "evidence_quality": sum(1 for f in extraction.facts if f.evidence) / max(1, len(extraction.facts)),
            "question_diversity": len(set(q.cognitive_level for q in extraction.questions)) / 6.0,
            "confidence": extraction.overall_confidence,
            "overall": 0.0  # Will be calculated
        }
    
    async def _get_full_content(self, pdf_path: str) -> str:
        """Get full content from PDF."""
        try:
            chunks = await self.pdf_processor.process_pdf(pdf_path)
            return "\n\n".join([chunk.content for chunk in chunks])
        except:
            return ""
    
    async def extract(self, pdf_path: str) -> ExtractedKnowledge:
        """Standard document-aware extraction (for comparison)."""
        # This would be the original extract method
        # For now, using baseline as placeholder
        content = await self._get_full_content(pdf_path)
        if content:
            return await self.baseline_extractor.extract(
                chunk_id=f"standard_{pdf_path}",
                content=content[:10000]  # Limit for baseline
            )
        return self._create_fallback_extraction("No content")