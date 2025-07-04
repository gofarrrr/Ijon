"""
Stateless extractors following 12-factor principles.

Each extractor is a pure function: (content, config) -> ExtractedKnowledge
"""

from typing import Dict, Any, Optional
import json
import os
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

from extraction.models import ExtractedKnowledge, Topic, Fact, Relationship, Question
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StatelessExtractor:
    """Base class for stateless extractors - all methods are static."""
    
    @staticmethod
    def build_prompt(content: str, instructions: str) -> str:
        """Build extraction prompt - pure function."""
        return f"""Extract structured knowledge from the following text.

{instructions}

Text to analyze:
{content}

Respond with a JSON object containing:
- topics: List of main topics/concepts
- facts: List of factual claims with evidence
- relationships: How concepts relate to each other
- questions: Questions for understanding
- summary: Brief summary
- overall_confidence: Your confidence (0.0-1.0)
"""

    @staticmethod
    async def call_llm(client, 
                      prompt: str,
                      model: str = "gpt-3.5-turbo",
                      temperature: float = 0.3) -> Dict[str, Any]:
        """Call LLM - supports both OpenAI and Gemini models."""
        
        # Handle Gemini models
        if model.startswith("gemini"):
            return await StatelessExtractor._call_gemini(prompt, model, temperature)
        
        # Handle OpenAI models
        if not isinstance(client, AsyncOpenAI):
            raise ValueError("OpenAI client required for OpenAI models")
            
        # Create completion request - handle models that don't support JSON mode
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise knowledge extractor. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        
        # Only add JSON response format for models that support it
        if model in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]:
            request_params["response_format"] = {"type": "json_object"}
        
        response = await client.chat.completions.create(**request_params)
        
        return json.loads(response.choices[0].message.content)
    
    @staticmethod
    async def _call_gemini(prompt: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.3) -> Dict[str, Any]:
        """Call Gemini API directly."""
        # Configure Gemini
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=gemini_api_key)
        
        # Create model instance
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction="You are a precise knowledge extractor. Respond with valid JSON only."
        )
        
        # Generate content
        response = await model_instance.generate_content_async(prompt)
        
        return json.loads(response.text)

    @staticmethod
    def parse_response(response: Dict[str, Any]) -> ExtractedKnowledge:
        """Parse LLM response - pure function."""
        # Handle case variations
        if any(k.isupper() for k in response.keys()):
            response = {k.lower(): v for k, v in response.items()}
        
        # Handle summary as dict
        summary = response.get("summary", "")
        if isinstance(summary, dict):
            summary = summary.get("text", "") or str(summary)
        
        # Convert simple strings to proper model objects
        topics = []
        for topic in response.get("topics", []):
            if isinstance(topic, str):
                topics.append(Topic(
                    name=topic,
                    description=f"Topic: {topic}",
                    confidence=0.8
                ))
            else:
                # Ensure required fields
                if "confidence" not in topic:
                    topic["confidence"] = 0.8
                if "description" not in topic:
                    topic["description"] = topic.get("name", "Unknown topic")
                topics.append(Topic(**topic))
        
        # Convert facts to proper Fact objects
        facts = []
        for fact in response.get("facts", []):
            if isinstance(fact, dict) and "fact" in fact:
                # Handle old format
                facts.append(Fact(
                    claim=fact["fact"],
                    evidence=fact.get("evidence", ""),
                    confidence=0.7
                ))
            elif isinstance(fact, str):
                facts.append(Fact(
                    claim=fact,
                    confidence=0.7
                ))
            else:
                # Ensure confidence is present
                if "confidence" not in fact:
                    fact["confidence"] = 0.7
                facts.append(Fact(**fact))
        
        # Convert relationships - handle both our format and Gemini's format
        relationships = []
        rel_data = response.get("relationships", [])
        if isinstance(rel_data, dict):
            # Convert dict to list of relationships
            for key, desc in rel_data.items():
                if " and " in key:
                    src, tgt = key.split(" and ", 1)
                    relationships.append(Relationship(
                        source_entity=src,
                        target_entity=tgt,
                        relationship_type="related_to",
                        description=desc,
                        confidence=0.7
                    ))
        elif isinstance(rel_data, list):
            for rel in rel_data:
                if isinstance(rel, dict):
                    # Handle Gemini format: subject, relation/predicate, object
                    if "subject" in rel and "object" in rel and ("relation" in rel or "predicate" in rel):
                        rel_type = rel.get("relation") or rel.get("predicate", "related_to")
                        relationships.append(Relationship(
                            source_entity=rel["subject"],
                            target_entity=rel["object"],
                            relationship_type=rel_type,
                            description=f"{rel['subject']} {rel_type} {rel['object']}",
                            confidence=rel.get("confidence", 0.7)
                        ))
                    # Handle our standard format
                    elif "source_entity" in rel and "target_entity" in rel:
                        if "confidence" not in rel:
                            rel["confidence"] = 0.7
                        relationships.append(Relationship(**rel))
                    # Handle string relationships
                    elif isinstance(rel, str):
                        relationships.append(Relationship(
                            source_entity="Unknown",
                            target_entity="Unknown", 
                            relationship_type="described_as",
                            description=rel,
                            confidence=0.6
                        ))
        
        # Convert questions - handle both our format and Gemini's format
        questions = []
        for q in response.get("questions", []):
            if isinstance(q, str):
                questions.append(Question(
                    question_text=q,
                    confidence=0.7
                ))
            elif isinstance(q, dict):
                # Handle Gemini format with 'question' field
                if "question" in q and "question_text" not in q:
                    q["question_text"] = q.pop("question")
                # Ensure required fields
                if "confidence" not in q:
                    q["confidence"] = 0.7
                questions.append(Question(**q))
        
        return ExtractedKnowledge(
            topics=topics,
            facts=facts,
            relationships=relationships,
            questions=questions,
            summary=summary,
            overall_confidence=response.get("overall_confidence", 0.5)
        )


class BaselineExtractor(StatelessExtractor):
    """Simple baseline extraction - stateless."""
    
    @staticmethod
    async def extract(content: str, 
                     client: AsyncOpenAI,
                     model: str = "gpt-3.5-turbo") -> ExtractedKnowledge:
        """
        Extract knowledge - pure function.
        
        Args:
            content: Text to extract from
            client: OpenAI client (passed in, not stored)
            model: Model to use
            
        Returns:
            Extracted knowledge
        """
        # Build prompt
        instructions = """Focus on:
1. Main topics and concepts
2. Factual claims with supporting evidence
3. How concepts relate to each other
4. Questions that test understanding
5. A comprehensive summary"""
        
        prompt = BaselineExtractor.build_prompt(content, instructions)
        
        # Call LLM
        response = await BaselineExtractor.call_llm(client, prompt, model)
        
        # Parse response
        return BaselineExtractor.parse_response(response)


class AcademicExtractor(StatelessExtractor):
    """Academic-focused extraction - stateless."""
    
    @staticmethod
    async def extract(content: str,
                     client: AsyncOpenAI, 
                     model: str = "gpt-4") -> ExtractedKnowledge:
        """Extract with academic focus - pure function."""
        
        instructions = """Focus on academic elements:
1. Research topics and theoretical frameworks
2. Empirical findings with citations
3. Methodological approaches
4. Research questions and hypotheses
5. Academic contributions and limitations

Pay special attention to:
- Statistical results (p-values, confidence intervals)
- Citation patterns
- Research methodology
- Theoretical implications"""

        prompt = AcademicExtractor.build_prompt(content, instructions)
        response = await AcademicExtractor.call_llm(client, prompt, model, temperature=0.2)
        
        extraction = AcademicExtractor.parse_response(response)
        
        # Post-process for academic features
        extraction = AcademicExtractor._enhance_academic_features(extraction)
        
        return extraction
    
    @staticmethod
    def _enhance_academic_features(extraction: ExtractedKnowledge) -> ExtractedKnowledge:
        """Enhance academic-specific features - pure function."""
        # This is still a pure function - takes extraction, returns modified extraction
        # Add academic-specific enhancements
        for fact in extraction.facts:
            # Boost confidence for facts with citations
            if any(indicator in str(fact.evidence) for indicator in ["et al.", "p <", "n ="]):
                fact.confidence = min(1.0, fact.confidence * 1.1)
        
        return extraction


class TechnicalExtractor(StatelessExtractor):
    """Technical document extraction - stateless."""
    
    @staticmethod
    async def extract(content: str,
                     client: AsyncOpenAI,
                     model: str = "gpt-3.5-turbo") -> ExtractedKnowledge:
        """Extract with technical focus - pure function."""
        
        instructions = """Focus on technical elements:
1. Technical concepts and specifications
2. Implementation details and requirements
3. Code examples and configurations
4. Troubleshooting and solutions
5. Best practices and warnings

Pay special attention to:
- Exact commands and syntax
- Version numbers and dependencies
- Configuration parameters
- Error messages and solutions"""

        prompt = TechnicalExtractor.build_prompt(content, instructions)
        response = await TechnicalExtractor.call_llm(client, prompt, model, temperature=0.1)
        
        return TechnicalExtractor.parse_response(response)


# Model router - pure function for model selection
def select_model_for_document(doc_type: str,
                            doc_length: int,
                            quality_required: bool = False,
                            budget_conscious: bool = False,
                            use_gemini: bool = True) -> Dict[str, Any]:
    """
    Select best model for document - now includes powerful Gemini models!
    
    Returns dict with model and extractor class.
    """
    # Rule 1: Very long documents - use Gemini 2.5 Pro (2M token context + best reasoning!)
    if doc_length > 50000:
        if use_gemini:
            return {
                "model": "gemini-2.5-pro",
                "extractor": BaselineExtractor,
                "reason": "Long document needs Gemini 2.5 Pro's massive context and superior reasoning"
            }
        return {
            "model": "claude-3-opus",
            "extractor": BaselineExtractor,
            "reason": "Long document needs larger context"
        }
    
    # Rule 2: Academic documents - Gemini 2.5 Pro excels at reasoning and research
    if doc_type == "academic":
        if quality_required and use_gemini:
            return {
                "model": "gemini-2.5-pro",
                "extractor": AcademicExtractor,
                "reason": "Gemini 2.5 Pro provides flagship-level reasoning for academic content"
            }
        elif quality_required:
            return {
                "model": "gpt-4",
                "extractor": AcademicExtractor,
                "reason": "Academic document needs precision"
            }
        elif budget_conscious:
            return {
                "model": "gpt-3.5-turbo",
                "extractor": BaselineExtractor,
                "reason": "Academic with budget constraint uses 3.5"
            }
        else:
            return {
                "model": "gemini-1.5-flash" if use_gemini else "gpt-3.5-turbo",
                "extractor": AcademicExtractor,
                "reason": "Academic document with standard requirements"
            }
    
    # Rule 3: Technical documents - Gemini 2.5 Pro excels at complex technical reasoning
    if doc_type == "technical":
        if quality_required and not budget_conscious and use_gemini:
            return {
                "model": "gemini-2.5-pro",
                "extractor": TechnicalExtractor,
                "reason": "Gemini 2.5 Pro excels at complex technical reasoning and code analysis"
            }
        elif quality_required and not budget_conscious:
            return {
                "model": "gpt-4",
                "extractor": TechnicalExtractor,
                "reason": "Technical with quality requirement"
            }
        else:
            return {
                "model": "gemini-1.5-flash" if use_gemini else "gpt-3.5-turbo",
                "extractor": TechnicalExtractor,
                "reason": "Technical docs work well with fast models"
            }
    
    # Rule 4: Unknown/default with quality requirement - Use best available
    if quality_required and not budget_conscious:
        if use_gemini:
            return {
                "model": "gemini-2.5-pro",
                "extractor": BaselineExtractor,
                "reason": "Quality required, using Gemini 2.5 Pro (flagship model)"
            }
        return {
            "model": "gpt-4",
            "extractor": BaselineExtractor,
            "reason": "Quality required, using GPT-4"
        }
    
    # Rule 5: Budget conscious - Gemini 1.5 Flash is very cost-effective
    if budget_conscious:
        return {
            "model": "gemini-1.5-flash" if use_gemini else "gpt-3.5-turbo",
            "extractor": BaselineExtractor,
            "reason": "Budget mode with fast model"
        }
    
    # Default - Use Gemini 1.5 Flash as it's very capable and cost-effective
    return {
        "model": "gemini-1.5-flash" if use_gemini else "gpt-3.5-turbo",
        "extractor": BaselineExtractor,
        "reason": "Default extraction with balanced performance"
    }