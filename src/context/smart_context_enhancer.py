"""
Smart context enhancement for chunks.
Adds semantic context to improve retrieval quality.
"""

from typing import Dict, Optional, Tuple
import re
from dataclasses import dataclass


@dataclass
class ContextMetadata:
    """Metadata for context enhancement."""
    title: str
    page_num: int
    category: Optional[str] = None
    chapter: Optional[str] = None


class SmartContextEnhancer:
    """
    Add intelligent context to chunks for better retrieval.
    Supports multiple levels of context enhancement.
    """
    
    def __init__(self):
        """Initialize the context enhancer with mental model patterns."""
        # Mental model detection patterns
        self.mental_model_patterns = {
            # Efficiency and Prioritization
            r'\b(pareto|80[/-]?20)\b': {
                'model': 'Pareto Principle',
                'application': 'prioritization and resource allocation',
                'keywords': ['efficiency', 'focus', 'vital few', 'trivial many']
            },
            
            # Cognitive Biases
            r'\bconfirmation\s+bias\b': {
                'model': 'Confirmation Bias',
                'application': 'avoiding biased decision-making',
                'keywords': ['beliefs', 'evidence', 'assumptions']
            },
            r'\bavailability\s+(heuristic|bias)\b': {
                'model': 'Availability Heuristic',
                'application': 'probability and risk assessment',
                'keywords': ['memory', 'likelihood', 'recent events']
            },
            r'\banchoring\s+(bias|effect)\b': {
                'model': 'Anchoring Bias',
                'application': 'negotiation and estimation',
                'keywords': ['first impression', 'reference point']
            },
            
            # Decision Making
            r'\b(opportunity\s+cost|trade[- ]?off)\b': {
                'model': 'Opportunity Cost',
                'application': 'resource allocation and decision-making',
                'keywords': ['alternatives', 'trade-offs', 'choices']
            },
            r'\basymmetric\s+risk\b': {
                'model': 'Asymmetric Risk',
                'application': 'evaluating opportunities with skewed payoffs',
                'keywords': ['downside', 'upside', 'risk-reward']
            },
            
            # Problem Solving
            r'\bfirst\s+principles?\b': {
                'model': 'First Principles Thinking',
                'application': 'innovative problem-solving',
                'keywords': ['fundamental', 'building blocks', 'reasoning']
            },
            r'\b(inversion|invert)\b.*\b(thinking|principle|approach)\b': {
                'model': 'Inversion',
                'application': 'avoiding failure and negative outcomes',
                'keywords': ['avoid', 'failure', 'opposite']
            },
            
            # Systems and Complexity
            r'\bsystems?\s+thinking\b': {
                'model': 'Systems Thinking',
                'application': 'understanding complex interactions',
                'keywords': ['interconnections', 'holistic', 'feedback loops']
            },
            r'\b(compound|compounding)\s+(effect|interest|growth)\b': {
                'model': 'Compounding',
                'application': 'long-term growth and accumulation',
                'keywords': ['exponential', 'time', 'accumulation']
            },
            
            # Strategy and Planning
            r'\bsecond[- ]?order\s+(effect|thinking|consequence)\b': {
                'model': 'Second-Order Thinking',
                'application': 'anticipating consequences',
                'keywords': ['consequences', 'ripple effects', 'long-term']
            },
            r'\b(margin\s+of\s+safety|safety\s+margin)\b': {
                'model': 'Margin of Safety',
                'application': 'risk management and buffer creation',
                'keywords': ['buffer', 'cushion', 'protection']
            }
        }
    
    def detect_mental_model(self, text: str) -> Optional[Dict[str, str]]:
        """Detect mental models in the text."""
        text_lower = text.lower()
        
        for pattern, info in self.mental_model_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                # Check for keyword relevance
                keyword_score = sum(1 for kw in info['keywords'] if kw in text_lower)
                if keyword_score >= 1 or len(text_lower) < 200:  # Short chunks get a pass
                    return info
        
        return None
    
    def add_context(self, chunk: str, metadata: ContextMetadata, level: int = 2) -> str:
        """
        Add context to a chunk based on the specified level.
        
        Args:
            chunk: The text chunk to enhance
            metadata: Metadata about the chunk
            level: Context enhancement level (1-3)
                1: Basic context (title, page)
                2: Smart context (mental model + application)
                3: Analytical context (detailed analysis)
        
        Returns:
            Enhanced chunk with context prefix
        """
        if level == 1:
            return self._add_basic_context(chunk, metadata)
        elif level == 2:
            return self._add_smart_context(chunk, metadata)
        elif level == 3:
            return self._add_analytical_context(chunk, metadata)
        else:
            return chunk
    
    def _add_basic_context(self, chunk: str, metadata: ContextMetadata) -> str:
        """Add basic bibliographic context."""
        # Shorten long titles
        title = metadata.title
        if len(title) > 40:
            title = title[:37] + "..."
        
        context = f"[{title}, p.{metadata.page_num}]"
        
        if metadata.chapter:
            context = f"[{title}, {metadata.chapter}, p.{metadata.page_num}]"
        
        return f"{context} {chunk}"
    
    def _add_smart_context(self, chunk: str, metadata: ContextMetadata) -> str:
        """Add smart context with mental model detection."""
        # Detect mental model
        model_info = self.detect_mental_model(chunk)
        
        if model_info:
            # Mental model detected - add specific context
            context = f"[{metadata.category or 'Mental Models'} - {model_info['model']} for {model_info['application']}]"
        else:
            # No specific model detected - use general context
            chunk_lower = chunk.lower()
            
            # Try to detect general topics
            if any(word in chunk_lower for word in ['decision', 'choice', 'choose', 'decide']):
                context = f"[{metadata.category or 'Mental Models'} - Decision Making Framework]"
            elif any(word in chunk_lower for word in ['bias', 'cognitive', 'thinking error']):
                context = f"[{metadata.category or 'Mental Models'} - Cognitive Bias Awareness]"
            elif any(word in chunk_lower for word in ['system', 'complex', 'interaction']):
                context = f"[{metadata.category or 'Mental Models'} - Systems Analysis]"
            elif any(word in chunk_lower for word in ['creative', 'innovation', 'idea']):
                context = f"[{metadata.category or 'Mental Models'} - Creative Thinking]"
            else:
                # Fallback to basic context
                return self._add_basic_context(chunk, metadata)
        
        return f"{context} {chunk}"
    
    def _add_analytical_context(self, chunk: str, metadata: ContextMetadata) -> str:
        """Add detailed analytical context."""
        # Detect mental model
        model_info = self.detect_mental_model(chunk)
        
        if model_info:
            # Build rich context
            model = model_info['model']
            application = model_info['application']
            
            # Analyze chunk for additional context
            chunk_lower = chunk.lower()
            
            # Detect key concepts
            concepts = []
            if 'example' in chunk_lower or 'for instance' in chunk_lower:
                concepts.append('Examples')
            if 'how to' in chunk_lower or 'steps' in chunk_lower:
                concepts.append('Implementation')
            if 'why' in chunk_lower or 'because' in chunk_lower:
                concepts.append('Reasoning')
            if 'when' in chunk_lower:
                concepts.append('Timing')
            
            # Build context string
            context_parts = [
                f"Model: {model}",
                f"Application: {application}"
            ]
            
            if concepts:
                context_parts.append(f"Content: {', '.join(concepts)}")
            
            if metadata.chapter:
                context_parts.append(f"Section: {metadata.chapter}")
            
            context = f"[{' | '.join(context_parts)}]"
        else:
            # No model detected - analyze content type
            chunk_lower = chunk.lower()
            
            content_type = "General Discussion"
            if 'definition' in chunk_lower or 'is defined as' in chunk_lower:
                content_type = "Definition"
            elif 'example' in chunk_lower:
                content_type = "Examples"
            elif 'research' in chunk_lower or 'study' in chunk_lower:
                content_type = "Research"
            elif 'conclusion' in chunk_lower or 'summary' in chunk_lower:
                content_type = "Summary"
            
            context = f"[{metadata.category or 'Content'} | Type: {content_type} | Page: {metadata.page_num}]"
        
        return f"{context} {chunk}"
    
    def extract_topics(self, chunk: str) -> Tuple[str, ...]:
        """Extract main topics from a chunk for indexing."""
        topics = []
        
        # Check for mental models
        model_info = self.detect_mental_model(chunk)
        if model_info:
            topics.append(model_info['model'])
        
        # Check for general topics
        chunk_lower = chunk.lower()
        
        topic_keywords = {
            'decision making': ['decision', 'choice', 'choose'],
            'cognitive bias': ['bias', 'cognitive', 'heuristic'],
            'problem solving': ['problem', 'solution', 'solve'],
            'risk management': ['risk', 'uncertainty', 'probability'],
            'systems thinking': ['system', 'complex', 'interaction'],
            'creativity': ['creative', 'innovation', 'idea'],
            'productivity': ['efficiency', 'productive', 'output'],
            'psychology': ['psychological', 'behavior', 'mind']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in chunk_lower for kw in keywords):
                topics.append(topic)
        
        return tuple(topics[:3])  # Return top 3 topics max


def create_context_enhancer() -> SmartContextEnhancer:
    """Create a context enhancer instance."""
    return SmartContextEnhancer()